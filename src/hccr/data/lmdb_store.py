"""Single-file LMDB image store for high-throughput HCCR reads."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import lmdb
from PIL import Image
from tqdm.auto import tqdm

SCHEMA_VERSION = 1
METADATA_KEY = b"__hccr_metadata__"
IMAGE_KEY_PREFIX = b"image:"
_ENVIRONMENTS: dict[tuple[int, Path], tuple[lmdb.Environment, int]] = {}


def manifest_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_key(sample_id: str) -> bytes:
    return IMAGE_KEY_PREFIX + sample_id.encode("utf-8")


class LMDBImageStore:
    """Lazily open one read-only LMDB environment per DataLoader process."""

    def __init__(self, path: Path, expected_manifest_digest: str | None = None) -> None:
        self.path = Path(path)
        self.expected_manifest_digest = expected_manifest_digest
        self._environment: lmdb.Environment | None = None
        self._registered_key: tuple[int, Path] | None = None
        self._metadata: dict[str, Any] | None = None

    @property
    def metadata(self) -> dict[str, Any]:
        if self._metadata is None:
            with self._open().begin(buffers=True) as transaction:
                encoded = transaction.get(METADATA_KEY)
            if encoded is None:
                raise ValueError(f"LMDB metadata is missing: {self.path}")
            self._metadata = json.loads(bytes(encoded))
            if self._metadata.get("schema_version") != SCHEMA_VERSION:
                raise ValueError(
                    "unsupported LMDB schema version: "
                    f"{self._metadata.get('schema_version')}"
                )
            actual_digest = self._metadata.get("manifest_digest")
            if (
                self.expected_manifest_digest is not None
                and actual_digest != self.expected_manifest_digest
            ):
                raise ValueError(
                    "LMDB was built from a different manifest: "
                    f"expected {self.expected_manifest_digest}, got {actual_digest}"
                )
        return self._metadata

    def read_image(self, sample_id: str) -> Image.Image:
        _ = self.metadata
        with self._open().begin(buffers=True) as transaction:
            encoded = transaction.get(image_key(sample_id))
            if encoded is None:
                raise KeyError(f"sample is missing from LMDB: {sample_id}")
            payload = bytes(encoded)
        with Image.open(BytesIO(payload)) as image:
            return image.convert("L")

    def close(self) -> None:
        if self._registered_key is not None:
            environment, references = _ENVIRONMENTS[self._registered_key]
            if references == 1:
                environment.close()
                del _ENVIRONMENTS[self._registered_key]
            else:
                _ENVIRONMENTS[self._registered_key] = (environment, references - 1)
        self._environment = None
        self._registered_key = None

    def __del__(self) -> None:
        self.close()

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_environment"] = None
        state["_registered_key"] = None
        return state

    def _open(self) -> lmdb.Environment:
        if self._environment is None:
            if not self.path.is_file():
                raise FileNotFoundError(f"LMDB dataset does not exist: {self.path}")
            registry_key = (os.getpid(), self.path.resolve())
            registered = _ENVIRONMENTS.get(registry_key)
            if registered is None:
                self._environment = lmdb.open(
                    str(self.path),
                    subdir=False,
                    readonly=True,
                    lock=False,
                    readahead=False,
                    meminit=False,
                    max_readers=512,
                )
                _ENVIRONMENTS[registry_key] = (self._environment, 1)
            else:
                self._environment = registered[0]
                _ENVIRONMENTS[registry_key] = (registered[0], registered[1] + 1)
            self._registered_key = registry_key
        return self._environment


def build_lmdb_image_store(
    manifest_path: Path,
    rows: list[dict[str, str]],
    source_root: Path,
    output_path: Path,
    *,
    map_size: int | None = None,
    commit_interval: int = 10_000,
    show_progress: bool = False,
) -> dict[str, Any]:
    """Pack encoded source images into one atomic single-file LMDB dataset."""
    resolved_map_size = map_size or _estimate_map_size(
        rows, lambda row: _filesystem_image_path(source_root, row).stat().st_size
    )
    return _build_lmdb_image_store(
        manifest_path,
        rows,
        output_path,
        lambda row: _read_filesystem_image(source_root, row),
        resolved_map_size,
        commit_interval,
        show_progress,
        source_kind="filesystem",
    )


def build_lmdb_image_store_from_zip(
    manifest_path: Path,
    rows: list[dict[str, str]],
    source_zip: Path,
    output_path: Path,
    *,
    zip_prefix: str = "",
    map_size: int | None = None,
    commit_interval: int = 10_000,
    show_progress: bool = False,
) -> dict[str, Any]:
    """Pack images directly from a ZIP archive without extracting small files."""
    source_zip = Path(source_zip)
    if not source_zip.is_file():
        raise FileNotFoundError(f"source ZIP does not exist: {source_zip}")
    normalized_prefix = _normalize_zip_path(zip_prefix, allow_empty=True)
    with ZipFile(source_zip) as archive:

        def member_for_row(row: dict[str, str]) -> str:
            return _zip_member_name(row["source_file"], normalized_prefix)

        resolved_map_size = map_size or _estimate_map_size(
            rows, lambda row: archive.getinfo(member_for_row(row)).file_size
        )
        return _build_lmdb_image_store(
            manifest_path,
            rows,
            output_path,
            lambda row: _read_zip_image(archive, member_for_row(row), zip_prefix),
            resolved_map_size,
            commit_interval,
            show_progress,
            source_kind="zip",
            source_prefix=normalized_prefix,
        )


def _build_lmdb_image_store(
    manifest_path: Path,
    rows: list[dict[str, str]],
    output_path: Path,
    read_payload: Callable[[dict[str, str]], bytes],
    resolved_map_size: int,
    commit_interval: int,
    show_progress: bool,
    *,
    source_kind: str,
    source_prefix: str = "",
) -> dict[str, Any]:
    if commit_interval < 1:
        raise ValueError("commit_interval must be positive")
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite LMDB dataset: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f"{output_path.name}.building")
    temporary_lock = Path(f"{temporary_path}-lock")
    if temporary_path.exists() or temporary_lock.exists():
        raise FileExistsError(f"stale LMDB build output exists: {temporary_path}")
    environment = lmdb.open(
        str(temporary_path),
        subdir=False,
        map_size=resolved_map_size,
        lock=True,
        meminit=False,
        map_async=True,
    )
    transaction = environment.begin(write=True)
    source_bytes = 0
    try:
        progress = tqdm(
            rows,
            desc="pack LMDB",
            unit="image",
            disable=not show_progress,
            dynamic_ncols=True,
        )
        for index, row in enumerate(progress, start=1):
            payload = read_payload(row)
            source_bytes += len(payload)
            if not transaction.put(
                image_key(row["sample_id"]), payload, overwrite=False
            ):
                raise ValueError(f"duplicate LMDB sample_id: {row['sample_id']}")
            if index % commit_interval == 0:
                transaction.commit()
                transaction = environment.begin(write=True)
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "manifest_digest": manifest_digest(manifest_path),
            "sample_count": len(rows),
            "source_bytes": source_bytes,
            "source_kind": source_kind,
            "source_prefix": source_prefix,
            "value_encoding": "original_encoded_image_bytes",
        }
        transaction.put(METADATA_KEY, json.dumps(metadata).encode("utf-8"))
        transaction.commit()
        transaction = None
        environment.sync(True)
    except Exception:
        if transaction is not None:
            transaction.abort()
        environment.close()
        if temporary_path.exists():
            temporary_path.unlink()
        if temporary_lock.exists():
            temporary_lock.unlink()
        raise
    else:
        environment.close()
    temporary_path.replace(output_path)
    if temporary_lock.exists():
        temporary_lock.unlink()
    return {**metadata, "path": str(output_path), "map_size": resolved_map_size}


def _estimate_map_size(
    rows: list[dict[str, str]], size_for_row: Callable[[dict[str, str]], int]
) -> int:
    if not rows:
        return 64 * 1024**2
    sample_count = min(10_000, len(rows))
    sample_bytes = 0
    for row in rows[:sample_count]:
        try:
            sample_bytes += size_for_row(row)
        except (FileNotFoundError, KeyError) as error:
            raise FileNotFoundError(
                f"manifest image is missing: {row['source_file']}"
            ) from error
    average_bytes = sample_bytes / sample_count
    return max(
        64 * 1024**2,
        int((average_bytes + 512) * len(rows) * 1.5),
    )


def _filesystem_image_path(source_root: Path, row: dict[str, str]) -> Path:
    return source_root / row["source_file"]


def _read_filesystem_image(source_root: Path, row: dict[str, str]) -> bytes:
    image_path = _filesystem_image_path(source_root, row)
    try:
        return image_path.read_bytes()
    except FileNotFoundError as error:
        raise FileNotFoundError(f"manifest image is missing: {image_path}") from error


def _normalize_zip_path(value: str, *, allow_empty: bool) -> str:
    normalized = value.replace("\\", "/").strip("/")
    parts = [part for part in normalized.split("/") if part not in {"", "."}]
    if any(part == ".." for part in parts):
        raise ValueError(f"ZIP path must not contain '..': {value}")
    result = "/".join(parts)
    if not result and not allow_empty:
        raise ValueError("manifest source_file must not be empty")
    return result


def _zip_member_name(source_file: str, zip_prefix: str) -> str:
    relative_path = _normalize_zip_path(source_file, allow_empty=False)
    return f"{zip_prefix}/{relative_path}" if zip_prefix else relative_path


def _read_zip_image(archive: ZipFile, member: str, original_prefix: str) -> bytes:
    try:
        return archive.read(member)
    except KeyError as error:
        raise FileNotFoundError(
            f"ZIP member is missing: {member} (zip_prefix={original_prefix!r})"
        ) from error
