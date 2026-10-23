"""Resumable preparation of official datasets without launching training."""

# ruff: noqa: UP039

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import io
import json
import os
import platform
import re
import shlex
import shutil
import stat
import subprocess
import sys
import time
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path, PurePosixPath, PureWindowsPath
from urllib.parse import parse_qs, urlsplit
from zipfile import ZipFile

STAGES = ("check", "download", "extract", "manifest", "lmdb", "config")
MINIMUM_PYTHON = (3, 12)
DATASETS = {
    "primary": {
        "env_key": "gdown_hwdb10_11_benchmark",
        "archive": "HWDB-1.0-1.1-Benchmark.zip",
        "protocol": "hwdb1.0-1.1-official",
        "merged": "HWDB1.0-1.1",
        "classes": 3755,
        "observed_archive_bytes": 6591447542,
    },
    "secondary": {
        "env_key": "gdown_hwdb10_12",
        "archive": "HWDB1.0-1.2.zip",
        "protocol": "hwdb1.0-1.2-official",
        "merged": "HWDB1.0-1.2",
        "classes": 7185,
        "observed_archive_bytes": 8721690095,
    },
}


class BootstrapError(RuntimeError):
    """A fixed diagnostic code safe to persist without secret-bearing exceptions."""


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    path.chmod(0o600)


def processed_artifact_inventory(root: Path) -> list[dict]:
    """Inspect existing audits and maps without scanning millions of CSV rows."""
    result = []
    processed = root / "data" / "processed"
    directories = {
        path.parent
        for pattern in (
            "*/manifest.csv",
            "*/audit_report.json",
            "*/class_mapping_full.json",
        )
        for path in processed.glob(pattern)
    }
    for directory in sorted(directories):
        manifest = directory / "manifest.csv"
        report_path = manifest.with_name("audit_report.json")
        mapping_path = manifest.with_name("class_mapping_full.json")
        item = {
            "manifest": str(manifest),
            "manifest_present": manifest.is_file(),
            "class_mapping": str(mapping_path),
            "class_mapping_available": mapping_path.is_file(),
            "mapping_digest_verified": False,
            "official_schema_candidate": False,
            "official_ready": False,
        }
        try:
            report = (
                json.loads(report_path.read_text(encoding="utf-8-sig"))
                if report_path.is_file()
                else {}
            )
            columns = set()
            if manifest.is_file():
                with manifest.open(encoding="utf-8", newline="") as stream:
                    columns = set(next(csv.reader(stream), []))
            schema = report.get("schema_version")
            calibration = report.get("split_counts", {}).get("calibration", 0) > 0
            source_fields = {
                "source_version",
                "official_partition",
                "source_writer_id",
                "dataset_protocol",
            }.issubset(columns)
            item.update(
                {
                    "schema_version": schema,
                    "protocol": report.get("protocol"),
                    "class_count": report.get("class_count"),
                    "has_calibration": calibration,
                    "has_source_version_fields": source_fields,
                    "readiness": "legacy_schema_requires_new_protocol_build",
                }
            )
            if mapping_path.is_file():
                labels = json.loads(mapping_path.read_text(encoding="utf-8-sig"))[
                    "labels"
                ]
                valid_mapping = (
                    isinstance(labels, list)
                    and {int(label["class_id"]) for label in labels}
                    == set(range(len(labels)))
                    and len({label["unicode_label"] for label in labels}) == len(labels)
                    and all(len(label["unicode_label"]) == 1 for label in labels)
                )
                verified = (
                    valid_mapping
                    and report.get("class_count") == len(labels)
                    and report.get("label_mapping_digest") == digest_file(mapping_path)
                )
                item.update(
                    {
                        "mapping_class_count": len(labels),
                        "mapping_digest_verified": verified,
                        "secondary_vocabulary_candidate": verified
                        and len(labels) == 7185,
                    }
                )
            candidate = (
                manifest.is_file()
                and schema == 4
                and calibration
                and source_fields
                and report.get("protocol")
                in {dataset["protocol"] for dataset in DATASETS.values()}
            )
            item["official_schema_candidate"] = candidate
            if candidate:
                item["readiness"] = "requires_official_preflight_validation"
        except (OSError, ValueError, KeyError, TypeError):
            item["readiness"] = "artifact_inventory_invalid"
        result.append(item)
    return result


def read_download_environment(path: Path) -> dict[str, str]:
    """Read only dataset keys as data; never source or interpolate the file."""
    if not path.is_file():
        return {}
    result = {}
    keys = {item["env_key"] for item in DATASETS.values()}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        match = re.match(r"^\s*(?:export\s+)?([A-Za-z_]\w*)\s*=(.*)$", line)
        if match is None or match[1] not in keys:
            continue
        try:
            tokens = shlex.split(match[2], comments=True)
        except ValueError:
            raise BootstrapError("invalid_download_environment") from None
        result[match[1]] = tokens[0] if len(tokens) == 1 else shlex.join(tokens)
    return result


def drive_file_id(value: str) -> str:
    """Accept a Drive URL, ID, or whitelisted gdown command without execution."""
    try:
        tokens = shlex.split(value)
    except ValueError:
        raise BootstrapError("invalid_drive_reference") from None
    if tokens and tokens[0] in {"gdown", "gdown.exe"}:
        candidates = []
        index = 1
        while index < len(tokens):
            token = tokens[index]
            if token in {"-O", "--output"}:
                index += 2
                if index > len(tokens):
                    raise BootstrapError("invalid_gdown_options")
                continue
            if token in {"--fuzzy", "--id", "--continue", "-c", "--quiet", "-q"}:
                index += 1
                continue
            if token.startswith("-"):
                raise BootstrapError("unsupported_gdown_option")
            candidates.append(token)
            index += 1
        tokens = candidates
    if len(tokens) != 1:
        raise BootstrapError("invalid_drive_reference")
    reference = tokens[0]
    if reference.startswith("https://"):
        url = urlsplit(reference)
        if url.hostname not in {"drive.google.com", "docs.google.com"}:
            raise BootstrapError("unsupported_download_host")
        match = re.search(r"/file/d/([A-Za-z0-9_-]+)(?:/|$)", url.path)
        identifiers = [match[1]] if match else parse_qs(url.query).get("id", [])
        if len(identifiers) != 1:
            raise BootstrapError("invalid_drive_reference")
        reference = identifiers[0]
    if re.fullmatch(r"[A-Za-z0-9_-]{10,200}", reference) is None:
        raise BootstrapError("invalid_drive_reference")
    return reference


def zip_inventory(path: Path) -> dict:
    """Validate the complete central directory before extracting any member."""
    names: set[str] = set()
    total = 0
    files = 0
    with ZipFile(path) as archive:
        for item in archive.infolist():
            name = item.filename.replace("\\", "/")
            member = PurePosixPath(name)
            kind = stat.S_IFMT(item.external_attr >> 16)
            if (
                not name
                or name.startswith("/")
                or PureWindowsPath(name).drive
                or ".." in member.parts
                or "\x00" in name
                or member.parts[0].startswith(".bootstrap")
                or kind not in {0, stat.S_IFREG, stat.S_IFDIR}
                or item.flag_bits & 1
            ):
                raise BootstrapError("unsafe_zip_member")
            normalized = member.as_posix().rstrip("/").casefold()
            if normalized in names:
                raise BootstrapError("duplicate_zip_member")
            names.add(normalized)
            total += item.file_size
            files += not item.is_dir()
    if not files:
        raise BootstrapError("empty_archive")
    return {"file_count": files, "uncompressed_bytes": total}


def require_space(path: Path, required: int, *, files: int = 0) -> None:
    parent = path
    while not parent.exists():
        parent = parent.parent
    if shutil.disk_usage(parent).free < required:
        raise BootstrapError("insufficient_disk_space")
    if hasattr(os, "statvfs"):
        info = os.statvfs(parent)
        if info.f_favail and info.f_favail < files + 32:
            raise BootstrapError("insufficient_free_inodes")


def download_archive(
    value: str, output: Path, *, expected_sha256: str | None = None, downloader=None
) -> dict:
    identifier = drive_file_id(value)
    if expected_sha256 and re.fullmatch(r"[a-fA-F0-9]{64}", expected_sha256) is None:
        raise BootstrapError("invalid_archive_sha256")
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.exists():
        require_space(output.parent, 256 * 1024**2)
        partial = output.with_name(output.name + ".part")
        marker = partial.with_name(partial.name + ".source.json")
        identity = hashlib.sha256(identifier.encode()).hexdigest()
        if marker.exists() and json.loads(marker.read_text())["identity"] != identity:
            raise BootstrapError("partial_download_source_changed")
        atomic_json(marker, {"identity": identity})
        try:
            if downloader is None:
                downloader = importlib.import_module("gdown").download
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                result = downloader(
                    id=identifier,
                    output=str(partial),
                    quiet=True,
                    resume=True,
                    verify=True,
                    use_cookies=False,
                )
            if not result or not partial.is_file():
                raise BootstrapError("download_incomplete")
            inventory = zip_inventory(partial)
            sha256 = digest_file(partial)
            if expected_sha256 and sha256 != expected_sha256.lower():
                raise BootstrapError("archive_sha256_mismatch")
            partial.replace(output)
            marker.unlink(missing_ok=True)
        except BootstrapError:
            raise
        except Exception:
            raise BootstrapError(
                "download_failed_check_network_and_drive_access"
            ) from None
    else:
        inventory = zip_inventory(output)
        sha256 = digest_file(output)
        if expected_sha256 and sha256 != expected_sha256.lower():
            raise BootstrapError("archive_sha256_mismatch")
    return {**inventory, "sha256": sha256, "size_bytes": output.stat().st_size}


def extract_archive(archive_path: Path, output: Path) -> dict:
    inventory = zip_inventory(archive_path)
    archive_digest = digest_file(archive_path)
    marker = output / ".bootstrap-extraction.json"
    if output.exists():
        if not marker.is_file():
            raise BootstrapError("unowned_extraction_directory")
        report = json.loads(marker.read_text(encoding="utf-8"))
        if report.get("archive_sha256") != archive_digest:
            raise BootstrapError("extraction_archive_changed")
        with ZipFile(archive_path) as archive:
            expected = {
                PurePosixPath(item.filename.replace("\\", "/")).as_posix()
                for item in archive.infolist()
                if not item.is_dir()
            }
        if {item["path"] for item in report["files"]} != expected:
            raise BootstrapError("extraction_inventory_changed")
        for item in report["files"]:
            path = output / item["path"]
            if (
                not path.resolve().is_relative_to(output.resolve())
                or path.is_symlink()
                or not path.is_file()
                or digest_file(path) != item["sha256"]
            ):
                raise BootstrapError("extracted_source_changed")
        return {**inventory, "archive_sha256": archive_digest, "reused": True}
    require_space(
        output.parent,
        inventory["uncompressed_bytes"]
        + max(256 * 1024**2, inventory["uncompressed_bytes"] // 10),
        files=inventory["file_count"],
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(output.name + ".bootstrap-extracting")
    if staging.exists():
        ownership = staging / ".bootstrap-owner.json"
        if (
            not ownership.is_file()
            or json.loads(ownership.read_text()).get("archive_sha256") != archive_digest
        ):
            raise BootstrapError("unowned_extraction_staging")
        shutil.rmtree(staging)
    staging.mkdir()
    atomic_json(staging / ".bootstrap-owner.json", {"archive_sha256": archive_digest})
    files = []
    with ZipFile(archive_path) as archive:
        for item in archive.infolist():
            target = staging / PurePosixPath(item.filename.replace("\\", "/"))
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            with archive.open(item) as source, target.open("xb") as destination:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    destination.write(chunk)
                    digest.update(chunk)
            if target.stat().st_size != item.file_size:
                raise BootstrapError("extraction_size_mismatch")
            files.append(
                {
                    "path": target.relative_to(staging).as_posix(),
                    "sha256": digest.hexdigest(),
                }
            )
    atomic_json(
        staging / marker.name,
        {
            "archive_sha256": archive_digest,
            "files": files,
            **inventory,
        },
    )
    staging.rename(output)
    return {**inventory, "archive_sha256": archive_digest, "reused": False}


def _read_limit(path: Path) -> int | None:
    try:
        text = path.read_text().strip()
        value = int(text)
        return value if 0 < value < 2**60 else None
    except (OSError, ValueError):
        return None


def hardware_report() -> dict:
    cpu_count = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else os.cpu_count() or 1
    )
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            cpu_count = min(cpu_count, max(1, int(quota) // int(period)))
    except (OSError, ValueError):
        quota = _read_limit(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"))
        period = _read_limit(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us"))
        if quota and period:
            cpu_count = min(cpu_count, max(1, quota // period))
    memory = None
    if hasattr(os, "sysconf"):
        memory = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    limits = [
        value
        for path in (
            "/sys/fs/cgroup/memory.max",
            "/sys/fs/cgroup/memory/memory.limit_in_bytes",
        )
        if (value := _read_limit(Path(path))) is not None
    ]
    if limits:
        memory = min([memory, *limits]) if memory else min(limits)
    available_memory = memory
    try:
        match = re.search(
            r"^MemAvailable:\s+(\d+) kB$",
            Path("/proc/meminfo").read_text(),
            re.MULTILINE,
        )
        if match:
            available_memory = min(memory or int(match[1]) * 1024, int(match[1]) * 1024)
    except OSError:
        pass
    current = _read_limit(Path("/sys/fs/cgroup/memory.current")) or _read_limit(
        Path("/sys/fs/cgroup/memory/memory.usage_in_bytes")
    )
    if limits and current:
        available_memory = min(
            available_memory or min(limits), max(0, min(limits) - current)
        )
    gpus = []
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
        for line in result.stdout.splitlines():
            name, total, free, driver = [item.strip() for item in line.split(",")]
            gpus.append(
                {
                    "name": name,
                    "total_vram_mib": int(total),
                    "free_vram_mib": int(free),
                    "driver": driver,
                }
            )
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    torch_report = {"available": False, "cuda_available": False}
    try:
        torch = importlib.import_module("torch")
        cuda = torch.cuda.is_available()
        torch_report = {
            "available": True,
            "version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": cuda,
            "device_count": torch.cuda.device_count() if cuda else 0,
            "bf16_supported": bool(
                cuda and torch.cuda.is_bf16_supported(including_emulation=False)
            ),
        }
    except (ImportError, RuntimeError, OSError):
        pass
    shm = shutil.disk_usage("/dev/shm").free if Path("/dev/shm").is_dir() else None
    return {
        "platform": platform.system(),
        "python": platform.python_version(),
        "cpu_count": cpu_count,
        "memory_limit_bytes": memory,
        "memory_available_bytes": available_memory,
        "shared_memory_free_bytes": shm,
        "gpus": gpus,
        "torch": torch_report,
    }


def configure_runtime(
    template: Path,
    output: Path,
    manifest: Path,
    hardware: dict,
    *,
    preset: str,
    concurrent_jobs: int,
    cache_backend: str,
    classes: int,
) -> dict:
    import yaml

    if concurrent_jobs < 1:
        raise BootstrapError("concurrent_jobs_must_be_positive")
    config = yaml.safe_load(template.read_text(encoding="utf-8"))
    base = config.setdefault("base_args", {})
    workers = min(8, max(0, (hardware["cpu_count"] - 4) // concurrent_jobs))
    shared_memory = hardware.get("shared_memory_free_bytes")
    if shared_memory is not None:
        batch_bytes = (
            int(base.get("batch_size", 256)) * int(base.get("image_size", 96)) ** 2 * 4
        )
        workers = min(
            workers, int(shared_memory // concurrent_jobs // 2 // max(1, batch_bytes))
        )
    cuda = hardware["torch"].get("cuda_available", False)
    if preset == "bf16" and not hardware["torch"].get("bf16_supported", False):
        raise BootstrapError("bf16_runtime_requires_supported_cuda")
    cache_backend = cache_backend or ("auto" if preset == "bf16" else "none")
    if preset == "strict" and cache_backend != "none":
        raise BootstrapError("strict_runtime_preserves_cpu_augmentation_without_cache")
    if cache_backend == "cuda" and not cuda:
        raise BootstrapError("cuda_cache_requires_cuda")
    memory_budget = hardware.get("memory_available_bytes")
    if memory_budget is None:
        memory_budget = hardware.get("memory_limit_bytes") or 64 * 1024**3
    cache_max = min(32.0, memory_budget / 1024**3 / concurrent_jobs / 2)
    if cache_max <= 0:
        raise BootstrapError("insufficient_memory_headroom")
    if cache_backend == "cuda" and hardware.get("gpus"):
        cache_max = min(
            cache_max,
            min(item["free_vram_mib"] for item in hardware["gpus"]) / 1024 - 16,
        )
        if cache_max <= 0:
            raise BootstrapError("insufficient_cuda_cache_headroom")
    base.update(
        {
            "device": "cuda" if cuda else "cpu",
            "precision": "bfloat16" if preset == "bf16" else "float32",
            "num_workers": workers,
            "prefetch_factor": 1,
            "persistent_workers": workers > 0,
            "dataset_backend": "lmdb",
            "num_classes": classes,
            "augmentation_backend": "gpu" if preset == "bf16" else "cpu",
            "training_cache": cache_backend,
            "training_cache_max_gib": round(cache_max, 2),
            "training_cache_reserve_gib": 16.0,
        }
    )
    if workers:
        base["dataloader_start_method"] = "spawn"
    config["manifest"] = str(manifest.resolve())
    config["full_class_num_classes"] = classes
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    temporary.chmod(0o600)
    temporary.replace(output)
    return {
        "config_sha256": digest_file(output),
        "runtime_preset": preset,
        "cache_backend": cache_backend,
        "workers_used_for": "initial_cache_fill_and_validation"
        if cache_backend != "none"
        else "training_and_validation",
        "num_workers_per_job": workers,
        "concurrent_jobs": concurrent_jobs,
        "batch_size": base.get("batch_size"),
        "epochs": base.get("epochs"),
        "manifest_ready": manifest.is_file(),
        "training_started": False,
    }


@contextmanager
def stage_lock(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ".bootstrap.lock"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT, 0o600)
    try:
        if os.name == "posix":
            import fcntl

            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise BootstrapError("bootstrap_already_running") from None
        else:
            import msvcrt

            try:
                msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
            except OSError:
                raise BootstrapError("bootstrap_already_running") from None
        yield
    finally:
        os.close(descriptor)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--dataset", choices=DATASETS, default="primary")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--through", choices=STAGES, default="config")
    selection.add_argument("--stage", choices=STAGES, action="append")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--archive-sha256")
    parser.add_argument(
        "--raw-root",
        type=Path,
        help="Reuse local GNT directories without downloading or extracting an archive",
    )
    parser.add_argument("--existing-manifest", type=Path)
    parser.add_argument("--source-version-map", type=Path)
    parser.add_argument("--writer-identity-map", type=Path)
    parser.add_argument("--vocabulary", type=Path)
    parser.add_argument("--config-template", type=Path)
    parser.add_argument("--config-output", type=Path)
    parser.add_argument(
        "--runtime-preset", choices=("strict", "bf16"), default="strict"
    )
    parser.add_argument("--concurrent-jobs", type=int, default=2)
    parser.add_argument("--cache-backend", choices=("none", "auto", "cpu", "cuda"))
    return parser


class Bootstrap:
    def __init__(self, args: argparse.Namespace, *, hardware=None, downloader=None):
        self.args = args
        self.dataset = DATASETS[args.dataset]
        self.root = args.root.resolve()
        self.work = (
            args.work_dir or self.root / "artifacts" / f"vast-bootstrap-{args.dataset}"
        ).resolve()
        self.environment_path = args.env_file or self.root / ".env"
        self.archive = (
            args.archive or self.root / "data" / "archives" / self.dataset["archive"]
        )
        local_raw = self.root / "data" / "raw"
        required_versions = ("HWDB1.0", "HWDB1.1") + (
            ("HWDB1.2",) if args.dataset == "secondary" else ()
        )
        local_layout_available = all(
            (local_raw / f"{version}_{partition}").is_dir()
            for version in required_versions
            for partition in ("Train", "Test")
        ) and (args.dataset == "secondary" or (local_raw / "ICDAR2013_Test").is_dir())
        self.reuse_raw = args.raw_root is not None or local_layout_available
        self.extracted = (
            args.raw_root or local_raw
            if self.reuse_raw
            else local_raw / f"vast-{args.dataset}"
        ).resolve()
        self.processed = (
            self.root / "data" / "processed" / f"vast-{args.dataset}-official"
        )
        self.manifest = self.processed / "manifest.csv"
        bundled_versions = self.root / "metadata" / "source_versions.json"
        if args.source_version_map is None and bundled_versions.is_file():
            args.source_version_map = bundled_versions
        self.source_version_audit = (
            args.source_version_map.with_name("audit.json")
            if args.source_version_map
            else None
        )
        self.inventory = processed_artifact_inventory(self.root)
        candidates = [
            item
            for item in self.inventory
            if item["official_schema_candidate"]
            and item.get("protocol") == self.dataset["protocol"]
        ]
        self.existing_manifest = args.existing_manifest or (
            Path(candidates[0]["manifest"]) if candidates else None
        )
        self.existing_validated = False
        if self.existing_manifest:
            self.manifest = self.existing_manifest.resolve()
            self.processed = self.manifest.parent
        if args.dataset == "secondary" and args.vocabulary is None:
            vocabulary = next(
                (
                    item
                    for item in self.inventory
                    if item.get("secondary_vocabulary_candidate")
                ),
                None,
            )
            if vocabulary:
                args.vocabulary = Path(vocabulary["class_mapping"])
        self.template = (
            args.config_template
            or self.root
            / "configs"
            / "experiment"
            / "official_classifier_backbones.yaml"
        )
        self.config_output = args.config_output or self.work / "runtime.yaml"
        self.hardware = hardware
        self.downloader = downloader

    def plan(self) -> dict:
        selected = self.args.stage or list(
            STAGES[: STAGES.index(self.args.through) + 1]
        )
        if self.reuse_raw:
            selected = [
                stage for stage in selected if stage not in {"download", "extract"}
            ]
        return {
            "dataset": self.args.dataset,
            "stages": selected,
            "execute": self.args.execute,
            "download_environment_key": self.dataset["env_key"],
            "archive": str(self.archive),
            "extraction_root": str(self.extracted),
            "reuse_local_raw": self.reuse_raw,
            "source_layout": self.source_layout(),
            "manifest": str(self.manifest),
            "config": str(self.config_output),
            "source_version_map_supplied": self.args.source_version_map is not None,
            "source_version_audit": str(self.source_version_audit)
            if self.source_version_audit and self.source_version_audit.is_file()
            else None,
            "canonical_vocabulary_supplied": self.args.vocabulary is not None,
            "vocabulary_path": str(self.args.vocabulary)
            if self.args.vocabulary
            else None,
            "existing_artifacts": self.inventory,
            "existing_manifest_selected": str(self.existing_manifest)
            if self.existing_manifest
            else None,
            "training_started": False,
        }

    def source_layout(self) -> dict:
        versions = ("HWDB1.0", "HWDB1.1") + (
            ("HWDB1.2",) if self.args.dataset == "secondary" else ()
        )
        directories = {
            partition.lower(): [
                self.extracted / f"{version}_{partition}" for version in versions
            ]
            for partition in ("Train", "Test")
        }
        separated = all(
            path.is_dir() for paths in directories.values() for path in paths
        )
        if not separated:
            directories = {
                partition.lower(): [
                    self.extracted / (self.dataset["merged"] + "_" + partition)
                ]
                for partition in ("Train", "Test")
            }
        icdar = (
            self.extracted / "ICDAR2013_Test"
            if self.args.dataset == "primary"
            else None
        )
        return {
            "kind": "version_separated" if separated else "merged",
            "train_directories": [str(path) for path in directories["train"]],
            "test_directories": [str(path) for path in directories["test"]],
            "icdar_directory": str(icdar) if icdar else None,
            "directories_present": all(
                path.is_dir() for paths in directories.values() for path in paths
            )
            and (icdar is None or icdar.is_dir()),
        }

    def run(self) -> dict:
        plan = self.plan()
        if not self.args.execute:
            return plan
        with stage_lock(self.work):
            status_path = self.work / "status.json"
            state = {
                "dataset": self.args.dataset,
                "status": "running",
                "stages": {},
                "training_started": False,
            }
            if status_path.is_file():
                state["stages"] = json.loads(status_path.read_text())["stages"]
            for stage in plan["stages"]:
                entry = {"status": "running", "started_at": time.time()}
                state["stages"][stage] = entry
                atomic_json(status_path, state)
                try:
                    entry["input_identity"] = self.stage_input_identity(stage)
                    entry["result"] = getattr(self, "stage_" + stage)()
                    entry["status"] = "completed"
                except Exception as error:
                    entry["status"] = (
                        "blocked" if isinstance(error, BootstrapError) else "failed"
                    )
                    entry["error_code"] = (
                        str(error)
                        if isinstance(error, BootstrapError)
                        else "stage_failed_" + type(error).__name__
                    )
                    state["status"] = entry["status"]
                    entry["finished_at"] = time.time()
                    atomic_json(status_path, state)
                    return state
                entry["finished_at"] = time.time()
                atomic_json(status_path, state)
            state["status"] = "completed"
            atomic_json(status_path, state)
            return state

    def stage_input_identity(self, stage: str) -> dict:
        inputs = {
            "dataset": self.args.dataset,
            "protocol": self.dataset["protocol"],
            "implementation_sha256": digest_file(Path(__file__)),
        }
        for name in ("source_version_map", "writer_identity_map", "vocabulary"):
            path = getattr(self.args, name)
            if path and path.is_file():
                inputs[name + "_sha256"] = digest_file(path)
        if self.source_version_audit and self.source_version_audit.is_file():
            inputs["source_version_audit_sha256"] = digest_file(
                self.source_version_audit
            )
        if stage in {"lmdb", "config"} and self.manifest.is_file():
            inputs["manifest_sha256"] = digest_file(self.manifest)
        if stage == "config" and self.template.is_file():
            inputs["template_sha256"] = digest_file(self.template)
        return inputs

    def validate_existing_manifest(self) -> dict:
        from hccr.data.manifest import read_manifest
        from hccr.training.proposal import validate_official_source

        if not self.existing_validated:
            validate_official_source(self.manifest, read_manifest(self.manifest))
            report = json.loads(
                self.manifest.with_name("audit_report.json").read_text()
            )
            if report["protocol"] != self.dataset["protocol"]:
                raise BootstrapError(
                    "existing_manifest_protocol_does_not_match_dataset"
                )
            self.existing_validated = True
        return {
            "manifest_sha256": digest_file(self.manifest),
            "reused_existing_official_manifest": True,
        }

    def stage_check(self) -> dict:
        if sys.version_info < MINIMUM_PYTHON:
            raise BootstrapError("python_3_12_or_newer_required")
        self.hardware = self.hardware or hardware_report()
        report = {
            **self.hardware,
            "disk_free_bytes": shutil.disk_usage(self.work).free,
            "download_environment_keys_present": sorted(
                read_download_environment(self.environment_path)
            ),
            "source_version_map_present": bool(
                self.args.source_version_map and self.args.source_version_map.is_file()
            ),
            "canonical_vocabulary_present": bool(
                self.args.vocabulary and self.args.vocabulary.is_file()
            ),
            "existing_artifacts": self.inventory,
        }
        atomic_json(self.work / "hardware.json", report)
        return report

    def stage_download(self) -> dict:
        if self.existing_manifest:
            return self.validate_existing_manifest()
        if self.reuse_raw:
            return {"reused_local_raw": True, "source_layout": self.source_layout()}
        if self.archive.exists():
            sha256 = digest_file(self.archive)
            if self.args.archive_sha256 and sha256 != self.args.archive_sha256.lower():
                raise BootstrapError("archive_sha256_mismatch")
            return {**zip_inventory(self.archive), "sha256": sha256, "reused": True}
        environment = read_download_environment(self.environment_path)
        value = environment.get(
            self.dataset["env_key"], os.environ.get(self.dataset["env_key"], "")
        )
        if not value:
            raise BootstrapError("missing_dataset_download_environment_key")
        require_space(
            self.archive.parent, self.dataset["observed_archive_bytes"] + 256 * 1024**2
        )
        return download_archive(
            value,
            self.archive,
            expected_sha256=self.args.archive_sha256,
            downloader=self.downloader,
        )

    def stage_extract(self) -> dict:
        if self.existing_manifest:
            return self.validate_existing_manifest()
        if self.reuse_raw:
            return {"reused_local_raw": True, "source_layout": self.source_layout()}
        if not self.archive.is_file():
            raise BootstrapError("archive_required_before_extraction")
        return extract_archive(self.archive, self.extracted)

    def stage_manifest(self) -> dict:
        from hccr.data.casia_manifest import build_casia_manifest
        from hccr.data.manifest import read_manifest
        from hccr.training.proposal import validate_official_source

        if self.existing_manifest:
            return self.validate_existing_manifest()
        if self.manifest.exists():
            validate_official_source(self.manifest, read_manifest(self.manifest))
            report = json.loads(
                self.manifest.with_name("audit_report.json").read_text()
            )
            for argument, field in (
                (self.args.source_version_map, "source_version_mapping"),
                (self.args.writer_identity_map, "writer_identity_mapping"),
            ):
                if argument and json.loads(argument.read_text()) != report[field]:
                    raise BootstrapError("manifest_preparation_inputs_changed")
            for source in report["sources"]:
                path = self.extracted / source["source_file"]
                if not path.is_file() or digest_file(path) != source["sha256"]:
                    raise BootstrapError("audited_manifest_source_changed")
            return {"manifest_sha256": digest_file(self.manifest), "reused": True}
        layout = self.source_layout()
        if layout["kind"] != "version_separated" and (
            not self.args.source_version_map
            or not self.args.source_version_map.is_file()
        ):
            raise BootstrapError(
                "official_sources_require_evidence_backed_source_version_map"
            )
        if not layout["directories_present"]:
            raise BootstrapError("required_official_source_directories_missing")
        if self.args.dataset == "secondary" and (
            not self.args.vocabulary or not self.args.vocabulary.is_file()
        ):
            raise BootstrapError("secondary_requires_audited_7185_label_vocabulary")
        versions = (
            json.loads(self.args.source_version_map.read_text(encoding="utf-8"))
            if self.args.source_version_map
            else {}
        )
        if self.source_version_audit and self.source_version_audit.is_file():
            audit = json.loads(self.source_version_audit.read_text(encoding="utf-8"))
            if (
                audit.get("status") != "completed"
                or audit.get("hashing_performed") is not True
                or audit.get("mapping_sha256")
                != digest_file(self.args.source_version_map)
            ):
                raise BootstrapError("source_version_mapping_audit_mismatch")
            audited = {
                item["source_file"]: item["sha256"]
                for entries in audit["inventory"].values()
                for item in entries
            }
            for directory in layout["train_directories"] + layout["test_directories"]:
                for path in sorted(Path(directory).rglob("*.gnt")):
                    relative = path.relative_to(self.extracted).as_posix()
                    if relative not in versions or audited.get(relative) != digest_file(
                        path
                    ):
                        raise BootstrapError("source_version_audited_content_mismatch")
        identities = (
            json.loads(self.args.writer_identity_map.read_text(encoding="utf-8"))
            if self.args.writer_identity_map
            else None
        )
        vocabulary = (
            json.loads(self.args.vocabulary.read_text(encoding="utf-8"))
            if self.args.vocabulary
            else None
        )
        if isinstance(vocabulary, dict):
            vocabulary = [item["unicode_label"] for item in vocabulary["labels"]]
        primary = self.args.dataset == "primary"
        report = build_casia_manifest(
            [Path(path) for path in layout["train_directories"]],
            [Path(path) for path in layout["test_directories"]],
            self.extracted,
            self.processed,
            protocol=self.dataset["protocol"],
            label_set="gb2312-level1" if primary else "all-hanzi",
            seed=20261022,
            calibration_fraction=0.1,
            validation_fraction=0.1,
            audit_pixel_duplicates=True,
            source_versions=versions,
            writer_identity_map=identities,
            vocabulary_labels=vocabulary,
            icdar_dir=self.extracted / "ICDAR2013_Test" if primary else None,
            expected_development_writers=576 if primary else 816,
            expected_test_writers=144 if primary else 204,
            expected_icdar_writers=60 if primary else None,
            expected_icdar_samples=224419 if primary else None,
        )
        validate_official_source(self.manifest, read_manifest(self.manifest))
        return {
            "manifest_sha256": report["manifest_digest"],
            "class_count": report["class_count"],
            "reused": False,
        }

    def stage_lmdb(self) -> dict:
        from hccr.data.lmdb_store import LMDBImageStore, build_lmdb_image_store
        from hccr.data.manifest import read_manifest
        from hccr.training.proposal import validate_official_source

        if not self.manifest.is_file():
            raise BootstrapError("official_manifest_required_before_lmdb")
        rows = read_manifest(self.manifest)
        validate_official_source(self.manifest, rows)
        output = self.processed / "images.lmdb"
        if output.exists():
            store = LMDBImageStore(output, digest_file(self.manifest))
            try:
                metadata = dict(store.metadata)
                if metadata["sample_count"] != len(rows):
                    raise BootstrapError("lmdb_sample_count_mismatch")
            finally:
                store.close()
            return {**metadata, "reused": True}
        required = sum(int(row["width"]) * int(row["height"]) + 1536 for row in rows)
        require_space(self.processed, int(required * 1.5))
        staging = self.processed / ".bootstrap-lmdb"
        staging.mkdir(exist_ok=True)
        for name in ("images.lmdb.building", "images.lmdb.building-lock"):
            (staging / name).unlink(missing_ok=True)
        temporary = staging / "images.lmdb"
        if temporary.exists():
            temporary.unlink()
        source_root = self.extracted
        if self.existing_manifest:
            from hccr.data import resolve_data_root

            source_root = resolve_data_root(self.manifest, rows)
        report = build_lmdb_image_store(self.manifest, rows, source_root, temporary)
        temporary.replace(output)
        return {
            "manifest_sha256": report["manifest_digest"],
            "sample_count": report["sample_count"],
            "reused": False,
        }

    def stage_config(self) -> dict:
        if self.existing_manifest:
            self.validate_existing_manifest()
        protected = [
            self.environment_path,
            self.archive,
            self.manifest,
            self.template,
            self.processed / "images.lmdb",
            self.args.source_version_map,
            self.source_version_audit,
            self.args.writer_identity_map,
            self.args.vocabulary,
        ]
        if self.config_output.resolve() in {
            path.resolve() for path in protected if path is not None
        }:
            raise BootstrapError("config_output_must_not_overwrite_source_artifacts")
        self.hardware = self.hardware or hardware_report()
        atomic_json(self.work / "hardware.json", self.hardware)
        return configure_runtime(
            self.template,
            self.config_output,
            self.manifest,
            self.hardware,
            preset=self.args.runtime_preset,
            concurrent_jobs=self.args.concurrent_jobs,
            cache_backend=self.args.cache_backend,
            classes=self.dataset["classes"],
        )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = Bootstrap(args).run()
    except Exception as error:
        report = {
            "status": "failed",
            "error_code": str(error)
            if isinstance(error, BootstrapError)
            else "bootstrap_failed_" + type(error).__name__,
            "training_started": False,
        }
    print(json.dumps(report, indent=2))
    return 0 if report.get("status", "completed") == "completed" else 1
