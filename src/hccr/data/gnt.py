"""Read CASIA GNT records without expanding the dataset into image files."""

from __future__ import annotations

import hashlib
import os
import struct
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import BinaryIO

from PIL import Image

_HEADER = struct.Struct("<I2sHH")


@dataclass(frozen=True)
class GNTRecord:
    offset: int
    size: int
    tag: bytes
    width: int
    height: int
    pixel_sha256: str = ""


def _read_header(
    stream: BinaryIO, file_size: int, *, allow_empty: bool = False
) -> GNTRecord:
    offset = stream.tell()
    header = stream.read(_HEADER.size)
    if len(header) != _HEADER.size:
        raise ValueError(f"truncated GNT header at {stream.name}:{offset}")
    size, tag, width, height = _HEADER.unpack(header)
    empty = width == height == 0 and size == _HEADER.size
    if (not width or not height) and not (allow_empty and empty):
        raise ValueError(f"invalid GNT dimensions/size at {stream.name}:{offset}")
    if size != _HEADER.size + width * height:
        raise ValueError(f"invalid GNT dimensions/size at {stream.name}:{offset}")
    if offset + size > file_size:
        raise ValueError(f"truncated GNT bitmap at {stream.name}:{offset}")
    return GNTRecord(offset, size, tag, width, height)


def iter_gnt_records(
    path: Path, *, pixel_hashes: bool = False, allow_empty: bool = False
) -> Iterator[GNTRecord]:
    """Validate boundaries, optionally hash native pixels for leakage audits."""
    with path.open("rb") as stream:
        size = os.fstat(stream.fileno()).st_size
        while stream.tell() < size:
            record = _read_header(stream, size, allow_empty=allow_empty)
            if pixel_hashes and record.width and record.height:
                pixels = stream.read(record.width * record.height)
                digest = hashlib.sha256(struct.pack("<HH", record.width, record.height))
                digest.update(pixels)
                record = replace(record, pixel_sha256=digest.hexdigest())
            yield record
            stream.seek(record.offset + record.size)


def read_gnt_image(path: Path, offset: int) -> Image.Image:
    """Decode one raw grayscale record, preserving native size and polarity."""
    if offset < 0:
        raise ValueError("GNT record_offset must be non-negative")
    with path.open("rb") as stream:
        stream.seek(offset)
        record = _read_header(stream, os.fstat(stream.fileno()).st_size)
        pixels = stream.read(record.width * record.height)
    return Image.frombytes("L", (record.width, record.height), pixels)


def load_source_image(root: Path, row: Mapping[str, str]) -> Image.Image:
    """Load either an encoded image or a GNT record from a manifest row."""
    path = root / row["source_file"]
    if path.suffix.lower() == ".gnt":
        offset = row.get("record_offset", "")
        if not offset:
            raise ValueError(f"GNT source requires record_offset: {path}")
        return read_gnt_image(path, int(offset))
    with Image.open(path) as image:
        return image.convert("L")
