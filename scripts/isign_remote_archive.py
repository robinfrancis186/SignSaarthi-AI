#!/usr/bin/env python3
"""Range-read individual files from the gated multipart iSign pose archive.

The upstream archive is a standard ZIP split into four raw byte parts. This
module reads the ZIP directory and selected members with HTTP Range requests;
it never reconstructs or stores the 170 GB source archive.
"""

from __future__ import annotations

import bz2
import hashlib
import io
import lzma
import struct
import threading
import time
import zipfile
import zlib
from dataclasses import dataclass
from typing import Callable, Iterable


ISIGN_REPO_ID = "Exploration-Lab/iSign"
ISIGN_DATASET_REVISION = "e4ee6c5f0d9dfcbc74205e3f1388ce94da26c298"
ISIGN_POSE_PARTS = tuple(f"iSign-poses_v1.1_part_a{suffix}" for suffix in "abcd")
ISIGN_POSE_PREFIX = "iSign-poses_v1.1/"


class ArchiveAccessError(RuntimeError):
    """Raised when a remote member cannot be read or verified."""


@dataclass(frozen=True)
class ArchivePart:
    name: str
    start: int
    size: int
    location: str
    etag: str

    @property
    def end(self) -> int:
        return self.start + self.size


class ConcatenatedRangeReader(io.RawIOBase):
    """Seekable view over several byte ranges presented as one file."""

    def __init__(
        self,
        parts: Iterable[ArchivePart],
        fetch_range: Callable[[ArchivePart, int, int], bytes],
    ) -> None:
        super().__init__()
        self.parts = tuple(parts)
        if not self.parts:
            raise ValueError("At least one archive part is required.")
        expected_start = 0
        for part in self.parts:
            if part.start != expected_start or part.size <= 0:
                raise ValueError("Archive parts must be positive and contiguous.")
            expected_start = part.end
        self.total_size = expected_start
        self.fetch_range = fetch_range
        self.position = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self.position + offset
        elif whence == io.SEEK_END:
            position = self.total_size + offset
        else:
            raise ValueError(f"Unsupported seek mode: {whence}")
        if position < 0:
            raise ValueError("Cannot seek before the start of the archive.")
        self.position = position
        return position

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = self.total_size - self.position
        size = min(size, self.total_size - self.position)
        if size <= 0:
            return b""
        data = self.read_at(self.position, size)
        self.position += len(data)
        return data

    def read_at(self, offset: int, size: int) -> bytes:
        if offset < 0 or size < 0 or offset + size > self.total_size:
            raise ArchiveAccessError("Requested range is outside the multipart archive.")
        chunks: list[bytes] = []
        position = offset
        remaining = size
        while remaining:
            part = next((candidate for candidate in self.parts if candidate.start <= position < candidate.end), None)
            if part is None:
                raise ArchiveAccessError(f"No archive part covers byte {position}.")
            local_offset = position - part.start
            take = min(remaining, part.end - position)
            chunk = self.fetch_range(part, local_offset, take)
            if len(chunk) != take:
                raise ArchiveAccessError(
                    f"Short range response for {part.name}: expected {take}, received {len(chunk)}."
                )
            chunks.append(chunk)
            position += take
            remaining -= take
        return b"".join(chunks)


class ISignRemotePoseArchive:
    """Authenticated, bounded reader for individual iSign `.pose` files."""

    def __init__(
        self,
        repo_id: str = ISIGN_REPO_ID,
        revision: str = ISIGN_DATASET_REVISION,
    ) -> None:
        try:
            import httpx
            from huggingface_hub import get_hf_file_metadata, get_token, hf_hub_url
        except ImportError as exc:
            raise ArchiveAccessError(
                "Install scripts/requirements-isign.txt before reading the iSign archive."
            ) from exc

        token = get_token()
        if not token:
            raise ArchiveAccessError("Hugging Face authentication is required for gated iSign access.")

        parts: list[ArchivePart] = []
        start = 0
        for name in ISIGN_POSE_PARTS:
            url = hf_hub_url(repo_id, name, repo_type="dataset", revision=revision)
            metadata = get_hf_file_metadata(url, token=token)
            if not metadata.size or not metadata.location:
                raise ArchiveAccessError(f"Hugging Face returned incomplete metadata for {name}.")
            parts.append(
                ArchivePart(
                    name=name,
                    start=start,
                    size=int(metadata.size),
                    location=str(metadata.location),
                    etag=str(metadata.etag or ""),
                )
            )
            start += int(metadata.size)

        self._httpx = httpx
        self._client = httpx.Client(
            follow_redirects=True,
            timeout=httpx.Timeout(60.0, connect=20.0),
            limits=httpx.Limits(max_connections=6, max_keepalive_connections=2),
            headers={"User-Agent": "SignSaarthi-iSign-range-reader/1.0"},
        )
        self._counter_lock = threading.Lock()
        self.range_requests = 0
        self.bytes_transferred = 0
        self.reader = ConcatenatedRangeReader(parts, self._fetch_range)
        try:
            self.zip_file = zipfile.ZipFile(self.reader)
        except (OSError, zipfile.BadZipFile) as exc:
            self.close()
            raise ArchiveAccessError("The remote iSign pose parts do not form a valid ZIP archive.") from exc
        self.members = {info.filename: info for info in self.zip_file.infolist() if not info.is_dir()}
        directory_digest = hashlib.sha256()
        for name, info in sorted(self.members.items()):
            directory_digest.update(
                f"{name}\0{info.CRC}\0{info.file_size}\0{info.compress_size}\0{info.header_offset}\n".encode(
                    "utf-8"
                )
            )
        self.archive_directory_sha256 = directory_digest.hexdigest()

    def __enter__(self) -> "ISignRemotePoseArchive":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    @property
    def archive_size(self) -> int:
        return self.reader.total_size

    @property
    def part_fingerprints(self) -> list[dict[str, object]]:
        return [
            {"name": part.name, "size": part.size, "etag": part.etag}
            for part in self.reader.parts
        ]

    def close(self) -> None:
        zip_file = getattr(self, "zip_file", None)
        if zip_file is not None:
            zip_file.close()
        client = getattr(self, "_client", None)
        if client is not None:
            client.close()

    def pose_info(self, uid: str) -> zipfile.ZipInfo:
        name = f"{ISIGN_POSE_PREFIX}{uid}.pose"
        info = self.members.get(name)
        if info is None:
            raise ArchiveAccessError("The requested pose UID is absent from the gated archive.")
        return info

    def read_pose(self, uid: str) -> bytes:
        return self.read_member(self.pose_info(uid))

    def read_member(self, info: zipfile.ZipInfo) -> bytes:
        if info.flag_bits & 0x1:
            raise ArchiveAccessError("Encrypted ZIP members are unsupported.")
        header = self.reader.read_at(info.header_offset, 30)
        signature, *_middle, name_length, extra_length = struct.unpack("<I5H3L2H", header)
        if signature != 0x04034B50:
            raise ArchiveAccessError("Invalid local ZIP member header.")
        data_offset = info.header_offset + 30 + name_length + extra_length
        compressed = self.reader.read_at(data_offset, info.compress_size)
        try:
            if info.compress_type == zipfile.ZIP_STORED:
                data = compressed
            elif info.compress_type == zipfile.ZIP_DEFLATED:
                data = zlib.decompress(compressed, -15)
            elif info.compress_type == zipfile.ZIP_BZIP2:
                data = bz2.decompress(compressed)
            elif info.compress_type == zipfile.ZIP_LZMA:
                data = lzma.decompress(compressed)
            else:
                raise ArchiveAccessError(f"Unsupported ZIP compression method: {info.compress_type}")
        except (OSError, EOFError, zlib.error, lzma.LZMAError) as exc:
            raise ArchiveAccessError("Unable to decompress the selected iSign pose member.") from exc
        if len(data) != info.file_size:
            raise ArchiveAccessError("Decompressed iSign pose size does not match the ZIP directory.")
        if (zlib.crc32(data) & 0xFFFFFFFF) != info.CRC:
            raise ArchiveAccessError("Decompressed iSign pose failed its ZIP CRC check.")
        return data

    def _fetch_range(self, part: ArchivePart, offset: int, size: int) -> bytes:
        last_error: Exception | None = None
        for attempt in range(4):
            try:
                response = self._client.get(
                    part.location,
                    headers={"Range": f"bytes={offset}-{offset + size - 1}"},
                )
                data = response.content
                with self._counter_lock:
                    self.range_requests += 1
                    self.bytes_transferred += len(data)
                if response.status_code != 206:
                    raise ArchiveAccessError(
                        f"Range request for {part.name} returned HTTP {response.status_code}."
                    )
                if len(data) != size:
                    raise ArchiveAccessError(
                        f"Short range response for {part.name}: expected {size}, received {len(data)}."
                    )
                return data
            except (self._httpx.TimeoutException, self._httpx.TransportError, ArchiveAccessError) as exc:
                last_error = exc
                if attempt < 3:
                    time.sleep(0.75 * (attempt + 1))
        raise ArchiveAccessError(
            f"Range request for {part.name} failed after four bounded attempts."
        ) from last_error
