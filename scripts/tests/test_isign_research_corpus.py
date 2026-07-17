from __future__ import annotations

import io
import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from build_isign_research_corpus import (  # noqa: E402
    CaptionRecord,
    balanced_sentence_subset,
    grouped_split,
    normalize_for_training,
    parse_bytes,
    resample_track,
    trim_motion_window,
    video_group_id,
)
from isign_remote_archive import (  # noqa: E402
    ArchivePart,
    ConcatenatedRangeReader,
    ISignRemotePoseArchive,
)


class RemoteArchiveTests(unittest.TestCase):
    def test_concatenated_reader_crosses_part_boundaries(self) -> None:
        payloads = {"aa": b"abc", "ab": b"defg", "ac": b"hi"}
        parts = []
        start = 0
        for name, payload in payloads.items():
            parts.append(ArchivePart(name, start, len(payload), name, name))
            start += len(payload)

        def fetch(part: ArchivePart, offset: int, size: int) -> bytes:
            return payloads[part.name][offset : offset + size]

        reader = ConcatenatedRangeReader(parts, fetch)
        self.assertEqual(reader.read_at(2, 6), b"cdefgh")
        reader.seek(-4, io.SEEK_END)
        self.assertEqual(reader.read(), b"fghi")

    def test_range_fetch_retries_a_transient_timeout(self) -> None:
        class TimeoutErrorForTest(Exception):
            pass

        class FakeHttpx:
            TimeoutException = TimeoutErrorForTest
            TransportError = TimeoutErrorForTest

        class Response:
            status_code = 206
            content = b"abc"

        class Client:
            calls = 0

            def get(self, *_args: object, **_kwargs: object) -> Response:
                self.calls += 1
                if self.calls == 1:
                    raise TimeoutErrorForTest()
                return Response()

        archive = ISignRemotePoseArchive.__new__(ISignRemotePoseArchive)
        archive._httpx = FakeHttpx
        archive._client = Client()
        archive._counter_lock = threading.Lock()
        archive.range_requests = 0
        archive.bytes_transferred = 0
        part = ArchivePart("aa", 0, 3, "https://example.invalid/archive", "etag")

        with patch("isign_remote_archive.time.sleep"):
            self.assertEqual(archive._fetch_range(part, 0, 3), b"abc")
        self.assertEqual(archive._client.calls, 2)
        self.assertEqual(archive.range_requests, 1)


class CorpusPlanningTests(unittest.TestCase):
    def test_video_group_parsing_keeps_hyphenated_video_ids(self) -> None:
        self.assertEqual(video_group_id("abc-def--19"), "abc-def-")
        self.assertEqual(video_group_id("1782bea75c7d-7"), "1782bea75c7d")
        self.assertEqual(video_group_id("isolated_word_a12"), "isolated_word")

    def test_grouped_split_is_stable(self) -> None:
        self.assertEqual(grouped_split("video-1"), grouped_split("video-1"))
        self.assertIn(grouped_split("video-2"), {"train", "val", "test"})

    def test_balanced_subset_preserves_group_isolation(self) -> None:
        records = []
        for index in range(600):
            video_id = f"video-{index}"
            tokens = ("rare", f"token{index}", "three", "four", "five")
            records.append(
                CaptionRecord(
                    uid=f"{video_id}-1",
                    text=" ".join(tokens),
                    normalized_text=" ".join(tokens),
                    tokens=tokens,
                    video_id=video_id,
                    split=grouped_split(video_id),
                )
            )
        selected = balanced_sentence_subset(records, 100)
        self.assertEqual(len(selected), 100)
        by_video = {}
        for row in selected:
            by_video.setdefault(row.video_id, set()).add(row.split)
        self.assertTrue(all(len(splits) == 1 for splits in by_video.values()))

    def test_resampling_and_normalization_are_finite(self) -> None:
        values = np.zeros((4, 93, 2), dtype=np.float32)
        values[:, :, 0] = np.arange(4, dtype=np.float32)[:, None]
        values[:, 0, 1] = -1
        values[:, 11] = (0, 0)
        values[:, 12] = (2, 0)
        confidence = np.ones((4, 93), dtype=np.float32)
        resampled, resampled_confidence = resample_track(values, confidence, 8)
        normalized = normalize_for_training(resampled, resampled_confidence)
        self.assertEqual(resampled.shape, (8, 93, 2))
        self.assertEqual(resampled_confidence.shape, (8, 93))
        self.assertTrue(np.isfinite(normalized).all())

    def test_byte_parser_is_unambiguous(self) -> None:
        self.assertEqual(parse_bytes("2GiB"), 2 * 1024**3)
        self.assertEqual(parse_bytes("2GB"), 2 * 1000**3)

    def test_word_motion_trimming_keeps_the_high_energy_window(self) -> None:
        values = np.zeros((200, 93, 2), dtype=np.float32)
        confidence = np.ones((200, 93), dtype=np.float32)
        values[:, 11] = (0, 0)
        values[:, 12] = (1, 0)
        values[90:120, 33:75, 0] = np.arange(30, dtype=np.float32)[:, None]
        trimmed, trimmed_confidence = trim_motion_window(values, confidence, fps=25)
        self.assertLessEqual(trimmed.shape[0], 80)
        self.assertEqual(trimmed.shape[:2], trimmed_confidence.shape)
        self.assertGreater(float(np.ptp(trimmed[:, 33:75, 0])), 0)


if __name__ == "__main__":
    unittest.main()
