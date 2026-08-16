from __future__ import annotations

import os
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from deinterlace_studio.health import (
    SourceHealthCancelled,
    _parse_compact_packet,
    assess_packet_timeline,
    health_details,
    health_headline,
    health_matches_source,
    scan_source_health,
)
from deinterlace_studio.models import JobSettings
from deinterlace_studio.planner import build_plan
from deinterlace_tests.test_core import capabilities, media, report


class SourceHealthClassificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.source = Path(self.temporary.name) / "影片 source.mkv"
        self.source.write_bytes(b"source identity")
        self.stat = self.source.stat()

    def _assess(self, source_media, pts, *, packet_count=None, warnings=(), structural=0, code=0, error=None):
        return assess_packet_timeline(
            source_media,
            source_size=self.stat.st_size,
            source_mtime_ns=self.stat.st_mtime_ns,
            packet_count=len(pts) if packet_count is None else packet_count,
            pts_values=pts,
            elapsed_seconds=1.25,
            warning_samples=warnings,
            demux_warning_count=len(warnings),
            structural_warning_count=structural,
            ffprobe_returncode=code,
            scan_error=error,
        )

    def test_material_field_packet_gap_requires_repair_without_counting_packets_as_frames(self) -> None:
        source_media = replace(media(self.source), duration=32.0)
        pts = [index * 0.02 for index in range(100)] + [30.0 + index * 0.02 for index in range(100)]
        health = self._assess(source_media, pts)
        self.assertEqual(health.status, "repair_required")
        self.assertEqual(health.packet_count, 200)
        self.assertEqual(health.unique_timestamp_count, 200)
        self.assertAlmostEqual(health.largest_gaps[0].duration, 28.02, places=6)
        self.assertIn("DAMAGE LIKELY", health_headline(health))
        self.assertIn("Automatic QTGMC recovery enabled", health_details(health))
        self.assertIn("Explicit FFmpeg BWDIF CPU/CUDA bypasses automatic repair", health_details(health))
        self.assertIn("never rewrites the selected source", health_details(health))

    def test_continuous_progressive_packet_timeline_is_clear(self) -> None:
        source_media = replace(media(self.source, progressive=True), duration=5.0)
        pts = [index * 0.04 for index in range(125)]
        health = self._assess(source_media, pts)
        self.assertEqual(health.status, "clear")
        self.assertFalse(health.repair_required)
        self.assertAlmostEqual(health.packet_timeline_span_seconds or 0.0, 5.0, places=6)
        self.assertIn("no obvious damage", health_headline(health))

    def test_duration_span_mismatch_requires_repair_even_without_internal_gap(self) -> None:
        base = media(self.source)
        source_media = replace(
            base,
            duration=10.0,
            streams=(replace(base.video, duration=10.0),) + base.streams[1:],
        )
        pts = [index * 0.04 for index in range(100)]
        health = self._assess(source_media, pts)
        self.assertEqual(health.status, "repair_required")
        self.assertEqual(health.material_gap_count, 0)
        self.assertAlmostEqual(health.duration_difference_seconds or 0.0, 6.0, places=6)

    def test_format_only_duration_mismatch_warns_instead_of_false_repair_block(self) -> None:
        source_media = replace(media(self.source), duration=10.0)
        pts = [index * 0.04 for index in range(100)]
        health = self._assess(source_media, pts)
        self.assertEqual(health.status, "warning")
        self.assertIn("no video-stream-specific duration", health.reason)

    def test_structural_diagnostic_warns_without_false_repair_requirement(self) -> None:
        source_media = replace(media(self.source), duration=4.0)
        pts = [index * 0.04 for index in range(100)]
        health = self._assess(
            source_media,
            pts,
            warnings=("Element exceeds containing master element",),
            structural=1,
        )
        self.assertEqual(health.status, "warning")
        self.assertFalse(health.repair_required)

    def test_failed_or_empty_scan_is_inconclusive(self) -> None:
        source_media = media(self.source)
        health = self._assess(source_media, [], packet_count=0, code=1, error="test failure")
        self.assertEqual(health.status, "inconclusive")
        self.assertIn("test failure", health.reason)

    def test_health_identity_invalidates_on_source_change(self) -> None:
        source_media = replace(media(self.source), duration=4.0)
        health = self._assess(source_media, [index * 0.04 for index in range(100)])
        self.assertTrue(health_matches_source(health, self.source))
        self.source.write_bytes(b"changed source identity")
        os.utime(self.source, None)
        self.assertFalse(health_matches_source(health, self.source))

    def test_compact_packet_parser_handles_missing_and_invalid_values(self) -> None:
        self.assertEqual(_parse_compact_packet("pts_time=1.250000|pos=4096\n"), (1.25, 4096))
        self.assertEqual(_parse_compact_packet("pts_time=N/A|pos=N/A\n"), (None, None))
        self.assertEqual(_parse_compact_packet("pts_time=nan|pos=-1\n"), (None, None))

    def test_pre_canceled_scan_does_not_start_ffprobe(self) -> None:
        canceled = threading.Event()
        canceled.set()
        with self.assertRaises(SourceHealthCancelled):
            scan_source_health(Path("missing-ffprobe.exe"), media(self.source), cancel_event=canceled)

    def test_active_scan_cancellation_terminates_ffprobe(self) -> None:
        class EmptyStderr:
            def __iter__(self):
                return iter(())

            def close(self):
                return None

        class SlowStdout:
            def __init__(self, owner):
                self.owner = owner

            def __iter__(self):
                for index in range(1000):
                    if self.owner.terminated:
                        break
                    time.sleep(0.005)
                    yield f"pts_time={index * 0.04:.6f}|pos={index * 188}\n"

            def close(self):
                return None

        class FakeProcess:
            def __init__(self):
                self.terminated = False
                self.returncode = None
                self.stdout = SlowStdout(self)
                self.stderr = EmptyStderr()

            def poll(self):
                return -15 if self.terminated else self.returncode

            def terminate(self):
                self.terminated = True
                self.returncode = -15

            def kill(self):
                self.terminate()

            def wait(self, timeout=None):
                self.returncode = -15 if self.terminated else 0
                return self.returncode

        cancel = threading.Event()
        fake = FakeProcess()
        timer = threading.Timer(0.03, cancel.set)
        timer.start()
        self.addCleanup(timer.cancel)
        with patch("deinterlace_studio.health.subprocess.Popen", return_value=fake):
            with self.assertRaises(SourceHealthCancelled):
                scan_source_health(Path("ffprobe.exe"), media(self.source), cancel_event=cancel, timeout=5)
        self.assertTrue(fake.terminated)


class SourceHealthPlannerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.source = Path(self.temporary.name) / "source.mkv"
        self.output = Path(self.temporary.name) / "output.mkv"
        self.source.write_bytes(b"source")
        self.media = replace(media(self.source), duration=32.0)
        stat = self.source.stat()
        pts = [index * 0.02 for index in range(100)] + [30.0 + index * 0.02 for index in range(100)]
        self.health = assess_packet_timeline(
            self.media,
            source_size=stat.st_size,
            source_mtime_ns=stat.st_mtime_ns,
            packet_count=len(pts),
            pts_values=pts,
            elapsed_seconds=1.0,
        )

    def _settings(self, backend: str) -> JobSettings:
        return JobSettings(
            input_path=self.source,
            output_path=self.output,
            backend=backend,
            family="ffv1",
            bit_depth=16,
            denoise_enabled=False,
        )

    def test_qtgmc_is_blocked_before_processing_when_repair_is_required(self) -> None:
        plan = build_plan(
            self._settings("auto"),
            self.media,
            report("tff"),
            capabilities(),
            source_health=self.health,
        )
        self.assertEqual(plan.selected_backend, "vapoursynth_qtgmc")
        self.assertFalse(plan.valid)
        self.assertTrue(any("Repair required" in error for error in plan.errors))
        self.assertIs(plan.source_health, self.health)

    def test_bwdif_remains_explicit_with_missing_picture_warning(self) -> None:
        plan = build_plan(
            self._settings("ffmpeg_bwdif"),
            self.media,
            report("tff"),
            capabilities(),
            source_health=self.health,
        )
        self.assertTrue(plan.valid, plan.errors)
        self.assertTrue(any("cannot restore missing/corrupt pictures" in warning for warning in plan.warnings))

    def test_stale_health_report_blocks_plan_and_is_not_serialized_as_current(self) -> None:
        self.source.write_bytes(b"changed after scan")
        plan = build_plan(
            self._settings("ffmpeg_bwdif"),
            self.media,
            report("tff"),
            capabilities(),
            source_health=self.health,
        )
        self.assertFalse(plan.valid)
        self.assertTrue(any("changed after its fast health precheck" in error for error in plan.errors))
        self.assertIsNone(plan.source_health)


if __name__ == "__main__":
    unittest.main()
