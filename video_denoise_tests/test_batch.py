from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from video_processing_core.media.models import ValidationResult

from video_denoise_studio.batch import BatchQueue, BatchRunner
from video_denoise_studio.models import BatchRunOptions, DenoiseResult, DenoiseSettings

from video_denoise_tests.helpers import fake_capabilities


class BatchQueueTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def video(self, name: str) -> Path:
        path = self.root / name
        path.write_bytes(b"video")
        return path

    def test_add_duplicate_reorder_remove_and_capacity(self) -> None:
        first = self.video("first.mkv")
        second = self.video("second.mov")
        unsupported = self.video("notes.txt")
        queue = BatchQueue(maximum=2)
        result = queue.add_paths((first, second, unsupported, first))
        self.assertEqual(len(result.added), 2)
        self.assertEqual(result.unsupported, (unsupported,))
        self.assertEqual(result.duplicates, (first,))
        identifiers = [record.identifier for record in queue.records]
        queue.move((identifiers[1],), -1)
        self.assertEqual(queue.records[0].source_path, second.resolve())
        removed = queue.remove((identifiers[0],))
        self.assertEqual(removed[0].source_path, first.resolve())

    def test_folder_scan_obeys_subfolder_switch(self) -> None:
        self.video("top.mkv")
        nested = self.root / "nested"
        nested.mkdir()
        (nested / "inside.mp4").write_bytes(b"video")
        queue = BatchQueue()
        self.assertEqual(len(queue.add_paths((self.root,), include_subfolders=False).added), 1)
        queue.clear()
        self.assertEqual(len(queue.add_paths((self.root,), include_subfolders=True).added), 2)

    def test_runner_streams_diagnostics_and_surfaces_validation_detail(self) -> None:
        source = self.video("failure.mp4")
        queue = BatchQueue()
        record = queue.add_paths((source,)).added[0]
        output = self.root / "failure.denoised.mp4"
        settings = DenoiseSettings(source, output)
        plan = SimpleNamespace(
            expected=SimpleNamespace(frame_count=100),
            profile=SimpleNamespace(label="HEVC NVIDIA 10-bit — P7 UHQ"),
            selected_denoise_backend="vszipcu",
            container="mp4",
        )
        validation = ValidationResult(
            valid=False,
            errors=("First exact validation error.", "Second exact validation error."),
            warnings=(),
            output_probe=None,
        )
        result = DenoiseResult(
            success=False,
            canceled=False,
            message="DenoiseProcessingError: The partial output failed validation.",
            output_path=None,
            log_path=self.root / "failure.denoised.mp4.Denoise.log",
            report_path=self.root / "failure.denoised.mp4.Denoise.json",
            script_path=None,
            output_sha256=None,
            validation=validation,
            quarantine_path=self.root / "failure.rejected.mp4",
        )
        events: list[tuple[str, object, object]] = []
        runner = BatchRunner(event_callback=lambda kind, row, payload: events.append((kind, row, payload)))
        runner._resolve_row = Mock(return_value=(settings, plan, ()))

        class FakeProcessor:
            def run(self, _plan, *, log_callback=None, progress_callback=None):
                if log_callback:
                    log_callback("Validation error: First exact validation error.")
                return result

            def cancel(self) -> None:
                pass

        with patch("video_denoise_studio.batch.DenoiseProcessor", FakeProcessor):
            summary = runner.run(queue, settings, fake_capabilities(), BatchRunOptions(None))

        self.assertEqual(summary.failed, 1)
        self.assertEqual(record.state, "Failed")
        self.assertIn("First exact validation error", record.error)
        self.assertIn("+1 more", record.error)
        rendered_log = "\n".join(str(payload) for kind, _row, payload in events if kind == "log")
        self.assertIn("Validation error: First exact validation error.", rendered_log)
        self.assertIn(str(result.log_path), rendered_log)
        self.assertIn(str(result.report_path), rendered_log)
        self.assertIn(str(result.quarantine_path), rendered_log)
        self.assertIn("Batch summary: completed 0/1; failed 1", rendered_log)


if __name__ == "__main__":
    unittest.main()
