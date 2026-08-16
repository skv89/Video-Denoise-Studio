from __future__ import annotations

import sys
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from pathlib import Path

from video_denoise_studio.models import DenoiseSettings
from video_denoise_studio.planner import build_plan
from video_denoise_studio.probe import ProbeCancelled, _run_json
from video_denoise_studio.processor import DenoiseProcessor

from video_denoise_tests.helpers import fake_capabilities, fake_media


class CancellationTests(unittest.TestCase):
    def test_cancelable_probe_terminates_active_child(self) -> None:
        cancel = threading.Event()
        outcome: list[BaseException] = []

        def worker() -> None:
            try:
                _run_json([sys.executable, "-c", "import time; time.sleep(30)"], cancel, 60)
            except BaseException as exc:
                outcome.append(exc)

        thread = threading.Thread(target=worker)
        thread.start()
        time.sleep(0.2)
        cancel.set()
        thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        self.assertIsInstance(outcome[0], ProbeCancelled)

    def test_processor_cancel_removes_partial_and_never_promotes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mkv"
            source.write_bytes(b"source")
            output = root / "output.mkv"
            settings = DenoiseSettings(source, output, denoiser="ffmpeg_atadenoise")
            plan = build_plan(settings, fake_media(source), fake_capabilities(), run_id="cancel")
            self.assertTrue(plan.valid, plan.errors)
            slow = replace(
                plan,
                ffmpeg_command=(sys.executable, "-c", "import time; time.sleep(30)"),
                vspipe_command=None,
                vapoursynth_script=None,
                script_path=None,
                temporary_script_path=None,
            )
            processor = DenoiseProcessor()
            results = []
            thread = threading.Thread(target=lambda: results.append(processor.run(slow)))
            thread.start()
            deadline = time.monotonic() + 5
            while not slow.log_path.exists() and time.monotonic() < deadline:
                time.sleep(0.02)
            processor.cancel()
            thread.join(timeout=8)
            self.assertFalse(thread.is_alive())
            self.assertTrue(results[0].canceled)
            self.assertFalse(output.exists())
            self.assertFalse(slow.partial_path.exists())
            self.assertTrue(slow.log_path.exists())
            self.assertTrue(slow.report_path.exists())


if __name__ == "__main__":
    unittest.main()

