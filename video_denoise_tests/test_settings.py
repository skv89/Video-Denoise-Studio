from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from video_denoise_studio.settings import (
    DEFAULTS,
    default_settings_path,
    load_settings,
    reference_settings_path,
    save_settings,
)


class SettingsTests(unittest.TestCase):
    def test_distinct_application_directory(self) -> None:
        self.assertEqual(default_settings_path().parent.name, "VideoDenoiseStudio")
        self.assertEqual(reference_settings_path().parent.name, "DeinterlaceStudio")

    def test_atomic_round_trip_and_type_guard(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            payload = dict(DEFAULTS)
            payload["container"] = "mp4"
            save_settings(payload, path)
            self.assertEqual(load_settings(path)["container"], "mp4")
            raw = json.loads(path.read_text(encoding="utf-8"))
            raw["container"] = 72
            path.write_text(json.dumps(raw), encoding="utf-8")
            self.assertEqual(load_settings(path)["container"], DEFAULTS["container"])

    def test_v1_settings_migrate_live_preview_to_frame_preview(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            path.write_text(
                json.dumps({"settings_schema_version": 1, "live_preview_enabled": True}),
                encoding="utf-8",
            )
            loaded = load_settings(path)
            self.assertTrue(loaded["frame_preview_enabled"])
            self.assertEqual(loaded["settings_schema_version"], 2)

    def test_first_run_reuses_only_reference_tool_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.json"
            reference.write_text(
                json.dumps(
                    {
                        "ffmpeg_path": "C:/tools/ffmpeg.exe",
                        "ffprobe_path": "C:/tools/ffprobe.exe",
                        "vspipe_path": "C:/tools/vspipe.exe",
                        "denoiser": "ffmpeg_nlmeans",
                        "denoise_strength": 10,
                    }
                ),
                encoding="utf-8",
            )
            loaded = load_settings(root / "new-settings.json", reference_path=reference)
            self.assertEqual(loaded["ffmpeg_path"], "C:/tools/ffmpeg.exe")
            self.assertEqual(loaded["ffprobe_path"], "C:/tools/ffprobe.exe")
            self.assertEqual(loaded["vspipe_path"], "C:/tools/vspipe.exe")
            self.assertEqual(loaded["denoiser"], DEFAULTS["denoiser"])
            self.assertEqual(loaded["denoise_strength"], DEFAULTS["denoise_strength"])

    def test_existing_denoise_settings_are_never_repopulated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            own = root / "own.json"
            reference = root / "reference.json"
            own.write_text(json.dumps({"ffmpeg_path": ""}), encoding="utf-8")
            reference.write_text(json.dumps({"ffmpeg_path": "C:/tools/ffmpeg.exe"}), encoding="utf-8")
            loaded = load_settings(own, reference_path=reference)
            self.assertEqual(loaded["ffmpeg_path"], "")


if __name__ == "__main__":
    unittest.main()
