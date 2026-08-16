from __future__ import annotations

import hashlib
import subprocess
import tempfile
import unittest
from pathlib import Path

from deinterlace_studio.capabilities import find_binary
from deinterlace_studio.compatibility import CompatibilityCopyRequest, MOVCompatibilityCopier
from deinterlace_studio.probe import probe_media


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest().upper()


class MOVCompatibilityCopyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ffmpeg = find_binary("ffmpeg")
        cls.ffprobe = cls.ffmpeg.with_name("ffprobe.exe") if cls.ffmpeg else None
        if cls.ffprobe and not cls.ffprobe.is_file():
            cls.ffprobe = find_binary("ffprobe")

    def test_real_prores_mkv_becomes_validated_mov_without_video_or_audio_reencode(self) -> None:
        if not self.ffmpeg or not self.ffprobe:
            self.skipTest("A paired FFmpeg/FFprobe installation is unavailable")
        with tempfile.TemporaryDirectory(prefix="DeinterlaceCompatibilityTest-") as directory:
            root = Path(directory)
            subtitle = root / "subtitle.srt"
            subtitle.write_text(
                "1\n00:00:00,050 --> 00:00:00,700\nCompatibility subtitle\n",
                encoding="utf-8",
            )
            source = root / "completed prores master.mkv"
            generated = subprocess.run(
                [
                    str(self.ffmpeg),
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc2=size=160x120:rate=25:duration=1",
                    "-f",
                    "lavfi",
                    "-i",
                    "sine=frequency=440:sample_rate=48000:duration=1",
                    "-f",
                    "srt",
                    "-i",
                    str(subtitle),
                    "-map",
                    "0:v:0",
                    "-map",
                    "1:a:0",
                    "-map",
                    "2:s:0",
                    "-vf",
                    "format=yuv444p10le",
                    "-c:v",
                    "prores_ks",
                    "-profile:v",
                    "4444xq",
                    "-bits_per_mb",
                    "8000",
                    "-alpha_bits",
                    "0",
                    "-c:a",
                    "mp3",
                    "-b:a",
                    "128k",
                    "-c:s",
                    "srt",
                    "-shortest",
                    str(source),
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=120,
                check=False,
            )
            self.assertEqual(generated.returncode, 0, generated.stderr)
            source_hash = sha256(source)
            output = root / "native compatibility.mov"
            result = MOVCompatibilityCopier().run(
                CompatibilityCopyRequest(source, output),
                self.ffmpeg,
                self.ffprobe,
            )
            self.assertTrue(result.success, result.message)
            self.assertFalse(result.canceled)
            self.assertEqual(sha256(source), source_hash)
            self.assertTrue(output.is_file())
            self.assertEqual(result.source_video_packets, result.output_video_packets)
            self.assertGreater(result.source_video_packets or 0, 0)
            self.assertEqual(result.copied_audio_tracks, 1)
            self.assertEqual(result.converted_subtitle_tracks, 1)
            self.assertTrue(result.video_essence_sha256)
            self.assertTrue(result.log_path and result.log_path.is_file())
            self.assertTrue(result.report_path and result.report_path.is_file())

            source_probe = probe_media(self.ffprobe, source, sample_frames=8)
            output_probe = probe_media(self.ffprobe, output, sample_frames=8)
            self.assertEqual(output_probe.video.codec_name, source_probe.video.codec_name)
            self.assertEqual(output_probe.video.pix_fmt, source_probe.video.pix_fmt)
            self.assertEqual(output_probe.audio_count, source_probe.audio_count)
            self.assertEqual(output_probe.streams_of_type("audio")[0].codec_name, "mp3")
            self.assertEqual(output_probe.subtitle_count, 1)
            self.assertEqual(output_probe.streams_of_type("subtitle")[0].codec_name, "mov_text")

    def test_existing_output_is_never_overwritten(self) -> None:
        if not self.ffmpeg or not self.ffprobe:
            self.skipTest("A paired FFmpeg/FFprobe installation is unavailable")
        with tempfile.TemporaryDirectory(prefix="DeinterlaceCompatibilityCollision-") as directory:
            root = Path(directory)
            source = root / "source.mkv"
            source.write_bytes(b"not needed because collision is checked before probing")
            output = root / "existing.mov"
            output.write_bytes(b"protected")
            result = MOVCompatibilityCopier().run(
                CompatibilityCopyRequest(source, output),
                self.ffmpeg,
                self.ffprobe,
            )
            self.assertFalse(result.success)
            self.assertIn("already exist", result.message)
            self.assertEqual(output.read_bytes(), b"protected")


if __name__ == "__main__":
    unittest.main()

