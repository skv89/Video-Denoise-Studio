from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from video_denoise_studio.models import PreviewRequest
from video_denoise_studio.preview import PreviewError, PreviewRenderer
from video_denoise_studio.timeline import (
    frame_from_timeline_position,
    source_frame_count,
    source_video_duration,
    timeline_render_delay_ms,
)

from video_denoise_tests.helpers import fake_capabilities, fake_media


class PreviewContractTests(unittest.TestCase):
    def test_matroska_video_duration_tag_prevents_audio_tail_phantom_frame(self) -> None:
        media = fake_media(Path("source.mkv"))
        video = replace(media.video, nb_frames=None, duration=None, tags={"DURATION": "00:00:10.010000000"})
        media = replace(media, duration=10.045, streams=(video, *media.streams[1:]))
        self.assertAlmostEqual(source_video_duration(media), 10.01)
        self.assertEqual(source_frame_count(media), 300)

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "source.mkv"
        self.source.write_bytes(b"source")
        self.media = fake_media(self.source)
        self.caps = fake_capabilities()
        self.renderer = PreviewRenderer()

    def tearDown(self) -> None:
        self.renderer.close()
        self.temp.cleanup()

    def request(self, **values) -> PreviewRequest:
        defaults = dict(
            source=self.source,
            media=self.media,
            capabilities=self.caps,
            denoiser="ffmpeg_atadenoise",
            strength=4,
            temporal_radius=3,
            target_frame=100,
        )
        defaults.update(values)
        return PreviewRequest(**defaults)

    def test_temporal_preview_derives_exact_hidden_context_for_one_target(self) -> None:
        target, start, count, leading, trailing, total = self.renderer._context(
            self.request(), 30000 / 1001
        )
        self.assertEqual((target, start, count, leading, trailing, total), (100, 97, 7, 3, 3, 300))

    def test_radius_four_and_six_derive_exact_nine_and_thirteen_frame_windows(self) -> None:
        radius_four = self.renderer._context(self.request(temporal_radius=4), 30000 / 1001)
        radius_six = self.renderer._context(self.request(temporal_radius=6), 30000 / 1001)
        self.assertEqual(radius_four, (100, 96, 9, 4, 4, 300))
        self.assertEqual(radius_six, (100, 94, 13, 6, 6, 300))

    def test_source_boundaries_reduce_only_unavailable_context(self) -> None:
        first = self.renderer._context(self.request(target_frame=1), 30000 / 1001)
        last = self.renderer._context(self.request(target_frame=299), 30000 / 1001)
        self.assertEqual(first, (1, 0, 5, 1, 3, 300))
        self.assertEqual(last, (299, 296, 4, 3, 0, 300))

    def test_all_six_denoisers_derive_their_actual_context(self) -> None:
        expected = {
            "ffmpeg_fftdnoiz": (1, 3),
            "ffmpeg_atadenoise": (3, 7),
            "vs_bm3d": (3, 7),
            "vs_dfttest": (3, 7),
            "vs_mvtools": (3, 7),
            "vs_nlmeans": (3, 7),
        }
        for identifier, (radius, count) in expected.items():
            with self.subTest(identifier=identifier):
                context = self.renderer._context(
                    self.request(denoiser=identifier, temporal_radius=radius), 30000 / 1001
                )
                self.assertEqual((context[2], context[3], context[4]), (count, radius, radius))

    def test_ffmpeg_comparison_fast_seeks_and_trims_to_one_aligned_target(self) -> None:
        directory = self.root / "preview"
        (directory / "original").mkdir(parents=True)
        (directory / "processed").mkdir()
        command = self.renderer._ffmpeg_comparison_command(
            self.request(), directory, 97, 7, 3, 30000 / 1001
        )
        graph = command[command.index("-filter_complex") + 1]
        self.assertIn("trim=start_frame=0:end_frame=7", graph)
        self.assertIn("split=2[base][work]", graph)
        self.assertEqual(graph.count("trim=start_frame=3:end_frame=4"), 2)
        self.assertIn("atadenoise=", graph)
        self.assertIn("-ss", command)
        self.assertEqual(command.count("-map"), 2)

    def test_ffmpeg_preview_graph_uses_current_strength_and_radius(self) -> None:
        directory = self.root / "preview-current-controls"
        (directory / "original").mkdir(parents=True)
        (directory / "processed").mkdir()
        request = self.request(strength=8, temporal_radius=4)
        command = self.renderer._ffmpeg_comparison_command(
            request, directory, 96, 9, 4, 30000 / 1001
        )
        graph = command[command.index("-filter_complex") + 1]
        self.assertIn("trim=start_frame=0:end_frame=9", graph)
        self.assertEqual(graph.count("trim=start_frame=4:end_frame=5"), 2)
        self.assertIn("atadenoise=0a=0.0400:0b=0.0800", graph)
        self.assertIn(":s=9:", graph)

    def test_source_only_preview_fast_seeks_and_limits_output_to_one_frame(self) -> None:
        directory = self.root / "source-only"
        (directory / "original").mkdir(parents=True)
        command = self.renderer._original_command(self.request(), directory, 100, 30000 / 1001)
        self.assertIn("-ss", command)
        self.assertIn("-frames:v", command)
        self.assertEqual(command[command.index("-frames:v") + 1], "1")
        self.assertNotIn("scale=", " ".join(command))

    def test_vapoursynth_preview_stacks_one_aligned_original_and_processed_target(self) -> None:
        directory = self.root / "preview-vs"
        (directory / "original").mkdir(parents=True)
        (directory / "processed").mkdir()
        request = self.request(denoiser="vs_bm3d")
        vspipe, ffmpeg, script_path = self.renderer._vspipe_comparison(
            request, directory, 97, 7, 3, "bm3dcpu"
        )
        script = script_path.read_text(encoding="utf-8")
        self.assertIn("source = source[97:104]", script)
        self.assertIn("source = source[3:4]", script)
        self.assertIn("processed = processed[3:4]", script)
        self.assertIn("StackHorizontal([source, processed])", script)
        self.assertIn("--container", vspipe)
        self.assertEqual(ffmpeg.count("-map"), 2)

    def test_vapoursynth_preview_script_uses_current_strength_and_thirteen_frame_window(self) -> None:
        directory = self.root / "preview-vs-current-controls"
        (directory / "original").mkdir(parents=True)
        (directory / "processed").mkdir()
        request = self.request(denoiser="vs_bm3d", strength=8, temporal_radius=6)
        _vspipe, _ffmpeg, script_path = self.renderer._vspipe_comparison(
            request, directory, 94, 13, 6, "bm3dcpu"
        )
        script = script_path.read_text(encoding="utf-8")
        self.assertIn("source = source[94:107]", script)
        self.assertIn("sigma=2.25, tr=6", script)
        self.assertIn("source = source[6:7]", script)
        self.assertIn("processed = processed[6:7]", script)

    def test_interlaced_dfttest_preview_filters_parities_and_reweaves_before_target_trim(self) -> None:
        directory = self.root / "preview-dft-interlaced"
        (directory / "original").mkdir(parents=True)
        (directory / "processed").mkdir()
        request = self.request(
            media=fake_media(self.source, field_order="tb", interlaced=64),
            denoiser="vs_dfttest",
            strength=10,
            temporal_radius=3,
        )
        _vspipe, _ffmpeg, script_path = self.renderer._vspipe_comparison(
            request, directory, 97, 7, 3, "dfttest_nvrtc"
        )
        script = script_path.read_text(encoding="utf-8")
        self.assertIn("source = source[97:104]", script)
        self.assertIn("SeparateFields(processed, tff=True)", script)
        self.assertIn("processed_first_parity", script)
        self.assertIn("processed_second_parity", script)
        self.assertEqual(script.count("DFTTest(backend=DFTTest.Backend.NVRTC).denoise"), 2)
        self.assertEqual(script.count("tr=3, sigma=20.00"), 2)
        self.assertIn("DoubleWeave(processed_fields, tff=True)", script)
        self.assertIn("SetFieldBased(processed, value=2)", script)
        self.assertIn("processed = processed[3:4]", script)

    def test_bff_dfttest_preview_restores_bottom_field_order(self) -> None:
        directory = self.root / "preview-dft-bff"
        (directory / "original").mkdir(parents=True)
        (directory / "processed").mkdir()
        request = self.request(
            media=fake_media(self.source, field_order="bt", interlaced=64),
            denoiser="vs_dfttest",
            strength=4,
            temporal_radius=3,
        )
        _vspipe, _ffmpeg, script_path = self.renderer._vspipe_comparison(
            request, directory, 97, 7, 3, "dfttest_cpu"
        )
        script = script_path.read_text(encoding="utf-8")
        self.assertIn("SeparateFields(processed, tff=False)", script)
        self.assertIn("DoubleWeave(processed_fields, tff=False)", script)
        self.assertIn("SetFieldBased(processed, value=1)", script)

    def test_absolute_timeline_position_maps_to_clicked_frame(self) -> None:
        total = 1001
        width = 1001
        expected = ((0, 0), (250, 250), (500, 500), (750, 750), (1000, 1000))
        for pointer_x, frame in expected:
            with self.subTest(pointer_x=pointer_x):
                self.assertEqual(frame_from_timeline_position(pointer_x, width, total), frame)
        self.assertEqual(frame_from_timeline_position(-50, width, total), 0)
        self.assertEqual(frame_from_timeline_position(5000, width, total), 1000)

    def test_source_only_timeline_seek_has_no_intentional_debounce(self) -> None:
        self.assertEqual(timeline_render_delay_ms(False, False), 0)
        self.assertEqual(timeline_render_delay_ms(False, True), 0)
        self.assertEqual(timeline_render_delay_ms(True, False), 400)
        self.assertEqual(timeline_render_delay_ms(True, True), 20)

    def test_source_raster_single_frame_storage_guard(self) -> None:
        request = self.request(width=3840, height=2160)
        with patch(
            "video_denoise_studio.preview.shutil.disk_usage",
            return_value=SimpleNamespace(free=256 * 1024 * 1024),
        ):
            with self.assertRaisesRegex(PreviewError, "Not enough temporary-drive space"):
                self.renderer._verify_preview_storage(request)


if __name__ == "__main__":
    unittest.main()
