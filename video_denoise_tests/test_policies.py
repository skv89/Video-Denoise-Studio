from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from video_processing_core.media.models import StreamInfo

from video_denoise_studio.denoiser_policy import (
    denoiser_backend_status,
    denoiser_control_policy,
    denoiser_ranking,
    denoiser_rankings_guide,
    normalize_temporal_radius,
    validate_denoiser_controls,
)
from video_denoise_studio.models import DenoiseSettings
from video_denoise_studio.output_policy import (
    container_compatibility_errors,
    encoder_args,
    encoder_control_policy,
    resolve_container,
    select_output_profile,
    valid_container_ids,
)

from video_denoise_tests.helpers import fake_capabilities, fake_media


class DenoiserPolicyTests(unittest.TestCase):
    def test_strength_applies_to_all_six_with_native_mapping_help(self) -> None:
        identifiers = (
            "ffmpeg_fftdnoiz",
            "ffmpeg_atadenoise",
            "vs_bm3d",
            "vs_dfttest",
            "vs_mvtools",
            "vs_nlmeans",
        )
        for identifier in identifiers:
            with self.subTest(identifier=identifier):
                policy = denoiser_control_policy(identifier, 4, 3)
                self.assertIn("Strength 4/10", policy.strength_help)
                self.assertTrue(policy.overview)

    def test_radius_is_fixed_only_for_fftdnoiz_and_atadenoise_minimum_is_two(self) -> None:
        fixed = denoiser_control_policy("ffmpeg_fftdnoiz", 4, 6)
        self.assertFalse(fixed.radius_enabled)
        self.assertEqual((fixed.radius_minimum, fixed.radius_maximum), (1, 1))
        self.assertEqual((fixed.normalized_radius, fixed.window_frames), (1, 3))
        self.assertEqual(normalize_temporal_radius("ffmpeg_atadenoise", 1), 2)
        atadenoise = denoiser_control_policy("ffmpeg_atadenoise", 4, 2)
        self.assertEqual((atadenoise.radius_minimum, atadenoise.radius_maximum), (2, 6))
        self.assertEqual(atadenoise.window_frames, 5)
        self.assertTrue(validate_denoiser_controls("ffmpeg_atadenoise", 4, 1))
        self.assertFalse(validate_denoiser_controls("vs_bm3d", 4, 1))
        self.assertEqual(normalize_temporal_radius("vs_dfttest", 6), 3)
        dfttest = denoiser_control_policy("vs_dfttest", 4, 6)
        self.assertEqual((dfttest.radius_minimum, dfttest.radius_maximum), (1, 3))
        self.assertTrue(validate_denoiser_controls("vs_dfttest", 4, 4))
        for identifier in ("vs_bm3d", "vs_mvtools", "vs_nlmeans"):
            with self.subTest(identifier=identifier):
                policy = denoiser_control_policy(identifier, 4, 6)
                self.assertEqual((policy.radius_minimum, policy.radius_maximum), (1, 6))

    def test_quality_and_speed_rankings_cover_one_through_six_exactly(self) -> None:
        identifiers = (
            "ffmpeg_fftdnoiz",
            "ffmpeg_atadenoise",
            "vs_bm3d",
            "vs_dfttest",
            "vs_mvtools",
            "vs_nlmeans",
        )
        rankings = tuple(denoiser_ranking(identifier) for identifier in identifiers)
        self.assertEqual({ranking.quality_score for ranking in rankings}, set(range(1, 7)))
        self.assertEqual({ranking.speed_score for ranking in rankings}, set(range(1, 7)))
        self.assertIn("guidance—not universal lab scores", denoiser_rankings_guide())

    def test_backend_status_reports_actual_gpu_cpu_and_unavailable_routes(self) -> None:
        capabilities = fake_capabilities()
        cpu = denoiser_backend_status("vs_mvtools", capabilities, 1920, 1080)
        self.assertTrue(cpu.available)
        self.assertFalse(cpu.gpu_active)
        self.assertEqual(cpu.classification, "CPU only")
        capabilities.denoise_backends["vs_bm3d"] = "vszipcu"
        gpu = denoiser_backend_status("vs_bm3d", capabilities, 1920, 1080)
        self.assertTrue(gpu.gpu_active)
        self.assertEqual(gpu.classification, "NVIDIA GPU active")
        self.assertIn("Automatic", gpu.summary)
        capabilities.denoise_capabilities["vs_nlmeans"] = False
        unavailable = denoiser_backend_status("vs_nlmeans", capabilities)
        self.assertFalse(unavailable.available)
        self.assertEqual(unavailable.classification, "unavailable")


class OutputPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "source.mkv"
        self.source.write_bytes(b"source")
        self.media = fake_media(self.source)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def settings(self, **values) -> DenoiseSettings:
        defaults = dict(input_path=self.source, output_path=self.root / "out.mkv")
        defaults.update(values)
        return DenoiseSettings(**defaults)

    def test_container_matrix_and_source_aware_auto_recommendation(self) -> None:
        self.assertEqual(valid_container_ids("ffv1"), ("auto", "mkv"))
        self.assertEqual(valid_container_ids("prores"), ("auto", "mov"))
        self.assertEqual(resolve_container(self.settings(family="hevc", bit_depth=10), self.media).identifier, "mp4")
        self.assertEqual(resolve_container(self.settings(family="av1", bit_depth=10), self.media).identifier, "mkv")
        subtitle = StreamInfo(index=2, codec_type="subtitle", codec_name="subrip")
        with_subtitle = replace(self.media, streams=self.media.streams + (subtitle,))
        self.assertEqual(resolve_container(self.settings(family="hevc", bit_depth=10), with_subtitle).identifier, "mkv")

    def test_codec_controls_match_actual_encoder(self) -> None:
        ffv1, _ = select_output_profile(self.settings(), self.media)
        ffv1_policy = encoder_control_policy(ffv1, "ffv1", False)
        self.assertFalse(ffv1_policy.quality_enabled)
        self.assertFalse(ffv1_policy.tune_grain_enabled)
        prores, _ = select_output_profile(
            self.settings(family="prores", bit_depth=10, container="mov"), self.media
        )
        self.assertFalse(encoder_control_policy(prores, "prores", False).quality_enabled)
        x265, _ = select_output_profile(
            self.settings(family="hevc", bit_depth=10, container="mkv"), self.media
        )
        self.assertTrue(encoder_control_policy(x265, "hevc", False).tune_grain_enabled)

    def test_nvenc_contract_is_p7_uhq_temporal_aq_without_explicit_lookahead(self) -> None:
        for family in ("hevc", "av1"):
            with self.subTest(family=family):
                profile, _ = select_output_profile(
                    self.settings(
                        family=family,
                        bit_depth=10,
                        hardware_encode=True,
                        container="mkv",
                    ),
                    self.media,
                )
                args = encoder_args(profile, 16, True)
                rendered = " ".join(args)
                self.assertIn("-preset p7", rendered)
                self.assertIn("-tune uhq", rendered)
                self.assertIn("-rc vbr", rendered)
                self.assertIn("-cq 16", rendered)
                self.assertIn("-multipass fullres", rendered)
                self.assertIn("-temporal-aq 1", rendered)
                self.assertNotIn("rc-lookahead", rendered)
                self.assertNotIn("lookahead_level", rendered)
                self.assertNotIn("spatial-aq", rendered)

    def test_iso_containers_reject_unpreservable_selected_tracks(self) -> None:
        attachment = StreamInfo(index=2, codec_type="attachment", codec_name="ttf")
        media = replace(self.media, streams=self.media.streams + (attachment,))
        errors = container_compatibility_errors(
            self.settings(family="hevc", bit_depth=10, container="mp4"), media, "mp4"
        )
        self.assertTrue(any("attachment" in error.casefold() and "MKV" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
