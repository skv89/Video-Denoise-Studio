from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from video_denoise_studio.models import DenoiseSettings
from video_denoise_studio.planner import build_plan, source_is_interlaced, unique_output_path

from video_denoise_tests.helpers import fake_capabilities, fake_media


class PlannerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "source.mkv"
        self.source.write_bytes(b"source")
        self.output = self.root / "source.denoised.mkv"
        self.caps = fake_capabilities()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_ffmpeg_plan_is_denoise_only_and_preserves_geometry(self) -> None:
        media = fake_media(self.source)
        settings = DenoiseSettings(self.source, self.output, denoiser="ffmpeg_atadenoise")
        plan = build_plan(settings, media, self.caps, run_id="test")
        self.assertTrue(plan.valid, plan.errors)
        command = " ".join(plan.ffmpeg_command)
        self.assertIn("atadenoise=", command)
        self.assertNotIn("bwdif", command)
        self.assertNotIn("deinterlace", command.casefold())
        self.assertNotIn("qtgmc", command.casefold())
        self.assertEqual((plan.expected.width, plan.expected.height), (1280, 720))
        self.assertEqual(plan.expected.frame_count, 300)
        self.assertTrue(plan.expected.progressive)
        self.assertEqual(plan.expected.expected_audio[0].codec_name, "aac")

    def test_vapoursynth_plan_contains_denoise_but_no_qtgmc(self) -> None:
        media = fake_media(self.source)
        plan = build_plan(DenoiseSettings(self.source, self.output), media, self.caps, run_id="test")
        self.assertTrue(plan.valid, plan.errors)
        self.assertIsNotNone(plan.vapoursynth_script)
        script = plan.vapoursynth_script.casefold()
        self.assertIn("bm3d", script)
        self.assertNotIn("qtempgaussmc", script)
        self.assertNotIn(".bob(", script)
        self.assertNotIn("deinterlace(", script)
        self.assertIn("no field separation or deinterlacing", script)

    def test_interlaced_tff_contract_is_preserved(self) -> None:
        media = fake_media(self.source, field_order="tt", interlaced=64)
        plan = build_plan(
            DenoiseSettings(self.source, self.output, denoiser="ffmpeg_atadenoise"),
            media,
            self.caps,
            run_id="test",
        )
        self.assertTrue(plan.valid, plan.errors)
        self.assertTrue(source_is_interlaced(media))
        self.assertFalse(plan.expected.progressive)
        self.assertIn("setfield=mode=tff", " ".join(plan.ffmpeg_command))
        self.assertIn("-field_order tt", " ".join(plan.ffmpeg_command))

    def test_dfttest_tff_uses_two_independent_parities_and_reweaves(self) -> None:
        media = fake_media(self.source, field_order="tb", interlaced=64)
        plan = build_plan(
            DenoiseSettings(
                self.source,
                self.output,
                denoiser="vs_dfttest",
                denoise_strength=10,
                denoise_temporal_radius=3,
            ),
            media,
            self.caps,
            run_id="dft-tff",
        )
        self.assertTrue(plan.valid, plan.errors)
        script = plan.vapoursynth_script or ""
        self.assertIn("SeparateFields(clip, tff=True)", script)
        self.assertIn("SelectEvery(clip_fields, cycle=2, offsets=0)", script)
        self.assertIn("SelectEvery(clip_fields, cycle=2, offsets=1)", script)
        self.assertEqual(script.count("DFTTest(backend=DFTTest.Backend.CPU).denoise"), 2)
        self.assertIn("clip_first_parity, tr=3, sigma=20.00", script)
        self.assertIn("clip_second_parity, tr=3, sigma=20.00", script)
        self.assertIn("Interleave([clip_first_parity, clip_second_parity], modify_duration=True)", script)
        self.assertIn("DoubleWeave(clip_fields, tff=True)", script)
        self.assertIn("SetFieldBased(clip, value=2)", script)
        self.assertIn("not deinterlacing", " ".join(plan.warnings).casefold())
        self.assertFalse(plan.expected.progressive)
        self.assertIn("setfield=mode=tff", " ".join(plan.ffmpeg_command))
        self.assertIn("-field_order tt", " ".join(plan.ffmpeg_command))

    def test_dfttest_bff_reweaves_bottom_field_first(self) -> None:
        media = fake_media(self.source, field_order="bt", interlaced=64)
        plan = build_plan(
            DenoiseSettings(self.source, self.output, denoiser="vs_dfttest", denoise_temporal_radius=3),
            media,
            self.caps,
            run_id="dft-bff",
        )
        self.assertTrue(plan.valid, plan.errors)
        script = plan.vapoursynth_script or ""
        self.assertIn("SeparateFields(clip, tff=False)", script)
        self.assertIn("DoubleWeave(clip_fields, tff=False)", script)
        self.assertIn("SetFieldBased(clip, value=1)", script)
        self.assertIn("setfield=mode=bff", " ".join(plan.ffmpeg_command))
        self.assertIn("-field_order bb", " ".join(plan.ffmpeg_command))

    def test_progressive_dfttest_and_interlaced_bm3d_keep_established_graphs(self) -> None:
        progressive = build_plan(
            DenoiseSettings(self.source, self.output, denoiser="vs_dfttest", denoise_temporal_radius=3),
            fake_media(self.source),
            self.caps,
            run_id="dft-progressive",
        )
        self.assertTrue(progressive.valid, progressive.errors)
        progressive_script = progressive.vapoursynth_script or ""
        self.assertEqual(progressive_script.count("DFTTest(backend=DFTTest.Backend.CPU).denoise"), 1)
        self.assertNotIn("SeparateFields", progressive_script)
        self.assertNotIn("DoubleWeave", progressive_script)

        bm3d = build_plan(
            DenoiseSettings(self.source, self.output, denoiser="vs_bm3d"),
            fake_media(self.source, field_order="tb", interlaced=64),
            self.caps,
            run_id="bm3d-interlaced",
        )
        self.assertTrue(bm3d.valid, bm3d.errors)
        self.assertNotIn("SeparateFields", bm3d.vapoursynth_script or "")
        self.assertNotIn("DoubleWeave", bm3d.vapoursynth_script or "")

    def test_unknown_interlaced_order_is_rejected(self) -> None:
        media = fake_media(self.source, field_order="unknown", interlaced=64)
        plan = build_plan(DenoiseSettings(self.source, self.output), media, self.caps, run_id="test")
        self.assertFalse(plan.valid)
        self.assertTrue(any("TFF/BFF" in error for error in plan.errors))

    def test_existing_output_and_sidecars_are_never_overwritten(self) -> None:
        media = fake_media(self.source)
        self.output.write_bytes(b"existing")
        plan = build_plan(DenoiseSettings(self.source, self.output), media, self.caps, run_id="test")
        self.assertFalse(plan.valid)
        self.assertTrue(any("already exists" in error for error in plan.errors))
        candidate = unique_output_path(self.output)
        self.assertNotEqual(candidate, self.output)
        self.assertFalse(candidate.exists())

    def test_retained_failure_sidecar_forces_a_new_retry_output_name(self) -> None:
        retained_log = self.output.with_name(self.output.name + ".Denoise.log")
        retained_log.write_text("previous failed run", encoding="utf-8")
        candidate = unique_output_path(self.output)
        self.assertEqual(candidate, self.root / "source.denoised-2.mkv")
        self.assertFalse(candidate.exists())

    def test_muxer_owned_mp4_identity_tags_are_regenerated_not_strictly_copied(self) -> None:
        media = replace(
            fake_media(self.source),
            format_name="mov,mp4,m4a,3gp,3g2,mj2",
            format_long_name="QuickTime / MOV",
            format_tags={
                "major_brand": "isom",
                "minor_version": "512",
                "compatible_brands": "isomiso2avc1mp41",
                "title": "Preserve this title",
                "comment": "Preserve this comment",
                "encoder": "source muxer",
            },
        )
        output = self.root / "source.denoised.mp4"
        settings = DenoiseSettings(
            self.source,
            output,
            denoiser="ffmpeg_atadenoise",
            denoise_temporal_radius=2,
            family="hevc",
            container="mp4",
            bit_depth=10,
            hardware_encode=True,
        )
        plan = build_plan(settings, media, self.caps, run_id="mp4-tags")
        self.assertTrue(plan.valid, plan.errors)
        self.assertEqual(
            plan.expected.expected_format_tags,
            {"title": "Preserve this title", "comment": "Preserve this comment"},
        )
        arguments = list(plan.ffmpeg_command)
        for tag in ("major_brand=", "minor_version=", "compatible_brands="):
            self.assertIn(tag, arguments)
            index = arguments.index(tag)
            self.assertEqual(arguments[index - 1], "-metadata")

    def test_mov_incompatible_attachment_falls_back_only_in_batch(self) -> None:
        media = fake_media(self.source)
        attachment = replace(media.streams[1], index=2, codec_type="attachment", codec_name="ttf")
        media = replace(media, streams=media.streams + (attachment,))
        output = self.root / "source.denoised.mov"
        settings = DenoiseSettings(self.source, output, denoiser="ffmpeg_atadenoise", family="prores", bit_depth=10)
        plan = build_plan(settings, media, self.caps, run_id="test")
        self.assertFalse(plan.valid)
        self.assertTrue(any("attachment" in error.casefold() for error in plan.errors))

    def test_hevc_nvenc_mp4_plan_uses_corrected_p7_uhq_contract(self) -> None:
        media = fake_media(self.source)
        output = self.root / "source.denoised.mp4"
        settings = DenoiseSettings(
            self.source,
            output,
            denoiser="ffmpeg_atadenoise",
            denoise_temporal_radius=2,
            family="hevc",
            container="mp4",
            bit_depth=10,
            hardware_encode=True,
            quality=16,
        )
        plan = build_plan(settings, media, self.caps, run_id="nvenc")
        self.assertTrue(plan.valid, plan.errors)
        self.assertEqual(plan.container, "mp4")
        command = " ".join(plan.ffmpeg_command)
        self.assertIn("-loglevel verbose", command)
        self.assertIn("-c:v hevc_nvenc", command)
        self.assertIn("-preset p7", command)
        self.assertIn("-tune uhq", command)
        self.assertIn("-rc vbr", command)
        self.assertIn("-cq 16", command)
        self.assertIn("-b:v 0", command)
        self.assertIn("-multipass fullres", command)
        self.assertIn("-temporal-aq 1", command)
        self.assertIn("-tag:v hvc1", command)
        self.assertIn("+faststart", command)
        self.assertNotIn("rc-lookahead", command)
        self.assertNotIn("spatial-aq", command)

    def test_av1_nvenc_mp4_uses_av01_tag_and_hardware_cq_range(self) -> None:
        media = fake_media(self.source)
        output = self.root / "source.denoised.mp4"
        settings = DenoiseSettings(
            self.source,
            output,
            denoiser="ffmpeg_atadenoise",
            denoise_temporal_radius=2,
            family="av1",
            container="mp4",
            bit_depth=10,
            hardware_encode=True,
            quality=63,
        )
        plan = build_plan(settings, media, self.caps, run_id="av1")
        self.assertTrue(plan.valid, plan.errors)
        command = " ".join(plan.ffmpeg_command)
        self.assertIn("-c:v av1_nvenc", command)
        self.assertIn("-tag:v av01", command)
        self.assertIn("-cq 63", command)
        invalid = build_plan(replace(settings, quality=64), media, self.caps, run_id="av1-invalid")
        self.assertFalse(invalid.valid)
        self.assertTrue(any("0 through 63" in error for error in invalid.errors))

    def test_ffv1_quality_value_is_irrelevant_and_not_forwarded(self) -> None:
        media = fake_media(self.source)
        plan = build_plan(
            DenoiseSettings(self.source, self.output, denoiser="ffmpeg_atadenoise", quality=999),
            media,
            self.caps,
            run_id="ffv1-quality",
        )
        self.assertTrue(plan.valid, plan.errors)
        command = " ".join(plan.ffmpeg_command)
        self.assertNotIn("-cq", command)
        self.assertNotIn("-crf", command)

    def test_explicit_mp4_rejects_attachment_with_mkv_recommendation(self) -> None:
        media = fake_media(self.source)
        attachment = replace(media.streams[1], index=2, codec_type="attachment", codec_name="ttf")
        media = replace(media, streams=media.streams + (attachment,))
        settings = DenoiseSettings(
            self.source,
            self.root / "source.denoised.mp4",
            denoiser="ffmpeg_atadenoise",
            denoise_temporal_radius=2,
            family="hevc",
            container="mp4",
            bit_depth=10,
        )
        plan = build_plan(settings, media, self.caps, run_id="mp4-attachment")
        self.assertFalse(plan.valid)
        self.assertTrue(any("Choose MKV" in error for error in plan.errors))


if __name__ == "__main__":
    unittest.main()
