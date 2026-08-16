from __future__ import annotations

import subprocess
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

from deinterlace_studio.capabilities import _inspect_vulkan_nnedi3
from deinterlace_studio.models import (
    CapabilityReport,
    IDetCounts,
    IDetReport,
    JobSettings,
    MediaProbe,
    OutputExpectation,
    StreamInfo,
)
from deinterlace_studio.planner import build_plan
from deinterlace_studio.presets import PROFILES, select_profile
from deinterlace_studio.scheduling import VapourSynthSchedule, choose_vapoursynth_schedule
from deinterlace_studio.settings import load_settings
from deinterlace_studio.validation import validate_output


def _schedule(*, requests: int, vulkan: bool = False) -> VapourSynthSchedule:
    mode = "Vulkan NNEDI3" if vulkan else "CPU NNEDI3"
    return VapourSynthSchedule(
        core_threads=16,
        requests=requests,
        logical_threads=32,
        memory_mib=98304,
        estimated_request_mib=144,
        memory_budget_mib=24576,
        rationale=f"Adaptive {mode} deterministic test schedule.",
    )


def _capabilities(*, vulkan: bool = True) -> CapabilityReport:
    formats: dict[str, tuple[str, ...]] = {}
    for profile in PROFILES.values():
        formats[profile.encoder] = tuple(
            dict.fromkeys(formats.get(profile.encoder, ()) + (profile.pix_fmt,))
        )
    return CapabilityReport(
        ffmpeg_path=Path("C:/Tools/ffmpeg.exe"),
        ffprobe_path=Path("C:/Tools/ffprobe.exe"),
        ffmpeg_version="ffmpeg version 9.0",
        ffmpeg_configuration="",
        filters=frozenset({"idet", "bwdif", "bwdif_cuda", "fftdnoiz", "atadenoise"}),
        encoders=frozenset(formats),
        encoder_pixel_formats=formats,
        hwaccels=frozenset({"cuda"}),
        vspipe_path=Path("C:/Tools/vspipe.exe"),
        vapoursynth_version="R79",
        qtgmc_ready=True,
        qtgmc_diagnostic="CPU graph passed.",
        qtgmc_install_command=None,
        vulkan_nnedi3_ready=vulkan,
        vulkan_nnedi3_diagnostic=(
            "Vulkan graph emitted 8 frames." if vulkan else "Vulkan 1.4 graph failed."
        ),
        vulkan_nnedi3_package_version="1.0" if vulkan else None,
    )


def _media(path: Path, pix_fmt: str = "yuv420p") -> MediaProbe:
    return MediaProbe(
        path=path,
        format_name="matroska",
        format_long_name="Matroska",
        duration=4.0,
        size=path.stat().st_size,
        bit_rate=1,
        start_time=0.0,
        streams=(
            StreamInfo(
                index=0,
                codec_type="video",
                codec_name="ffv1",
                width=720,
                height=576,
                pix_fmt=pix_fmt,
                bits_per_raw_sample=8,
                sample_aspect_ratio=Fraction(349, 240),
                display_aspect_ratio=Fraction(349, 192),
                r_frame_rate=Fraction(25, 1),
                avg_frame_rate=Fraction(25, 1),
                field_order="tt",
                nb_frames=100,
                color_range="pc",
                color_space="bt470bg",
                color_transfer="bt470bg",
                color_primaries="bt470bg",
            ),
        ),
    )


def _idet() -> IDetReport:
    return IDetReport(
        mode="sampled",
        segments=(),
        aggregate=IDetCounts(multi_tff=100),
        classification="tff",
        dominant_field_order="tff",
        confidence=1.0,
        rationale="test",
    )


class AdaptiveSchedulingTests(unittest.TestCase):
    def test_measured_sd_optima_are_selected_for_cpu_and_vulkan(self) -> None:
        cpu = choose_vapoursynth_schedule(
            720,
            576,
            "yuv420p",
            logical_threads=32,
            memory_mib=98304,
        )
        vulkan = choose_vapoursynth_schedule(
            720,
            576,
            "yuv420p",
            vulkan_nnedi3=True,
            logical_threads=32,
            memory_mib=98304,
        )
        self.assertEqual((cpu.core_threads, cpu.requests), (16, 24))
        self.assertEqual((vulkan.core_threads, vulkan.requests), (16, 16))
        self.assertIn("estimated", cpu.rationale)
        self.assertIn("bounded memory budget", cpu.rationale)

    def test_large_rasters_low_memory_and_temporal_graphs_reduce_requests(self) -> None:
        constrained = choose_vapoursynth_schedule(
            3840,
            2160,
            "yuv444p16le",
            temporal_denoise=True,
            logical_threads=64,
            memory_mib=4096,
        )
        self.assertLessEqual(constrained.requests, 12)
        self.assertGreaterEqual(constrained.requests, 1)
        # A single UHD 4:4:4 temporal request can itself exceed the deliberately
        # conservative 25% budget estimate.  Concurrency must still bottom out
        # at one (a runnable graph), never claim that multiple requests fit.
        if constrained.estimated_request_mib > (constrained.memory_budget_mib or 0):
            self.assertEqual(constrained.requests, 1)
        else:
            self.assertLessEqual(
                constrained.requests * constrained.estimated_request_mib,
                constrained.memory_budget_mib or 0,
            )
        no_memory_evidence = choose_vapoursynth_schedule(
            3840,
            2160,
            "yuv420p16le",
            logical_threads=64,
            memory_mib=0,
        )
        self.assertLessEqual(no_memory_evidence.requests, 8)


class FFV1NativeChromaTests(unittest.TestCase):
    def test_native_profile_maps_every_supported_subsampling(self) -> None:
        cases = {
            "yuvj420p": ("ffv1_intra_16_native_420", "yuv420p16le", "4:2:0"),
            "yuv422p10le": ("ffv1_intra_16_native_422", "yuv422p16le", "4:2:2"),
            "yuv444p12le": ("ffv1_intra_16_native_444", "yuv444p16le", "4:4:4"),
        }
        for source, expected in cases.items():
            with self.subTest(source=source):
                profile = select_profile(
                    "ffv1",
                    16,
                    False,
                    ffv1_chroma_mode="native",
                    source_pix_fmt=source,
                )
                self.assertEqual((profile.id, profile.pix_fmt, profile.chroma), expected)

    def test_explicit_444_is_retained_and_unknown_native_format_fails_closed(self) -> None:
        explicit = select_profile(
            "ffv1",
            16,
            False,
            ffv1_chroma_mode="444",
            source_pix_fmt="yuv420p",
        )
        self.assertEqual(explicit.id, "ffv1_intra_16")
        self.assertEqual(explicit.pix_fmt, "yuv444p16le")
        with self.assertRaisesRegex(ValueError, "cannot safely classify"):
            select_profile(
                "ffv1",
                16,
                False,
                ffv1_chroma_mode="native",
                source_pix_fmt="xyz12",
            )


class PlannerPerformanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "source.mkv"
        self.source.write_bytes(b"source")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _settings(self, **changes) -> JobSettings:
        values = {
            "input_path": self.source,
            "output_path": self.root / "out.mkv",
            "backend": "vapoursynth_qtgmc",
            "field_order": "auto",
            "output_cadence": "field_rate",
            "family": "ffv1",
            "bit_depth": 16,
            "ffv1_chroma_mode": "native",
            "denoise_enabled": False,
        }
        values.update(changes)
        return JobSettings(**values)

    def test_cpu_plan_uses_native_chroma_and_audited_adaptive_schedule(self) -> None:
        with patch(
            "deinterlace_studio.planner.choose_vapoursynth_schedule",
            return_value=_schedule(requests=24),
        ):
            plan = build_plan(
                self._settings(),
                _media(self.source),
                _idet(),
                _capabilities(),
                run_id="cpu",
            )
        self.assertTrue(plan.valid, plan.errors)
        self.assertEqual(plan.profile_id, "ffv1_intra_16_native_420")
        self.assertEqual(plan.expected.pix_fmts, ("yuv420p16le",))
        self.assertEqual(plan.vspipe_requests, 24)
        self.assertEqual(plan.vapoursynth_threads, 16)
        self.assertIn("24", plan.vapoursynth_schedule_note or "")
        self.assertNotIn("from vsaa import NNEDI3", plan.vapoursynth_script or "")
        self.assertIn("format=yuv420p16le", " ".join(plan.ffmpeg_command))

    def test_vulkan_plan_is_opt_in_graph_gated_and_uses_its_measured_schedule(self) -> None:
        with patch(
            "deinterlace_studio.planner.choose_vapoursynth_schedule",
            return_value=_schedule(requests=16, vulkan=True),
        ):
            plan = build_plan(
                self._settings(vulkan_nnedi3=True),
                _media(self.source),
                _idet(),
                _capabilities(vulkan=True),
                run_id="vk",
            )
        self.assertTrue(plan.valid, plan.errors)
        self.assertTrue(plan.vulkan_nnedi3_active)
        self.assertEqual(plan.vspipe_requests, 16)
        self.assertIn("from vsaa import NNEDI3", plan.vapoursynth_script or "")
        self.assertIn("basic_bobber=NNEDI3(nsize=1, gpu=True)", plan.vapoursynth_script or "")

        blocked = build_plan(
            self._settings(vulkan_nnedi3=True),
            _media(self.source),
            _idet(),
            _capabilities(vulkan=False),
            run_id="blocked",
        )
        self.assertFalse(blocked.valid)
        self.assertTrue(any("Vulkan NNEDI3 was selected" in error for error in blocked.errors))

        wrong_backend = build_plan(
            self._settings(backend="ffmpeg_bwdif", vulkan_nnedi3=True),
            _media(self.source),
            _idet(),
            _capabilities(vulkan=True),
            run_id="wrong",
        )
        self.assertFalse(wrong_backend.valid)
        self.assertTrue(any("applies only" in error for error in wrong_backend.errors))


class ConsolidatedValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.output = self.root / "out.mkv"
        self.output.write_bytes(b"output")
        self.video = StreamInfo(
            index=0,
            codec_type="video",
            codec_name="ffv1",
            width=720,
            height=576,
            pix_fmt="yuv420p16le",
            bits_per_raw_sample=16,
            sample_aspect_ratio=Fraction(349, 240),
            display_aspect_ratio=Fraction(349, 192),
            r_frame_rate=Fraction(50, 1),
            avg_frame_rate=Fraction(50, 1),
            field_order="progressive",
            nb_frames=200,
        )
        self.probe = MediaProbe(
            path=self.output,
            format_name="matroska",
            format_long_name="Matroska",
            duration=4.0,
            size=self.output.stat().st_size,
            bit_rate=1,
            start_time=0.0,
            streams=(self.video,),
        )
        self.expected = OutputExpectation(
            codec_names=("ffv1",),
            pix_fmts=("yuv420p16le",),
            width=720,
            height=576,
            sar=Fraction(349, 240),
            dar=Fraction(349, 192),
            frame_rate=Fraction(50, 1),
            progressive=True,
            lossless=True,
            bit_depth=16,
            expected_audio=(),
            expected_subtitles=(),
            expected_attachments=(),
            duration=4.0,
            frame_count=200,
        )
        self.settings = JobSettings(
            self.root / "source.mkv",
            self.output,
            family="ffv1",
            ffv1_chroma_mode="native",
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_one_ffv1_scan_proves_both_frame_count_and_all_intra(self) -> None:
        frames = [{"interlaced_frame": 0, "pix_fmt": "yuv420p16le"} for _ in range(8)]
        with patch("deinterlace_studio.validation.probe_media", return_value=self.probe), patch(
            "deinterlace_studio.validation.probe_frame_samples", return_value=frames
        ), patch(
            "deinterlace_studio.validation._count_intra_packet_flags", return_value=(200, 200)
        ) as flags, patch(
            "deinterlace_studio.validation.count_video_packets"
        ) as redundant:
            result = validate_output(
                Path("ffprobe.exe"),
                self.output,
                self.expected,
                self.settings,
                thorough_packet_count=True,
            )
        self.assertTrue(result.valid, result.errors)
        flags.assert_called_once()
        redundant.assert_not_called()
        self.assertEqual(result.verified_packet_count, 200)
        self.assertEqual(result.verified_key_packet_count, 200)
        self.assertTrue(result.thorough_packet_scan_completed)

    def test_bounded_final_reopen_never_repeats_a_full_packet_scan(self) -> None:
        frames = [{"interlaced_frame": 0, "pix_fmt": "yuv420p16le"} for _ in range(8)]
        with patch("deinterlace_studio.validation.probe_media", return_value=self.probe), patch(
            "deinterlace_studio.validation.probe_frame_samples", return_value=frames
        ), patch(
            "deinterlace_studio.validation._count_intra_packet_flags"
        ) as flags, patch(
            "deinterlace_studio.validation.count_video_packets"
        ) as packets:
            result = validate_output(
                Path("ffprobe.exe"),
                self.output,
                self.expected,
                self.settings,
                thorough_packet_count=False,
            )
        self.assertTrue(result.valid, result.errors)
        flags.assert_not_called()
        packets.assert_not_called()
        self.assertFalse(result.thorough_packet_scan_completed)


class VulkanCapabilityTests(unittest.TestCase):
    def test_real_graph_success_and_package_version_are_required(self) -> None:
        graph = subprocess.CompletedProcess([], 0, "", "Output 8 frames")
        version = subprocess.CompletedProcess([], 0, "1.0\n", "")
        with patch("deinterlace_studio.capabilities._run", side_effect=(graph, version)) as run, patch(
            "deinterlace_studio.capabilities._infer_vspipe_python",
            return_value=Path("C:/runtime/python.exe"),
        ):
            ready, diagnostic, package = _inspect_vulkan_nnedi3(Path("C:/runtime/vspipe.exe"))
        self.assertTrue(ready)
        self.assertEqual(package, "1.0")
        self.assertIn("emitted 8 frames", diagnostic)
        self.assertIn("--end", run.call_args_list[0].args[0])

        failure = subprocess.CompletedProcess([], 1, "", "Vulkan 1.4 is required")
        with patch("deinterlace_studio.capabilities._run", return_value=failure):
            ready, diagnostic, package = _inspect_vulkan_nnedi3(Path("vspipe.exe"))
        self.assertFalse(ready)
        self.assertIsNone(package)
        self.assertIn("Vulkan 1.4", diagnostic)


class SettingsMigrationTests(unittest.TestCase):
    def test_old_settings_receive_safe_new_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            path.write_text('{"denoise_enabled": true}', encoding="utf-8")
            values = load_settings(path)
        self.assertEqual(values["ffv1_chroma_mode"], "native")
        self.assertFalse(values["vulkan_nnedi3"])


if __name__ == "__main__":
    unittest.main()
