from __future__ import annotations

import subprocess
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

from deinterlace_main import _windows_extended_tcl_path
from unittest.mock import patch

from deinterlace_studio.capabilities import (
    _d3d12_error_excerpt,
    _inspect_ffmpeg_interlace_runtime,
    _inspect_vapoursynth_denoisers,
    _parse_named_components,
    _parse_pixel_formats,
)
from deinterlace_studio.denoise import (
    DFTTEST_ADAPTIVE_CPU_CUFFT,
    DFTTEST_ADAPTIVE_CPU_NVRTC,
    DFTTEST_ADAPTIVE_NVRTC_CUFFT,
    resolve_denoiser_backend,
    vapoursynth_denoise_lines,
)
from deinterlace_studio.idet import classify_idet, parse_idet_output, sample_intervals
from deinterlace_studio.models import (
    CapabilityReport,
    IDetCounts,
    IDetReport,
    JobSettings,
    MediaProbe,
    StreamInfo,
)
from deinterlace_studio.planner import (
    QTGMC_CORE_THREAD_CAP,
    QTGMC_VSPIPE_REQUESTS,
    _execution_vspipe_path,
    build_plan,
)
from deinterlace_studio.presets import (
    PROFILES,
    nvenc_maximum_quality_args,
    profile_capability_error,
    selectable_bit_depths,
)
from deinterlace_studio.probe import count_video_frames, parse_probe_json, pixel_format_depth
from deinterlace_studio.rationals import derive_dar, exact_square_pixel_raster, parse_fraction


class FrozenTclPathTests(unittest.TestCase):
    def test_windows_drive_paths_use_extended_prefix(self) -> None:
        self.assertEqual(
            _windows_extended_tcl_path(r"C:\Users\ExampleUser\AppData\Local\Temp\_MEI123\_tcl_data"),
            "//?/C:/Users/ExampleUser/AppData/Local/Temp/_MEI123/_tcl_data",
        )
        self.assertEqual(
            _windows_extended_tcl_path("//?/C:/Users/ExampleUser/AppData/Local/Temp/_MEI123/_tk_data"),
            "//?/C:/Users/ExampleUser/AppData/Local/Temp/_MEI123/_tk_data",
        )
        self.assertEqual(_windows_extended_tcl_path("relative/tcl"), "relative/tcl")


def capabilities(*, qtgmc: bool = True, dnx12: bool = False) -> CapabilityReport:
    encoders = {profile.encoder for profile in PROFILES.values()}
    formats: dict[str, tuple[str, ...]] = {}
    for profile in PROFILES.values():
        formats.setdefault(profile.encoder, tuple())
        formats[profile.encoder] = tuple(dict.fromkeys(formats[profile.encoder] + (profile.pix_fmt,)))
    if not dnx12:
        formats["dnxhd"] = ("yuv422p", "yuv422p10le", "yuv444p10le", "gbrp10le")
    return CapabilityReport(
        ffmpeg_path=Path("C:/Tools/ffmpeg.exe"),
        ffprobe_path=Path("C:/Tools/ffprobe.exe"),
        ffmpeg_version="ffmpeg version test",
        ffmpeg_configuration="",
        filters=frozenset({"bwdif", "bwdif_cuda", "idet", "zscale", "fftdnoiz", "atadenoise"}),
        encoders=frozenset(encoders),
        encoder_pixel_formats=formats,
        hwaccels=frozenset({"cuda"}),
        vspipe_path=Path("C:/Tools/vspipe.exe"),
        vapoursynth_version="R78",
        qtgmc_ready=qtgmc,
        qtgmc_diagnostic="ready" if qtgmc else "missing plugins",
        qtgmc_install_command="python -m pip install vsjetpack[deinterlace]",
        gpu_name="RTX PRO 6000",
        gpu_memory_mib=97887,
        gpu_driver="610.88",
        encoder_verified_bit_depths={"hevc_nvenc": (10, 12), "av1_nvenc": (10, 12)},
        encoder_runtime_diagnostics={"hevc_nvenc": "test", "av1_nvenc": "test"},
        denoise_capabilities={
            "ffmpeg_fftdnoiz": True,
            "ffmpeg_atadenoise": True,
            "vs_bm3d": True,
            "vs_dfttest": True,
            "vs_mvtools": True,
            "vs_nlmeans": True,
        },
        denoise_backends={
            "ffmpeg_fftdnoiz": "ffmpeg",
            "ffmpeg_atadenoise": "ffmpeg",
            "vs_bm3d": "bm3dcpu",
            "vs_dfttest": "dfttest_cpu",
            "vs_mvtools": "mvtools",
            "vs_nlmeans": "nlm_ispc",
        },
        denoise_diagnostics={identifier: "graph passed" for identifier in (
            "ffmpeg_fftdnoiz", "ffmpeg_atadenoise", "vs_bm3d", "vs_dfttest", "vs_mvtools", "vs_nlmeans"
        )},
    )


def media(
    path: Path,
    *,
    progressive: bool = False,
    subtitle: str | None = None,
    audio: str = "ac3",
) -> MediaProbe:
    streams = [
        StreamInfo(
            index=0,
            codec_type="video",
            codec_name="h264",
            width=720,
            height=576,
            pix_fmt="yuvj420p",
            bits_per_raw_sample=8,
            sample_aspect_ratio=Fraction(349, 240),
            display_aspect_ratio=Fraction(349, 192),
            r_frame_rate=Fraction(25, 1),
            avg_frame_rate=Fraction(25, 1),
            field_order="progressive" if progressive else "tt",
            color_range="pc",
            color_space="bt470bg",
            color_transfer="bt470bg",
            color_primaries="bt470bg",
        ),
        StreamInfo(index=1, codec_type="audio", codec_name=audio, tags={"language": "chi"}),
    ]
    if subtitle:
        streams.append(StreamInfo(index=2, codec_type="subtitle", codec_name=subtitle, tags={"language": "chi"}))
    return MediaProbe(
        path=path,
        format_name="matroska",
        format_long_name="Matroska",
        duration=60.0,
        size=1000,
        bit_rate=1000000,
        start_time=0.0,
        streams=tuple(streams),
    )


def report(classification: str) -> IDetReport:
    if classification == "progressive":
        counts = IDetCounts(multi_progressive=1000)
        order = None
    elif classification == "tff":
        counts = IDetCounts(multi_tff=990, multi_progressive=10)
        order = "tff"
    elif classification == "bff":
        counts = IDetCounts(multi_bff=990, multi_progressive=10)
        order = "bff"
    else:
        counts = IDetCounts(multi_tff=400, multi_bff=20, multi_progressive=400)
        order = None
    return IDetReport(
        mode="sampled",
        segments=(),
        aggregate=counts,
        classification=classification,
        dominant_field_order=order,
        confidence=0.99,
        rationale="test evidence",
    )


class RationalTests(unittest.TestCase):
    def test_exact_rational_and_dar(self) -> None:
        self.assertEqual(parse_fraction("349:240"), Fraction(349, 240))
        self.assertEqual(derive_dar(720, 576, Fraction(349, 240)), Fraction(349, 192))

    def test_exact_square_pixel_raster_never_downscales(self) -> None:
        self.assertEqual(exact_square_pixel_raster(Fraction(349, 192), 720, 576), (1396, 768))
        self.assertEqual(exact_square_pixel_raster(Fraction(16, 9), 1920, 1080), (1920, 1080))


class ProbeTests(unittest.TestCase):
    @patch("deinterlace_studio.probe._run_json")
    def test_decoded_frame_count_does_not_use_field_packet_count(self, run_json) -> None:
        run_json.return_value = {
            "streams": [{"nb_read_frames": "1498", "nb_read_packets": "2988"}]
        }
        counted = count_video_frames(Path("ffprobe.exe"), Path("field-coded.mkv"))
        self.assertEqual(counted, 1498)
        command = run_json.call_args.args[0]
        self.assertIn("-count_frames", command)
        self.assertNotIn("-count_packets", command)

    def test_probe_parser_preserves_unicode_exact_rates_and_frame_flags(self) -> None:
        path = Path("D:/影片/節目.mkv")
        payload = {
            "streams": [
                {
                    "index": 0,
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": 720,
                    "height": 576,
                    "pix_fmt": "yuvj420p",
                    "sample_aspect_ratio": "349:240",
                    "display_aspect_ratio": "349:192",
                    "r_frame_rate": "25/1",
                    "avg_frame_rate": "25/1",
                    "field_order": "tt",
                }
            ],
            "format": {"format_name": "matroska", "duration": "60.0", "size": "1234"},
        }
        parsed = parse_probe_json(
            path,
            payload,
            [{"interlaced_frame": 1, "top_field_first": 1}, {"interlaced_frame": 0, "top_field_first": 0}],
        )
        self.assertEqual(parsed.path, path)
        self.assertEqual(parsed.video.display_aspect_ratio, Fraction(349, 192))
        self.assertEqual(parsed.sampled_interlaced_frames, 1)
        self.assertEqual(parsed.sampled_tff_frames, 1)
        self.assertEqual(parsed.sampled_progressive_frames, 1)

    def test_pixel_depth_detection(self) -> None:
        self.assertEqual(pixel_format_depth("yuv444p16le"), 16)
        self.assertEqual(pixel_format_depth("p010le"), 10)
        self.assertEqual(pixel_format_depth("yuvj420p"), 8)


class CapabilityParserTests(unittest.TestCase):
    def test_dfttest_capability_retains_cpu_and_nvrtc_for_raster_adaptation(self) -> None:
        def fake_run(args, **_kwargs):
            script = Path(args[-2]).read_text(encoding="utf-8")
            returncode = 1 if "dfttest2_cuda" in script else 0
            return subprocess.CompletedProcess(args, returncode, "", "cuFFT unavailable" if returncode else "")

        with patch("deinterlace_studio.capabilities._run", side_effect=fake_run):
            ready, backends, diagnostics = _inspect_vapoursynth_denoisers(Path("C:/tools/vspipe.exe"))
        self.assertTrue(all(ready.values()), diagnostics)
        self.assertEqual(backends["vs_dfttest"], DFTTEST_ADAPTIVE_CPU_NVRTC)
        self.assertIn("dfttest_cpu: graph emitted 4 frames", diagnostics["vs_dfttest"])
        self.assertIn("dfttest_nvrtc: graph emitted 4 frames", diagnostics["vs_dfttest"])
        self.assertIn("dfttest_cufft: unavailable", diagnostics["vs_dfttest"])

    def test_ffmpeg_component_and_pixel_format_parsers(self) -> None:
        component_text = """
 Filters:
  TSC bwdif             V->V       Deinterlace the input image.
  ... idet              V->V       Interlace detect filter.
 Encoders:
  V....D libx265        libx265 H.265 / HEVC
  V....D hevc_nvenc     NVIDIA NVENC hevc encoder
 """
        self.assertEqual(
            _parse_named_components(component_text),
            frozenset({"bwdif", "idet", "libx265", "hevc_nvenc"}),
        )
        self.assertEqual(
            _parse_pixel_formats("Encoder hevc_nvenc\n    Supported pixel formats: yuv420p nv12 p010le p012le\n"),
            ("yuv420p", "nv12", "p010le", "p012le"),
        )
        self.assertEqual(_parse_pixel_formats("Encoder without a format list"), ())

    def test_d3d12_probe_requires_real_processor_output_and_keeps_bob_non_quality(self) -> None:
        custom_failure = subprocess.CompletedProcess(
            args=[],
            returncode=-542398533,
            stdout="",
            stderr=(
                "Using custom (driver-defined) deinterlacing\n"
                "Failed to create video processor: HRESULT 0x887A0005\n"
                "Conversion failed!\n"
            ),
        )
        bob_success = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="",
            stderr=(
                "D3D12 deinterlace processor successfully configured\n"
                "Output stream #0:0 (video): 4 frames encoded; 4 packets muxed\n"
            ),
        )
        p010_failure = subprocess.CompletedProcess(
            args=[],
            returncode=-40,
            stdout="",
            stderr="No deinterlacing methods supported by hardware\nConversion failed!\n",
        )
        with patch(
            "deinterlace_studio.capabilities._run",
            side_effect=(custom_failure, bob_success, p010_failure, p010_failure),
        ) as run:
            verified, diagnostics = _inspect_ffmpeg_interlace_runtime(
                Path("C:/Tools/ffmpeg.exe"),
                frozenset({"bwdif", "bwdif_cuda", "deinterlace_d3d12"}),
                frozenset({"cuda", "d3d12va"}),
            )

        self.assertEqual(run.call_count, 4)
        self.assertTrue(all("-nostdin" in call.args[0] for call in run.call_args_list))
        self.assertFalse(verified["d3d12_custom"])
        self.assertTrue(verified["d3d12_bob"])
        self.assertFalse(verified["d3d12_custom_p010"])
        self.assertFalse(verified["d3d12_bob_p010"])
        self.assertIn("0x887A0005", diagnostics["d3d12_custom"])
        self.assertIn("excluded", diagnostics["d3d12_custom"])
        self.assertIn("better quality", diagnostics["d3d12_bob"])
        self.assertIn("No deinterlacing methods", diagnostics["d3d12_custom_p010"])
        self.assertIn("not a new FFmpeg 9 quality mode", diagnostics["bwdif_cuda"])

    def test_d3d12_filter_absence_is_reported_without_runtime_probe(self) -> None:
        with patch("deinterlace_studio.capabilities._run") as run:
            verified, diagnostics = _inspect_ffmpeg_interlace_runtime(
                Path("C:/Tools/ffmpeg.exe"), frozenset({"bwdif"}), frozenset()
            )
        run.assert_not_called()
        self.assertEqual(verified, {})
        self.assertIn("Not present", diagnostics["d3d12_custom"])

    def test_d3d12_error_excerpt_prefers_processor_hresult(self) -> None:
        excerpt = _d3d12_error_excerpt(
            "first line\nFailed to create video processor: HRESULT 0x887A0005\nConversion failed!\n"
        )
        self.assertTrue(excerpt.startswith("Failed to create video processor"))
        self.assertIn("Conversion failed", excerpt)


class IDetTests(unittest.TestCase):
    def test_parser_uses_nonzero_final_summary(self) -> None:
        text = """
Repeated Fields: Neither: 0 Top: 0 Bottom: 0
Single frame detection: TFF: 0 BFF: 0 Progressive: 0 Undetermined: 0
Multi frame detection: TFF: 0 BFF: 0 Progressive: 0 Undetermined: 0
Repeated Fields: Neither: 300 Top: 1 Bottom: 1
Single frame detection: TFF: 250 BFF: 0 Progressive: 20 Undetermined: 32
Multi frame detection: TFF: 300 BFF: 0 Progressive: 2 Undetermined: 0
"""
        counts = parse_idet_output(text)
        self.assertEqual(counts.multi_tff, 300)
        self.assertEqual(counts.repeated_neither, 300)

    def test_progressive_and_field_order_classification(self) -> None:
        self.assertEqual(classify_idet(IDetCounts(multi_progressive=1732))[0], "progressive")
        result = classify_idet(IDetCounts(multi_tff=1700, multi_progressive=60, multi_undetermined=40))
        self.assertEqual(result[0], "tff")
        self.assertEqual(result[1], "tff")
        self.assertEqual(classify_idet(IDetCounts(multi_tff=400, multi_progressive=400))[0], "mixed_or_ambiguous")

    def test_samples_are_distributed_away_from_exact_edges(self) -> None:
        intervals = sample_intervals(100.0, count=4, seconds=10.0)
        self.assertEqual(len(intervals), 4)
        self.assertGreater(intervals[0][0], 0)
        self.assertLess(intervals[-1][0] + intervals[-1][1], 100.0)


class ProfileTests(unittest.TestCase):
    def test_dnxhr_12_is_honestly_capability_gated(self) -> None:
        caps = capabilities(dnx12=False)
        self.assertIsNone(profile_capability_error(PROFILES["dnxhr_444_10"], caps))
        message = profile_capability_error(PROFILES["dnxhr_444_12"], caps)
        self.assertIsNotNone(message)
        self.assertIn("does not support yuv444p12le", message or "")

    def test_dnxhr_depth_choices_hide_only_unproven_12_bit(self) -> None:
        self.assertEqual(selectable_bit_depths("dnxhr", capabilities(dnx12=False)), (10,))
        self.assertEqual(selectable_bit_depths("dnxhr", capabilities(dnx12=True)), (10, 12))

    def test_nvenc_maximum_quality_contract_contains_requested_and_stronger_controls(self) -> None:
        args = nvenc_maximum_quality_args(14)
        rendered = " ".join(args)
        self.assertIn("-preset p7", rendered)
        self.assertIn("-tune uhq", rendered)
        self.assertIn("-multipass fullres", rendered)
        self.assertIn("-temporal-aq 1", rendered)
        self.assertIn("-rc-lookahead 32", rendered)
        self.assertNotIn("-tune hq", rendered)

    def test_hardware_depth_requires_coded_runtime_proof(self) -> None:
        caps = capabilities()
        caps = CapabilityReport(
            **{
                **caps.__dict__,
                "encoder_verified_bit_depths": {"hevc_nvenc": (10,), "av1_nvenc": (10,)},
                "encoder_runtime_diagnostics": {
                    "hevc_nvenc": "12-bit request decoded as yuv420p10le",
                    "av1_nvenc": "12-bit request decoded as yuv420p10le",
                },
            }
        )
        self.assertIsNone(profile_capability_error(PROFILES["hevc_nvenc_10"], caps))
        message = profile_capability_error(PROFILES["hevc_nvenc_12"], caps)
        self.assertIn("did not produce a true 12-bit stream", message or "")


class DenoiseGraphTests(unittest.TestCase):
    def test_dfttest_backend_prefers_verified_nvrtc_at_sd_and_hd(self) -> None:
        self.assertEqual(
            resolve_denoiser_backend("vs_dfttest", DFTTEST_ADAPTIVE_CPU_NVRTC, 720, 576),
            "dfttest_nvrtc",
        )

    def test_dfttest_backend_uses_fastest_member_of_each_verified_pair(self) -> None:
        self.assertEqual(
            resolve_denoiser_backend("vs_dfttest", DFTTEST_ADAPTIVE_CPU_CUFFT, 720, 576),
            "dfttest_cufft",
        )
        self.assertEqual(
            resolve_denoiser_backend("vs_dfttest", DFTTEST_ADAPTIVE_NVRTC_CUFFT, 720, 576),
            "dfttest_nvrtc",
        )
        self.assertEqual(
            resolve_denoiser_backend("vs_dfttest", DFTTEST_ADAPTIVE_CPU_NVRTC, None, None),
            "dfttest_nvrtc",
        )
        self.assertEqual(
            resolve_denoiser_backend("vs_dfttest", DFTTEST_ADAPTIVE_CPU_NVRTC, 1280, 720),
            "dfttest_nvrtc",
        )
        self.assertEqual(
            resolve_denoiser_backend("vs_dfttest", DFTTEST_ADAPTIVE_CPU_NVRTC, 1920, 1080),
            "dfttest_nvrtc",
        )

    def test_dfttest_graph_maps_strength_radius_and_named_backend(self) -> None:
        graph = "\n".join(vapoursynth_denoise_lines("vs_dfttest", 4, 3, "dfttest_nvrtc"))
        self.assertIn("DFTTest.Backend.NVRTC", graph)
        self.assertIn("tr=3", graph)
        self.assertIn("sigma=8.00", graph)
        with self.assertRaisesRegex(ValueError, "Unsupported DFTTest2 implementation"):
            vapoursynth_denoise_lines("vs_dfttest", 4, 3, DFTTEST_ADAPTIVE_CPU_NVRTC)


class PlannerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "本港台 節目.mkv"
        self.source.write_bytes(b"source")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def settings(self, **changes) -> JobSettings:
        values = dict(
            input_path=self.source,
            output_path=self.root / "out.mkv",
            backend="auto",
            field_order="auto",
            output_cadence="field_rate",
            aspect_mode="preserve",
            family="ffv1",
            bit_depth=16,
            # Planner tests opt into denoise explicitly when that stage is
            # under test; the application-level default is verified below.
            denoise_enabled=False,
        )
        values.update(changes)
        return JobSettings(**values)

    def test_job_settings_defaults_match_source_rate_and_enable_requested_denoise(self) -> None:
        defaults = JobSettings(input_path=self.source, output_path=self.root / "default.mkv")
        self.assertEqual(defaults.output_cadence, "frame_rate")
        self.assertEqual(defaults.hardware_decode, "auto")
        self.assertTrue(defaults.denoise_enabled)
        self.assertEqual(defaults.denoiser, "vs_bm3d")
        self.assertEqual(defaults.denoise_strength, 4)
        self.assertEqual(defaults.denoise_temporal_radius, 3)

    def test_wheel_vspipe_relay_is_replaced_by_its_native_child(self) -> None:
        wrapper = self.root / "Python" / "Scripts" / "vspipe.exe"
        native = self.root / "Python" / "Lib" / "site-packages" / "vapoursynth" / "vspipe.exe"
        wrapper.parent.mkdir(parents=True)
        native.parent.mkdir(parents=True)
        wrapper.write_bytes(b"relay")
        native.write_bytes(b"native")
        self.assertEqual(_execution_vspipe_path(wrapper), native)

    def test_progressive_auto_selects_passthrough(self) -> None:
        settings = self.settings()
        plan = build_plan(settings, media(self.source, progressive=True), report("progressive"), capabilities(), run_id="test")
        self.assertTrue(plan.valid, plan.errors)
        self.assertEqual(plan.selected_backend, "progressive")
        self.assertNotIn("bwdif=", " ".join(plan.ffmpeg_command))
        self.assertEqual(plan.expected.frame_rate, Fraction(25, 1))

    def test_qtgmc_quality_plan_is_reproducible_and_double_rate(self) -> None:
        settings = self.settings(backend="vapoursynth_qtgmc")
        plan = build_plan(settings, media(self.source), report("tff"), capabilities(), run_id="test")
        self.assertTrue(plan.valid, plan.errors)
        self.assertIn("QTempGaussMC", plan.vapoursynth_script or "")
        self.assertIn("QTempGaussMC.SourceMatchMode.TWICE_REFINED", plan.vapoursynth_script or "")
        self.assertIn("QTempGaussMC.LosslessMode.POSTSMOOTH", plan.vapoursynth_script or "")
        self.assertIn("cachemode=1, cachepath=CACHE_ROOT", plan.vapoursynth_script or "")
        self.assertIn("BestSource Cache", plan.vapoursynth_script or "")
        self.assertIn("tff=True", plan.vapoursynth_script or "")
        self.assertEqual(plan.expected.frame_rate, Fraction(50, 1))
        self.assertIn(repr(str(self.source)), plan.vapoursynth_script or "")
        self.assertIn("--container", plan.vspipe_command or ())
        self.assertIn("y4m", plan.vspipe_command or ())
        self.assertIn("--requests", plan.vspipe_command or ())
        request_index = (plan.vspipe_command or ()).index("--requests")
        self.assertEqual((plan.vspipe_command or ())[request_index + 1], str(QTGMC_VSPIPE_REQUESTS))
        self.assertIn(f"core.num_threads = min({QTGMC_CORE_THREAD_CAP}", plan.vapoursynth_script or "")
        self.assertIn("setsar=sar=349/240:max=349", " ".join(plan.ffmpeg_command))
        self.assertNotIn("-aspect", plan.ffmpeg_command)

    def test_qtgmc_source_rate_plan_is_25p_and_not_bobbed(self) -> None:
        settings = self.settings(backend="vapoursynth_qtgmc", output_cadence="frame_rate")
        plan = build_plan(settings, media(self.source), report("tff"), capabilities(), run_id="test")
        self.assertTrue(plan.valid, plan.errors)
        self.assertEqual(plan.expected.frame_rate, Fraction(25, 1))
        self.assertIn("motion_blur(fps_divisor=2)", plan.vapoursynth_script or "")
        self.assertIn("qtgmc.deinterlace", plan.vapoursynth_script or "")
        self.assertNotIn("qtgmc.bob", plan.vapoursynth_script or "")

    def test_bwdif_source_rate_uses_send_frame(self) -> None:
        settings = self.settings(backend="ffmpeg_bwdif", output_cadence="frame_rate")
        plan = build_plan(settings, media(self.source), report("tff"), capabilities(), run_id="test")
        self.assertTrue(plan.valid, plan.errors)
        self.assertEqual(plan.expected.frame_rate, Fraction(25, 1))
        self.assertIn("bwdif=mode=send_frame:parity=tff:deint=all", " ".join(plan.ffmpeg_command))

    def test_auto_uses_explicit_bwdif_when_qtgmc_is_missing(self) -> None:
        plan = build_plan(self.settings(), media(self.source), report("tff"), capabilities(qtgmc=False), run_id="test")
        self.assertTrue(plan.valid, plan.errors)
        self.assertEqual(plan.selected_backend, "ffmpeg_bwdif")
        self.assertIn("bwdif=mode=send_field:parity=tff:deint=all", " ".join(plan.ffmpeg_command))
        self.assertTrue(any("Auto selected FFmpeg" in warning for warning in plan.warnings))

    def test_progressive_deinterlace_requires_deliberate_override(self) -> None:
        plan = build_plan(
            self.settings(backend="ffmpeg_bwdif"),
            media(self.source, progressive=True),
            report("progressive"),
            capabilities(),
            run_id="test",
        )
        self.assertFalse(plan.valid)
        self.assertTrue(any("classifi" in error for error in plan.errors))
        override = build_plan(
            self.settings(backend="ffmpeg_bwdif", allow_progressive_override=True, field_order="tff"),
            media(self.source, progressive=True),
            report("progressive"),
            capabilities(),
            run_id="test",
        )
        self.assertTrue(override.valid, override.errors)

    def test_square_pixels_preserve_exact_dar(self) -> None:
        plan = build_plan(
            self.settings(backend="ffmpeg_bwdif", aspect_mode="square"),
            media(self.source),
            report("tff"),
            capabilities(),
            run_id="test",
        )
        self.assertTrue(plan.valid, plan.errors)
        self.assertEqual((plan.expected.width, plan.expected.height), (1396, 768))
        self.assertEqual(plan.expected.sar, Fraction(1, 1))
        self.assertEqual(plan.expected.dar, Fraction(349, 192))
        self.assertIn("zscale=w=1396:h=768", " ".join(plan.ffmpeg_command))

    def test_cuda_bwdif_downloads_after_deinterlacing(self) -> None:
        plan = build_plan(
            self.settings(backend="ffmpeg_bwdif_cuda", hardware_decode="cuda"),
            media(self.source),
            report("tff"),
            capabilities(),
            run_id="test",
        )
        self.assertTrue(plan.valid, plan.errors)
        filters = plan.ffmpeg_command[plan.ffmpeg_command.index("-vf") + 1]
        self.assertTrue(filters.startswith("bwdif_cuda="), filters)
        self.assertIn("bwdif_cuda=mode=send_field:parity=tff:deint=all,hwdownload,format=nv12", filters)

    def test_cuda_bwdif_can_software_decode_then_upload_for_ffv1_source(self) -> None:
        source_media = media(self.source)
        source_media = MediaProbe(
            **{
                **source_media.__dict__,
                "streams": (
                    StreamInfo(**{**source_media.video.__dict__, "codec_name": "ffv1"}),
                    *source_media.streams[1:],
                ),
            }
        )
        plan = build_plan(
            self.settings(backend="ffmpeg_bwdif_cuda", hardware_decode="off"),
            source_media,
            report("tff"),
            capabilities(),
            run_id="test",
        )
        self.assertTrue(plan.valid, plan.errors)
        self.assertNotIn("-hwaccel", plan.ffmpeg_command)
        filters = plan.ffmpeg_command[plan.ffmpeg_command.index("-vf") + 1]
        self.assertTrue(filters.startswith("format=nv12,hwupload_cuda,bwdif_cuda="), filters)
        self.assertIn("bwdif_cuda=mode=send_field:parity=tff:deint=all,hwdownload,format=nv12", filters)

    def test_cuda_bwdif_automatic_decode_is_codec_aware(self) -> None:
        direct = build_plan(
            self.settings(backend="ffmpeg_bwdif_cuda", hardware_decode="auto"),
            media(self.source),
            report("tff"),
            capabilities(),
            run_id="direct",
        )
        self.assertTrue(direct.valid, direct.errors)
        self.assertIn("-hwaccel", direct.ffmpeg_command)

        ffv1_media = media(self.source)
        ffv1_media = MediaProbe(
            **{
                **ffv1_media.__dict__,
                "streams": (
                    StreamInfo(**{**ffv1_media.video.__dict__, "codec_name": "ffv1"}),
                    *ffv1_media.streams[1:],
                ),
            }
        )
        uploaded = build_plan(
            self.settings(backend="ffmpeg_bwdif_cuda", hardware_decode="auto"),
            ffv1_media,
            report("tff"),
            capabilities(),
            run_id="upload",
        )
        self.assertTrue(uploaded.valid, uploaded.errors)
        self.assertNotIn("-hwaccel", uploaded.ffmpeg_command)
        graph = uploaded.ffmpeg_command[uploaded.ffmpeg_command.index("-vf") + 1]
        self.assertTrue(graph.startswith("format=nv12,hwupload_cuda,bwdif_cuda="), graph)

    def test_ffmpeg_temporal_denoisers_follow_deinterlace_and_color_properties(self) -> None:
        for identifier, marker in (
            ("ffmpeg_fftdnoiz", "fftdnoiz="),
            ("ffmpeg_atadenoise", "atadenoise="),
        ):
            with self.subTest(identifier=identifier):
                plan = build_plan(
                    self.settings(
                        backend="ffmpeg_bwdif",
                        denoise_enabled=True,
                        denoiser=identifier,
                        denoise_strength=4,
                        denoise_temporal_radius=2,
                    ),
                    media(self.source),
                    report("tff"),
                    capabilities(),
                    run_id="test",
                )
                self.assertTrue(plan.valid, plan.errors)
                graph = plan.ffmpeg_command[plan.ffmpeg_command.index("-vf") + 1]
                self.assertLess(graph.index("bwdif="), graph.index("setfield=mode=prog"))
                self.assertLess(graph.index("setfield=mode=prog"), graph.index("setparams="))
                self.assertLess(graph.index("setparams="), graph.index(marker))
                self.assertLess(graph.index(marker), graph.index("setsar="))
                self.assertIn("color_trc=bt470bg", graph)
                self.assertEqual(plan.selected_denoiser, identifier)
                self.assertEqual(plan.selected_denoise_backend, "ffmpeg")

    def test_qtgmc_ffmpeg_denoiser_runs_after_y4m_color_restoration(self) -> None:
        plan = build_plan(
            self.settings(
                backend="vapoursynth_qtgmc",
                denoise_enabled=True,
                denoiser="ffmpeg_atadenoise",
            ),
            media(self.source),
            report("tff"),
            capabilities(),
            run_id="test",
        )
        self.assertTrue(plan.valid, plan.errors)
        graph = plan.ffmpeg_command[plan.ffmpeg_command.index("-vf") + 1]
        self.assertLess(graph.index("setparams="), graph.index("atadenoise="))
        self.assertNotIn("atadenoise", plan.vapoursynth_script or "")

    def test_each_vapoursynth_denoiser_follows_qtgmc_in_the_generated_graph(self) -> None:
        expected_markers = {
            "vs_bm3d": "clip = bm3d(",
            "vs_dfttest": "clip = DFTTest(",
            "vs_mvtools": "clip = mc_degrain(",
            "vs_nlmeans": "clip = nl_means(",
        }
        for identifier, marker in expected_markers.items():
            with self.subTest(identifier=identifier):
                plan = build_plan(
                    self.settings(
                        backend="vapoursynth_qtgmc",
                        denoise_enabled=True,
                        denoiser=identifier,
                    ),
                    media(self.source),
                    report("tff"),
                    capabilities(),
                    run_id="test",
                )
                self.assertTrue(plan.valid, plan.errors)
                script = plan.vapoursynth_script or ""
                self.assertLess(script.index("clip = qtgmc.bob"), script.index(marker))
                self.assertIn("Temporal denoise is deliberately applied after deinterlacing", script)
                self.assertEqual(plan.selected_denoiser, identifier)
                self.assertTrue(plan.selected_denoise_backend)

    def test_vapoursynth_denoiser_can_process_progressive_without_qtgmc(self) -> None:
        plan = build_plan(
            self.settings(denoise_enabled=True, denoiser="vs_nlmeans"),
            media(self.source, progressive=True),
            report("progressive"),
            capabilities(),
            run_id="test",
        )
        self.assertTrue(plan.valid, plan.errors)
        self.assertEqual(plan.selected_backend, "progressive")
        self.assertIsNotNone(plan.vspipe_command)
        self.assertIn("cachemode=1, cachepath=CACHE_ROOT", plan.vapoursynth_script or "")
        self.assertIn("clip = nl_means(", plan.vapoursynth_script or "")
        self.assertNotIn("QTempGaussMC", plan.vapoursynth_script or "")
        self.assertEqual(plan.expected.frame_rate, Fraction(25, 1))

    def test_vapoursynth_denoiser_with_bwdif_is_rejected_without_substitution(self) -> None:
        plan = build_plan(
            self.settings(
                backend="ffmpeg_bwdif",
                denoise_enabled=True,
                denoiser="vs_bm3d",
            ),
            media(self.source),
            report("tff"),
            capabilities(),
            run_id="test",
        )
        self.assertFalse(plan.valid)
        joined = "\n".join(plan.errors)
        self.assertIn("cannot follow FFmpeg BWDIF", joined)
        self.assertIn("will not denoise interlaced fields before BWDIF", joined)
        self.assertEqual(plan.selected_backend, "ffmpeg_bwdif")

    def test_denoiser_capability_and_numeric_bounds_fail_closed(self) -> None:
        ready = capabilities()
        missing = CapabilityReport(
            **{
                **ready.__dict__,
                "denoise_capabilities": {**ready.denoise_capabilities, "vs_bm3d": False},
                "denoise_diagnostics": {**ready.denoise_diagnostics, "vs_bm3d": "test graph failure"},
            }
        )
        unavailable = build_plan(
            self.settings(backend="vapoursynth_qtgmc", denoise_enabled=True, denoiser="vs_bm3d"),
            media(self.source),
            report("tff"),
            missing,
            run_id="test",
        )
        self.assertFalse(unavailable.valid)
        self.assertTrue(any("test graph failure" in error for error in unavailable.errors))
        invalid = build_plan(
            self.settings(
                backend="ffmpeg_bwdif",
                denoise_enabled=True,
                denoiser="ffmpeg_fftdnoiz",
                denoise_strength=11,
                denoise_temporal_radius=0,
            ),
            media(self.source),
            report("tff"),
            ready,
            run_id="test",
        )
        self.assertFalse(invalid.valid)
        self.assertTrue(any("strength" in error.lower() for error in invalid.errors))
        self.assertTrue(any("radius" in error.lower() for error in invalid.errors))

    def test_source_output_collision_is_never_authorizable(self) -> None:
        plan = build_plan(
            self.settings(output_path=self.source, overwrite_approved=True),
            media(self.source),
            report("tff"),
            capabilities(),
            run_id="test",
        )
        self.assertFalse(plan.valid)
        self.assertTrue(any("Source media" in error for error in plan.errors))

    def test_mov_converts_subrip_to_native_mov_text_without_changing_video_or_audio(self) -> None:
        plan = build_plan(
            self.settings(output_path=self.root / "out.mov", family="prores", bit_depth=10),
            media(self.source, subtitle="subrip"),
            report("tff"),
            capabilities(),
            run_id="test",
        )
        self.assertTrue(plan.valid, plan.errors)
        rendered = " ".join(plan.ffmpeg_command)
        self.assertIn("-c:s:0 mov_text", rendered)
        self.assertIn("-c:a copy", rendered)
        self.assertEqual(plan.expected.expected_subtitles[0].codec_name, "mov_text")
        self.assertTrue(any("native mov_text" in warning for warning in plan.warnings))

    def test_mov_blocks_incompatible_direct_copy_audio(self) -> None:
        plan = build_plan(
            self.settings(output_path=self.root / "out.mov", family="prores", bit_depth=10),
            media(self.source, audio="flac"),
            report("tff"),
            capabilities(),
            run_id="test",
        )
        self.assertFalse(plan.valid)
        self.assertTrue(any("audio" in error.lower() and "flac" in error.lower() for error in plan.errors))

    def test_resolve_editor_profile_builds_dnxhr_444_10_bit_mov_with_compatible_audio(self) -> None:
        plan = build_plan(
            self.settings(
                output_path=self.root / "resolve-editor.mov",
                family="dnxhr",
                bit_depth=10,
                hardware_encode=False,
                copy_audio=True,
                copy_subtitles=False,
                copy_attachments=False,
                copy_data=False,
            ),
            media(self.source, audio="ac3"),
            report("tff"),
            capabilities(),
            run_id="resolve",
        )
        self.assertTrue(plan.valid, plan.errors)
        command = plan.ffmpeg_command
        self.assertIn("dnxhd", command)
        self.assertIn("dnxhr_444", command)
        self.assertIn("yuv444p10le", command)
        self.assertIn("-c:a", command)
        self.assertIn("copy", command)
        self.assertIn("-metadata:s:a:0", command)
        self.assertIn("language=chi", command)
        self.assertEqual(plan.expected.frame_rate, Fraction(50, 1))
        self.assertEqual(plan.expected.pix_fmts, ("yuv444p10le",))

    def test_ffv1_plan_identifies_archive_editor_interchange_boundary(self) -> None:
        plan = build_plan(
            self.settings(),
            media(self.source),
            report("tff"),
            capabilities(),
            run_id="archive",
        )
        self.assertTrue(plan.valid, plan.errors)
        warning_text = "\n".join(plan.warnings)
        self.assertIn("archival master", warning_text)
        self.assertIn("Resolve editor preset", warning_text)
        self.assertIn("DNxHR 444 10-bit MOV", warning_text)

    def test_prores_in_matroska_warns_about_mpc_and_fast_mov_copy(self) -> None:
        plan = build_plan(
            self.settings(
                output_path=self.root / "prores-master.mkv",
                family="prores",
                bit_depth=10,
            ),
            media(self.source),
            report("tff"),
            capabilities(),
            run_id="prores-mkv-warning",
        )
        self.assertTrue(plan.valid, plan.errors)
        warning_text = "\n".join(plan.warnings)
        self.assertIn("DirectShow/MPC", warning_text)
        self.assertIn("Create fast MOV compatibility copy", warning_text)

    def test_resolve_mov_normalizes_extended_chinese_language_tag_without_losing_track(self) -> None:
        source_media = media(self.source, audio="ac3")
        source_media = MediaProbe(
            **{
                **source_media.__dict__,
                "streams": (
                    source_media.video,
                    StreamInfo(
                        **{
                            **source_media.streams_of_type("audio")[0].__dict__,
                            "tags": {"language": "yue", "title": "Cantonese"},
                        }
                    ),
                ),
            }
        )
        plan = build_plan(
            self.settings(
                output_path=self.root / "resolve-language.mov",
                family="dnxhr",
                bit_depth=10,
                copy_subtitles=False,
                copy_attachments=False,
                copy_data=False,
            ),
            source_media,
            report("tff"),
            capabilities(),
            run_id="language",
        )
        self.assertTrue(plan.valid, plan.errors)
        self.assertIn("language=chi", plan.ffmpeg_command)
        self.assertIn("title=Cantonese", plan.ffmpeg_command)
        self.assertEqual(plan.expected.expected_audio[0].tags["language"], "chi")
        self.assertTrue(any("'yue'→'chi'" in warning for warning in plan.warnings))

    def test_quality_and_ffmpeg_ratio_bounds_fail_closed(self) -> None:
        bad_quality = build_plan(
            self.settings(quality=41), media(self.source), report("tff"), capabilities(), run_id="test"
        )
        self.assertFalse(bad_quality.valid)
        self.assertTrue(any("0 through 40" in error for error in bad_quality.errors))
        huge_dar = build_plan(
            self.settings(aspect_mode="manual", manual_dar=f"{2**31}:1"),
            media(self.source),
            report("tff"),
            capabilities(),
            run_id="test",
        )
        self.assertFalse(huge_dar.valid)
        self.assertTrue(any("integer range" in error for error in huge_dar.errors))


if __name__ == "__main__":
    unittest.main()
