from __future__ import annotations

import json
import io
import subprocess
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

from deinterlace_studio.models import (
    AutomaticRecoveryAudit,
    CapabilityReport,
    IDetCounts,
    IDetReport,
    JobSettings,
    MediaProbe,
    OutputExpectation,
    PacketTimelineGap,
    SourceHealthReport,
    SourcePreflightEvidence,
    StreamInfo,
    ValidationResult,
)
from deinterlace_studio.planner import build_plan
from deinterlace_studio.processor import (
    JobProcessor,
    FastPreflightUnavailable,
    ProcessingCancelled,
    ProcessingError,
    SourceRepairRequiredError,
    describe_vspipe_exit,
    qtgmc_timeline_integrity_error,
    vapoursynth_timeline_integrity_error,
)
from deinterlace_studio.validation import validate_output


def video(*, codec: str = "ffv1", interlaced: bool = False) -> StreamInfo:
    return StreamInfo(
        index=0,
        codec_type="video",
        codec_name=codec,
        width=320,
        height=240,
        pix_fmt="yuv444p16le" if codec == "ffv1" else "yuv420p",
        bits_per_raw_sample=16 if codec == "ffv1" else 8,
        sample_aspect_ratio=Fraction(4, 3),
        display_aspect_ratio=Fraction(16, 9),
        r_frame_rate=Fraction(50, 1),
        avg_frame_rate=Fraction(50, 1),
        field_order="tt" if interlaced else "progressive",
        nb_frames=3,
        color_range="tv",
        color_space="bt470bg",
        color_transfer="bt470bg",
        color_primaries="bt470bg",
    )


def media(path: Path, *, output: bool = False) -> MediaProbe:
    return MediaProbe(
        path=path,
        format_name="matroska",
        format_long_name="Matroska",
        duration=0.06,
        size=path.stat().st_size if path.exists() else 1,
        bit_rate=1,
        start_time=0.0,
        streams=(
            video(codec="ffv1" if output else "mpeg2video", interlaced=not output),
            StreamInfo(index=1, codec_type="audio", codec_name="flac", tags={"language": "yue"}),
            StreamInfo(index=2, codec_type="data", codec_name="bin_data", tags={"title": "markers"}),
        ),
        chapters=({"id": 0, "start_time": "0", "end_time": "0.06", "tags": {"title": "Opening"}},),
        format_tags={"title": "Preserved title"},
    )


def caps() -> CapabilityReport:
    return CapabilityReport(
        ffmpeg_path=Path("ffmpeg.exe"),
        ffprobe_path=Path("ffprobe.exe"),
        ffmpeg_version="test",
        ffmpeg_configuration="",
        filters=frozenset({"bwdif", "idet", "fftdnoiz", "atadenoise"}),
        encoders=frozenset({"ffv1"}),
        encoder_pixel_formats={"ffv1": ("yuv420p16le", "yuv422p16le", "yuv444p16le")},
        hwaccels=frozenset(),
        vspipe_path=None,
        vapoursynth_version=None,
        qtgmc_ready=False,
        qtgmc_diagnostic="not installed",
        qtgmc_install_command=None,
        denoise_capabilities={
            "ffmpeg_fftdnoiz": True,
            "ffmpeg_atadenoise": True,
            "vs_bm3d": False,
            "vs_mvtools": False,
            "vs_nlmeans": False,
        },
        denoise_backends={
            "ffmpeg_fftdnoiz": "ffmpeg",
            "ffmpeg_atadenoise": "ffmpeg",
        },
    )


def idet() -> IDetReport:
    return IDetReport(
        mode="full",
        segments=(),
        aggregate=IDetCounts(multi_tff=3),
        classification="tff",
        dominant_field_order="tff",
        confidence=1.0,
        rationale="test",
    )


class ValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.output = self.root / "out.mkv"
        self.output.write_bytes(b"output")
        self.actual = media(self.output, output=True)
        self.expected = OutputExpectation(
            codec_names=("ffv1",),
            pix_fmts=("yuv444p16le",),
            width=320,
            height=240,
            sar=Fraction(4, 3),
            dar=Fraction(16, 9),
            frame_rate=Fraction(50, 1),
            progressive=True,
            lossless=True,
            bit_depth=16,
            expected_audio=self.actual.streams_of_type("audio"),
            expected_subtitles=(),
            expected_attachments=(),
            duration=0.06,
            frame_count=3,
            expected_data=self.actual.streams_of_type("data"),
            expected_chapter_count=1,
            expected_format_tags={"title": "Preserved title"},
            color_range="tv",
            color_space="bt470bg",
            color_transfer="bt470bg",
            color_primaries="bt470bg",
        )
        self.settings = JobSettings(self.root / "source.mkv", self.output, family="ffv1", bit_depth=16)

    def tearDown(self) -> None:
        self.temp.cleanup()

    @patch("deinterlace_studio.validation._count_intra_packet_flags", return_value=(3, 3))
    @patch("deinterlace_studio.validation.count_video_packets", return_value=3)
    @patch(
        "deinterlace_studio.validation.probe_frame_samples",
        return_value=[{"interlaced_frame": 0} for _ in range(8)],
    )
    @patch("deinterlace_studio.validation.probe_media")
    def test_full_contract_passes(self, probe_mock, *_mocks) -> None:
        probe_mock.return_value = self.actual
        result = validate_output(Path("ffprobe.exe"), self.output, self.expected, self.settings)
        self.assertTrue(result.valid, result.errors)
        self.assertEqual(result.checked_progressive_frames, 8)

    @patch("deinterlace_studio.validation._count_intra_packet_flags", return_value=(3, 3))
    @patch("deinterlace_studio.validation.count_video_packets", return_value=3)
    @patch("deinterlace_studio.validation.probe_frame_samples", return_value=[{"interlaced_frame": 1}])
    @patch("deinterlace_studio.validation.probe_media")
    def test_color_chapters_metadata_and_progressive_fail_closed(self, probe_mock, *_mocks) -> None:
        bad_video = StreamInfo(**{**video().__dict__, "color_space": None, "field_order": "tt"})
        bad = MediaProbe(
            **{
                **self.actual.__dict__,
                "streams": (bad_video,) + self.actual.streams[1:],
                "chapters": (),
                "format_tags": {},
            }
        )
        probe_mock.return_value = bad
        result = validate_output(Path("ffprobe.exe"), self.output, self.expected, self.settings)
        self.assertFalse(result.valid)
        joined = "\n".join(result.errors)
        self.assertIn("color matrix", joined)
        self.assertIn("Chapter count", joined)
        self.assertIn("Container metadata", joined)
        self.assertIn("remain flagged interlaced", joined)

    @patch("deinterlace_studio.validation._count_intra_packet_flags", return_value=(3, 3))
    @patch("deinterlace_studio.validation.count_video_packets", return_value=3)
    @patch("deinterlace_studio.validation.probe_frame_samples", return_value=[{"interlaced_frame": 0}])
    @patch("deinterlace_studio.validation.probe_media")
    def test_near_aspect_approximation_fails_exact_contract(self, probe_mock, *_mocks) -> None:
        approximated_video = StreamInfo(
            **{
                **video().__dict__,
                "sample_aspect_ratio": Fraction(996, 685),
                "display_aspect_ratio": Fraction(249, 137),
            }
        )
        probe_mock.return_value = MediaProbe(
            **{**self.actual.__dict__, "streams": (approximated_video,) + self.actual.streams[1:]}
        )
        result = validate_output(Path("ffprobe.exe"), self.output, self.expected, self.settings)
        self.assertFalse(result.valid)
        self.assertTrue(any("Output SAR" in error for error in result.errors))
        self.assertTrue(any("Output DAR" in error for error in result.errors))

    @patch("deinterlace_studio.validation._count_intra_packet_flags", return_value=(3, 3))
    @patch("deinterlace_studio.validation.count_video_packets", return_value=3)
    @patch("deinterlace_studio.validation.probe_frame_samples")
    @patch("deinterlace_studio.validation.probe_media")
    def test_decoded_frame_properties_prove_stream_omissions(self, probe_mock, frame_mock, *_mocks) -> None:
        omitted = StreamInfo(
            **{
                **video().__dict__,
                "pix_fmt": None,
                "color_transfer": None,
                "color_primaries": None,
            }
        )
        probe_mock.return_value = MediaProbe(
            **{**self.actual.__dict__, "streams": (omitted,) + self.actual.streams[1:]}
        )
        frame_mock.return_value = [
            {
                "interlaced_frame": 0,
                "pix_fmt": "yuv444p16le",
                "color_range": "tv",
                "color_space": "bt470bg",
                "color_transfer": "bt470bg",
                "color_primaries": "bt470bg",
            }
            for _ in range(12)
        ]
        result = validate_output(Path("ffprobe.exe"), self.output, self.expected, self.settings)
        self.assertTrue(result.valid, result.errors)
        joined = "\n".join(result.warnings)
        self.assertIn("omitted the stream-level pixel format", joined)
        self.assertIn("omitted stream-level color transfer", joined)
        self.assertIn("omitted stream-level color primaries", joined)

    @patch("deinterlace_studio.validation._count_intra_packet_flags", return_value=(3, 3))
    @patch("deinterlace_studio.validation.count_video_packets", return_value=3)
    @patch("deinterlace_studio.validation.probe_frame_samples")
    @patch("deinterlace_studio.validation.probe_media")
    def test_decoded_frame_fallback_never_accepts_missing_or_mismatched_evidence(
        self, probe_mock, frame_mock, *_mocks
    ) -> None:
        omitted = StreamInfo(
            **{
                **video().__dict__,
                "pix_fmt": None,
                "color_transfer": None,
                "color_primaries": None,
            }
        )
        probe_mock.return_value = MediaProbe(
            **{**self.actual.__dict__, "streams": (omitted,) + self.actual.streams[1:]}
        )
        frame_mock.return_value = [
            {
                "interlaced_frame": 0,
                "pix_fmt": "yuv420p10le",
                "color_transfer": "bt709",
                "color_primaries": "bt709",
            }
        ]
        result = validate_output(Path("ffprobe.exe"), self.output, self.expected, self.settings)
        self.assertFalse(result.valid)
        joined = "\n".join(result.errors)
        self.assertIn("bounded decoded frame samples did not consistently prove yuv444p16le", joined)
        self.assertIn("preserved value bt470bg", joined)

        mismatched_stream = StreamInfo(**{**video().__dict__, "color_transfer": "bt709"})
        probe_mock.return_value = MediaProbe(
            **{**self.actual.__dict__, "streams": (mismatched_stream,) + self.actual.streams[1:]}
        )
        frame_mock.return_value = [
            {
                "interlaced_frame": 0,
                "pix_fmt": "yuv444p16le",
                "color_transfer": "bt470bg",
                "color_primaries": "bt470bg",
            }
        ]
        mismatch = validate_output(Path("ffprobe.exe"), self.output, self.expected, self.settings)
        self.assertFalse(mismatch.valid)
        self.assertTrue(any("Output color transfer is bt709" in error for error in mismatch.errors))


class ProcessorSafetyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "source.mkv"
        self.source.write_bytes(b"source")
        self.output = self.root / "out.mkv"
        self.source_probe = media(self.source, output=False)
        self.settings = JobSettings(
            input_path=self.source,
            output_path=self.output,
            backend="ffmpeg_bwdif",
            family="ffv1",
            bit_depth=16,
            denoise_enabled=False,
            copy_data=True,
            overwrite_approved=True,
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def plan(self):
        return build_plan(self.settings, self.source_probe, idet(), caps(), run_id="safety")

    @staticmethod
    def _valid() -> ValidationResult:
        return ValidationResult(True, (), (), None)

    def test_success_replaces_only_after_validation_and_writes_sidecars(self) -> None:
        self.output.write_bytes(b"old")
        old_log = self.output.with_name(self.output.name + ".Deinterlace.log")
        old_log.write_text("old log", encoding="utf-8")
        plan = replace(
            self.plan(),
            vapoursynth_threads=16,
            vspipe_requests=24,
            vapoursynth_schedule_note=(
                "Adaptive CPU NNEDI3 selected from measured bounds. "
                "Applied schedule: core threads=16; VSPipe requests=24."
            ),
            vulkan_nnedi3_active=False,
        )
        processor = JobProcessor()

        def encode(_plan, *_args):
            _plan.partial_path.write_bytes(b"new")

        partial_valid = ValidationResult(
            True,
            (),
            (),
            None,
            verified_packet_count=3,
            verified_key_packet_count=3,
            thorough_packet_scan_completed=True,
        )
        with patch.object(processor, "_run_pipeline", side_effect=encode), patch(
            "deinterlace_studio.processor.validate_output",
            side_effect=(partial_valid, self._valid()),
        ) as validate:
            result = processor.run(plan, self.source_probe, idet(), caps())
        self.assertTrue(result.success, result.message)
        self.assertEqual(self.output.read_bytes(), b"new")
        self.assertTrue(result.log_path.is_file())
        self.assertTrue(result.report_path.is_file())
        log_text = result.log_path.read_text(encoding="utf-8")
        self.assertIn("core threads=16; VSPipe requests=24", log_text)
        self.assertIn("Vulkan NNEDI3 interpolation: disabled (CPU NNEDI3)", log_text)
        payload = json.loads(result.report_path.read_text(encoding="utf-8"))
        self.assertEqual(
            payload["source_preflight"]["method"],
            "not_required_timestamp_aware_ffmpeg",
        )
        self.assertEqual(validate.call_count, 2)
        self.assertTrue(validate.call_args_list[0].kwargs["thorough_packet_count"])
        self.assertFalse(validate.call_args_list[1].kwargs["thorough_packet_count"])
        self.assertEqual(payload["final_validation"]["verified_packet_count"], 3)
        self.assertEqual(payload["validation_strategy"]["partial_thorough_packet_scans"], 1)
        self.assertEqual(payload["validation_strategy"]["promoted_full_packet_rescans"], 0)
        self.assertTrue(payload["validation_strategy"]["same_file_atomic_promotion_verified"])
        self.assertFalse(any(self.root.glob("*.backup.*")))

    def test_atomic_promotion_identity_mismatch_quarantines_and_restores_prior_output(self) -> None:
        self.output.write_bytes(b"old")
        plan = self.plan()
        processor = JobProcessor()

        def encode(_plan, *_args):
            _plan.partial_path.write_bytes(b"candidate")

        with patch.object(processor, "_run_pipeline", side_effect=encode), patch(
            "deinterlace_studio.processor.validate_output", return_value=self._valid()
        ), patch(
            "deinterlace_studio.processor._promotion_identity",
            side_effect=((9, 1, 1, 10), (9, 1, 1, 11)),
        ):
            result = processor.run(plan, self.source_probe, idet(), caps())
        self.assertFalse(result.success)
        self.assertIn("exact file identity", result.message)
        self.assertEqual(self.output.read_bytes(), b"old")
        self.assertIsNotNone(result.quarantine_path)
        self.assertEqual(result.quarantine_path.read_bytes(), b"candidate")

    def test_enabled_denoiser_is_serialized_in_log_and_json_audit(self) -> None:
        self.settings = JobSettings(
            **{
                **self.settings.__dict__,
                "denoise_enabled": True,
                "denoiser": "ffmpeg_fftdnoiz",
                "denoise_strength": 3,
                "denoise_temporal_radius": 1,
            }
        )
        plan = self.plan()
        self.assertTrue(plan.valid, plan.errors)
        processor = JobProcessor()

        def encode(_plan, *_args):
            _plan.partial_path.write_bytes(b"new")

        with patch.object(processor, "_run_pipeline", side_effect=encode), patch(
            "deinterlace_studio.processor.validate_output", return_value=self._valid()
        ):
            result = processor.run(plan, self.source_probe, idet(), caps())
        self.assertTrue(result.success, result.message)
        log_text = result.log_path.read_text(encoding="utf-8")
        self.assertIn("Temporal denoise: enabled after deinterlacing", log_text)
        self.assertIn("algorithm=ffmpeg_fftdnoiz", log_text)
        payload = json.loads(result.report_path.read_text(encoding="utf-8"))
        self.assertTrue(payload["plan"]["settings"]["denoise_enabled"])
        self.assertEqual(payload["plan"]["selected_denoiser"], "ffmpeg_fftdnoiz")
        self.assertEqual(payload["plan"]["selected_denoise_backend"], "ffmpeg")

    def test_automatic_recovery_audit_is_linked_in_log_and_json_sidecar(self) -> None:
        stat = self.source.stat()
        trigger = SourceHealthReport(
            path=self.source,
            source_size=stat.st_size,
            source_mtime_ns=stat.st_mtime_ns,
            status="repair_required",
            reason="Fast packet scan measured a material timestamp hole.",
            elapsed_seconds=0.1,
            packet_count=3,
            timestamped_packet_count=3,
            unique_timestamp_count=3,
            first_pts=0.0,
            last_pts=0.06,
            typical_step_seconds=0.02,
            packet_timeline_span_seconds=0.06,
            reported_duration_seconds=0.06,
            duration_difference_seconds=0.0,
            gap_threshold_seconds=0.5,
            material_gap_count=1,
            largest_gaps=(PacketTimelineGap(0.02, 1.02, 1.0),),
            demux_warning_count=0,
            structural_warning_count=0,
            warning_samples=(),
            ffprobe_returncode=0,
        )
        repair = self.root / "source.qtgmc-repair.mkv"
        audit = AutomaticRecoveryAudit(
            original_source=self.source,
            trigger_health=trigger,
            requested_output=self.output,
            selected_output=self.output,
            repair_output=repair,
            repair_method="ffv1_rescue",
            repair_output_sha256="A" * 64,
            repair_log_path=self.root / "repair.log",
            repair_report_path=self.root / "repair.json",
            repeated_frames=7,
            dropped_frames=0,
            storage_preflight="PASS C: conservative storage check",
        )
        plan = build_plan(
            self.settings,
            self.source_probe,
            idet(),
            caps(),
            run_id="audit",
            automatic_recovery=audit,
        )
        processor = JobProcessor()

        def encode(_plan, *_args):
            _plan.partial_path.write_bytes(b"new")

        with patch.object(processor, "_run_pipeline", side_effect=encode), patch(
            "deinterlace_studio.processor.validate_output", return_value=self._valid()
        ):
            result = processor.run(plan, self.source_probe, idet(), caps())
        self.assertTrue(result.success, result.message)
        log_text = result.log_path.read_text(encoding="utf-8")
        self.assertIn("Automatic recovery chain", log_text)
        self.assertIn(str(self.source), log_text)
        self.assertIn(str(repair), log_text)
        payload = json.loads(result.report_path.read_text(encoding="utf-8"))
        recovery = payload["plan"]["automatic_recovery"]
        self.assertEqual(recovery["original_source"], str(self.source))
        self.assertEqual(recovery["repair_output"], str(repair))
        self.assertEqual(recovery["repair_method"], "ffv1_rescue")
        self.assertEqual(recovery["repeated_frames"], 7)

    def test_final_reopen_failure_quarantines_candidate_and_restores_prior_output(self) -> None:
        self.output.write_bytes(b"old")
        plan = self.plan()
        processor = JobProcessor()

        def encode(_plan, *_args):
            _plan.partial_path.write_bytes(b"candidate")

        invalid = ValidationResult(False, ("reopen mismatch",), (), None)
        with patch.object(processor, "_run_pipeline", side_effect=encode), patch(
            "deinterlace_studio.processor.validate_output", side_effect=[self._valid(), invalid]
        ):
            result = processor.run(plan, self.source_probe, idet(), caps())
        self.assertFalse(result.success)
        self.assertEqual(self.output.read_bytes(), b"old")
        self.assertIsNotNone(result.quarantine_path)
        self.assertEqual(result.quarantine_path.read_bytes(), b"candidate")

    def test_cancel_removes_current_partial_and_preserves_prior_output(self) -> None:
        self.output.write_bytes(b"old")
        plan = self.plan()
        processor = JobProcessor()

        def canceled(_plan, *_args):
            _plan.partial_path.write_bytes(b"partial")
            raise ProcessingCancelled("Processing canceled")

        with patch.object(processor, "_run_pipeline", side_effect=canceled):
            result = processor.run(plan, self.source_probe, idet(), caps())
        self.assertTrue(result.canceled)
        self.assertEqual(self.output.read_bytes(), b"old")
        self.assertFalse(plan.partial_path.exists())
        self.assertTrue(result.log_path.is_file())

    def test_source_change_during_encode_blocks_promotion(self) -> None:
        plan = self.plan()
        processor = JobProcessor()

        def encode(_plan, *_args):
            _plan.partial_path.write_bytes(b"candidate")
            self.source.write_bytes(b"changed source identity")

        with patch.object(processor, "_run_pipeline", side_effect=encode), patch(
            "deinterlace_studio.processor.validate_output", return_value=self._valid()
        ):
            result = processor.run(plan, self.source_probe, idet(), caps())
        self.assertFalse(result.success)
        self.assertIn("source file changed", result.message)
        self.assertFalse(self.output.exists())

    def test_windows_access_violation_is_reported_actionably(self) -> None:
        message = describe_vspipe_exit(3221225477)
        self.assertIn("0xC0000005", message)
        self.assertIn("native VapourSynth/plugin failure", message)
        self.assertIn("Dependency Doctor", message)

    def test_qtgmc_timeline_gap_is_blocked_before_cfr_pipe(self) -> None:
        source = media(self.source, output=False)
        self.assertIsNone(qtgmc_timeline_integrity_error(source, 3))
        damaged = MediaProbe(
            path=source.path,
            format_name=source.format_name,
            format_long_name=source.format_long_name,
            duration=4.06,
            size=source.size,
            bit_rate=source.bit_rate,
            start_time=source.start_time,
            streams=source.streams,
            chapters=source.chapters,
            format_tags=source.format_tags,
        )
        message = qtgmc_timeline_integrity_error(damaged, 3)
        self.assertIn("timeline integrity check failed", message or "")
        self.assertIn("BWDIF CPU/CUDA", message or "")
        self.assertIn("blocked every time QTGMC is selected", message or "")
        self.assertIn("timestamp-aware fallback, not a repair", message or "")
        self.assertIn("Use Repair source…", message or "")
        self.assertIn("lossless stream-copy remuxing", message or "")
        self.assertIn("FFV1 lossless rescue", message or "")
        self.assertIn("Missing pictures are not reconstructed", message or "")
        progressive_message = vapoursynth_timeline_integrity_error(damaged, 3)
        self.assertIn("VapourSynth temporal-denoise source timeline integrity check failed", progressive_message or "")
        self.assertIn("choose one of the FFmpeg temporal denoisers", progressive_message or "")


class SourcePreflightTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "clean-repair.mkv"
        self.source.write_bytes(b"three ffv1 packets")
        self.output = self.root / "out.mkv"
        base_probe = media(self.source, output=False)
        ffv1_video = StreamInfo(
            **{
                **base_probe.video.__dict__,
                "codec_name": "ffv1",
                "nb_frames": 3,
            }
        )
        self.source_probe = MediaProbe(
            **{
                **base_probe.__dict__,
                "streams": (ffv1_video,) + base_probe.streams[1:],
            }
        )
        stat = self.source.stat()
        self.health = SourceHealthReport(
            path=self.source,
            source_size=stat.st_size,
            source_mtime_ns=stat.st_mtime_ns,
            status="clear",
            reason="Fast full-file packet/timestamp scan found no material gap or structural warning.",
            elapsed_seconds=0.01,
            packet_count=3,
            timestamped_packet_count=3,
            unique_timestamp_count=3,
            first_pts=0.0,
            last_pts=0.04,
            typical_step_seconds=0.02,
            packet_timeline_span_seconds=0.06,
            reported_duration_seconds=0.06,
            duration_difference_seconds=0.0,
            gap_threshold_seconds=0.5,
            material_gap_count=0,
            largest_gaps=(),
            demux_warning_count=0,
            structural_warning_count=0,
            warning_samples=(),
            ffprobe_returncode=0,
            scan_error=None,
        )
        settings = JobSettings(
            input_path=self.source,
            output_path=self.output,
            backend="ffmpeg_bwdif",
            output_cadence="field_rate",
            family="ffv1",
            bit_depth=16,
            denoise_enabled=False,
            overwrite_approved=True,
        )
        base_plan = build_plan(settings, self.source_probe, idet(), caps(), run_id="preflight")
        self.assertTrue(base_plan.valid, base_plan.errors)
        assert base_plan.expected
        temporary_script_path = self.root / ".out.Deinterlace.preflight.vpy"
        self.plan = replace(
            base_plan,
            selected_backend="vapoursynth_qtgmc",
            vspipe_command=("vspipe.exe", "--progress", str(temporary_script_path), "-"),
            vapoursynth_script="clip.set_output()",
            temporary_script_path=temporary_script_path,
            expected=replace(base_plan.expected, frame_count=None),
            source_health=self.health,
        )
        self.plan.temporary_script_path.write_text("clip.set_output()", encoding="utf-8")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_clear_unchanged_ffv1_uses_indexed_graph_packet_contract(self) -> None:
        processor = JobProcessor()
        events: list[dict[str, str]] = []
        with patch.object(processor, "_run_vspipe_info", return_value=6) as info, patch.object(
            processor, "_run_decoded_preflight"
        ) as decoded:
            evidence = processor._source_preflight(
                self.plan,
                self.source_probe,
                caps(),
                lambda _message: None,
                events.append,
            )
        info.assert_called_once()
        decoded.assert_not_called()
        self.assertEqual(evidence.method, "vspipe_info_ffv1_packet_contract")
        self.assertEqual(evidence.source_frames, 3)
        self.assertEqual(evidence.expected_output_frames, 6)
        self.assertEqual(evidence.packet_count, 3)
        self.assertTrue(evidence.fast_path_eligible)
        self.assertEqual(events[-1]["phase"], "preflight_complete")

    def test_warning_health_forces_managed_full_decode(self) -> None:
        warning_plan = replace(self.plan, source_health=replace(self.health, status="warning"))
        processor = JobProcessor()
        with patch.object(processor, "_run_vspipe_info") as info, patch.object(
            processor, "_run_decoded_preflight", return_value=3
        ) as decoded:
            evidence = processor._source_preflight(
                warning_plan,
                self.source_probe,
                caps(),
                lambda _message: None,
                None,
            )
        info.assert_not_called()
        decoded.assert_called_once()
        self.assertEqual(evidence.method, "managed_full_decode")
        self.assertFalse(evidence.fast_path_eligible)
        self.assertIn("warning", evidence.fallback_reason or "")

    def test_every_indexed_eligibility_guard_rejects_missing_or_contradictory_evidence(self) -> None:
        changed_stat = replace(self.health, source_mtime_ns=self.health.source_mtime_ns + 1)
        cases = {
            "no health": (replace(self.plan, source_health=None), self.source_probe),
            "source identity changed": (replace(self.plan, source_health=changed_stat), self.source_probe),
            "status not clear": (replace(self.plan, source_health=replace(self.health, status="warning")), self.source_probe),
            "scan exit failed": (replace(self.plan, source_health=replace(self.health, ffprobe_returncode=1)), self.source_probe),
            "scan error": (replace(self.plan, source_health=replace(self.health, scan_error="timeout")), self.source_probe),
            "no packets": (replace(self.plan, source_health=replace(self.health, packet_count=0)), self.source_probe),
            "missing timestamp": (
                replace(self.plan, source_health=replace(self.health, timestamped_packet_count=2)),
                self.source_probe,
            ),
            "duplicate timestamp": (
                replace(self.plan, source_health=replace(self.health, unique_timestamp_count=2)),
                self.source_probe,
            ),
            "material gap": (
                replace(self.plan, source_health=replace(self.health, material_gap_count=1)),
                self.source_probe,
            ),
            "demux warning": (
                replace(self.plan, source_health=replace(self.health, demux_warning_count=1)),
                self.source_probe,
            ),
            "structural warning": (
                replace(self.plan, source_health=replace(self.health, structural_warning_count=1)),
                self.source_probe,
            ),
            "not ffv1": (
                self.plan,
                MediaProbe(
                    **{
                        **self.source_probe.__dict__,
                        "streams": (
                            StreamInfo(**{**self.source_probe.video.__dict__, "codec_name": "h264"}),
                        )
                        + self.source_probe.streams[1:],
                    }
                ),
            ),
        }
        for label, (plan, probe) in cases.items():
            with self.subTest(label=label):
                self.assertIsNotNone(JobProcessor._fast_preflight_ineligibility(plan, probe))

    def test_vspipe_info_parses_frames_and_fails_over_on_exit_or_timeout(self) -> None:
        class InfoProcess:
            def __init__(self, output: str, returncode: int = 0, *, timeout: bool = False) -> None:
                self.output = output
                self.returncode = None if timeout else returncode
                self.timeout = timeout
                self.stdout = io.StringIO("")
                self.stderr = io.StringIO("")
                self.terminated = False

            def communicate(self, timeout=None):
                if self.timeout and not self.terminated:
                    raise subprocess.TimeoutExpired("vspipe", timeout)
                return self.output, ""

            def poll(self):
                return self.returncode

            def wait(self, timeout=None):
                return self.returncode

            def terminate(self):
                self.terminated = True
                self.returncode = -15

            def kill(self):
                self.terminated = True
                self.returncode = -9

        processor = JobProcessor()
        success = InfoProcess("Width: 320\nHeight: 240\nFrames: 6\nFPS: 100/1\n")
        with patch("deinterlace_studio.processor.subprocess.Popen", return_value=success):
            self.assertEqual(
                processor._run_vspipe_info(
                    self.plan,
                    lambda _message: None,
                    None,
                ),
                6,
            )

        failed = InfoProcess("", returncode=2)
        with patch("deinterlace_studio.processor.subprocess.Popen", return_value=failed):
            with self.assertRaisesRegex(FastPreflightUnavailable, "exited with code 2"):
                processor._run_vspipe_info(self.plan, lambda _message: None, None)

        timed_out = InfoProcess("", timeout=True)
        with patch("deinterlace_studio.processor.subprocess.Popen", return_value=timed_out):
            with self.assertRaisesRegex(FastPreflightUnavailable, "did not finish"):
                processor._run_vspipe_info(
                    self.plan,
                    lambda _message: None,
                    None,
                    timeout=0.01,
                )
        self.assertTrue(timed_out.terminated)

    def test_vspipe_info_cancel_is_immediate_and_managed(self) -> None:
        class WaitingProcess:
            def __init__(self) -> None:
                self.returncode = None
                self.stdout = io.StringIO("")
                self.stderr = io.StringIO("")
                self.terminated = False

            def communicate(self, timeout=None):
                raise subprocess.TimeoutExpired("vspipe", timeout)

            def poll(self):
                return self.returncode

            def wait(self, timeout=None):
                return self.returncode

            def terminate(self):
                self.terminated = True
                self.returncode = -15

            def kill(self):
                self.terminated = True
                self.returncode = -9

        waiting = WaitingProcess()
        processor = JobProcessor()
        processor.cancel_event.set()
        with patch("deinterlace_studio.processor.subprocess.Popen", return_value=waiting):
            with self.assertRaises(ProcessingCancelled):
                processor._run_vspipe_info(self.plan, lambda _message: None, None)
        self.assertTrue(waiting.terminated)

    def test_graph_packet_contradiction_falls_back_then_fails_closed(self) -> None:
        processor = JobProcessor()
        with patch.object(processor, "_run_vspipe_info", return_value=8), patch.object(
            processor, "_run_decoded_preflight", return_value=3
        ):
            with self.assertRaisesRegex(ProcessingError, "graph reported 8 output frames"):
                processor._source_preflight(
                    self.plan,
                    self.source_probe,
                    caps(),
                    lambda _message: None,
                    None,
                )

    def test_run_propagates_contract_to_audit_and_blocks_vspipe_completion_mismatch(self) -> None:
        contract = SourcePreflightEvidence(
            method="vspipe_info_ffv1_packet_contract",
            source_frames=3,
            expected_output_frames=6,
            elapsed_seconds=0.01,
            packet_count=3,
            graph_output_frames=6,
            fast_path_eligible=True,
        )
        processor = JobProcessor()

        def mismatched_pipeline(_plan, *_args):
            _plan.partial_path.write_bytes(b"incomplete")
            return 5

        with patch.object(processor, "_source_preflight", return_value=contract), patch.object(
            processor, "_run_pipeline", side_effect=mismatched_pipeline
        ):
            mismatch = processor.run(self.plan, self.source_probe, idet(), caps())
        self.assertFalse(mismatch.success)
        self.assertIn("source preflight expected 6", mismatch.message)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.plan.partial_path.exists())

        processor = JobProcessor()

        def complete_pipeline(_plan, *_args):
            _plan.partial_path.write_bytes(b"complete")
            return 6

        valid = ValidationResult(True, (), (), None)
        with patch.object(processor, "_source_preflight", return_value=contract), patch.object(
            processor, "_run_pipeline", side_effect=complete_pipeline
        ), patch("deinterlace_studio.processor.validate_output", return_value=valid):
            result = processor.run(self.plan, self.source_probe, idet(), caps())
        self.assertTrue(result.success, result.message)
        payload = json.loads(result.report_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["source_preflight"]["method"], contract.method)
        self.assertEqual(payload["effective_output_expectation"]["frame_count"], 6)

    def test_full_decode_progress_is_reported_and_positive_frame_count_is_required(self) -> None:
        class FinishedProcess:
            def __init__(self) -> None:
                self.stdout = io.StringIO(
                    "frame=1\nspeed=2.0x\nprogress=continue\n"
                    "frame=3\nspeed=2.5x\nprogress=end\n"
                )
                self.stderr = io.StringIO("")
                self.returncode = 0

            def poll(self):
                return self.returncode

            def wait(self, timeout=None):
                return self.returncode

            def terminate(self):
                self.returncode = 1

            def kill(self):
                self.returncode = 1

        processor = JobProcessor()
        events: list[dict[str, str]] = []
        with patch("deinterlace_studio.processor.subprocess.Popen", return_value=FinishedProcess()):
            count = processor._run_decoded_preflight(
                Path("ffmpeg.exe"),
                self.plan,
                self.source_probe,
                lambda _message: None,
                events.append,
            )
        self.assertEqual(count, 3)
        progress = [event for event in events if event["phase"] == "preflight_full_progress"]
        self.assertTrue(progress)
        self.assertEqual(progress[-1]["frame"], "3")
        self.assertEqual(progress[-1]["percent"], "100.000")
        self.assertIn("eta_seconds", progress[0])

    def test_full_decode_fails_closed_on_exit_diagnostic_and_empty_result(self) -> None:
        class StaticProcess:
            def __init__(self, stdout: str, stderr: str, returncode: int) -> None:
                self.stdout = io.StringIO(stdout)
                self.stderr = io.StringIO(stderr)
                self.returncode = returncode

            def poll(self):
                return self.returncode

            def wait(self, timeout=None):
                return self.returncode

            def terminate(self):
                self.returncode = 1

            def kill(self):
                self.returncode = 1

        cases = (
            (
                StaticProcess("frame=1\nprogress=end\n", "", 3),
                "exited with code 3",
                ProcessingError,
            ),
            (
                StaticProcess("frame=1\nprogress=end\n", "corrupt decoded frame\n", 0),
                "decoder-integrity fault",
                SourceRepairRequiredError,
            ),
            (
                StaticProcess("frame=1\nprogress=end\n", "corrupt decoded frame\n", 3),
                "decoder-integrity fault",
                SourceRepairRequiredError,
            ),
            (
                StaticProcess("progress=end\n", "", 0),
                "without a positive frame count",
                ProcessingError,
            ),
        )
        for process, message, error_type in cases:
            with self.subTest(message=message), patch(
                "deinterlace_studio.processor.subprocess.Popen", return_value=process
            ):
                with self.assertRaisesRegex(error_type, message):
                    JobProcessor()._run_decoded_preflight(
                        Path("ffmpeg.exe"),
                        self.plan,
                        self.source_probe,
                        lambda _message: None,
                        None,
                    )

    def test_decoder_integrity_fault_is_structured_for_gui_recovery(self) -> None:
        processor = JobProcessor()
        with patch.object(
            processor,
            "_source_preflight",
            side_effect=SourceRepairRequiredError(
                "Full decoded source preflight reported a decoder-integrity fault (corrupt decoded frame)."
            ),
        ):
            result = processor.run(self.plan, self.source_probe, idet(), caps())
        self.assertFalse(result.success)
        self.assertFalse(result.canceled)
        self.assertEqual(result.failure_code, "source_repair_required")
        self.assertIn("decoder-integrity fault", result.message)
        self.assertIsNotNone(result.log_path)
        self.assertTrue(result.log_path.is_file())
        self.assertFalse(self.plan.partial_path.exists())

    def test_cancel_terminates_managed_full_decode_promptly(self) -> None:
        class BlockingStream:
            def __init__(self, stopped: threading.Event) -> None:
                self.stopped = stopped

            def __iter__(self):
                return self

            def __next__(self):
                self.stopped.wait()
                raise StopIteration

            def close(self):
                self.stopped.set()

        class BlockingProcess:
            def __init__(self) -> None:
                self.stopped = threading.Event()
                self.stdout = BlockingStream(self.stopped)
                self.stderr = BlockingStream(self.stopped)
                self.returncode = None
                self.terminate_called = False

            def poll(self):
                return self.returncode

            def wait(self, timeout=None):
                if not self.stopped.wait(timeout):
                    raise TimeoutError
                return self.returncode

            def terminate(self):
                self.terminate_called = True
                self.returncode = 1
                self.stopped.set()

            def kill(self):
                self.returncode = 1
                self.stopped.set()

        fake = BlockingProcess()
        processor = JobProcessor()
        captured: list[BaseException] = []

        def run_preflight() -> None:
            try:
                processor._run_decoded_preflight(
                    Path("ffmpeg.exe"),
                    self.plan,
                    self.source_probe,
                    lambda _message: None,
                    None,
                )
            except BaseException as exc:
                captured.append(exc)

        with patch("deinterlace_studio.processor.subprocess.Popen", return_value=fake):
            worker = threading.Thread(target=run_preflight)
            worker.start()
            deadline = time.monotonic() + 1.0
            while not processor._processes and time.monotonic() < deadline:
                time.sleep(0.01)
            processor.cancel()
            worker.join(timeout=2.0)
        self.assertFalse(worker.is_alive())
        self.assertTrue(fake.terminate_called)
        self.assertTrue(captured)
        self.assertIsInstance(captured[0], ProcessingCancelled)


if __name__ == "__main__":
    unittest.main()
