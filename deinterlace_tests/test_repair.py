from __future__ import annotations

import json
import os
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

from deinterlace_studio.models import CapabilityReport, MediaProbe, StreamInfo
from deinterlace_studio.repair import (
    RepairCancelled,
    RepairError,
    RepairRequest,
    RepairValidation,
    SourceRepairer,
    TimelineDiagnosis,
    TimelineGap,
    _compare_stream_group,
    build_remux_command,
    build_rescue_command,
    choose_ffv1_pixel_format,
    classify_timeline_condition,
)


def source_media(path: Path) -> MediaProbe:
    return MediaProbe(
        path=path,
        format_name="matroska,webm",
        format_long_name="Matroska",
        duration=5.0,
        size=path.stat().st_size,
        bit_rate=1,
        start_time=0.0,
        streams=(
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
                field_order="tt",
                color_range="pc",
                color_space="bt470bg",
                color_transfer="bt470bg",
                color_primaries="bt470bg",
                tags={"title": "Main video", "DURATION": "00:00:05.000"},
                disposition={"default": 1},
            ),
            StreamInfo(
                index=1,
                codec_type="audio",
                codec_name="flac",
                tags={"language": "yue", "title": "Cantonese"},
                disposition={"default": 1},
            ),
            StreamInfo(
                index=2,
                codec_type="subtitle",
                codec_name="subrip",
                tags={"language": "chi"},
                disposition={"default": 1},
            ),
        ),
        chapters=({"id": 0, "start_time": "0.0", "end_time": "5.0", "tags": {"title": "Opening"}},),
        format_tags={"title": "Episode", "encoder": "old muxer"},
    )


def capabilities() -> CapabilityReport:
    return CapabilityReport(
        ffmpeg_path=Path("ffmpeg.exe"),
        ffprobe_path=Path("ffprobe.exe"),
        ffmpeg_version="9.0",
        ffmpeg_configuration="",
        filters=frozenset({"fps", "setfield"}),
        encoders=frozenset({"ffv1"}),
        encoder_pixel_formats={"ffv1": ("yuv420p", "yuv420p10le", "yuv444p16le")},
        hwaccels=frozenset(),
        vspipe_path=None,
        vapoursynth_version=None,
        qtgmc_ready=False,
        qtgmc_diagnostic="not needed",
        qtgmc_install_command=None,
    )


def diagnosis(path: Path, *, method: str = "ffv1_rescue") -> TimelineDiagnosis:
    gap = TimelineGap(50, 2.0, 3.0, 1.0, 24)
    return TimelineDiagnosis(
        source_path=path,
        nominal_rate=Fraction(25, 1),
        reported_video_duration=5.0,
        container_duration=5.0,
        decoded_frames=100,
        timestamped_frames=100,
        decoded_cfr_span_seconds=4.0,
        first_pts=0.0,
        last_pts=4.96,
        pts_duration_seconds=5.0,
        material_gaps=(gap,) if method == "ffv1_rescue" else (),
        non_monotonic_steps=0,
        sampled_interlaced_frames=100,
        sampled_progressive_frames=0,
        decoder_warning_count=1 if method == "ffv1_rescue" else 0,
        decoder_warning_samples=("corrupt input",) if method == "ffv1_rescue" else (),
        severe_warning_count=1 if method == "ffv1_rescue" else 0,
        qtgmc_error="timeline mismatch" if method != "none" else None,
        classification="timestamp_gap_with_decode_errors" if method == "ffv1_rescue" else "healthy",
        recommended_method=method,
    )


class RepairDecisionTests(unittest.TestCase):
    def test_stream_sync_validation_uses_normalized_stream_timestamps(self) -> None:
        source_stream = StreamInfo(
            index=3,
            codec_type="subtitle",
            codec_name="subrip",
            start_time=0.488,
            disposition={"default": 1},
        )
        output_stream = StreamInfo(
            index=3,
            codec_type="subtitle",
            codec_name="subrip",
            start_time=0.488,
            disposition={"default": 1},
        )
        errors: list[str] = []
        _compare_stream_group(
            "subtitle",
            (source_stream,),
            (output_stream,),
            errors,
            source_format_start=0.0,
        )
        self.assertEqual(errors, [])

    def test_classification_routes_only_metadata_to_remux(self) -> None:
        self.assertEqual(
            classify_timeline_condition(
                qtgmc_error=None,
                material_gap_count=0,
                non_monotonic_steps=0,
                severe_warning_count=0,
                timestamps_complete=True,
            ),
            ("healthy", "none"),
        )
        self.assertEqual(
            classify_timeline_condition(
                qtgmc_error="duration mismatch",
                material_gap_count=0,
                non_monotonic_steps=0,
                severe_warning_count=0,
                timestamps_complete=True,
            ),
            ("container_or_duration_metadata", "stream_copy_remux"),
        )
        self.assertEqual(
            classify_timeline_condition(
                qtgmc_error="duration mismatch",
                material_gap_count=1,
                non_monotonic_steps=0,
                severe_warning_count=2,
                timestamps_complete=True,
            ),
            ("timestamp_gap_with_decode_errors", "ffv1_rescue"),
        )
        self.assertEqual(
            classify_timeline_condition(
                qtgmc_error="duration mismatch",
                material_gap_count=0,
                non_monotonic_steps=0,
                severe_warning_count=0,
                timestamps_complete=False,
            ),
            ("undetermined_missing_timestamps", "blocked"),
        )

    def test_ffv1_pixel_format_is_never_silently_reduced(self) -> None:
        supported = ("yuv420p", "yuv420p10le", "yuv444p16le")
        self.assertEqual(choose_ffv1_pixel_format("yuv420p10le", supported), "yuv420p10le")
        self.assertEqual(choose_ffv1_pixel_format("yuvj420p", supported), "yuv420p")
        with self.assertRaises(RepairError):
            choose_ffv1_pixel_format("xyz12", supported)

    def test_commands_preserve_unicode_tracks_geometry_and_gap_timestamps(self) -> None:
        source = Path("D:/影像/輸入 video.mkv")
        partial = Path("D:/影像/.輸出 partial.mkv")
        media = source_media(Path(__file__))
        media = MediaProbe(**{**media.__dict__, "path": source})
        diag = diagnosis(source)
        remux = build_remux_command(Path("ffmpeg.exe"), source, partial)
        self.assertIn(str(source), remux)
        self.assertIn("0:a?", remux)
        self.assertIn("0:s?", remux)
        self.assertNotIn("discardcorrupt", " ".join(remux))
        rescue, target = build_rescue_command(
            Path("ffmpeg.exe"), source, partial, media, diag, ("yuv420p",)
        )
        rendered = " ".join(rescue)
        self.assertEqual(target, "yuv420p")
        self.assertIn("setpts=PTS-STARTPTS", rendered)
        self.assertIn("fps=fps=25/1", rendered)
        self.assertIn("scale=in_range=pc:out_range=pc", rendered)
        self.assertIn("setsar=sar=349/240", rendered)
        self.assertIn("setfield=mode=tff", rendered)
        self.assertIn("-color_range pc", rendered)
        self.assertIn("-c:v:0 ffv1", rendered)
        self.assertIn("-c copy", rendered)


class RepairTransactionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "來源.mkv"
        self.source.write_bytes(b"original source")
        self.output = self.root / "來源.qtgmc-repair.mkv"
        self.media = source_media(self.source)
        self.caps = capabilities()

    def tearDown(self) -> None:
        self.temp.cleanup()

    @staticmethod
    def _healthy_output(path: Path, frames: int = 125) -> TimelineDiagnosis:
        return TimelineDiagnosis(
            source_path=path,
            nominal_rate=Fraction(25, 1),
            reported_video_duration=5.0,
            container_duration=5.0,
            decoded_frames=frames,
            timestamped_frames=frames,
            decoded_cfr_span_seconds=5.0,
            first_pts=0.0,
            last_pts=4.96,
            pts_duration_seconds=5.0,
            material_gaps=(),
            non_monotonic_steps=0,
            sampled_interlaced_frames=frames,
            sampled_progressive_frames=0,
            decoder_warning_count=0,
            decoder_warning_samples=(),
            severe_warning_count=0,
            qtgmc_error=None,
            classification="healthy",
            recommended_method="none",
        )

    def test_rescue_reuses_full_decode_only_after_identical_atomic_promotion(self) -> None:
        repairer = SourceRepairer()
        source_diag = diagnosis(self.source)
        output_diag = self._healthy_output(self.output)
        valid = RepairValidation(True, (), (), self.media, output_diag)

        def encode(command, **_kwargs):
            Path(command[-1]).write_bytes(b"validated rescue")
            return {"progress": "end"}

        with patch.object(repairer, "diagnose", return_value=source_diag), patch.object(
            repairer, "_run_ffmpeg", side_effect=encode
        ), patch.object(repairer, "_validate_candidate", side_effect=[valid, valid]) as validate:
            result = repairer.run(
                RepairRequest(self.source, self.output),
                self.media,
                self.caps,
            )
        self.assertTrue(result.success, result.message)
        self.assertEqual(validate.call_count, 2)
        self.assertNotIn("prior_full_validation", validate.call_args_list[0].kwargs)
        self.assertIs(validate.call_args_list[1].kwargs["prior_full_validation"], valid)
        self.assertEqual(self.source.read_bytes(), b"original source")
        self.assertEqual(self.output.read_bytes(), b"validated rescue")
        self.assertEqual(result.method, "ffv1_rescue")
        self.assertEqual(result.repeated_frames, 25)
        self.assertTrue(result.log_path.is_file())
        self.assertTrue(result.report_path.is_file())
        payload = json.loads(result.report_path.read_text(encoding="utf-8"))
        strategy = payload["validation_strategy"]
        self.assertEqual(strategy["partial_full_decode_validations"], 1)
        self.assertEqual(strategy["promoted_full_decode_repeats"], 0)
        self.assertTrue(strategy["same_file_atomic_promotion_verified"])
        self.assertEqual(strategy["sha256_passes"], 1)

    def test_atomic_promotion_identity_mismatch_quarantines_and_restores_old_output(self) -> None:
        self.output.write_bytes(b"old repair")
        repairer = SourceRepairer()
        source_diag = diagnosis(self.source)
        output_diag = self._healthy_output(self.output)
        valid = RepairValidation(True, (), (), self.media, output_diag)

        def encode(command, **_kwargs):
            Path(command[-1]).write_bytes(b"new candidate")
            return {}

        with patch.object(repairer, "diagnose", return_value=source_diag), patch.object(
            repairer, "_run_ffmpeg", side_effect=encode
        ), patch.object(repairer, "_validate_candidate", return_value=valid), patch(
            "deinterlace_studio.repair._promotion_identity",
            side_effect=[(13, 10, 1, 2), (13, 10, 1, 3)],
        ):
            result = repairer.run(
                RepairRequest(self.source, self.output, overwrite_approved=True),
                self.media,
                self.caps,
            )
        self.assertFalse(result.success)
        self.assertIn("exact file identity", result.message)
        self.assertEqual(self.output.read_bytes(), b"old repair")
        self.assertIsNotNone(result.quarantine_path)
        self.assertEqual(result.quarantine_path.read_bytes(), b"new candidate")

    def test_final_validation_failure_quarantines_candidate_and_restores_old_output(self) -> None:
        self.output.write_bytes(b"old repair")
        repairer = SourceRepairer()
        source_diag = diagnosis(self.source)
        output_diag = self._healthy_output(self.output)
        valid = RepairValidation(True, (), (), self.media, output_diag)
        invalid = RepairValidation(False, ("full decode failed",), (), self.media, output_diag)

        def encode(command, **_kwargs):
            Path(command[-1]).write_bytes(b"new candidate")
            return {}

        with patch.object(repairer, "diagnose", return_value=source_diag), patch.object(
            repairer, "_run_ffmpeg", side_effect=encode
        ), patch.object(repairer, "_validate_candidate", side_effect=[valid, invalid]):
            result = repairer.run(
                RepairRequest(self.source, self.output, overwrite_approved=True),
                self.media,
                self.caps,
            )
        self.assertFalse(result.success)
        self.assertEqual(self.output.read_bytes(), b"old repair")
        self.assertIsNotNone(result.quarantine_path)
        self.assertEqual(result.quarantine_path.read_bytes(), b"new candidate")

    def test_bounded_final_reopen_does_not_repeat_diagnosis_or_subtitle_scan(self) -> None:
        repairer = SourceRepairer()
        source_diag = diagnosis(self.source, method="stream_copy_remux")
        output_diag = self._healthy_output(self.output, frames=source_diag.decoded_frames)
        prior = RepairValidation(True, (), (), self.media, output_diag)
        with patch("deinterlace_studio.repair.probe_media", return_value=self.media), patch.object(
            repairer, "diagnose"
        ) as diagnose_call, patch.object(repairer, "_packet_fingerprints") as packet_scan:
            result = repairer._validate_candidate(
                self.caps.ffprobe_path,
                self.media,
                source_diag,
                self.output,
                method="stream_copy_remux",
                target_pix_fmt=None,
                log_callback=lambda _line: None,
                progress_callback=None,
                phase="bounded final reopen",
                prior_full_validation=prior,
            )
        self.assertTrue(result.valid, result.errors)
        diagnose_call.assert_not_called()
        packet_scan.assert_not_called()
        self.assertIn("identical thoroughly validated partial", " ".join(result.warnings))

    def test_cancel_removes_partial_and_preserves_existing_output(self) -> None:
        self.output.write_bytes(b"old repair")
        repairer = SourceRepairer()

        def cancel_encode(command, **_kwargs):
            Path(command[-1]).write_bytes(b"partial")
            raise RepairCancelled("Source repair canceled")

        with patch.object(repairer, "diagnose", return_value=diagnosis(self.source)), patch.object(
            repairer, "_run_ffmpeg", side_effect=cancel_encode
        ):
            result = repairer.run(
                RepairRequest(self.source, self.output, overwrite_approved=True),
                self.media,
                self.caps,
            )
        self.assertTrue(result.canceled)
        self.assertEqual(self.output.read_bytes(), b"old repair")
        self.assertFalse(any(self.root.glob("*.partial.*")))
        self.assertTrue(result.log_path.is_file())
        self.assertTrue(result.report_path.is_file())

    def test_cancel_requested_before_worker_enters_run_is_not_lost(self) -> None:
        repairer = SourceRepairer()
        repairer.cancel()

        def canceled_diagnosis(*_args, **_kwargs):
            repairer._check_canceled()

        with patch.object(repairer, "diagnose", side_effect=canceled_diagnosis):
            result = repairer.run(RepairRequest(self.source, self.output), self.media, self.caps)
        self.assertTrue(result.canceled)
        self.assertFalse(self.output.exists())
        self.assertEqual(self.source.read_bytes(), b"original source")

    def test_sidecar_promotion_failure_restores_prior_artifacts_and_retains_evidence(self) -> None:
        self.output.write_bytes(b"old repair")
        final_log = self.output.with_name(self.output.name + ".Repair.log")
        final_report = self.output.with_name(self.output.name + ".Repair.json")
        final_log.write_text("old log", encoding="utf-8")
        final_report.write_text("old report", encoding="utf-8")
        repairer = SourceRepairer()
        source_diag = diagnosis(self.source)
        output_diag = self._healthy_output(self.output)
        valid = RepairValidation(True, (), (), self.media, output_diag)

        def encode(command, **_kwargs):
            Path(command[-1]).write_bytes(b"new candidate")
            return {}

        real_replace = os.replace
        injected = False

        def fail_report_promotion(source, destination):
            nonlocal injected
            if Path(destination) == final_report and not injected:
                injected = True
                raise OSError("simulated report promotion failure")
            return real_replace(source, destination)

        with patch.object(repairer, "diagnose", return_value=source_diag), patch.object(
            repairer, "_run_ffmpeg", side_effect=encode
        ), patch.object(repairer, "_validate_candidate", side_effect=[valid, valid]), patch(
            "deinterlace_studio.repair.os.replace", side_effect=fail_report_promotion
        ):
            result = repairer.run(
                RepairRequest(self.source, self.output, overwrite_approved=True),
                self.media,
                self.caps,
            )
        self.assertFalse(result.success)
        self.assertEqual(self.output.read_bytes(), b"old repair")
        self.assertEqual(final_log.read_text(encoding="utf-8"), "old log")
        self.assertEqual(final_report.read_text(encoding="utf-8"), "old report")
        self.assertIsNotNone(result.quarantine_path)
        self.assertTrue(result.quarantine_path.is_file())
        self.assertIsNotNone(result.log_path)
        self.assertTrue(result.log_path.is_file())
        self.assertIsNotNone(result.report_path)
        self.assertTrue(result.report_path.is_file())

    def test_source_change_before_promotion_is_blocked(self) -> None:
        repairer = SourceRepairer()
        source_diag = diagnosis(self.source)
        output_diag = self._healthy_output(self.output)
        valid = RepairValidation(True, (), (), self.media, output_diag)

        def encode(command, **_kwargs):
            Path(command[-1]).write_bytes(b"candidate")
            self.source.write_bytes(b"changed source")
            return {}

        with patch.object(repairer, "diagnose", return_value=source_diag), patch.object(
            repairer, "_run_ffmpeg", side_effect=encode
        ), patch.object(repairer, "_validate_candidate", return_value=valid):
            result = repairer.run(RepairRequest(self.source, self.output), self.media, self.caps)
        self.assertFalse(result.success)
        self.assertIn("source file changed", result.message.lower())
        self.assertFalse(self.output.exists())

    def test_automatic_remux_failure_switches_to_validated_ffv1_rescue(self) -> None:
        repairer = SourceRepairer()
        source_diag = diagnosis(self.source, method="stream_copy_remux")
        output_diag = self._healthy_output(self.output)
        invalid_remux = RepairValidation(False, ("stale duration remains",), (), self.media, source_diag)
        valid_rescue = RepairValidation(True, (), (), self.media, output_diag)
        commands: list[tuple[str, ...]] = []

        def encode(command, **_kwargs):
            commands.append(command)
            Path(command[-1]).write_bytes(b"candidate")
            return {}

        with patch.object(repairer, "diagnose", return_value=source_diag), patch.object(
            repairer, "_run_ffmpeg", side_effect=encode
        ), patch.object(
            repairer,
            "_validate_candidate",
            side_effect=[invalid_remux, valid_rescue, valid_rescue],
        ):
            result = repairer.run(RepairRequest(self.source, self.output), self.media, self.caps)
        self.assertTrue(result.success, result.message)
        self.assertEqual(result.method, "ffv1_rescue")
        self.assertEqual(len(commands), 2)
        self.assertIn("-c:v:0", commands[1])
        self.assertEqual(self.source.read_bytes(), b"original source")

    def test_output_collision_is_never_authorizable(self) -> None:
        result = SourceRepairer().run(
            RepairRequest(self.source, self.source, overwrite_approved=True),
            self.media,
            self.caps,
        )
        self.assertFalse(result.success)
        self.assertIn("different paths", result.message)

    def test_existing_artifact_requires_explicit_replacement_approval(self) -> None:
        self.output.write_bytes(b"prior repair")
        result = SourceRepairer().run(
            RepairRequest(self.source, self.output),
            self.media,
            self.caps,
        )
        self.assertFalse(result.success)
        self.assertIn("explicit replacement approval", result.message)
        self.assertEqual(self.output.read_bytes(), b"prior repair")

    def test_healthy_source_creates_audit_only_and_preserves_prior_output(self) -> None:
        self.output.write_bytes(b"prior repair")
        repairer = SourceRepairer()
        healthy = diagnosis(self.source, method="none")
        with patch.object(repairer, "diagnose", return_value=healthy), patch.object(
            repairer, "_run_ffmpeg"
        ) as encode:
            result = repairer.run(
                RepairRequest(self.source, self.output, overwrite_approved=True),
                self.media,
                self.caps,
            )
        self.assertTrue(result.success, result.message)
        self.assertEqual(result.method, "none")
        self.assertIsNone(result.output_path)
        self.assertEqual(self.output.read_bytes(), b"prior repair")
        self.assertTrue(result.log_path.is_file())
        self.assertTrue(result.report_path.is_file())
        encode.assert_not_called()

    def test_non_matroska_output_is_rejected_before_diagnosis(self) -> None:
        output = self.root / "bad-output.mp4"
        repairer = SourceRepairer()
        with patch.object(repairer, "diagnose") as diagnose_call:
            result = repairer.run(RepairRequest(self.source, output), self.media, self.caps)
        self.assertFalse(result.success)
        self.assertIn("Matroska", result.message)
        self.assertFalse(output.exists())
        diagnose_call.assert_not_called()


if __name__ == "__main__":
    unittest.main()
