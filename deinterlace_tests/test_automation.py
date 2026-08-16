from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from tkinter import TclError, Tk
from unittest.mock import patch

from deinterlace_studio.automation import (
    DEINTERLACE_ARTIFACT_SUFFIXES,
    REPAIR_ARTIFACT_SUFFIXES,
    AutomaticRecoveryWorkflow,
    VolumeStorageCheck,
    automatic_recovery_applies_to_backend,
    choose_available_artifact_path,
    completed_artifacts,
    estimate_output_bytes,
    estimate_repair_bytes,
    storage_preflight,
    storage_summary,
)
from deinterlace_studio.models import (
    SOURCE_REPAIR_REQUIRED_FAILURE,
    JobSettings,
    PacketTimelineGap,
    SourceHealthReport,
)
from deinterlace_studio.gui import DeinterlaceStudioApp, ENGINE_LABELS, FAMILY_LABELS
from deinterlace_studio.planner import build_plan
from deinterlace_studio.settings import load_settings, save_settings
from deinterlace_tests.test_core import capabilities, media, report


def damaged_health(path: Path) -> SourceHealthReport:
    stat = path.stat()
    return SourceHealthReport(
        path=path,
        source_size=stat.st_size,
        source_mtime_ns=stat.st_mtime_ns,
        status="repair_required",
        reason="Fast packet scan found a 28.020-second timestamp hole.",
        elapsed_seconds=1.5,
        packet_count=204786,
        timestamped_packet_count=204786,
        unique_timestamp_count=204732,
        first_pts=0.016,
        last_pts=4125.921,
        typical_step_seconds=0.02,
        packet_timeline_span_seconds=4125.925,
        reported_duration_seconds=4125.945,
        duration_difference_seconds=0.02,
        gap_threshold_seconds=0.5,
        material_gap_count=1,
        largest_gaps=(PacketTimelineGap(137.877, 165.897, 28.020),),
        demux_warning_count=1,
        structural_warning_count=1,
        warning_samples=("container warning",),
        ffprobe_returncode=0,
    )


class AutomaticRecoveryHelperTests(unittest.TestCase):
    def test_automatic_recovery_is_exclusively_a_resolved_qtgmc_workflow(self) -> None:
        self.assertTrue(automatic_recovery_applies_to_backend("vapoursynth_qtgmc"))
        for backend in ("ffmpeg_bwdif", "ffmpeg_bwdif_cuda", "progressive", None):
            with self.subTest(backend=backend):
                self.assertFalse(automatic_recovery_applies_to_backend(backend))

    def test_unique_completed_artifact_set_never_overwrites_or_uses_reserved_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            preferred = root / "episode.deinterlaced.mkv"
            preferred.write_bytes(b"old output")
            preferred.with_name(preferred.name + ".Deinterlace.json").write_text("old")
            chosen = choose_available_artifact_path(
                preferred,
                DEINTERLACE_ARTIFACT_SUFFIXES,
                reserved=(root / "episode.deinterlaced.2.mkv",),
            )
            self.assertEqual(chosen.name, "episode.deinterlaced.3.mkv")
            self.assertEqual(preferred.read_bytes(), b"old output")
            self.assertFalse(any(path.exists() for path in completed_artifacts(chosen, DEINTERLACE_ARTIFACT_SUFFIXES)))

    def test_repair_sidecar_collision_advances_the_entire_set(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            preferred = root / "episode.qtgmc-repair.mkv"
            preferred.with_name(preferred.name + ".Repair.log").write_text("retained")
            chosen = choose_available_artifact_path(preferred, REPAIR_ARTIFACT_SUFFIXES)
            self.assertEqual(chosen.name, "episode.qtgmc-repair.2.mkv")
            self.assertEqual(
                preferred.with_name(preferred.name + ".Repair.log").read_text(),
                "retained",
            )

    def test_storage_preflight_groups_repair_and_output_on_one_volume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mkv"
            source.write_bytes(b"source")
            probed = media(source)
            settings = JobSettings(source, root / "final.mkv")
            plan = build_plan(settings, probed, report("tff"), capabilities())
            self.assertTrue(plan.valid, plan.errors)
            self.assertGreater(estimate_repair_bytes(probed), 0)
            self.assertGreater(estimate_output_bytes(plan, probed), estimate_repair_bytes(probed))
            with patch(
                "deinterlace_studio.automation.shutil.disk_usage",
                return_value=SimpleNamespace(free=10**15),
            ):
                checks = storage_preflight(
                    probed,
                    plan,
                    root / "repair.mkv",
                    root / "final.mkv",
                )
            self.assertEqual(len(checks), 1)
            self.assertTrue(checks[0].sufficient)
            self.assertIn("PASS", storage_summary(checks))
            self.assertGreater(
                checks[0].required_bytes,
                estimate_repair_bytes(probed) + estimate_output_bytes(plan, probed),
            )

    def test_low_storage_is_reported_before_an_unattended_chain(self) -> None:
        check = VolumeStorageCheck("D:\\", Path("D:/"), required_bytes=100, free_bytes=99)
        self.assertFalse(check.sufficient)
        self.assertIn("INSUFFICIENT", check.summary)

    def test_workflow_audit_links_original_repair_and_final_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mkv"
            source.write_bytes(b"source")
            requested = JobSettings(source, root / "chosen.mkv", family="hevc", bit_depth=10)
            selected = replace(requested, output_path=root / "chosen.2.mkv")
            workflow = AutomaticRecoveryWorkflow(
                original_source=source,
                trigger_health=damaged_health(source),
                requested_settings=requested,
                final_settings=selected,
                repair_output=root / "source.qtgmc-repair.mkv",
                analysis_mode="sampled",
                storage_preflight_summary="PASS D: requires 100 GiB; 1000 GiB free",
                stage="deinterlacing",
                validated_repair_source=root / "source.qtgmc-repair.mkv",
                repair_method="ffv1_rescue",
                repair_output_sha256="A" * 64,
                repair_log_path=root / "repair.log",
                repair_report_path=root / "repair.json",
                repeated_frames=703,
            )
            audit = workflow.audit()
            self.assertIsNotNone(audit)
            self.assertEqual(audit.original_source, source)
            self.assertEqual(audit.requested_output, root / "chosen.mkv")
            self.assertEqual(audit.selected_output, root / "chosen.2.mkv")
            self.assertEqual(audit.repair_method, "ffv1_rescue")
            self.assertEqual(audit.repeated_frames, 703)

    def test_automatic_recovery_setting_defaults_on_and_persists_off(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings_path = Path(directory) / "settings.json"
            defaults = load_settings(settings_path)
            self.assertTrue(defaults["automatic_repair_and_continue"])
            self.assertTrue(defaults["denoise_enabled"])
            self.assertEqual(defaults["denoiser"], "vs_bm3d")
            self.assertEqual(defaults["denoise_strength"], 4)
            self.assertEqual(defaults["denoise_temporal_radius"], 3)
            self.assertEqual(defaults["settings_schema_version"], 2)
            self.assertEqual(defaults["batch_output_dir"], "")
            self.assertFalse(defaults["batch_include_subfolders"])
            self.assertTrue(defaults["batch_continue_after_error"])
            values = load_settings(settings_path)
            values["automatic_repair_and_continue"] = False
            values["denoise_enabled"] = True
            values["denoiser"] = "vs_mvtools"
            values["denoise_strength"] = 3
            values["denoise_temporal_radius"] = 4
            save_settings(values, settings_path)
            saved = load_settings(settings_path)
            self.assertFalse(saved["automatic_repair_and_continue"])
            self.assertTrue(saved["denoise_enabled"])
            self.assertEqual(saved["denoiser"], "vs_mvtools")
            self.assertEqual(saved["denoise_strength"], 3)
            self.assertEqual(saved["denoise_temporal_radius"], 4)

            settings_path.write_text(
                '{"denoise_enabled": false, "denoiser": "vs_mvtools", '
                '"denoise_strength": 7, "denoise_temporal_radius": 2, '
                '"ffmpeg_path": "C:/Tools/ffmpeg.exe"}',
                encoding="utf-8",
            )
            migrated = load_settings(settings_path)
            self.assertEqual(migrated["settings_schema_version"], 2)
            self.assertTrue(migrated["denoise_enabled"])
            self.assertEqual(migrated["denoise_temporal_radius"], 3)
            self.assertEqual(migrated["denoiser"], "vs_mvtools")
            self.assertEqual(migrated["denoise_strength"], 7)
            self.assertEqual(migrated["ffmpeg_path"], "C:/Tools/ffmpeg.exe")


class AutomaticRecoveryGuiTests(unittest.TestCase):
    def _app(self, directory: str) -> tuple[Tk, DeinterlaceStudioApp, Path]:
        try:
            root = Tk()
        except TclError as exc:
            self.skipTest(f"Tk unavailable in this test runtime: {exc}")
        root.withdraw()
        app = DeinterlaceStudioApp(
            root,
            initial_capabilities=capabilities(),
            settings_path=Path(directory) / "settings.json",
        )
        # Build the damaged-source fixture without allowing the real Tk idle loop
        # to launch the workflow before an individual test can install its mocks.
        app.auto_repair_continue_var.set(False)
        source = Path(directory) / "damaged source.mkv"
        source.write_bytes(b"protected original")
        app._select_input_path(source)
        app.media = media(source)
        app.analysis = report("tff")
        app._set_source_health(damaged_health(source))
        app._update_cadence_labels()
        app._refresh_control_states()
        app._suggest_output()
        root.update()
        with patch.object(root, "after_idle"):
            app.auto_repair_continue_var.set(True)
        return root, app, source

    @staticmethod
    def _clear_health(path: Path) -> SourceHealthReport:
        return replace(
            damaged_health(path),
            status="clear",
            reason="Fast scan found no obvious damage.",
            material_gap_count=0,
            largest_gaps=(),
            demux_warning_count=0,
            structural_warning_count=0,
            warning_samples=(),
        )

    def test_disabled_or_healthy_analysis_never_starts_automatic_repair(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root, app, source = self._app(directory)
            try:
                app.auto_repair_continue_var.set(False)
                with patch.object(root, "after_idle") as after_idle:
                    self.assertFalse(app._maybe_begin_automatic_recovery())
                after_idle.assert_not_called()
                self.assertIsNone(app.auto_workflow)

                app._set_source_health(self._clear_health(source))
                app.auto_repair_continue_var.set(True)
                with patch.object(root, "after_idle") as after_idle:
                    self.assertFalse(app._maybe_begin_automatic_recovery())
                after_idle.assert_not_called()
                self.assertIsNone(app.auto_workflow)
            finally:
                app._on_close()

    def test_late_decoded_damage_automatically_enters_recovery_without_failure_popup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root, app, source = self._app(directory)
            try:
                app._set_source_health(self._clear_health(source))
                app._refresh_plan()
                self.assertTrue(app.plan.valid, app.plan.errors)
                preflight_log = Path(directory) / "retained-preflight.log"
                preflight_log.write_text("corrupt decoded frame", encoding="utf-8")
                failed = SimpleNamespace(
                    success=False,
                    canceled=False,
                    output_path=None,
                    output_sha256=None,
                    log_path=preflight_log,
                    message=(
                        "Full decoded source preflight reported a decoder-integrity fault "
                        "(corrupt decoded frame)."
                    ),
                    quarantine_path=None,
                    failure_code=SOURCE_REPAIR_REQUIRED_FAILURE,
                )
                with patch.object(root, "after_idle") as after_idle, patch(
                    "deinterlace_studio.gui.messagebox.showerror"
                ) as error:
                    app._handle_run_done(failed)
                error.assert_not_called()
                after_idle.assert_called_once_with(app._maybe_begin_automatic_recovery)
                self.assertTrue(app.source_health.repair_required)
                self.assertIn("fast compressed-packet scan could not prove", app.source_health.reason)
                self.assertIn("automatic QTGMC recovery is starting", app.status_var.get())
                self.assertIn(str(preflight_log), app.status_var.get())
                self.assertEqual(app.progress_var.get(), 0.0)
                self.assertIsNone(app.auto_workflow)

                storage = (
                    VolumeStorageCheck("C:\\", Path(directory), required_bytes=100, free_bytes=1000),
                )
                with patch("deinterlace_studio.gui.storage_preflight", return_value=storage), patch.object(
                    root, "after_idle"
                ) as recovery_idle:
                    self.assertTrue(app._maybe_begin_automatic_recovery())
                self.assertEqual(app.auto_workflow.stage, "repairing")
                self.assertEqual(app.auto_workflow.trigger_health.reason, app.source_health.reason)
                self.assertTrue(
                    any(call.args and call.args[0] == app._start_automatic_repair for call in recovery_idle.call_args_list)
                )
                self.assertEqual(source.read_bytes(), b"protected original")
            finally:
                app._on_close()

    def test_late_decoded_damage_remains_actionable_failure_when_automation_is_off(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root, app, source = self._app(directory)
            try:
                app._set_source_health(self._clear_health(source))
                app.auto_repair_continue_var.set(False)
                app._refresh_plan()
                failed = SimpleNamespace(
                    success=False,
                    canceled=False,
                    output_path=None,
                    output_sha256=None,
                    log_path=Path(directory) / "retained-preflight.log",
                    message="Full decoded source preflight reported a decoder-integrity fault.",
                    quarantine_path=None,
                    failure_code=SOURCE_REPAIR_REQUIRED_FAILURE,
                )
                with patch.object(root, "after_idle") as after_idle, patch(
                    "deinterlace_studio.gui.messagebox.showerror"
                ) as error:
                    app._handle_run_done(failed)
                error.assert_called_once()
                after_idle.assert_not_called()
                self.assertTrue(app.source_health.repair_required)
                self.assertIsNone(app.auto_workflow)
                self.assertIn("Processing failed", error.call_args.args[0])
                self.assertIn("Enable Automatic QTGMC recovery", error.call_args.args[1])
                self.assertEqual(source.read_bytes(), b"protected original")
            finally:
                app._on_close()

    def test_explicit_bwdif_cpu_and_cuda_bypass_automatic_repair(self) -> None:
        for backend in ("ffmpeg_bwdif", "ffmpeg_bwdif_cuda"):
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as directory:
                root, app, source = self._app(directory)
                try:
                    label = next(label for label, value in ENGINE_LABELS.items() if value == backend)
                    with patch.object(root, "after_idle"):
                        app.denoise_enabled_var.set(False)
                        app.engine_var.set(label)
                        app._refresh_control_states()
                    app._refresh_plan()
                    requested_output = app.output_var.get()

                    with patch("deinterlace_studio.gui.choose_available_artifact_path") as choose_path, patch(
                        "deinterlace_studio.gui.storage_preflight"
                    ) as storage, patch.object(root, "after_idle") as after_idle:
                        self.assertFalse(app._maybe_begin_automatic_recovery())

                    choose_path.assert_not_called()
                    storage.assert_not_called()
                    after_idle.assert_not_called()
                    self.assertIsNone(app.auto_workflow)
                    self.assertEqual(app.output_var.get(), requested_output)
                    self.assertTrue(app.plan.valid, app.plan.errors)
                    self.assertEqual(app.plan.selected_backend, backend)
                    self.assertEqual(str(app.start_button.cget("state")), "normal")
                    self.assertEqual(str(app.repair_button.cget("text")), "Repair required…")
                    self.assertEqual(source.read_bytes(), b"protected original")
                    self.assertIn("automatic repair is skipped", app.status_var.get())
                    self.assertTrue(
                        any("cannot restore missing/corrupt pictures" in warning for warning in app.plan.warnings)
                    )
                finally:
                    root.update_idletasks()
                    app._on_close()

    def test_automatic_backend_fallback_to_bwdif_also_skips_repair(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root, app, _source = self._app(directory)
            try:
                app.denoise_enabled_var.set(False)
                app.capabilities = capabilities(qtgmc=False)
                app._refresh_plan()
                requested_output = app.output_var.get()
                with patch("deinterlace_studio.gui.choose_available_artifact_path") as choose_path, patch(
                    "deinterlace_studio.gui.storage_preflight"
                ) as storage, patch.object(root, "after_idle") as after_idle:
                    self.assertFalse(app._maybe_begin_automatic_recovery())
                choose_path.assert_not_called()
                storage.assert_not_called()
                after_idle.assert_not_called()
                self.assertIsNone(app.auto_workflow)
                self.assertEqual(app.output_var.get(), requested_output)
                self.assertEqual(app.plan.selected_backend, "ffmpeg_bwdif")
                self.assertTrue(app.plan.valid, app.plan.errors)
                self.assertIn("BWDIF CPU will process the original directly", app.status_var.get())
            finally:
                app._on_close()

    def test_repair_success_reanalysis_and_final_success_continue_without_intermediate_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root, app, source = self._app(directory)
            try:
                hevc_label = next(label for label, value in FAMILY_LABELS.items() if value == "hevc")
                with patch.object(root, "after_idle"):
                    app.family_var.set(hevc_label)
                    app.bit_depth_var.set("12")
                    app.hardware_encode_var.set(True)
                    app.quality_var.set("17")
                    app.copy_subtitles_var.set(False)
                requested_output = Path(app.output_var.get())
                requested_output.write_bytes(b"retained prior output")
                storage = (
                    VolumeStorageCheck("C:\\", Path(directory), required_bytes=100, free_bytes=1000),
                )
                with patch("deinterlace_studio.gui.storage_preflight", return_value=storage), patch.object(
                    root, "after_idle"
                ) as after_idle:
                    self.assertTrue(app._maybe_begin_automatic_recovery())
                self.assertIsNotNone(app.auto_workflow)
                workflow = app.auto_workflow
                self.assertEqual(workflow.stage, "repairing")
                self.assertEqual(workflow.requested_settings.output_path, requested_output)
                self.assertEqual(workflow.requested_settings.family, "hevc")
                self.assertEqual(workflow.requested_settings.bit_depth, 12)
                self.assertTrue(workflow.requested_settings.hardware_encode)
                self.assertEqual(workflow.requested_settings.quality, 17)
                self.assertFalse(workflow.requested_settings.copy_subtitles)
                self.assertEqual(workflow.final_settings.output_path.name, requested_output.stem + ".2" + requested_output.suffix)
                self.assertEqual(requested_output.read_bytes(), b"retained prior output")
                self.assertNotEqual(workflow.repair_output, workflow.final_settings.output_path)
                self.assertTrue(
                    any(call.args and call.args[0] == app._start_automatic_repair for call in after_idle.call_args_list)
                )

                repaired = Path(directory) / "damaged source.qtgmc-repair.mkv"
                repaired.write_bytes(b"validated repair")
                repair_result = SimpleNamespace(
                    success=True,
                    canceled=False,
                    output_path=repaired,
                    method="ffv1_rescue",
                    repeated_frames=703,
                    dropped_frames=0,
                    output_sha256="A" * 64,
                    report_path=repaired.with_name(repaired.name + ".Repair.json"),
                    log_path=repaired.with_name(repaired.name + ".Repair.log"),
                    source_diagnosis=None,
                    quarantine_path=None,
                    message="validated",
                )
                with patch.object(app, "_analyze") as fresh_analysis, patch.object(
                    root, "after_idle", side_effect=lambda callback: callback()
                ), patch("deinterlace_studio.gui.messagebox.showinfo") as info:
                    app._handle_repair_done(repair_result)
                info.assert_not_called()
                fresh_analysis.assert_called_once_with("sampled")
                self.assertEqual(Path(app.input_var.get()), repaired)
                self.assertEqual(Path(app.output_var.get()), workflow.final_settings.output_path)
                self.assertEqual(workflow.stage, "reanalyzing")

                clear = self._clear_health(repaired)
                app._cancel_event_poll()
                with patch.object(app, "_start_processing") as start_processing, patch.object(
                    root, "after_idle", side_effect=lambda callback: callback()
                ), patch.object(root, "after", return_value="poll-id"):
                    app.events.put(("analysis_done", (media(repaired), report("tff"), clear)))
                    app._poll_events()
                start_processing.assert_called_once_with(automatic=True)
                self.assertEqual(workflow.stage, "deinterlacing")
                self.assertTrue(app.plan.valid, app.plan.errors)
                self.assertIsNotNone(app.plan.automatic_recovery)
                self.assertEqual(app.plan.automatic_recovery.original_source, source)
                self.assertEqual(app.plan.settings.input_path, repaired)
                self.assertEqual(app.plan.output_path, workflow.final_settings.output_path)
                self.assertEqual(app.plan.settings.family, "hevc")
                self.assertEqual(app.plan.settings.bit_depth, 12)
                self.assertTrue(app.plan.settings.hardware_encode)
                self.assertEqual(app.plan.settings.quality, 17)
                self.assertFalse(app.plan.settings.copy_subtitles)

                final = SimpleNamespace(
                    success=True,
                    canceled=False,
                    output_path=workflow.final_settings.output_path,
                    output_sha256="B" * 64,
                    log_path=workflow.final_settings.output_path.with_name("final.log"),
                    message="ok",
                    quarantine_path=None,
                )
                with patch("deinterlace_studio.gui.messagebox.showinfo") as info:
                    app._handle_run_done(final)
                info.assert_called_once()
                self.assertIn("Automatic repair", info.call_args.args[0])
                self.assertIsNone(app.auto_workflow)
                self.assertEqual(source.read_bytes(), b"protected original")
                self.assertEqual(requested_output.read_bytes(), b"retained prior output")
            finally:
                if root.winfo_exists():
                    app._on_close()

    def test_repaired_source_still_damaged_stops_before_encode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root, app, source = self._app(directory)
            try:
                requested = app._collect_settings()
                repaired = Path(directory) / "repair.mkv"
                repaired.write_bytes(b"still damaged")
                workflow = AutomaticRecoveryWorkflow(
                    original_source=source,
                    trigger_health=damaged_health(source),
                    requested_settings=requested,
                    final_settings=requested,
                    repair_output=repaired,
                    analysis_mode="sampled",
                    storage_preflight_summary="PASS",
                    stage="reanalyzing",
                    validated_repair_source=repaired,
                    repair_method="ffv1_rescue",
                )
                app.auto_workflow = workflow
                app._restoring_automatic_settings = True
                try:
                    app._select_input_path(repaired, suggest_output=False)
                finally:
                    app._restoring_automatic_settings = False
                app._restore_automatic_settings(requested)
                app._cancel_event_poll()
                with patch.object(app, "_start_processing") as start_processing, patch(
                    "deinterlace_studio.gui.messagebox.showerror"
                ) as error, patch.object(root, "after", return_value="poll-id"):
                    app.events.put(
                        ("analysis_done", (media(repaired), report("tff"), damaged_health(repaired)))
                    )
                    app._poll_events()
                start_processing.assert_not_called()
                error.assert_called_once()
                self.assertIsNone(app.auto_workflow)
                self.assertIn("still has a repair-required", app.status_var.get())
            finally:
                app._on_close()

    def test_low_storage_and_invalid_plan_stop_before_repair(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root, app, _source = self._app(directory)
            try:
                low = (
                    VolumeStorageCheck("C:\\", Path(directory), required_bytes=1000, free_bytes=1),
                )
                with patch("deinterlace_studio.gui.storage_preflight", return_value=low), patch(
                    "deinterlace_studio.gui.messagebox.showerror"
                ) as error, patch.object(root, "after_idle") as after_idle:
                    self.assertFalse(app._maybe_begin_automatic_recovery())
                error.assert_called_once()
                after_idle.assert_not_called()
                self.assertIsNone(app.auto_workflow)
                self.assertIn("storage preflight failed", app.status_var.get())

                app.analysis = report("mixed")
                with patch("deinterlace_studio.gui.storage_preflight") as storage, patch.object(
                    root, "after_idle"
                ) as after_idle:
                    self.assertFalse(app._maybe_begin_automatic_recovery())
                storage.assert_not_called()
                after_idle.assert_not_called()
                self.assertIsNone(app.auto_workflow)
                self.assertIn("final plan is invalid", app.status_var.get())
            finally:
                app._on_close()

    def test_complete_diagnosis_no_repair_needed_continues_and_cancel_stops_chain(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root, app, source = self._app(directory)
            try:
                settings = app._collect_settings()
                workflow = AutomaticRecoveryWorkflow(
                    original_source=source,
                    trigger_health=damaged_health(source),
                    requested_settings=settings,
                    final_settings=settings,
                    repair_output=Path(directory) / "repair.mkv",
                    analysis_mode="sampled",
                    storage_preflight_summary="PASS",
                )
                app.auto_workflow = workflow
                no_repair = SimpleNamespace(
                    success=True,
                    canceled=False,
                    output_path=None,
                    method="none",
                    repeated_frames=0,
                    dropped_frames=0,
                    output_sha256=None,
                    report_path=Path(directory) / "diagnosis.Repair.json",
                    log_path=Path(directory) / "diagnosis.Repair.log",
                    source_diagnosis=None,
                    quarantine_path=None,
                    message="healthy",
                )
                with patch.object(app, "_start_processing") as start_processing, patch.object(
                    root, "after_idle", side_effect=lambda callback: callback()
                ), patch("deinterlace_studio.gui.messagebox.showinfo") as info:
                    app._handle_repair_done(no_repair)
                info.assert_not_called()
                start_processing.assert_called_once_with(automatic=True)
                self.assertEqual(workflow.stage, "deinterlacing")
                self.assertEqual(workflow.repair_method, "none")
                self.assertEqual(app.source_health.status, "warning")
                self.assertTrue(app.plan.valid, app.plan.errors)
                self.assertEqual(app.plan.automatic_recovery.repair_method, "none")

                workflow.stage = "repairing"
                canceled = SimpleNamespace(
                    success=False,
                    canceled=True,
                    output_path=None,
                    method=None,
                    repeated_frames=0,
                    dropped_frames=0,
                    output_sha256=None,
                    report_path=None,
                    log_path=Path(directory) / "canceled.log",
                    source_diagnosis=None,
                    quarantine_path=None,
                    message="canceled",
                )
                app._handle_repair_done(canceled)
                self.assertIsNone(app.auto_workflow)
                self.assertIn("canceled during repair", app.status_var.get())
            finally:
                app._on_close()

    def test_repair_failure_stops_chain_and_retains_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root, app, source = self._app(directory)
            try:
                settings = app._collect_settings()
                app.auto_workflow = AutomaticRecoveryWorkflow(
                    original_source=source,
                    trigger_health=damaged_health(source),
                    requested_settings=settings,
                    final_settings=settings,
                    repair_output=Path(directory) / "repair.mkv",
                    analysis_mode="sampled",
                    storage_preflight_summary="PASS",
                )
                failed = SimpleNamespace(
                    success=False,
                    canceled=False,
                    output_path=None,
                    method="ffv1_rescue",
                    repeated_frames=0,
                    dropped_frames=0,
                    output_sha256=None,
                    report_path=Path(directory) / "failed.Repair.json",
                    log_path=Path(directory) / "failed.Repair.log",
                    source_diagnosis=None,
                    quarantine_path=Path(directory) / "failed.partial.quarantine",
                    message="validation rejected the repaired candidate",
                )
                with patch.object(app, "_analyze") as analyze, patch.object(
                    app, "_start_processing"
                ) as start_processing, patch("deinterlace_studio.gui.messagebox.showerror") as error:
                    app._handle_repair_done(failed)
                analyze.assert_not_called()
                start_processing.assert_not_called()
                error.assert_called_once()
                self.assertIsNone(app.auto_workflow)
                self.assertIn("validation rejected", app.status_var.get())
                self.assertIn("failed.Repair.log", error.call_args.args[1])
                self.assertEqual(source.read_bytes(), b"protected original")
            finally:
                app._on_close()

    def test_final_cancel_or_failure_clears_chain_and_retains_stage_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root, app, source = self._app(directory)
            try:
                settings = app._collect_settings()

                def workflow() -> AutomaticRecoveryWorkflow:
                    return AutomaticRecoveryWorkflow(
                        original_source=source,
                        trigger_health=damaged_health(source),
                        requested_settings=settings,
                        final_settings=settings,
                        repair_output=Path(directory) / "repair.mkv",
                        analysis_mode="sampled",
                        storage_preflight_summary="PASS",
                        stage="deinterlacing",
                        validated_repair_source=Path(directory) / "repair.mkv",
                        repair_method="ffv1_rescue",
                    )

                app.auto_workflow = workflow()
                canceled = SimpleNamespace(
                    success=False,
                    canceled=True,
                    output_path=None,
                    output_sha256=None,
                    log_path=Path(directory) / "final-canceled.log",
                    message="canceled",
                    quarantine_path=None,
                )
                with patch("deinterlace_studio.gui.messagebox.showerror") as error:
                    app._handle_run_done(canceled)
                error.assert_not_called()
                self.assertIsNone(app.auto_workflow)
                self.assertIn("Final processing canceled", app.status_var.get())

                app.auto_workflow = workflow()
                failed = SimpleNamespace(
                    success=False,
                    canceled=False,
                    output_path=None,
                    output_sha256=None,
                    log_path=Path(directory) / "final-failed.log",
                    message="final validation rejected the candidate",
                    quarantine_path=Path(directory) / "final.quarantine",
                    failure_code=SOURCE_REPAIR_REQUIRED_FAILURE,
                )
                with patch("deinterlace_studio.gui.messagebox.showerror") as error:
                    app._handle_run_done(failed)
                error.assert_called_once()
                self.assertIsNone(app.auto_workflow)
                self.assertIn("Final processing stopped", app.status_var.get())
                self.assertIn("final validation rejected", error.call_args.args[1])
                self.assertEqual(source.read_bytes(), b"protected original")
            finally:
                app._on_close()


if __name__ == "__main__":
    unittest.main()
