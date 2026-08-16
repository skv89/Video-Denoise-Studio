from __future__ import annotations

import tempfile
import time
import unittest
from fractions import Fraction
from pathlib import Path
from tkinter import TclError, Tk
from types import SimpleNamespace
from unittest.mock import patch

from deinterlace_studio.gui import (
    BACKEND_GPU_GUIDE_TEXT,
    CADENCE_GUIDE_TEXT,
    DENOISE_GUIDE_TEXT,
    DENOISER_LABELS,
    DeinterlaceStudioApp,
    ENGINE_LABELS,
    FAMILY_LABELS,
    HW_DECODE_LABELS,
    SOURCE_TIMELINE_GUIDE_TEXT,
    cadence_labels_for_media,
)
from deinterlace_studio.models import CapabilityReport, PacketTimelineGap, SourceHealthReport
from deinterlace_tests.test_core import (
    capabilities as test_capabilities,
    media as test_media,
    report as test_idet_report,
)


class GuiSmokeTests(unittest.TestCase):
    def test_prores_suggestion_stays_native_mov_when_subrip_can_be_converted(self) -> None:
        try:
            root = Tk()
        except TclError as exc:
            self.skipTest(f"Tk unavailable in this test runtime: {exc}")
        with tempfile.TemporaryDirectory() as directory:
            try:
                root.withdraw()
                source = Path(directory) / "interlaced source.mkv"
                source.write_bytes(b"fixture")
                app = DeinterlaceStudioApp(
                    root,
                    initial_capabilities=test_capabilities(),
                    settings_path=Path(directory) / "settings.json",
                )
                app.input_var.set(str(source))
                app.media = test_media(source, subtitle="subrip")
                prores_label = next(
                    label for label, value in FAMILY_LABELS.items() if value == "prores"
                )
                app.family_var.set(prores_label)
                app.bit_depth_var.set("10")
                app.copy_subtitles_var.set(True)
                app._suggest_output()
                self.assertTrue(app.output_var.get().endswith(".deinterlaced.mov"))
            finally:
                root.destroy()

    def test_dnxhr_gui_hides_unproven_12_bit_and_future_support_restores_it(self) -> None:
        try:
            root = Tk()
        except TclError as exc:
            self.skipTest(f"Tk unavailable in this test runtime: {exc}")
        with tempfile.TemporaryDirectory() as directory:
            try:
                root.withdraw()
                app = DeinterlaceStudioApp(
                    root,
                    initial_capabilities=test_capabilities(dnx12=False),
                    settings_path=Path(directory) / "settings.json",
                )
                dnxhr_label = next(label for label, value in FAMILY_LABELS.items() if value == "dnxhr")
                app.family_var.set(dnxhr_label)
                root.update_idletasks()
                self.assertEqual(tuple(app.depth_combo.cget("values")), ("10",))
                self.assertEqual(app.bit_depth_var.get(), "10")
                self.assertIn("12-bit is hidden", app.depth_hint_var.get())

                app.capabilities = test_capabilities(dnx12=True)
                app._refresh_control_states()
                root.update_idletasks()
                self.assertEqual(tuple(app.depth_combo.cget("values")), ("10", "12"))
                self.assertEqual(app.depth_hint_var.get(), "")
            finally:
                root.destroy()

    def test_backend_help_confirms_bwdif_temporal_check_and_nvenc_contract(self) -> None:
        normalized = " ".join(BACKEND_GPU_GUIDE_TEXT.split())
        self.assertIn("BWDIF is temporal as well as spatial", normalized)
        self.assertIn("previous, current, and next", normalized)
        self.assertIn("not “turn temporal analysis off.”", normalized)
        self.assertIn("preset=p7", normalized)
        self.assertIn("tune=uhq", normalized)
        self.assertIn("multipass=fullres", normalized)

    def test_resolve_editor_preset_selects_dnxhr_mov_and_safe_track_set(self) -> None:
        try:
            root = Tk()
        except TclError as exc:
            self.skipTest(f"Tk unavailable in this test runtime: {exc}")
        with tempfile.TemporaryDirectory() as directory:
            try:
                root.withdraw()
                source = Path(directory) / "interlaced source.mkv"
                source.write_bytes(b"fixture")
                app = DeinterlaceStudioApp(
                    root,
                    initial_capabilities=test_capabilities(),
                    settings_path=Path(directory) / "settings.json",
                )
                app.input_var.set(str(source))
                app.media = test_media(source, subtitle="subrip")
                app.analysis = test_idet_report("tff")
                app.output_var.set(str(Path(directory) / "old-output.mkv"))
                with patch("deinterlace_studio.gui.messagebox.askyesno", return_value=True):
                    app._apply_resolve_editor_preset()
                self.assertEqual(app.family_var.get(), "Avid DNxHR 444 (editor interchange)")
                self.assertEqual(app.bit_depth_var.get(), "10")
                self.assertFalse(app.hardware_encode_var.get())
                self.assertTrue(app.copy_audio_var.get())
                self.assertFalse(app.copy_subtitles_var.get())
                self.assertFalse(app.copy_attachments_var.get())
                self.assertFalse(app.copy_data_var.get())
                self.assertTrue(app.copy_chapters_var.get())
                self.assertTrue(app.copy_metadata_var.get())
                self.assertTrue(app.output_var.get().endswith(".deinterlaced.resolve-editor.mov"))
                self.assertIn("DNxHR 444 10-bit MOV", app.status_var.get())
            finally:
                root.destroy()

    def test_high_dpi_scrollable_startup_state(self) -> None:
        try:
            root = Tk()
        except TclError as exc:
            self.skipTest(f"Tk unavailable in this test runtime: {exc}")
        with tempfile.TemporaryDirectory() as directory:
            try:
                root.withdraw()
                root.tk.call("tk", "scaling", 2.0)
                root.geometry("980x720")
                caps = CapabilityReport(
                    ffmpeg_path=Path("ffmpeg.exe"),
                    ffprobe_path=Path("ffprobe.exe"),
                    ffmpeg_version="ffmpeg test",
                    ffmpeg_configuration="",
                    filters=frozenset({"bwdif", "idet", "fftdnoiz", "atadenoise"}),
                    encoders=frozenset({"ffv1"}),
                    encoder_pixel_formats={"ffv1": ("yuv444p16le",)},
                    hwaccels=frozenset(),
                    vspipe_path=None,
                    vapoursynth_version="R78",
                    qtgmc_ready=False,
                    qtgmc_diagnostic="missing",
                    qtgmc_install_command="python -m pip install vsjetpack[deinterlace]",
                    interlace_runtime_verified={
                        "d3d12_custom": False,
                        "d3d12_bob": True,
                        "d3d12_custom_p010": False,
                        "d3d12_bob_p010": False,
                    },
                    interlace_runtime_diagnostics={
                        "ffmpeg9_source_audit": "No new FFmpeg 9 deinterlacing-quality algorithm.",
                        "d3d12_custom": "Runtime probe failed; excluded from output backends.",
                        "d3d12_bob": "Runtime probe passed; lower quality than BWDIF/QTGMC.",
                        "d3d12_custom_p010": "No 10-bit deinterlacing methods supported by hardware.",
                        "d3d12_bob_p010": "No 10-bit deinterlacing methods supported by hardware.",
                    },
                    ffmpeg_selection_source="process PATH entry 1",
                    ffmpeg_discovery_diagnostics=(
                        "SELECTED [process PATH entry 1] C:\\FFmpeg — unconfirmed Git/date-stamped banner",
                        "NOT SELECTED [process PATH entry 2] D:\\Other — unconfirmed Git/date-stamped banner",
                    ),
                    ffprobe_version="ffprobe test",
                    denoise_capabilities={
                        "ffmpeg_fftdnoiz": True,
                        "ffmpeg_atadenoise": True,
                        "vs_bm3d": False,
                        "vs_dfttest": False,
                        "vs_mvtools": False,
                        "vs_nlmeans": False,
                    },
                    denoise_backends={
                        "ffmpeg_fftdnoiz": "ffmpeg",
                        "ffmpeg_atadenoise": "ffmpeg",
                    },
                )
                app = DeinterlaceStudioApp(
                    root,
                    initial_capabilities=caps,
                    settings_path=Path(directory) / "settings.json",
                )
                root.update_idletasks()
                result = app.self_test()
                self.assertEqual(result["application_version"], "1.10.1")
                self.assertIn("Deinterlace Studio 1.10.1", result["title"])
                self.assertEqual(result["engine_choices"], 5)
                self.assertEqual(result["family_choices"], 5)
                self.assertTrue(result["start_initially_disabled"])
                self.assertEqual(result["default_output_cadence"], "frame_rate")
                self.assertEqual(result["default_hardware_decode"], "auto")
                self.assertEqual(result["ffv1_chroma_default"], "native")
                self.assertEqual(result["ffv1_chroma_choices"], 2)
                self.assertFalse(result["vulkan_nnedi3_default_enabled"])
                self.assertTrue(result["drag_drop_enabled"], result["drag_drop_diagnostic"])
                self.assertTrue(result["backend_guide_has_rankings"])
                self.assertTrue(result["unified_backend_gpu_guide_available"])
                self.assertTrue(result["qtgmc_guide_explains_exact_parameters"])
                self.assertTrue(result["qtgmc_acceleration_preserves_quality_parameters"])
                self.assertTrue(result["help_excludes_job_specific_chat_analysis"])
                self.assertTrue(result["resolve_editor_preset_available"])
                self.assertTrue(result["speed_gpu_modes_available"])
                self.assertTrue(result["speed_gpu_guide_has_measured_tradeoffs"])
                self.assertTrue(result["bwdif_temporal_analysis_retained"])
                self.assertTrue(result["nvenc_maximum_quality_contract_documented"])
                self.assertTrue(result["mov_compatibility_copy_available"])
                self.assertEqual(result["dnxhr_selectable_depths"], (10,))
                self.assertEqual(result["denoiser_choices"], 6)
                self.assertTrue(result["denoise_default_enabled"])
                self.assertEqual(result["default_denoiser"], "vs_bm3d")
                self.assertEqual(result["default_denoise_strength"], "4")
                self.assertEqual(result["default_denoise_radius"], "3")
                self.assertTrue(result["denoise_guide_has_evidence_based_choices"])
                self.assertTrue(result["denoise_guide_requires_deinterlace_first"])
                self.assertTrue(result["denoise_guide_explains_temporal_radius"])
                self.assertTrue(result["cadence_guide_preserves_duration"])
                self.assertTrue(result["timeline_guide_explains_repeat_failure"])
                self.assertTrue(result["timeline_guide_explains_in_app_repair"])
                self.assertTrue(result["timeline_guide_explains_source_protection"])
                self.assertTrue(result["timeline_guide_explains_fast_precheck"])
                self.assertTrue(result["timeline_guide_explains_indexed_contract"])
                self.assertTrue(result["phase_aware_progress_available"])
                self.assertTrue(result["timeline_guide_explains_bwdif_repair_bypass"])
                self.assertTrue(result["source_health_banner_available"])
                self.assertTrue(result["automatic_recovery_control_available"])
                self.assertTrue(result["automatic_recovery_enabled"])
                self.assertTrue(result["repair_workflow_available"])
                self.assertEqual(result["repair_default_mode"], "automatic")
                self.assertTrue(result["dependency_installer_available"])
                self.assertFalse(result["dependency_ready"])
                self.assertIn("ffmpeg", result["dependency_issues"])
                self.assertIn("vapoursynth", result["dependency_issues"])
                self.assertFalse(result["interlace_runtime_verified"]["d3d12_custom"])
                self.assertTrue(result["interlace_runtime_verified"]["d3d12_bob"])
                self.assertEqual(result["ffmpeg_selection_source"], "process PATH entry 1")
                self.assertEqual(len(result["ffmpeg_discovery_diagnostics"]), 2)
                self.assertIn("evaluated 2 paired FFmpeg installations", app.status_var.get())
                self.assertEqual(root.minsize(), (980, 720))
                self.assertEqual(app._cadence_value(), "frame_rate")
                self.assertEqual(result["drag_drop_provider"], "TkDND")
                self.assertTrue(result["drag_drop_provider_version"])
                self.assertEqual(result["drag_drop_package_version"], "0.6.2")
                self.assertGreater(result["drag_drop_surface_count"], 0)
                self.assertEqual(result["drag_drop_registration_error_count"], 0)
                self.assertTrue(result["batch_tab_available"])
                self.assertEqual(result["batch_max_files"], 99)
                self.assertEqual(
                    result["batch_queue_columns"],
                    ("state", "analysis", "effective", "output", "progress"),
                )
                self.assertTrue(result["batch_delete_binding_available"])
                self.assertTrue(result["batch_drag_bindings_available"])
                self.assertTrue(result["batch_drop_route_test_passed"], result["batch_drop_route_test_detail"])
                self.assertTrue(result["batch_controls_share_settings"])
                self.assertTrue(result["drag_drop_route_test_passed"], result["drag_drop_route_test_detail"])
                self.assertIn("#1  VapourSynth QTGMC", BACKEND_GPU_GUIDE_TEXT)
                self.assertIn("BWDIF CUDA", BACKEND_GPU_GUIDE_TEXT)
                self.assertIn("analyze_refine=2", BACKEND_GPU_GUIDE_TEXT)
                self.assertIn("source_match(tr=2, TWICE_REFINED)", BACKEND_GPU_GUIDE_TEXT)
                self.assertIn("identical QTGMC parameters", BACKEND_GPU_GUIDE_TEXT)
                self.assertNotIn("WHAT TOOK SO LONG", BACKEND_GPU_GUIDE_TEXT)
                self.assertNotIn("8.50 times faster", BACKEND_GPU_GUIDE_TEXT)
                self.assertIn("60-minute input remains 60 minutes", CADENCE_GUIDE_TEXT)
                self.assertIn("twice the declared frame rate", CADENCE_GUIDE_TEXT)
                self.assertIn("blocked every time QTGMC is selected", SOURCE_TIMELINE_GUIDE_TEXT)
                self.assertIn(
                    "WHAT AUTOMATIC QTGMC RECOVERY AND THE “REPAIR SOURCE…” BUTTON DO",
                    SOURCE_TIMELINE_GUIDE_TEXT,
                )
                self.assertIn("Explicit BWDIF CPU/CUDA bypasses automatic repair", SOURCE_TIMELINE_GUIDE_TEXT)
                self.assertIn("FFV1 v3 intra lossless rescue", SOURCE_TIMELINE_GUIDE_TEXT)
                self.assertIn("cannot reconstruct pictures", SOURCE_TIMELINE_GUIDE_TEXT)
                normalized_timeline_guide = " ".join(SOURCE_TIMELINE_GUIDE_TEXT.split())
                self.assertIn("fast VSPipe graph-info check", normalized_timeline_guide)
                self.assertIn("managed full decoded fallback", normalized_timeline_guide)
                self.assertIn("immediate Cancel support", normalized_timeline_guide)
                self.assertIn("VapourSynth V-BM3D", DENOISE_GUIDE_TEXT)
                self.assertIn("VapourSynth DFTTest2", DENOISE_GUIDE_TEXT)
                self.assertIn("NO UNIVERSAL RANKING", DENOISE_GUIDE_TEXT)
                self.assertIn("always deinterlaces first", DENOISE_GUIDE_TEXT)
                self.assertIn("WHAT TEMPORAL RADIUS MEANS", DENOISE_GUIDE_TEXT)
                self.assertIn("(2 × N) + 1 frames", DENOISE_GUIDE_TEXT)
                self.assertIn("does not change frame rate, playing speed, or video duration", DENOISE_GUIDE_TEXT)
                self.assertIn("minimum five-frame window", DENOISE_GUIDE_TEXT)
                self.assertEqual(str(app.denoiser_combo.cget("state")), "readonly")
                app._handle_run_progress(
                    {
                        "phase": "preflight_full_progress",
                        "frame": "25",
                        "expected_frames": "100",
                        "percent": "25.0",
                        "speed": "2.0x",
                        "eta_seconds": "45",
                    }
                )
                self.assertEqual(app.progress_var.get(), 25.0)
                self.assertIn("no output encode has started", app.status_var.get())
                self.assertIn("ETA 00:45", app.run_detail_var.get())
                app._handle_run_progress({"phase": "encode_start", "expected_frames": "200"})
                self.assertEqual(app.progress_var.get(), 0.0)
                self.assertIn("starting the unique partial", app.status_var.get())
                app._handle_run_progress(
                    {
                        "phase": "encode_progress",
                        "frame": "100",
                        "expected_frames": "200",
                        "speed": "0.5x",
                    }
                )
                self.assertEqual(app.progress_var.get(), 50.0)
                self.assertIn("encoding", app.run_detail_var.get())
                app.auto_workflow = SimpleNamespace(stage="deinterlacing")
                app._handle_run_progress(
                    {
                        "phase": "encode_progress",
                        "frame": "120",
                        "expected_frames": "200",
                        "speed": "0.5x",
                    }
                )
                self.assertTrue(app.run_detail_var.get().startswith("Final processing 3/3 · encoding"))
                self.assertIn("Final processing 3/3:", app.status_var.get())
                self.assertNotIn("Automatic recovery 3/3", app.run_detail_var.get())
                app.auto_workflow = None
                app.denoise_enabled_var.set(True)
                root.update_idletasks()
                self.assertEqual(str(app.denoiser_combo.cget("state")), "readonly")
                fft_label = next(label for label, value in DENOISER_LABELS.items() if value == "ffmpeg_fftdnoiz")
                app.denoiser_var.set(fft_label)
                root.update_idletasks()
                self.assertEqual(app.denoise_radius_var.get(), "1")
                self.assertEqual(str(app.denoise_radius_spin.cget("state")), "disabled")

                app._apply_setup_layout(app.setup_wide_breakpoint - 1)
                self.assertEqual(app.setup_layout_mode, "stacked")
                self.assertEqual(int(app.deint_box.grid_info()["row"]), 1)
                self.assertEqual(int(app.output_box.grid_info()["row"]), 2)
                root.tk.call("tk", "scaling", 1.0)
                root.attributes("-alpha", 0.0)
                root.geometry("1900x1050+0+0")
                root.deiconify()
                root.update()
                root.update_idletasks()
                synthetic_wide_viewport = False
                if app.setup_layout_mode != "wide":
                    root.geometry(f"{app.setup_wide_breakpoint + 160}x1050+0+0")
                    root.update()
                    root.update_idletasks()
                if app.setup_layout_mode != "wide":
                    # Hosted Windows runners expose a narrow virtual desktop
                    # and clamp even explicitly oversized top-level windows.
                    # Exercise the responsive-layout decision with a simulated
                    # wide viewport when the display cannot provide one.
                    synthetic_wide_viewport = True
                    app._apply_setup_layout(app.setup_wide_breakpoint + 160)
                self.assertEqual(
                    app.setup_layout_mode,
                    "wide",
                    (
                        f"available={app.setup_parent.winfo_width()}, breakpoint={app.setup_wide_breakpoint}, "
                        f"deint_req={app.deint_box.winfo_reqwidth()}, output_req={app.output_box.winfo_reqwidth()}"
                    ),
                )
                self.assertEqual(int(app.deint_box.grid_info()["row"]), 1)
                self.assertEqual(int(app.output_box.grid_info()["row"]), 1)
                self.assertEqual(int(app.deint_box.grid_info()["column"]), 0)
                self.assertEqual(int(app.output_box.grid_info()["column"]), 1)
                self.assertGreaterEqual(app.summary_text.winfo_height(), 250)
                natural_summary_height = app.summary_text.winfo_height()
                simulated_tall_viewport = app.setup_scroll.inner.winfo_reqheight() + 300
                app.setup_scroll._start_content_measurement(
                    app.setup_parent.winfo_width(),
                    simulated_tall_viewport,
                )
                root.update()
                root.update_idletasks()
                if synthetic_wide_viewport:
                    self.assertGreaterEqual(app.summary_text.winfo_height(), natural_summary_height)
                else:
                    self.assertGreater(app.summary_text.winfo_height(), natural_summary_height + 200)
                root.geometry("980x720")
                root.update()
                root.update_idletasks()
                self.assertEqual(app.setup_layout_mode, "stacked")

                app._set_text(app.summary_text, "\n".join(f"old line {index}" for index in range(80)))
                app.summary_text.yview_moveto(1.0)
                app._set_text(app.summary_text, "new top line\n" + "\n".join("detail" for _ in range(80)))
                self.assertEqual(app.summary_text.get("1.0", "1.end"), "new top line")
                self.assertAlmostEqual(app.summary_text.yview()[0], 0.0)
                dropped = Path(directory) / "拖放 測試.mkv"
                dropped.write_bytes(b"video")
                root.tk.call("set", "::deinterlace_gui_drop_test", (str(dropped),))
                raw_drop = root.tk.eval("set ::deinterlace_gui_drop_test")
                action = app.file_drop_target._on_drop(SimpleNamespace(data=raw_drop))
                root.update()
                self.assertEqual(action, "copy")
                self.assertEqual(Path(app.input_var.get()), dropped)
                self.assertIn("deinterlaced", app.output_var.get())
                self.assertIn("drag-and-drop", app.status_var.get())
                self.assertEqual(str(app.repair_button.cget("state")), "normal")
                dropped_stat = dropped.stat()
                damaged_health = SourceHealthReport(
                    path=dropped,
                    source_size=dropped_stat.st_size,
                    source_mtime_ns=dropped_stat.st_mtime_ns,
                    status="repair_required",
                    reason=(
                        "Fast packet scan found a 28.020-second video timestamp hole at "
                        "137.877→165.897 seconds."
                    ),
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
                    warning_samples=("Element exceeds containing master element",),
                    ffprobe_returncode=0,
                )
                app.auto_repair_continue_var.set(False)
                with patch("deinterlace_studio.gui.probe_media", return_value=test_media(dropped)), patch(
                    "deinterlace_studio.gui.scan_source_health", return_value=damaged_health
                ) as health_scan, patch(
                    "deinterlace_studio.gui.scan_idet", return_value=test_idet_report("tff")
                ):
                    app._analyze("sampled")
                    deadline = time.monotonic() + 3.0
                    while app.busy_kind and time.monotonic() < deadline:
                        root.update()
                        time.sleep(0.01)
                    root.update()
                self.assertIsNone(app.busy_kind)
                health_scan.assert_called_once()
                self.assertIn("DAMAGE LIKELY", app.source_health_var.get())
                self.assertIn("28.020s", app.source_health_var.get())
                self.assertEqual(str(app.source_health_label.cget("style")), "HealthError.TLabel")
                self.assertEqual(str(app.repair_button.cget("text")), "Repair required…")
                self.assertIn("SOURCE HEALTH — DAMAGE LIKELY", app.summary_text.get("1.0", "end"))
                with patch("deinterlace_studio.gui.probe_media", return_value=test_media(dropped)), patch(
                    "deinterlace_studio.gui.scan_source_health"
                ) as repeated_health_scan, patch(
                    "deinterlace_studio.gui.scan_idet", return_value=test_idet_report("tff")
                ):
                    app._analyze("sampled")
                    deadline = time.monotonic() + 3.0
                    while app.busy_kind and time.monotonic() < deadline:
                        root.update()
                        time.sleep(0.01)
                    root.update()
                self.assertIsNone(app.busy_kind)
                repeated_health_scan.assert_not_called()
                self.assertEqual(app.analysis_progress_var.get(), "Sampled IDet complete · TFF")
                app._show_repair_dialog()
                root.update_idletasks()
                self.assertIsNotNone(app.repair_dialog)
                self.assertEqual(app.repair_mode_var.get(), "automatic")
                self.assertTrue(app.repair_output_var.get().endswith(".qtgmc-repair.mkv"))
                self.assertIn("safe QTGMC", app.repair_dialog.title())
                app._close_repair_dialog()
                repaired = Path(directory) / "拖放 測試.qtgmc-repair.mkv"
                repaired.write_bytes(b"validated repair")
                success = SimpleNamespace(
                    success=True,
                    output_path=repaired,
                    method="ffv1_rescue",
                    repeated_frames=25,
                    output_sha256="A" * 64,
                    report_path=repaired.with_name(repaired.name + ".Repair.json"),
                    message="ok",
                    source_diagnosis=None,
                    canceled=False,
                    log_path=None,
                    quarantine_path=None,
                )
                with patch("deinterlace_studio.gui.messagebox.showinfo"), patch.object(
                    app, "_analyze"
                ) as fresh_analysis:
                    app._handle_repair_done(success)
                    root.update()
                self.assertEqual(Path(app.input_var.get()), repaired)
                fresh_analysis.assert_called_once_with("sampled")
                failed = SimpleNamespace(
                    success=False,
                    output_path=None,
                    method="ffv1_rescue",
                    repeated_frames=0,
                    output_sha256=None,
                    report_path=Path(directory) / "failed.json",
                    message="validation rejected candidate",
                    source_diagnosis=None,
                    canceled=False,
                    log_path=Path(directory) / "failed.log",
                    quarantine_path=None,
                )
                with patch("deinterlace_studio.gui.messagebox.showerror"):
                    app._handle_repair_done(failed)
                self.assertEqual(Path(app.input_var.get()), repaired)
                app._handle_dropped_paths((dropped, dropped))
                self.assertEqual(Path(app.input_var.get()), repaired)
                self.assertEqual(len(app.batch_queue.records), 1)
                self.assertEqual(app.batch_queue.records[0].source_path, dropped.resolve())
                self.assertTrue(app._is_batch_tab())
                app._show_backend_gpu_guide()
                root.update_idletasks()
                self.assertIsNotNone(app.backend_gpu_dialog)
                self.assertIn("QTGMC parameters", app.backend_gpu_dialog.title())
                merged_text = app.backend_gpu_text.get("1.0", "end")
                self.assertIn("QUALITY RANKING", merged_text)
                self.assertIn("analyze_force_tr=3", merged_text)
                self.assertNotIn("WHAT TOOK SO LONG", merged_text)
                app._close_backend_gpu_guide()
                app._show_cadence_guide()
                root.update_idletasks()
                self.assertIn("60-minute input remains 60 minutes", app.cadence_guide_text.get("1.0", "end"))
                app.cadence_guide_dialog.destroy()
                app._show_timeline_guide()
                root.update_idletasks()
                timeline_text = app.timeline_guide_text.get("1.0", "end")
                self.assertIn("WILL RETRYING THE SAME FILE WITH QTGMC WORK?", timeline_text)
                self.assertIn(
                    "WHAT AUTOMATIC QTGMC RECOVERY AND THE “REPAIR SOURCE…” BUTTON DO",
                    timeline_text,
                )
                self.assertIn("AUTOMATIC QTGMC RECOVERY — DEFAULT ENABLED", timeline_text)
                self.assertIn("many times the source size", timeline_text)
                app.timeline_guide_dialog.destroy()
                cuda_engine = next(label for label, value in ENGINE_LABELS.items() if value == "ffmpeg_bwdif_cuda")
                app.engine_var.set(cuda_engine)
                root.update_idletasks()
                self.assertEqual(str(app.hw_decode_combo.cget("state")), "readonly")
                progressive_engine = next(label for label, value in ENGINE_LABELS.items() if value == "progressive")
                app.engine_var.set(progressive_engine)
                root.update_idletasks()
                self.assertEqual(app._cadence_value(), "frame_rate")
                self.assertEqual(str(app.cadence_combo.cget("state")), "disabled")
                self.assertEqual(str(app.field_combo.cget("state")), "disabled")
                app._dependency_doctor()
                root.update_idletasks()
                doctor_text = app.dependency_doctor_text.get("1.0", "end")
                self.assertIn("Managed runtime folder", doctor_text)
                self.assertIn("FFmpeg 9 interlace capability audit", doctor_text)
                self.assertIn("FFmpeg discovery evidence", doctor_text)
                self.assertIn("process PATH entry 1", doctor_text)
                self.assertIn("excluded from output backends", doctor_text)
                self.assertEqual(str(app.dependency_cancel_button.cget("state")), "disabled")
                app._close_dependency_doctor()
                with patch("deinterlace_studio.gui.messagebox.askyesno", return_value=False) as prompt:
                    app._offer_app_local_install()
                prompt_text = prompt.call_args.args[1]
                self.assertIn("dependency scan completed", prompt_text)
                self.assertIn("Selected FFmpeg", prompt_text)
                self.assertIn("process PATH entry 1", prompt_text)
                self.assertIn("Dependency doctor", prompt_text)

                ntsc_media = SimpleNamespace(
                    video=SimpleNamespace(
                        r_frame_rate=Fraction(30000, 1001),
                        avg_frame_rate=Fraction(30000, 1001),
                    )
                )
                ntsc_labels = cadence_labels_for_media(ntsc_media)
                field_label = next(label for label, value in ntsc_labels.items() if value == "field_rate")
                frame_label = next(label for label, value in ntsc_labels.items() if value == "frame_rate")
                self.assertIn("59.94 fields/s", field_label)
                self.assertIn("59.94p", field_label)
                self.assertIn("29.97 interlaced frames/s", frame_label)
                self.assertNotIn("25i", " ".join(ntsc_labels))
            finally:
                root.destroy()

    def test_verified_git_pair_is_ready_and_does_not_offer_install(self) -> None:
        try:
            root = Tk()
        except TclError as exc:
            self.skipTest(f"Tk unavailable in this test runtime: {exc}")
        libraries = {
            "libavutil": (61, 5, 100),
            "libavcodec": (63, 7, 100),
            "libavformat": (63, 5, 101),
            "libavfilter": (12, 3, 101),
        }
        with tempfile.TemporaryDirectory() as directory:
            try:
                root.withdraw()
                caps = CapabilityReport(
                    ffmpeg_path=Path("C:/Program Files (x86)/FFMPEG/ffmpeg.exe"),
                    ffprobe_path=Path("C:/Program Files (x86)/FFMPEG/ffprobe.exe"),
                    ffmpeg_version="ffmpeg version 2026-08-06-git-95c43d7df7-full_build-www.gyan.dev",
                    ffmpeg_configuration="--enable-gpl",
                    filters=frozenset({"idet", "bwdif", "fftdnoiz", "atadenoise"}),
                    encoders=frozenset({"libx265", "libaom-av1", "libsvtav1", "ffv1", "prores_ks", "dnxhd"}),
                    encoder_pixel_formats={},
                    hwaccels=frozenset(),
                    vspipe_path=Path("C:/tools/vspipe.exe"),
                    vapoursynth_version="R78",
                    qtgmc_ready=True,
                    qtgmc_diagnostic="ready",
                    qtgmc_install_command=None,
                    ffprobe_version="ffprobe version 2026-08-06-git-95c43d7df7-full_build-www.gyan.dev",
                    ffmpeg_version_kind="verified_git",
                    ffmpeg_version_diagnostic=(
                        "verified Git snapshot revision 95c43d7df7; FFmpeg/FFprobe revisions and required "
                        "library versions match, and the FFmpeg 9.0 library floor is met"
                    ),
                    ffmpeg_git_revision="95c43d7df7",
                    ffprobe_git_revision="95c43d7df7",
                    ffmpeg_library_versions=libraries,
                    ffprobe_library_versions=libraries,
                    ffmpeg_selection_source="process PATH entry 1",
                    ffmpeg_discovery_diagnostics=(
                        "SELECTED [process PATH entry 1] C:\\Program Files (x86)\\FFMPEG — verified Git snapshot revision 95c43d7df7",
                    ),
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
                )
                app = DeinterlaceStudioApp(
                    root,
                    initial_capabilities=caps,
                    settings_path=Path(directory) / "settings.json",
                )
                result = app.self_test()
                self.assertTrue(result["dependency_ready"])
                self.assertEqual(result["ffmpeg_version_kind"], "verified_git")
                self.assertEqual(result["ffmpeg_git_revision"], "95c43d7df7")
                self.assertEqual(result["ffmpeg_library_versions"]["libavfilter"], (12, 3, 101))
                app._dependency_doctor()
                root.update_idletasks()
                doctor_text = app.dependency_doctor_text.get("1.0", "end")
                self.assertIn("Version classification: verified_git", doctor_text)
                self.assertIn("FFmpeg 9.0 library floor is met", doctor_text)
                app._close_dependency_doctor()
                with patch("deinterlace_studio.gui.messagebox.askyesno") as prompt:
                    app._offer_app_local_install()
                prompt.assert_not_called()
            finally:
                root.destroy()


if __name__ == "__main__":
    unittest.main()
