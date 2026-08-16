from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from tkinter import TclError, Tk
from unittest.mock import Mock, patch

from PIL import Image

from video_denoise_studio.gui import ComparisonViewer, VideoDenoiseStudioApp
from video_denoise_studio.models import PreviewFrames
from video_denoise_studio.timeline import frame_from_timeline_position

from video_denoise_tests.helpers import fake_capabilities, fake_media


class GuiContractTests(unittest.TestCase):
    def setUp(self) -> None:
        try:
            self.root = Tk()
        except TclError as exc:
            self.skipTest(f"Tk is unavailable: {exc}")
        self.root.withdraw()

    def tearDown(self) -> None:
        if hasattr(self, "root"):
            try:
                self.root.update_idletasks()
                self.root.update()
            except TclError:
                pass
            try:
                self.root.destroy()
            except TclError:
                pass

    def _dispose_app(self, app: VideoDenoiseStudioApp) -> None:
        app.closing = True
        if app.preview_after_id is not None:
            self.root.after_cancel(app.preview_after_id)
            app.preview_after_id = None
        if app.drop_after_id is not None:
            self.root.after_cancel(app.drop_after_id)
            app.drop_after_id = None
        if app.drop_target:
            app.drop_target.close()

    def test_two_top_level_tabs_timeline_and_automatic_preflight_contract(self) -> None:
        app = VideoDenoiseStudioApp(self.root, initial_capabilities=fake_capabilities())
        self.root.update_idletasks()
        result = app.self_test()
        self.assertEqual(result["top_level_tabs"], ("Single file + frame preview", "Batch processing"))
        self.assertTrue(result["frame_preview_toggle_available"])
        self.assertTrue(result["manual_preview_frame_count_removed"])
        self.assertTrue(result["full_source_timeline_available"])
        self.assertTrue(result["hold_to_original_available"])
        self.assertTrue(result["pan_available"])
        self.assertTrue(result["wheel_zoom_available"])
        self.assertTrue(result["check_plan_button_removed"])
        self.assertTrue(result["automatic_preflight_on_process"])
        self.assertGreaterEqual(result["preflight_log_preferred_lines"], 10)
        self.assertTrue(result["preserved_track_controls_one_row"])
        self.assertTrue(result["container_selector_available"])
        self.assertEqual(result["batch_max_files"], 99)
        self.assertTrue(result["shared_settings_identity"])
        self._dispose_app(app)

    def test_viewer_hold_release_pan_zoom_and_fit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original_path = root / "original.png"
            processed_path = root / "processed.png"
            Image.new("RGB", (40, 30), (255, 0, 0)).save(original_path)
            Image.new("RGB", (40, 30), (0, 0, 255)).save(processed_path)
            viewer = ComparisonViewer(self.root)
            viewer.canvas.configure(width=200, height=120)
            viewer.pack(fill="both", expand=True)
            frames = PreviewFrames(
                token="test",
                directory=root,
                original_frame=original_path,
                processed_frame=processed_path,
                target_frame=10,
                total_frames=300,
                temporal_radius=1,
                leading_context=1,
                trailing_context=1,
                fps=24.0,
                selected_backend="test",
                status="ready",
            )
            viewer.set_frames(frames, show_processed=True)
            self.root.update_idletasks()
            viewer.fit()
            self.assertEqual(viewer._active_image()[0], "DENOISED")
            viewer._press_left(SimpleNamespace(x=100, y=60))
            self.assertEqual(viewer._active_image()[0], "ORIGINAL")
            viewer._mouse_wheel(SimpleNamespace(x=100, y=60, delta=120))
            zoomed = viewer.scale
            viewer._drag_left(SimpleNamespace(x=90, y=50))
            self.assertFalse(viewer.fit_mode)
            viewer._release_left(SimpleNamespace(x=90, y=50))
            self.assertEqual(viewer._active_image()[0], "DENOISED")
            self.assertGreater(zoomed, viewer.fit_scale)
            viewer.scale = 8.0
            viewer.fit_mode = False
            viewer.origin_x = -1500
            viewer.origin_y = -1000
            viewer._render(force=True)
            self.assertLessEqual(viewer._last_render_size[0], viewer.canvas.winfo_width() + 18)
            self.assertLessEqual(viewer._last_render_size[1], viewer.canvas.winfo_height() + 18)
            viewer.fit()
            self.assertTrue(viewer.fit_mode)
            self.assertAlmostEqual(viewer.scale, viewer.fit_scale)
            viewer.destroy()

    def test_same_frame_replacement_preserves_viewport_and_new_frame_resets_fit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for index, color in enumerate(((255, 0, 0), (0, 255, 0), (0, 0, 255))):
                path = root / f"frame-{index}.png"
                Image.new("RGB", (320, 180), color).save(path)
                paths.append(path)
            viewer = ComparisonViewer(self.root)
            viewer.canvas.configure(width=640, height=360)
            viewer.pack(fill="both", expand=True)

            def frames(token: str, target: int, image: Path) -> PreviewFrames:
                return PreviewFrames(
                    token=token,
                    directory=root,
                    original_frame=image,
                    processed_frame=image,
                    target_frame=target,
                    total_frames=300,
                    temporal_radius=4,
                    leading_context=4,
                    trailing_context=4,
                    fps=24.0,
                    selected_backend="test",
                    status="ready",
                    strength=6,
                    window_frames=9,
                )

            viewer.set_frames(frames("initial", 10, paths[0]), show_processed=True)
            self.root.update_idletasks()
            self.root.update()
            viewer.fit()
            viewer._mouse_wheel(SimpleNamespace(x=430, y=170, delta=120))
            viewer._press_left(SimpleNamespace(x=430, y=170))
            viewer._drag_left(SimpleNamespace(x=390, y=145))
            viewer._release_left(SimpleNamespace(x=390, y=145))
            expected = (viewer.scale, viewer.origin_x, viewer.origin_y)
            self.assertFalse(viewer.fit_mode)

            # A setting render adopts the viewport as it exists when the result
            # arrives, rather than a stale pre-render snapshot.
            viewer.set_frames(frames("settings-changed", 10, paths[1]), show_processed=True)
            self.root.update_idletasks()
            self.root.update()
            self.assertFalse(viewer.fit_mode)
            self.assertEqual((viewer.scale, viewer.origin_x, viewer.origin_y), expected)

            viewer.set_frames(frames("new-frame", 11, paths[2]), show_processed=True)
            self.root.update_idletasks()
            self.root.update()
            self.assertTrue(viewer.fit_mode)
            self.assertAlmostEqual(viewer.scale, viewer.fit_scale)
            viewer.destroy()

    def test_timeline_first_previous_next_last_and_keyboard_step(self) -> None:
        app = VideoDenoiseStudioApp(self.root, initial_capabilities=fake_capabilities())
        app.media = fake_media(Path("source.mkv"))
        app._refresh_timeline_range()
        self.assertEqual(app.timeline_total_frames, 300)
        self.assertTrue(app.timeline_scale.bind("<Left>"))
        self.assertTrue(app.timeline_scale.bind("<Right>"))
        app._step_timeline("last")
        self.assertEqual(app._current_timeline_frame(), 299)
        generation_after_last = app.preview_generation
        app._step_timeline(-1)
        self.assertEqual(app._current_timeline_frame(), 298)
        self.assertGreater(app.preview_generation, generation_after_last)
        app._step_timeline(1)
        self.assertEqual(app._current_timeline_frame(), 299)
        app._step_timeline("first")
        self.assertEqual(app._current_timeline_frame(), 0)
        self.assertTrue(app.timeline_label_var.get().startswith("Frame 1 / 300"))
        self._dispose_app(app)

    def test_timeline_pointer_click_is_absolute_and_source_only_render_is_immediate(self) -> None:
        app = VideoDenoiseStudioApp(self.root, initial_capabilities=fake_capabilities())
        app.media = fake_media(Path("source.mkv"))
        app._refresh_timeline_range()
        self.root.update_idletasks()
        width = max(2, app.timeline_scale.winfo_width())
        pointer_x = round((width - 1) * 0.75)
        expected = frame_from_timeline_position(pointer_x, width, 300)
        app.frame_preview_var.set(False)
        app._schedule_frame_render = Mock()
        event = SimpleNamespace(x=pointer_x)
        self.assertEqual(app._timeline_pressed(event), "break")
        self.assertEqual(app._current_timeline_frame(), expected)
        app._schedule_frame_render.assert_called_once_with(immediate=True)
        self.assertEqual(app._timeline_released(event), "break")
        self.assertEqual(app._schedule_frame_render.call_count, 1)

        app.frame_preview_var.set(True)
        app._schedule_frame_render.reset_mock()
        self.assertEqual(app._timeline_pressed(SimpleNamespace(x=0)), "break")
        app._schedule_frame_render.assert_called_once_with()
        self.assertEqual(app._timeline_released(SimpleNamespace(x=0)), "break")
        app._schedule_frame_render.assert_called_with(immediate=True)
        self._dispose_app(app)

    def test_self_test_cleanup_can_close_without_writing_preferences(self) -> None:
        app = VideoDenoiseStudioApp(self.root, initial_capabilities=fake_capabilities())
        with patch("video_denoise_studio.gui.save_settings") as save_settings:
            app.close(save_preferences=False)
        save_settings.assert_not_called()
        self.assertTrue(app.closing)


if __name__ == "__main__":
    unittest.main()
