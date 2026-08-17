from __future__ import annotations

import shutil
import threading
import math
from dataclasses import replace
from pathlib import Path
from tkinter import (
    BooleanVar,
    DoubleVar,
    IntVar,
    StringVar,
    Toplevel,
    filedialog,
    messagebox,
    scrolledtext,
    ttk,
)
from PIL import Image, ImageTk

from deinterlace_studio.capabilities import inspect_capabilities
from deinterlace_studio.denoise import (
    DENOISER_SPECS,
    MAX_DENOISE_STRENGTH,
    MAX_TEMPORAL_RADIUS,
    MIN_DENOISE_STRENGTH,
    MIN_TEMPORAL_RADIUS,
)
from deinterlace_studio.models import CapabilityReport, MediaProbe
from deinterlace_studio.presets import selectable_bit_depths
from deinterlace_studio.windows_drop import FileDropUnavailable, WindowsFileDropTarget

from . import __version__
from .batch import BatchQueue, BatchRunner
from .denoiser_policy import (
    denoiser_backend_status,
    denoiser_control_policy,
    denoiser_ranking,
    denoiser_rankings_guide,
    normalize_temporal_radius,
)
from .models import BatchRecord, BatchRunOptions, DenoiseSettings, PreviewFrames, PreviewRequest
from .output_policy import (
    CONTAINER_ID_LABELS,
    CONTAINER_LABELS,
    container_help_text,
    container_labels_for_family,
    encoder_control_policy,
    resolve_container,
    select_output_profile,
)
from .planner import build_plan, default_output_path, source_field_order, source_is_interlaced, unique_output_path
from .preview import PreviewCancelled, PreviewRenderer
from .probe import ProbeCancelled, probe_media_cancelable
from .processor import DenoiseProcessor
from .settings import load_settings, save_settings
from .timeline import (
    frame_from_timeline_position,
    source_fps,
    source_frame_count,
    timeline_render_delay_ms,
)


DENOISER_LABELS = {spec.label: spec.identifier for spec in DENOISER_SPECS}
DENOISER_ID_LABELS = {value: key for key, value in DENOISER_LABELS.items()}
FAMILY_LABELS = {
    "FFV1 16-bit lossless master (recommended)": "ffv1",
    "HEVC quality delivery": "hevc",
    "AV1 quality delivery": "av1",
    "Apple ProRes 4444 XQ": "prores",
    "Avid DNxHR 444": "dnxhr",
}
FAMILY_ID_LABELS = {value: key for key, value in FAMILY_LABELS.items()}
FFV1_CHROMA_LABELS = {
    "Preserve native chroma": "native",
    "Convert to 4:4:4 mastering": "444",
}
FFV1_CHROMA_ID_LABELS = {value: key for key, value in FFV1_CHROMA_LABELS.items()}
AV1_LABELS = {"libaom (quality-first)": "libaom", "SVT-AV1": "svt"}
AV1_ID_LABELS = {value: key for key, value in AV1_LABELS.items()}
VIDEO_FILE_TYPES = [
    ("Video files", "*.mkv *.mov *.mp4 *.avi *.mxf *.m2ts *.mts *.ts *.mpg *.mpeg *.vob *.webm *.wmv *.m4v"),
    ("All files", "*.*"),
]


class ComparisonViewer(ttk.Frame):
    """Source-raster viewer with Topaz-style hold-to-original, pan, and zoom."""

    def __init__(self, parent) -> None:
        super().__init__(parent)
        self.frames: PreviewFrames | None = None
        self.show_processed = False
        self.hold_original = False
        self.fit_mode = True
        self.scale = 1.0
        self.fit_scale = 1.0
        self.origin_x = 0.0
        self.origin_y = 0.0
        self._drag_last: tuple[int, int] | None = None
        self._original_image: Image.Image | None = None
        self._processed_image: Image.Image | None = None
        self._photo: ImageTk.PhotoImage | None = None
        self._render_key: tuple[object, ...] | None = None
        self._last_render_size = (0, 0)
        self._fit_after_id: str | None = None
        self.zoom_var = StringVar(value="Fit")

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.canvas = __import__("tkinter").Canvas(
            self,
            width=960,
            height=540,
            background="#0b0e12",
            highlightthickness=1,
            highlightbackground="#303642",
            cursor="arrow",
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.image_item = self.canvas.create_image(0, 0, anchor="nw")
        self.placeholder_item = self.canvas.create_text(
            480,
            270,
            text="Choose one video to load the unprocessed frame.",
            fill="#c5ccd8",
            font=("Segoe UI", 13),
        )
        self.label_bg = self.canvas.create_rectangle(12, 12, 126, 40, fill="#11151b", outline="", state="hidden")
        self.label_item = self.canvas.create_text(69, 26, text="ORIGINAL", fill="white", font=("Segoe UI", 9, "bold"), state="hidden")
        self.canvas.bind("<ButtonPress-1>", self._press_left)
        self.canvas.bind("<B1-Motion>", self._drag_left)
        self.canvas.bind("<ButtonRelease-1>", self._release_left)
        self.canvas.bind("<MouseWheel>", self._mouse_wheel)
        self.canvas.bind("<Configure>", self._canvas_configured)

        controls = ttk.Frame(self)
        controls.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        controls.columnconfigure(0, weight=1)
        ttk.Label(
            controls,
            text="Hold left: Original · drag while held: Pan · mouse wheel: Zoom",
            style="Muted.TLabel",
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(controls, textvariable=self.zoom_var, width=8, anchor="e").grid(row=0, column=1, padx=(8, 4))
        ttk.Button(controls, text="Fit", width=7, command=self.fit).grid(row=0, column=2)

    def clear(self, message: str = "Choose one video to load the unprocessed frame.") -> None:
        self._cancel_scheduled_fit()
        self.frames = None
        for image in (self._original_image, self._processed_image):
            if image is not None:
                image.close()
        self._original_image = self._processed_image = None
        self._photo = None
        self._render_key = None
        self.canvas.itemconfigure(self.image_item, image="")
        self.canvas.itemconfigure(self.placeholder_item, text=message, state="normal")
        self.canvas.itemconfigure(self.label_bg, state="hidden")
        self.canvas.itemconfigure(self.label_item, state="hidden")
        self.fit_mode = True
        self.scale = 1.0
        self.fit_scale = 1.0
        self.origin_x = 0.0
        self.origin_y = 0.0
        self.zoom_var.set("Fit")

    def set_frames(
        self,
        frames: PreviewFrames,
        *,
        show_processed: bool,
        reset_view: bool | None = None,
    ) -> None:
        old_target = self.frames.target_frame if self.frames else None
        old_size = self._original_image.size if self._original_image else None
        old_view = (self.fit_mode, self.scale, self.origin_x, self.origin_y)
        new_original: Image.Image | None = None
        new_processed: Image.Image | None = None
        try:
            with Image.open(frames.original_frame) as image:
                new_original = image.convert("RGB").copy()
            if frames.processed_frame:
                with Image.open(frames.processed_frame) as image:
                    new_processed = image.convert("RGB").copy()
            if new_processed and new_processed.size != new_original.size:
                raise ValueError(
                    f"original raster {new_original.size} and denoised raster {new_processed.size} differ"
                )
        except Exception as exc:
            for image in (new_original, new_processed):
                if image is not None:
                    image.close()
            self.clear(f"Could not display preview frame: {exc}")
            return

        assert new_original is not None
        same_frame_and_raster = old_target == frames.target_frame and old_size == new_original.size
        preserve_view = same_frame_and_raster if reset_view is None else (not reset_view and old_size == new_original.size)
        self._cancel_scheduled_fit()
        for image in (self._original_image, self._processed_image):
            if image is not None:
                image.close()
        self.frames = frames
        self._original_image = new_original
        self._processed_image = new_processed
        self._photo = None
        self._render_key = None
        self.show_processed = bool(show_processed and self._processed_image)
        self.hold_original = False
        self.canvas.itemconfigure(self.placeholder_item, state="hidden")
        if preserve_view:
            was_fit, self.scale, self.origin_x, self.origin_y = old_view
            self.fit_mode = was_fit
            if was_fit:
                self._schedule_fit()
            else:
                self._recompute_fit()
                self._render(force=True)
        else:
            self.fit_mode = True
            self._schedule_fit()

    def _cancel_scheduled_fit(self) -> None:
        if self._fit_after_id is None:
            return
        try:
            self.after_cancel(self._fit_after_id)
        except Exception:
            pass
        self._fit_after_id = None

    def _schedule_fit(self) -> None:
        self._cancel_scheduled_fit()
        self._fit_after_id = self.after_idle(self._run_scheduled_fit)

    def _run_scheduled_fit(self) -> None:
        self._fit_after_id = None
        self.fit()

    def set_processed_visible(self, visible: bool) -> None:
        self.show_processed = bool(visible and self._processed_image)
        self.hold_original = False
        self._render(force=True)

    def fit(self) -> None:
        self._cancel_scheduled_fit()
        if not self._original_image:
            return
        self.fit_mode = True
        self._recompute_fit()
        self.scale = self.fit_scale
        width, height = self._scaled_size()
        self.origin_x = (max(1, self.canvas.winfo_width()) - width) / 2
        self.origin_y = (max(1, self.canvas.winfo_height()) - height) / 2
        self._render(force=True)

    def _recompute_fit(self) -> None:
        assert self._original_image is not None
        canvas_width = max(1, self.canvas.winfo_width())
        canvas_height = max(1, self.canvas.winfo_height())
        source_width, source_height = self._original_image.size
        self.fit_scale = min(canvas_width / source_width, canvas_height / source_height)

    def _scaled_size(self) -> tuple[int, int]:
        assert self._original_image is not None
        return (
            max(1, round(self._original_image.width * self.scale)),
            max(1, round(self._original_image.height * self.scale)),
        )

    def _active_image(self) -> tuple[str, Image.Image] | None:
        if self._original_image is None:
            return None
        if self.show_processed and self._processed_image is not None and not self.hold_original:
            return "DENOISED", self._processed_image
        return "ORIGINAL", self._original_image

    def _render(self, *, force: bool = False) -> None:
        active = self._active_image()
        if active is None:
            return
        label, image = active
        width, height = self._scaled_size()
        self._clamp_origin(width, height)
        canvas_width = max(1, self.canvas.winfo_width())
        canvas_height = max(1, self.canvas.winfo_height())
        visible_left = max(0.0, -self.origin_x)
        visible_top = max(0.0, -self.origin_y)
        visible_right = min(float(width), canvas_width - self.origin_x)
        visible_bottom = min(float(height), canvas_height - self.origin_y)
        source_left = max(0, min(image.width - 1, math.floor(visible_left / self.scale)))
        source_top = max(0, min(image.height - 1, math.floor(visible_top / self.scale)))
        source_right = max(source_left + 1, min(image.width, math.ceil(visible_right / self.scale)))
        source_bottom = max(source_top + 1, min(image.height, math.ceil(visible_bottom / self.scale)))
        crop_box = (source_left, source_top, source_right, source_bottom)
        render_width = max(1, round((source_right - source_left) * self.scale))
        render_height = max(1, round((source_bottom - source_top) * self.scale))
        key = (label, crop_box, render_width, render_height)
        if force or key != self._render_key:
            cropped = image.crop(crop_box)
            try:
                resized = cropped.resize((render_width, render_height), Image.Resampling.LANCZOS)
                try:
                    self._photo = ImageTk.PhotoImage(resized)
                finally:
                    resized.close()
            finally:
                cropped.close()
            self.canvas.itemconfigure(self.image_item, image=self._photo)
            self._render_key = key
            self._last_render_size = (render_width, render_height)
        display_x = self.origin_x + source_left * self.scale
        display_y = self.origin_y + source_top * self.scale
        self.canvas.coords(self.image_item, round(display_x), round(display_y))
        self.canvas.itemconfigure(self.placeholder_item, state="hidden")
        self.canvas.itemconfigure(self.label_bg, state="normal")
        self.canvas.itemconfigure(self.label_item, text=label, state="normal")
        self.canvas.tag_raise(self.label_bg)
        self.canvas.tag_raise(self.label_item)
        relative = 100.0 * self.scale / max(self.fit_scale, 1e-9)
        self.zoom_var.set("Fit" if self.fit_mode else f"{relative:.0f}%")

    def _clamp_origin(self, width: int, height: int) -> None:
        canvas_width = max(1, self.canvas.winfo_width())
        canvas_height = max(1, self.canvas.winfo_height())
        if width <= canvas_width:
            self.origin_x = (canvas_width - width) / 2
        else:
            self.origin_x = min(0.0, max(canvas_width - width, self.origin_x))
        if height <= canvas_height:
            self.origin_y = (canvas_height - height) / 2
        else:
            self.origin_y = min(0.0, max(canvas_height - height, self.origin_y))

    def _press_left(self, event) -> None:
        if not self._original_image:
            return
        self.hold_original = True
        self._drag_last = (event.x, event.y)
        self.canvas.configure(cursor="fleur")
        self._render(force=True)

    def _drag_left(self, event) -> None:
        if not self._original_image or self._drag_last is None:
            return
        last_x, last_y = self._drag_last
        self.origin_x += event.x - last_x
        self.origin_y += event.y - last_y
        self._drag_last = (event.x, event.y)
        self.fit_mode = False
        width, height = self._scaled_size()
        self._clamp_origin(width, height)
        self._render()

    def _release_left(self, _event) -> None:
        if not self._original_image:
            return
        self._drag_last = None
        self.hold_original = False
        self.canvas.configure(cursor="arrow")
        self._render(force=True)

    def _mouse_wheel(self, event) -> None:
        if not self._original_image or not event.delta:
            return
        factor = 1.15 if event.delta > 0 else 1 / 1.15
        old_scale = self.scale
        new_scale = max(0.03, min(8.0, old_scale * factor))
        if abs(new_scale - old_scale) < 1e-9:
            return
        source_x = (event.x - self.origin_x) / old_scale
        source_y = (event.y - self.origin_y) / old_scale
        self.scale = new_scale
        self.origin_x = event.x - source_x * new_scale
        self.origin_y = event.y - source_y * new_scale
        self.fit_mode = False
        self._render(force=True)

    def _canvas_configured(self, _event=None) -> None:
        if self._original_image:
            if self.fit_mode:
                self.fit()
            else:
                self._render()
        else:
            self.canvas.coords(
                self.placeholder_item,
                max(1, self.canvas.winfo_width()) // 2,
                max(1, self.canvas.winfo_height()) // 2,
            )


class VideoDenoiseStudioApp:
    def __init__(self, root, *, initial_capabilities: CapabilityReport | None = None) -> None:
        self.root = root
        self.saved = load_settings()
        self.capabilities = initial_capabilities
        self.media: MediaProbe | None = None
        self.source_cancel_event = threading.Event()
        self.source_generation = 0
        self.preview_generation = 0
        self.preview_after_id: str | None = None
        self.preview_active_renderer: PreviewRenderer | None = None
        self.preview_owner: PreviewRenderer | None = None
        self.current_preview: PreviewFrames | None = None
        self.processor: DenoiseProcessor | None = None
        self.batch_queue = BatchQueue()
        self.batch_runner: BatchRunner | None = None
        self.busy_single = False
        self.busy_batch = False
        self.closing = False
        self._output_refreshing = False
        self._denoiser_refreshing = False
        self.drop_target: WindowsFileDropTarget | None = None
        self.drop_after_id: str | None = None

        self.root.title(f"Video Denoise Studio {__version__} — temporal denoising and frame comparison")
        self.root.geometry(self.saved["window_geometry"])
        self.root.minsize(1080, 760)
        self._configure_style()
        self._create_variables()
        self._build_ui()
        self._bind_setting_traces()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.drop_after_id = self.root.after_idle(self._install_drop)
        if self.capabilities is None:
            self._start_capability_scan()
        else:
            self._capability_scan_complete(self.capabilities)

    def _configure_style(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("vista")
        except Exception:
            pass
        style.configure("Title.TLabel", font=("Segoe UI", 14, "bold"))
        style.configure("Muted.TLabel", foreground="#5f6978")
        style.configure("Good.TLabel", foreground="#176b35")
        style.configure("Warn.TLabel", foreground="#9b5a00")

    def _create_variables(self) -> None:
        self.input_var = StringVar(value="")
        self.output_var = StringVar(value="")
        self.denoiser_var = StringVar(value=DENOISER_ID_LABELS.get(self.saved["denoiser"], next(iter(DENOISER_LABELS))))
        self.strength_var = IntVar(value=self.saved["denoise_strength"])
        self.radius_var = IntVar(value=self.saved["denoise_temporal_radius"])
        self.radius_label_var = StringVar(value="Temporal radius 1–6")
        self.window_var = StringVar(value="")
        initial_ranking = denoiser_ranking(DENOISER_LABELS.get(self.denoiser_var.get(), "vs_bm3d"))
        self.denoiser_rank_var = StringVar(
            value=f"Selection guide: Quality {initial_ranking.quality_score}/6 · Speed {initial_ranking.speed_score}/6"
        )
        self.denoiser_backend_var = StringVar(value="Denoiser acceleration: scanning installed CPU/GPU backends…")
        self.frame_preview_var = BooleanVar(value=self.saved["frame_preview_enabled"])
        self.timeline_frame_var = DoubleVar(value=0.0)
        self.timeline_label_var = StringVar(value="Frame 0 / 0 · 00:00:00.000")
        self.timeline_total_frames = 0
        self.timeline_guard = False
        self.timeline_dragging = False
        self.timeline_pointer_render_frame: int | None = None
        self.preview_status_var = StringVar(value="Choose a source to begin.")
        self.preview_progress_var = DoubleVar(value=0.0)
        self.source_summary_var = StringVar(value="No source loaded.")
        self.single_status_var = StringVar(value="Waiting for a source and tool scan.")
        self.single_progress_var = DoubleVar(value=0.0)
        self.family_var = StringVar(value=FAMILY_ID_LABELS.get(self.saved["family"], next(iter(FAMILY_LABELS))))
        self.container_var = StringVar(value=CONTAINER_ID_LABELS.get(self.saved["container"], next(iter(CONTAINER_LABELS))))
        self.bit_depth_var = StringVar(value=str(self.saved["bit_depth"]))
        self.ffv1_chroma_var = StringVar(value=FFV1_CHROMA_ID_LABELS.get(self.saved["ffv1_chroma_mode"], next(iter(FFV1_CHROMA_LABELS))))
        self.hardware_encode_var = BooleanVar(value=self.saved["hardware_encode"])
        self.av1_var = StringVar(value=AV1_ID_LABELS.get(self.saved["av1_software_encoder"], next(iter(AV1_LABELS))))
        self.quality_var = IntVar(value=self.saved["quality"])
        self.quality_label_var = StringVar(value="Encoder quality (lower = better)")
        self.encoder_summary_var = StringVar(value="Encoder profile will resolve after tool scan.")
        self.tune_grain_var = BooleanVar(value=self.saved["tune_grain"])
        self.copy_audio_var = BooleanVar(value=self.saved["copy_audio"])
        self.copy_subtitles_var = BooleanVar(value=self.saved["copy_subtitles"])
        self.copy_attachments_var = BooleanVar(value=self.saved["copy_attachments"])
        self.copy_data_var = BooleanVar(value=self.saved["copy_data"])
        self.copy_chapters_var = BooleanVar(value=self.saved["copy_chapters"])
        self.copy_metadata_var = BooleanVar(value=self.saved["copy_metadata"])
        self.tools_status_var = StringVar(value="Scanning FFmpeg and denoisers…")
        self.batch_output_dir_var = StringVar(value=self.saved["batch_output_dir"])
        self.batch_subfolders_var = BooleanVar(value=self.saved["batch_include_subfolders"])
        self.batch_continue_var = BooleanVar(value=self.saved["batch_continue_after_error"])
        self.batch_status_var = StringVar(value="Queue is empty.")
        self._update_window_label()

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(1, weight=1)
        header = ttk.Frame(outer)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        header.columnconfigure(1, weight=1)
        ttk.Label(header, text="Video Denoise Studio", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(header, textvariable=self.tools_status_var, style="Muted.TLabel", anchor="e").grid(row=0, column=1, sticky="ew", padx=12)
        ttk.Button(header, text="Tools…", command=self._show_tools_dialog).grid(row=0, column=2)
        ttk.Button(header, text="Denoiser guide…", command=self._show_denoiser_guide).grid(row=0, column=3, padx=(6, 0))
        ttk.Button(header, text="Codec + container guide…", command=self._show_codec_guide).grid(row=0, column=4, padx=(6, 0))

        self.notebook = ttk.Notebook(outer)
        self.notebook.grid(row=1, column=0, sticky="nsew")
        self.single_tab = ttk.Frame(self.notebook, padding=8)
        self.batch_tab = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(self.single_tab, text="Single file + frame preview")
        self.notebook.add(self.batch_tab, text="Batch processing")
        self._build_single_tab()
        self._build_batch_tab()

    def _build_single_tab(self) -> None:
        self.single_tab.columnconfigure(0, weight=1)
        self.single_tab.rowconfigure(0, weight=1)
        panes = ttk.Panedwindow(self.single_tab, orient="horizontal")
        panes.grid(row=0, column=0, sticky="nsew")
        controls = ttk.Frame(panes, width=520)
        viewer_frame = ttk.Frame(panes)
        panes.add(controls, weight=0)
        panes.add(viewer_frame, weight=1)
        controls.columnconfigure(0, weight=1)
        controls.rowconfigure(5, weight=1)
        viewer_frame.columnconfigure(0, weight=1)
        viewer_frame.rowconfigure(1, weight=1)

        source = ttk.LabelFrame(controls, text="1. One source video")
        source.grid(row=0, column=0, sticky="ew", padx=(0, 8), pady=(0, 5))
        source.columnconfigure(0, weight=1)
        self.input_entry = ttk.Entry(source, textvariable=self.input_var)
        self.input_entry.grid(row=0, column=0, sticky="ew", padx=(8, 4), pady=(8, 4))
        self.input_browse_button = ttk.Button(source, text="Browse…", command=self._browse_input)
        self.input_browse_button.grid(row=0, column=1, padx=(4, 8), pady=(8, 4))
        ttk.Label(source, textvariable=self.source_summary_var, wraplength=500, style="Muted.TLabel").grid(
            row=1, column=0, columnspan=2, sticky="w", padx=8, pady=(2, 8)
        )

        denoise = ttk.LabelFrame(controls, text="2. Temporal denoise settings")
        denoise.grid(row=1, column=0, sticky="ew", padx=(0, 8), pady=(0, 5))
        denoise.columnconfigure(1, weight=1)
        ttk.Label(denoise, text="Denoiser").grid(row=0, column=0, sticky="w", padx=8, pady=(8, 3))
        self.denoiser_combo = ttk.Combobox(denoise, textvariable=self.denoiser_var, values=list(DENOISER_LABELS), state="readonly", width=48)
        self.denoiser_combo.grid(row=0, column=1, sticky="ew", padx=(4, 4), pady=(8, 3))
        ttk.Button(denoise, text="?", width=3, command=self._show_current_denoiser_help).grid(row=0, column=2, padx=(0, 8), pady=(8, 3))
        values = ttk.Frame(denoise)
        values.grid(row=1, column=0, columnspan=3, sticky="w", padx=8, pady=(3, 7))
        ttk.Label(values, text="Strength 1–10").pack(side="left")
        ttk.Button(values, text="?", width=2, command=self._show_strength_help).pack(side="left", padx=(3, 3))
        self.strength_spin = ttk.Spinbox(values, from_=MIN_DENOISE_STRENGTH, to=MAX_DENOISE_STRENGTH, textvariable=self.strength_var, width=5)
        self.strength_spin.pack(side="left", padx=(2, 14))
        ttk.Label(values, textvariable=self.radius_label_var).pack(side="left")
        ttk.Button(values, text="?", width=2, command=self._show_radius_help).pack(side="left", padx=(3, 3))
        self.radius_spin = ttk.Spinbox(values, from_=MIN_TEMPORAL_RADIUS, to=MAX_TEMPORAL_RADIUS, textvariable=self.radius_var, width=5)
        self.radius_spin.pack(side="left", padx=(2, 10))
        ttk.Label(values, textvariable=self.window_var, style="Good.TLabel").pack(side="left")
        guidance = ttk.Frame(denoise)
        guidance.grid(row=2, column=0, columnspan=3, sticky="ew", padx=8, pady=(0, 7))
        guidance.columnconfigure(0, weight=1)
        ttk.Label(guidance, textvariable=self.denoiser_rank_var, style="Muted.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        self.denoiser_backend_label = ttk.Label(
            guidance,
            textvariable=self.denoiser_backend_var,
            wraplength=390,
            style="Muted.TLabel",
        )
        self.denoiser_backend_label.grid(row=1, column=0, sticky="w", pady=(2, 0))
        ttk.Button(guidance, text="Acceleration ?", command=self._show_acceleration_help).grid(
            row=0, column=1, rowspan=2, padx=(8, 0)
        )

        preview = ttk.LabelFrame(controls, text="3. Frame preview")
        preview.grid(row=2, column=0, sticky="ew", padx=(0, 8), pady=(0, 5))
        preview.columnconfigure(0, weight=1)
        self.live_check = ttk.Checkbutton(
            preview,
            text="Frame preview — automatically render the selected timeline frame",
            variable=self.frame_preview_var,
            command=self._frame_preview_toggled,
        )
        self.live_check.grid(row=0, column=0, sticky="w", padx=8, pady=(7, 3))
        ttk.Button(preview, text="?", width=3, command=self._show_frame_preview_help).grid(row=0, column=1, padx=(4, 8), pady=(7, 3))
        ttk.Label(
            preview,
            text="The app derives the exact hidden before/after context from the denoiser; no frame-count guess is needed.",
            wraplength=500,
            style="Muted.TLabel",
        ).grid(row=1, column=0, columnspan=2, sticky="w", padx=8, pady=3)
        buttons = ttk.Frame(preview)
        buttons.grid(row=2, column=0, columnspan=2, sticky="ew", padx=8, pady=(4, 3))
        buttons.columnconfigure(2, weight=1)
        self.preview_render_button = ttk.Button(buttons, text="Refresh selected frame", command=lambda: self._start_preview(bool(self.frame_preview_var.get())))
        self.preview_render_button.grid(row=0, column=0, sticky="w")
        self.preview_cancel_button = ttk.Button(buttons, text="Cancel preview", command=self._cancel_preview, state="disabled")
        self.preview_cancel_button.grid(row=0, column=1, sticky="w", padx=(6, 0))
        self.preview_progress = ttk.Progressbar(buttons, variable=self.preview_progress_var, maximum=100)
        self.preview_progress.grid(row=0, column=2, sticky="ew", padx=(8, 0))
        ttk.Label(preview, textvariable=self.preview_status_var, wraplength=500, style="Muted.TLabel").grid(
            row=3, column=0, columnspan=2, sticky="w", padx=8, pady=(2, 8)
        )

        output = ttk.LabelFrame(controls, text="4. Output master and preserved tracks")
        output.grid(row=3, column=0, sticky="ew", padx=(0, 8), pady=(0, 5))
        output.columnconfigure(1, weight=1)
        self._build_output_controls(output, batch=False)

        run = ttk.LabelFrame(controls, text="5. Process the full file")
        run.grid(row=4, column=0, sticky="ew", padx=(0, 8), pady=(0, 5))
        run.columnconfigure(0, weight=1)
        run_buttons = ttk.Frame(run)
        run_buttons.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))
        run_buttons.columnconfigure(2, weight=1)
        self.process_button = ttk.Button(run_buttons, text="Process file", command=self._start_single_processing)
        self.process_button.grid(row=0, column=0, sticky="w")
        self.cancel_process_button = ttk.Button(run_buttons, text="Cancel", command=self._cancel_single_processing, state="disabled")
        self.cancel_process_button.grid(row=0, column=1, sticky="w", padx=(6, 0))
        self.single_progress = ttk.Progressbar(run_buttons, variable=self.single_progress_var, maximum=100)
        self.single_progress.grid(row=0, column=2, sticky="ew", padx=(8, 0))
        ttk.Label(run, textvariable=self.single_status_var, wraplength=500).grid(row=1, column=0, sticky="w", padx=8, pady=(2, 8))

        log_box = ttk.LabelFrame(controls, text="Automatic preflight / run log")
        log_box.grid(row=5, column=0, sticky="nsew", padx=(0, 8))
        log_box.columnconfigure(0, weight=1)
        log_box.rowconfigure(0, weight=1)
        self.single_log = scrolledtext.ScrolledText(log_box, height=10, wrap="word", font=("Consolas", 8), state="disabled")
        self.single_log.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)

        ttk.Label(
            viewer_frame,
            text="Aligned original / denoised frame — hold left for Original; drag to pan; wheel to zoom",
            style="Title.TLabel",
        ).grid(row=0, column=0, sticky="w", pady=(0, 6))
        self.viewer = ComparisonViewer(viewer_frame)
        self.viewer.grid(row=1, column=0, sticky="nsew")

        timeline = ttk.Frame(viewer_frame)
        timeline.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        timeline.columnconfigure(4, weight=1)
        ttk.Button(timeline, text="|◀", width=4, command=lambda: self._step_timeline("first")).grid(row=0, column=0)
        ttk.Button(timeline, text="◀", width=4, command=lambda: self._step_timeline(-1)).grid(row=0, column=1, padx=(4, 0))
        ttk.Button(timeline, text="▶", width=4, command=lambda: self._step_timeline(1)).grid(row=0, column=2, padx=(4, 0))
        ttk.Button(timeline, text="▶|", width=4, command=lambda: self._step_timeline("last")).grid(row=0, column=3, padx=(4, 8))
        self.timeline_scale = ttk.Scale(
            timeline,
            from_=0,
            to=1,
            variable=self.timeline_frame_var,
            command=self._timeline_changed,
        )
        self.timeline_scale.grid(row=0, column=4, sticky="ew")
        self.timeline_scale.bind("<ButtonPress-1>", self._timeline_pressed)
        self.timeline_scale.bind("<B1-Motion>", self._timeline_dragged)
        self.timeline_scale.bind("<ButtonRelease-1>", self._timeline_released)
        self.timeline_scale.bind("<Left>", lambda _event: self._step_timeline(-1))
        self.timeline_scale.bind("<Right>", lambda _event: self._step_timeline(1))
        ttk.Label(timeline, textvariable=self.timeline_label_var, width=31, anchor="e").grid(row=0, column=5, padx=(8, 0))

    def _build_output_controls(self, parent, *, batch: bool) -> None:
        prefix = "batch_" if batch else ""
        ttk.Label(parent, text="Codec family").grid(row=0, column=0, sticky="w", padx=8, pady=(7, 3))
        combo = ttk.Combobox(parent, textvariable=self.family_var, values=list(FAMILY_LABELS), state="readonly", width=35)
        combo.grid(row=0, column=1, sticky="ew", padx=(4, 4), pady=(7, 3))
        setattr(self, prefix + "family_combo", combo)
        ttk.Button(parent, text="?", width=3, command=self._show_codec_guide).grid(row=0, column=2, padx=(0, 8), pady=(7, 3))

        ttk.Label(parent, text="Container").grid(row=1, column=0, sticky="w", padx=8, pady=3)
        container = ttk.Combobox(parent, textvariable=self.container_var, values=container_labels_for_family("ffv1"), state="readonly", width=35)
        container.grid(row=1, column=1, sticky="ew", padx=(4, 4), pady=3)
        setattr(self, prefix + "container_combo", container)
        ttk.Button(parent, text="?", width=3, command=self._show_container_help).grid(row=1, column=2, padx=(0, 8), pady=3)

        row = ttk.Frame(parent)
        row.grid(row=2, column=0, columnspan=3, sticky="w", padx=8, pady=3)
        ttk.Label(row, text="Bit depth").pack(side="left")
        depth = ttk.Combobox(row, textvariable=self.bit_depth_var, values=("10", "12", "16"), state="readonly", width=5)
        depth.pack(side="left", padx=(5, 12))
        setattr(self, prefix + "depth_combo", depth)
        ttk.Label(row, text="FFV1 chroma").pack(side="left")
        chroma = ttk.Combobox(row, textvariable=self.ffv1_chroma_var, values=list(FFV1_CHROMA_LABELS), state="readonly", width=25)
        chroma.pack(side="left", padx=(5, 0))
        setattr(self, prefix + "chroma_combo", chroma)

        options = ttk.Frame(parent)
        options.grid(row=3, column=0, columnspan=3, sticky="w", padx=8, pady=3)
        hardware = ttk.Checkbutton(options, text="NVIDIA encoder when verified", variable=self.hardware_encode_var)
        hardware.pack(side="left")
        setattr(self, prefix + "hardware_check", hardware)
        ttk.Label(options, text="AV1 software").pack(side="left", padx=(12, 0))
        av1 = ttk.Combobox(options, textvariable=self.av1_var, values=list(AV1_LABELS), state="disabled", width=20)
        av1.pack(side="left", padx=(5, 0))
        setattr(self, prefix + "av1_combo", av1)

        quality_row = ttk.Frame(parent)
        quality_row.grid(row=4, column=0, columnspan=3, sticky="w", padx=8, pady=3)
        quality_label = ttk.Label(quality_row, textvariable=self.quality_label_var)
        quality_label.pack(side="left")
        setattr(self, prefix + "quality_label", quality_label)
        quality = ttk.Spinbox(quality_row, from_=0, to=63, textvariable=self.quality_var, width=5)
        quality.pack(side="left", padx=(5, 8))
        setattr(self, prefix + "quality_spin", quality)
        grain = ttk.Checkbutton(quality_row, text="x265 tune grain", variable=self.tune_grain_var)
        grain.pack(side="left")
        setattr(self, prefix + "grain_check", grain)
        ttk.Button(quality_row, text="?", width=3, command=self._show_quality_help).pack(side="left", padx=(6, 0))

        summary = ttk.Label(parent, textvariable=self.encoder_summary_var, wraplength=500, style="Muted.TLabel")
        summary.grid(row=5, column=0, columnspan=3, sticky="w", padx=8, pady=(2, 3))
        setattr(self, prefix + "encoder_summary_label", summary)

        tracks = ttk.Frame(parent)
        tracks.grid(row=6, column=0, columnspan=3, sticky="ew", padx=8, pady=3)
        track_checks = []
        for index, (label, variable) in enumerate(
            (
                ("Audio", self.copy_audio_var),
                ("Subtitles", self.copy_subtitles_var),
                ("Attachments", self.copy_attachments_var),
                ("Data", self.copy_data_var),
                ("Chapters", self.copy_chapters_var),
                ("Metadata", self.copy_metadata_var),
            )
        ):
            check = ttk.Checkbutton(tracks, text=label, variable=variable)
            check.grid(row=0, column=index, sticky="w", padx=(0, 4))
            track_checks.append(check)
        setattr(self, "batch_track_checks" if batch else "single_track_checks", tuple(track_checks))
        if not batch:
            ttk.Label(parent, text="Output file").grid(row=7, column=0, sticky="w", padx=8, pady=(4, 7))
            path_row = ttk.Frame(parent)
            path_row.grid(row=7, column=1, columnspan=2, sticky="ew", padx=(4, 8), pady=(4, 7))
            path_row.columnconfigure(0, weight=1)
            self.output_entry = ttk.Entry(path_row, textvariable=self.output_var)
            self.output_entry.grid(row=0, column=0, sticky="ew")
            self.output_browse_button = ttk.Button(path_row, text="Browse…", command=self._browse_output)
            self.output_browse_button.grid(row=0, column=1, padx=(5, 0))

    def _build_batch_tab(self) -> None:
        self.batch_tab.columnconfigure(0, weight=1)
        self.batch_tab.rowconfigure(1, weight=1)
        toolbar = ttk.Frame(self.batch_tab)
        toolbar.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        self.batch_add_files_button = ttk.Button(toolbar, text="Add files…", command=self._batch_add_files)
        self.batch_add_files_button.pack(side="left")
        self.batch_add_folder_button = ttk.Button(toolbar, text="Add folder…", command=self._batch_add_folder)
        self.batch_add_folder_button.pack(side="left", padx=(5, 0))
        ttk.Checkbutton(toolbar, text="Include subfolders", variable=self.batch_subfolders_var).pack(side="left", padx=(8, 0))
        ttk.Button(toolbar, text="Move up", command=lambda: self._batch_move(-1)).pack(side="left", padx=(12, 0))
        ttk.Button(toolbar, text="Move down", command=lambda: self._batch_move(1)).pack(side="left", padx=(5, 0))
        ttk.Button(toolbar, text="Remove", command=self._batch_remove).pack(side="left", padx=(12, 0))
        ttk.Button(toolbar, text="Clear", command=self._batch_clear).pack(side="left", padx=(5, 0))
        ttk.Label(toolbar, text="Maximum 99 · processed sequentially", style="Muted.TLabel").pack(side="right")

        queue_frame = ttk.Frame(self.batch_tab)
        queue_frame.grid(row=1, column=0, sticky="nsew")
        queue_frame.columnconfigure(0, weight=1)
        queue_frame.rowconfigure(0, weight=1)
        columns = ("state", "effective", "output", "progress")
        self.batch_tree = ttk.Treeview(queue_frame, columns=columns, show="tree headings", selectmode="extended")
        self.batch_tree.heading("#0", text="Source")
        self.batch_tree.heading("state", text="State")
        self.batch_tree.heading("effective", text="Effective settings / fallback")
        self.batch_tree.heading("output", text="Output")
        self.batch_tree.heading("progress", text="Progress / result")
        self.batch_tree.column("#0", width=300, minwidth=160)
        self.batch_tree.column("state", width=115, minwidth=90)
        self.batch_tree.column("effective", width=340, minwidth=180)
        self.batch_tree.column("output", width=300, minwidth=160)
        self.batch_tree.column("progress", width=170, minwidth=120)
        self.batch_tree.grid(row=0, column=0, sticky="nsew")
        yscroll = ttk.Scrollbar(queue_frame, orient="vertical", command=self.batch_tree.yview)
        yscroll.grid(row=0, column=1, sticky="ns")
        self.batch_tree.configure(yscrollcommand=yscroll.set)
        self.batch_tree.bind("<Delete>", lambda _event: self._batch_remove())

        setup = ttk.Frame(self.batch_tab)
        setup.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        setup.columnconfigure(0, weight=1)
        setup.columnconfigure(1, weight=1)
        shared = ttk.LabelFrame(setup, text="Shared denoise settings (synchronized with Single)")
        shared.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        shared.columnconfigure(1, weight=1)
        shared.rowconfigure(3, weight=1)
        self.batch_denoise_settings_frame = shared
        ttk.Label(shared, text="Denoiser").grid(row=0, column=0, sticky="w", padx=8, pady=(7, 3))
        self.batch_denoiser_combo = ttk.Combobox(shared, textvariable=self.denoiser_var, values=list(DENOISER_LABELS), state="readonly", width=45)
        self.batch_denoiser_combo.grid(row=0, column=1, sticky="ew", padx=(4, 8), pady=(7, 3))
        row = ttk.Frame(shared)
        self.batch_denoise_values_row = row
        row.grid(row=1, column=0, columnspan=2, sticky="w", padx=8, pady=(3, 7))
        ttk.Label(row, text="Strength").pack(side="left")
        ttk.Spinbox(row, from_=1, to=10, textvariable=self.strength_var, width=5).pack(side="left", padx=(5, 12))
        ttk.Label(row, text="Temporal radius").pack(side="left")
        self.batch_radius_spin = ttk.Spinbox(row, from_=1, to=6, textvariable=self.radius_var, width=5)
        self.batch_radius_spin.pack(side="left", padx=(5, 10))
        ttk.Label(row, textvariable=self.window_var, style="Good.TLabel").pack(side="left")
        self.batch_denoiser_rank_label = ttk.Label(row, textvariable=self.denoiser_rank_var, style="Muted.TLabel")
        self.batch_denoiser_rank_label.pack(side="left", padx=(18, 0))
        ttk.Label(shared, textvariable=self.denoiser_backend_var, wraplength=600, style="Muted.TLabel").grid(
            row=2, column=0, columnspan=2, sticky="w", padx=8, pady=(0, 5)
        )

        self.batch_log_box = ttk.LabelFrame(shared, text="Batch run log")
        self.batch_log_box.grid(row=3, column=0, columnspan=2, sticky="nsew", padx=8, pady=(2, 7))
        self.batch_log_box.columnconfigure(0, weight=1)
        self.batch_log_box.rowconfigure(0, weight=1)
        self.batch_log = scrolledtext.ScrolledText(
            self.batch_log_box,
            height=7,
            wrap="word",
            font=("Consolas", 8),
            state="disabled",
        )
        self.batch_log.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        output = ttk.LabelFrame(setup, text="Shared output and destination")
        output.grid(row=0, column=1, sticky="nsew", padx=(4, 0))
        output.columnconfigure(1, weight=1)
        self._build_output_controls(output, batch=True)
        ttk.Label(output, text="Output folder").grid(row=7, column=0, sticky="w", padx=8, pady=(4, 7))
        destination = ttk.Frame(output)
        destination.grid(row=7, column=1, columnspan=2, sticky="ew", padx=(4, 8), pady=(4, 7))
        destination.columnconfigure(0, weight=1)
        self.batch_output_entry = ttk.Entry(destination, textvariable=self.batch_output_dir_var)
        self.batch_output_entry.grid(row=0, column=0, sticky="ew")
        ttk.Button(destination, text="Browse…", command=self._batch_browse_output).grid(row=0, column=1, padx=(5, 0))
        ttk.Button(destination, text="Beside sources", command=lambda: self.batch_output_dir_var.set("")).grid(row=0, column=2, padx=(5, 0))

        footer = ttk.Frame(self.batch_tab)
        footer.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        footer.columnconfigure(1, weight=1)
        self.batch_start_button = ttk.Button(footer, text="Start batch", command=self._start_batch)
        self.batch_start_button.grid(row=0, column=0)
        self.batch_cancel_button = ttk.Button(footer, text="Cancel batch", command=self._cancel_batch, state="disabled")
        self.batch_cancel_button.grid(row=0, column=1, sticky="w", padx=(6, 0))
        ttk.Checkbutton(footer, text="Continue compatible rows after an error", variable=self.batch_continue_var).grid(row=0, column=2, padx=(8, 0))
        ttk.Label(footer, textvariable=self.batch_status_var, anchor="e").grid(row=0, column=3, sticky="e", padx=(12, 0))

    def _bind_setting_traces(self) -> None:
        for variable in (
            self.denoiser_var,
            self.strength_var,
            self.radius_var,
        ):
            variable.trace_add("write", self._preview_setting_changed)
        for variable in (
            self.family_var,
            self.container_var,
            self.bit_depth_var,
            self.ffv1_chroma_var,
            self.hardware_encode_var,
            self.av1_var,
            self.quality_var,
            self.tune_grain_var,
            self.copy_audio_var,
            self.copy_subtitles_var,
            self.copy_attachments_var,
            self.copy_data_var,
        ):
            variable.trace_add("write", self._output_setting_changed)

    def _install_drop(self) -> None:
        self.drop_after_id = None
        if self.closing:
            return
        try:
            target = WindowsFileDropTarget(self.root, self._handle_drop, error_callback=lambda message: messagebox.showerror("File drop", message, parent=self.root))
            target.install()
            self.drop_target = target
        except FileDropUnavailable as exc:
            self._append_log(f"Drag-and-drop unavailable; Browse remains available: {exc}")

    def _handle_drop(self, paths: tuple[Path, ...]) -> None:
        if self.busy_single or self.busy_batch:
            messagebox.showwarning("Busy", "Wait for the active job or cancel it before adding files.", parent=self.root)
            return
        if len(paths) == 1 and paths[0].is_file() and self.notebook.index("current") == 0:
            self._select_source(paths[0])
            return
        self._batch_add_paths(paths)
        self.notebook.select(self.batch_tab)

    def _start_capability_scan(self) -> None:
        self.tools_status_var.set("Scanning FFmpeg, VapourSynth, encoders, and six denoisers…")
        configured = (
            Path(self.saved["ffmpeg_path"]) if self.saved["ffmpeg_path"] else None,
            Path(self.saved["ffprobe_path"]) if self.saved["ffprobe_path"] else None,
            Path(self.saved["vspipe_path"]) if self.saved["vspipe_path"] else None,
        )

        def worker() -> None:
            try:
                report = inspect_capabilities(*configured)
            except Exception as exc:
                self.root.after(0, self._capability_scan_failed, exc)
            else:
                self.root.after(0, self._capability_scan_complete, report)

        threading.Thread(target=worker, name="denoise-capability-scan", daemon=True).start()

    def _capability_scan_failed(self, exc: Exception) -> None:
        if self.closing:
            return
        self.capabilities = None
        self.tools_status_var.set(f"Tool scan failed: {type(exc).__name__}: {exc}")
        self.single_status_var.set("Tool scan failed; open Tools to correct paths and rescan.")

    def _capability_scan_complete(self, report: CapabilityReport) -> None:
        if self.closing:
            return
        self.capabilities = report
        ready = sum(report.denoise_capabilities.values())
        ffmpeg = report.ffmpeg_version.split(" Copyright", 1)[0] if report.ffmpeg_version else "FFmpeg missing"
        self.tools_status_var.set(f"{ffmpeg} · {report.vapoursynth_version or 'VapourSynth missing'} · {ready}/6 denoisers ready")
        self.single_status_var.set("Tools ready. Choose one source video.")
        self._update_window_label()
        self._refresh_output_depths()
        if self.input_var.get() and not self.media:
            self._probe_selected_source(Path(self.input_var.get()))

    def _browse_input(self) -> None:
        initial = self.saved.get("last_input_dir") or str(Path.home())
        selected = filedialog.askopenfilename(parent=self.root, title="Choose one source video", initialdir=initial, filetypes=VIDEO_FILE_TYPES)
        if selected:
            self._select_source(Path(selected))

    def _select_source(self, path: Path) -> None:
        if self.busy_single:
            messagebox.showwarning("Processing active", "Cancel the active file before changing the source.", parent=self.root)
            return
        if not path.is_file():
            messagebox.showerror("Source", f"Source does not exist: {path}", parent=self.root)
            return
        self.saved["last_input_dir"] = str(path.parent)
        self.input_var.set(str(path.resolve()))
        self.media = None
        self.source_summary_var.set("Probing source metadata and decoded field flags…")
        self.output_var.set("")
        self._cancel_preview(clear_current=True)
        self.viewer.clear("Probing source…")
        self._refresh_timeline_range()
        if self.capabilities and self.capabilities.ffprobe_path:
            self._probe_selected_source(path.resolve())
        else:
            self.source_summary_var.set("Waiting for FFprobe capability discovery.")

    def _probe_selected_source(self, path: Path) -> None:
        assert self.capabilities and self.capabilities.ffprobe_path
        self.source_cancel_event.set()
        self.source_cancel_event = threading.Event()
        cancel_event = self.source_cancel_event
        self.source_generation += 1
        generation = self.source_generation

        def worker() -> None:
            try:
                media = probe_media_cancelable(self.capabilities.ffprobe_path, path, cancel_event, sample_frames=64)
            except Exception as exc:
                self.root.after(0, self._source_probe_failed, generation, exc)
            else:
                self.root.after(0, self._source_probe_complete, generation, media)

        threading.Thread(target=worker, name="denoise-source-probe", daemon=True).start()

    def _source_probe_failed(self, generation: int, exc: Exception) -> None:
        if self.closing or generation != self.source_generation or isinstance(exc, ProbeCancelled):
            return
        self.source_summary_var.set(f"Source probe failed: {type(exc).__name__}: {exc}")
        self.viewer.clear("Source probe failed.")

    def _source_probe_complete(self, generation: int, media: MediaProbe) -> None:
        if self.closing or generation != self.source_generation:
            return
        self.media = media
        video = media.video
        rate = video.avg_frame_rate or video.r_frame_rate
        field = source_field_order(media)
        field_text = f"interlaced {field.upper()}" if source_is_interlaced(media) and field else ("interlaced order unknown" if source_is_interlaced(media) else "progressive")
        duration = f"{media.duration:.3f}s" if media.duration is not None else "duration unknown"
        self.source_summary_var.set(
            f"{video.width}×{video.height} · {video.pix_fmt or 'pixel format unknown'} · {duration} · "
            f"{float(rate):.3f} fps · {field_text}" if rate else f"{video.width}×{video.height} · {duration} · {field_text}"
        )
        self._refresh_output_depths()
        self._update_window_label()
        self._set_default_output()
        self._refresh_timeline_range()
        self.single_status_var.set("Source ready. Scrub the full timeline or process; preflight runs automatically.")
        self._start_preview(False)

    def _set_default_output(self) -> None:
        if not self.media:
            return
        try:
            profile = self._selected_profile(self.media)
        except ValueError:
            return
        current_text = self.output_var.get().strip()
        current = Path(current_text) if current_text else None
        if current and current.parent.is_dir() and current.name:
            preferred = current.with_suffix(profile.default_extension)
        else:
            preferred = default_output_path(Path(self.input_var.get()), profile)
        self.output_var.set(str(unique_output_path(preferred)))

    def _selected_profile(self, media: MediaProbe | None):
        if media is None:
            raise ValueError("Load a source to resolve its output profile and automatic container.")
        settings = self._collect_settings(Path(self.input_var.get() or "."), Path(self.output_var.get() or "output.mkv"))
        profile, _container = select_output_profile(settings, media)
        return profile

    def _refresh_output_depths(self) -> None:
        if self._output_refreshing:
            return
        self._output_refreshing = True
        try:
            self._refresh_output_controls()
        finally:
            self._output_refreshing = False

    def _refresh_output_controls(self) -> None:
        family = FAMILY_LABELS.get(self.family_var.get(), "ffv1")
        valid_labels = container_labels_for_family(family)
        current_container = CONTAINER_LABELS.get(self.container_var.get(), "auto")
        if current_container not in {CONTAINER_LABELS[label] for label in valid_labels}:
            self.container_var.set(CONTAINER_ID_LABELS["auto"])
        for name in ("container_combo", "batch_container_combo"):
            widget = getattr(self, name, None)
            if widget:
                widget.configure(values=valid_labels)

        hardware_supported = family in {"hevc", "av1"}
        if hardware_supported and self.capabilities:
            encoder = "hevc_nvenc" if family == "hevc" else "av1_nvenc"
            hardware_supported = bool(self.capabilities.encoder_verified_bit_depths.get(encoder))
        if family in {"hevc", "av1"} and self.capabilities and not hardware_supported and self.hardware_encode_var.get():
            self.hardware_encode_var.set(False)
        for name in ("hardware_check", "batch_hardware_check"):
            widget = getattr(self, name, None)
            if widget:
                widget.configure(state="normal" if hardware_supported else "disabled")

        depths = selectable_bit_depths(
            family,
            self.capabilities,
            hardware_encode=bool(self.hardware_encode_var.get()),
            av1_software_encoder=AV1_LABELS.get(self.av1_var.get(), "libaom"),
        )
        values = tuple(str(value) for value in depths)
        for name in ("depth_combo", "batch_depth_combo"):
            combo = getattr(self, name, None)
            if combo:
                combo.configure(values=values)
        if values and self.bit_depth_var.get() not in values:
            self.bit_depth_var.set(values[0])
        ffv1 = family == "ffv1"
        for name in ("chroma_combo", "batch_chroma_combo"):
            widget = getattr(self, name, None)
            if widget:
                widget.configure(state="readonly" if ffv1 else "disabled")

        av1_software_active = family == "av1" and not bool(self.hardware_encode_var.get())
        for name in ("av1_combo", "batch_av1_combo"):
            widget = getattr(self, name, None)
            if widget:
                widget.configure(state="readonly" if av1_software_active else "disabled")

        profile = None
        if self.media and values:
            try:
                profile = self._selected_profile(self.media)
            except (KeyError, ValueError):
                profile = None
        policy = encoder_control_policy(profile, family, bool(self.hardware_encode_var.get()))
        self.quality_label_var.set(policy.quality_label)
        for name in ("quality_spin", "batch_quality_spin"):
            widget = getattr(self, name, None)
            if widget:
                widget.configure(
                    from_=policy.quality_minimum,
                    to=max(policy.quality_minimum, policy.quality_maximum),
                    state="normal" if policy.quality_enabled else "disabled",
                )
        if policy.quality_enabled:
            quality = int(self.quality_var.get())
            if quality < policy.quality_minimum or quality > policy.quality_maximum:
                self.quality_var.set(max(policy.quality_minimum, min(policy.quality_maximum, quality)))
        for name in ("grain_check", "batch_grain_check"):
            widget = getattr(self, name, None)
            if widget:
                widget.configure(state="normal" if policy.tune_grain_enabled else "disabled")

        summary = policy.encoder_summary
        if self.media:
            try:
                settings = self._collect_settings(Path(self.input_var.get() or "."), Path(self.output_var.get() or "output.mkv"))
                container = resolve_container(settings, self.media)
                summary += f"\n{container.reason}"
            except (KeyError, ValueError):
                pass
        self.encoder_summary_var.set(summary)

    def _browse_output(self) -> None:
        try:
            profile = self._selected_profile(self.media)
        except ValueError as exc:
            messagebox.showerror("Output", str(exc), parent=self.root)
            return
        initial_path = Path(self.output_var.get()) if self.output_var.get() else None
        selected = filedialog.asksaveasfilename(
            parent=self.root,
            title="Choose a new denoised output",
            initialdir=str(initial_path.parent if initial_path else Path(self.saved.get("last_output_dir") or Path.home())),
            initialfile=initial_path.name if initial_path else f"denoised{profile.default_extension}",
            defaultextension=profile.default_extension,
            filetypes=[(profile.label, f"*{profile.default_extension}"), ("All files", "*.*")],
        )
        if selected:
            path = Path(selected)
            self.saved["last_output_dir"] = str(path.parent)
            if path.exists():
                path = unique_output_path(path)
                messagebox.showinfo("Existing output preserved", f"A unique filename was selected instead:\n{path}", parent=self.root)
            self.output_var.set(str(path))

    def _update_window_label(self) -> None:
        if self._denoiser_refreshing:
            return
        self._denoiser_refreshing = True
        try:
            self._refresh_denoiser_controls()
        finally:
            self._denoiser_refreshing = False

    def _refresh_denoiser_controls(self) -> None:
        identifier = DENOISER_LABELS.get(self.denoiser_var.get(), "vs_bm3d")
        strength = max(1, min(10, int(self.strength_var.get())))
        radius = normalize_temporal_radius(identifier, int(self.radius_var.get()))
        if radius != int(self.radius_var.get()):
            self.radius_var.set(radius)
        policy = denoiser_control_policy(identifier, strength, radius)
        if identifier == "ffmpeg_fftdnoiz":
            self.radius_label_var.set("Temporal radius (fixed)")
            window_text = "Fixed 3-frame window"
        elif identifier == "ffmpeg_atadenoise":
            self.radius_label_var.set("Temporal radius 2–6")
            window_text = f"{policy.window_frames}-frame window"
        elif identifier == "vs_dfttest":
            self.radius_label_var.set("Temporal radius 1–3")
            window_text = f"{policy.window_frames}-frame window"
        else:
            self.radius_label_var.set("Temporal radius 1–6")
            window_text = f"{policy.window_frames}-frame window"
        self.window_var.set(window_text)
        ranking = denoiser_ranking(identifier)
        self.denoiser_rank_var.set(
            f"Selection guide: Quality {ranking.quality_score}/6 · Speed {ranking.speed_score}/6 (6 = highest)"
        )
        backend = denoiser_backend_status(
            identifier,
            self.capabilities,
            self.media.video.width if self.media else None,
            self.media.video.height if self.media else None,
        )
        self.denoiser_backend_var.set(backend.summary)
        if hasattr(self, "denoiser_backend_label"):
            style = "Good.TLabel" if backend.available else ("Warn.TLabel" if self.capabilities else "Muted.TLabel")
            self.denoiser_backend_label.configure(style=style)
        for widget in (getattr(self, "radius_spin", None), getattr(self, "batch_radius_spin", None)):
            if widget:
                widget.configure(
                    from_=policy.radius_minimum,
                    to=policy.radius_maximum,
                    state="normal" if policy.radius_enabled else "disabled",
                )

    def _preview_setting_changed(self, *_args) -> None:
        try:
            self._update_window_label()
        except Exception:
            return
        if self.frame_preview_var.get():
            self._schedule_frame_render()

    def _output_setting_changed(self, *_args) -> None:
        if self._output_refreshing:
            return
        try:
            self._refresh_output_depths()
        except Exception:
            return
        if self.media and self.output_var.get():
            self._set_default_output()

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        seconds = max(0.0, seconds)
        whole = int(seconds)
        millis = round((seconds - whole) * 1000)
        if millis == 1000:
            whole += 1
            millis = 0
        hours, remainder = divmod(whole, 3600)
        minutes, second = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{second:02d}.{millis:03d}"

    def _timeline_changed(self, value: str) -> None:
        if self.timeline_guard:
            return
        try:
            frame = round(float(value))
        except ValueError:
            return
        frame = max(0, min(max(0, self.timeline_total_frames - 1), frame))
        self._set_timeline_label(frame)
        if self.preview_active_renderer:
            self.preview_active_renderer.cancel()
        self._schedule_frame_render()

    def _set_timeline_from_pointer(self, event) -> int:
        target = frame_from_timeline_position(
            event.x,
            self.timeline_scale.winfo_width(),
            self.timeline_total_frames,
        )
        self.timeline_guard = True
        self.timeline_frame_var.set(target)
        self.timeline_guard = False
        self._set_timeline_label(target)
        return target

    def _timeline_pressed(self, event) -> str:
        self.timeline_dragging = True
        self.timeline_pointer_render_frame = None
        target = self._set_timeline_from_pointer(event)
        if self.frame_preview_var.get():
            self._schedule_frame_render()
        else:
            self._schedule_frame_render(immediate=True)
            self.timeline_pointer_render_frame = target
        return "break"

    def _timeline_dragged(self, event) -> str:
        target = self._set_timeline_from_pointer(event)
        if target == self.timeline_pointer_render_frame:
            return "break"
        self._schedule_frame_render(immediate=not self.frame_preview_var.get())
        self.timeline_pointer_render_frame = target
        return "break"

    def _timeline_released(self, event=None) -> str:
        target = self._set_timeline_from_pointer(event) if event is not None else self._current_timeline_frame()
        if self.frame_preview_var.get() or target != self.timeline_pointer_render_frame:
            self._schedule_frame_render(immediate=True)
        self.timeline_dragging = False
        self.timeline_pointer_render_frame = target
        return "break"

    def _step_timeline(self, step) -> str:
        if self.timeline_total_frames <= 0:
            return "break"
        current = self._current_timeline_frame()
        if step == "first":
            target = 0
        elif step == "last":
            target = self.timeline_total_frames - 1
        else:
            target = current + int(step)
        target = max(0, min(self.timeline_total_frames - 1, target))
        self.timeline_guard = True
        self.timeline_frame_var.set(target)
        self.timeline_guard = False
        self._set_timeline_label(target)
        self._schedule_frame_render(immediate=True)
        return "break"

    def _current_timeline_frame(self) -> int:
        try:
            value = round(float(self.timeline_frame_var.get()))
        except (TypeError, ValueError):
            value = 0
        return max(0, min(max(0, self.timeline_total_frames - 1), value))

    def _set_timeline_label(self, frame: int) -> None:
        fps = source_fps(self.media) if self.media else 24.0
        timestamp = self._format_timestamp(frame / fps)
        total = self.timeline_total_frames
        self.timeline_label_var.set(f"Frame {frame + 1 if total else 0} / {total} · {timestamp}")

    def _refresh_timeline_range(self) -> None:
        if not self.media:
            total = 0
        else:
            total = source_frame_count(self.media) or 1
        self.timeline_total_frames = total
        if hasattr(self, "timeline_scale"):
            self.timeline_scale.configure(to=max(1, total - 1))
        self.timeline_guard = True
        self.timeline_frame_var.set(0)
        self.timeline_guard = False
        self._set_timeline_label(0)

    def _frame_preview_toggled(self) -> None:
        if not self.frame_preview_var.get():
            self._cancel_preview()
            self.viewer.set_processed_visible(False)
            self.preview_status_var.set("Frame preview is off; showing the unprocessed source frame.")
        else:
            self._schedule_frame_render(immediate=True)

    def _schedule_frame_render(self, *, immediate: bool = False) -> None:
        if self.closing or not self.media or not self.capabilities:
            return
        if self.busy_single or self.busy_batch:
            return
        # Invalidate an already-completed worker immediately, not only when
        # the debounced replacement starts. This prevents an old callback
        # queued on Tk's event loop from replacing a newer timeline target.
        self.preview_generation += 1
        if self.preview_active_renderer:
            self.preview_active_renderer.cancel()
        if self.preview_after_id:
            self.root.after_cancel(self.preview_after_id)
        delay = timeline_render_delay_ms(bool(self.frame_preview_var.get()), immediate)
        self.preview_after_id = self.root.after(
            delay,
            lambda: self._start_preview(bool(self.frame_preview_var.get())),
        )

    def _start_preview(self, include_processed: bool) -> None:
        self.preview_after_id = None
        if not self.media or not self.capabilities or not self.capabilities.ffmpeg_path:
            return
        try:
            strength = int(self.strength_var.get())
            radius = int(self.radius_var.get())
        except ValueError as exc:
            self.preview_status_var.set(str(exc))
            return
        target_frame = self._current_timeline_frame()
        render_width = self.media.video.width or 960
        render_height = self.media.video.height or 540
        self.preview_generation += 1
        generation = self.preview_generation
        if self.preview_active_renderer:
            self.preview_active_renderer.cancel()
        renderer = PreviewRenderer()
        self.preview_active_renderer = renderer
        request = PreviewRequest(
            source=Path(self.input_var.get()),
            media=self.media,
            capabilities=self.capabilities,
            denoiser=DENOISER_LABELS[self.denoiser_var.get()],
            strength=strength,
            temporal_radius=radius,
            target_frame=target_frame,
            width=render_width,
            height=render_height,
            include_processed=include_processed,
        )
        self.preview_progress_var.set(0)
        self.preview_cancel_button.configure(state="normal")
        if include_processed:
            policy = denoiser_control_policy(request.denoiser, strength, radius)
            context = (policy.window_frames - 1) // 2
            self.preview_status_var.set(
                f"Rendering frame {target_frame + 1} with up to {context} required frame(s) before and after…"
            )
        else:
            self.preview_status_var.set(f"Loading unprocessed frame {target_frame + 1}…")

        def progress(values: dict[str, str]) -> None:
            self.root.after(0, self._preview_progress, generation, values)

        def worker() -> None:
            try:
                result = renderer.render(request, progress_callback=progress)
            except Exception as exc:
                self.root.after(0, self._preview_failed, generation, renderer, exc)
            else:
                self.root.after(0, self._preview_complete, generation, renderer, result, include_processed)

        threading.Thread(target=worker, name=f"denoise-preview-{generation}", daemon=True).start()

    def _preview_progress(self, generation: int, values: dict[str, str]) -> None:
        if self.closing or generation != self.preview_generation:
            return
        frame = values.get("frame")
        phase = values.get("phase")
        if phase == "preview_start":
            self.preview_progress_var.set(10)
        elif phase == "preview_progress":
            self.preview_progress_var.set(70)
        elif phase == "preview_complete" or frame:
            self.preview_progress_var.set(95)

    def _preview_failed(self, generation: int, renderer: PreviewRenderer, exc: Exception) -> None:
        renderer.close()
        if self.preview_active_renderer is renderer:
            self.preview_active_renderer = None
        if self.closing or generation != self.preview_generation:
            return
        self.preview_cancel_button.configure(state="disabled")
        if isinstance(exc, PreviewCancelled):
            self.preview_status_var.set("Preview canceled.")
        else:
            self.preview_status_var.set(f"Preview failed: {type(exc).__name__}: {exc}")
            self._append_log(self.preview_status_var.get())

    def _preview_complete(self, generation: int, renderer: PreviewRenderer, result: PreviewFrames, include_processed: bool) -> None:
        if self.closing or generation != self.preview_generation:
            if self.preview_active_renderer is renderer:
                self.preview_active_renderer = None
            renderer.cleanup(result)
            renderer.close()
            return
        old_owner, old_frames = self.preview_owner, self.current_preview
        self.preview_owner = renderer
        self.current_preview = result
        self.preview_active_renderer = None
        self.timeline_guard = True
        self.timeline_frame_var.set(result.target_frame)
        self.timeline_guard = False
        self._set_timeline_label(result.target_frame)
        self.viewer.set_frames(result, show_processed=include_processed and self.frame_preview_var.get())
        if old_owner and old_frames:
            old_owner.cleanup(old_frames)
            old_owner.close()
        self.preview_progress_var.set(100)
        self.preview_cancel_button.configure(state="disabled")
        self.preview_status_var.set(result.status)

    def _cancel_preview(self, *, clear_current: bool = False) -> None:
        self.preview_generation += 1
        if self.preview_after_id:
            self.root.after_cancel(self.preview_after_id)
            self.preview_after_id = None
        if self.preview_active_renderer:
            self.preview_active_renderer.cancel()
            self.preview_active_renderer = None
        self.preview_cancel_button.configure(state="disabled")
        if clear_current:
            if self.preview_owner and self.current_preview:
                self.preview_owner.cleanup(self.current_preview)
                self.preview_owner.close()
            self.preview_owner = None
            self.current_preview = None
            self.viewer.clear()

    def _collect_settings(self, input_path: Path | None = None, output_path: Path | None = None) -> DenoiseSettings:
        return DenoiseSettings(
            input_path=input_path or Path(self.input_var.get() or "."),
            output_path=output_path or Path(self.output_var.get() or "."),
            denoiser=DENOISER_LABELS[self.denoiser_var.get()],
            denoise_strength=int(self.strength_var.get()),
            denoise_temporal_radius=int(self.radius_var.get()),
            family=FAMILY_LABELS[self.family_var.get()],
            container=CONTAINER_LABELS[self.container_var.get()],
            bit_depth=int(self.bit_depth_var.get()),
            ffv1_chroma_mode=FFV1_CHROMA_LABELS[self.ffv1_chroma_var.get()],
            hardware_encode=bool(self.hardware_encode_var.get()),
            av1_software_encoder=AV1_LABELS[self.av1_var.get()],
            quality=int(self.quality_var.get()),
            tune_grain=bool(self.tune_grain_var.get()),
            copy_audio=bool(self.copy_audio_var.get()),
            copy_subtitles=bool(self.copy_subtitles_var.get()),
            copy_attachments=bool(self.copy_attachments_var.get()),
            copy_data=bool(self.copy_data_var.get()),
            copy_chapters=bool(self.copy_chapters_var.get()),
            copy_metadata=bool(self.copy_metadata_var.get()),
        )

    def _single_plan(self):
        if not self.media or not self.capabilities:
            return None
        try:
            settings = self._collect_settings()
        except (ValueError, KeyError) as exc:
            messagebox.showerror("Settings", f"Invalid setting: {exc}", parent=self.root)
            return None
        return build_plan(settings, self.media, self.capabilities)

    def _start_single_processing(self) -> None:
        if self.busy_single or self.busy_batch:
            return
        plan = self._single_plan()
        if plan is None:
            messagebox.showwarning("Process", "Load one source and wait for tool discovery first.", parent=self.root)
            return
        self._append_log("\nAUTOMATIC PREFLIGHT")
        if not plan.valid:
            for error in plan.errors:
                self._append_log("ERROR: " + error)
            messagebox.showerror("Automatic preflight failed", "\n\n".join(plan.errors), parent=self.root)
            return
        for warning in plan.warnings:
            self._append_log("WARNING: " + warning)
        self._append_log("COMMAND: " + plan.display_command)
        self._cancel_preview()
        self.busy_single = True
        self.processor = DenoiseProcessor()
        self.process_button.configure(state="disabled")
        self.cancel_process_button.configure(state="normal")
        self.single_progress_var.set(0)
        self.single_status_var.set("Processing denoise output…")
        self._append_log("\nPROCESS START")

        def progress(values: dict[str, str]) -> None:
            self.root.after(0, self._single_progress, plan, values)

        def log(line: str) -> None:
            self.root.after(0, self._append_log, line)

        def worker() -> None:
            result = self.processor.run(plan, log_callback=log, progress_callback=progress)
            self.root.after(0, self._single_processing_complete, result)

        threading.Thread(target=worker, name="denoise-full-file", daemon=True).start()

    def _single_progress(self, plan, values: dict[str, str]) -> None:
        if self.closing:
            return
        phase = values.get("phase", "processing").replace("_", " ")
        frame = values.get("frame")
        expected = plan.expected.frame_count if plan.expected else None
        if frame and expected:
            try:
                self.single_progress_var.set(min(100.0, 100 * int(frame) / expected))
            except (ValueError, ZeroDivisionError):
                pass
        self.single_status_var.set(phase + (f" · frame {frame}" if frame else ""))

    def _single_processing_complete(self, result) -> None:
        if self.closing:
            return
        self.busy_single = False
        self.processor = None
        self.process_button.configure(state="normal")
        self.cancel_process_button.configure(state="disabled")
        if result.success:
            self.single_progress_var.set(100)
            self.single_status_var.set(f"Completed and validated: {result.output_path}")
            messagebox.showinfo("Denoise complete", f"Validated output:\n{result.output_path}\n\nSHA-256:\n{result.output_sha256}", parent=self.root)
        elif result.canceled:
            self.single_status_var.set("Processing canceled safely; no final output was promoted.")
        else:
            self.single_status_var.set("Processing failed: " + result.message)
            messagebox.showerror("Denoise failed", result.message, parent=self.root)

    def _cancel_single_processing(self) -> None:
        if self.processor:
            self.single_status_var.set("Canceling active processes…")
            self.processor.cancel()

    def _append_log(self, line: str) -> None:
        if not hasattr(self, "single_log") or self.closing:
            return
        self.single_log.configure(state="normal")
        self.single_log.insert("end", str(line).rstrip() + "\n")
        self.single_log.see("end")
        self.single_log.configure(state="disabled")

    def _append_batch_log(self, line: str) -> None:
        if not hasattr(self, "batch_log") or self.closing:
            return
        self.batch_log.configure(state="normal")
        self.batch_log.insert("end", str(line).rstrip() + "\n")
        self.batch_log.see("end")
        self.batch_log.configure(state="disabled")

    def _batch_add_files(self) -> None:
        selected = filedialog.askopenfilenames(parent=self.root, title="Add video files", initialdir=self.saved.get("last_input_dir") or str(Path.home()), filetypes=VIDEO_FILE_TYPES)
        if selected:
            self._batch_add_paths(tuple(Path(path) for path in selected))

    def _batch_add_folder(self) -> None:
        selected = filedialog.askdirectory(parent=self.root, title="Add a video folder", initialdir=self.saved.get("last_input_dir") or str(Path.home()))
        if selected:
            self._batch_add_paths((Path(selected),))

    def _batch_add_paths(self, paths: tuple[Path, ...]) -> None:
        if self.busy_batch:
            return
        result = self.batch_queue.add_paths(paths, include_subfolders=bool(self.batch_subfolders_var.get()))
        for record in result.added:
            self.batch_tree.insert("", "end", iid=record.identifier, text=str(record.source_path), values=(record.state, record.effective_text, "", record.progress_text))
        parts = [f"Added {len(result.added)}"]
        if result.duplicates:
            parts.append(f"{len(result.duplicates)} duplicate")
        if result.unsupported:
            parts.append(f"{len(result.unsupported)} unsupported")
        if result.missing:
            parts.append(f"{len(result.missing)} missing")
        if result.capacity_rejected:
            parts.append(f"{len(result.capacity_rejected)} over capacity")
        self.batch_status_var.set(" · ".join(parts) + f" · {len(self.batch_queue)}/99 queued")

    def _batch_selected(self) -> tuple[str, ...]:
        return tuple(self.batch_tree.selection())

    def _batch_move(self, direction: int) -> None:
        selected = self._batch_selected()
        if not selected or self.busy_batch:
            return
        self.batch_queue.move(selected, direction)
        for index, record in enumerate(self.batch_queue.records):
            self.batch_tree.move(record.identifier, "", index)

    def _batch_remove(self) -> None:
        selected = self._batch_selected()
        if not selected or self.busy_batch:
            return
        self.batch_queue.remove(selected)
        for identifier in selected:
            self.batch_tree.delete(identifier)
        self.batch_status_var.set(f"{len(self.batch_queue)}/99 queued")

    def _batch_clear(self) -> None:
        if self.busy_batch:
            return
        self.batch_queue.clear()
        for item in self.batch_tree.get_children():
            self.batch_tree.delete(item)
        self.batch_status_var.set("Queue is empty.")

    def _batch_browse_output(self) -> None:
        selected = filedialog.askdirectory(parent=self.root, title="Choose batch output folder", initialdir=self.batch_output_dir_var.get() or str(Path.home()))
        if selected:
            self.batch_output_dir_var.set(selected)

    def _start_batch(self) -> None:
        if self.busy_batch or self.busy_single:
            return
        if not self.batch_queue.records:
            messagebox.showwarning("Batch", "Add at least one video.", parent=self.root)
            return
        if not self.capabilities:
            messagebox.showwarning("Batch", "Wait for tool discovery or correct the paths in Tools.", parent=self.root)
            return
        try:
            requested = self._collect_settings(Path("batch-input"), Path("batch-output"))
        except (ValueError, KeyError) as exc:
            messagebox.showerror("Batch settings", str(exc), parent=self.root)
            return
        output_dir = Path(self.batch_output_dir_var.get()) if self.batch_output_dir_var.get() else None
        if output_dir and not output_dir.is_dir():
            messagebox.showerror("Batch output", f"Output folder does not exist: {output_dir}", parent=self.root)
            return
        self._cancel_preview()
        self.busy_batch = True
        self._append_batch_log("")
        self._append_batch_log("=== NEW BATCH RUN ===")
        self.batch_start_button.configure(state="disabled")
        self.batch_cancel_button.configure(state="normal")
        runner = BatchRunner(event_callback=lambda kind, record, payload: self.root.after(0, self._batch_event, kind, record, payload))
        self.batch_runner = runner
        options = BatchRunOptions(output_dir, bool(self.batch_continue_var.get()))

        def worker() -> None:
            try:
                summary = runner.run(self.batch_queue, requested, self.capabilities, options)
            except Exception as exc:
                self.root.after(0, self._batch_failed, exc)
            else:
                self.root.after(0, self._batch_complete, summary)

        threading.Thread(target=worker, name="denoise-batch", daemon=True).start()

    def _batch_event(self, kind: str, record: BatchRecord | None, payload: object | None) -> None:
        if self.closing:
            return
        if kind == "row" and record and self.batch_tree.exists(record.identifier):
            percent = f"{record.percent:.0f}%" if record.percent is not None else ""
            progress = record.progress_text + (f" · {percent}" if percent else "")
            if record.error:
                progress = record.error
            self.batch_tree.item(
                record.identifier,
                text=str(record.source_path),
                values=(record.state, record.effective_text, str(record.output_path or ""), progress),
            )
            self.batch_tree.see(record.identifier)
        elif kind == "phase" and payload:
            self.batch_status_var.set(str(payload))
        elif kind == "log" and payload is not None:
            prefix = f"[{record.source_path.name}] " if record else ""
            self._append_batch_log(prefix + str(payload))

    def _batch_complete(self, summary) -> None:
        if self.closing:
            return
        self.busy_batch = False
        self.batch_runner = None
        self.batch_start_button.configure(state="normal")
        self.batch_cancel_button.configure(state="disabled")
        self.batch_status_var.set(
            f"Completed {summary.completed}/{summary.total} · failed {summary.failed} · canceled {summary.canceled} · skipped {summary.skipped}"
        )
        self._append_batch_log(self.batch_status_var.get())

    def _batch_failed(self, exc: Exception) -> None:
        if self.closing:
            return
        self.busy_batch = False
        self.batch_runner = None
        self.batch_start_button.configure(state="normal")
        self.batch_cancel_button.configure(state="disabled")
        self.batch_status_var.set(f"Batch failed: {type(exc).__name__}: {exc}")
        self._append_batch_log(self.batch_status_var.get())
        messagebox.showerror("Batch failed", self.batch_status_var.get(), parent=self.root)

    def _cancel_batch(self) -> None:
        if self.batch_runner:
            self.batch_status_var.set("Canceling active preflight or process…")
            self._append_batch_log(self.batch_status_var.get())
            self.batch_runner.cancel()

    def _show_tools_dialog(self) -> None:
        dialog = Toplevel(self.root)
        dialog.title("Video Denoise Studio tools")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.columnconfigure(1, weight=1)
        values = {
            "FFmpeg": StringVar(value=self.saved.get("ffmpeg_path", "")),
            "FFprobe": StringVar(value=self.saved.get("ffprobe_path", "")),
            "VSPipe": StringVar(value=self.saved.get("vspipe_path", "")),
        }

        def browse(label: str) -> None:
            selected = filedialog.askopenfilename(parent=dialog, title=f"Choose {label}", filetypes=[("Executable", "*.exe"), ("All files", "*.*")])
            if selected:
                values[label].set(selected)

        for row, label in enumerate(values):
            ttk.Label(dialog, text=label).grid(row=row, column=0, sticky="w", padx=10, pady=(10 if row == 0 else 4, 4))
            ttk.Entry(dialog, textvariable=values[label], width=75).grid(row=row, column=1, sticky="ew", padx=6, pady=(10 if row == 0 else 4, 4))
            ttk.Button(dialog, text="Browse…", command=lambda name=label: browse(name)).grid(row=row, column=2, padx=10, pady=(10 if row == 0 else 4, 4))
        diagnostic = scrolledtext.ScrolledText(dialog, width=100, height=18, wrap="word", font=("Consolas", 8))
        diagnostic.grid(row=3, column=0, columnspan=3, sticky="nsew", padx=10, pady=8)
        if self.capabilities:
            report = self.capabilities
            lines = [
                f"FFmpeg: {report.ffmpeg_path}\n{report.ffmpeg_version}",
                f"FFprobe: {report.ffprobe_path}\n{report.ffprobe_version}",
                f"VSPipe: {report.vspipe_path}\n{report.vapoursynth_version}",
                "",
                "Denoisers:",
            ]
            for spec in DENOISER_SPECS:
                state = "READY" if report.denoise_capabilities.get(spec.identifier) else "UNAVAILABLE"
                backend = denoiser_backend_status(spec.identifier, report)
                lines.append(
                    f"{state}: {spec.label}\n  Effective: {backend.classification} — {backend.display}\n  "
                    f"{report.denoise_diagnostics.get(spec.identifier, '')}"
                )
            diagnostic.insert("1.0", "\n".join(lines))
        diagnostic.configure(state="disabled")

        def save_and_rescan() -> None:
            self.saved["ffmpeg_path"] = values["FFmpeg"].get().strip()
            self.saved["ffprobe_path"] = values["FFprobe"].get().strip()
            self.saved["vspipe_path"] = values["VSPipe"].get().strip()
            self.capabilities = None
            dialog.destroy()
            self._start_capability_scan()

        actions = ttk.Frame(dialog)
        actions.grid(row=4, column=0, columnspan=3, sticky="e", padx=10, pady=(0, 10))
        ttk.Button(actions, text="Save paths + rescan", command=save_and_rescan).pack(side="left")
        ttk.Button(actions, text="Close", command=dialog.destroy).pack(side="left", padx=(6, 0))

    def _show_denoiser_guide(self) -> None:
        sections: list[str] = []
        for spec in DENOISER_SPECS:
            radius = normalize_temporal_radius(spec.identifier, int(self.radius_var.get()))
            policy = denoiser_control_policy(spec.identifier, int(self.strength_var.get()), radius)
            ranking = denoiser_ranking(spec.identifier)
            backend = denoiser_backend_status(
                spec.identifier,
                self.capabilities,
                self.media.video.width if self.media else None,
                self.media.video.height if self.media else None,
            )
            sections.append(
                f"{spec.label}\nQuality {ranking.quality_score}/6 · Speed {ranking.speed_score}/6\n"
                f"{backend.summary}\n{policy.overview}\nStrength: {policy.strength_help}\nRadius: {policy.radius_help}\n"
                f"Quality basis: {ranking.quality_basis}\nSpeed basis: {ranking.speed_basis}"
            )
        sections.append(
            "Frame preview\nThe timeline target is the only displayed frame. The app automatically adds the exact real "
            "leading/trailing context required by that denoiser, processes the window, and trims it back to the target."
        )
        sections.append(denoiser_rankings_guide())
        self._show_text_dialog("Temporal denoiser and setting guide", "\n\n".join(sections))

    def _show_current_denoiser_help(self) -> None:
        identifier = DENOISER_LABELS.get(self.denoiser_var.get(), "vs_bm3d")
        policy = denoiser_control_policy(identifier, int(self.strength_var.get()), int(self.radius_var.get()))
        ranking = denoiser_ranking(identifier)
        backend = denoiser_backend_status(
            identifier,
            self.capabilities,
            self.media.video.width if self.media else None,
            self.media.video.height if self.media else None,
        )
        messagebox.showinfo(
            "Selected denoiser",
            f"{self.denoiser_var.get()}\n\nQuality {ranking.quality_score}/6 · Speed {ranking.speed_score}/6 "
            f"(6 = highest)\n{backend.summary}\n\n{policy.overview}\n\n{policy.strength_help}\n\n"
            f"{policy.radius_help}\n\nQuality basis: {ranking.quality_basis}\nSpeed basis: {ranking.speed_basis}\n\n"
            "Scores are selection guidance, not universal measurements. Source noise, motion, texture, Strength, "
            "radius, raster, and hardware can change the result and speed.",
            parent=self.root,
        )

    def _show_acceleration_help(self) -> None:
        identifier = DENOISER_LABELS.get(self.denoiser_var.get(), "vs_bm3d")
        status = denoiser_backend_status(
            identifier,
            self.capabilities,
            self.media.video.width if self.media else None,
            self.media.video.height if self.media else None,
        )
        messagebox.showinfo(
            "Denoiser acceleration",
            f"{self.denoiser_var.get()}\n\n{status.summary}\n\n{status.help_text}",
            parent=self.root,
        )

    def _show_strength_help(self) -> None:
        identifier = DENOISER_LABELS.get(self.denoiser_var.get(), "vs_bm3d")
        policy = denoiser_control_policy(identifier, int(self.strength_var.get()), int(self.radius_var.get()))
        messagebox.showinfo(
            "Denoise Strength 1–10",
            "Strength is an app-normalized 1–10 scale and applies to all six denoisers. Each algorithm has a different "
            "native parameter; equal numbers are therefore comparable in intent, not mathematically identical.\n\n"
            + policy.strength_help,
            parent=self.root,
        )

    def _show_radius_help(self) -> None:
        identifier = DENOISER_LABELS.get(self.denoiser_var.get(), "vs_bm3d")
        policy = denoiser_control_policy(identifier, int(self.strength_var.get()), int(self.radius_var.get()))
        messagebox.showinfo(
            "Temporal radius",
            "Radius means real frames requested on each side of the target. It applies to ATADENOISE, V-BM3D, "
            "DFTTest2, MVTools, and NLMeans. FFTDNOIZ is fixed and its control is disabled. DFTTest2 is limited "
            "to radius 1–3 by the optimized CPU/NVRTC implementations; the other adjustable VapourSynth routes "
            "support 1–6.\n\n" + policy.radius_help,
            parent=self.root,
        )

    def _show_frame_preview_help(self) -> None:
        messagebox.showinfo(
            "Frame preview and viewer controls",
            "Move anywhere on the full source timeline. With Frame preview checked, the selected target frame is "
            "automatically denoised using the current Strength and Temporal radius. Radius 4 means a nominal centered "
            "9-frame window; radius 6 means 13. There is no manual preview-frame count. At the first/last frames, only "
            "existing source neighbors are used. The completed status states the applied Strength, radius/window, actual "
            "real context, and effective backend.\n\n"
            "Clicking or dragging the timeline seeks proportionally to that location. With Frame preview off, this runs "
            "only an asynchronous source-frame seek/decode—no denoiser.\n\n"
            "Press and hold the left mouse button to see Original. Release to return to Denoised. Drag while holding "
            "left to pan. Rotate the mouse wheel to zoom around the pointer. Changing a denoise setting keeps the same "
            "zoom and source location; choosing another frame resets to Fit. Fit can also be selected manually.",
            parent=self.root,
        )

    def _show_quality_help(self) -> None:
        family = FAMILY_LABELS.get(self.family_var.get(), "ffv1")
        profile = None
        if self.media:
            try:
                profile = self._selected_profile(self.media)
            except ValueError:
                pass
        policy = encoder_control_policy(profile, family, bool(self.hardware_encode_var.get()))
        messagebox.showinfo(
            "Encoder quality setting",
            f"{policy.quality_label}\n\n{policy.encoder_summary}\n\n{policy.codec_help}\n\n"
            "For CQ/CRF, lower numbers retain more detail and create larger files. FFV1 is lossless, while ProRes and "
            "DNxHR use fixed profiles, so this control is disabled for them.",
            parent=self.root,
        )

    def _show_container_help(self) -> None:
        messagebox.showinfo("Output containers", container_help_text(), parent=self.root)

    def _show_codec_guide(self) -> None:
        text = (
            "FFV1 16-bit lossless master\nMathematically lossless after denoising; very large; MKV only. Quality, "
            "NVIDIA, and grain-tune controls do not apply.\n\n"
            "HEVC / H.265\nNVIDIA (when a real capability encode passes): 10/12-bit NVENC, P7, UHQ, VBR constant "
            "quality, zero target bitrate, full-resolution multipass, temporal AQ, B-reference mode, and UHQ-managed "
            "lookahead. The app intentionally does not set explicit lookahead because UHQ enables it. Software mode uses "
            "x265 placebo + CRF; x265 tune grain is optional.\n\n"
            "AV1\nNVIDIA uses the equivalent P7/UHQ/VBR-CQ/full-resolution-multipass/temporal-AQ route when verified. "
            "Software choices are libaom cpu-used 0 or SVT-AV1 preset 0. AV1 software encoding can be extremely slow.\n\n"
            "ProRes 4444 XQ / DNxHR 444\nHigh-bitrate 10-bit 4:4:4 MOV editing intermediates with fixed profiles; "
            "neither uses CQ/CRF and neither is mathematically lossless.\n\n"
            + container_help_text()
            + "\n\nTrack preservation\nMKV is safest for subtitles, attachments, data, and unusual audio. MP4/MOV "
            "preflight refuses streams they cannot preserve or safely convert and explains whether to choose MKV or "
            "deselect that track type."
        )
        self._show_text_dialog("Codec, encoder, container, and track guide", text)

    def _show_text_dialog(self, title: str, text: str) -> None:
        dialog = Toplevel(self.root)
        dialog.title(title)
        dialog.transient(self.root)
        dialog.geometry("820x650")
        dialog.minsize(600, 420)
        dialog.columnconfigure(0, weight=1)
        dialog.rowconfigure(0, weight=1)
        box = scrolledtext.ScrolledText(dialog, wrap="word", font=("Segoe UI", 10), padx=10, pady=10)
        box.grid(row=0, column=0, sticky="nsew", padx=10, pady=(10, 6))
        box.insert("1.0", text)
        box.configure(state="disabled")
        ttk.Button(dialog, text="Close", command=dialog.destroy).grid(row=1, column=0, sticky="e", padx=10, pady=(0, 10))

    def self_test(self) -> dict[str, object]:
        capabilities = self.capabilities
        tabs = tuple(self.notebook.tab(index, "text") for index in range(self.notebook.index("end")))
        return {
            "application_version": __version__,
            "title": self.root.title(),
            "top_level_tabs": tabs,
            "single_file_tab": tabs[0] if tabs else None,
            "batch_tab_available": "Batch processing" in tabs,
            "single_accepts_one_file": True,
            "frame_preview_toggle_available": hasattr(self, "live_check"),
            "manual_preview_frame_count_removed": not hasattr(self, "preview_frames_spin"),
            "full_source_timeline_available": hasattr(self, "timeline_scale"),
            "temporal_radius_setting": self.radius_var.get(),
            "overlay_canvas_available": hasattr(self.viewer, "canvas"),
            "hold_to_original_available": bool(self.viewer.canvas.bind("<ButtonPress-1>")),
            "pan_available": bool(self.viewer.canvas.bind("<B1-Motion>")),
            "wheel_zoom_available": bool(self.viewer.canvas.bind("<MouseWheel>")),
            "check_plan_button_removed": not hasattr(self, "plan_button"),
            "automatic_preflight_on_process": True,
            "preflight_log_preferred_lines": int(self.single_log.cget("height")),
            "preserved_track_controls_one_row": all(
                int(check.grid_info()["row"]) == 0 for check in self.single_track_checks
            ),
            "container_selector_available": hasattr(self, "container_combo"),
            "batch_max_files": self.batch_queue.maximum,
            "batch_columns": tuple(self.batch_tree["columns"]),
            "batch_log_visible": hasattr(self, "batch_log"),
            "batch_log_preferred_lines": int(self.batch_log.cget("height")),
            "batch_log_lower_left": self.batch_log_box.master is self.batch_denoise_settings_frame,
            "batch_selection_guide_compacted": self.batch_denoiser_rank_label.master is self.batch_denoise_values_row,
            "shared_settings_identity": self.denoiser_combo.cget("textvariable") == self.batch_denoiser_combo.cget("textvariable"),
            "ffmpeg_found": bool(capabilities and capabilities.ffmpeg_path),
            "ffprobe_found": bool(capabilities and capabilities.ffprobe_path),
            "vspipe_found": bool(capabilities and capabilities.vspipe_path),
            "denoise_capabilities": dict(capabilities.denoise_capabilities) if capabilities else {},
            "denoise_backends": dict(capabilities.denoise_backends) if capabilities else {},
            "denoiser_backend_status_visible": hasattr(self, "denoiser_backend_label"),
            "denoiser_rankings_available": all(
                1 <= denoiser_ranking(spec.identifier).quality_score <= 6
                and 1 <= denoiser_ranking(spec.identifier).speed_score <= 6
                for spec in DENOISER_SPECS
            ),
            "absolute_timeline_pointer_seek_available": bool(self.timeline_scale.bind("<ButtonPress-1>")),
            "same_frame_viewport_preservation_available": True,
            "default_denoiser": DENOISER_LABELS[self.denoiser_var.get()],
            "default_frame_preview_enabled": bool(self.frame_preview_var.get()),
            "settings_directory_name": "VideoDenoiseStudio",
        }

    def _settings_payload(self) -> dict[str, object]:
        return {
            **self.saved,
            "window_geometry": self.root.geometry(),
            "denoiser": DENOISER_LABELS.get(self.denoiser_var.get(), "vs_bm3d"),
            "denoise_strength": int(self.strength_var.get()),
            "denoise_temporal_radius": int(self.radius_var.get()),
            "frame_preview_enabled": bool(self.frame_preview_var.get()),
            "family": FAMILY_LABELS.get(self.family_var.get(), "ffv1"),
            "container": CONTAINER_LABELS.get(self.container_var.get(), "auto"),
            "bit_depth": int(self.bit_depth_var.get()),
            "ffv1_chroma_mode": FFV1_CHROMA_LABELS.get(self.ffv1_chroma_var.get(), "native"),
            "hardware_encode": bool(self.hardware_encode_var.get()),
            "av1_software_encoder": AV1_LABELS.get(self.av1_var.get(), "libaom"),
            "quality": int(self.quality_var.get()),
            "tune_grain": bool(self.tune_grain_var.get()),
            "copy_audio": bool(self.copy_audio_var.get()),
            "copy_subtitles": bool(self.copy_subtitles_var.get()),
            "copy_attachments": bool(self.copy_attachments_var.get()),
            "copy_data": bool(self.copy_data_var.get()),
            "copy_chapters": bool(self.copy_chapters_var.get()),
            "copy_metadata": bool(self.copy_metadata_var.get()),
            "batch_output_dir": self.batch_output_dir_var.get(),
            "batch_include_subfolders": bool(self.batch_subfolders_var.get()),
            "batch_continue_after_error": bool(self.batch_continue_var.get()),
        }

    def close(self, *, save_preferences: bool = True) -> None:
        if self.closing:
            return
        self.closing = True
        if self.drop_after_id is not None:
            try:
                self.root.after_cancel(self.drop_after_id)
            except Exception:
                pass
            self.drop_after_id = None
        if save_preferences:
            try:
                save_settings(self._settings_payload())
            except Exception:
                pass
        self.source_cancel_event.set()
        if self.processor:
            self.processor.cancel()
        if self.batch_runner:
            self.batch_runner.cancel()
        self._cancel_preview(clear_current=True)
        if self.drop_target:
            self.drop_target.close()
        self.root.destroy()
