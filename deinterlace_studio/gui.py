from __future__ import annotations

import os
import queue
import subprocess
import tempfile
import threading
import time
from dataclasses import replace
from pathlib import Path
from tkinter import BooleanVar, DoubleVar, StringVar, TclError, Tk, Toplevel, filedialog, messagebox
from tkinter import scrolledtext, ttk
from types import SimpleNamespace

from . import __version__
from .acceleration import (
    DIRECT_NVDEC_CODECS,
    FAST_EVERYDAY_GPU,
    HYBRID_QTGMC_GPU,
    MAXIMUM_FIDELITY,
    SPEED_MODES,
    apply_speed_mode,
    speed_mode_unavailable_reason,
)
from .automation import (
    DEINTERLACE_ARTIFACT_SUFFIXES,
    REPAIR_ARTIFACT_SUFFIXES,
    AutomaticRecoveryWorkflow,
    automatic_recovery_applies_to_backend,
    choose_available_artifact_path,
    completed_artifacts,
    storage_preflight,
    storage_summary,
)
from .batch import MAX_BATCH_FILES, BatchQueue
from .batch_runner import BatchRunOptions, BatchRunSummary, BatchRunner
from .capabilities import inspect_capabilities
from .compatibility import CompatibilityCopyRequest, MOVCompatibilityCopier
from .dependencies import (
    DependencyInstallCancelled,
    dependency_issues,
    install_latest_dependencies,
    managed_runtime_root,
    resolve_latest_releases,
)
from .denoise import (
    DEFAULT_DENOISER,
    DENOISER_BY_ID,
    DENOISER_LABELS,
    MAX_DENOISE_STRENGTH,
    MAX_TEMPORAL_RADIUS,
    MIN_DENOISE_STRENGTH,
    MIN_TEMPORAL_RADIUS,
    denoiser_backend_has_gpu,
    denoiser_backend_display,
)
from .idet import AnalysisCancelled, scan_idet
from .health import (
    SourceHealthCancelled,
    health_details,
    health_headline,
    health_matches_source,
    health_summary,
    scan_source_health,
    source_identity,
)
from .models import (
    SOURCE_REPAIR_REQUIRED_FAILURE,
    CapabilityReport,
    IDetReport,
    JobSettings,
    MediaProbe,
    ProcessingPlan,
    SourceHealthReport,
)
from .planner import MOV_AUDIO_CODECS, MOV_SUBTITLE_CODECS, build_plan
from .presets import PROFILES, profile_capability_error, select_profile, selectable_bit_depths
from .probe import ProbeError, probe_media
from .processor import JobProcessor
from .repair import RepairRequest, SourceRepairer, diagnosis_summary
from .rationals import fraction_text, rate_text
from .settings import load_settings, save_settings
from .windows_drop import FileDropUnavailable, WindowsFileDropTarget


ENGINE_LABELS = {
    "Automatic (recommended) — max-quality QTGMC for interlace; progressive passthrough": "auto",
    "Force QTGMC — same max-quality graph (manual override)": "vapoursynth_qtgmc",
    "FFmpeg BWDIF CPU — best built-in FFmpeg baseline": "ffmpeg_bwdif",
    "FFmpeg BWDIF CUDA — GPU deinterlacing (fast BWDIF path)": "ffmpeg_bwdif_cuda",
    "Progressive passthrough — encode only, no deinterlacing": "progressive",
}
FIELD_LABELS = {"Automatic from analysis": "auto", "Top field first (TFF)": "tff", "Bottom field first (BFF)": "bff"}
CADENCE_LABELS = {
    "Match source nominal frame rate — progressive output (same duration)": "frame_rate",
    "Preserve every temporal field — field-rate progressive output (same duration)": "field_rate",
}
ASPECT_LABELS = {
    "Preserve stored raster + exact source SAR/DAR (recommended)": "preserve",
    "Square pixels + exact source DAR (high-quality scale)": "square",
    "Manual DAR metadata, no scale/crop": "manual",
}
FAMILY_LABELS = {
    "FFV1 Intra 16-bit (lossless archival; not editor interchange)": "ffv1",
    "HEVC / H.265": "hevc",
    "AV1": "av1",
    "Apple ProRes 4444 XQ": "prores",
    "Avid DNxHR 444 (editor interchange)": "dnxhr",
}
FFV1_CHROMA_LABELS = {
    "Native source chroma at 16-bit (recommended; smallest/lossless)": "native",
    "Explicit 4:4:4 mastering (larger; derived chroma for 4:2:0/4:2:2 sources)": "444",
}
HW_DECODE_LABELS = {
    "Automatic hardware decode": "auto",
    "Off — software decode": "off",
    "NVIDIA CUDA decode": "cuda",
}
AV1_ENCODER_LABELS = {"libaom — reference quality / slowest": "libaom", "SVT-AV1 — preset 0": "svt"}
INTERLACE_DIAGNOSTIC_LABELS = {
    "ffmpeg9_source_audit": "Official 8.1→9.0 source audit",
    "bwdif_cpu": "CPU BWDIF",
    "bwdif_cuda": "CUDA BWDIF",
    "d3d12_custom": "D3D12 custom (8-bit NV12)",
    "d3d12_bob": "D3D12 bob (8-bit NV12)",
    "d3d12_custom_p010": "D3D12 custom (10-bit P010)",
    "d3d12_bob_p010": "D3D12 bob (10-bit P010)",
}
FINAL_PROCESSING_STAGE = "Final processing 3/3"

BACKEND_GPU_GUIDE_TEXT = """BACKEND CHOICE AND ACCELERATION ARE SEPARATE DECISIONS

Automatic routing is the recommended selection policy. It is not a lower-
quality deinterlacer: after measured IDet analysis it runs the same maximum-
quality QTGMC graph described below for TFF/BFF interlaced material, and it
passes measured progressive material through without deinterlacing. “Force
VapourSynth QTGMC” is a manual override for exceptional cases; it does not use
stronger QTGMC settings than Automatic routing.

QUALITY RANKING FOR GENUINELY INTERLACED VIDEO

#1  VapourSynth QTGMC — maximum-quality graph
    Multi-frame motion analysis, refined vectors, source matching, exact-field
    restoration, final temporal smoothing, and anti-comb cleanup in a 16-bit
    working pipeline. This is the best-quality path in the app.

#2 (tie)  FFmpeg BWDIF CPU and FFmpeg BWDIF CUDA
    Both use BWDIF. CUDA is the much faster GPU implementation, not a QTGMC
    accelerator or a quality upgrade. BWDIF usually gives good results, but it
    performs less temporal reconstruction than QTGMC and can lose more detail
    on difficult motion, fine diagonals, line twitter, and noisy sources.

    BWDIF is temporal as well as spatial: its normal algorithm checks the
    previous, current, and next field/frame neighborhoods when deciding how to
    interpolate missing lines. The app does not disable that temporal check.
    `mode=send_field` preserves every temporal field; `mode=send_frame` keeps
    the nominal frame rate; `deint=all` means process every selected input
    frame, not “turn temporal analysis off.” BWDIF is not motion-compensated
    multi-frame reconstruction or a temporal denoiser, so its temporal reach
    and sophistication remain below QTGMC.

Progressive passthrough is not a deinterlacer. It is the correct highest-
fidelity route for genuinely progressive video because it avoids unnecessary
processing.

THE EXACT MAXIMUM-QUALITY QTGMC GRAPH

Both QTGMC modes below use these same processing and quality parameters. The
accelerated mode changes only verified execution stages; it does not select a
faster QTGMC preset or weaken motion analysis.

• 16-bit working precision — input is promoted before QTGMC to prevent extra
  rounding during repeated filtering. It cannot recreate precision absent from
  an 8-bit source.

• analyze_force_tr=3 — motion vectors are always analyzed to temporal radius
  three, even if a later stage requests less. A larger radius supplies more
  distant motion references but increases MVTools work and memory use.

• analyze_blksize=16 — the first motion search uses 16×16 blocks. Larger blocks
  are faster and more resistant to noise; smaller blocks can follow local
  motion more precisely but cost more and can chase noise.

• analyze_overlap=2 — neighboring motion blocks overlap by half a block. This
  reduces block-edge artifacts while increasing the number of calculations.

• analyze_refine=2 — motion vectors are recalculated twice at successively
  smaller block sizes: 16→8→4. This is expensive but improves difficult local
  motion and is one reason this graph is not a “fast” QTGMC preset.

• prefilter_tr=2 — the motion-analysis prefilter uses a temporal binomial
  window with radius two. It stabilizes noisy input before vector estimation.

• basic_tr=2 — the main motion-compensated temporal smooth uses radius two.
  It suppresses shimmer/noise while retaining motion-compensated detail.

• final_tr=3 — the final motion-compensated smooth uses radius three. It adds
  temporal stability after reconstruction, at substantial CPU cost.

• source_match(tr=2, TWICE_REFINED) — two source-matching refinement passes use
  radius two to recover source detail and reduce over-smoothing, sharpening
  error, and halos introduced by interpolation.

• lossless(POSTSMOOTH, anti_comb=True) — exact source fields are restored after
  the final smooth, then anti-comb cleanup targets residual comb artifacts.
  “Lossless” here describes source-field restoration inside QTGMC; the final
  output is mathematically lossless only when a lossless codec such as FFV1 is
  selected.

• bob output — Preserve every temporal field produces one progressive frame per
  field at the field rate and preserves duration. Match source nominal frame
  rate selects one progressive frame per source interlaced frame instead.

QTGMC SPEED / QUALITY MODES

QTGMC maximum quality — CPU reference
    Automatic routing, the full parameter set above, CPU NNEDI3 interpolation,
    BestSource-managed software decode, and software HEVC/AV1 encoding. This is
    the conservative reference mode. FFV1, ProRes, and DNxHR remain CPU encodes
    because the app has no NVIDIA encoder for those codec families.

QTGMC maximum quality — accelerated where beneficial
    Automatic routing and the identical QTGMC parameters above. Verified
    Vulkan NNEDI3 moves only spatial interpolation to the GPU when its local
    graph passes. If that optional graph is unavailable or unstable, this mode
    stays available and safely keeps CPU NNEDI3. MVTools motion search/
    compensation, source matching, field restoration, and final smooth remain
    CPU work. If a CUDA temporal denoiser is active, NNEDI3 also stays on the
    CPU because simultaneous Vulkan and CUDA stages can contend and run slower.
    For HEVC or AV1 output, verified NVENC is enabled; that speeds compression
    but is lossy and does not change or accelerate QTGMC itself.

Fast GPU — BWDIF CUDA + HEVC NVENC
    This is the large-speed option, but it changes the deinterlacing algorithm
    from QTGMC to BWDIF and changes output to lossy 10-bit HEVC. It should not
    be described as hardware-accelerated QTGMC.

WHAT THE INDIVIDUAL HARDWARE CONTROLS CAN AND CANNOT DO

• Hardware decode affects direct FFmpeg paths. QTGMC uses BestSource inside
  VapourSynth, so the FFmpeg NVDEC control does not accelerate QTGMC input.
• Vulkan NNEDI3 can accelerate only QTGMC's spatial interpolation and is used
  only after its real graph passes the local capability test.
• CUDA temporal denoisers accelerate the optional post-deinterlace denoiser,
  not QTGMC motion analysis.
• NVENC accelerates HEVC/AV1 encoding only. It is visually lossy, whereas FFV1
  is mathematically lossless. NVIDIA does not provide FFV1, ProRes, or DNxHR
  encoders through this app.

NVENC MAXIMUM-QUALITY CONTRACT FOR HEVC AND AV1

When NVIDIA hardware encoding is enabled, capability discovery performs a real
bounded encode using the same complete option contract used by the job:
`preset=p7`, `tune=uhq`, VBR constant-quality control, `multipass=fullres`, a
32-frame lookahead at level 3, spatial and temporal adaptive quantization,
AQ strength 8, and middle B-reference mode. On FFmpeg builds that expose both
HQ and UHQ, UHQ is the stronger quality tune, so the app deliberately uses UHQ.
The output bit depth is enabled only after the bounded file decodes at that
true coded precision. These settings maximize NVENC quality, but a software
encoder can still compress more efficiently at the same file size.

The mode buttons below only propose visible setting changes and ask for
confirmation. They never start processing, and every control remains editable.
"""

# Backward-compatible exported names.  Both intentionally point to the one
# reusable guide because the GUI presents a single merged help surface.
BACKEND_GUIDE_TEXT = BACKEND_GPU_GUIDE_TEXT

CADENCE_GUIDE_TEXT = """Both cadence choices preserve the source running time.
A 60-minute input remains 60 minutes.

MATCH SOURCE NOMINAL FRAME RATE
Creates one progressive frame per interlaced source frame and keeps the source
timestamps. For example, 25 interlaced frames/s becomes 25p; 29.97 interlaced
frames/s becomes 29.97p. This is the shared Single/Batch default. It cannot
retain both distinct field-
time motion samples contained in each interlaced frame.

PRESERVE EVERY TEMPORAL FIELD — FIELD-RATE OUTPUT
Creates one progressive frame per temporal field. For example, 50 fields/s
becomes 50p and 59.94 fields/s becomes 59.94p. The output has twice the frame
count and twice the declared frame rate, so playback speed and duration remain unchanged.
Select this optional mode when retaining every distinct field-time motion sample
matters more than matching the source's nominal frame rate.

After analysis, the Output cadence control shows the measured source and target
rates instead of assuming a 25-fps source. Progressive passthrough keeps the
progressive source cadence because there are no interlaced fields to bob.
"""

SOURCE_TIMELINE_GUIDE_TEXT = """WHY QTGMC CAN REJECT A SOURCE BEFORE ENCODING

Every normal Probe/IDet analysis first performs a fast full-file scan of
compressed video packet timestamps and container diagnostics. This usually
takes seconds rather than a complete decoded-frame pass. A prominent source-
health banner reports no obvious damage, warning, repair needed, or
inconclusive. A measured material timestamp hole blocks QTGMC immediately.

The fast scan is deliberately not described as a complete decoded-picture
guarantee. When QTGMC Start is allowed, the app first establishes an exact
source/output frame contract. An unchanged packet-clean FFV1 source uses a fast
VSPipe graph-info check only when graph frames, cadence, and one-to-one packet
count all agree. Every other VapourSynth source uses a managed full decoded
fallback with live frame/percent/speed progress and immediate Cancel support.
No output encode begins until that check passes. FFmpeg-only BWDIF jobs skip the
redundant preflight and decode the timestamp-aware source once in the real job.

AUTOMATIC QTGMC RECOVERY — DEFAULT ENABLED

When the measured result is repair needed and the current plan resolves to
QTGMC, Automatic QTGMC recovery creates and fully validates a separate repair
copy, runs fresh health and IDet analysis on that copy, then starts the selected
valid output plan. The original source is never rewritten. Existing artifacts
are never silently replaced; numbered filenames are selected automatically. A
conservative storage preflight runs before the potentially large FFV1 chain.
Disable the checkbox to retain the manual Repair required… QTGMC workflow.
Warning and inconclusive results do not trigger automatic repair. A clear fast
scan never triggers repair by itself; healthy analysis still waits for the Start button. If Start's full
decoded preflight then confirms corrupt pictures or an unsafe timeline that the
packet scan could not see, Source health changes to SOURCE REPAIR NEEDED and the
same recovery chain begins automatically. Its third stage is labelled Final
processing 3/3 because the repair and re-analysis have already passed.

Explicit BWDIF CPU/CUDA bypasses automatic repair even when that preference is
checked. Start processes the original file directly through FFmpeg's timestamp-
aware pipeline. If Automatic falls back to BWDIF because QTGMC is unavailable,
it follows the same direct behavior. Manual Repair required… remains available
if you deliberately want a validated separate copy first.

QTGMC is delivered to FFmpeg through a constant-rate VapourSynth/Y4M stream.
For that path to preserve the complete source safely, the number of decoded
video frames must agree with the source video timeline. If they differ by more
than a small tolerance, Deinterlace Studio stops before creating a partial
encode instead of silently producing shortened or out-of-sync video.

WILL RETRYING THE SAME FILE WITH QTGMC WORK?

No. An unchanged file with the same measured timeline discrepancy will be
blocked every time QTGMC is selected. Changing the output codec does not repair
the source timeline.

WHAT BWDIF DOES

FFmpeg BWDIF CPU or CUDA can be selected in the Backend control. It remains in
FFmpeg's timestamp-aware pipeline and may process a file containing timestamp
gaps that the constant-rate QTGMC pipe cannot preserve. Selecting either BWDIF
backend explicitly skips the long automatic repair. BWDIF is a fallback, not a
repair: it cannot reconstruct missing or corrupt pictures, so inspect audio/
video sync and motion around any damaged region.

WHAT AUTOMATIC QTGMC RECOVERY AND THE “REPAIR SOURCE…” BUTTON DO

The button performs a complete decoded-frame/timestamp scan before choosing a
method. It never rewrites the selected source:

• If the pictures and timestamps are continuous and only container/duration
  metadata is wrong, it creates a separate stream-copy Matroska remux. That
  candidate is kept only if a full reopen/decode proves it is QTGMC-compatible.

• If decoded timestamps prove a real gap or the decoder reports recoverable
  corruption, it creates a separate FFV1 v3 intra lossless rescue master at the
  exact nominal source rate. Whole interlaced frames are repeated across missing
  time so the video timeline, audio sync, and playing duration remain aligned.
  FFV1 preserves the decoded pixels but can require many times the source size.

The rescue cannot reconstruct pictures that are absent from the file. Its audit
report states the measured gap and the net number of materialized/repeated frame
slots. A clean re-rip or replacement remains the only way to recover the real
missing scene. Every candidate uses a unique partial file and is promoted only
after full validation; the original remains the authoritative fallback.
"""

DENOISE_GUIDE_TEXT = """TEMPORAL DENOISE ORDER

For genuinely interlaced video, Deinterlace Studio always deinterlaces first
and denoises the resulting progressive frames second. Adjacent fields represent
different moments in time; denoising the woven fields first can mistake comb
teeth for detail or create motion ghosts. A measured progressive source is
denoised directly. When enabled, both stages run in the same background job and
share cancellation, validation, rollback, and audit reporting.

WHAT TEMPORAL RADIUS MEANS

Temporal Radius controls how far the denoiser may look on each side of the
current progressive frame. Radius N normally means up to N earlier frames + the
current frame + N later frames: a centered window of up to (2 × N) + 1 frames.
For example, radius 1 uses a 3-frame window, radius 2 uses 5 frames, and radius
6 uses 13 frames. At 50p those windows span about 0.04, 0.08, and 0.24 seconds
from first frame to last; at 25p they span about 0.08, 0.16, and 0.48 seconds.
Temporal Radius does not change frame rate, playing speed, or video duration.

A wider radius provides more observations of persistent random noise, but also
increases CPU/GPU work and RAM/VRAM use. Imperfect matching can smear moving
texture, leave trails, or erase real grain and small motion. Radius 1–2 is the
conservative starting range; radius 3 is a deliberate quality-first default for
V-BM3D. Inspect motion, faces, text, grain, and dark scenes before going wider.
This control belongs to the optional post-deinterlace denoiser and is separate
from QTGMC's internal temporal analysis.

Algorithm details: FFmpeg fftdnoiz is fixed at radius 1 (three frames).
FFmpeg atadenoise has a minimum five-frame window, so radius 1 and radius 2 both
resolve to five frames. V-BM3D, DFTTest2, MVTools, and temporal NLMeans use the
selected radius on each side. Plan & command reports the resolved frame window.

ALGORITHM AND QUALITY TRADE-OFFS — NO UNIVERSAL RANKING

• VapourSynth V-BM3D — the default reconstruction-oriented choice. It searches
  similar patches in a spatio-temporal neighborhood, creates a basic estimate,
  then performs a refined final pass. A verified CUDA implementation is used
  when available; the optimized CPU implementation is the safe fallback. It is
  normally the slowest choice and remains the best starting point for important
  masters when texture retention matters more than speed.

• VapourSynth DFTTest2 — temporal frequency-domain filtering. It is much faster
  than V-BM3D and can be excellent for fine, approximately stationary noise,
  but it is not motion compensated and excessive settings can soften texture or
  ring near edges. After an exact graph test, the app prefers NVIDIA NVRTC at
  every raster: clean five-repeat 1,000-frame trials found it fastest at
  720×576, 1280×720, and 1920×1080. Optimized CPU remains the safe fallback if
  NVRTC is absent or fails. Two repeated real-episode renders per backend were
  pixel-deterministic; CPU versus NVRTC measured 129.12-dB PSNR and 1.000000
  SSIM, consistent with negligible backend rounding rather than visible loss.

• VapourSynth MVTools degrain — block-motion estimation followed by weighted
  temporal averaging. It can handle coherent camera/object motion well, but bad
  vectors, occlusions, flashes, and scene boundaries can produce trails or
  displaced detail. Scene-change rejection remains enabled. This path is CPU.

• VapourSynth temporal NLMeans — non-local neighborhood matching across space
  and time. CUDA is preferred after a real graph test; ISPC CPU is the fallback.
  It is fast on the verified CUDA path and effective on random noise, but can
  give repeated texture or skin a waxy appearance at high strength.

• FFmpeg fftdnoiz — a limited three-frame 3D FFT/Wiener option. It is the more
  reconstruction-like native FFmpeg temporal choice, but its radius is fixed.

• FFmpeg atadenoise — adaptive temporal averaging with a configurable window.
  It is practical and usually faster, but has no motion compensation and can
  smear moving texture sooner as strength/radius rise.

FFmpeg's filters named bm3d, nlmeans, nlmeans_opencl, and nlmeans_vulkan are
spatial filters: their block/patch search is within each frame. GPU execution
does not turn them into multi-frame temporal denoisers, so they are intentionally
not presented in this Temporal denoiser list.

WHY THERE IS NO “HYBRID MOTION-COMPENSATED BM3D = ALWAYS BEST” PRESET

V-BM3D already performs temporal patch matching. Feeding a separate MVTools
degrain result as BM3D's reference replaces BM3D's own basic estimate; it does
not simply add perfect motion tracking. It can help a mild setting on a suitable
shot, but motion-vector errors and scene boundaries can also make it worse. The
app therefore keeps the separately validated V-BM3D and MVTools algorithms and
does not label an unproven hybrid as an absolute-quality upgrade.

No automatic denoiser can know which grain is artistically intentional. The
shared default remains two-pass V-BM3D at strength 4 and radius 3. For
irreplaceable material, compare a short denoise-off sample. Higher strength
removes more signal; higher radius uses more neighboring frames, memory, and
time.

FFmpeg denoisers can follow QTGMC, BWDIF CPU/CUDA, or progressive passthrough.
VapourSynth denoisers can follow QTGMC or process a measured progressive source.
They cannot follow FFmpeg BWDIF in the same correct-order pipeline; that pairing
is rejected before processing rather than silently changing algorithms or
creating an unrequested intermediate.
"""

SPEED_GPU_GUIDE_TEXT = BACKEND_GPU_GUIDE_TEXT


def _short_rate(value) -> str:
    rendered = f"{float(value):.3f}"
    return rendered.rstrip("0").rstrip(".")


def cadence_labels_for_media(media: MediaProbe | None) -> dict[str, str]:
    if media is None:
        return dict(CADENCE_LABELS)
    rate = media.video.r_frame_rate or media.video.avg_frame_rate
    if rate is None or rate <= 0:
        return dict(CADENCE_LABELS)
    source_rate = _short_rate(rate)
    field_rate = _short_rate(rate * 2)
    return {
        f"Match source nominal frame rate — {source_rate} interlaced frames/s → {source_rate}p (same duration)": "frame_rate",
        f"Preserve every temporal field — {field_rate} fields/s → {field_rate}p (same duration)": "field_rate",
    }


class ScrollFrame(ttk.Frame):
    def __init__(self, parent) -> None:
        super().__init__(parent)
        self.canvas = __import__("tkinter").Canvas(self, highlightthickness=0, borderwidth=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas, padding=(12, 10, 18, 18))
        self.window = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self._resize_after_id: str | None = None
        self._pending_viewport_height = 1
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.inner.bind("<Configure>", self._inner_configured)
        self.canvas.bind("<Configure>", self._canvas_configured)
        self.bind("<Destroy>", self._destroyed, add="+")

    def refresh_content_size(self) -> None:
        """Remeasure natural content, then fill only genuinely surplus height."""

        self._start_content_measurement(self.canvas.winfo_width(), self.canvas.winfo_height())

    def _start_content_measurement(self, width: int, viewport_height: int) -> None:
        self._pending_viewport_height = max(1, viewport_height)
        # Height zero restores the embedded frame's natural requested height.
        # Measuring while an explicit viewport height is still applied can make
        # weighted rows appear artificially compressible and hide real content.
        self.canvas.itemconfigure(self.window, width=max(1, width), height=0)
        if self._resize_after_id is None:
            self._resize_after_id = self.after_idle(self._finish_content_measurement)

    def _finish_content_measurement(self) -> None:
        self._resize_after_id = None
        try:
            natural_height = max(1, self.inner.winfo_reqheight())
            desired_height = self._pending_viewport_height if natural_height < self._pending_viewport_height else 0
            self.canvas.itemconfigure(self.window, height=desired_height)
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        except TclError:
            return

    def _inner_configured(self, _event) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _canvas_configured(self, event) -> None:
        self._start_content_measurement(event.width, event.height)

    def _destroyed(self, event) -> None:
        if event.widget is not self or self._resize_after_id is None:
            return
        try:
            self.after_cancel(self._resize_after_id)
        except TclError:
            pass
        self._resize_after_id = None


class DeinterlaceStudioApp:
    def __init__(
        self,
        root: Tk,
        *,
        initial_capabilities: CapabilityReport | None = None,
        settings_path: Path | None = None,
    ) -> None:
        self.root = root
        self.settings_path = settings_path
        self.persisted = load_settings(settings_path)
        self.capabilities = initial_capabilities
        self.media: MediaProbe | None = None
        self.analysis: IDetReport | None = None
        self.source_health: SourceHealthReport | None = None
        self.source_health_cache: dict[tuple[str, int, int], SourceHealthReport] = {}
        self.health_scan_media: MediaProbe | None = None
        self.plan: ProcessingPlan | None = None
        self.processor: JobProcessor | None = None
        self.batch_queue = BatchQueue()
        self.batch_runner: BatchRunner | None = None
        self.batch_last_output: Path | None = None
        self.batch_drag_row: str | None = None
        self.batch_dragging = False
        self.repairer: SourceRepairer | None = None
        self.compatibility_copier: MOVCompatibilityCopier | None = None
        self.auto_workflow: AutomaticRecoveryWorkflow | None = None
        self._restoring_automatic_settings = False
        self.analysis_cancel = threading.Event()
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.busy_kind: str | None = None
        self.started_at: float | None = None
        self.run_phase_detail = "Idle"
        self.capability_scan_started_at: float | None = None
        self.last_capability_scan_seconds: float | None = None
        self.last_completed_output: Path | None = None
        self.close_pending = False
        self.auto_dependency_offer = initial_capabilities is None
        self.dependency_offer_shown = False
        self.dependency_cancel = threading.Event()
        self.dependency_dialog: Toplevel | None = None
        self.dependency_doctor_text = None
        self.dependency_progress = None
        self.dependency_install_button = None
        self.dependency_cancel_button = None
        self.file_drop_target: WindowsFileDropTarget | None = None
        self.file_drop_diagnostic = "Native drag-and-drop has not been initialized."
        self.poll_after_id: str | None = None
        self.plan_refresh_after_id: str | None = None
        self.backend_gpu_dialog: Toplevel | None = None
        self.backend_gpu_text = None
        self.cadence_guide_dialog: Toplevel | None = None
        self.cadence_guide_text = None
        self.timeline_guide_dialog: Toplevel | None = None
        self.timeline_guide_text = None
        self.denoise_guide_dialog: Toplevel | None = None
        self.denoise_guide_text = None
        self.repair_dialog: Toplevel | None = None
        self.notebook: ttk.Notebook | None = None
        self.setup_tab = None
        self.batch_tab = None
        self.plan_tab = None
        self.log_tab = None
        self.batch_mutation_controls: list[object] = []
        self.batch_layout_mode = "wide"
        self.setup_layout_mode = "stacked"
        self.setup_wide_breakpoint = 0

        self.root.title(
            f"Deinterlace Studio {__version__} — quality-first FFmpeg + VapourSynth"
        )
        self.root.geometry(self.persisted.get("window_geometry") or "1180x900")
        self.root.minsize(980, 720)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.bind("<Destroy>", self._root_destroyed, add="+")
        self._configure_style()
        self._build_variables()
        self._build_menu()
        self._build_ui()
        self._install_file_drop()
        self._refresh_control_states()
        self._poll_events()
        if self.capabilities:
            self._show_capabilities()
        else:
            self._refresh_capabilities()

    def _configure_style(self) -> None:
        style = ttk.Style(self.root)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("Title.TLabel", font=("Segoe UI Semibold", 16))
        style.configure("Section.TLabelframe.Label", font=("Segoe UI Semibold", 10))
        style.configure("Status.TLabel", padding=(8, 5))
        style.configure("Good.TLabel", foreground="#146c2e")
        style.configure("Warn.TLabel", foreground="#9a5b00")
        style.configure("Error.TLabel", foreground="#a51d20")
        style.configure("HealthGood.TLabel", foreground="#146c2e", font=("Segoe UI Semibold", 10))
        style.configure("HealthWarn.TLabel", foreground="#9a5b00", font=("Segoe UI Semibold", 10))
        style.configure("HealthError.TLabel", foreground="#a51d20", font=("Segoe UI Semibold", 10))

    def _build_variables(self) -> None:
        self.input_var = StringVar()
        self.output_var = StringVar()
        self.engine_var = StringVar(value=next(iter(ENGINE_LABELS)))
        self.field_var = StringVar(value=next(iter(FIELD_LABELS)))
        self.cadence_label_values = dict(CADENCE_LABELS)
        self.cadence_var = StringVar(
            value=next(label for label, value in self.cadence_label_values.items() if value == "frame_rate")
        )
        self.progressive_override_var = BooleanVar(value=False)
        self.aspect_var = StringVar(value=next(iter(ASPECT_LABELS)))
        self.manual_dar_var = StringVar(value="16:9")
        self.family_var = StringVar(value=next(iter(FAMILY_LABELS)))
        self.bit_depth_var = StringVar(value="16")
        self.depth_hint_var = StringVar(value="")
        persisted_ffv1_chroma = str(self.persisted.get("ffv1_chroma_mode") or "native")
        if persisted_ffv1_chroma not in set(FFV1_CHROMA_LABELS.values()):
            persisted_ffv1_chroma = "native"
        self.ffv1_chroma_var = StringVar(
            value=next(label for label, value in FFV1_CHROMA_LABELS.items() if value == persisted_ffv1_chroma)
        )
        self.hardware_encode_var = BooleanVar(value=False)
        self.hardware_decode_var = StringVar(value=next(iter(HW_DECODE_LABELS)))
        self.vulkan_nnedi3_var = BooleanVar(value=bool(self.persisted.get("vulkan_nnedi3", False)))
        self.av1_encoder_var = StringVar(value=next(iter(AV1_ENCODER_LABELS)))
        self.quality_var = StringVar(value="14")
        self.tune_grain_var = BooleanVar(value=True)
        persisted_denoiser = str(self.persisted.get("denoiser") or DEFAULT_DENOISER)
        if persisted_denoiser not in DENOISER_BY_ID:
            persisted_denoiser = DEFAULT_DENOISER
        persisted_strength = int(self.persisted.get("denoise_strength", 4))
        persisted_radius = int(self.persisted.get("denoise_temporal_radius", 3))
        persisted_strength = min(MAX_DENOISE_STRENGTH, max(MIN_DENOISE_STRENGTH, persisted_strength))
        persisted_radius = min(MAX_TEMPORAL_RADIUS, max(MIN_TEMPORAL_RADIUS, persisted_radius))
        self.denoise_enabled_var = BooleanVar(value=bool(self.persisted.get("denoise_enabled", True)))
        self.denoiser_var = StringVar(
            value=next(label for label, value in DENOISER_LABELS.items() if value == persisted_denoiser)
        )
        self.denoise_strength_var = StringVar(value=str(persisted_strength))
        self.denoise_radius_var = StringVar(value=str(persisted_radius))
        self.copy_audio_var = BooleanVar(value=True)
        self.copy_subtitles_var = BooleanVar(value=True)
        self.copy_attachments_var = BooleanVar(value=True)
        self.copy_data_var = BooleanVar(value=False)
        self.copy_chapters_var = BooleanVar(value=True)
        self.copy_metadata_var = BooleanVar(value=True)
        self.status_var = StringVar(value="Discovering dependencies…")
        self.dependency_var = StringVar(value="FFmpeg / VapourSynth capability scan pending")
        self.analysis_progress_var = StringVar(value="No input analyzed")
        self.source_hint_var = StringVar(
            value="Browse for one video, or drop one file here; multi-file/folder drops are added to Batch."
        )
        self.source_health_var = StringVar(
            value="Source health: not checked — normal analysis includes a fast full-file timeline precheck."
        )
        self.run_detail_var = StringVar(value="Idle")
        self.progress_var = DoubleVar(value=0.0)
        self.repair_output_var = StringVar()
        self.repair_mode_var = StringVar(value="automatic")
        self.auto_repair_continue_var = BooleanVar(
            value=bool(self.persisted.get("automatic_repair_and_continue", True))
        )
        self.batch_output_dir_var = StringVar(
            value=str(self.persisted.get("batch_output_dir") or "")
        )
        self.batch_include_subfolders_var = BooleanVar(
            value=bool(self.persisted.get("batch_include_subfolders", False))
        )
        self.batch_continue_var = BooleanVar(
            value=bool(self.persisted.get("batch_continue_after_error", True))
        )
        self.batch_status_var = StringVar(
            value=f"Batch queue is empty · up to {MAX_BATCH_FILES} files"
        )

        self.input_var.trace_add("write", self._input_path_changed)

        for variable in (
            self.output_var,
            self.engine_var,
            self.field_var,
            self.cadence_var,
            self.aspect_var,
            self.family_var,
            self.bit_depth_var,
            self.ffv1_chroma_var,
            self.hardware_encode_var,
            self.hardware_decode_var,
            self.vulkan_nnedi3_var,
            self.av1_encoder_var,
            self.quality_var,
            self.tune_grain_var,
            self.denoise_enabled_var,
            self.denoiser_var,
            self.denoise_strength_var,
            self.denoise_radius_var,
            self.copy_audio_var,
            self.copy_subtitles_var,
            self.copy_attachments_var,
            self.copy_data_var,
            self.copy_chapters_var,
            self.copy_metadata_var,
            self.progressive_override_var,
            self.manual_dar_var,
        ):
            variable.trace_add("write", self._settings_changed)
        self.auto_repair_continue_var.trace_add("write", self._automatic_recovery_setting_changed)
        self.batch_output_dir_var.trace_add("write", self._batch_option_changed)
        self.batch_include_subfolders_var.trace_add("write", self._batch_option_changed)
        self.batch_continue_var.trace_add("write", self._batch_option_changed)

    def _build_menu(self) -> None:
        import tkinter as tk

        menu = tk.Menu(self.root)
        tools = tk.Menu(menu, tearoff=False)
        tools.add_command(label="Select FFmpeg and FFprobe…", command=self._select_ffmpeg)
        tools.add_command(label="Select VSPipe…", command=self._select_vspipe)
        tools.add_command(label="Refresh capability scan", command=self._refresh_capabilities)
        tools.add_separator()
        tools.add_command(label="Install/update app-local dependencies…", command=self._start_dependency_install)
        tools.add_command(label="Dependency doctor…", command=self._dependency_doctor)
        tools.add_separator()
        tools.add_command(
            label="Create fast MOV compatibility copy…",
            command=self._start_mov_compatibility_copy,
        )
        menu.add_cascade(label="Tools", menu=tools)
        help_menu = tk.Menu(menu, tearoff=False)
        help_menu.add_command(
            label="Backend, QTGMC parameters & GPU guide…",
            command=self._show_backend_gpu_guide,
        )
        help_menu.add_command(label="Temporal denoiser quality & ordering guide…", command=self._show_denoise_guide)
        help_menu.add_command(label="Output cadence guide…", command=self._show_cadence_guide)
        help_menu.add_command(label="QTGMC source timeline guide…", command=self._show_timeline_guide)
        help_menu.add_separator()
        help_menu.add_command(label="About", command=self._about)
        menu.add_cascade(label="Help", menu=help_menu)
        self.root.configure(menu=menu)

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=(12, 10, 12, 8))
        outer.pack(fill="both", expand=True)
        header = ttk.Frame(outer)
        header.pack(fill="x")
        ttk.Label(
            header,
            text=f"Deinterlace Studio {__version__}",
            style="Title.TLabel",
        ).pack(side="left")
        ttk.Label(
            header,
            text="Quality-first · FFmpeg BWDIF · VapourSynth QTGMC · safe partial promotion",
        ).pack(side="left", padx=(14, 0), pady=(5, 0))
        self.dependency_label = ttk.Label(outer, textvariable=self.dependency_var, style="Warn.TLabel", wraplength=1120)
        self.dependency_label.pack(fill="x", pady=(6, 8))

        notebook = ttk.Notebook(outer)
        self.notebook = notebook
        notebook.pack(fill="both", expand=True)
        setup_scroll = ScrollFrame(notebook)
        self.setup_scroll = setup_scroll
        self.setup_tab = setup_scroll
        notebook.add(setup_scroll, text="  Input, analysis & output  ")
        self._build_setup_tab(setup_scroll.inner)

        batch_scroll = ScrollFrame(notebook)
        self.batch_scroll = batch_scroll
        self.batch_tab = batch_scroll
        notebook.add(batch_scroll, text="  Batch processing  ")
        self._build_batch_tab(batch_scroll.inner)

        plan_tab = ttk.Frame(notebook, padding=10)
        self.plan_tab = plan_tab
        notebook.add(plan_tab, text="  Plan & command  ")
        self.plan_text = scrolledtext.ScrolledText(plan_tab, wrap="word", font=("Cascadia Mono", 9), undo=False)
        self.plan_text.pack(fill="both", expand=True)
        self.plan_text.insert("1.0", "Analyze an input to build a reproducible processing plan.")
        self.plan_text.configure(state="disabled")

        log_tab = ttk.Frame(notebook, padding=10)
        self.log_tab = log_tab
        notebook.add(log_tab, text="  Run log  ")
        self.log_text = scrolledtext.ScrolledText(log_tab, wrap="word", font=("Cascadia Mono", 9), undo=False)
        self.log_text.pack(fill="both", expand=True)
        self.log_text.configure(state="disabled")

        run_bar = ttk.Frame(outer)
        run_bar.pack(fill="x", pady=(9, 0))
        self.start_button = ttk.Button(
            run_bar,
            text="Start deinterlacing",
            command=self._start_requested,
            state="disabled",
        )
        self.start_button.pack(side="left")
        self.cancel_button = ttk.Button(run_bar, text="Cancel", command=self._cancel_active, state="disabled")
        self.cancel_button.pack(side="left", padx=(6, 0))
        self.open_button = ttk.Button(run_bar, text="Open output folder", command=self._open_output_folder)
        self.open_button.pack(side="left", padx=(6, 0))
        ttk.Progressbar(run_bar, variable=self.progress_var, maximum=100).pack(side="left", fill="x", expand=True, padx=12)
        # Keep the source-check frame count, measured speed, ETA, and total
        # elapsed time visible.  The former 28-character field silently clipped
        # those diagnostics and made a long full-decode fallback look stalled.
        ttk.Label(run_bar, textvariable=self.run_detail_var, width=68, anchor="e").pack(side="right")
        ttk.Label(outer, textvariable=self.status_var, style="Status.TLabel", anchor="w").pack(fill="x", pady=(4, 0))
        notebook.bind("<<NotebookTabChanged>>", self._notebook_tab_changed, add="+")

    def _build_batch_tab(self, parent: ttk.Frame) -> None:
        """Build the ordered queue and the shared settings surface."""

        parent.columnconfigure(0, weight=1)
        # Do not let a narrow window collapse the queue controls out of view.
        # The Batch tab itself scrolls, so retaining a usable queue is safer
        # than squeezing its footer or data rows to zero height.
        parent.rowconfigure(0, weight=3, minsize=270)
        parent.rowconfigure(1, weight=2)

        queue_box = ttk.LabelFrame(
            parent,
            text="1. Ordered batch queue — every row is preflighted before any long encode",
            style="Section.TLabelframe",
        )
        queue_box.grid(row=0, column=0, sticky="nsew", pady=(0, 8))
        queue_box.columnconfigure(0, weight=1)
        # A weighted row may be compressed below the Treeview's requested
        # height while the surrounding ScrollFrame is measuring a stacked
        # layout.  Preserve enough room for the headings plus several files;
        # the outer canvas can scroll to the shared settings below.
        queue_box.rowconfigure(1, weight=1, minsize=150)

        toolbar = ttk.Frame(queue_box)
        toolbar.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 5))
        self.batch_add_files_button = ttk.Button(toolbar, text="Add files…", command=self._batch_add_files)
        self.batch_add_files_button.pack(side="left")
        self.batch_add_folder_button = ttk.Button(toolbar, text="Add folder…", command=self._batch_add_folder)
        self.batch_add_folder_button.pack(side="left", padx=(5, 0))
        self.batch_include_subfolders_check = ttk.Checkbutton(
            toolbar,
            text="Include subfolders",
            variable=self.batch_include_subfolders_var,
        )
        self.batch_include_subfolders_check.pack(side="left", padx=(8, 14))
        self.batch_move_up_button = ttk.Button(toolbar, text="Move up", command=lambda: self._batch_move(-1))
        self.batch_move_up_button.pack(side="left")
        self.batch_move_down_button = ttk.Button(toolbar, text="Move down", command=lambda: self._batch_move(1))
        self.batch_move_down_button.pack(side="left", padx=(5, 0))
        self.batch_remove_button = ttk.Button(toolbar, text="Remove", command=self._batch_remove_selected)
        self.batch_remove_button.pack(side="left", padx=(12, 0))
        self.batch_clear_button = ttk.Button(toolbar, text="Clear", command=self._batch_clear)
        self.batch_clear_button.pack(side="left", padx=(5, 0))
        ttk.Label(
            toolbar,
            text=f"Drag files/folders here · Delete removes selected rows · maximum {MAX_BATCH_FILES}",
            style="Good.TLabel",
        ).pack(side="right")

        tree_frame = ttk.Frame(queue_box)
        tree_frame.grid(row=1, column=0, sticky="nsew", padx=8)
        tree_frame.columnconfigure(0, weight=1)
        tree_frame.rowconfigure(0, weight=1)
        columns = ("state", "analysis", "effective", "output", "progress")
        tree = ttk.Treeview(
            tree_frame,
            columns=columns,
            show="tree headings",
            selectmode="extended",
            height=9,
        )
        self.batch_tree = tree
        tree.heading("#0", text="Source file")
        tree.heading("state", text="State")
        tree.heading("analysis", text="Analysis")
        tree.heading("effective", text="Effective per-file settings / fallback")
        tree.heading("output", text="Output")
        tree.heading("progress", text="Progress / result")
        tree.column("#0", width=350, minwidth=180, stretch=True)
        tree.column("state", width=105, minwidth=90, stretch=False)
        tree.column("analysis", width=145, minwidth=110, stretch=False)
        tree.column("effective", width=390, minwidth=220, stretch=True)
        tree.column("output", width=330, minwidth=180, stretch=True)
        tree.column("progress", width=210, minwidth=130, stretch=True)
        vertical = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
        horizontal = ttk.Scrollbar(tree_frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        tree.grid(row=0, column=0, sticky="nsew")
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal.grid(row=1, column=0, sticky="ew")
        tree.bind("<Delete>", self._batch_delete_key, add="+")
        tree.bind("<ButtonPress-1>", self._batch_drag_start, add="+")
        tree.bind("<B1-Motion>", self._batch_drag_motion, add="+")
        tree.bind("<ButtonRelease-1>", self._batch_drag_end, add="+")

        queue_footer = ttk.Frame(queue_box)
        self.batch_queue_footer = queue_footer
        queue_footer.grid(row=2, column=0, sticky="ew", padx=8, pady=(5, 8))
        queue_footer.columnconfigure(1, weight=1)
        self.batch_auto_repair_check = ttk.Checkbutton(
            queue_footer,
            text="Automatically repair only QTGMC rows when strict decoded preflight proves damage",
            variable=self.auto_repair_continue_var,
        )
        self.batch_auto_repair_check.grid(row=0, column=0, sticky="w")
        self.batch_continue_check = ttk.Checkbutton(
            queue_footer,
            text="Continue compatible rows after an error",
            variable=self.batch_continue_var,
        )
        self.batch_continue_check.grid(row=0, column=1, sticky="w", padx=(16, 0))
        self.batch_status_label = ttk.Label(queue_footer, textvariable=self.batch_status_var, anchor="e")
        self.batch_status_label.grid(
            row=0, column=2, sticky="e", padx=(12, 0)
        )

        settings = ttk.Frame(parent)
        self.batch_settings_frame = settings
        settings.grid(row=1, column=0, sticky="nsew")
        settings.columnconfigure(0, weight=1, uniform="batch-settings")
        settings.columnconfigure(1, weight=1, uniform="batch-settings")

        deint = ttk.LabelFrame(
            settings,
            text="2. Shared deinterlace, geometry, and temporal denoise settings",
            style="Section.TLabelframe",
        )
        self.batch_deint_box = deint
        deint.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        deint.columnconfigure(1, weight=1)
        self.batch_engine_combo = self._combo_row(deint, 0, "Backend", self.engine_var, list(ENGINE_LABELS), 48)
        ttk.Button(deint, text="Backend / QTGMC / GPU guide…", command=self._show_backend_gpu_guide).grid(
            row=0, column=2, sticky="w", padx=(2, 8), pady=3
        )
        self.batch_field_combo = self._combo_row(deint, 1, "Field order", self.field_var, list(FIELD_LABELS), 34)
        self.batch_cadence_combo = self._combo_row(
            deint, 2, "Output cadence", self.cadence_var, list(self.cadence_label_values), 48
        )
        ttk.Button(deint, text="Cadence guide…", command=self._show_cadence_guide).grid(
            row=2, column=2, sticky="w", padx=(2, 8), pady=3
        )
        self.batch_hw_decode_combo = self._combo_row(
            deint, 3, "Hardware decode", self.hardware_decode_var, list(HW_DECODE_LABELS), 34
        )
        self.batch_vulkan_check = ttk.Checkbutton(
            deint,
            text="Use verified Vulkan NNEDI3 when compatible",
            variable=self.vulkan_nnedi3_var,
        )
        self.batch_vulkan_check.grid(row=4, column=1, sticky="w", padx=6, pady=3)
        self.batch_aspect_combo = self._combo_row(deint, 5, "Display aspect", self.aspect_var, list(ASPECT_LABELS), 48)
        manual = ttk.Frame(deint)
        manual.grid(row=6, column=1, sticky="w", padx=6, pady=3)
        ttk.Label(manual, text="Manual DAR").pack(side="left")
        self.batch_manual_dar_entry = ttk.Entry(manual, textvariable=self.manual_dar_var, width=12)
        self.batch_manual_dar_entry.pack(side="left", padx=(6, 0))
        self.batch_denoise_check = ttk.Checkbutton(
            deint,
            text="Enable temporal denoise after deinterlacing (shared default)",
            variable=self.denoise_enabled_var,
        )
        self.batch_denoise_check.grid(row=7, column=1, sticky="w", padx=6, pady=3)
        self.batch_denoiser_combo = self._combo_row(
            deint, 8, "Temporal denoiser", self.denoiser_var, list(DENOISER_LABELS), 48
        )
        ttk.Button(deint, text="Denoiser guide…", command=self._show_denoise_guide).grid(
            row=8, column=2, sticky="w", padx=(2, 8), pady=3
        )
        denoise = ttk.Frame(deint)
        denoise.grid(row=9, column=1, sticky="w", padx=6, pady=3)
        ttk.Label(denoise, text="Strength (1–10)").pack(side="left")
        self.batch_denoise_strength_spin = ttk.Spinbox(
            denoise,
            from_=MIN_DENOISE_STRENGTH,
            to=MAX_DENOISE_STRENGTH,
            textvariable=self.denoise_strength_var,
            width=5,
        )
        self.batch_denoise_strength_spin.pack(side="left", padx=(6, 12))
        self.batch_denoise_radius_label = ttk.Label(denoise, text="Temporal radius (1–6)")
        self.batch_denoise_radius_label.pack(side="left")
        self.batch_denoise_radius_spin = ttk.Spinbox(
            denoise,
            from_=MIN_TEMPORAL_RADIUS,
            to=MAX_TEMPORAL_RADIUS,
            textvariable=self.denoise_radius_var,
            width=5,
        )
        self.batch_denoise_radius_spin.pack(side="left", padx=(6, 0))
        self.batch_progressive_override_check = ttk.Checkbutton(
            deint,
            text="Allow forced deinterlacing when measured progressive (not recommended)",
            variable=self.progressive_override_var,
        )
        self.batch_progressive_override_check.grid(row=10, column=1, sticky="w", padx=6, pady=(3, 8))

        output = ttk.LabelFrame(
            settings,
            text="3. Shared output master, destination, and preserved tracks",
            style="Section.TLabelframe",
        )
        self.batch_output_box = output
        output.grid(row=0, column=1, sticky="nsew", padx=(4, 0))
        output.columnconfigure(1, weight=1)
        self.batch_family_combo = self._combo_row(output, 0, "Codec family", self.family_var, list(FAMILY_LABELS), 44)
        self.batch_depth_combo = self._combo_row(
            output, 1, "Pipeline bit depth", self.bit_depth_var, ["10", "12", "16"], 10, expand=False
        )
        self.batch_ffv1_chroma_combo = self._combo_row(
            output, 2, "FFV1 chroma storage", self.ffv1_chroma_var, list(FFV1_CHROMA_LABELS), 44
        )
        self.batch_hardware_encode_check = ttk.Checkbutton(
            output,
            text="Enable supported NVIDIA hardware encoder (P7/UHQ/full multipass)",
            variable=self.hardware_encode_var,
        )
        self.batch_hardware_encode_check.grid(row=3, column=0, columnspan=3, sticky="w", padx=8, pady=3)
        self.batch_av1_combo = self._combo_row(
            output, 4, "AV1 software encoder", self.av1_encoder_var, list(AV1_ENCODER_LABELS), 36
        )
        quality = ttk.Frame(output)
        quality.grid(row=5, column=0, columnspan=3, sticky="w", padx=8, pady=3)
        ttk.Label(quality, text="Quality (lower = larger/better)").pack(side="left")
        self.batch_quality_spin = ttk.Spinbox(quality, from_=0, to=40, textvariable=self.quality_var, width=6)
        self.batch_quality_spin.pack(side="left", padx=(6, 10))
        self.batch_grain_check = ttk.Checkbutton(
            quality,
            text="Preserve grain/detail",
            variable=self.tune_grain_var,
        )
        self.batch_grain_check.pack(side="left")
        tracks = ttk.Frame(output)
        tracks.grid(row=6, column=0, columnspan=3, sticky="w", padx=8, pady=4)
        self.batch_track_checks = []
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
            check.grid(row=index // 3, column=index % 3, sticky="w", padx=(0, 10), pady=1)
            self.batch_track_checks.append(check)
        ttk.Label(output, text="Output folder").grid(row=7, column=0, sticky="w", padx=8, pady=(5, 3))
        self.batch_output_dir_entry = ttk.Entry(output, textvariable=self.batch_output_dir_var)
        self.batch_output_dir_entry.grid(row=7, column=1, sticky="ew", padx=6, pady=(5, 3))
        destination_buttons = ttk.Frame(output)
        destination_buttons.grid(row=7, column=2, sticky="e", padx=(0, 8), pady=(5, 3))
        self.batch_output_browse_button = ttk.Button(
            destination_buttons,
            text="Browse…",
            command=self._batch_browse_output_dir,
        )
        self.batch_output_browse_button.pack(side="left")
        self.batch_output_clear_button = ttk.Button(
            destination_buttons,
            text="Beside sources",
            command=lambda: self.batch_output_dir_var.set(""),
        )
        self.batch_output_clear_button.pack(side="left", padx=(5, 0))
        ttk.Label(
            output,
            text=(
                "Blank output folder writes beside each source. Existing completed outputs are never silently "
                "overwritten; a unique compatible name is reserved per row."
            ),
            wraplength=650,
        ).grid(row=8, column=0, columnspan=3, sticky="w", padx=8, pady=(3, 8))

        self.batch_mutation_controls = [
            self.batch_add_files_button,
            self.batch_add_folder_button,
            self.batch_include_subfolders_check,
            self.batch_move_up_button,
            self.batch_move_down_button,
            self.batch_remove_button,
            self.batch_clear_button,
            self.batch_output_dir_entry,
            self.batch_output_browse_button,
            self.batch_output_clear_button,
            self.batch_continue_check,
        ]
        parent.bind("<Configure>", self._batch_parent_configured, add="+")
        self._apply_batch_layout(parent.winfo_width())

    def _batch_parent_configured(self, event) -> None:
        if event.widget is getattr(self.batch_scroll, "inner", None):
            self._apply_batch_layout(event.width)

    def _apply_batch_layout(self, available_width: int) -> None:
        if not hasattr(self, "batch_settings_frame"):
            return
        requested = "wide" if available_width >= self._required_wide_setup_width() else "stacked"
        if requested == self.batch_layout_mode:
            return
        if requested == "wide":
            self.batch_settings_frame.columnconfigure(0, weight=1, uniform="batch-settings")
            self.batch_settings_frame.columnconfigure(1, weight=1, uniform="batch-settings")
            self.batch_deint_box.grid_configure(row=0, column=0, sticky="nsew", padx=(0, 4), pady=0)
            self.batch_output_box.grid_configure(row=0, column=1, sticky="nsew", padx=(4, 0), pady=0)
            self.batch_auto_repair_check.grid_configure(row=0, column=0, columnspan=1, sticky="w")
            self.batch_continue_check.grid_configure(row=0, column=1, sticky="w", padx=(16, 0))
            self.batch_status_label.grid_configure(
                row=0, column=2, columnspan=1, sticky="e", padx=(12, 0), pady=0
            )
        else:
            self.batch_settings_frame.columnconfigure(0, weight=1, uniform="")
            self.batch_settings_frame.columnconfigure(1, weight=0, uniform="")
            self.batch_deint_box.grid_configure(row=0, column=0, sticky="nsew", padx=0, pady=(0, 8))
            self.batch_output_box.grid_configure(row=1, column=0, sticky="nsew", padx=0, pady=0)
            self.batch_auto_repair_check.grid_configure(row=0, column=0, columnspan=1, sticky="w")
            self.batch_continue_check.grid_configure(row=0, column=1, sticky="w", padx=(16, 0))
            self.batch_status_label.grid_configure(
                row=1, column=0, columnspan=3, sticky="w", padx=0, pady=(3, 0)
            )
        self.batch_layout_mode = requested
        self.batch_scroll.refresh_content_size()

    def _build_setup_tab(self, parent: ttk.Frame) -> None:
        self.setup_parent = parent
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
        source_box = ttk.LabelFrame(parent, text="1. Source and measured interlace analysis", style="Section.TLabelframe")
        self.source_box = source_box
        source_box.grid(row=0, column=0, columnspan=2, sticky="nsew", pady=(0, 9))
        source_box.columnconfigure(1, weight=1)
        source_box.rowconfigure(5, weight=1)
        ttk.Label(source_box, text="Source video").grid(row=0, column=0, sticky="w", padx=8, pady=(9, 4))
        self.input_entry = ttk.Entry(source_box, textvariable=self.input_var)
        self.input_entry.grid(row=0, column=1, sticky="ew", padx=6, pady=(9, 4))
        self.input_browse_button = ttk.Button(source_box, text="Browse…", command=self._browse_input)
        self.input_browse_button.grid(row=0, column=2, padx=(0, 8), pady=(9, 4))
        controls = ttk.Frame(source_box)
        controls.grid(row=1, column=0, columnspan=3, sticky="ew", padx=8, pady=4)
        controls.columnconfigure(2, weight=1)
        self.sample_button = ttk.Button(controls, text="Probe + distributed IDet samples", command=lambda: self._analyze("sampled"))
        self.sample_button.grid(row=0, column=0, sticky="w")
        self.full_button = ttk.Button(controls, text="Thorough full-file IDet scan", command=lambda: self._analyze("full"))
        self.full_button.grid(row=0, column=1, sticky="w", padx=(6, 0))
        self.analysis_progress_label = ttk.Label(controls, textvariable=self.analysis_progress_var, anchor="w")
        self.analysis_progress_label.grid(row=0, column=2, sticky="ew", padx=12)
        self.timeline_help_button = ttk.Button(controls, text="Timeline help…", command=self._show_timeline_guide)
        self.timeline_help_button.grid(row=0, column=4, sticky="e")
        self.repair_button = ttk.Button(controls, text="Repair source…", command=self._show_repair_dialog, state="disabled")
        self.repair_button.grid(row=0, column=3, sticky="e", padx=(0, 6))
        self.auto_repair_check = ttk.Checkbutton(
            source_box,
            text=(
                "Automatic QTGMC recovery: repair only for QTGMC; BWDIF runs the original directly"
            ),
            variable=self.auto_repair_continue_var,
        )
        self.auto_repair_check.grid(row=2, column=0, columnspan=3, sticky="w", padx=8, pady=(1, 3))
        self.source_hint_label = ttk.Label(
            source_box,
            textvariable=self.source_hint_var,
            style="Good.TLabel",
            wraplength=1000,
        )
        self.source_hint_label.grid(row=3, column=0, columnspan=3, sticky="w", padx=8, pady=(0, 3))
        self.source_health_label = ttk.Label(
            source_box,
            textvariable=self.source_health_var,
            style="HealthWarn.TLabel",
            wraplength=1500,
        )
        self.source_health_label.grid(row=4, column=0, columnspan=3, sticky="ew", padx=8, pady=(2, 2))
        self.summary_text = scrolledtext.ScrolledText(source_box, height=14, wrap="word", font=("Segoe UI", 9), undo=False)
        self.summary_text.grid(row=5, column=0, columnspan=3, sticky="nsew", padx=8, pady=(4, 9))
        self.summary_text.insert("1.0", "No source has been analyzed. Metadata alone is not used as proof of interlacing.")
        self.summary_text.configure(state="disabled")

        deint_box = ttk.LabelFrame(parent, text="2. Deinterlacing and display geometry", style="Section.TLabelframe")
        self.deint_box = deint_box
        deint_box.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 9))
        deint_box.columnconfigure(1, weight=1)
        self.engine_combo = self._combo_row(deint_box, 0, "Backend", self.engine_var, list(ENGINE_LABELS), width=66)
        backend_actions = ttk.Frame(deint_box)
        backend_actions.grid(row=0, column=2, sticky="w", padx=(2, 8), pady=3)
        self.backend_gpu_button = ttk.Button(
            backend_actions,
            text="Backend / QTGMC / GPU guide…",
            command=self._show_backend_gpu_guide,
        )
        self.backend_gpu_button.pack(side="left")
        self.field_combo = self._combo_row(deint_box, 1, "Field order", self.field_var, list(FIELD_LABELS), width=46)
        self.cadence_combo = self._combo_row(deint_box, 2, "Output cadence", self.cadence_var, list(CADENCE_LABELS), width=66)
        ttk.Button(deint_box, text="Cadence guide…", command=self._show_cadence_guide).grid(
            row=2, column=2, sticky="w", padx=(2, 8), pady=3
        )
        self.hw_decode_combo = self._combo_row(deint_box, 3, "Hardware decode", self.hardware_decode_var, list(HW_DECODE_LABELS), width=46)
        self.vulkan_nnedi3_check = ttk.Checkbutton(
            deint_box,
            text="Use verified Vulkan NNEDI3 interpolation (optional; QTGMC only; CPU remains default)",
            variable=self.vulkan_nnedi3_var,
        )
        self.vulkan_nnedi3_check.grid(row=4, column=1, sticky="w", padx=6, pady=(3, 2))
        self.aspect_combo = self._combo_row(deint_box, 5, "Display aspect", self.aspect_var, list(ASPECT_LABELS), width=66)
        manual_frame = ttk.Frame(deint_box)
        manual_frame.grid(row=6, column=1, sticky="w", padx=6, pady=3)
        ttk.Label(manual_frame, text="Manual DAR").pack(side="left")
        self.manual_dar_entry = ttk.Entry(manual_frame, textvariable=self.manual_dar_var, width=14)
        self.manual_dar_entry.pack(side="left", padx=(6, 0))
        ttk.Label(manual_frame, text="Example: 16:9 or 349:192").pack(side="left", padx=(8, 0))
        self.denoise_check = ttk.Checkbutton(
            deint_box,
            text="Enable temporal denoise after deinterlacing (same job; shared default)",
            variable=self.denoise_enabled_var,
        )
        self.denoise_check.grid(row=7, column=1, sticky="w", padx=6, pady=(5, 2))
        self.denoiser_combo = self._combo_row(
            deint_box,
            8,
            "Temporal denoiser",
            self.denoiser_var,
            list(DENOISER_LABELS),
            width=66,
        )
        ttk.Button(deint_box, text="Denoiser guide…", command=self._show_denoise_guide).grid(
            row=8, column=2, sticky="w", padx=(2, 8), pady=3
        )
        denoise_controls = ttk.Frame(deint_box)
        denoise_controls.grid(row=9, column=1, sticky="w", padx=6, pady=3)
        ttk.Label(denoise_controls, text="Strength (1–10)").pack(side="left")
        self.denoise_strength_spin = ttk.Spinbox(
            denoise_controls,
            from_=MIN_DENOISE_STRENGTH,
            to=MAX_DENOISE_STRENGTH,
            textvariable=self.denoise_strength_var,
            width=5,
        )
        self.denoise_strength_spin.pack(side="left", padx=(6, 14))
        self.denoise_radius_label = ttk.Label(denoise_controls, text="Temporal radius (1–6)")
        self.denoise_radius_label.pack(side="left")
        self.denoise_radius_spin = ttk.Spinbox(
            denoise_controls,
            from_=MIN_TEMPORAL_RADIUS,
            to=MAX_TEMPORAL_RADIUS,
            textvariable=self.denoise_radius_var,
            width=5,
        )
        self.denoise_radius_spin.pack(side="left", padx=(6, 0))
        self.progressive_override_check = ttk.Checkbutton(
            deint_box,
            text="Deliberately allow deinterlacing even when measured analysis says progressive",
            variable=self.progressive_override_var,
        )
        self.progressive_override_check.grid(row=10, column=1, sticky="w", padx=6, pady=(4, 9))

        output_box = ttk.LabelFrame(parent, text="3. Output master and preserved tracks", style="Section.TLabelframe")
        self.output_box = output_box
        output_box.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0, 9))
        output_box.columnconfigure(1, weight=1)
        self.family_combo = self._combo_row(output_box, 0, "Codec family", self.family_var, list(FAMILY_LABELS), width=58)
        editor_actions = ttk.Frame(output_box)
        editor_actions.grid(row=0, column=2, sticky="w", padx=(2, 8), pady=3)
        self.resolve_preset_button = ttk.Button(
            editor_actions,
            text="Resolve editor preset…",
            command=self._apply_resolve_editor_preset,
        )
        self.resolve_preset_button.pack(side="left")
        self.compatibility_copy_button = ttk.Button(
            editor_actions,
            text="Fast MOV copy…",
            command=self._start_mov_compatibility_copy,
        )
        self.compatibility_copy_button.pack(side="left", padx=(5, 0))
        self.depth_combo = self._combo_row(
            output_box,
            1,
            "Pipeline bit depth",
            self.bit_depth_var,
            ["10", "12", "16"],
            width=12,
            expand=False,
        )
        self.depth_hint_label = ttk.Label(
            output_box,
            textvariable=self.depth_hint_var,
            style="Warn.TLabel",
            wraplength=360,
        )
        self.depth_hint_label.grid(row=1, column=2, sticky="w", padx=(2, 8), pady=3)
        self.ffv1_chroma_combo = self._combo_row(
            output_box,
            2,
            "FFV1 chroma storage",
            self.ffv1_chroma_var,
            list(FFV1_CHROMA_LABELS),
            width=58,
        )
        encode_frame = ttk.Frame(output_box)
        encode_frame.grid(row=3, column=0, columnspan=3, sticky="w", padx=8, pady=3)
        self.hardware_encode_check = ttk.Checkbutton(
            encode_frame,
            text="Enable supported NVIDIA hardware encoder (P7/UHQ/full multipass)",
            variable=self.hardware_encode_var,
        )
        self.hardware_encode_check.pack(side="left")
        self.av1_combo = self._combo_row(output_box, 4, "AV1 software encoder", self.av1_encoder_var, list(AV1_ENCODER_LABELS), width=42)
        quality_frame = ttk.Frame(output_box)
        quality_frame.grid(row=5, column=0, columnspan=3, sticky="w", padx=8, pady=3)
        ttk.Label(quality_frame, text="Quality (lower = larger/better)").pack(side="left")
        self.quality_spin = ttk.Spinbox(quality_frame, from_=0, to=40, textvariable=self.quality_var, width=6)
        self.quality_spin.pack(side="left", padx=(6, 0))
        self.grain_check = ttk.Checkbutton(quality_frame, text="Preserve grain/detail (x265 tune grain)", variable=self.tune_grain_var)
        self.grain_check.pack(side="left", padx=(12, 0))
        track_frame = ttk.Frame(output_box)
        track_frame.grid(row=6, column=0, columnspan=3, sticky="w", padx=8, pady=5)
        self.track_checks: list[ttk.Checkbutton] = []
        for index, (text, variable) in enumerate((
            ("Audio", self.copy_audio_var),
            ("Subtitles", self.copy_subtitles_var),
            ("Attachments", self.copy_attachments_var),
            ("Data", self.copy_data_var),
            ("Chapters", self.copy_chapters_var),
            ("Metadata", self.copy_metadata_var),
        )):
            check = ttk.Checkbutton(track_frame, text=text, variable=variable)
            check.grid(row=0, column=index, sticky="w", padx=(0, 10))
            self.track_checks.append(check)
        ttk.Label(output_box, text="Output file").grid(row=7, column=0, sticky="w", padx=8, pady=(5, 9))
        self.output_entry = ttk.Entry(output_box, textvariable=self.output_var)
        self.output_entry.grid(row=7, column=1, sticky="ew", padx=6, pady=(5, 9))
        output_buttons = ttk.Frame(output_box)
        output_buttons.grid(row=7, column=2, sticky="e", padx=(0, 8), pady=(5, 9))
        ttk.Button(output_buttons, text="Suggest", command=self._suggest_output).pack(side="left")
        ttk.Button(output_buttons, text="Browse…", command=self._browse_output).pack(side="left", padx=(5, 0))

        preview_box = ttk.Frame(parent)
        self.preview_box = preview_box
        preview_box.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(0, 4))
        ttk.Button(preview_box, text="Build / refresh plan", command=self._refresh_plan).pack(side="left")
        ttk.Button(preview_box, text="Dependency doctor…", command=self._dependency_doctor).pack(side="left", padx=(6, 0))
        self.preview_note_label = ttk.Label(
            preview_box,
            text=(
                "Start remains blocked until Probe/IDet and the fast full-file health precheck complete. "
                "Automatic repair applies only to QTGMC; BWDIF processes the original directly with a damage warning."
            ),
            wraplength=760,
        )
        self.preview_note_label.pack(side="left", padx=12)
        parent.bind("<Configure>", self._setup_parent_configured, add="+")
        self._apply_setup_layout(parent.winfo_width())

    def _setup_parent_configured(self, event) -> None:
        if event.widget is self.setup_parent:
            self.dependency_label.configure(wraplength=max(600, event.width - 20))
            self.preview_note_label.configure(wraplength=max(420, event.width - 360))
            self.source_health_label.configure(wraplength=max(400, event.width - 80))
            self._apply_setup_layout(event.width)

    def _required_wide_setup_width(self) -> int:
        try:
            scaling = float(self.root.tk.call("tk", "scaling"))
        except (TypeError, ValueError):
            scaling = 1.333
        return max(1480, round(scaling * 850))

    def _configure_setup_density(self, *, wide: bool) -> None:
        widths = (
            (self.engine_combo, 50 if wide else 66),
            (self.field_combo, 34 if wide else 46),
            (self.cadence_combo, 50 if wide else 66),
            (self.hw_decode_combo, 34 if wide else 46),
            (self.aspect_combo, 50 if wide else 66),
            (self.denoiser_combo, 50 if wide else 66),
            (self.family_combo, 42 if wide else 58),
            (self.depth_combo, 10 if wide else 12),
            (self.ffv1_chroma_combo, 42 if wide else 58),
            (self.av1_combo, 32 if wide else 42),
        )
        for combo, width in widths:
            combo.configure(width=width)
        for index, check in enumerate(self.track_checks):
            if wide:
                row, column = divmod(index, 3)
            else:
                row, column = 0, index
            check.grid_configure(row=row, column=column, sticky="w", padx=(0, 10), pady=(0, 2) if wide else 0)

    def _apply_setup_layout(self, available_width: int) -> None:
        """Use compact side-by-side sections beyond a DPI-scaled wide breakpoint."""

        breakpoint = self._required_wide_setup_width()
        self.setup_wide_breakpoint = breakpoint
        requested_mode = "wide" if available_width >= breakpoint else "stacked"
        if requested_mode == self.setup_layout_mode:
            return

        if requested_mode == "wide":
            self._configure_setup_density(wide=True)
            self.setup_parent.columnconfigure(0, weight=1, uniform="setup-sections")
            self.setup_parent.columnconfigure(1, weight=1, uniform="setup-sections")
            self.deint_box.grid_configure(
                row=1,
                column=0,
                columnspan=1,
                sticky="nsew",
                padx=(0, 4),
                pady=(0, 9),
            )
            self.output_box.grid_configure(
                row=1,
                column=1,
                columnspan=1,
                sticky="nsew",
                padx=(4, 0),
                pady=(0, 9),
            )
            self.preview_box.grid_configure(row=2, column=0, columnspan=2, sticky="ew", padx=0, pady=(0, 4))
        else:
            self._configure_setup_density(wide=False)
            self.setup_parent.columnconfigure(0, weight=1, uniform="")
            self.setup_parent.columnconfigure(1, weight=0, uniform="")
            self.deint_box.grid_configure(
                row=1,
                column=0,
                columnspan=2,
                sticky="ew",
                padx=0,
                pady=(0, 9),
            )
            self.output_box.grid_configure(
                row=2,
                column=0,
                columnspan=2,
                sticky="ew",
                padx=0,
                pady=(0, 9),
            )
            self.preview_box.grid_configure(row=3, column=0, columnspan=2, sticky="ew", padx=0, pady=(0, 4))
        self.setup_layout_mode = requested_mode
        self.setup_scroll.refresh_content_size()

    def _combo_row(
        self,
        parent,
        row: int,
        label: str,
        variable: StringVar,
        values: list[str],
        width: int,
        *,
        expand: bool = True,
    ):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=8, pady=3)
        combo = ttk.Combobox(parent, textvariable=variable, values=values, state="readonly", width=width)
        combo.grid(row=row, column=1, sticky="ew" if expand else "w", padx=6, pady=3)
        return combo

    def _set_text(self, widget, text: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.mark_set("insert", "1.0")
        widget.yview_moveto(0.0)
        widget.configure(state="disabled")

    def _cadence_value(self) -> str:
        return self.cadence_label_values.get(self.cadence_var.get(), "frame_rate")

    def _cadence_label(self, value: str) -> str:
        return next(label for label, candidate in self.cadence_label_values.items() if candidate == value)

    def _update_cadence_labels(self) -> None:
        selected = self._cadence_value()
        self.cadence_label_values = cadence_labels_for_media(self.media)
        self.cadence_combo.configure(values=list(self.cadence_label_values))
        if hasattr(self, "batch_cadence_combo"):
            self.batch_cadence_combo.configure(values=list(self.cadence_label_values))
        self.cadence_var.set(self._cadence_label(selected))
        self._apply_setup_layout(self.setup_parent.winfo_width())

    def _append_log(self, text: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text.rstrip() + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _install_file_drop(self) -> None:
        try:
            target = WindowsFileDropTarget(
                self.root,
                self._handle_dropped_paths,
                error_callback=self._handle_drop_error,
            )
            target.install()
        except (FileDropUnavailable, OSError, RuntimeError) as exc:
            self.file_drop_target = None
            self.file_drop_diagnostic = f"Native drag-and-drop unavailable: {exc}"
            self.source_hint_var.set(
                "Browse for one video; use Batch > Add files/folder because native drag-and-drop is unavailable in this session."
            )
        else:
            self.file_drop_target = target
            self.file_drop_diagnostic = (
                "Ready for one Single-tab file or up to 99 Batch-tab files/folders dropped from Windows Explorer via "
                f"TkDND {target.provider_version} (TkinterDnD2 {target.package_version}); "
                f"{len(target.registrations)} GUI drop surfaces registered."
            )

    def _handle_drop_error(self, message: str) -> None:
        self.file_drop_diagnostic = message
        if not self.close_pending:
            messagebox.showerror("Drag-and-drop failed", message, parent=self.root)

    def _handle_dropped_paths(self, paths: tuple[Path, ...]) -> None:
        if self.busy_kind:
            messagebox.showwarning(
                "Files are locked while work is active",
                "Wait for the current operation to finish, or cancel it, before changing the source or batch queue.",
                parent=self.root,
            )
            return
        if not paths:
            return
        if len(paths) > 1 or self._is_batch_tab() or any(path.is_dir() for path in paths):
            self._batch_add_paths(paths, dropped=True)
            if self.notebook is not None and self.batch_tab is not None:
                self.notebook.select(self.batch_tab)
            return
        self._select_input_path(paths[0], dropped=True)

    def _is_batch_tab(self) -> bool:
        if self.notebook is None or self.batch_tab is None:
            return False
        try:
            return self.notebook.select() == str(self.batch_tab)
        except TclError:
            return False

    def _notebook_tab_changed(self, _event=None) -> None:
        self._update_start_button_state()
        if self._is_batch_tab():
            self.status_var.set(self.batch_status_var.get())

    def _batch_option_changed(self, *_args) -> None:
        if not hasattr(self, "batch_queue"):
            return
        self.persisted["batch_output_dir"] = self.batch_output_dir_var.get().strip()
        self.persisted["batch_include_subfolders"] = self.batch_include_subfolders_var.get()
        self.persisted["batch_continue_after_error"] = self.batch_continue_var.get()
        if not self.busy_kind:
            for record in self.batch_queue.records:
                record.reset_plan(retain_analysis=True)
            if hasattr(self, "batch_tree"):
                self._refresh_batch_tree()

    def _batch_add_files(self) -> None:
        if self.busy_kind:
            return
        initial = self.persisted.get("last_input_dir") or os.getcwd()
        chosen = filedialog.askopenfilenames(
            title=f"Add video files to batch (maximum {MAX_BATCH_FILES})",
            initialdir=initial,
            filetypes=[
                (
                    "Video files",
                    "*.3gp *.asf *.avi *.divx *.flv *.m2t *.m2ts *.m4v *.mkv *.mov *.mp4 "
                    "*.mpeg *.mpg *.mts *.mxf *.ogm *.rm *.rmvb *.ts *.vob *.webm *.wmv",
                ),
                ("All files", "*.*"),
            ],
        )
        if chosen:
            self._batch_add_paths(tuple(Path(path) for path in chosen))

    def _batch_add_folder(self) -> None:
        if self.busy_kind:
            return
        initial = self.persisted.get("last_input_dir") or os.getcwd()
        chosen = filedialog.askdirectory(title="Add videos from folder", initialdir=initial)
        if chosen:
            self._batch_add_paths((Path(chosen),))

    def _batch_add_paths(self, paths: tuple[Path, ...], *, dropped: bool = False) -> None:
        result = self.batch_queue.add_paths(
            paths,
            include_subfolders=self.batch_include_subfolders_var.get(),
        )
        if result.added:
            self.persisted["last_input_dir"] = str(result.added[-1].source_path.parent)
        self._refresh_batch_tree()
        pieces = [f"Added {len(result.added)}"]
        if result.duplicates:
            pieces.append(f"ignored {len(result.duplicates)} duplicate(s)")
        if result.unsupported:
            pieces.append(f"ignored {len(result.unsupported)} unsupported file(s)")
        if result.missing:
            pieces.append(f"ignored {len(result.missing)} missing path(s)")
        if result.capacity_rejected:
            pieces.append(f"rejected {len(result.capacity_rejected)} above the {MAX_BATCH_FILES}-file limit")
        message = " · ".join(pieces) + f" · {len(self.batch_queue)}/{MAX_BATCH_FILES} queued"
        self.batch_status_var.set(message)
        self.status_var.set(("Drop accepted · " if dropped else "") + message)
        if result.unsupported or result.missing or result.capacity_rejected:
            self._append_log("Batch add: " + message)
        self._update_start_button_state()

    def _refresh_batch_tree(self) -> None:
        if not hasattr(self, "batch_tree"):
            return
        selected = set(self.batch_tree.selection())
        authoritative = {record.identifier for record in self.batch_queue.records}
        for identifier in self.batch_tree.get_children():
            if identifier not in authoritative:
                self.batch_tree.delete(identifier)
        duplicate_names: dict[str, int] = {}
        for record in self.batch_queue.records:
            key = record.source_path.name.casefold()
            duplicate_names[key] = duplicate_names.get(key, 0) + 1
        for index, record in enumerate(self.batch_queue.records):
            source_text = record.source_path.name
            if duplicate_names[record.source_path.name.casefold()] > 1:
                source_text = f"{source_text} — {record.source_path.parent}"
            values = (
                record.state,
                record.analysis_text,
                record.effective_text,
                record.output_path.name if record.output_path else "Pending",
                record.progress_text,
            )
            if self.batch_tree.exists(record.identifier):
                self.batch_tree.item(record.identifier, text=source_text, values=values)
                self.batch_tree.move(record.identifier, "", index)
            else:
                self.batch_tree.insert(
                    "",
                    index,
                    iid=record.identifier,
                    text=source_text,
                    values=values,
                )
        retained = tuple(identifier for identifier in selected if identifier in authoritative)
        if retained:
            self.batch_tree.selection_set(retained)
        if not self.busy_kind:
            if self.batch_queue.records:
                self.batch_status_var.set(
                    f"{len(self.batch_queue)}/{MAX_BATCH_FILES} queued · settings apply to every row; fallbacks are shown explicitly"
                )
            else:
                self.batch_status_var.set(f"Batch queue is empty · up to {MAX_BATCH_FILES} files")

    def _batch_selected_ids(self) -> tuple[str, ...]:
        return tuple(str(identifier) for identifier in self.batch_tree.selection())

    def _batch_move(self, direction: int) -> None:
        if self.busy_kind:
            return
        selected = self._batch_selected_ids()
        if not selected:
            return
        self.batch_queue.move(selected, direction)
        self._refresh_batch_tree()
        self.batch_tree.selection_set(selected)
        self.batch_tree.see(selected[0])

    def _batch_remove_selected(self) -> None:
        if self.busy_kind:
            return
        selected = self._batch_selected_ids()
        if not selected:
            return
        removed = self.batch_queue.remove(selected)
        self._refresh_batch_tree()
        self.batch_status_var.set(
            f"Removed {len(removed)} row(s) · {len(self.batch_queue)}/{MAX_BATCH_FILES} remain"
        )
        self.status_var.set(self.batch_status_var.get())
        self._update_start_button_state()

    def _batch_clear(self) -> None:
        if self.busy_kind or not self.batch_queue.records:
            return
        removed = self.batch_queue.clear()
        self._refresh_batch_tree()
        self.batch_status_var.set(f"Cleared {len(removed)} row(s) · batch queue is empty")
        self.status_var.set(self.batch_status_var.get())
        self._update_start_button_state()

    def _batch_delete_key(self, _event=None):
        self._batch_remove_selected()
        return "break"

    def _batch_drag_start(self, event) -> None:
        if self.busy_kind:
            self.batch_drag_row = None
            return
        row = self.batch_tree.identify_row(event.y)
        self.batch_drag_row = row or None
        self.batch_dragging = False

    def _batch_drag_motion(self, event) -> None:
        source = self.batch_drag_row
        if self.busy_kind or not source or not self.batch_tree.exists(source):
            return
        target = self.batch_tree.identify_row(event.y)
        if not target or target == source:
            return
        children = list(self.batch_tree.get_children())
        target_index = children.index(target)
        self.batch_tree.move(source, "", target_index)
        self.batch_tree.selection_set(source)
        self.batch_tree.see(source)
        self.batch_dragging = True

    def _batch_drag_end(self, _event=None) -> None:
        if self.batch_dragging and not self.busy_kind:
            try:
                self.batch_queue.reorder(self.batch_tree.get_children())
            except ValueError as exc:
                self._refresh_batch_tree()
                self.status_var.set(f"Batch reorder was rejected safely: {exc}")
            else:
                self.batch_status_var.set(
                    f"Queue order updated · {len(self.batch_queue)}/{MAX_BATCH_FILES} rows"
                )
                self.status_var.set(self.batch_status_var.get())
        self.batch_drag_row = None
        self.batch_dragging = False

    def _batch_browse_output_dir(self) -> None:
        if self.busy_kind:
            return
        initial = self.batch_output_dir_var.get().strip() or self.persisted.get("last_output_dir") or os.getcwd()
        chosen = filedialog.askdirectory(title="Select batch output folder", initialdir=initial)
        if chosen:
            self.batch_output_dir_var.set(chosen)
            self.persisted["last_output_dir"] = chosen

    def _select_input_path(
        self,
        path: Path,
        *,
        dropped: bool = False,
        suggest_output: bool = True,
    ) -> bool:
        path = Path(path)
        if not path.is_file():
            messagebox.showerror(
                "Invalid source",
                f"The selected source is not an existing file:\n{path}",
                parent=self.root,
            )
            return False
        self.input_var.set(str(path))
        self.persisted["last_input_dir"] = str(path.parent)
        if suggest_output:
            self._suggest_output()
        if dropped:
            self.status_var.set("Video received by drag-and-drop. Run Probe + distributed IDet samples.")
        return True

    def _browse_input(self) -> None:
        if self.busy_kind:
            return
        initial = self.persisted.get("last_input_dir") or os.getcwd()
        chosen = filedialog.askopenfilename(
            title="Select source video",
            initialdir=initial,
            filetypes=[("Video files", "*.mkv *.mov *.mp4 *.m2ts *.mts *.ts *.avi *.mpg *.mpeg"), ("All files", "*.*")],
        )
        if chosen:
            self._select_input_path(Path(chosen))

    def _show_text_guide(self, *, title: str, text: str, dialog_attr: str, text_attr: str) -> None:
        existing = getattr(self, dialog_attr)
        if existing is not None and existing.winfo_exists():
            existing.deiconify()
            existing.lift()
            existing.focus_force()
            return

        dialog = Toplevel(self.root)
        dialog.title(title)
        dialog.geometry("940x650")
        dialog.minsize(720, 500)
        dialog.transient(self.root)
        frame = ttk.Frame(dialog, padding=12)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        guide = scrolledtext.ScrolledText(frame, wrap="word", font=("Segoe UI", 10), undo=False)
        guide.grid(row=0, column=0, sticky="nsew")
        guide.insert("1.0", text.strip() + "\n")
        guide.configure(state="disabled")
        setattr(self, dialog_attr, dialog)
        setattr(self, text_attr, guide)

        def close_dialog() -> None:
            setattr(self, dialog_attr, None)
            setattr(self, text_attr, None)
            dialog.destroy()

        ttk.Button(frame, text="Close", command=close_dialog).grid(row=1, column=0, sticky="e", pady=(10, 0))
        dialog.protocol("WM_DELETE_WINDOW", close_dialog)

    def _close_backend_gpu_guide(self) -> None:
        dialog = self.backend_gpu_dialog
        self.backend_gpu_dialog = None
        self.backend_gpu_text = None
        if dialog is not None and dialog.winfo_exists():
            dialog.destroy()

    def _show_backend_gpu_guide(self) -> None:
        if self.backend_gpu_dialog is not None and self.backend_gpu_dialog.winfo_exists():
            self.backend_gpu_dialog.deiconify()
            self.backend_gpu_dialog.lift()
            self.backend_gpu_dialog.focus_force()
            return

        dialog = Toplevel(self.root)
        self.backend_gpu_dialog = dialog
        dialog.title("Backends, QTGMC parameters, GPU acceleration, and quality modes")
        dialog.geometry("1040x780")
        dialog.minsize(820, 620)
        dialog.transient(self.root)
        frame = ttk.Frame(dialog, padding=12)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        guide = scrolledtext.ScrolledText(frame, wrap="word", font=("Segoe UI", 10), undo=False, height=24)
        guide.grid(row=0, column=0, sticky="nsew")
        guide.insert("1.0", BACKEND_GPU_GUIDE_TEXT.strip() + "\n")
        guide.configure(state="disabled")
        self.backend_gpu_text = guide

        modes = ttk.Frame(frame)
        modes.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        modes.columnconfigure(0, weight=1)
        classification = self.analysis.classification if self.analysis else None
        current_settings = self._collect_settings()
        for row, mode in enumerate(SPEED_MODES):
            box = ttk.LabelFrame(
                modes,
                text=f"{mode.label} · quality: {mode.quality} · speed: {mode.speed}",
            )
            box.grid(row=row, column=0, sticky="ew", pady=(0, 7))
            box.columnconfigure(0, weight=1)
            ttk.Label(box, text=mode.description, wraplength=790, justify="left").grid(
                row=0, column=0, sticky="w", padx=8, pady=7
            )
            unavailable = (
                "Capability scan is not ready."
                if self.capabilities is None
                else speed_mode_unavailable_reason(
                    mode.identifier,
                    self.capabilities,
                    source_classification=classification,
                    settings=current_settings,
                )
            )
            ttk.Label(
                box,
                text=("Ready on this system" if unavailable is None else "Unavailable: " + unavailable),
                style="Good.TLabel" if unavailable is None else "Warn.TLabel",
                wraplength=780,
            ).grid(row=1, column=0, sticky="w", padx=8, pady=(0, 7))
            ttk.Button(
                box,
                text="Review and apply…",
                command=lambda identifier=mode.identifier: self._apply_speed_mode(identifier),
                state="normal" if unavailable is None and not self.busy_kind else "disabled",
            ).grid(row=0, column=1, rowspan=2, sticky="e", padx=8, pady=7)

        ttk.Button(frame, text="Close", command=self._close_backend_gpu_guide).grid(
            row=2, column=0, sticky="e", pady=(3, 0)
        )
        dialog.protocol("WM_DELETE_WINDOW", self._close_backend_gpu_guide)

    # Compatibility aliases for scripts from earlier releases.  The GUI has
    # only one button and one Help-menu entry for this merged guide.
    def _show_backend_guide(self) -> None:
        self._show_backend_gpu_guide()

    def _show_speed_gpu_guide(self) -> None:
        self._show_backend_gpu_guide()

    def _close_speed_gpu_guide(self) -> None:
        self._close_backend_gpu_guide()

    def _apply_speed_mode(self, identifier: str) -> None:
        if self.busy_kind or self.capabilities is None:
            return
        classification = self.analysis.classification if self.analysis else None
        try:
            applied = apply_speed_mode(
                self._collect_settings(),
                identifier,
                self.capabilities,
                source_classification=classification,
                source_codec=self.media.video.codec_name if self.media else None,
                source_width=self.media.video.width if self.media else None,
                source_height=self.media.video.height if self.media else None,
            )
        except ValueError as exc:
            messagebox.showerror("Mode unavailable", str(exc), parent=self.backend_gpu_dialog or self.root)
            return
        details = "\n".join(f"• {item}" for item in applied.changes)
        cautions = "\n".join(f"• {item}" for item in applied.cautions)
        prompt = (
            f"Apply {applied.mode.label}?\n\nSETTINGS THAT WILL CHANGE\n{details}"
            f"\n\nQUALITY / WORKFLOW NOTES\n{cautions}"
            "\n\nNo processing will start. You can edit every control afterward."
        )
        if not messagebox.askyesno(
            "Confirm speed / quality settings",
            prompt,
            parent=self.backend_gpu_dialog or self.root,
        ):
            return
        self._restore_automatic_settings(applied.settings)
        self.run_detail_var.set(applied.mode.label)
        self.status_var.set(
            f"Applied {applied.mode.label}; review the Plan & command tab, then Start when ready."
        )
        if self.media and self.analysis:
            self._refresh_plan()
        self._close_backend_gpu_guide()
        messagebox.showinfo(
            "Speed / quality mode applied",
            f"{applied.mode.label} is now reflected in the visible controls.\n\nNo job was started.",
            parent=self.root,
        )

    def _show_denoise_guide(self) -> None:
        self._show_text_guide(
            title="Temporal denoiser quality, speed, and ordering guide",
            text=DENOISE_GUIDE_TEXT,
            dialog_attr="denoise_guide_dialog",
            text_attr="denoise_guide_text",
        )

    def _show_cadence_guide(self) -> None:
        self._show_text_guide(
            title="Output cadence and playback duration",
            text=CADENCE_GUIDE_TEXT,
            dialog_attr="cadence_guide_dialog",
            text_attr="cadence_guide_text",
        )

    def _show_timeline_guide(self) -> None:
        self._show_text_guide(
            title="QTGMC source timeline and repair guidance",
            text=SOURCE_TIMELINE_GUIDE_TEXT,
            dialog_attr="timeline_guide_dialog",
            text_attr="timeline_guide_text",
        )

    def _suggest_repair_output(self, source: Path) -> Path:
        return source.with_name(f"{source.stem}.qtgmc-repair.mkv")

    def _close_repair_dialog(self) -> None:
        dialog = self.repair_dialog
        self.repair_dialog = None
        if dialog is not None and dialog.winfo_exists():
            try:
                dialog.grab_release()
            except TclError:
                pass
            dialog.destroy()

    def _show_repair_dialog(self) -> None:
        if self.busy_kind:
            return
        if not self.capabilities or not self.capabilities.ffmpeg_path or not self.capabilities.ffprobe_path:
            messagebox.showerror(
                "Dependencies unavailable",
                "FFmpeg and FFprobe must pass the capability scan before source repair can run.",
                parent=self.root,
            )
            return
        source = Path(self.input_var.get().strip())
        if not source.is_file():
            messagebox.showerror("Missing source", f"Select an existing video first:\n{source}", parent=self.root)
            return
        if self.repair_dialog is not None and self.repair_dialog.winfo_exists():
            self.repair_dialog.deiconify()
            self.repair_dialog.lift()
            self.repair_dialog.focus_force()
            return

        self.repair_output_var.set(str(self._suggest_repair_output(source)))
        self.repair_mode_var.set("automatic")
        dialog = Toplevel(self.root)
        self.repair_dialog = dialog
        dialog.title("Repair source for safe QTGMC processing")
        dialog.geometry("980x620")
        dialog.minsize(820, 580)
        dialog.transient(self.root)
        frame = ttk.Frame(dialog, padding=14)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(0, weight=1)
        ttk.Label(frame, text="Repair a separate copy — the selected source is never changed", style="Section.TLabelframe.Label").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            frame,
            text=str(source),
            wraplength=900,
        ).grid(row=1, column=0, sticky="ew", pady=(5, 10))
        ttk.Label(
            frame,
            text=(
                "The complete video is decoded once to locate timestamp holes and decoder corruption. "
                "A metadata-only problem can be repaired by lossless stream-copy remuxing. A real gap uses "
                "an FFV1 v3 intra lossless rescue that repeats whole interlaced frames across unavailable time "
                "so audio sync and duration stay aligned. It does not recreate the missing scene."
            ),
            wraplength=900,
        ).grid(row=2, column=0, sticky="ew", pady=(0, 10))

        methods = ttk.LabelFrame(frame, text="Method")
        methods.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        methods.columnconfigure(0, weight=1)
        choices = (
            (
                "automatic",
                "Automatic safest validated repair (recommended)",
                "Diagnose first. Use stream-copy only for a proven metadata/container-only fault; otherwise create the FFV1 rescue.",
            ),
            (
                "remux_only",
                "Lossless stream-copy remux only",
                "Fast and small, but it will be rejected if timestamps contain a real gap or the compressed video remains corrupt.",
            ),
            (
                "rescue_only",
                "FFV1 lossless QTGMC rescue",
                "Rebuild a constant-rate decoded timeline directly. This is typically many times larger than the source and uses software decoding.",
            ),
        )
        for index, (value, title, description) in enumerate(choices):
            ttk.Radiobutton(methods, text=title, variable=self.repair_mode_var, value=value).grid(
                row=index * 2, column=0, sticky="w", padx=10, pady=(8 if index == 0 else 4, 0)
            )
            ttk.Label(methods, text=description, wraplength=850).grid(
                row=index * 2 + 1,
                column=0,
                sticky="ew",
                padx=(34, 10),
                pady=(1, 6 if index == len(choices) - 1 else 2),
            )

        output_frame = ttk.LabelFrame(frame, text="Separate repair output")
        output_frame.grid(row=4, column=0, sticky="ew", pady=(0, 10))
        output_frame.columnconfigure(0, weight=1)
        ttk.Entry(output_frame, textvariable=self.repair_output_var).grid(
            row=0, column=0, sticky="ew", padx=(10, 6), pady=10
        )
        ttk.Button(output_frame, text="Browse…", command=self._browse_repair_output).grid(
            row=0, column=1, padx=(0, 10), pady=10
        )
        ttk.Label(
            frame,
            text=(
                "The repair runs to a unique partial file and fully decodes/validates it once. After same-directory "
                "atomic promotion, the app proves it is the identical file object, performs a bounded final-path "
                "reopen and structural comparison, then calculates SHA-256 once. "
                "Audio, subtitles, attachments, data, chapters, metadata, SAR/DAR, color tags, field parity, and nominal rate must survive."
            ),
            style="Warn.TLabel",
            wraplength=900,
        ).grid(row=5, column=0, sticky="ew", pady=(0, 12))
        buttons = ttk.Frame(frame)
        buttons.grid(row=6, column=0, sticky="e")
        ttk.Button(buttons, text="Start safe repair", command=self._start_repair_from_dialog).pack(side="left")
        ttk.Button(buttons, text="Cancel", command=self._close_repair_dialog).pack(side="left", padx=(8, 0))
        dialog.protocol("WM_DELETE_WINDOW", self._close_repair_dialog)
        dialog.grab_set()
        dialog.focus_force()

    def _browse_repair_output(self) -> None:
        source = Path(self.input_var.get().strip())
        current = Path(self.repair_output_var.get().strip()) if self.repair_output_var.get().strip() else self._suggest_repair_output(source)
        chosen = filedialog.asksaveasfilename(
            title="Choose separate QTGMC repair copy",
            initialdir=str(current.parent),
            initialfile=current.name,
            defaultextension=".mkv",
            filetypes=[("Matroska", "*.mkv")],
            parent=self.repair_dialog or self.root,
        )
        if chosen:
            self.repair_output_var.set(str(Path(chosen)))

    def _start_repair_from_dialog(self) -> None:
        if self.busy_kind or not self.capabilities:
            return
        source = Path(self.input_var.get().strip())
        output_text = self.repair_output_var.get().strip()
        mode = self.repair_mode_var.get()
        if not source.is_file():
            messagebox.showerror("Missing source", f"The selected source does not exist:\n{source}", parent=self.repair_dialog or self.root)
            return
        if not output_text:
            messagebox.showerror("Missing output", "Choose a separate repair output path.", parent=self.repair_dialog or self.root)
            return
        output = Path(output_text)
        artifacts = completed_artifacts(output, REPAIR_ARTIFACT_SUFFIXES)
        existing = [path for path in artifacts if path.exists()]
        overwrite_approved = False
        if existing:
            message = (
                "These completed repair artifacts already exist. They will remain untouched until a new partial "
                "passes validation, then be replaced with rollback protection:\n\n"
                + "\n".join(str(path) for path in existing)
                + "\n\nContinue?"
            )
            if not messagebox.askyesno("Confirm safe repair replacement", message, parent=self.repair_dialog or self.root):
                return
            overwrite_approved = True

        current_media = self.media
        if current_media is not None:
            try:
                if os.path.normcase(os.path.abspath(current_media.path)) != os.path.normcase(os.path.abspath(source)):
                    current_media = None
            except OSError:
                current_media = None
        request = RepairRequest(source, output, mode=mode, overwrite_approved=overwrite_approved)
        self._close_repair_dialog()
        self._launch_repair(request, current_media, automatic=False)

    def _launch_repair(
        self,
        request: RepairRequest,
        current_media: MediaProbe | None,
        *,
        automatic: bool,
    ) -> None:
        if self.busy_kind or not self.capabilities:
            return
        source = request.source_path
        active_capabilities = self.capabilities
        self.repairer = SourceRepairer()
        active_repairer = self.repairer
        self._set_busy("repair")
        self.started_at = time.monotonic()
        self.progress_var.set(0)
        self.run_detail_var.set(
            "Automatic recovery 1/3 · diagnosing" if automatic else "Diagnosing source…"
        )
        self.status_var.set(
            "Automatic recovery 1/3: performing the complete decoded diagnosis/repair…"
            if automatic
            else "Repair: performing a complete decoded timestamp and corruption scan…"
        )
        self._set_text(self.log_text, "")
        if automatic and self.auto_workflow:
            self._append_log(
                f"Automatic recovery triggered by: {self.auto_workflow.trigger_health.reason}"
            )
            self._append_log(f"Original source: {self.auto_workflow.original_source}")
            self._append_log(f"Reserved repair copy: {request.output_path}")
            self._append_log(f"Reserved final output: {self.auto_workflow.final_settings.output_path}")
            self._append_log(
                "Storage preflight: " + self.auto_workflow.storage_preflight_summary
            )

        def log_callback(line: str) -> None:
            self.events.put(("repair_log", line))

        def progress_callback(values: dict[str, str]) -> None:
            self.events.put(("repair_progress", values))

        def worker() -> None:
            try:
                media = current_media or probe_media(active_capabilities.ffprobe_path, source, sample_frames=64)
                result = active_repairer.run(
                    request,
                    media,
                    active_capabilities,
                    log_callback=log_callback,
                    progress_callback=progress_callback,
                )
            except Exception as exc:
                result = None
                self.events.put(("repair_start_error", str(exc)))
            if result is not None:
                self.events.put(("repair_done", result))

        threading.Thread(target=worker, daemon=True).start()
        self._update_elapsed()

    def _browse_output(self) -> None:
        initial_path = Path(self.output_var.get()) if self.output_var.get().strip() else None
        initial_dir = (initial_path.parent if initial_path else Path(self.persisted.get("last_output_dir") or os.getcwd()))
        chosen = filedialog.asksaveasfilename(
            title="Choose deinterlaced output",
            initialdir=str(initial_dir),
            initialfile=initial_path.name if initial_path else "deinterlaced.mkv",
            defaultextension=".mkv",
            filetypes=[("Matroska", "*.mkv"), ("QuickTime MOV", "*.mov")],
        )
        if chosen:
            self.output_var.set(str(Path(chosen)))
            self.persisted["last_output_dir"] = str(Path(chosen).parent)

    def _suggest_output(self) -> None:
        source_text = self.input_var.get().strip()
        if not source_text:
            return
        source = Path(source_text)
        try:
            profile = select_profile(
                FAMILY_LABELS[self.family_var.get()],
                int(self.bit_depth_var.get()),
                self.hardware_encode_var.get(),
                AV1_ENCODER_LABELS[self.av1_encoder_var.get()],
                FFV1_CHROMA_LABELS[self.ffv1_chroma_var.get()],
                self.media.video.pix_fmt if self.media else None,
            )
        except (KeyError, ValueError):
            return
        extension = profile.default_extension
        if extension == ".mov" and self.media:
            incompatible_subtitles = any(
                stream.codec_name not in MOV_SUBTITLE_CODECS for stream in self.media.streams_of_type("subtitle")
            )
            incompatible_audio = any(
                stream.codec_name not in MOV_AUDIO_CODECS for stream in self.media.streams_of_type("audio")
            )
            if (
                (self.copy_attachments_var.get() and self.media.attachment_count)
                or (self.copy_subtitles_var.get() and incompatible_subtitles)
                or (self.copy_audio_var.get() and incompatible_audio)
                or (self.copy_data_var.get() and self.media.data_count)
            ):
                extension = ".mkv"
        directory = Path(self.persisted.get("last_output_dir") or source.parent)
        self.output_var.set(str(directory / f"{source.stem}.deinterlaced{extension}"))

    def _start_mov_compatibility_copy(self) -> None:
        """Remux a completed ProRes/DNxHR MKV without rerunning image processing."""

        if self.busy_kind:
            self.status_var.set("Finish or cancel the active operation before starting a compatibility copy.")
            return
        if not self.capabilities or not self.capabilities.ffmpeg_path or not self.capabilities.ffprobe_path:
            messagebox.showerror(
                "FFmpeg unavailable",
                "A validated FFmpeg/FFprobe pair is required. Use Tools → Dependency doctor first.",
                parent=self.root,
            )
            return

        candidates = [
            self.last_completed_output,
            Path(self.output_var.get().strip()) if self.output_var.get().strip() else None,
            Path(self.input_var.get().strip()) if self.input_var.get().strip() else None,
        ]
        initial = next(
            (
                path
                for path in candidates
                if path is not None and path.is_file() and path.suffix.casefold() == ".mkv"
            ),
            None,
        )
        chosen = filedialog.askopenfilename(
            title="Choose a completed ProRes or DNxHR MKV",
            initialdir=str(initial.parent if initial else Path(self.persisted.get("last_output_dir") or os.getcwd())),
            initialfile=initial.name if initial else "",
            filetypes=[("Matroska video", "*.mkv"), ("All files", "*.*")],
        )
        if not chosen:
            return
        source = Path(chosen)
        suggested = source.with_name(f"{source.stem}.compatibility.mov")
        destination = filedialog.asksaveasfilename(
            title="Choose the native MOV compatibility output",
            initialdir=str(suggested.parent),
            initialfile=suggested.name,
            defaultextension=".mov",
            filetypes=[("QuickTime MOV", "*.mov")],
        )
        if not destination:
            return
        output = Path(destination)
        if output.exists():
            messagebox.showerror(
                "Choose a new output name",
                "The compatibility workflow never overwrites an existing file. Choose a new MOV filename.",
                parent=self.root,
            )
            return
        if not messagebox.askyesno(
            "Create fast native-MOV compatibility copy?",
            "This workflow is for a completed ProRes or DNxHR MKV that decodes in FFmpeg/VLC but is unreliable "
            "in a DirectShow/MPC or editor path.\n\n"
            "• Video and MOV-compatible audio are stream-copied; QTGMC, BWDIF, and denoising do not run again.\n"
            "• Supported text subtitles are converted to MOV's native mov_text format.\n"
            "• Attachments, data streams, and unsupported subtitle types remain available in the unchanged MKV.\n"
            "• The app compares video packet count and video-essence SHA-256, then strictly decodes the whole MOV "
            "before it is promoted.\n"
            "• Existing files are never overwritten.\n\n"
            f"Source (unchanged):\n{source}\n\nOutput:\n{output}\n\nContinue?",
            parent=self.root,
        ):
            return

        copier = MOVCompatibilityCopier()
        self.compatibility_copier = copier
        active_ffmpeg = self.capabilities.ffmpeg_path
        active_ffprobe = self.capabilities.ffprobe_path
        self._set_busy("compatibility")
        self.started_at = time.monotonic()
        self.progress_var.set(0)
        self.run_phase_detail = "Fast MOV compatibility copy · probing"
        self.run_detail_var.set(self.run_phase_detail)
        self.status_var.set(
            "Checking the completed MKV and preparing a no-video/no-audio-reencode native MOV copy…"
        )
        self._set_text(self.log_text, "")

        def log_callback(line: str) -> None:
            self.events.put(("compatibility_log", line))

        def progress_callback(values: dict[str, str]) -> None:
            self.events.put(("compatibility_progress", values))

        def worker() -> None:
            result = copier.run(
                CompatibilityCopyRequest(source, output),
                active_ffmpeg,
                active_ffprobe,
                log_callback=log_callback,
                progress_callback=progress_callback,
            )
            self.events.put(("compatibility_done", result))

        threading.Thread(target=worker, daemon=True).start()
        self._update_elapsed()

    def _handle_compatibility_progress(self, values: dict[str, str]) -> None:
        phase = values.get("phase", "compatibility_validate")
        labels = {
            "compatibility_probe": ("probing source", "Checking codec, container, and tracks…", 0.0, 5.0),
            "compatibility_remux": ("stream-copying to MOV", "Stream-copying video/audio to a unique MOV partial…", 5.0, 50.0),
            "compatibility_validate": ("validating structure", "Validating streams, geometry, chapters, and video packet count…", 50.0, 55.0),
            "compatibility_hash": ("proving identical video essence", "Comparing source and MOV video-essence SHA-256…", 55.0, 65.0),
            "compatibility_full_decode": ("strict full MOV decode", "Strictly decoding every MOV video frame before promotion…", 65.0, 98.0),
            "compatibility_promote": ("promoting validated MOV", "Validation passed; promoting the exact checked MOV file…", 98.0, 99.0),
            "compatibility_complete": ("validated", "Native MOV compatibility copy completed and validated.", 100.0, 100.0),
        }
        detail, status, floor, ceiling = labels.get(
            phase,
            (phase.replace("_", " "), "Validating the compatibility copy…", 50.0, 99.0),
        )
        progress = None
        try:
            duration_us = int(values.get("duration_us", "0"))
            out_time_us = int(values.get("out_time_us", values.get("out_time_ms", "0")))
            if duration_us > 0:
                progress = floor + (ceiling - floor) * min(1.0, max(0.0, out_time_us / duration_us))
        except ValueError:
            progress = None
        if progress is None:
            try:
                progress = float(values.get("percent", floor))
            except ValueError:
                progress = floor
        self.progress_var.set(min(100.0, max(0.0, progress)))
        self.run_phase_detail = f"Fast MOV compatibility copy · {detail}"
        self.run_detail_var.set(self.run_phase_detail)
        self.status_var.set(status)

    def _handle_compatibility_done(self, result) -> None:
        self._set_busy(None)
        self.started_at = None
        self.compatibility_copier = None
        if self.close_pending:
            self._finish_pending_close()
            return
        if result.success and result.output_path:
            self.progress_var.set(100)
            self.last_completed_output = result.output_path
            self.run_phase_detail = "MOV compatibility copy validated"
            self.run_detail_var.set(self.run_phase_detail)
            self.status_var.set(f"Completed and validated native MOV copy: {result.output_path}")
            omissions = (
                "\n\nTracks retained only in the unchanged MKV:\n• " + "\n• ".join(result.omitted_tracks)
                if result.omitted_tracks
                else ""
            )
            messagebox.showinfo(
                "MOV compatibility copy complete",
                f"Validated output:\n{result.output_path}\n\n"
                "Video re-encoded: no\nAudio re-encoded: no\n"
                f"Video packets: {result.source_video_packets} → {result.output_video_packets}\n"
                f"Video-essence SHA-256:\n{result.video_essence_sha256}\n"
                f"Text subtitles converted to mov_text: {result.converted_subtitle_tracks}"
                f"{omissions}\n\nAudit report:\n{result.report_path}",
                parent=self.root,
            )
        elif result.canceled:
            self.run_phase_detail = "MOV compatibility copy canceled"
            self.run_detail_var.set(self.run_phase_detail)
            self.status_var.set(f"Compatibility copy canceled. Log retained at {result.log_path}")
        else:
            self.run_phase_detail = "MOV compatibility copy failed"
            self.run_detail_var.set(self.run_phase_detail)
            self.status_var.set(f"Compatibility copy failed: {result.message}")
            messagebox.showerror(
                "MOV compatibility copy failed",
                f"{result.message}\n\nThe source was not changed and no final MOV was promoted."
                f"\nDiagnostic log: {result.log_path or 'unavailable'}"
                f"\nDiagnostic report: {result.report_path or 'unavailable'}",
                parent=self.root,
            )

    def _apply_resolve_editor_preset(self) -> None:
        """Select a conservative Windows Resolve editor-interchange master."""

        if self.busy_kind:
            return
        if self.media is None or self.capabilities is None:
            messagebox.showwarning(
                "Analyze a source first",
                "Load and analyze the source before applying the Resolve editor preset so its tracks and "
                "installed DNxHR encoder can be checked.",
                parent=self.root,
            )
            return
        profile = select_profile("dnxhr", 10, False)
        unavailable = profile_capability_error(profile, self.capabilities)
        if unavailable:
            messagebox.showerror(
                "Resolve editor preset unavailable",
                "The selected FFmpeg installation cannot create the preset's DNxHR 444 10-bit video:\n\n"
                + unavailable,
                parent=self.root,
            )
            return

        incompatible_audio = sorted(
            {
                stream.codec_name
                for stream in self.media.streams_of_type("audio")
                if stream.codec_name not in MOV_AUDIO_CODECS
            }
        )
        keep_audio = bool(self.media.audio_count) and not incompatible_audio
        omitted: list[str] = []
        if self.media.subtitle_count:
            omitted.append(f"{self.media.subtitle_count} subtitle track(s)")
        if self.media.attachment_count:
            omitted.append(f"{self.media.attachment_count} attachment(s)")
        if self.media.data_count:
            omitted.append(f"{self.media.data_count} data stream(s)")
        if incompatible_audio:
            omitted.append("audio using MOV-incompatible codec(s): " + ", ".join(incompatible_audio))

        source = self.media.path
        output_text = self.output_var.get().strip()
        if output_text and Path(output_text).parent != Path("."):
            directory = Path(output_text).parent
        else:
            directory = Path(self.persisted.get("last_output_dir") or source.parent)
        stem = source.stem
        suffix = ".resolve-editor.mov" if stem.casefold().endswith(".deinterlaced") else ".deinterlaced.resolve-editor.mov"
        output = directory / f"{stem}{suffix}"

        omission_text = (
            "\n\nThe editor master will omit: " + ", ".join(omitted) + "."
            if omitted
            else "\n\nNo selected source tracks require omission for this MOV preset."
        )
        prompt = (
            "Apply the DaVinci Resolve editor-master preset?\n\n"
            "OUTPUT CHANGES\n"
            "• Avid DNxHR 444, 10-bit, progressive MOV\n"
            "• CPU intra-frame encoding (NVIDIA does not provide a DNxHR encoder here)\n"
            f"• Compatible audio: {'preserved by direct copy' if keep_audio else 'not included'}\n"
            "• Chapters and metadata: preserved\n"
            "• Output name: " + output.name +
            omission_text +
            "\n\nThis avoids the high-bit-depth FFV1 variant that produced Media Offline in Resolve. "
            "It does not change the selected deinterlacer, cadence, aspect ratio, denoiser, or source file. "
            "The original remains available for any omitted tracks. No processing will start."
        )
        if not messagebox.askyesno(
            "Confirm Resolve editor preset",
            prompt,
            parent=self.root,
        ):
            return

        updated = replace(
            self._collect_settings(),
            output_path=output,
            family="dnxhr",
            bit_depth=10,
            hardware_encode=False,
            copy_audio=keep_audio,
            copy_subtitles=False,
            copy_attachments=False,
            copy_data=False,
            copy_chapters=True,
            copy_metadata=True,
        )
        self._restore_automatic_settings(updated)
        self.persisted["last_output_dir"] = str(output.parent)
        if self.analysis:
            self._refresh_plan()
        self.run_detail_var.set("Resolve editor preset applied")
        self.status_var.set(
            "Resolve editor preset applied: DNxHR 444 10-bit MOV. Review the plan, then Start when ready."
        )

    def _settings_changed(self, *_args) -> None:
        if self._restoring_automatic_settings or not hasattr(self, "start_button"):
            return
        if self.auto_workflow is not None:
            intended = self.auto_workflow.final_settings
            self.status_var.set(
                "Automatic recovery has locked the captured output settings; cancel the chain before changing them."
            )
            self.root.after_idle(lambda: self._restore_automatic_settings(intended))
            return
        self._refresh_control_states()
        if not self.busy_kind and hasattr(self, "batch_queue"):
            for record in self.batch_queue.records:
                record.reset_plan(retain_analysis=True)
            self._refresh_batch_tree()
        if self.media and self.analysis and self.capabilities:
            self._schedule_refresh_plan()

    def _schedule_refresh_plan(self) -> None:
        if self.plan_refresh_after_id is None:
            self.plan_refresh_after_id = self.root.after_idle(self._run_scheduled_refresh_plan)

    def _run_scheduled_refresh_plan(self) -> None:
        self.plan_refresh_after_id = None
        try:
            if self.root.winfo_exists():
                self._refresh_plan()
        except TclError:
            return

    def _automatic_recovery_setting_changed(self, *_args) -> None:
        self.persisted["automatic_repair_and_continue"] = self.auto_repair_continue_var.get()
        if not self.busy_kind and hasattr(self, "batch_queue"):
            for record in self.batch_queue.records:
                record.reset_plan(retain_analysis=True)
            if hasattr(self, "batch_tree"):
                self._refresh_batch_tree()
        if not hasattr(self, "status_var") or self.auto_workflow is not None:
            return
        if not self.auto_repair_continue_var.get():
            if self.source_health and self.source_health.repair_required:
                if self.media and self.analysis and self.capabilities:
                    self._refresh_plan()
                else:
                    self.status_var.set(
                        "Automatic QTGMC recovery is off. Use Repair required… for the manual workflow."
                    )
            return
        if (
            not self.busy_kind
            and not self._is_batch_tab()
            and self.media
            and self.analysis
            and self.source_health
            and self.source_health.repair_required
        ):
            self.root.after_idle(self._maybe_begin_automatic_recovery)

    @staticmethod
    def _label_for(mapping: dict[str, str], value: str) -> str:
        return next(label for label, candidate in mapping.items() if candidate == value)

    def _restore_automatic_settings(self, settings: JobSettings) -> None:
        """Restore the user's pre-repair output intent without changing the repaired input."""

        self._restoring_automatic_settings = True
        try:
            self.output_var.set(str(settings.output_path))
            self.engine_var.set(self._label_for(ENGINE_LABELS, settings.backend))
            self.field_var.set(self._label_for(FIELD_LABELS, settings.field_order))
            self.cadence_var.set(self._cadence_label(settings.output_cadence))
            self.progressive_override_var.set(settings.allow_progressive_override)
            self.aspect_var.set(self._label_for(ASPECT_LABELS, settings.aspect_mode))
            self.manual_dar_var.set(settings.manual_dar)
            self.family_var.set(self._label_for(FAMILY_LABELS, settings.family))
            self.bit_depth_var.set(str(settings.bit_depth))
            self.ffv1_chroma_var.set(self._label_for(FFV1_CHROMA_LABELS, settings.ffv1_chroma_mode))
            self.hardware_encode_var.set(settings.hardware_encode)
            self.hardware_decode_var.set(self._label_for(HW_DECODE_LABELS, settings.hardware_decode))
            self.vulkan_nnedi3_var.set(settings.vulkan_nnedi3)
            self.av1_encoder_var.set(
                self._label_for(AV1_ENCODER_LABELS, settings.av1_software_encoder)
            )
            self.quality_var.set(str(settings.quality))
            self.tune_grain_var.set(settings.tune_grain)
            self.denoise_enabled_var.set(settings.denoise_enabled)
            self.denoiser_var.set(self._label_for(DENOISER_LABELS, settings.denoiser))
            self.denoise_strength_var.set(str(settings.denoise_strength))
            self.denoise_radius_var.set(str(settings.denoise_temporal_radius))
            self.copy_audio_var.set(settings.copy_audio)
            self.copy_subtitles_var.set(settings.copy_subtitles)
            self.copy_attachments_var.set(settings.copy_attachments)
            self.copy_data_var.set(settings.copy_data)
            self.copy_chapters_var.set(settings.copy_chapters)
            self.copy_metadata_var.set(settings.copy_metadata)
        finally:
            self._restoring_automatic_settings = False
        self._refresh_control_states()

    def _input_path_changed(self, *_args) -> None:
        if not hasattr(self, "start_button"):
            return
        self._refresh_repair_button_state()
        current = Path(self.input_var.get().strip()) if self.input_var.get().strip() else None
        if self.media and current:
            try:
                if os.path.normcase(os.path.abspath(current)) == os.path.normcase(os.path.abspath(self.media.path)):
                    return
            except OSError:
                pass
        self._invalidate_analysis()

    def _refresh_repair_button_state(self) -> None:
        if not hasattr(self, "repair_button"):
            return
        source_text = self.input_var.get().strip()
        ready = bool(
            not self.busy_kind
            and source_text
            and Path(source_text).is_file()
            and self.capabilities
            and self.capabilities.ffmpeg_path
            and self.capabilities.ffprobe_path
        )
        repair_text = "Repair source…"
        if self.source_health and health_matches_source(self.source_health, Path(source_text)):
            if self.source_health.repair_required:
                repair_text = "Repair required…"
            elif self.source_health.status in {"warning", "inconclusive"}:
                repair_text = "Diagnose / repair…"
        self.repair_button.configure(text=repair_text, state="normal" if ready else "disabled")

    def _set_source_health(self, report: SourceHealthReport | None) -> None:
        self.source_health = report
        if report is None:
            self.source_health_var.set(
                "Source health: not checked — normal analysis includes a fast full-file timeline precheck."
            )
            style = "HealthWarn.TLabel"
            if hasattr(self, "source_hint_label"):
                self.source_hint_label.grid()
        else:
            self.source_health_var.set(health_headline(report))
            style = {
                "clear": "HealthGood.TLabel",
                "warning": "HealthWarn.TLabel",
                "inconclusive": "HealthWarn.TLabel",
                "repair_required": "HealthError.TLabel",
            }.get(report.status, "HealthWarn.TLabel")
            if hasattr(self, "source_hint_label"):
                self.source_hint_label.grid_remove()
        if hasattr(self, "source_health_label"):
            self.source_health_label.configure(style=style)
        self._refresh_repair_button_state()

    def _refresh_control_states(self) -> None:
        family = FAMILY_LABELS.get(self.family_var.get(), "ffv1")
        engine = ENGINE_LABELS.get(self.engine_var.get(), "auto")
        denoise_enabled = self.denoise_enabled_var.get()
        denoiser = DENOISER_LABELS.get(self.denoiser_var.get(), DEFAULT_DENOISER)
        resolved_engine = engine
        if engine == "auto" and self.analysis is not None:
            if self.analysis.classification == "progressive":
                resolved_engine = "progressive"
            elif self.analysis.classification in {"tff", "bff"}:
                resolved_engine = (
                    "vapoursynth_qtgmc"
                    if self.capabilities and self.capabilities.qtgmc_ready
                    else "ffmpeg_bwdif"
                )
        resolved_progressive = engine == "progressive" or (
            engine == "auto" and self.analysis is not None and self.analysis.classification == "progressive"
        )
        av1_encoder = AV1_ENCODER_LABELS.get(self.av1_encoder_var.get(), "libaom")
        available_depths = selectable_bit_depths(
            family,
            self.capabilities,
            hardware_encode=self.hardware_encode_var.get(),
            av1_software_encoder=av1_encoder,
        )
        depth_values = [str(value) for value in available_depths]
        if self.bit_depth_var.get() not in depth_values:
            self.bit_depth_var.set(depth_values[0])
        self.depth_combo.configure(values=depth_values, state="readonly")
        self.depth_hint_var.set("")
        if family == "ffv1":
            self.ffv1_chroma_combo.configure(state="readonly")
        elif family in {"prores"}:
            self.ffv1_chroma_combo.configure(state="disabled")
        elif family in {"hevc", "av1", "dnxhr"}:
            self.ffv1_chroma_combo.configure(state="disabled")
        if family == "dnxhr" and 12 not in available_depths:
            supported = (
                self.capabilities.encoder_pixel_formats.get("dnxhd", ())
                if self.capabilities
                else ()
            )
            detail = f" ({', '.join(supported)})" if supported else ""
            self.depth_hint_var.set(
                "12-bit is hidden because this FFmpeg DNxHR encoder does not expose yuv444p12le"
                f"{detail}; 10-bit is the highest valid DNxHR 444 choice."
            )

        hardware_family = family in {"hevc", "av1"}
        if not hardware_family and self.hardware_encode_var.get():
            self.hardware_encode_var.set(False)
        self.hardware_encode_check.configure(state="normal" if hardware_family else "disabled")
        self.av1_combo.configure(
            state="readonly" if family == "av1" and not self.hardware_encode_var.get() else "disabled"
        )
        self.quality_spin.configure(state="normal" if family in {"hevc", "av1"} else "disabled")
        self.grain_check.configure(
            state="normal" if family == "hevc" and not self.hardware_encode_var.get() else "disabled"
        )
        self.manual_dar_entry.configure(state="normal" if ASPECT_LABELS.get(self.aspect_var.get()) == "manual" else "disabled")
        if resolved_progressive:
            frame_rate_label = self._cadence_label("frame_rate")
            if self.cadence_var.get() != frame_rate_label:
                self.cadence_var.set(frame_rate_label)
            self.cadence_combo.configure(state="disabled")
            self.field_combo.configure(state="disabled")
        else:
            self.cadence_combo.configure(state="readonly")
            self.field_combo.configure(state="readonly")
        uses_vapoursynth_graph = resolved_engine == "vapoursynth_qtgmc" or (
            resolved_engine == "progressive"
            and denoise_enabled
            and DENOISER_BY_ID[denoiser].engine == "vapoursynth"
        )
        if uses_vapoursynth_graph:
            self.hw_decode_combo.configure(state="disabled")
            if HW_DECODE_LABELS.get(self.hardware_decode_var.get()) == "cuda":
                self.hardware_decode_var.set(self._label_for(HW_DECODE_LABELS, "auto"))
        else:
            self.hw_decode_combo.configure(state="readonly")

        vulkan_applicable = resolved_engine == "vapoursynth_qtgmc"
        vulkan_ready = bool(self.capabilities and self.capabilities.vulkan_nnedi3_ready)
        self.vulkan_nnedi3_check.configure(
            state="normal" if vulkan_applicable and vulkan_ready and not self.busy_kind else "disabled"
        )
        if self.analysis is not None and not vulkan_applicable and self.vulkan_nnedi3_var.get():
            self.vulkan_nnedi3_var.set(False)

        if denoiser == "ffmpeg_fftdnoiz" and self.denoise_radius_var.get() != "1":
            self.denoise_radius_var.set("1")
        self.denoiser_combo.configure(state="readonly" if denoise_enabled else "disabled")
        self.denoise_strength_spin.configure(state="normal" if denoise_enabled else "disabled")
        radius_supported = denoiser != "ffmpeg_fftdnoiz"
        self.denoise_radius_spin.configure(
            state="normal" if denoise_enabled and radius_supported else "disabled"
        )
        self.denoise_radius_label.configure(
            text=(
                "Temporal radius (fixed at 1 for fftdnoiz)"
                if denoiser == "ffmpeg_fftdnoiz"
                else "Temporal radius (1–6)"
            )
        )
        if self.busy_kind:
            for widget in (
                self.engine_combo,
                self.field_combo,
                self.cadence_combo,
                self.hw_decode_combo,
                self.vulkan_nnedi3_check,
                self.aspect_combo,
                self.manual_dar_entry,
                self.denoise_check,
                self.denoiser_combo,
                self.denoise_strength_spin,
                self.denoise_radius_spin,
                self.progressive_override_check,
                self.family_combo,
                self.depth_combo,
                self.ffv1_chroma_combo,
                self.hardware_encode_check,
                self.av1_combo,
                self.quality_spin,
                self.grain_check,
                *self.track_checks,
            ):
                widget.configure(state="disabled")
        self._refresh_batch_control_states()

    def _refresh_batch_control_states(self) -> None:
        """Mirror shared-variable validity without assuming every row has one analysis result."""

        if not hasattr(self, "batch_engine_combo"):
            return
        family = FAMILY_LABELS.get(self.family_var.get(), "ffv1")
        engine = ENGINE_LABELS.get(self.engine_var.get(), "auto")
        denoise_enabled = self.denoise_enabled_var.get()
        denoiser = DENOISER_LABELS.get(self.denoiser_var.get(), DEFAULT_DENOISER)
        av1_encoder = AV1_ENCODER_LABELS.get(self.av1_encoder_var.get(), "libaom")
        available_depths = selectable_bit_depths(
            family,
            self.capabilities,
            hardware_encode=self.hardware_encode_var.get(),
            av1_software_encoder=av1_encoder,
        )
        self.batch_depth_combo.configure(values=[str(value) for value in available_depths])
        self.batch_ffv1_chroma_combo.configure(
            state="readonly" if family == "ffv1" else "disabled"
        )
        hardware_family = family in {"hevc", "av1"}
        self.batch_hardware_encode_check.configure(
            state="normal" if hardware_family else "disabled"
        )
        self.batch_av1_combo.configure(
            state="readonly" if family == "av1" and not self.hardware_encode_var.get() else "disabled"
        )
        self.batch_quality_spin.configure(
            state="normal" if family in {"hevc", "av1"} else "disabled"
        )
        self.batch_grain_check.configure(
            state="normal" if family == "hevc" and not self.hardware_encode_var.get() else "disabled"
        )
        self.batch_manual_dar_entry.configure(
            state="normal" if ASPECT_LABELS.get(self.aspect_var.get()) == "manual" else "disabled"
        )
        if engine == "progressive":
            self.batch_cadence_combo.configure(state="disabled")
            self.batch_field_combo.configure(state="disabled")
        else:
            self.batch_cadence_combo.configure(state="readonly")
            self.batch_field_combo.configure(state="readonly")
        self.batch_hw_decode_combo.configure(state="readonly")
        vulkan_ready = bool(self.capabilities and self.capabilities.vulkan_nnedi3_ready)
        self.batch_vulkan_check.configure(
            state="normal" if engine in {"auto", "vapoursynth_qtgmc"} and vulkan_ready else "disabled"
        )
        self.batch_denoiser_combo.configure(state="readonly" if denoise_enabled else "disabled")
        self.batch_denoise_strength_spin.configure(state="normal" if denoise_enabled else "disabled")
        radius_supported = denoiser != "ffmpeg_fftdnoiz"
        self.batch_denoise_radius_spin.configure(
            state="normal" if denoise_enabled and radius_supported else "disabled"
        )
        self.batch_denoise_radius_label.configure(
            text=(
                "Temporal radius (fixed at 1 for fftdnoiz)"
                if denoiser == "ffmpeg_fftdnoiz"
                else "Temporal radius (1–6)"
            )
        )
        if self.busy_kind:
            for widget in (
                self.batch_engine_combo,
                self.batch_field_combo,
                self.batch_cadence_combo,
                self.batch_hw_decode_combo,
                self.batch_vulkan_check,
                self.batch_aspect_combo,
                self.batch_manual_dar_entry,
                self.batch_denoise_check,
                self.batch_denoiser_combo,
                self.batch_denoise_strength_spin,
                self.batch_denoise_radius_spin,
                self.batch_progressive_override_check,
                self.batch_family_combo,
                self.batch_depth_combo,
                self.batch_ffv1_chroma_combo,
                self.batch_hardware_encode_check,
                self.batch_av1_combo,
                self.batch_quality_spin,
                self.batch_grain_check,
                *self.batch_track_checks,
            ):
                widget.configure(state="disabled")

    def _invalidate_analysis(self) -> None:
        self.media = None
        self.analysis = None
        self.health_scan_media = None
        self._set_source_health(None)
        self.plan = None
        self._update_cadence_labels()
        self.analysis_progress_var.set("No current analysis")
        self._set_text(self.summary_text, "Source selection changed. Probe and IDet analysis are required again.")
        self._set_text(self.plan_text, "Analysis is stale. Run Probe + IDet again.")
        self._update_start_button_state()

    def _analyze(self, mode: str) -> None:
        if self.busy_kind:
            return
        if not self.capabilities or not self.capabilities.ffprobe_path or not self.capabilities.ffmpeg_path:
            messagebox.showerror("Dependencies unavailable", "FFmpeg and FFprobe must pass discovery first.", parent=self.root)
            return
        source = Path(self.input_var.get().strip())
        if not source.is_file():
            messagebox.showerror("Missing source", f"The selected source does not exist:\n{source}", parent=self.root)
            return
        self.analysis_cancel.clear()
        self.media = None
        self.analysis = None
        self.health_scan_media = None
        self.plan = None
        self._set_source_health(None)
        self._update_start_button_state()
        self._set_text(self.summary_text, "Probing media and preparing the fast full-file source-health precheck…")
        self._set_busy("analysis")
        self.analysis_progress_var.set("Probing media…")
        if self.auto_workflow and self.auto_workflow.stage == "reanalyzing":
            self.run_detail_var.set("Automatic recovery 2/3 · re-analysis")
            self.status_var.set(
                "Automatic recovery 2/3: verifying the repaired copy with fresh health and IDet analysis…"
            )
        else:
            self.status_var.set("Analyzing metadata and decoded field evidence…")

        def worker() -> None:
            try:
                media = probe_media(self.capabilities.ffprobe_path, source, sample_frames=64)

                cache_key = source_identity(source)
                health = self.source_health_cache.get(cache_key)
                cached = health is not None and health_matches_source(health, source)
                if not cached:
                    self.events.put(("analysis_health_started", None))

                    def health_progress(packets: int, fraction: float | None) -> None:
                        self.events.put(("analysis_health_progress", (packets, fraction)))

                    health = scan_source_health(
                        self.capabilities.ffprobe_path,
                        media,
                        cancel_event=self.analysis_cancel,
                        progress=health_progress,
                    )
                assert health is not None
                self.events.put(("analysis_health_done", (media, health, cache_key, cached)))

                def progress(done: int, total: int, offset: float) -> None:
                    self.events.put(("analysis_progress", (done, total, offset, mode)))

                analysis = scan_idet(
                    self.capabilities.ffmpeg_path,
                    media,
                    mode=mode,
                    sample_count=8,
                    sample_seconds=12.0,
                    cancel_event=self.analysis_cancel,
                    progress=progress,
                )
                self.events.put(("analysis_done", (media, analysis, health)))
            except (AnalysisCancelled, SourceHealthCancelled) as exc:
                self.events.put(("analysis_canceled", str(exc)))
            except Exception as exc:
                self.events.put(("analysis_error", str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def _analysis_summary(self) -> str:
        if not self.media or not self.analysis:
            return "No analysis."
        video = self.media.video
        counts = self.analysis.aggregate
        recommendation = {
            "progressive": "DO NOT DEINTERLACE. Encode/pass through progressively unless visual inspection disproves the scan.",
            "tff": "Deinterlace as TFF. The default progressive output matches the source's nominal frame rate; optional field-rate output preserves every temporal field at the original playing speed.",
            "bff": "Deinterlace as BFF. The default progressive output matches the source's nominal frame rate; optional field-rate output preserves every temporal field at the original playing speed.",
            "mixed_or_ambiguous": "Do not start automatically. Inspect cadence/field order and consider telecine or mixed material.",
            "insufficient": "Insufficient evidence. Run the full-file scan or inspect representative frames.",
        }.get(self.analysis.classification, "Inspect before processing.")
        return "\n".join(
            [
                health_summary(self.source_health) if self.source_health else "SOURCE HEALTH — NOT CHECKED",
                f"Path: {self.media.path}",
                f"Video: {video.codec_name} {video.profile or ''} · {video.width}x{video.height} · {video.pix_fmt} · {video.bits_per_raw_sample or '?'}-bit",
                f"Metadata: field_order={video.field_order or 'unknown'} · nominal={rate_text(video.r_frame_rate)} fps · average={rate_text(video.avg_frame_rate)} fps",
                f"Geometry: SAR {fraction_text(video.sample_aspect_ratio)} · DAR {fraction_text(video.display_aspect_ratio)}",
                f"Color: range={video.color_range or 'unspecified'} · matrix={video.color_space or 'unspecified'} · transfer={video.color_transfer or 'unspecified'} · primaries={video.color_primaries or 'unspecified'}",
                f"Tracks: {self.media.audio_count} audio · {self.media.subtitle_count} subtitles · {self.media.attachment_count} attachments · {len(self.media.chapters)} chapters",
                f"Decoded frame flags (first {self.media.sampled_interlaced_frames + self.media.sampled_progressive_frames}): interlaced={self.media.sampled_interlaced_frames}, progressive={self.media.sampled_progressive_frames}, TFF={self.media.sampled_tff_frames}, BFF={self.media.sampled_bff_frames}",
                "",
                f"{self.analysis.mode.title()} IDet result: {self.analysis.classification.upper()} ({self.analysis.confidence:.1%} confidence score)",
                (
                    f"Multi-frame totals: TFF={counts.multi_tff}, BFF={counts.multi_bff}, "
                    f"progressive={counts.multi_progressive}, undetermined={counts.multi_undetermined} · "
                    f"repeated fields: neither={counts.repeated_neither}, top={counts.repeated_top}, "
                    f"bottom={counts.repeated_bottom}"
                ),
                self.analysis.rationale,
                "",
                "Recommendation: " + recommendation,
            ]
        )

    def _collect_settings(
        self,
        *,
        overwrite_approved: bool = False,
        input_path: Path | None = None,
        output_path: Path | None = None,
    ) -> JobSettings:
        try:
            quality = int(self.quality_var.get())
        except ValueError:
            quality = -1
        try:
            denoise_strength = int(self.denoise_strength_var.get())
        except ValueError:
            denoise_strength = -1
        try:
            denoise_radius = int(self.denoise_radius_var.get())
        except ValueError:
            denoise_radius = -1
        return JobSettings(
            input_path=input_path or Path(self.input_var.get().strip()),
            output_path=output_path or Path(self.output_var.get().strip()),
            backend=ENGINE_LABELS.get(self.engine_var.get(), "auto"),
            field_order=FIELD_LABELS.get(self.field_var.get(), "auto"),
            output_cadence=self._cadence_value(),
            allow_progressive_override=self.progressive_override_var.get(),
            aspect_mode=ASPECT_LABELS.get(self.aspect_var.get(), "preserve"),
            manual_dar=self.manual_dar_var.get(),
            family=FAMILY_LABELS.get(self.family_var.get(), "ffv1"),
            bit_depth=int(self.bit_depth_var.get()),
            ffv1_chroma_mode=FFV1_CHROMA_LABELS.get(self.ffv1_chroma_var.get(), "native"),
            hardware_encode=self.hardware_encode_var.get(),
            hardware_decode=HW_DECODE_LABELS.get(self.hardware_decode_var.get(), "auto"),
            vulkan_nnedi3=self.vulkan_nnedi3_var.get(),
            av1_software_encoder=AV1_ENCODER_LABELS.get(self.av1_encoder_var.get(), "libaom"),
            quality=quality,
            tune_grain=self.tune_grain_var.get(),
            denoise_enabled=self.denoise_enabled_var.get(),
            denoiser=DENOISER_LABELS.get(self.denoiser_var.get(), DEFAULT_DENOISER),
            denoise_strength=denoise_strength,
            denoise_temporal_radius=denoise_radius,
            copy_audio=self.copy_audio_var.get(),
            copy_subtitles=self.copy_subtitles_var.get(),
            copy_attachments=self.copy_attachments_var.get(),
            copy_data=self.copy_data_var.get(),
            copy_chapters=self.copy_chapters_var.get(),
            copy_metadata=self.copy_metadata_var.get(),
            overwrite_approved=overwrite_approved,
        )

    def _automatic_recovery_audit(self):
        return self.auto_workflow.audit() if self.auto_workflow else None

    def _stop_automatic_recovery(
        self,
        message: str,
        *,
        title: str = "Automatic recovery stopped",
        show_error: bool = False,
    ) -> None:
        self.auto_workflow = None
        self.run_detail_var.set("Automatic recovery stopped")
        self.status_var.set(message)
        if hasattr(self, "auto_repair_check"):
            self.auto_repair_check.configure(state="normal" if not self.busy_kind else "disabled")
        self._refresh_repair_button_state()
        if show_error and not self.close_pending:
            messagebox.showerror(title, message, parent=self.root)

    def _maybe_begin_automatic_recovery(self) -> bool:
        if (
            self.auto_workflow is not None
            or self.busy_kind
            or not self.auto_repair_continue_var.get()
            or not self.media
            or not self.analysis
            or not self.source_health
            or not self.source_health.repair_required
            or not self.capabilities
        ):
            return False
        source = Path(self.input_var.get().strip())
        if not source.is_file() or not health_matches_source(self.source_health, source):
            self.status_var.set("Automatic recovery did not start because the source changed; analyze it again.")
            return False

        requested_settings = self._collect_settings()
        routing_preview = build_plan(
            requested_settings,
            self.media,
            self.analysis,
            self.capabilities,
            source_health=None,
        )
        if not routing_preview.valid or routing_preview.expected is None:
            detail = "; ".join(routing_preview.errors) or "the output plan is incomplete"
            self.status_var.set(
                "Automatic recovery did not start because the intended final plan is invalid: " + detail
            )
            self.run_detail_var.set("Automation waiting for a valid plan")
            return False
        if not automatic_recovery_applies_to_backend(routing_preview.selected_backend):
            # Non-QTGMC plans remain in FFmpeg's timestamp-aware path. Do not
            # reserve names, estimate a large repair, or queue any repair work.
            self._refresh_plan()
            return False
        try:
            final_output = choose_available_artifact_path(
                requested_settings.output_path,
                DEINTERLACE_ARTIFACT_SUFFIXES,
                reserved=(source,),
            )
            final_settings = replace(requested_settings, output_path=final_output)
            preview = build_plan(
                final_settings,
                self.media,
                self.analysis,
                self.capabilities,
                source_health=None,
            )
            if not preview.valid or preview.expected is None:
                detail = "; ".join(preview.errors) or "the output plan is incomplete"
                self.status_var.set(
                    "Automatic recovery did not start because the intended final plan is invalid: " + detail
                )
                self.run_detail_var.set("Automation waiting for a valid plan")
                return False
            repair_preferred = final_output.parent / f"{source.stem}.qtgmc-repair.mkv"
            repair_output = choose_available_artifact_path(
                repair_preferred,
                REPAIR_ARTIFACT_SUFFIXES,
                reserved=(source, final_output),
            )
            checks = storage_preflight(self.media, preview, repair_output, final_output)
        except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
            self._stop_automatic_recovery(
                f"Automatic recovery could not prepare a safe workflow: {exc}",
                show_error=True,
            )
            return False
        summary = storage_summary(checks)
        insufficient = [check for check in checks if not check.sufficient]
        if insufficient:
            self._stop_automatic_recovery(
                "Automatic recovery stopped before repair because its conservative storage preflight failed. "
                + summary
                + ". Choose another output drive/profile, or disable automation and use the manual workflow.",
                title="Insufficient storage for automatic recovery",
                show_error=True,
            )
            return False

        self.output_var.set(str(final_output))
        self.auto_workflow = AutomaticRecoveryWorkflow(
            original_source=source,
            trigger_health=self.source_health,
            requested_settings=requested_settings,
            final_settings=final_settings,
            repair_output=repair_output,
            analysis_mode=self.analysis.mode,
            storage_preflight_summary=summary,
        )
        self.run_detail_var.set("Automatic recovery 1/3 · repair")
        renamed = "" if final_output == requested_settings.output_path else f" Final output reserved as {final_output}."
        self.status_var.set(
            "Damage detected — automatic recovery will create and validate a separate repair copy. "
            + summary
            + "."
            + renamed
        )
        self.root.after_idle(self._start_automatic_repair)
        return True

    def _start_automatic_repair(self) -> None:
        workflow = self.auto_workflow
        if not workflow or workflow.stage != "repairing" or self.busy_kind:
            return
        source = workflow.original_source
        if (
            not source.is_file()
            or not self.source_health
            or not health_matches_source(workflow.trigger_health, source)
        ):
            self._stop_automatic_recovery(
                "Automatic recovery stopped because the original source changed after analysis.",
                show_error=True,
            )
            return
        # Close the tiny race between reservation and launch without overwriting a new artifact.
        try:
            workflow.repair_output = choose_available_artifact_path(
                workflow.repair_output,
                REPAIR_ARTIFACT_SUFFIXES,
                reserved=(source, workflow.final_settings.output_path),
            )
        except RuntimeError as exc:
            self._stop_automatic_recovery(
                f"Automatic recovery stopped before repair because a safe repair filename could not be reserved: {exc}",
                show_error=True,
            )
            return
        request = RepairRequest(source, workflow.repair_output, mode="automatic", overwrite_approved=False)
        self._launch_repair(request, self.media, automatic=True)

    def _refresh_plan(self) -> None:
        if not self.capabilities:
            return
        settings = self._collect_settings()
        self.plan = build_plan(
            settings,
            self.media,
            self.analysis,
            self.capabilities,
            source_health=self.source_health,
            automatic_recovery=self._automatic_recovery_audit(),
        )
        selected_profile = PROFILES.get(self.plan.profile_id or "")
        if self.plan.vspipe_command:
            decode_stage = "BestSource-managed decode (FFmpeg NVDEC control is not used)"
        elif self.plan.selected_backend == "ffmpeg_bwdif_cuda":
            codec = self.media.video.codec_name.casefold() if self.media else ""
            if self.plan.settings.hardware_decode == "cuda" or (
                self.plan.settings.hardware_decode == "auto" and codec in DIRECT_NVDEC_CODECS
            ):
                decode_stage = "NVIDIA CUDA/NVDEC"
            else:
                decode_stage = "software decode, then CUDA upload for GPU deinterlacing"
        elif self.plan.settings.hardware_decode == "cuda":
            decode_stage = "NVIDIA CUDA/NVDEC"
        elif self.plan.settings.hardware_decode == "auto":
            decode_stage = "FFmpeg automatic hardware decode"
        else:
            decode_stage = "software decode"
        if self.plan.selected_backend == "vapoursynth_qtgmc":
            deinterlace_stage = (
                "QTGMC + Vulkan NNEDI3; MVTools remains CPU"
                if self.plan.vulkan_nnedi3_active
                else "QTGMC CPU NNEDI3 + CPU MVTools"
            )
        elif self.plan.selected_backend == "ffmpeg_bwdif_cuda":
            deinterlace_stage = "NVIDIA CUDA BWDIF"
        elif self.plan.selected_backend == "ffmpeg_bwdif":
            deinterlace_stage = "FFmpeg CPU BWDIF"
        else:
            deinterlace_stage = "none (progressive passthrough)"
        if self.plan.selected_denoiser and self.plan.selected_denoise_backend:
            denoise_stage = denoiser_backend_display(
                self.plan.selected_denoiser,
                self.plan.selected_denoise_backend,
            )
        elif self.plan.settings.denoise_enabled:
            denoise_stage = "requested but unresolved"
        else:
            denoise_stage = "off"
        encode_stage = (
            f"NVIDIA NVENC ({selected_profile.encoder})"
            if selected_profile and selected_profile.hardware
            else f"software/CPU ({selected_profile.encoder})"
            if selected_profile
            else "unresolved"
        )
        lines = [
            f"Plan status: {'VALID' if self.plan.valid else 'BLOCKED'}",
            f"Backend: {self.plan.selected_backend or 'unresolved'}",
            f"Field order: {(self.plan.selected_field_order or 'n/a').upper()}",
            (
                "Acceleration map: "
                f"decode={decode_stage} · deinterlace={deinterlace_stage} · "
                f"denoise={denoise_stage} · encode={encode_stage}"
            ),
            (
                "QTGMC interpolation: Vulkan NNEDI3 (opt-in; graph verified)"
                if self.plan.vulkan_nnedi3_active
                else "QTGMC interpolation: CPU NNEDI3 (maximum-fidelity default)"
                if self.plan.selected_backend == "vapoursynth_qtgmc"
                else "QTGMC interpolation: not applicable"
            ),
            (
                "Temporal denoise: disabled"
                if not self.plan.settings.denoise_enabled
                else (
                    f"Temporal denoise: {self.plan.selected_denoiser or self.plan.settings.denoiser} · "
                    f"implementation {denoise_stage} · "
                    f"strength {self.plan.settings.denoise_strength}/10 · "
                    f"radius {self.plan.settings.denoise_temporal_radius} · after deinterlacing"
                )
            ),
            f"Output profile: {self.plan.profile_label or 'unresolved'}",
            f"Output: {self.plan.output_path}",
        ]
        if self.plan.vspipe_requests is not None:
            lines.append(
                f"Adaptive VapourSynth schedule: {self.plan.vapoursynth_threads} core threads · "
                f"{self.plan.vspipe_requests} VSPipe requests"
            )
            if self.plan.vapoursynth_schedule_note:
                lines.append("Schedule rationale: " + self.plan.vapoursynth_schedule_note)
        if selected_profile:
            lines += [
                f"Encoding contract: {'lossless' if selected_profile.lossless else 'lossy'} · "
                f"{selected_profile.bit_depth}-bit · {selected_profile.chroma} · "
                f"{'intra-only' if selected_profile.intra_only else 'inter-frame'} · {selected_profile.encoder}",
                f"Profile note: {selected_profile.description}",
            ]
        if self.plan.expected:
            exp = self.plan.expected
            lines += [
                f"Expected raster/SAR/DAR: {exp.width}x{exp.height} · {exp.sar} · {exp.dar}",
                f"Expected progressive rate: {rate_text(exp.frame_rate)} fps",
                f"Expected direct-copy tracks: {len(exp.expected_audio)} audio · {len(exp.expected_subtitles)} subtitles · {len(exp.expected_attachments)} attachments",
            ]
        if self.plan.errors:
            lines += ["", "BLOCKING ERRORS:"] + [f"  • {error}" for error in self.plan.errors]
        if self.plan.warnings:
            lines += ["", "WARNINGS / TRADEOFFS:"] + [f"  • {warning}" for warning in self.plan.warnings]
        if self.plan.vapoursynth_script:
            lines += ["", "GENERATED VAPOURSYNTH SCRIPT:", self.plan.vapoursynth_script]
        if self.plan.display_command:
            lines += ["", "EXACT COMMAND:", self.plan.display_command]
        self._set_text(self.plan_text, "\n".join(lines))
        self._update_start_button_state()
        if (
            self.source_health
            and self.source_health.repair_required
            and self.plan.selected_backend == "vapoursynth_qtgmc"
        ):
            self.status_var.set(
                (
                    "Repair is required before QTGMC. Automatic recovery will use the current output settings."
                    if self.auto_repair_continue_var.get()
                    else "Repair is required before QTGMC. Click Repair required… or choose a clean source."
                )
            )
        elif (
            self.source_health
            and self.source_health.repair_required
            and self.plan.selected_backend in {"ffmpeg_bwdif", "ffmpeg_bwdif_cuda"}
            and self.plan.valid
        ):
            backend_name = (
                "BWDIF CUDA" if self.plan.selected_backend == "ffmpeg_bwdif_cuda" else "BWDIF CPU"
            )
            self.status_var.set(
                f"Damage detected · {backend_name} will process the original directly; automatic repair is skipped. "
                "Missing/corrupt pictures cannot be restored. Use Repair required… first if desired."
            )
        else:
            self.status_var.set("Plan is ready." if self.plan.valid else "Plan is blocked; see Plan & command.")

    def _update_start_button_state(self) -> None:
        if not hasattr(self, "start_button"):
            return
        if self._is_batch_tab():
            self.start_button.configure(
                text="Start batch",
                state=(
                    "normal"
                    if (
                        not self.busy_kind
                        and bool(self.batch_queue.records)
                        and self.capabilities is not None
                        and self.capabilities.ffmpeg_path is not None
                        and self.capabilities.ffprobe_path is not None
                    )
                    else "disabled"
                ),
            )
        else:
            self.start_button.configure(
                text="Start deinterlacing",
                state="normal" if not self.busy_kind and self.plan and self.plan.valid else "disabled",
            )

    def _start_requested(self) -> None:
        if self._is_batch_tab():
            self._start_batch()
        else:
            self._start_processing()

    def _start_batch(self) -> None:
        if self.busy_kind or not self.batch_queue.records or not self.capabilities:
            return
        if not self.capabilities.ffmpeg_path or not self.capabilities.ffprobe_path:
            messagebox.showerror(
                "Dependencies unavailable",
                "FFmpeg and FFprobe must pass discovery before batch preflight.",
                parent=self.root,
            )
            return
        output_directory: Path | None = None
        destination_text = self.batch_output_dir_var.get().strip()
        if destination_text:
            output_directory = Path(destination_text)
            try:
                output_directory.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                messagebox.showerror(
                    "Batch output folder unavailable",
                    f"The selected batch output folder could not be created or opened:\n{output_directory}\n\n{exc}",
                    parent=self.root,
                )
                return
            if not output_directory.is_dir():
                messagebox.showerror(
                    "Batch output folder unavailable",
                    f"The selected batch output path is not a folder:\n{output_directory}",
                    parent=self.root,
                )
                return

        first_source = self.batch_queue.records[0].source_path
        requested = self._collect_settings(
            input_path=first_source,
            output_path=first_source.with_name(first_source.stem + ".batch-template.mkv"),
        )
        options = BatchRunOptions(
            output_directory=output_directory,
            auto_repair=self.auto_repair_continue_var.get(),
            continue_after_error=self.batch_continue_var.get(),
            analysis_mode="sampled",
        )

        def event_callback(kind, record, payload) -> None:
            self.events.put(("batch_event", (kind, record, payload)))

        runner = BatchRunner(event_callback)
        self.batch_runner = runner
        self._set_busy("batch")
        self.started_at = time.monotonic()
        self.progress_var.set(0)
        self.run_phase_detail = "Batch preflight · preparing every row"
        self.run_detail_var.set(self.run_phase_detail)
        self.batch_status_var.set(
            f"Preflighting {len(self.batch_queue)} rows before any long encode starts"
        )
        self.status_var.set(self.batch_status_var.get())
        self._set_text(self.log_text, "")
        self._append_log(
            f"Deinterlace Studio {__version__} batch started with {len(self.batch_queue)} ordered row(s)."
        )
        self._append_log(
            "Shared request: "
            f"backend={requested.backend}; cadence={requested.output_cadence}; decode={requested.hardware_decode}; "
            f"output={requested.family}/{requested.bit_depth}-bit; denoise="
            f"{requested.denoiser if requested.denoise_enabled else 'off'}."
        )

        def worker() -> None:
            try:
                runner.run(self.batch_queue, requested, self.capabilities, options)
            except Exception as exc:
                self.events.put(("batch_start_error", f"{type(exc).__name__}: {exc}"))

        threading.Thread(target=worker, daemon=True).start()
        self._update_elapsed()

    def _handle_batch_event(self, kind: str, record, payload) -> None:
        if kind == "row":
            self._refresh_batch_tree()
            return
        if kind == "log":
            if record is not None:
                try:
                    index = self.batch_queue.records.index(record) + 1
                except ValueError:
                    index = 0
                prefix = f"[{index}/{len(self.batch_queue)}] {record.source_path.name}"
            else:
                prefix = "[batch]"
            self._append_log(f"{prefix} | {payload}")
            return
        if kind == "phase":
            text = str(payload)
            self.batch_status_var.set(text)
            self.status_var.set(text)
            self.run_phase_detail = text
            self.run_detail_var.set(text)
            return
        if kind == "overall":
            values = payload if isinstance(payload, dict) else {}
            phase = str(values.get("phase") or "batch")
            index = int(values.get("index") or 1)
            total = max(1, int(values.get("total") or len(self.batch_queue) or 1))
            if phase == "preflight":
                overall = 40.0 * (index - 1) / total
                detail = f"Batch preflight · row {index}/{total}"
            else:
                overall = 40.0 + 60.0 * (index - 1) / total
                detail = f"Batch processing · compatible row {index}/{total}"
            self.progress_var.set(overall)
            self.run_phase_detail = detail
            self.run_detail_var.set(detail)
            self.batch_status_var.set(detail)
            self.status_var.set(detail)
            return
        if kind == "complete" and isinstance(payload, BatchRunSummary):
            self._handle_batch_complete(payload)

    def _handle_batch_complete(self, summary: BatchRunSummary) -> None:
        self._set_busy(None)
        self.started_at = None
        self.batch_runner = None
        completed_outputs = [
            record.result_output
            for record in self.batch_queue.records
            if record.state == "Completed" and record.result_output is not None
        ]
        if completed_outputs:
            self.batch_last_output = completed_outputs[-1]
            self.last_completed_output = completed_outputs[-1]
        self._refresh_batch_tree()
        self.progress_var.set(100 if summary.completed + summary.failed + summary.needs_review + summary.canceled + summary.skipped else 0)
        detail = (
            f"Batch complete · {summary.completed} completed · {summary.failed} failed · "
            f"{summary.needs_review} need review · {summary.canceled} canceled · {summary.skipped} skipped"
        )
        self.run_phase_detail = detail
        self.run_detail_var.set(detail)
        self.batch_status_var.set(detail)
        self.status_var.set(detail)
        self._append_log(detail)
        self._update_start_button_state()
        if not self.close_pending:
            messagebox.showinfo(
                "Batch processing complete",
                f"Total rows: {summary.total}\n"
                f"Completed: {summary.completed}\n"
                f"Failed: {summary.failed}\n"
                f"Needs review: {summary.needs_review}\n"
                f"Canceled: {summary.canceled}\n"
                f"Skipped: {summary.skipped}\n\n"
                "Every fallback or incompatibility remains visible in the queue and Run log.",
                parent=self.root,
            )
        elif self.close_pending:
            self._finish_pending_close()

    def _start_processing(self, *, automatic: bool = False) -> None:
        if self.busy_kind or not self.plan or not self.plan.valid or not self.media or not self.analysis or not self.capabilities:
            return
        artifacts = completed_artifacts(self.plan.output_path, DEINTERLACE_ARTIFACT_SUFFIXES)
        existing = [path for path in artifacts if path.exists()]
        settings = self.plan.settings
        if existing:
            if automatic and self.auto_workflow:
                try:
                    replacement = choose_available_artifact_path(
                        self.plan.output_path,
                        DEINTERLACE_ARTIFACT_SUFFIXES,
                        reserved=(Path(self.input_var.get()), self.auto_workflow.original_source),
                    )
                except RuntimeError as exc:
                    self._stop_automatic_recovery(
                        f"Automatic recovery stopped before encoding because a safe output name could not be reserved: {exc}",
                        show_error=True,
                    )
                    return
                self.auto_workflow.final_settings = replace(
                    self.auto_workflow.final_settings,
                    output_path=replacement,
                )
                self._restore_automatic_settings(self.auto_workflow.final_settings)
                settings = self._collect_settings()
            else:
                message = "These completed artifacts already exist and will be replaced only after the new partial passes validation:\n\n"
                message += "\n".join(str(path) for path in existing)
                message += "\n\nContinue?"
                if not messagebox.askyesno("Confirm safe replacement", message, parent=self.root):
                    return
                settings = replace(settings, overwrite_approved=True)
            self.plan = build_plan(
                settings,
                self.media,
                self.analysis,
                self.capabilities,
                source_health=self.source_health,
                automatic_recovery=self._automatic_recovery_audit(),
            )
            if not self.plan.valid:
                self._refresh_plan()
                if automatic:
                    self._stop_automatic_recovery(
                        "Automatic recovery stopped because the rebuilt final plan is invalid; see Plan & command.",
                        show_error=True,
                    )
                return

        self.processor = JobProcessor()
        active_processor = self.processor
        active_plan = self.plan
        active_media = self.media
        active_analysis = self.analysis
        active_capabilities = self.capabilities
        self._set_busy("processing")
        self.started_at = time.monotonic()
        self.progress_var.set(0)
        self.run_phase_detail = (
            f"{FINAL_PROCESSING_STAGE} · preparing source contract"
            if automatic
            else "Preparing source safety contract"
        )
        self.run_detail_var.set(self.run_phase_detail)
        self.status_var.set(
            f"{FINAL_PROCESSING_STAGE}: checking the repaired source contract before output encoding…"
            if automatic
            else (
                "Checking the source contract before output encoding; clear FFV1/VSPipe jobs use a fast indexed "
                "check, while other VSPipe sources show live full-decode progress."
            )
        )
        self._set_text(self.log_text, "")
        if automatic and self.auto_workflow:
            self._append_log(
                "Final processing is continuing from validated repair "
                f"{self.auto_workflow.validated_repair_source or self.auto_workflow.original_source} "
                f"to {active_plan.output_path}."
            )

        def log_callback(line: str) -> None:
            self.events.put(("run_log", line))

        def progress_callback(values: dict[str, str]) -> None:
            self.events.put(("run_progress", values))

        def worker() -> None:
            result = active_processor.run(
                active_plan,
                active_media,
                active_analysis,
                active_capabilities,
                log_callback=log_callback,
                progress_callback=progress_callback,
            )
            self.events.put(("run_done", result))

        threading.Thread(target=worker, daemon=True).start()
        self._update_elapsed()

    def _cancel_active(self) -> None:
        if self.busy_kind == "analysis":
            self.analysis_cancel.set()
            self.status_var.set("Canceling analysis…")
        elif self.busy_kind == "processing" and self.processor:
            self.processor.cancel()
            self.status_var.set("Canceling the active source check, encoder, and VSPipe process…")
        elif self.busy_kind == "repair" and self.repairer:
            self.repairer.cancel()
            self.status_var.set("Canceling source diagnosis/repair and removing the current partial…")
        elif self.busy_kind == "compatibility" and self.compatibility_copier:
            self.compatibility_copier.cancel()
            self.status_var.set("Canceling the MOV compatibility copy and removing its unpromoted partial…")
        elif self.busy_kind == "dependencies":
            self.dependency_cancel.set()
            self.status_var.set("Canceling dependency installation; the previous active runtime will remain unchanged…")
        elif self.busy_kind == "batch" and self.batch_runner:
            self.batch_runner.cancel()
            self.status_var.set("Canceling the active batch row and marking remaining rows safely…")

    def _set_busy(self, kind: str | None) -> None:
        self.busy_kind = kind
        busy = kind is not None
        state = "disabled" if busy else "normal"
        self.sample_button.configure(state=state)
        self.full_button.configure(state=state)
        self.input_entry.configure(state="disabled" if busy else "normal")
        self.input_browse_button.configure(state="disabled" if busy else "normal")
        self.cancel_button.configure(state="normal" if busy else "disabled")
        self._update_start_button_state()
        self.auto_repair_check.configure(
            state="disabled" if busy or self.auto_workflow is not None else "normal"
        )
        if hasattr(self, "batch_auto_repair_check"):
            self.batch_auto_repair_check.configure(state="disabled" if busy else "normal")
        if hasattr(self, "batch_tree"):
            self.batch_tree.configure(selectmode="none" if busy else "extended")
        for control in getattr(self, "batch_mutation_controls", ()):
            try:
                control.configure(state="disabled" if busy else "normal")
            except TclError:
                pass
        self._refresh_repair_button_state()
        self._refresh_control_states()

    def _poll_events(self) -> None:
        self.poll_after_id = None
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "capabilities_done":
                    if self.capability_scan_started_at is not None:
                        self.last_capability_scan_seconds = time.monotonic() - self.capability_scan_started_at
                    self.capability_scan_started_at = None
                    self.capabilities = payload  # type: ignore[assignment]
                    self._set_busy(None)
                    self._show_capabilities()
                    if self.media and self.analysis:
                        self._refresh_plan()
                    if self.close_pending:
                        self._finish_pending_close()
                        return
                elif kind == "capabilities_error":
                    if self.capability_scan_started_at is not None:
                        self.last_capability_scan_seconds = time.monotonic() - self.capability_scan_started_at
                    self.capability_scan_started_at = None
                    self._set_busy(None)
                    self.dependency_var.set(f"Capability scan failed: {payload}")
                    self.dependency_label.configure(style="Error.TLabel")
                    if self.close_pending:
                        self._finish_pending_close()
                        return
                elif kind == "dependency_progress":
                    stage, message, current, total = payload  # type: ignore[misc]
                    self._show_dependency_progress(str(stage), str(message), current, total)
                elif kind == "dependency_log":
                    self._append_dependency_log(str(payload))
                elif kind == "dependency_done":
                    self._handle_dependency_done(payload)
                    if self.close_pending:
                        self._finish_pending_close()
                        return
                elif kind == "dependency_canceled":
                    self._handle_dependency_failure(str(payload), canceled=True)
                    if self.close_pending:
                        self._finish_pending_close()
                        return
                elif kind == "dependency_error":
                    self._handle_dependency_failure(str(payload), canceled=False)
                    if self.close_pending:
                        self._finish_pending_close()
                        return
                elif kind == "dependency_latest":
                    self._handle_latest_release_result(payload)
                elif kind == "dependency_latest_error":
                    self._append_dependency_log(f"Latest-version check failed: {payload}")
                    self.status_var.set("Latest dependency metadata check failed; existing capability results are unchanged.")
                elif kind == "analysis_health_started":
                    self.analysis_progress_var.set("Fast full-file source-health scan starting…")
                    self.source_health_var.set("Source health: scanning every compressed video packet timestamp…")
                    self.source_health_label.configure(style="HealthWarn.TLabel")
                elif kind == "analysis_health_progress":
                    packets, fraction = payload  # type: ignore[misc]
                    if fraction is None:
                        self.analysis_progress_var.set(f"Source-health scan · {int(packets):,} video packets")
                    else:
                        self.analysis_progress_var.set(
                            f"Source-health scan · {float(fraction):.0%} · {int(packets):,} video packets"
                        )
                elif kind == "analysis_health_done":
                    media, health, cache_key, cached = payload  # type: ignore[misc]
                    self.health_scan_media = media
                    self._set_source_health(health)
                    self.source_health_cache[cache_key] = health
                    while len(self.source_health_cache) > 8:
                        self.source_health_cache.pop(next(iter(self.source_health_cache)))
                    result_kind = "reused for unchanged file" if cached else "complete"
                    self.analysis_progress_var.set(f"Fast source-health scan {result_kind} · starting IDet…")
                    self._set_text(
                        self.summary_text,
                        health_details(health) + "\n\nInterlace analysis is still in progress…",
                    )
                elif kind == "analysis_progress":
                    done, total, offset, mode = payload  # type: ignore[misc]
                    if mode == "full":
                        self.analysis_progress_var.set("Full-file scan in progress…")
                    else:
                        self.analysis_progress_var.set(f"IDet sample {done}/{total} · near {offset:.1f}s")
                elif kind == "analysis_done":
                    self.media, self.analysis, health = payload  # type: ignore[misc]
                    self.health_scan_media = self.media
                    automatic_reanalysis = bool(
                        self.auto_workflow and self.auto_workflow.stage == "reanalyzing"
                    )
                    self._set_source_health(health)
                    self._set_busy(None)
                    if automatic_reanalysis:
                        self._restoring_automatic_settings = True
                        try:
                            self._update_cadence_labels()
                        finally:
                            self._restoring_automatic_settings = False
                    else:
                        self._update_cadence_labels()
                    self._refresh_control_states()
                    self.analysis_progress_var.set(
                        f"{self.analysis.mode.title()} IDet complete · {self.analysis.classification.upper()}"
                    )
                    self._set_text(self.summary_text, self._analysis_summary())
                    if automatic_reanalysis and self.auto_workflow:
                        self._restore_automatic_settings(self.auto_workflow.final_settings)
                    else:
                        self._suggest_output()
                    self._refresh_plan()
                    if automatic_reanalysis and self.auto_workflow:
                        if health.repair_required:
                            self._stop_automatic_recovery(
                                "Automatic recovery stopped: the validated repair copy still has a repair-required "
                                "packet timeline. No deinterlace encode was started.",
                                show_error=True,
                            )
                        elif not self.plan or not self.plan.valid:
                            self._stop_automatic_recovery(
                                "Automatic recovery stopped after repair because the fresh processing plan is invalid "
                                "or ambiguous; see Plan & command. No encode was started.",
                                show_error=True,
                            )
                        else:
                            self.auto_workflow.stage = "deinterlacing"
                            self.run_detail_var.set(f"{FINAL_PROCESSING_STAGE} · queued")
                            self.status_var.set(
                                "Repair and re-analysis passed; final processing is starting the validated output plan…"
                            )
                            self.root.after_idle(lambda: self._start_processing(automatic=True))
                    elif health.repair_required and self.auto_repair_continue_var.get():
                        self.root.after_idle(self._maybe_begin_automatic_recovery)
                    if self.close_pending:
                        self._finish_pending_close()
                        return
                elif kind == "analysis_canceled":
                    self._set_busy(None)
                    self.analysis_progress_var.set("Analysis canceled")
                    if self.source_health is None:
                        self.source_health_var.set("Source health: scan canceled — no current result.")
                        self.source_health_label.configure(style="HealthWarn.TLabel")
                    if self.auto_workflow and self.auto_workflow.stage == "reanalyzing":
                        self._stop_automatic_recovery(
                            "Automatic recovery canceled during repaired-source analysis; no encode was started."
                        )
                    else:
                        self.status_var.set(str(payload))
                    if self.close_pending:
                        self._finish_pending_close()
                        return
                elif kind == "analysis_error":
                    self._set_busy(None)
                    self.analysis_progress_var.set("Analysis failed")
                    if self.source_health is None:
                        self.source_health_var.set("Source health: analysis failed — no current result.")
                        self.source_health_label.configure(style="HealthError.TLabel")
                    automatic_reanalysis = bool(
                        self.auto_workflow and self.auto_workflow.stage == "reanalyzing"
                    )
                    if automatic_reanalysis:
                        self._stop_automatic_recovery(
                            f"Automatic recovery stopped because repaired-source analysis failed: {payload}",
                            show_error=False,
                        )
                    else:
                        self.status_var.set("Analysis failed.")
                    if self.close_pending:
                        self._finish_pending_close()
                        return
                    elif not automatic_reanalysis:
                        messagebox.showerror("Analysis failed", str(payload), parent=self.root)
                    else:
                        messagebox.showerror(
                            "Automatic recovery stopped",
                            f"Repaired-source analysis failed: {payload}\n\nNo encode was started.",
                            parent=self.root,
                        )
                elif kind == "run_log":
                    self._append_log(str(payload))
                elif kind == "run_progress":
                    self._handle_run_progress(payload)  # type: ignore[arg-type]
                elif kind == "run_done":
                    self._handle_run_done(payload)
                    if self.close_pending:
                        return
                elif kind == "batch_event":
                    batch_kind, record, batch_payload = payload  # type: ignore[misc]
                    self._handle_batch_event(str(batch_kind), record, batch_payload)
                    if self.close_pending and self.busy_kind is None:
                        return
                elif kind == "batch_start_error":
                    self._set_busy(None)
                    self.started_at = None
                    self.batch_runner = None
                    self.run_detail_var.set("Batch stopped")
                    self.batch_status_var.set(f"Batch stopped safely: {payload}")
                    self.status_var.set(self.batch_status_var.get())
                    self._append_log("Batch coordinator stopped: " + str(payload))
                    self._refresh_batch_tree()
                    if self.close_pending:
                        self._finish_pending_close()
                        return
                    messagebox.showerror(
                        "Batch processing stopped",
                        f"The batch coordinator stopped safely:\n\n{payload}\n\n"
                        "Completed outputs remain validated; no source file was modified.",
                        parent=self.root,
                    )
                elif kind == "repair_log":
                    self._append_log(str(payload))
                elif kind == "repair_progress":
                    self._handle_repair_progress(payload)  # type: ignore[arg-type]
                elif kind == "repair_done":
                    self._handle_repair_done(payload)
                    if self.close_pending:
                        return
                elif kind == "repair_start_error":
                    self._set_busy(None)
                    self.started_at = None
                    self.repairer = None
                    self.run_detail_var.set("Repair failed")
                    automatic_repair = bool(
                        self.auto_workflow and self.auto_workflow.stage == "repairing"
                    )
                    if automatic_repair:
                        self._stop_automatic_recovery(
                            f"Automatic recovery could not start source repair: {payload}",
                            show_error=False,
                        )
                    else:
                        self.status_var.set("Source repair could not start.")
                    if self.close_pending:
                        self._finish_pending_close()
                        return
                    messagebox.showerror(
                        "Automatic recovery stopped" if automatic_repair else "Source repair failed",
                        (
                            f"Source repair could not start: {payload}\n\nNo deinterlace encode was started."
                            if automatic_repair
                            else str(payload)
                        ),
                        parent=self.root,
                    )
                elif kind == "compatibility_log":
                    self._append_log(str(payload))
                elif kind == "compatibility_progress":
                    self._handle_compatibility_progress(payload)  # type: ignore[arg-type]
                elif kind == "compatibility_done":
                    self._handle_compatibility_done(payload)
                    if self.close_pending:
                        return
        except queue.Empty:
            pass
        self.poll_after_id = self.root.after(100, self._poll_events)

    def _cancel_event_poll(self) -> None:
        after_id = self.poll_after_id
        self.poll_after_id = None
        if after_id:
            try:
                self.root.after_cancel(after_id)
            except Exception:
                pass
        refresh_id = self.plan_refresh_after_id
        self.plan_refresh_after_id = None
        if refresh_id:
            try:
                self.root.after_cancel(refresh_id)
            except Exception:
                pass

    def _root_destroyed(self, event) -> None:
        if event.widget is self.root:
            self._cancel_event_poll()

    def _handle_run_progress(self, values: dict[str, str]) -> None:
        phase = values.get("phase", "encode_progress")
        prefix = f"{FINAL_PROCESSING_STAGE} · " if self.auto_workflow else ""

        def set_phase(detail: str, status: str, progress: float | None = None) -> None:
            self.run_phase_detail = prefix + detail
            self.run_detail_var.set(self.run_phase_detail)
            self.status_var.set(status)
            if progress is not None:
                self.progress_var.set(max(0.0, min(100.0, progress)))

        if phase == "preflight_indexed_start":
            set_phase(
                "indexed source check",
                "Checking the unchanged FFV1 packet timeline against the VSPipe graph; no frames are being encoded…",
                0.0,
            )
            return
        if phase == "preflight_full_start":
            set_phase(
                "full decoded preflight · starting",
                "Running a cancellable full decoded source check; no output encode has started…",
                0.0,
            )
            return
        if phase == "preflight_full_progress":
            frame = values.get("frame", "?")
            expected = values.get("expected_frames")
            speed = values.get("speed") or "calculating speed"
            eta = values.get("eta_seconds")
            try:
                percent = float(values["percent"])
            except (KeyError, ValueError):
                percent = None
            count = f"{frame}/{expected}" if expected else frame
            try:
                eta_text = (
                    f" · ETA {int(float(eta)) // 60:02d}:{int(float(eta)) % 60:02d}"
                    if eta is not None
                    else ""
                )
            except ValueError:
                eta_text = ""
            set_phase(
                f"source check · {count} · {speed}{eta_text}",
                "Verifying every source frame before the VSPipe job; no output encode has started…",
                percent,
            )
            return
        if phase == "preflight_complete":
            method = values.get("method", "verified")
            elapsed = values.get("elapsed_seconds")
            try:
                elapsed_text = f" · {float(elapsed):.1f}s" if elapsed else ""
            except ValueError:
                elapsed_text = ""
            label = (
                "indexed source contract"
                if method == "vspipe_info_ffv1_packet_contract"
                else "source preflight"
            )
            set_phase(
                f"{label} complete{elapsed_text}",
                "Source contract passed; preparing the encoder…",
                100.0,
            )
            return
        if phase == "encode_start":
            set_phase(
                "encoder/VSPipe starting",
                "Source check passed; starting the unique partial output encode…",
                0.0,
            )
            return
        if phase == "encode_progress":
            frame = values.get("frame", "?").strip()
            expected = values.get("expected_frames")
            speed = values.get("speed", "?")
            progress: float | None = None
            try:
                if expected and int(expected) > 0:
                    progress = 100.0 * int(frame) / int(expected)
            except ValueError:
                progress = None
            if progress is None:
                try:
                    out_time = int(
                        values.get("out_time_us", values.get("out_time_ms", "0"))
                    ) / 1_000_000
                except ValueError:
                    out_time = 0.0
                if self.media and self.media.duration:
                    progress = 100.0 * out_time / self.media.duration
            count = f"{frame}/{expected}" if expected else frame
            set_phase(
                f"encoding · frame {count} · {speed}",
                (
                    f"{FINAL_PROCESSING_STAGE}: encoding to a unique partial file…"
                    if self.auto_workflow and self.auto_workflow.stage == "deinterlacing"
                    else "Encoding to a unique partial file…"
                ),
                progress,
            )
            return
        if phase == "encode_complete":
            set_phase("encode complete · validating", "Encoder finished; validating the partial output…", 100.0)
            return
        if phase in {"validation_start", "final_validation_start"}:
            set_phase(
                "validating output",
                "Validating frame count, streams, geometry, and progressive output…",
                100.0,
            )
            return
        if phase in {"validation_complete", "final_validation_complete"}:
            set_phase(
                "validation passed",
                "Output validation passed; completing protected promotion…",
                100.0,
            )
            return
        if phase == "hash_start":
            set_phase(
                "calculating output SHA-256",
                "Final reopen passed; calculating the output checksum…",
                100.0,
            )
            return
        if phase == "hash_complete":
            set_phase(
                "checksum complete",
                "Checksum complete; promoting the linked audit sidecars…",
                100.0,
            )
            return
        if phase == "job_complete":
            set_phase("validated", "Processing and all final validation completed.", 100.0)
            return

        set_phase(f"processing · {phase}", "Processing the selected video…")

    def _queue_late_source_fault_recovery(self, result) -> bool:
        """Promote newly decoded source damage into the existing QTGMC recovery chain."""

        if (
            getattr(result, "failure_code", None) != SOURCE_REPAIR_REQUIRED_FAILURE
            or not self.plan
            or not automatic_recovery_applies_to_backend(self.plan.selected_backend)
            or not self.media
            or not self.analysis
            or not self.source_health
        ):
            return False
        source = Path(self.input_var.get().strip())
        if (
            not source.is_file()
            or not health_matches_source(self.source_health, source)
        ):
            return False

        reason = (
            "The managed full decoded source preflight found picture/timeline damage that the fast compressed-"
            f"packet scan could not prove: {result.message}"
        )
        self._set_source_health(replace(self.source_health, status="repair_required", reason=reason))
        self._set_text(self.summary_text, self._analysis_summary())
        self._refresh_plan()
        if self.auto_workflow is not None or not self.auto_repair_continue_var.get():
            return False
        self.progress_var.set(0)
        self.run_phase_detail = "Automatic recovery 1/3 · queued after decoded-damage confirmation"
        self.run_detail_var.set(self.run_phase_detail)
        retained_log = f" Preflight log retained at {result.log_path}." if result.log_path else ""
        self.status_var.set(
            "Decoded source damage confirmed; automatic QTGMC recovery is starting a separate repair copy."
            + retained_log
        )
        self.root.after_idle(self._maybe_begin_automatic_recovery)
        return True

    def _handle_run_done(self, result) -> None:
        automatic_workflow = (
            self.auto_workflow
            if self.auto_workflow and self.auto_workflow.stage == "deinterlacing"
            else None
        )
        self._set_busy(None)
        self.started_at = None
        self.processor = None
        if self.close_pending:
            self._finish_pending_close()
            return
        if not result.success and not result.canceled and self._queue_late_source_fault_recovery(result):
            return
        if result.success:
            self.progress_var.set(100)
            self.last_completed_output = result.output_path
            self.status_var.set(f"Completed and validated: {result.output_path}")
            self.run_phase_detail = "Validated"
            self.run_detail_var.set("Validated")
            if automatic_workflow:
                repair_source = (
                    automatic_workflow.validated_repair_source
                    or automatic_workflow.original_source
                )
                repair_source_label = (
                    "Validated repair copy"
                    if automatic_workflow.validated_repair_source
                    else "Validated source (complete diagnosis required no repair copy)"
                )
                messagebox.showinfo(
                    "Automatic repair and processing complete",
                    f"Original source (unchanged):\n{automatic_workflow.original_source}"
                    f"\n\n{repair_source_label}:\n{repair_source}"
                    f"\nRepair method: {automatic_workflow.repair_method}"
                    f"\nRepair audit: {automatic_workflow.repair_report_path}"
                    f"\n\nValidated final output:\n{result.output_path}"
                    f"\nSHA-256:\n{result.output_sha256}",
                    parent=self.root,
                )
            else:
                messagebox.showinfo(
                    "Deinterlacing complete",
                    f"Validated output:\n{result.output_path}\n\nSHA-256:\n{result.output_sha256}",
                    parent=self.root,
                )
        elif result.canceled:
            self.status_var.set(
                (
                    "Final processing canceled. "
                    f"Diagnostic log retained at {result.log_path}"
                )
                if automatic_workflow
                else f"Canceled. Diagnostic log retained at {result.log_path}"
            )
            self.run_detail_var.set("Final processing canceled" if automatic_workflow else "Canceled")
            self.run_phase_detail = self.run_detail_var.get()
        else:
            self.status_var.set(
                f"Final processing stopped: {result.message}"
                if automatic_workflow
                else f"Failed: {result.message}"
            )
            self.run_detail_var.set("Final processing failed" if automatic_workflow else "Failed")
            self.run_phase_detail = self.run_detail_var.get()
            messagebox.showerror(
                "Final processing stopped" if automatic_workflow else "Processing failed",
                f"{result.message}\n\nDiagnostic log: {result.log_path or 'unavailable'}"
                + (
                    "\n\nSource health is now SOURCE REPAIR NEEDED. Enable Automatic QTGMC recovery and "
                    "start again, or click Repair required… to create a separate validated copy."
                    if getattr(result, "failure_code", None) == SOURCE_REPAIR_REQUIRED_FAILURE
                    and not automatic_workflow
                    else "\n\nThe repaired copy failed its decoded source contract; recursive repair was not started."
                    if getattr(result, "failure_code", None) == SOURCE_REPAIR_REQUIRED_FAILURE
                    and automatic_workflow
                    else ""
                )
                + (f"\nQuarantined candidate: {result.quarantine_path}" if result.quarantine_path else ""),
                parent=self.root,
            )
        if automatic_workflow:
            self.auto_workflow = None
            self.auto_repair_check.configure(state="normal")
        completed_status = self.status_var.get()
        self._refresh_plan()
        self.status_var.set(completed_status)

    def _handle_repair_progress(self, values: dict[str, str]) -> None:
        phase = values.get("phase", "Repair")
        current_us = 0
        duration_us = 0
        try:
            current_us = int(values.get("out_time_us", values.get("out_time_ms", "0")))
            duration_us = int(values.get("duration_us", "0"))
        except ValueError:
            pass
        if duration_us <= 0 and self.media and self.media.duration:
            duration_us = round(self.media.duration * 1_000_000)
        if duration_us > 0:
            self.progress_var.set(min(100.0, max(0.0, 100.0 * current_us / duration_us)))
        frame = values.get("frame")
        speed = values.get("speed")
        details = [phase]
        if frame:
            details.append(f"frame {frame}")
        if speed:
            details.append(speed)
        automatic = bool(self.auto_workflow and self.auto_workflow.stage == "repairing")
        prefix = "Automatic recovery 1/3 · " if automatic else ""
        self.run_detail_var.set(prefix + " · ".join(details))
        self.status_var.set(
            f"Automatic recovery 1/3: {phase}…" if automatic else f"Repair: {phase}…"
        )

    def _handle_repair_done(self, result) -> None:
        automatic_repair = bool(
            self.auto_workflow and self.auto_workflow.stage == "repairing"
        )
        self._set_busy(None)
        self.started_at = None
        self.repairer = None
        if self.close_pending:
            self._finish_pending_close()
            return
        if automatic_repair:
            self._handle_automatic_repair_done(result)
            return
        if result.success and result.output_path:
            self.progress_var.set(100)
            self.last_completed_output = result.output_path
            self.run_detail_var.set("Repair validated")
            self.status_var.set(f"Validated repair copy: {result.output_path}")
            limitation = ""
            if result.method == "ffv1_rescue":
                limitation = (
                    f"\n\nMeasured net repeated/materialized frame slots: {result.repeated_frames}."
                    "\nUnavailable pictures were not reconstructed; inspect the reported damaged interval."
                )
            messagebox.showinfo(
                "Source repair complete",
                f"Validated repair copy:\n{result.output_path}\n\nMethod: {result.method}"
                f"\nSHA-256:\n{result.output_sha256}{limitation}"
                f"\n\nAudit report:\n{result.report_path}"
                "\n\nThe repair copy will now become the selected source and receive fresh Probe/IDet analysis.",
                parent=self.root,
            )
            self._select_input_path(result.output_path)
            self.root.after_idle(lambda: self._analyze("sampled"))
        elif result.success:
            self.progress_var.set(100)
            self.run_detail_var.set("No repair needed")
            self.status_var.set(result.message)
            detail = diagnosis_summary(result.source_diagnosis) if result.source_diagnosis else result.message
            messagebox.showinfo(
                "Source timeline is healthy",
                f"{result.message}\n\n{detail}\n\nDiagnostic report:\n{result.report_path}",
                parent=self.root,
            )
        elif result.canceled:
            self.run_detail_var.set("Repair canceled")
            self.status_var.set(f"Source repair canceled. Diagnostic log retained at {result.log_path}")
        else:
            self.run_detail_var.set("Repair failed")
            self.status_var.set(f"Source repair failed: {result.message}")
            measured = (
                "\n\nMeasured diagnosis:\n" + diagnosis_summary(result.source_diagnosis)
                if result.source_diagnosis
                else ""
            )
            messagebox.showerror(
                "Source repair failed",
                f"{result.message}{measured}\n\nDiagnostic log: {result.log_path or 'unavailable'}"
                f"\nDiagnostic report: {result.report_path or 'unavailable'}"
                + (f"\nQuarantined candidate: {result.quarantine_path}" if result.quarantine_path else ""),
                parent=self.root,
            )
        if self.media and self.analysis:
            self._refresh_plan()

    def _handle_automatic_repair_done(self, result) -> None:
        workflow = self.auto_workflow
        if not workflow or workflow.stage != "repairing":
            return
        if result.success:
            workflow.repair_method = result.method or "none"
            workflow.repair_output_sha256 = result.output_sha256
            workflow.repair_log_path = result.log_path
            workflow.repair_report_path = result.report_path
            workflow.repeated_frames = result.repeated_frames
            workflow.dropped_frames = result.dropped_frames
        if result.success and result.output_path:
            workflow.validated_repair_source = result.output_path
            workflow.stage = "reanalyzing"
            self.progress_var.set(100)
            self.run_detail_var.set("Automatic recovery 2/3 · queued")
            self.status_var.set(
                "Automatic recovery 1/3 passed. Selecting the validated repair copy and starting fresh analysis…"
            )
            self._restoring_automatic_settings = True
            try:
                self._select_input_path(result.output_path, suggest_output=False)
            finally:
                self._restoring_automatic_settings = False
            self._restore_automatic_settings(workflow.final_settings)
            self.root.after_idle(lambda: self._analyze(workflow.analysis_mode))
            return
        if result.success:
            # Complete decoded diagnosis is authoritative when the fast packet scan was conservative.
            cleared = replace(
                workflow.trigger_health,
                status="warning",
                reason=(
                    "The fast packet precheck requested repair, but the complete decoded diagnosis and QTGMC "
                    "compatibility check found no repair was needed; automatic processing may continue."
                ),
            )
            self._set_source_health(cleared)
            workflow.stage = "deinterlacing"
            self._restore_automatic_settings(workflow.final_settings)
            self._refresh_plan()
            if not self.plan or not self.plan.valid:
                self._stop_automatic_recovery(
                    "Automatic recovery stopped after the complete diagnosis because the final plan is invalid; "
                    "see Plan & command. No encode was started.",
                    show_error=True,
                )
                return
            self.run_detail_var.set(f"{FINAL_PROCESSING_STAGE} · queued")
            self.status_var.set(
                "Complete diagnosis found no repair copy was needed; starting the validated final plan…"
            )
            self.root.after_idle(lambda: self._start_processing(automatic=True))
            return
        if result.canceled:
            self._stop_automatic_recovery(
                f"Automatic recovery canceled during repair. Diagnostic log retained at {result.log_path}"
            )
            return
        measured = (
            "\n\nMeasured diagnosis:\n" + diagnosis_summary(result.source_diagnosis)
            if result.source_diagnosis
            else ""
        )
        message = (
            f"Automatic recovery stopped because source repair failed: {result.message}{measured}"
            f"\n\nDiagnostic log: {result.log_path or 'unavailable'}"
            f"\nDiagnostic report: {result.report_path or 'unavailable'}"
            + (f"\nQuarantined candidate: {result.quarantine_path}" if result.quarantine_path else "")
            + "\n\nNo deinterlace encode was started."
        )
        self._stop_automatic_recovery(message, show_error=True)

    def _update_elapsed(self) -> None:
        if self.busy_kind in {"processing", "repair", "compatibility", "batch"} and self.started_at is not None:
            elapsed = int(time.monotonic() - self.started_at)
            if self.busy_kind in {"processing", "compatibility", "batch"}:
                self.run_detail_var.set(
                    f"{self.run_phase_detail} · total {elapsed // 60:02d}:{elapsed % 60:02d}"
                )
            elif "frame" not in self.run_detail_var.get():
                self.run_detail_var.set(f"elapsed {elapsed // 60:02d}:{elapsed % 60:02d}")
            self.root.after(1000, self._update_elapsed)

    def _refresh_capabilities(self) -> None:
        if self.busy_kind:
            return
        self._set_busy("capabilities")
        self.capability_scan_started_at = time.monotonic()
        self.dependency_var.set(
            "Inspecting FFmpeg filters/encoders, VSPipe/QTGMC, Vulkan NNEDI3, temporal-denoise graphs, and NVIDIA capabilities…"
        )
        self.dependency_label.configure(style="Warn.TLabel")

        def worker() -> None:
            try:
                caps = inspect_capabilities(
                    self.persisted.get("ffmpeg_path") or None,
                    self.persisted.get("ffprobe_path") or None,
                    self.persisted.get("vspipe_path") or None,
                )
                self.events.put(("capabilities_done", caps))
            except Exception as exc:
                self.events.put(("capabilities_error", str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def _show_capabilities(self) -> None:
        assert self.capabilities
        ffmpeg = self.capabilities.ffmpeg_version or "FFmpeg not found"
        if len(ffmpeg) > 110:
            ffmpeg = ffmpeg[:107] + "…"
        vs = self.capabilities.vapoursynth_version or "not found"
        qtgmc = "QTGMC ready" if self.capabilities.qtgmc_ready else "QTGMC dependencies missing"
        vulkan = "Vulkan NNEDI3 ready" if self.capabilities.vulkan_nnedi3_ready else "Vulkan NNEDI3 optional/unavailable"
        gpu = self.capabilities.gpu_name or "NVIDIA GPU not detected"
        denoise_ready = sum(
            bool(self.capabilities.denoise_capabilities.get(identifier, False))
            for identifier in DENOISER_BY_ID
        )
        gpu_denoise = any(
            denoiser_backend_has_gpu(self.capabilities.denoise_backends.get(identifier))
            for identifier in DENOISER_BY_ID
        )
        denoise_status = f"temporal denoisers {denoise_ready}/{len(DENOISER_BY_ID)} ready"
        if self.capabilities.gpu_name:
            denoise_status += " (GPU graphs active)" if gpu_denoise else " (verified CPU graphs; optional GPU update available)"
        issues = dependency_issues(self.capabilities)
        self.dependency_var.set(
            f"{ffmpeg} · VapourSynth {vs} · {qtgmc} · {vulkan} · {denoise_status} · {gpu}"
        )
        self.dependency_label.configure(style="Good.TLabel" if not issues else "Warn.TLabel")
        self._refresh_repair_button_state()
        self._refresh_control_states()
        elapsed = (
            f" in {self.last_capability_scan_seconds:.1f} s"
            if self.last_capability_scan_seconds is not None
            else ""
        )
        candidate_count = sum(
            line.startswith("SELECTED [") or line.startswith("NOT SELECTED [")
            for line in self.capabilities.ffmpeg_discovery_diagnostics
        )
        candidate_summary = (
            f"; evaluated {candidate_count} paired FFmpeg installation{'s' if candidate_count != 1 else ''}"
            if candidate_count
            else ""
        )
        self.status_var.set(f"Dependency scan completed{elapsed}{candidate_summary}. Select and analyze a source.")
        if issues and self.auto_dependency_offer and not self.dependency_offer_shown and not self.close_pending:
            self.dependency_offer_shown = True
            self.root.after(250, self._offer_app_local_install)

    def _select_ffmpeg(self) -> None:
        chosen = filedialog.askopenfilename(title="Select ffmpeg.exe", filetypes=[("FFmpeg", "ffmpeg.exe"), ("Executables", "*.exe")])
        if not chosen:
            return
        ffmpeg = Path(chosen)
        ffprobe = ffmpeg.with_name("ffprobe.exe")
        self.persisted["ffmpeg_path"] = str(ffmpeg)
        self.persisted["ffprobe_path"] = str(ffprobe) if ffprobe.is_file() else ""
        self._refresh_capabilities()

    def _select_vspipe(self) -> None:
        chosen = filedialog.askopenfilename(title="Select vspipe.exe", filetypes=[("VSPipe", "vspipe.exe"), ("Executables", "*.exe")])
        if chosen:
            self.persisted["vspipe_path"] = str(Path(chosen))
            self._refresh_capabilities()

    def _dependency_doctor(self) -> None:
        if self.dependency_dialog and self.dependency_dialog.winfo_exists():
            self.dependency_dialog.deiconify()
            self.dependency_dialog.lift()
            return
        dialog = Toplevel(self.root)
        self.dependency_dialog = dialog
        dialog.title("Dependency doctor and app-local installer")
        dialog.geometry("900x650")
        dialog.minsize(760, 500)
        dialog.transient(self.root)
        text = scrolledtext.ScrolledText(dialog, wrap="word", font=("Cascadia Mono", 9))
        text.pack(fill="both", expand=True, padx=10, pady=10)
        self.dependency_doctor_text = text
        text.insert("1.0", self._dependency_doctor_content())
        text.configure(state="disabled")

        progress = ttk.Progressbar(dialog, mode="determinate", maximum=100)
        progress.pack(fill="x", padx=10, pady=(0, 8))
        self.dependency_progress = progress
        buttons = ttk.Frame(dialog, padding=(10, 0, 10, 10))
        buttons.pack(fill="x")

        def copy_command() -> None:
            if self.capabilities and self.capabilities.qtgmc_install_command:
                self.root.clipboard_clear()
                self.root.clipboard_append(self.capabilities.qtgmc_install_command)

        self.dependency_install_button = ttk.Button(
            buttons, text="Install/update app-local tools…", command=self._start_dependency_install
        )
        self.dependency_install_button.pack(side="left")
        ttk.Button(buttons, text="Check current stable versions", command=self._check_latest_dependencies).pack(
            side="left", padx=(6, 0)
        )
        ttk.Button(buttons, text="Copy manual QTGMC command", command=copy_command).pack(side="left", padx=(6, 0))
        self.dependency_cancel_button = ttk.Button(
            buttons, text="Cancel install", command=self._cancel_active, state="disabled"
        )
        self.dependency_cancel_button.pack(side="left", padx=(6, 0))
        ttk.Button(buttons, text="Close", command=self._close_dependency_doctor).pack(side="right")
        dialog.protocol("WM_DELETE_WINDOW", self._close_dependency_doctor)

    def _dependency_doctor_content(self) -> str:
        runtime = managed_runtime_root()
        if not self.capabilities:
            capability_lines = ["Capability scan has not completed."]
            issues: dict[str, tuple[str, ...]] = {}
        else:
            caps = self.capabilities
            issues = dependency_issues(caps)
            capability_lines = [
                f"FFmpeg: {caps.ffmpeg_path or 'NOT FOUND'}",
                f"FFprobe: {caps.ffprobe_path or 'NOT FOUND'}",
                f"Version: {caps.ffmpeg_version or 'unknown'}",
                f"FFprobe version: {caps.ffprobe_version or 'unknown'}",
                f"Version classification: {caps.ffmpeg_version_kind or 'unverified'}",
                f"Version assessment: {caps.ffmpeg_version_diagnostic or 'no version evidence'}",
                f"Git revisions: FFmpeg {caps.ffmpeg_git_revision or 'n/a'} · FFprobe {caps.ffprobe_git_revision or 'n/a'}",
                f"FFmpeg required libraries: {caps.ffmpeg_library_versions or 'not recorded'}",
                f"FFprobe required libraries: {caps.ffprobe_library_versions or 'not recorded'}",
                f"Selection source: {caps.ffmpeg_selection_source or 'none'}",
                f"BWDIF CPU/CUDA: {'bwdif' in caps.filters} / {'bwdif_cuda' in caps.filters}",
                f"VSPipe: {caps.vspipe_path or 'NOT FOUND'}",
                f"VapourSynth: {caps.vapoursynth_version or 'unknown'}",
                f"QTGMC ready: {caps.qtgmc_ready}",
                f"QTGMC diagnostic:\n{caps.qtgmc_diagnostic}",
                f"Vulkan NNEDI3 ready: {caps.vulkan_nnedi3_ready}",
                f"Vulkan NNEDI3 package: {caps.vulkan_nnedi3_package_version or 'not installed/verified'}",
                f"Vulkan NNEDI3 diagnostic:\n{caps.vulkan_nnedi3_diagnostic}",
                "Temporal denoiser graph checks:",
                *[
                    (
                        f"{DENOISER_BY_ID[identifier].label}: ready={caps.denoise_capabilities.get(identifier, False)}; "
                        f"implementation={denoiser_backend_display(identifier, caps.denoise_backends.get(identifier))}; "
                        f"diagnostic={caps.denoise_diagnostics.get(identifier, 'not scanned')}"
                    )
                    for identifier in DENOISER_BY_ID
                ],
                "",
                f"GPU: {caps.gpu_name or 'not detected'} · {caps.gpu_memory_mib or '?'} MiB · driver {caps.gpu_driver or '?'}",
                f"Verified NVENC coded depths: {caps.encoder_verified_bit_depths or 'none'}",
                *[f"{name}: {detail}" for name, detail in caps.encoder_runtime_diagnostics.items()],
                "",
                "FFmpeg 9 interlace capability audit:",
                *[
                    f"{INTERLACE_DIAGNOSTIC_LABELS.get(name, name)}: {detail}"
                    for name, detail in caps.interlace_runtime_diagnostics.items()
                ],
                "",
                "FFmpeg discovery evidence:",
                *(caps.ffmpeg_discovery_diagnostics or ("No discovery diagnostics were recorded.",)),
            ]
        issue_lines = [
            f"{component}: {message}"
            for component, messages in issues.items()
            for message in messages
        ]
        return "\n".join(
            [
                *capability_lines,
                "",
                "Compatibility assessment:",
                *(issue_lines or ["READY — the discovered toolchain passes the required capability checks."]),
                "",
                f"Managed runtime folder:\n{runtime}",
                "",
                "The installer resolves the current stable FFmpeg and VapourSynth releases when it starts, "
                "downloads only over HTTPS, verifies published SHA-256 values, safely stages/extracts the archives, "
                "and activates them only after FFmpeg, the full BestSource/QTGMC graph, and every temporal-denoise capability passes validation. "
                "The official Vulkan NNEDI3 wheel is installed only inside this staged runtime and is exposed only if its real QTGMC graph passes.",
                "It does not change system PATH, the registry, system Python, file associations, or existing system tools.",
                "A canceled or failed update leaves the previously selected runtime unchanged.",
                "Sources/licenses: Gyan full FFmpeg build (GPLv3), official VapourSynth portable release (LGPL-2.1+), "
                "Python Software Foundation embedded Python, VSJetpack (MIT), and optional vapoursynth-nnedi3vk (GPL-3.0) plus native plugins under their upstream licenses. "
                "Optional NVIDIA graphs may install BM3DCUDA (GPL-2.0-or-later), DFTTest2 NVRTC (GPL-3.0), "
                "vszipcu (MIT), and NVIDIA CUDA NVRTC (proprietary). CPU V-BM3D, DFTTest2, MVTools, and "
                "NLMeans remain required fallbacks.",
                "",
                "Cadence reminder: field-rate output preserves one progressive frame for every temporal field; "
                "frame-rate output creates one progressive frame per interlaced source frame. Both preserve the source playing speed, and the control shows exact rates after analysis.",
                "",
                "Current FFmpeg DNxHR encoding is limited to 10-bit. Twelve-bit DNxHR remains disabled unless a future selected encoder explicitly reports yuv444p12le.",
            ]
        )

    def _close_dependency_doctor(self) -> None:
        if self.busy_kind == "dependencies":
            self.status_var.set("Dependency installation is still active. Cancel it before closing the dependency window.")
            return
        if self.dependency_dialog:
            self.dependency_dialog.destroy()
        self.dependency_dialog = None
        self.dependency_doctor_text = None
        self.dependency_progress = None
        self.dependency_install_button = None
        self.dependency_cancel_button = None

    def _offer_app_local_install(self) -> None:
        if self.close_pending or self.busy_kind or not self.capabilities:
            return
        issues = dependency_issues(self.capabilities)
        if not issues:
            return
        rendered = "\n".join(
            f"• {component}: {message}" for component, messages in issues.items() for message in messages
        )
        selected = ""
        if self.capabilities.ffmpeg_path:
            selected = (
                "\n\nSelected FFmpeg:\n"
                f"{self.capabilities.ffmpeg_path}\n"
                f"Source: {self.capabilities.ffmpeg_selection_source or 'unknown'}\n"
                f"Reported: {self.capabilities.ffmpeg_version or 'unknown'}"
            )
        if messagebox.askyesno(
            "Install missing video tools locally?",
            "The dependency scan completed and found compatibility issues:\n\n"
            f"{rendered}{selected}\n\n"
                "Tools → Dependency doctor shows every FFmpeg candidate that was evaluated.\n\n"
                "Install the current stable missing components into this app's own subfolder? "
                "This will not change system PATH, Python, registry, or existing installations.\n\n"
                "Sources include the GPLv3 Gyan FFmpeg build, LGPL-2.1+ VapourSynth, PSF Python, MIT VSJetpack, "
                "and narrowly scoped optional BM3DCUDA (GPL-2.0-or-later), DFTTest2 NVRTC (GPL-3.0), "
                "vszipcu (MIT), and NVIDIA CUDA NVRTC (proprietary) packages.",
            parent=self.root,
        ):
            self._dependency_doctor()
            self._start_dependency_install(confirmed=True, requested_components=frozenset(issues))

    def _start_dependency_install(
        self,
        *,
        confirmed: bool = False,
        requested_components: frozenset[str] | None = None,
    ) -> None:
        if self.busy_kind:
            return
        if not self.dependency_dialog or not self.dependency_dialog.winfo_exists():
            self._dependency_doctor()
        issues = dependency_issues(self.capabilities)
        components = requested_components or frozenset(issues) or frozenset({"ffmpeg", "vapoursynth"})
        if not confirmed:
            labels = " and ".join(sorted(components))
            confirmed = messagebox.askyesno(
                "Install app-local dependencies?",
                f"Install/update {labels} using current stable releases?\n\n"
                f"Destination:\n{managed_runtime_root()}\n\n"
                "The full FFmpeg package is roughly 250 MB to download; the tested complete runtime with optional "
                "NVIDIA denoisers occupies about 1.74 GiB. "
                "The current active runtime remains selected until every staged check passes.\n\n"
                "Sources/licenses: Gyan FFmpeg full build (GPLv3), VapourSynth (LGPL-2.1+), PSF Python, "
                "VSJetpack (MIT), and optional graph-tested BM3DCUDA (GPL-2.0-or-later), DFTTest2 NVRTC "
                "(GPL-3.0), vszipcu (MIT), and NVIDIA CUDA NVRTC (proprietary) packages.",
                parent=self.dependency_dialog or self.root,
            )
        if not confirmed:
            return
        self.dependency_cancel.clear()
        self._set_busy("dependencies")
        if self.dependency_install_button:
            self.dependency_install_button.configure(state="disabled")
        if self.dependency_cancel_button:
            self.dependency_cancel_button.configure(state="normal")
        if self.dependency_progress:
            self.dependency_progress.configure(mode="indeterminate")
            self.dependency_progress.start(12)
        self.status_var.set("Resolving and staging app-local dependencies…")
        self._append_dependency_log("\nStarting a new staged app-local dependency installation.")

        def progress(stage: str, message: str, current: int | None, total: int | None) -> None:
            self.events.put(("dependency_progress", (stage, message, current, total)))

        def log(line: str) -> None:
            self.events.put(("dependency_log", line))

        def worker() -> None:
            try:
                result = install_latest_dependencies(
                    components=components,
                    cancel_event=self.dependency_cancel,
                    progress=progress,
                    log=log,
                )
                self.events.put(("dependency_done", result))
            except DependencyInstallCancelled as exc:
                self.events.put(("dependency_canceled", str(exc)))
            except Exception as exc:
                self.events.put(("dependency_error", str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def _check_latest_dependencies(self) -> None:
        if self.busy_kind:
            return
        self.status_var.set("Checking official current stable release metadata…")
        self._append_dependency_log(
            "Checking current stable FFmpeg, VapourSynth, Python, pip, VSJetpack, and Vulkan NNEDI3 metadata…"
        )

        def worker() -> None:
            try:
                self.events.put(("dependency_latest", resolve_latest_releases()))
            except Exception as exc:
                self.events.put(("dependency_latest_error", str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def _handle_latest_release_result(self, plan) -> None:
        message = (
            f"Current stable releases: FFmpeg {plan.ffmpeg_version}; VapourSynth {plan.vapoursynth_version}; "
            f"VSJetpack {plan.vsjetpack_version}; Vulkan NNEDI3 {plan.nnedi3vk_version or 'optional metadata unavailable'}. "
            f"Portable support runtime: Python {plan.python_version}, pip {plan.pip_version}."
        )
        self._append_dependency_log(message)
        self.status_var.set(message)

    def _show_dependency_progress(
        self, stage: str, message: str, current: int | None, total: int | None
    ) -> None:
        detail = message
        if current is not None and total:
            percent = max(0.0, min(100.0, current * 100.0 / total))
            detail += f" — {percent:.1f}%"
            if self.dependency_progress:
                self.dependency_progress.stop()
                self.dependency_progress.configure(mode="determinate", value=percent)
        elif self.dependency_progress:
            self.dependency_progress.configure(mode="indeterminate")
            self.dependency_progress.start(12)
        self.status_var.set(f"{stage}: {detail}")

    def _append_dependency_log(self, line: str) -> None:
        widget = self.dependency_doctor_text
        if not widget or not widget.winfo_exists():
            return
        widget.configure(state="normal")
        widget.insert("end", line.rstrip() + "\n")
        widget.see("end")
        widget.configure(state="disabled")

    def _finish_dependency_ui(self) -> None:
        if self.dependency_progress:
            self.dependency_progress.stop()
            self.dependency_progress.configure(mode="determinate", value=0)
        if self.dependency_install_button:
            self.dependency_install_button.configure(state="normal")
        if self.dependency_cancel_button:
            self.dependency_cancel_button.configure(state="disabled")

    def _handle_dependency_done(self, result) -> None:
        self._finish_dependency_ui()
        self._set_busy(None)
        if result.ffmpeg_path:
            self.persisted["ffmpeg_path"] = str(result.ffmpeg_path)
        if result.ffprobe_path:
            self.persisted["ffprobe_path"] = str(result.ffprobe_path)
        if result.vspipe_path:
            self.persisted["vspipe_path"] = str(result.vspipe_path)
        self._append_dependency_log(f"Validated and activated: {result.manifest_path}")
        self.status_var.set("App-local dependencies were validated and activated; rescanning capabilities…")
        if not self.close_pending:
            messagebox.showinfo(
                "Dependencies ready",
                f"Validated app-local dependencies were activated at:\n{result.runtime_root}\n\n"
                "System PATH, registry, Python, and existing installations were not changed.",
                parent=self.dependency_dialog or self.root,
            )
            self._refresh_capabilities()

    def _handle_dependency_failure(self, message: str, *, canceled: bool) -> None:
        self._finish_dependency_ui()
        self._set_busy(None)
        prefix = "Canceled" if canceled else "Failed"
        self._append_dependency_log(f"{prefix}: {message}")
        self.status_var.set(
            f"Dependency installation {prefix.lower()}; the previously active runtime and all system tools are unchanged."
        )
        if not canceled and not self.close_pending:
            messagebox.showerror(
                "Dependency installation failed",
                f"{message}\n\nNothing new was activated; the previous runtime and system tools are unchanged.",
                parent=self.dependency_dialog or self.root,
            )

    def _open_output_folder(self) -> None:
        target = self.batch_last_output if self._is_batch_tab() else self.last_completed_output
        if target and target.is_file():
            subprocess.Popen(["explorer.exe", f"/select,{target}"], creationflags=CREATE_NO_WINDOW)
            return
        if self._is_batch_tab():
            batch_folder = self.batch_output_dir_var.get().strip()
            if batch_folder and Path(batch_folder).is_dir():
                subprocess.Popen(["explorer.exe", batch_folder], creationflags=CREATE_NO_WINDOW)
                return
            selected = self._batch_selected_ids() if hasattr(self, "batch_tree") else ()
            if selected:
                record = self.batch_queue.record(selected[0])
                if record and record.source_path.parent.is_dir():
                    subprocess.Popen(["explorer.exe", str(record.source_path.parent)], creationflags=CREATE_NO_WINDOW)
                    return
        output_text = self.output_var.get().strip()
        folder = Path(output_text).parent if output_text else None
        if folder and folder.is_dir():
            subprocess.Popen(["explorer.exe", str(folder)], creationflags=CREATE_NO_WINDOW)

    def _about(self) -> None:
        messagebox.showinfo(
            "About Deinterlace Studio",
            f"Deinterlace Studio {__version__}\n\nQuality-first FFmpeg BWDIF and VapourSynth QTGMC processing with adaptive bounded scheduling, capability-gated output depths, the complete P7/UHQ/full-resolution-multipass NVENC quality contract, native MOV ProRes/DNxHR editor masters, a validated no-rerender MKV-to-MOV compatibility-copy workflow, native-chroma or 4:4:4 FFV1 16-bit archival masters, optional graph-tested Vulkan NNEDI3 interpolation, capability-tested post-deinterlace temporal denoisers with verified NVRTC-first DFTTest2 acceleration and CPU fallback, fast full-file source-health checks, exact indexed FFV1/VSPipe frame contracts with a live cancellable decoded fallback, backend-aware automatic QTGMC repair and continuation, temporal/spatial timestamp-aware BWDIF processing, exact DAR preservation, consolidated partial validation with same-file atomic promotion, and linked audit sidecars.",
            parent=self.root,
        )

    def _save_persisted(self) -> None:
        self.persisted["window_geometry"] = self.root.geometry()
        self.persisted["settings_schema_version"] = 2
        self.persisted["automatic_repair_and_continue"] = self.auto_repair_continue_var.get()
        self.persisted["denoise_enabled"] = self.denoise_enabled_var.get()
        self.persisted["ffv1_chroma_mode"] = FFV1_CHROMA_LABELS.get(
            self.ffv1_chroma_var.get(), "native"
        )
        self.persisted["vulkan_nnedi3"] = self.vulkan_nnedi3_var.get()
        self.persisted["batch_output_dir"] = self.batch_output_dir_var.get().strip()
        self.persisted["batch_include_subfolders"] = self.batch_include_subfolders_var.get()
        self.persisted["batch_continue_after_error"] = self.batch_continue_var.get()
        self.persisted["denoiser"] = DENOISER_LABELS.get(self.denoiser_var.get(), DEFAULT_DENOISER)
        try:
            self.persisted["denoise_strength"] = min(
                MAX_DENOISE_STRENGTH,
                max(MIN_DENOISE_STRENGTH, int(self.denoise_strength_var.get())),
            )
        except ValueError:
            self.persisted["denoise_strength"] = 4
        try:
            self.persisted["denoise_temporal_radius"] = min(
                MAX_TEMPORAL_RADIUS,
                max(MIN_TEMPORAL_RADIUS, int(self.denoise_radius_var.get())),
            )
        except ValueError:
            self.persisted["denoise_temporal_radius"] = 3
        try:
            save_settings(self.persisted, self.settings_path)
        except OSError:
            pass

    def _on_close(self) -> None:
        if self.close_pending:
            return
        if self.busy_kind in {"analysis", "processing", "repair", "compatibility", "dependencies", "batch"}:
            if not messagebox.askyesno("Cancel active work?", "Cancel the active operation and close the application?", parent=self.root):
                return
            self.close_pending = True
            self._cancel_active()
            self.status_var.set("Waiting for active child processes to stop safely before closing…")
            return
        if self.busy_kind == "capabilities":
            self.close_pending = True
            self.status_var.set("Waiting for the bounded dependency scan to finish before closing…")
            return
        self._finish_pending_close()

    def _finish_pending_close(self) -> None:
        self._cancel_event_poll()
        self._save_persisted()
        if self.file_drop_target:
            try:
                self.file_drop_target.close()
            except OSError:
                pass
        self.root.destroy()

    def self_test(self) -> dict[str, object]:
        """Bounded packaged-startup smoke test used by the build gate."""

        self.root.update_idletasks()
        issues = dependency_issues(self.capabilities)
        drop_route_passed = False
        drop_route_detail = "Native drop target is unavailable."
        batch_drop_route_passed = False
        batch_drop_route_detail = "Native batch drop target is unavailable."
        if self.file_drop_target and self.file_drop_target.active:
            original_input = self.input_var.get()
            original_output = self.output_var.get()
            original_status = self.status_var.get()
            original_persisted = dict(self.persisted)
            original_batch_records = list(self.batch_queue.records)
            original_tab = self.notebook.select() if self.notebook is not None else None
            try:
                with tempfile.TemporaryDirectory(prefix="deinterlace-packaged-drop-") as directory:
                    dropped = Path(directory) / "拖放 {測試} video.mkv"
                    dropped.write_bytes(b"bounded packaged drop self-test")
                    variable = "::deinterlace_packaged_drop_test"
                    self.root.tk.call("set", variable, (str(dropped),))
                    try:
                        raw_drop = self.root.tk.eval(f"set {variable}")
                    finally:
                        self.root.tk.call("unset", variable)
                    action = self.file_drop_target._on_drop(SimpleNamespace(data=raw_drop))
                    self.root.update()
                    drop_route_passed = (
                        action == "copy"
                        and Path(self.input_var.get()) == dropped
                        and "deinterlaced" in self.output_var.get()
                    )
                    drop_route_detail = (
                        "Unicode/spaced/braced Tcl drop data reached the same source-selection path as Browse."
                        if drop_route_passed
                        else "The synthetic TkDND event did not reach the authoritative source-selection path."
                    )
                    first_batch = Path(directory) / "批次 first {A}.mkv"
                    second_batch = Path(directory) / "批次 second B.ts"
                    first_batch.write_bytes(b"bounded packaged batch drop self-test A")
                    second_batch.write_bytes(b"bounded packaged batch drop self-test B")
                    self.batch_queue.clear()
                    if self.notebook is not None and self.setup_tab is not None:
                        self.notebook.select(self.setup_tab)
                    self._handle_dropped_paths((first_batch, second_batch))
                    self.root.update()
                    batch_drop_route_passed = (
                        len(self.batch_queue.records) == 2
                        and self.batch_queue.records[0].source_path == first_batch.resolve()
                        and self.batch_queue.records[1].source_path == second_batch.resolve()
                        and self._is_batch_tab()
                    )
                    batch_drop_route_detail = (
                        "A two-file Unicode drop entered the ordered Batch queue and selected the Batch tab."
                        if batch_drop_route_passed
                        else "The two-file drop did not preserve its ordered Batch queue route."
                    )
            except Exception as exc:
                drop_route_detail = f"Bounded TkDND route test failed: {type(exc).__name__}: {exc}"
                batch_drop_route_detail = drop_route_detail
            finally:
                self.batch_queue.records[:] = original_batch_records
                self._refresh_batch_tree()
                self.input_var.set(original_input)
                self.output_var.set(original_output)
                self.persisted.clear()
                self.persisted.update(original_persisted)
                self.status_var.set(original_status)
                if self.notebook is not None and original_tab:
                    self.notebook.select(original_tab)
                self._update_start_button_state()
        normalized_timeline_guide = " ".join(SOURCE_TIMELINE_GUIDE_TEXT.split())
        normalized_backend_gpu_guide = " ".join(BACKEND_GPU_GUIDE_TEXT.split())
        normalized_denoise_guide = " ".join(DENOISE_GUIDE_TEXT.split())
        return {
            "application_version": __version__,
            "title": self.root.title(),
            "engine_choices": len(self.engine_combo.cget("values")),
            "family_choices": len(self.family_combo.cget("values")),
            "ffmpeg_found": bool(self.capabilities and self.capabilities.ffmpeg_path),
            "ffprobe_found": bool(self.capabilities and self.capabilities.ffprobe_path),
            "ffmpeg_version": self.capabilities.ffmpeg_version if self.capabilities else None,
            "ffprobe_version": self.capabilities.ffprobe_version if self.capabilities else None,
            "ffmpeg_version_kind": self.capabilities.ffmpeg_version_kind if self.capabilities else None,
            "ffmpeg_version_diagnostic": (
                self.capabilities.ffmpeg_version_diagnostic if self.capabilities else None
            ),
            "ffmpeg_git_revision": self.capabilities.ffmpeg_git_revision if self.capabilities else None,
            "ffprobe_git_revision": self.capabilities.ffprobe_git_revision if self.capabilities else None,
            "ffmpeg_library_versions": (
                self.capabilities.ffmpeg_library_versions if self.capabilities else {}
            ),
            "ffprobe_library_versions": (
                self.capabilities.ffprobe_library_versions if self.capabilities else {}
            ),
            "ffmpeg_selection_source": self.capabilities.ffmpeg_selection_source if self.capabilities else None,
            "ffmpeg_discovery_diagnostics": (
                self.capabilities.ffmpeg_discovery_diagnostics if self.capabilities else ()
            ),
            "vspipe_found": bool(self.capabilities and self.capabilities.vspipe_path),
            "vapoursynth_version": self.capabilities.vapoursynth_version if self.capabilities else None,
            "qtgmc_ready": bool(self.capabilities and self.capabilities.qtgmc_ready),
            "vulkan_nnedi3_ready": bool(self.capabilities and self.capabilities.vulkan_nnedi3_ready),
            "vulkan_nnedi3_diagnostic": (
                self.capabilities.vulkan_nnedi3_diagnostic if self.capabilities else None
            ),
            "vulkan_nnedi3_default_enabled": self.vulkan_nnedi3_var.get(),
            "ffv1_chroma_default": FFV1_CHROMA_LABELS.get(self.ffv1_chroma_var.get()),
            "ffv1_chroma_choices": len(self.ffv1_chroma_combo.cget("values")),
            "denoise_capabilities": (
                self.capabilities.denoise_capabilities if self.capabilities else {}
            ),
            "denoise_backends": self.capabilities.denoise_backends if self.capabilities else {},
            "denoiser_choices": len(self.denoiser_combo.cget("values")),
            "denoise_default_enabled": self.denoise_enabled_var.get(),
            "denoise_guide_has_evidence_based_choices": (
                "VapourSynth V-BM3D" in DENOISE_GUIDE_TEXT
                and "VapourSynth DFTTest2" in DENOISE_GUIDE_TEXT
                and "VapourSynth MVTools degrain" in DENOISE_GUIDE_TEXT
                and "VapourSynth temporal NLMeans" in DENOISE_GUIDE_TEXT
                and "FFmpeg fftdnoiz" in DENOISE_GUIDE_TEXT
                and "FFmpeg atadenoise" in DENOISE_GUIDE_TEXT
                and "NO UNIVERSAL RANKING" in DENOISE_GUIDE_TEXT
                and "not presented in this Temporal denoiser list" in DENOISE_GUIDE_TEXT
            ),
            "denoise_guide_requires_deinterlace_first": (
                "always deinterlaces first" in DENOISE_GUIDE_TEXT
                and "denoises the resulting progressive frames second" in DENOISE_GUIDE_TEXT
            ),
            "denoise_guide_explains_temporal_radius": (
                "WHAT TEMPORAL RADIUS MEANS" in DENOISE_GUIDE_TEXT
                and "(2 × N) + 1 frames" in DENOISE_GUIDE_TEXT
                and "does not change frame rate, playing speed, or video duration"
                in normalized_denoise_guide
                and "Radius 1–2 is the conservative starting" in normalized_denoise_guide
            ),
            "nvenc_verified_bit_depths": (
                self.capabilities.encoder_verified_bit_depths if self.capabilities else {}
            ),
            "interlace_runtime_verified": (
                self.capabilities.interlace_runtime_verified if self.capabilities else {}
            ),
            "interlace_runtime_diagnostics": (
                self.capabilities.interlace_runtime_diagnostics if self.capabilities else {}
            ),
            "start_initially_disabled": str(self.start_button.cget("state")) == "disabled",
            "default_output_cadence": self._cadence_value(),
            "default_hardware_decode": HW_DECODE_LABELS.get(self.hardware_decode_var.get()),
            "default_denoiser": DENOISER_LABELS.get(self.denoiser_var.get()),
            "default_denoise_strength": self.denoise_strength_var.get(),
            "default_denoise_radius": self.denoise_radius_var.get(),
            "drag_drop_enabled": bool(self.file_drop_target and self.file_drop_target.active),
            "drag_drop_diagnostic": self.file_drop_diagnostic,
            "drag_drop_provider": "TkDND" if self.file_drop_target else None,
            "drag_drop_provider_version": (
                self.file_drop_target.provider_version if self.file_drop_target else None
            ),
            "drag_drop_package_version": (
                self.file_drop_target.package_version if self.file_drop_target else None
            ),
            "drag_drop_surface_count": (
                len(self.file_drop_target.registrations) if self.file_drop_target else 0
            ),
            "drag_drop_registration_error_count": (
                len(self.file_drop_target.registration_errors) if self.file_drop_target else 0
            ),
            "drag_drop_route_test_passed": drop_route_passed,
            "drag_drop_route_test_detail": drop_route_detail,
            "batch_tab_available": self.batch_tab is not None,
            "batch_max_files": self.batch_queue.maximum,
            "batch_queue_columns": tuple(self.batch_tree.cget("columns")),
            "batch_delete_binding_available": bool(self.batch_tree.bind("<Delete>")),
            "batch_drag_bindings_available": all(
                bool(self.batch_tree.bind(sequence))
                for sequence in ("<ButtonPress-1>", "<B1-Motion>", "<ButtonRelease-1>")
            ),
            "batch_drop_route_test_passed": batch_drop_route_passed,
            "batch_drop_route_test_detail": batch_drop_route_detail,
            "batch_controls_share_settings": all(
                str(batch_control.cget("textvariable")) == str(single_control.cget("textvariable"))
                for batch_control, single_control in (
                    (self.batch_engine_combo, self.engine_combo),
                    (self.batch_field_combo, self.field_combo),
                    (self.batch_cadence_combo, self.cadence_combo),
                    (self.batch_hw_decode_combo, self.hw_decode_combo),
                    (self.batch_aspect_combo, self.aspect_combo),
                    (self.batch_denoiser_combo, self.denoiser_combo),
                    (self.batch_family_combo, self.family_combo),
                    (self.batch_depth_combo, self.depth_combo),
                    (self.batch_ffv1_chroma_combo, self.ffv1_chroma_combo),
                )
            ),
            "backend_guide_has_rankings": "#1  VapourSynth QTGMC" in BACKEND_GPU_GUIDE_TEXT,
            "unified_backend_gpu_guide_available": (
                hasattr(self, "backend_gpu_button")
                and "Backend / QTGMC / GPU guide" in str(self.backend_gpu_button.cget("text"))
            ),
            "qtgmc_guide_explains_exact_parameters": all(
                token in BACKEND_GPU_GUIDE_TEXT
                for token in (
                    "analyze_force_tr=3",
                    "analyze_blksize=16",
                    "analyze_overlap=2",
                    "analyze_refine=2",
                    "prefilter_tr=2",
                    "basic_tr=2",
                    "final_tr=3",
                    "source_match(tr=2, TWICE_REFINED)",
                    "lossless(POSTSMOOTH, anti_comb=True)",
                )
            ),
            "qtgmc_acceleration_preserves_quality_parameters": (
                "identical QTGMC parameters" in normalized_backend_gpu_guide
                and "does not select a faster QTGMC preset" in normalized_backend_gpu_guide
            ),
            "help_excludes_job_specific_chat_analysis": all(
                phrase not in BACKEND_GPU_GUIDE_TEXT
                for phrase in (
                    "WHAT TOOK SO LONG IN THE SUPPLIED COMPLETED JOB",
                    "2 hours 10 minutes",
                    "132.6-GiB",
                    "8.50 times faster",
                )
            ),
            "resolve_editor_preset_available": hasattr(self, "resolve_preset_button"),
            "speed_gpu_modes_available": len(SPEED_MODES) == 3,
            "speed_gpu_guide_has_measured_tradeoffs": (
                "BWDIF CUDA + HEVC NVENC" in BACKEND_GPU_GUIDE_TEXT
                and "changes the deinterlacing algorithm" in BACKEND_GPU_GUIDE_TEXT
                and "does not change or accelerate QTGMC itself" in BACKEND_GPU_GUIDE_TEXT
            ),
            "bwdif_temporal_analysis_retained": (
                "BWDIF is temporal as well as spatial" in normalized_backend_gpu_guide
                and "previous, current, and next" in normalized_backend_gpu_guide
                and "not “turn temporal analysis off.”" in normalized_backend_gpu_guide
            ),
            "nvenc_maximum_quality_contract_documented": all(
                token in normalized_backend_gpu_guide
                for token in ("preset=p7", "tune=uhq", "multipass=fullres")
            ),
            "mov_compatibility_copy_available": (
                hasattr(self, "compatibility_copy_button")
                and callable(self._start_mov_compatibility_copy)
                and "Fast MOV copy" in str(self.compatibility_copy_button.cget("text"))
            ),
            "dnxhr_selectable_depths": selectable_bit_depths(
                "dnxhr", self.capabilities, hardware_encode=False
            ),
            "cadence_guide_preserves_duration": "60-minute input remains 60 minutes" in CADENCE_GUIDE_TEXT,
            "timeline_guide_explains_repeat_failure": (
                "blocked every time QTGMC is selected" in SOURCE_TIMELINE_GUIDE_TEXT
            ),
            "timeline_guide_explains_in_app_repair": (
                "WHAT AUTOMATIC QTGMC RECOVERY AND THE “REPAIR SOURCE…” BUTTON DO"
                in SOURCE_TIMELINE_GUIDE_TEXT
            ),
            "timeline_guide_explains_source_protection": (
                "never rewrites the selected source" in SOURCE_TIMELINE_GUIDE_TEXT
            ),
            "timeline_guide_explains_fast_precheck": (
                "fast full-file scan" in SOURCE_TIMELINE_GUIDE_TEXT
                and "AUTOMATIC QTGMC RECOVERY — DEFAULT ENABLED" in SOURCE_TIMELINE_GUIDE_TEXT
                and "healthy analysis still waits for the Start button" in SOURCE_TIMELINE_GUIDE_TEXT
            ),
            "timeline_guide_explains_indexed_contract": (
                "fast VSPipe graph-info check" in normalized_timeline_guide
                and "one-to-one packet" in normalized_timeline_guide
                and "managed full decoded fallback" in normalized_timeline_guide
                and "immediate Cancel support" in normalized_timeline_guide
                and "No output encode begins" in normalized_timeline_guide
            ),
            "phase_aware_progress_available": (
                hasattr(self, "run_phase_detail") and callable(self._handle_run_progress)
            ),
            "timeline_guide_explains_bwdif_repair_bypass": (
                "Explicit BWDIF CPU/CUDA bypasses automatic repair" in SOURCE_TIMELINE_GUIDE_TEXT
                and "Manual Repair required… remains available" in SOURCE_TIMELINE_GUIDE_TEXT
            ),
            "source_health_banner_available": hasattr(self, "source_health_label"),
            "automatic_recovery_control_available": hasattr(self, "auto_repair_check"),
            "automatic_recovery_enabled": self.auto_repair_continue_var.get(),
            "repair_workflow_available": callable(self._show_repair_dialog),
            "repair_default_mode": self.repair_mode_var.get(),
            "setup_layout_mode": self.setup_layout_mode,
            "setup_wide_breakpoint": self.setup_wide_breakpoint,
            "setup_available_width": self.setup_parent.winfo_width(),
            "analysis_panel_height": self.summary_text.winfo_height(),
            "analysis_panel_requested_height": self.summary_text.winfo_reqheight(),
            "managed_runtime_root": str(managed_runtime_root()),
            "dependency_installer_available": callable(install_latest_dependencies),
            "dependency_ready": not issues,
            "dependency_issues": issues,
        }


# Avoid importing processor's private platform constant into the UI module.
CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0
