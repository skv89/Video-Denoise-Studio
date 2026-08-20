from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
import uuid

from video_processing_core.media.models import (
    CapabilityReport,
    MediaProbe,
    OutputExpectation,
    ValidationResult,
)
from video_processing_core.media.presets import OutputProfile
from video_processing_core.media.scheduling import VapourSynthSchedule


@dataclass(frozen=True)
class DenoiseSettings:
    input_path: Path
    output_path: Path
    denoiser: str = "vs_bm3d"
    denoise_strength: int = 4
    denoise_temporal_radius: int = 3
    family: str = "ffv1"
    container: str = "auto"
    bit_depth: int = 16
    ffv1_chroma_mode: str = "native"
    hardware_encode: bool = False
    av1_software_encoder: str = "libaom"
    quality: int = 14
    tune_grain: bool = True
    copy_audio: bool = True
    copy_subtitles: bool = True
    copy_attachments: bool = True
    copy_data: bool = False
    copy_chapters: bool = True
    copy_metadata: bool = True


@dataclass(frozen=True)
class DenoisePlan:
    valid: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    settings: DenoiseSettings
    media: MediaProbe | None
    ffprobe_path: Path | None
    profile: OutputProfile | None
    container: str | None
    selected_denoise_backend: str | None
    output_path: Path
    partial_path: Path | None
    log_path: Path | None
    report_path: Path | None
    script_path: Path | None
    temporary_script_path: Path | None
    ffmpeg_command: tuple[str, ...]
    vspipe_command: tuple[str, ...] | None
    vapoursynth_script: str | None
    display_command: str
    expected: OutputExpectation | None
    schedule: VapourSynthSchedule | None = None


@dataclass(frozen=True)
class DenoiseResult:
    success: bool
    canceled: bool
    message: str
    output_path: Path | None
    log_path: Path | None
    report_path: Path | None
    script_path: Path | None
    output_sha256: str | None
    validation: ValidationResult | None
    quarantine_path: Path | None = None


@dataclass(frozen=True)
class PreviewRequest:
    source: Path
    media: MediaProbe
    capabilities: CapabilityReport
    denoiser: str
    strength: int
    temporal_radius: int
    target_frame: int
    width: int = 960
    height: int = 540
    include_processed: bool = True


@dataclass(frozen=True)
class PreviewFrames:
    token: str
    directory: Path
    original_frame: Path
    processed_frame: Path | None
    target_frame: int
    total_frames: int | None
    temporal_radius: int
    leading_context: int
    trailing_context: int
    fps: float
    selected_backend: str | None
    status: str
    strength: int = 0
    window_frames: int = 1


@dataclass
class BatchRecord:
    source_path: Path
    identifier: str = field(default_factory=lambda: uuid.uuid4().hex)
    state: str = "Queued"
    progress_text: str = "Waiting"
    percent: float | None = 0.0
    output_path: Path | None = None
    effective_text: str = "Pending preflight"
    fallback_notes: tuple[str, ...] = ()
    error: str | None = None
    media: MediaProbe | None = None
    plan: DenoisePlan | None = None
    result: DenoiseResult | None = None

    def reset(self) -> None:
        self.state = "Queued"
        self.progress_text = "Waiting"
        self.percent = 0.0
        self.output_path = None
        self.effective_text = "Pending preflight"
        self.fallback_notes = ()
        self.error = None
        self.plan = None
        self.result = None


@dataclass(frozen=True)
class BatchRunOptions:
    output_directory: Path | None
    continue_after_error: bool = True


@dataclass(frozen=True)
class BatchRunSummary:
    total: int
    completed: int
    failed: int
    canceled: int
    skipped: int


ProgressCallback = Callable[[dict[str, str]], None]
LogCallback = Callable[[str], None]
BatchEventCallback = Callable[[str, BatchRecord | None, object | None], None]
