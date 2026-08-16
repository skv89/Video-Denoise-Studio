from __future__ import annotations

from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable


SOURCE_REPAIR_REQUIRED_FAILURE = "source_repair_required"


@dataclass(frozen=True)
class StreamInfo:
    index: int
    codec_type: str
    codec_name: str = "unknown"
    profile: str | None = None
    width: int | None = None
    height: int | None = None
    pix_fmt: str | None = None
    bits_per_raw_sample: int | None = None
    sample_aspect_ratio: Fraction | None = None
    display_aspect_ratio: Fraction | None = None
    r_frame_rate: Fraction | None = None
    avg_frame_rate: Fraction | None = None
    time_base: Fraction | None = None
    field_order: str | None = None
    start_time: float | None = None
    duration: float | None = None
    nb_frames: int | None = None
    color_range: str | None = None
    color_space: str | None = None
    color_transfer: str | None = None
    color_primaries: str | None = None
    tags: dict[str, str] = field(default_factory=dict)
    disposition: dict[str, int] = field(default_factory=dict)

    @property
    def dimensions(self) -> tuple[int, int] | None:
        if self.width and self.height:
            return self.width, self.height
        return None


@dataclass(frozen=True)
class MediaProbe:
    path: Path
    format_name: str
    format_long_name: str | None
    duration: float | None
    size: int | None
    bit_rate: int | None
    start_time: float | None
    streams: tuple[StreamInfo, ...]
    chapters: tuple[dict[str, Any], ...] = ()
    format_tags: dict[str, str] = field(default_factory=dict)
    sampled_interlaced_frames: int = 0
    sampled_progressive_frames: int = 0
    sampled_tff_frames: int = 0
    sampled_bff_frames: int = 0

    @property
    def video(self) -> StreamInfo:
        for stream in self.streams:
            if stream.codec_type == "video":
                return stream
        raise ValueError("No video stream is present")

    def streams_of_type(self, codec_type: str) -> tuple[StreamInfo, ...]:
        return tuple(stream for stream in self.streams if stream.codec_type == codec_type)

    @property
    def audio_count(self) -> int:
        return len(self.streams_of_type("audio"))

    @property
    def subtitle_count(self) -> int:
        return len(self.streams_of_type("subtitle"))

    @property
    def attachment_count(self) -> int:
        return len(self.streams_of_type("attachment"))

    @property
    def data_count(self) -> int:
        return len(self.streams_of_type("data"))


@dataclass
class IDetCounts:
    repeated_neither: int = 0
    repeated_top: int = 0
    repeated_bottom: int = 0
    single_tff: int = 0
    single_bff: int = 0
    single_progressive: int = 0
    single_undetermined: int = 0
    multi_tff: int = 0
    multi_bff: int = 0
    multi_progressive: int = 0
    multi_undetermined: int = 0

    def __iadd__(self, other: "IDetCounts") -> "IDetCounts":
        for name in self.__dataclass_fields__:
            setattr(self, name, getattr(self, name) + getattr(other, name))
        return self

    @property
    def multi_total(self) -> int:
        return self.multi_tff + self.multi_bff + self.multi_progressive + self.multi_undetermined

    @property
    def determined_total(self) -> int:
        return self.multi_tff + self.multi_bff + self.multi_progressive


@dataclass(frozen=True)
class IDetSegment:
    offset: float
    duration: float | None
    counts: IDetCounts


@dataclass(frozen=True)
class IDetReport:
    mode: str
    segments: tuple[IDetSegment, ...]
    aggregate: IDetCounts
    classification: str
    dominant_field_order: str | None
    confidence: float
    rationale: str
    command_lines: tuple[str, ...] = ()


@dataclass(frozen=True)
class CapabilityReport:
    ffmpeg_path: Path | None
    ffprobe_path: Path | None
    ffmpeg_version: str | None
    ffmpeg_configuration: str | None
    filters: frozenset[str]
    encoders: frozenset[str]
    encoder_pixel_formats: dict[str, tuple[str, ...]]
    hwaccels: frozenset[str]
    vspipe_path: Path | None
    vapoursynth_version: str | None
    qtgmc_ready: bool
    qtgmc_diagnostic: str
    qtgmc_install_command: str | None
    gpu_name: str | None = None
    gpu_memory_mib: int | None = None
    gpu_driver: str | None = None
    encoder_verified_bit_depths: dict[str, tuple[int, ...]] = field(default_factory=dict)
    encoder_runtime_diagnostics: dict[str, str] = field(default_factory=dict)
    interlace_runtime_verified: dict[str, bool] = field(default_factory=dict)
    interlace_runtime_diagnostics: dict[str, str] = field(default_factory=dict)
    ffmpeg_selection_source: str | None = None
    ffmpeg_discovery_diagnostics: tuple[str, ...] = ()
    ffprobe_version: str | None = None
    ffmpeg_version_kind: str | None = None
    ffmpeg_version_diagnostic: str = ""
    ffmpeg_git_revision: str | None = None
    ffprobe_git_revision: str | None = None
    ffmpeg_library_versions: dict[str, tuple[int, int, int]] = field(default_factory=dict)
    ffprobe_library_versions: dict[str, tuple[int, int, int]] = field(default_factory=dict)
    denoise_capabilities: dict[str, bool] = field(default_factory=dict)
    denoise_backends: dict[str, str] = field(default_factory=dict)
    denoise_diagnostics: dict[str, str] = field(default_factory=dict)
    vulkan_nnedi3_ready: bool = False
    vulkan_nnedi3_diagnostic: str = "Vulkan NNEDI3 was not capability-scanned."
    vulkan_nnedi3_package_version: str | None = None


@dataclass(frozen=True)
class JobSettings:
    input_path: Path
    output_path: Path
    backend: str = "auto"
    field_order: str = "auto"
    output_cadence: str = "frame_rate"
    allow_progressive_override: bool = False
    aspect_mode: str = "preserve"
    manual_dar: str = "16:9"
    family: str = "ffv1"
    bit_depth: int = 16
    ffv1_chroma_mode: str = "native"
    hardware_encode: bool = False
    hardware_decode: str = "auto"
    vulkan_nnedi3: bool = False
    av1_software_encoder: str = "libaom"
    quality: int = 14
    tune_grain: bool = True
    denoise_enabled: bool = True
    denoiser: str = "vs_bm3d"
    denoise_strength: int = 4
    denoise_temporal_radius: int = 3
    copy_audio: bool = True
    copy_subtitles: bool = True
    copy_attachments: bool = True
    copy_data: bool = False
    copy_chapters: bool = True
    copy_metadata: bool = True
    overwrite_approved: bool = False


@dataclass(frozen=True)
class OutputExpectation:
    codec_names: tuple[str, ...]
    pix_fmts: tuple[str, ...]
    width: int
    height: int
    sar: Fraction
    dar: Fraction
    frame_rate: Fraction | None
    progressive: bool
    lossless: bool
    bit_depth: int
    expected_audio: tuple[StreamInfo, ...]
    expected_subtitles: tuple[StreamInfo, ...]
    expected_attachments: tuple[StreamInfo, ...]
    duration: float | None
    frame_count: int | None = None
    expected_data: tuple[StreamInfo, ...] = ()
    expected_chapter_count: int = 0
    expected_format_tags: dict[str, str] = field(default_factory=dict)
    color_range: str | None = None
    color_space: str | None = None
    color_transfer: str | None = None
    color_primaries: str | None = None


@dataclass(frozen=True)
class PacketTimelineGap:
    before_pts: float
    after_pts: float
    duration: float


@dataclass(frozen=True)
class SourceHealthReport:
    path: Path
    source_size: int
    source_mtime_ns: int
    status: str
    reason: str
    elapsed_seconds: float
    packet_count: int
    timestamped_packet_count: int
    unique_timestamp_count: int
    first_pts: float | None
    last_pts: float | None
    typical_step_seconds: float | None
    packet_timeline_span_seconds: float | None
    reported_duration_seconds: float | None
    duration_difference_seconds: float | None
    gap_threshold_seconds: float
    material_gap_count: int
    largest_gaps: tuple[PacketTimelineGap, ...]
    demux_warning_count: int
    structural_warning_count: int
    warning_samples: tuple[str, ...]
    ffprobe_returncode: int | None = None
    scan_error: str | None = None

    @property
    def repair_required(self) -> bool:
        return self.status == "repair_required"


@dataclass(frozen=True)
class SourcePreflightEvidence:
    """Auditable frame-contract evidence gathered before output encoding."""

    method: str
    source_frames: int | None
    expected_output_frames: int | None
    elapsed_seconds: float
    packet_count: int | None = None
    graph_output_frames: int | None = None
    fast_path_eligible: bool = False
    fallback_reason: str | None = None


@dataclass(frozen=True)
class AutomaticRecoveryAudit:
    original_source: Path
    trigger_health: SourceHealthReport
    requested_output: Path
    selected_output: Path
    repair_output: Path | None
    repair_method: str
    repair_output_sha256: str | None
    repair_log_path: Path | None
    repair_report_path: Path | None
    repeated_frames: int
    dropped_frames: int
    storage_preflight: str


@dataclass(frozen=True)
class ProcessingPlan:
    valid: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    settings: JobSettings
    profile_id: str | None
    profile_label: str | None
    selected_backend: str | None
    selected_field_order: str | None
    selected_denoiser: str | None
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
    analysis_summary: str
    source_health: SourceHealthReport | None = None
    automatic_recovery: AutomaticRecoveryAudit | None = None
    vapoursynth_threads: int | None = None
    vspipe_requests: int | None = None
    vapoursynth_schedule_note: str | None = None
    vulkan_nnedi3_active: bool = False


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    output_probe: MediaProbe | None
    checked_frame_count: int = 0
    checked_progressive_frames: int = 0
    checked_interlaced_frames: int = 0
    verified_packet_count: int | None = None
    verified_key_packet_count: int | None = None
    thorough_packet_scan_completed: bool = False


@dataclass(frozen=True)
class ProcessingResult:
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
    failure_code: str | None = None


ProgressCallback = Callable[[dict[str, str]], None]
LogCallback = Callable[[str], None]


def json_safe(value: Any) -> Any:
    if isinstance(value, Fraction):
        return f"{value.numerator}/{value.denominator}"
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dataclass_fields__"):
        return json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [json_safe(item) for item in value]
    return value
