from __future__ import annotations

from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any


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
