from __future__ import annotations

import json
import hashlib
import os
import re
import subprocess
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable

from . import __version__
from .models import CapabilityReport, MediaProbe, StreamInfo, json_safe
from .probe import probe_media
from .processor import qtgmc_timeline_integrity_error, sha256_file


CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0
CREATE_NEW_PROCESS_GROUP = 0x00000200 if os.name == "nt" else 0
PROCESS_FLAGS = CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP
CLOCK_DURATION = re.compile(r"^(\d+):(\d{2}):(\d{2}(?:\.\d+)?)$")
SEVERE_DIAGNOSTIC_PATTERNS = (
    "corrupt",
    "error while decoding",
    "illegal short term",
    "exceeds max",
    "mmco:",
    "no frame returned",
    "invalid nal",
    "missing reference",
    "concealing",
    "bytestream",
    "exceeds containing master element",
)
FATAL_ZERO_EXIT_PATTERNS = (
    "conversion failed",
    "error writing trailer",
    "error closing file",
    "file duration too long for timebase",
)
TECHNICAL_TAG_PREFIXES = ("_STATISTICS_",)
TECHNICAL_TAGS = {
    "BPS",
    "DURATION",
    "ENCODER",
    "ENCODER_OPTIONS",
    "NUMBER_OF_BYTES",
    "NUMBER_OF_FRAMES",
}
SAFE_FFV1_PIXEL_FORMAT_EQUIVALENTS = {
    "yuvj420p": "yuv420p",
    "yuvj422p": "yuv422p",
    "yuvj444p": "yuv444p",
    "nv12": "yuv420p",
    "nv16": "yuv422p",
    "nv24": "yuv444p",
    "p010le": "yuv420p10le",
    "p012le": "yuv420p12le",
    "p016le": "yuv420p16le",
    "rgb24": "bgr0",
    "bgr24": "bgr0",
}


class RepairCancelled(RuntimeError):
    pass


class RepairError(RuntimeError):
    pass


@dataclass(frozen=True)
class TimelineGap:
    after_frame: int
    previous_pts: float
    current_pts: float
    duration_seconds: float
    estimated_missing_frames: int


@dataclass(frozen=True)
class PacketTimelineFingerprint:
    packet_count: int
    first_pts_us: int | None
    last_pts_us: int | None
    digest: str


@dataclass(frozen=True)
class TimelineDiagnosis:
    source_path: Path
    nominal_rate: Fraction | None
    reported_video_duration: float | None
    container_duration: float | None
    decoded_frames: int
    timestamped_frames: int
    decoded_cfr_span_seconds: float | None
    first_pts: float | None
    last_pts: float | None
    pts_duration_seconds: float | None
    material_gaps: tuple[TimelineGap, ...]
    non_monotonic_steps: int
    sampled_interlaced_frames: int
    sampled_progressive_frames: int
    decoder_warning_count: int
    decoder_warning_samples: tuple[str, ...]
    severe_warning_count: int
    qtgmc_error: str | None
    classification: str
    recommended_method: str

    @property
    def qtgmc_compatible(self) -> bool:
        return self.qtgmc_error is None

    @property
    def largest_gap(self) -> TimelineGap | None:
        return max(self.material_gaps, key=lambda gap: gap.duration_seconds, default=None)


@dataclass(frozen=True)
class RepairRequest:
    source_path: Path
    output_path: Path
    mode: str = "automatic"
    overwrite_approved: bool = False


@dataclass(frozen=True)
class RepairValidation:
    valid: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    media: MediaProbe | None
    diagnosis: TimelineDiagnosis | None


@dataclass(frozen=True)
class RepairResult:
    success: bool
    canceled: bool
    message: str
    output_path: Path | None
    log_path: Path | None
    report_path: Path | None
    output_sha256: str | None
    method: str | None
    source_diagnosis: TimelineDiagnosis | None
    output_diagnosis: TimelineDiagnosis | None
    repeated_frames: int
    dropped_frames: int
    quarantine_path: Path | None = None


RepairProgressCallback = Callable[[dict[str, str]], None]
RepairLogCallback = Callable[[str], None]


def _reported_video_duration(media: MediaProbe) -> float | None:
    if media.video.duration is not None:
        return media.video.duration
    for key in ("DURATION", "DURATION-eng"):
        value = media.video.tags.get(key)
        match = CLOCK_DURATION.fullmatch(value.strip()) if value else None
        if match:
            hours, minutes, seconds = match.groups()
            return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    return media.duration


def _nominal_rate(media: MediaProbe) -> Fraction | None:
    rate = media.video.avg_frame_rate or media.video.r_frame_rate
    return rate if rate and rate > 0 else None


def classify_timeline_condition(
    *,
    qtgmc_error: str | None,
    material_gap_count: int,
    non_monotonic_steps: int,
    severe_warning_count: int,
    timestamps_complete: bool,
) -> tuple[str, str]:
    """Return an evidence-based condition and the safest automatic action."""

    if not timestamps_complete:
        return "undetermined_missing_timestamps", "blocked"
    if qtgmc_error is None and material_gap_count == 0 and non_monotonic_steps == 0:
        if severe_warning_count:
            return "decodable_corruption", "ffv1_rescue"
        return "healthy", "none"
    if material_gap_count or non_monotonic_steps:
        if severe_warning_count:
            return "timestamp_gap_with_decode_errors", "ffv1_rescue"
        return "timestamp_discontinuity", "ffv1_rescue"
    if severe_warning_count:
        return "decodable_corruption", "ffv1_rescue"
    if qtgmc_error is not None:
        return "container_or_duration_metadata", "stream_copy_remux"
    return "healthy", "none"


def choose_ffv1_pixel_format(source_pix_fmt: str | None, supported: tuple[str, ...]) -> str:
    """Choose an FFV1 format without reducing decoded precision or chroma."""

    if not source_pix_fmt:
        raise RepairError("The source pixel format is unknown; a lossless FFV1 rescue cannot be proven safe.")
    supported_set = set(supported)
    if source_pix_fmt in supported_set:
        return source_pix_fmt
    equivalent = SAFE_FFV1_PIXEL_FORMAT_EQUIVALENTS.get(source_pix_fmt)
    if equivalent and equivalent in supported_set:
        return equivalent
    raise RepairError(
        f"FFV1 cannot encode source pixel format {source_pix_fmt!r} directly and no lossless equivalent "
        "was verified. Use a clean replacement source rather than silently reducing chroma or bit depth."
    )


def _same_path(left: Path, right: Path) -> bool:
    try:
        return os.path.normcase(os.path.abspath(left)) == os.path.normcase(os.path.abspath(right))
    except OSError:
        return False


def _snapshot(path: Path) -> tuple[bool, int | None, int | None, int | None]:
    try:
        stat = path.stat()
        return True, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns
    except FileNotFoundError:
        return False, None, None, None


def _promotion_identity(path: Path) -> tuple[int, int, int, int]:
    """Identify the exact file object across a same-directory atomic rename."""

    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_dev, stat.st_ino


def _repair_structure_signature(media: MediaProbe) -> tuple[Any, ...]:
    """Return stable reopen facts while deliberately excluding the file path."""

    return (
        media.format_name,
        media.duration,
        media.size,
        media.start_time,
        tuple(
            (
                stream.index,
                stream.codec_type,
                stream.codec_name,
                stream.profile,
                stream.width,
                stream.height,
                stream.pix_fmt,
                stream.bits_per_raw_sample,
                stream.sample_aspect_ratio,
                stream.display_aspect_ratio,
                stream.r_frame_rate,
                stream.avg_frame_rate,
                stream.time_base,
                stream.field_order,
                stream.start_time,
                stream.duration,
                _meaningful_tags(stream.tags),
                stream.disposition,
            )
            for stream in media.streams
        ),
        media.chapters,
        _meaningful_tags(media.format_tags),
    )


def _safe_unlink(path: Path | None) -> None:
    if path is None:
        return
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _format_rate(rate: Fraction) -> str:
    return f"{rate.numerator}/{rate.denominator}"


def _color_args(media: MediaProbe) -> list[str]:
    video = media.video
    args: list[str] = []
    if video.color_range:
        args += ["-color_range", video.color_range]
    if video.color_space:
        args += ["-colorspace", video.color_space]
    if video.color_transfer:
        args += ["-color_trc", video.color_transfer]
    if video.color_primaries:
        args += ["-color_primaries", video.color_primaries]
    return args


def _field_mode(field_order: str | None) -> str | None:
    normalized = (field_order or "").lower()
    if normalized in {"tt", "tb"}:
        return "tff"
    if normalized in {"bb", "bt"}:
        return "bff"
    if normalized == "progressive":
        return "prog"
    return None


def build_remux_command(ffmpeg: Path, source: Path, partial: Path) -> tuple[str, ...]:
    return tuple(
        [
            str(ffmpeg),
            "-hide_banner",
            "-nostdin",
            "-y",
            "-fflags",
            "+genpts",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-map",
            "0:s?",
            "-map",
            "0:t?",
            "-map",
            "0:d?",
            "-c",
            "copy",
            "-map_metadata",
            "0",
            "-map_chapters",
            "0",
            "-metadata:s:v:0",
            "DURATION=",
            "-metadata:s:v:0",
            "DURATION-eng=",
            "-max_muxing_queue_size",
            "4096",
            "-progress",
            "pipe:1",
            "-nostats",
            str(partial),
        ]
    )


def build_rescue_command(
    ffmpeg: Path,
    source: Path,
    partial: Path,
    media: MediaProbe,
    diagnosis: TimelineDiagnosis,
    supported_ffv1_formats: tuple[str, ...],
) -> tuple[tuple[str, ...], str]:
    rate = diagnosis.nominal_rate
    if rate is None:
        raise RepairError("The source nominal frame rate is unknown; a constant-rate QTGMC rescue is unsafe.")
    target_pix_fmt = choose_ffv1_pixel_format(media.video.pix_fmt, supported_ffv1_formats)
    format_start = media.start_time or 0.0
    first_video_pts = diagnosis.first_pts if diagnosis.first_pts is not None else format_start
    video_start_offset = max(0.0, first_video_pts - format_start)
    filters = [f"setpts=PTS-STARTPTS+{video_start_offset:.9f}/TB"]
    if (media.video.pix_fmt or "").startswith("yuvj"):
        filters.append("scale=in_range=pc:out_range=pc")
    filters.append(
        f"fps=fps={_format_rate(rate)}:start_time=0:round=near:eof_action=pass"
    )
    filters.append(f"format=pix_fmts={target_pix_fmt}")
    sar = media.video.sample_aspect_ratio or Fraction(1, 1)
    filters.append(f"setsar=sar={sar.numerator}/{sar.denominator}:max={max(sar.numerator, sar.denominator)}")
    field_mode = _field_mode(media.video.field_order)
    if field_mode:
        filters.append(f"setfield=mode={field_mode}")
    slices = "16" if (media.video.width or 0) >= 512 else "4"
    command = [
        str(ffmpeg),
        "-hide_banner",
        "-nostdin",
        "-y",
        "-fflags",
        "+genpts",
        "-err_detect",
        "ignore_err",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-map",
        "0:s?",
        "-map",
        "0:t?",
        "-map",
        "0:d?",
        "-vf",
        ",".join(filters),
        "-c",
        "copy",
        "-c:v:0",
        "ffv1",
        "-level:v:0",
        "3",
        "-coder:v:0",
        "2",
        "-context:v:0",
        "1",
        "-g:v:0",
        "1",
        "-slicecrc:v:0",
        "1",
        "-slices:v:0",
        slices,
        "-pix_fmt:v:0",
        target_pix_fmt,
        "-fps_mode:v:0",
        "cfr",
        "-map_metadata",
        "0",
        "-map_chapters",
        "0",
        "-metadata:s:v:0",
        "DURATION=",
        "-metadata:s:v:0",
        "DURATION-eng=",
    ]
    command += _color_args(media)
    command += [
        "-max_muxing_queue_size",
        "4096",
        "-progress",
        "pipe:1",
        "-nostats",
        str(partial),
    ]
    return tuple(command), target_pix_fmt


def _meaningful_tags(tags: dict[str, str]) -> dict[str, str]:
    meaningful: dict[str, str] = {}
    for key, value in tags.items():
        normalized = key.upper()
        base = normalized.removesuffix("-ENG")
        if base in TECHNICAL_TAGS or any(base.startswith(prefix) for prefix in TECHNICAL_TAG_PREFIXES):
            continue
        meaningful[key.casefold()] = value
    return meaningful


def _compare_stream_group(
    kind: str,
    source: tuple[StreamInfo, ...],
    output: tuple[StreamInfo, ...],
    errors: list[str],
    *,
    source_format_start: float | None,
) -> None:
    if len(source) != len(output):
        errors.append(f"{kind.title()} stream count changed from {len(source)} to {len(output)}.")
        return
    for position, (before, after) in enumerate(zip(source, output), start=1):
        if before.codec_name != after.codec_name:
            errors.append(
                f"{kind.title()} stream {position} codec changed from {before.codec_name} to {after.codec_name}."
            )
        if before.disposition != after.disposition:
            errors.append(f"{kind.title()} stream {position} disposition flags changed.")
        if kind != "subtitle" and before.start_time is not None and after.start_time is not None:
            # FFmpeg normalizes input timestamps by the source format start unless
            # -copyts is requested. Do not use the output container start as a
            # reference: Matroska can report the first subtitle cluster there even
            # when audio/video begin at zero. Stream-level timestamps are the
            # stable playback/sync evidence.
            before_relative = before.start_time - (source_format_start or 0.0)
            after_relative = after.start_time
            if abs(before_relative - after_relative) > 0.1:
                errors.append(
                    f"{kind.title()} stream {position} relative start changed from "
                    f"{before_relative:.6f}s to {after_relative:.6f}s."
                )
        expected_tags = _meaningful_tags(before.tags)
        actual_tags = _meaningful_tags(after.tags)
        for key, value in expected_tags.items():
            if actual_tags.get(key) != value:
                errors.append(f"{kind.title()} stream {position} metadata {key!r} was not preserved.")


def _compare_chapters(source: tuple[dict, ...], output: tuple[dict, ...], errors: list[str]) -> None:
    if len(source) != len(output):
        errors.append(f"Chapter count changed from {len(source)} to {len(output)}.")
        return
    for position, (before, after) in enumerate(zip(source, output), start=1):
        for key in ("start_time", "end_time"):
            try:
                expected = float(before[key])
                actual = float(after[key])
            except (KeyError, TypeError, ValueError):
                continue
            if abs(expected - actual) > 0.001:
                errors.append(
                    f"Chapter {position} {key.replace('_', ' ')} changed from {expected:.6f}s to {actual:.6f}s."
                )
        expected_tags = _meaningful_tags({str(key): str(value) for key, value in before.get("tags", {}).items()})
        actual_tags = _meaningful_tags({str(key): str(value) for key, value in after.get("tags", {}).items()})
        for key, value in expected_tags.items():
            if actual_tags.get(key) != value:
                errors.append(f"Chapter {position} metadata {key!r} was not preserved.")


def _field_parity(field_order: str | None) -> str | None:
    normalized = (field_order or "").lower()
    if normalized in {"tt", "tb"}:
        return "tff"
    if normalized in {"bb", "bt"}:
        return "bff"
    if normalized == "progressive":
        return "progressive"
    return None


class SourceRepairer:
    """Diagnose and create a separately validated QTGMC-compatible source copy."""

    def __init__(self) -> None:
        self.cancel_event = threading.Event()
        self._process_lock = threading.Lock()
        self._process: subprocess.Popen | None = None
        self._packet_fingerprint_cache: dict[
            tuple[str, int, int, str, int], dict[int, PacketTimelineFingerprint]
        ] = {}

    def cancel(self) -> None:
        self.cancel_event.set()
        with self._process_lock:
            process = self._process
        if process is not None and process.poll() is None:
            try:
                process.terminate()
            except OSError:
                pass

    def _register(self, process: subprocess.Popen) -> None:
        with self._process_lock:
            self._process = process

    def _clear_process(self, process: subprocess.Popen) -> None:
        with self._process_lock:
            if self._process is process:
                self._process = None

    def _check_canceled(self) -> None:
        if self.cancel_event.is_set():
            raise RepairCancelled("Source repair canceled")

    def diagnose(
        self,
        ffprobe: Path,
        media: MediaProbe,
        *,
        log_callback: RepairLogCallback | None = None,
        progress_callback: RepairProgressCallback | None = None,
        phase: str = "diagnosing source",
    ) -> TimelineDiagnosis:
        """Fully decode frame timestamps while retaining bounded diagnostics."""

        self._check_canceled()
        command = [
            str(ffprobe),
            "-v",
            "warning",
            "-select_streams",
            "v:0",
            "-show_entries",
            "frame=best_effort_timestamp_time,interlaced_frame,top_field_first",
            "-of",
            "compact=p=0:nk=0",
            "--",
            str(media.path),
        ]
        env = os.environ.copy()
        env["AV_LOG_FORCE_NOCOLOR"] = "1"
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=PROCESS_FLAGS,
            env=env,
        )
        self._register(process)
        assert process.stdout is not None
        assert process.stderr is not None
        warning_samples: deque[str] = deque(maxlen=80)
        warning_count = 0
        severe_warning_count = 0
        warning_lock = threading.Lock()

        def drain_stderr() -> None:
            nonlocal warning_count, severe_warning_count
            for raw in process.stderr:
                line = raw.strip()
                if not line:
                    continue
                lowered = line.casefold()
                with warning_lock:
                    warning_count += 1
                    warning_samples.append(line)
                    if any(pattern in lowered for pattern in SEVERE_DIAGNOSTIC_PATTERNS):
                        severe_warning_count += 1
                if log_callback:
                    log_callback(f"FFprobe: {line}")

        stderr_thread = threading.Thread(target=drain_stderr, daemon=True)
        stderr_thread.start()
        rate = _nominal_rate(media)
        period = 1.0 / float(rate) if rate else None
        gap_threshold = max((period or 0.04) * 2.5, 0.1)
        decoded_frames = 0
        timestamped_frames = 0
        interlaced_frames = 0
        progressive_frames = 0
        first_pts: float | None = None
        last_pts: float | None = None
        previous_pts: float | None = None
        non_monotonic = 0
        gaps: list[TimelineGap] = []
        reported_duration = _reported_video_duration(media)
        last_progress_frame = 0
        try:
            for raw in process.stdout:
                self._check_canceled()
                line = raw.strip()
                if not line:
                    continue
                fields: dict[str, str] = {}
                for item in line.split("|"):
                    key, separator, value = item.partition("=")
                    if separator:
                        fields[key] = value
                decoded_frames += 1
                if fields.get("interlaced_frame") == "1":
                    interlaced_frames += 1
                elif fields.get("interlaced_frame") == "0":
                    progressive_frames += 1
                raw_pts = fields.get("best_effort_timestamp_time")
                try:
                    pts = float(raw_pts) if raw_pts not in (None, "", "N/A") else None
                except ValueError:
                    pts = None
                if pts is not None:
                    timestamped_frames += 1
                    if first_pts is None:
                        first_pts = pts
                    if previous_pts is not None:
                        step = pts - previous_pts
                        if step <= 0:
                            non_monotonic += 1
                        elif step > gap_threshold:
                            missing = max(0, round(step / (period or step)) - 1)
                            gaps.append(
                                TimelineGap(
                                    after_frame=decoded_frames - 1,
                                    previous_pts=previous_pts,
                                    current_pts=pts,
                                    duration_seconds=step,
                                    estimated_missing_frames=missing,
                                )
                            )
                    previous_pts = pts
                    last_pts = pts
                if progress_callback and decoded_frames - last_progress_frame >= 250:
                    last_progress_frame = decoded_frames
                    progress: dict[str, str] = {"phase": phase, "frame": str(decoded_frames)}
                    if last_pts is not None:
                        progress["out_time_us"] = str(max(0, round((last_pts - (first_pts or 0.0)) * 1_000_000)))
                    if reported_duration:
                        progress["duration_us"] = str(round(reported_duration * 1_000_000))
                    progress_callback(progress)
            process.stdout.close()
            return_code = process.wait()
            stderr_thread.join(timeout=10)
            if self.cancel_event.is_set():
                raise RepairCancelled("Source repair canceled")
            if return_code != 0:
                raise RepairError(f"FFprobe decoded-timeline scan exited with code {return_code}.")
        finally:
            if process.poll() is None:
                try:
                    process.terminate()
                    process.wait(timeout=5)
                except (OSError, subprocess.TimeoutExpired):
                    try:
                        process.kill()
                    except OSError:
                        pass
            stderr_thread.join(timeout=5)
            for stream in (process.stdout, process.stderr):
                try:
                    stream.close()
                except (OSError, ValueError):
                    pass
            self._clear_process(process)

        decoded_span = decoded_frames / float(rate) if rate else None
        pts_duration = (
            last_pts - first_pts + period
            if first_pts is not None and last_pts is not None and period is not None
            else None
        )
        qtgmc_error = qtgmc_timeline_integrity_error(media, decoded_frames)
        classification, recommended = classify_timeline_condition(
            qtgmc_error=qtgmc_error,
            material_gap_count=len(gaps),
            non_monotonic_steps=non_monotonic,
            severe_warning_count=severe_warning_count,
            timestamps_complete=decoded_frames > 0 and decoded_frames == timestamped_frames,
        )
        return TimelineDiagnosis(
            source_path=media.path,
            nominal_rate=rate,
            reported_video_duration=reported_duration,
            container_duration=media.duration,
            decoded_frames=decoded_frames,
            timestamped_frames=timestamped_frames,
            decoded_cfr_span_seconds=decoded_span,
            first_pts=first_pts,
            last_pts=last_pts,
            pts_duration_seconds=pts_duration,
            material_gaps=tuple(gaps),
            non_monotonic_steps=non_monotonic,
            sampled_interlaced_frames=interlaced_frames,
            sampled_progressive_frames=progressive_frames,
            decoder_warning_count=warning_count,
            decoder_warning_samples=tuple(warning_samples),
            severe_warning_count=severe_warning_count,
            qtgmc_error=qtgmc_error,
            classification=classification,
            recommended_method=recommended,
        )

    def _run_ffmpeg(
        self,
        command: tuple[str, ...],
        *,
        phase: str,
        log_callback: RepairLogCallback,
        progress_callback: RepairProgressCallback | None,
    ) -> dict[str, str]:
        self._check_canceled()
        env = os.environ.copy()
        env["AV_LOG_FORCE_NOCOLOR"] = "1"
        process = subprocess.Popen(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=PROCESS_FLAGS,
            env=env,
        )
        self._register(process)
        assert process.stdout is not None
        assert process.stderr is not None
        stderr_tail: deque[str] = deque(maxlen=120)

        def drain_stderr() -> None:
            for raw in process.stderr:
                line = raw.rstrip()
                if line:
                    stderr_tail.append(line)
                    log_callback(f"FFmpeg: {line}")

        stderr_thread = threading.Thread(target=drain_stderr, daemon=True)
        stderr_thread.start()
        state: dict[str, str] = {}
        final_state: dict[str, str] = {}
        try:
            for raw in process.stdout:
                self._check_canceled()
                key, separator, value = raw.strip().partition("=")
                if separator:
                    state[key] = value
                    if key == "progress":
                        final_state.update(state)
                        if progress_callback:
                            progress_callback({"phase": phase, **state})
                        state.clear()
            process.stdout.close()
            return_code = process.wait()
            stderr_thread.join(timeout=10)
            if self.cancel_event.is_set():
                raise RepairCancelled("Source repair canceled")
            combined_tail = "\n".join(stderr_tail).casefold()
            fatal = next((pattern for pattern in FATAL_ZERO_EXIT_PATTERNS if pattern in combined_tail), None)
            if return_code != 0:
                raise RepairError(f"FFmpeg {phase} exited with code {return_code}. See the retained repair log.")
            if fatal:
                raise RepairError(f"FFmpeg {phase} emitted a fatal diagnostic despite exit code 0: {fatal}")
            return final_state
        finally:
            if process.poll() is None:
                try:
                    process.terminate()
                    process.wait(timeout=5)
                except (OSError, subprocess.TimeoutExpired):
                    try:
                        process.kill()
                    except OSError:
                        pass
            stderr_thread.join(timeout=5)
            for stream in (process.stdout, process.stderr):
                try:
                    stream.close()
                except (OSError, ValueError):
                    pass
            self._clear_process(process)

    def _packet_fingerprints(
        self,
        ffprobe: Path,
        path: Path,
        *,
        selector: str,
        origin: float,
        log_callback: RepairLogCallback,
        progress_callback: RepairProgressCallback | None,
        phase: str,
    ) -> dict[int, PacketTimelineFingerprint]:
        """Hash packet-level timing for copied streams without buffering probe output."""

        self._check_canceled()
        stat = path.stat()
        origin_us = round(origin * 1_000_000)
        key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns, selector, origin_us)
        cached = self._packet_fingerprint_cache.get(key)
        if cached is not None:
            return cached
        if progress_callback:
            progress_callback({"phase": phase})
        command = [
            str(ffprobe),
            "-v",
            "error",
            "-select_streams",
            selector,
            "-show_packets",
            "-show_entries",
            "packet=stream_index,pts_time,dts_time,duration_time,size",
            "-of",
            "compact=p=0:nk=0",
            "--",
            str(path),
        ]
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=PROCESS_FLAGS,
        )
        self._register(process)
        assert process.stdout is not None
        assert process.stderr is not None
        stderr_tail: deque[str] = deque(maxlen=40)

        def drain_stderr() -> None:
            for raw in process.stderr:
                line = raw.strip()
                if line:
                    stderr_tail.append(line)

        stderr_thread = threading.Thread(target=drain_stderr, daemon=True)
        stderr_thread.start()
        hashers: dict[int, Any] = {}
        counts: dict[int, int] = {}
        first_pts: dict[int, int | None] = {}
        last_pts: dict[int, int | None] = {}

        def timestamp_us(raw_value: str | None, *, subtract_origin: bool) -> int | None:
            if raw_value in (None, "", "N/A"):
                return None
            try:
                value = round(float(raw_value) * 1_000_000)
            except ValueError:
                return None
            return value - origin_us if subtract_origin else value

        try:
            for raw in process.stdout:
                self._check_canceled()
                fields: dict[str, str] = {}
                for item in raw.strip().split("|"):
                    field, separator, value = item.partition("=")
                    if separator:
                        fields[field] = value
                try:
                    index = int(fields["stream_index"])
                except (KeyError, ValueError):
                    continue
                pts = timestamp_us(fields.get("pts_time"), subtract_origin=True)
                dts = timestamp_us(fields.get("dts_time"), subtract_origin=True)
                duration = timestamp_us(fields.get("duration_time"), subtract_origin=False)
                digest = hashers.setdefault(index, hashlib.sha256())
                digest.update(f"{pts}|{dts}|{duration}|{fields.get('size', '')}\n".encode("ascii"))
                counts[index] = counts.get(index, 0) + 1
                if index not in first_pts:
                    first_pts[index] = pts
                last_pts[index] = pts
            process.stdout.close()
            return_code = process.wait()
            stderr_thread.join(timeout=10)
            if self.cancel_event.is_set():
                raise RepairCancelled("Source repair canceled")
            if return_code != 0:
                detail = f" ({stderr_tail[-1]})" if stderr_tail else ""
                raise RepairError(f"FFprobe packet-timeline scan exited with code {return_code}{detail}.")
        finally:
            if process.poll() is None:
                try:
                    process.terminate()
                    process.wait(timeout=5)
                except (OSError, subprocess.TimeoutExpired):
                    try:
                        process.kill()
                    except OSError:
                        pass
            stderr_thread.join(timeout=5)
            for stream in (process.stdout, process.stderr):
                try:
                    stream.close()
                except (OSError, ValueError):
                    pass
            self._clear_process(process)
        result = {
            index: PacketTimelineFingerprint(
                packet_count=counts[index],
                first_pts_us=first_pts.get(index),
                last_pts_us=last_pts.get(index),
                digest=digest.hexdigest(),
            )
            for index, digest in hashers.items()
        }
        self._packet_fingerprint_cache[key] = result
        log_callback(f"Validated packet-level timing fingerprints for {sum(counts.values())} {selector} packet(s).")
        return result

    def _validate_candidate(
        self,
        ffprobe: Path,
        source_media: MediaProbe,
        source_diagnosis: TimelineDiagnosis,
        candidate: Path,
        *,
        method: str,
        target_pix_fmt: str | None,
        log_callback: RepairLogCallback,
        progress_callback: RepairProgressCallback | None,
        phase: str,
        prior_full_validation: RepairValidation | None = None,
    ) -> RepairValidation:
        errors: list[str] = []
        warnings: list[str] = []
        try:
            output_media = probe_media(ffprobe, candidate, sample_frames=64, timeout=180.0)
            if prior_full_validation is None:
                output_diagnosis = self.diagnose(
                    ffprobe,
                    output_media,
                    log_callback=log_callback,
                    progress_callback=progress_callback,
                    phase=phase,
                )
            else:
                if (
                    not prior_full_validation.valid
                    or prior_full_validation.media is None
                    or prior_full_validation.diagnosis is None
                ):
                    raise RepairError(
                        "Bounded final reopen requires a valid prior full-decode validation."
                    )
                if _repair_structure_signature(output_media) != _repair_structure_signature(
                    prior_full_validation.media
                ):
                    raise RepairError(
                        "The promoted repair reopened with different container or stream structure."
                    )
                output_diagnosis = replace(
                    prior_full_validation.diagnosis,
                    source_path=candidate,
                )
                warnings.append(
                    "Full-decode evidence was retained from the identical thoroughly validated partial; "
                    "the final path was reopened with bounded decoded samples and an exact structural comparison."
                )
        except Exception as exc:
            action = "reopened" if prior_full_validation is not None else "reopened and fully decoded"
            return RepairValidation(False, (f"Candidate could not be {action}: {exc}",), (), None, None)

        before = source_media.video
        after = output_media.video
        if "matroska" not in output_media.format_name:
            errors.append(f"Repair output container is {output_media.format_name}, expected Matroska.")
        expected_codec = "ffv1" if method == "ffv1_rescue" else before.codec_name
        if after.codec_name != expected_codec:
            errors.append(f"Video codec is {after.codec_name}, expected {expected_codec} for {method}.")
        if target_pix_fmt and after.pix_fmt != target_pix_fmt:
            errors.append(f"Video pixel format is {after.pix_fmt}, expected exact rescue format {target_pix_fmt}.")
        if after.width != before.width or after.height != before.height:
            errors.append(
                f"Stored raster changed from {before.width}x{before.height} to {after.width}x{after.height}."
            )
        if after.sample_aspect_ratio != before.sample_aspect_ratio:
            errors.append(
                f"Sample aspect ratio changed from {before.sample_aspect_ratio} to {after.sample_aspect_ratio}."
            )
        if after.display_aspect_ratio != before.display_aspect_ratio:
            errors.append(
                f"Display aspect ratio changed from {before.display_aspect_ratio} to {after.display_aspect_ratio}."
            )
        if before.disposition != after.disposition:
            errors.append("Video stream disposition flags changed.")
        expected_rate = source_diagnosis.nominal_rate
        actual_rate = after.avg_frame_rate or after.r_frame_rate
        if expected_rate and actual_rate != expected_rate:
            errors.append(f"Nominal rate changed from {expected_rate} to {actual_rate}.")
        if (after.bits_per_raw_sample or 0) < (before.bits_per_raw_sample or 0):
            errors.append(
                f"Decoded bit depth fell from {before.bits_per_raw_sample} to {after.bits_per_raw_sample}."
            )
        if _field_parity(before.field_order) and _field_parity(after.field_order) != _field_parity(before.field_order):
            errors.append(f"Field parity changed from {before.field_order} to {after.field_order}.")
        for label, expected, actual in (
            ("color range", before.color_range, after.color_range),
            ("color matrix", before.color_space, after.color_space),
            ("color transfer", before.color_transfer, after.color_transfer),
            ("color primaries", before.color_primaries, after.color_primaries),
        ):
            if expected and actual != expected:
                errors.append(f"Video {label} changed from {expected} to {actual}.")
        for kind in ("audio", "subtitle", "attachment", "data"):
            _compare_stream_group(
                kind,
                source_media.streams_of_type(kind),
                output_media.streams_of_type(kind),
                errors,
                source_format_start=source_media.start_time,
            )
        source_subtitles = source_media.streams_of_type("subtitle")
        output_subtitles = output_media.streams_of_type("subtitle")
        if (
            prior_full_validation is None
            and source_subtitles
            and len(source_subtitles) == len(output_subtitles)
        ):
            try:
                source_packets = self._packet_fingerprints(
                    ffprobe,
                    source_media.path,
                    selector="s",
                    origin=source_media.start_time or 0.0,
                    log_callback=log_callback,
                    progress_callback=progress_callback,
                    phase=f"{phase}: checking source subtitle timestamps",
                )
                output_packets = self._packet_fingerprints(
                    ffprobe,
                    candidate,
                    selector="s",
                    origin=0.0,
                    log_callback=log_callback,
                    progress_callback=progress_callback,
                    phase=f"{phase}: checking repaired subtitle timestamps",
                )
                for position, (source_stream, output_stream) in enumerate(
                    zip(source_subtitles, output_subtitles), start=1
                ):
                    expected = source_packets.get(source_stream.index)
                    actual = output_packets.get(output_stream.index)
                    if expected != actual:
                        errors.append(
                            f"Subtitle stream {position} packet count, timestamps, durations, or payload sizes changed."
                        )
            except (OSError, RepairError) as exc:
                errors.append(f"Subtitle packet-timeline validation failed: {exc}")
        _compare_chapters(source_media.chapters, output_media.chapters, errors)
        expected_format_tags = _meaningful_tags(source_media.format_tags)
        actual_format_tags = _meaningful_tags(output_media.format_tags)
        for key, value in expected_format_tags.items():
            if actual_format_tags.get(key) != value:
                errors.append(f"Container metadata {key!r} was not preserved.")
        expected_video_tags = _meaningful_tags(before.tags)
        actual_video_tags = _meaningful_tags(after.tags)
        for key, value in expected_video_tags.items():
            if actual_video_tags.get(key) != value:
                errors.append(f"Video stream metadata {key!r} was not preserved.")
        if not output_diagnosis.qtgmc_compatible:
            errors.append(output_diagnosis.qtgmc_error or "Output does not satisfy the QTGMC timeline contract.")
        if output_diagnosis.material_gaps:
            errors.append(f"Output still contains {len(output_diagnosis.material_gaps)} material timestamp gap(s).")
        if output_diagnosis.non_monotonic_steps:
            errors.append(
                f"Output contains {output_diagnosis.non_monotonic_steps} non-monotonic decoded timestamp step(s)."
            )
        if output_diagnosis.severe_warning_count:
            errors.append(
                f"Output full decode emitted {output_diagnosis.severe_warning_count} severe corruption diagnostic(s)."
            )
        if output_diagnosis.decoded_frames != output_diagnosis.timestamped_frames:
            errors.append("Not every decoded output frame has a usable timestamp.")
        initial_video_offset = max(
            0.0,
            (source_diagnosis.first_pts or 0.0) - (source_media.start_time or 0.0),
        )
        target_duration = (
            source_diagnosis.pts_duration_seconds + initial_video_offset
            if source_diagnosis.pts_duration_seconds is not None
            else None
        )
        if target_duration is not None and output_diagnosis.pts_duration_seconds is not None and expected_rate:
            tolerance = max(1.0, 2.0 / float(expected_rate))
            difference = output_diagnosis.pts_duration_seconds - target_duration
            if abs(difference) > tolerance:
                errors.append(
                    f"Output video timeline {output_diagnosis.pts_duration_seconds:.6f}s differs from the measured "
                    f"source timestamp timeline {target_duration:.6f}s by {difference:+.6f}s."
                )
        if method == "stream_copy_remux" and output_diagnosis.decoded_frames != source_diagnosis.decoded_frames:
            errors.append(
                f"Stream-copy remux changed decoded video frame count from {source_diagnosis.decoded_frames} "
                f"to {output_diagnosis.decoded_frames}."
            )
        return RepairValidation(not errors, tuple(errors), tuple(warnings), output_media, output_diagnosis)

    def run(
        self,
        request: RepairRequest,
        source_media: MediaProbe,
        capabilities: CapabilityReport,
        *,
        log_callback: RepairLogCallback | None = None,
        progress_callback: RepairProgressCallback | None = None,
    ) -> RepairResult:
        if request.mode not in {"automatic", "remux_only", "rescue_only"}:
            return RepairResult(False, False, f"Unknown repair mode: {request.mode}", None, None, None, None, None, None, None, 0, 0)
        if not capabilities.ffmpeg_path or not capabilities.ffprobe_path:
            return RepairResult(False, False, "FFmpeg and FFprobe are required for source repair.", None, None, None, None, None, None, None, 0, 0)
        source = request.source_path
        output = request.output_path
        if not source.is_file() or not _same_path(source, source_media.path):
            return RepairResult(False, False, "The selected source no longer matches the analyzed media.", None, None, None, None, None, None, None, 0, 0)
        if output.suffix.casefold() != ".mkv":
            return RepairResult(False, False, "Repair output must be a separate Matroska .mkv file.", None, None, None, None, None, None, None, 0, 0)
        final_log = output.with_name(output.name + ".Repair.log")
        final_report = output.with_name(output.name + ".Repair.json")
        if any(_same_path(source, path) for path in (output, final_log, final_report)):
            return RepairResult(False, False, "The source and every repair artifact must use different paths.", None, None, None, None, None, None, None, 0, 0)
        output.parent.mkdir(parents=True, exist_ok=True)
        completed_artifacts = (output, final_log, final_report)
        snapshots = {path: _snapshot(path) for path in completed_artifacts}
        existing = [path for path, identity in snapshots.items() if identity[0]]
        if existing and not request.overwrite_approved:
            return RepairResult(
                False,
                False,
                "Existing repair artifacts require explicit replacement approval: " + ", ".join(str(path) for path in existing),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                0,
                0,
            )

        run_token = uuid.uuid4().hex[:12]
        partial = output.with_name(f".{output.stem}.Repair.partial.{run_token}.mkv")
        run_log = output.with_name(f".{output.stem}.Repair.{run_token}.run.log")
        report_temp = output.with_name(f".{output.stem}.Repair.{run_token}.json")
        if partial.exists() or run_log.exists() or report_temp.exists():
            return RepairResult(False, False, "A unique repair temporary path unexpectedly exists.", None, None, None, None, None, None, None, 0, 0)
        source_snapshot = _snapshot(source)
        diagnosis: TimelineDiagnosis | None = None
        partial_validation: RepairValidation | None = None
        final_validation: RepairValidation | None = None
        method: str | None = None
        target_pix_fmt: str | None = None
        command_history: list[tuple[str, ...]] = []
        progress_state: dict[str, str] = {}
        backups: dict[Path, Path] = {}
        promoted = False
        quarantine: Path | None = None
        output_hash: str | None = None
        partial_identity: tuple[int, int, int, int] | None = None
        repeated_frames = 0
        dropped_frames = 0
        error_message = ""
        status = "failed"
        log_lock = threading.Lock()

        with run_log.open("w", encoding="utf-8", buffering=1) as log:
            def write_log(message: str) -> None:
                rendered = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
                with log_lock:
                    log.write(rendered + "\n")
                    log.flush()
                if log_callback:
                    log_callback(rendered)

            try:
                write_log(f"Deinterlace Studio {__version__} source repair started")
                write_log(f"Source: {source}")
                write_log(f"Requested repair output: {output}")
                write_log(f"Mode: {request.mode}")
                diagnosis = self.diagnose(
                    capabilities.ffprobe_path,
                    source_media,
                    log_callback=write_log,
                    progress_callback=progress_callback,
                )
                write_log(
                    f"Diagnosis: {diagnosis.classification}; decoded={diagnosis.decoded_frames}; "
                    f"timestamp gaps={len(diagnosis.material_gaps)}; non-monotonic={diagnosis.non_monotonic_steps}; "
                    f"severe diagnostics={diagnosis.severe_warning_count}."
                )
                if diagnosis.largest_gap:
                    gap = diagnosis.largest_gap
                    write_log(
                        f"Largest gap: {gap.previous_pts:.6f}s -> {gap.current_pts:.6f}s "
                        f"({gap.duration_seconds:.6f}s; about {gap.estimated_missing_frames} unavailable frames)."
                    )
                self._check_canceled()
                if _snapshot(source) != source_snapshot:
                    raise RepairError("The source file changed during diagnosis; repair was aborted.")
                if diagnosis.recommended_method == "blocked":
                    raise RepairError(
                        "Not every decoded frame has a usable timestamp, so neither remux nor automatic CFR rescue can be proven safe."
                    )
                if request.mode == "remux_only":
                    method = "stream_copy_remux"
                elif request.mode == "rescue_only":
                    method = "ffv1_rescue"
                else:
                    method = diagnosis.recommended_method
                if method == "none":
                    write_log("The source already satisfies the complete QTGMC timeline and decode checks; no repair output was created.")
                    for artifact in (final_log, final_report):
                        if _snapshot(artifact) != snapshots[artifact]:
                            raise RepairError(f"Approved diagnostic sidecar changed during the health scan: {artifact}")
                        if artifact.exists():
                            backup = artifact.with_name(f".{artifact.name}.backup.{run_token}")
                            if backup.exists():
                                raise RepairError(f"Backup collision: {backup}")
                            os.replace(artifact, backup)
                            backups[artifact] = backup
                    status = "no_repair_needed"
                else:
                    if method == "stream_copy_remux":
                        command = build_remux_command(capabilities.ffmpeg_path, source, partial)
                    else:
                        supported = capabilities.encoder_pixel_formats.get("ffv1", ())
                        if "ffv1" not in capabilities.encoders or not supported:
                            raise RepairError("The selected FFmpeg build does not expose FFV1 with usable pixel formats.")
                        command, target_pix_fmt = build_rescue_command(
                            capabilities.ffmpeg_path,
                            source,
                            partial,
                            source_media,
                            diagnosis,
                            supported,
                        )
                    command_history.append(command)
                    write_log(f"Selected method: {method}")
                    write_log("Command: " + subprocess.list2cmdline(list(command)))
                    progress_state = self._run_ffmpeg(
                        command,
                        phase="creating repair candidate",
                        log_callback=write_log,
                        progress_callback=progress_callback,
                    )
                    partial_validation = self._validate_candidate(
                        capabilities.ffprobe_path,
                        source_media,
                        diagnosis,
                        partial,
                        method=method,
                        target_pix_fmt=target_pix_fmt,
                        log_callback=write_log,
                        progress_callback=progress_callback,
                        phase="validating repair candidate",
                    )
                    if not partial_validation.valid and method == "stream_copy_remux" and request.mode == "automatic":
                        for error in partial_validation.errors:
                            write_log("Remux validation: " + error)
                        write_log("Stream-copy remux did not repair the timeline; switching to FFV1 lossless rescue.")
                        _safe_unlink(partial)
                        method = "ffv1_rescue"
                        supported = capabilities.encoder_pixel_formats.get("ffv1", ())
                        if "ffv1" not in capabilities.encoders or not supported:
                            raise RepairError("Remux failed and the selected FFmpeg build cannot create the required FFV1 rescue.")
                        command, target_pix_fmt = build_rescue_command(
                            capabilities.ffmpeg_path,
                            source,
                            partial,
                            source_media,
                            diagnosis,
                            supported,
                        )
                        command_history.append(command)
                        write_log("Command: " + subprocess.list2cmdline(list(command)))
                        progress_state = self._run_ffmpeg(
                            command,
                            phase="creating FFV1 rescue",
                            log_callback=write_log,
                            progress_callback=progress_callback,
                        )
                        partial_validation = self._validate_candidate(
                            capabilities.ffprobe_path,
                            source_media,
                            diagnosis,
                            partial,
                            method=method,
                            target_pix_fmt=target_pix_fmt,
                            log_callback=write_log,
                            progress_callback=progress_callback,
                            phase="validating FFV1 rescue",
                        )
                    if not partial_validation.valid:
                        for error in partial_validation.errors:
                            write_log("Validation error: " + error)
                        raise RepairError("Repair candidate failed validation; no output was promoted.")
                    partial_identity = _promotion_identity(partial)
                    if _snapshot(source) != source_snapshot:
                        raise RepairError("The source file changed while creating the repair; promotion was aborted.")
                    for path, identity in snapshots.items():
                        if _snapshot(path) != identity:
                            raise RepairError(f"Approved repair artifact changed during processing: {path}")
                    for artifact in completed_artifacts:
                        if artifact.exists():
                            backup = artifact.with_name(f".{artifact.name}.backup.{run_token}")
                            if backup.exists():
                                raise RepairError(f"Backup collision: {backup}")
                            os.replace(artifact, backup)
                            backups[artifact] = backup
                    os.replace(partial, output)
                    promoted = True
                    if _promotion_identity(output) != partial_identity:
                        raise RepairError(
                            "Atomic promotion did not preserve the validated repair partial's exact file identity."
                        )
                    write_log(
                        "Atomic promotion preserved the thoroughly validated repair partial's exact file identity."
                    )
                    write_log(
                        "Performing bounded final-path reopen and structural comparison without repeating the full decode."
                    )
                    final_validation = self._validate_candidate(
                        capabilities.ffprobe_path,
                        source_media,
                        diagnosis,
                        output,
                        method=method,
                        target_pix_fmt=target_pix_fmt,
                        log_callback=write_log,
                        progress_callback=progress_callback,
                        phase="final repair validation",
                        prior_full_validation=partial_validation,
                    )
                    if not final_validation.valid:
                        for error in final_validation.errors:
                            write_log("Final validation error: " + error)
                        raise RepairError("Promoted repair failed bounded final reopen validation.")
                    if _snapshot(source) != source_snapshot:
                        raise RepairError("The source file changed before final acceptance; repair was rolled back.")
                    hash_snapshot = _snapshot(output)
                    write_log(
                        "Bounded final reopen passed; calculating SHA-256 once."
                    )
                    output_hash = sha256_file(output)
                    if _snapshot(output) != hash_snapshot:
                        raise RepairError("The repair output changed while its SHA-256 was calculated.")
                    output_diagnosis = final_validation.diagnosis
                    if output_diagnosis:
                        repeated_frames = max(0, output_diagnosis.decoded_frames - diagnosis.decoded_frames)
                        dropped_frames = max(0, diagnosis.decoded_frames - output_diagnosis.decoded_frames)
                    write_log(f"Repair output SHA-256: {output_hash}")
                    write_log(
                        f"Measured net materialized frame count: +{repeated_frames}; net removed/non-monotonic count: {dropped_frames}."
                    )
                    status = "success"

                report_payload = {
                    "application": "Deinterlace Studio",
                    "application_version": __version__,
                    "completed_at_local": time.strftime("%Y-%m-%d %H:%M:%S %z"),
                    "status": status,
                    "method": method,
                    "limitation": (
                        "Unavailable or corrupt pictures were not reconstructed. The FFV1 rescue repeats whole decoded "
                        "interlaced frames where timestamps prove picture time is missing."
                        if method == "ffv1_rescue"
                        else None
                    ),
                    "source_file": {
                        "path": str(source),
                        "size": source.stat().st_size,
                        "mtime_ns": source.stat().st_mtime_ns,
                    },
                    "source_probe": json_safe(source_media),
                    "source_diagnosis": json_safe(diagnosis),
                    "commands": [list(command) for command in command_history],
                    "partial_validation": json_safe(partial_validation),
                    "final_validation": json_safe(final_validation),
                    "validation_strategy": {
                        "partial_full_decode_validations": 1,
                        "promoted_full_decode_repeats": 0,
                        "same_file_atomic_promotion_verified": True,
                        "final_reopen": "bounded probe, decoded samples, and exact structural comparison",
                        "subtitle_packet_fingerprint_scans": 1 if source_media.subtitle_count else 0,
                        "sha256_passes": 1,
                    },
                    "progress_final": progress_state,
                    "repeated_frames_net": repeated_frames,
                    "dropped_frames_net": dropped_frames,
                    "output": (
                        {"path": str(output), "size": output.stat().st_size, "sha256": output_hash}
                        if status == "success"
                        else None
                    ),
                }
                report_temp.write_text(json.dumps(report_payload, ensure_ascii=False, indent=2), encoding="utf-8")
                write_log("Repair audit report prepared.")
            except RepairCancelled as exc:
                _safe_unlink(partial)
                _safe_unlink(report_temp)
                error_message = str(exc)
                write_log(error_message)
                status = "canceled"
            except Exception as exc:
                _safe_unlink(partial)
                _safe_unlink(report_temp)
                error_message = str(exc)
                write_log(f"Repair failed: {error_message}")
                if promoted and output.exists():
                    quarantine = output.with_name(f"{output.stem}.Repair.failed.{run_token}.mkv")
                    os.replace(output, quarantine)
                for original, backup in backups.items():
                    if backup.exists():
                        os.replace(backup, original)
                status = "failed"

        if status in {"failed", "canceled"}:
            destination_log = output.with_name(f"{output.stem}.Repair.{status}.{run_token}.log")
            destination_report: Path | None = output.with_name(
                f"{output.stem}.Repair.{status}.{run_token}.json"
            )
            try:
                os.replace(run_log, destination_log)
            except OSError:
                destination_log = run_log
            failure_payload = {
                "application": "Deinterlace Studio",
                "application_version": __version__,
                "completed_at_local": time.strftime("%Y-%m-%d %H:%M:%S %z"),
                "status": status,
                "message": error_message,
                "source_file": str(source),
                "requested_output": str(output),
                "method": method,
                "source_diagnosis": json_safe(diagnosis),
                "partial_validation": json_safe(partial_validation),
                "final_validation": json_safe(final_validation),
                "commands": [list(command) for command in command_history],
                "quarantine_path": str(quarantine) if quarantine else None,
            }
            try:
                destination_report.write_text(
                    json.dumps(failure_payload, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            except OSError:
                destination_report = None
            return RepairResult(
                False,
                status == "canceled",
                error_message,
                None,
                destination_log,
                destination_report,
                None,
                method,
                diagnosis,
                final_validation.diagnosis if final_validation else None,
                repeated_frames,
                dropped_frames,
                quarantine,
            )

        promoted_sidecars: list[Path] = []
        try:
            os.replace(run_log, final_log)
            promoted_sidecars.append(final_log)
            os.replace(report_temp, final_report)
            promoted_sidecars.append(final_report)
            for backup in backups.values():
                _safe_unlink(backup)
        except Exception as exc:
            failure_log = output.with_name(f"{output.stem}.Repair.failed-sidecars.{run_token}.log")
            failure_report = output.with_name(f"{output.stem}.Repair.failed-sidecars.{run_token}.json")
            if status == "success" and output.exists():
                quarantine = output.with_name(f"{output.stem}.Repair.failed-sidecars.{run_token}.mkv")
                os.replace(output, quarantine)
            try:
                if final_log in promoted_sidecars and final_log.exists():
                    os.replace(final_log, failure_log)
                elif run_log.exists():
                    os.replace(run_log, failure_log)
            except OSError:
                failure_log = final_log if final_log.exists() else (run_log if run_log.exists() else None)
            try:
                if report_temp.exists():
                    os.replace(report_temp, failure_report)
            except OSError:
                failure_report = report_temp if report_temp.exists() else None
            for original, backup in backups.items():
                if backup.exists():
                    os.replace(backup, original)
            return RepairResult(
                False,
                False,
                f"Repair sidecar promotion failed and prior artifacts were restored: {exc}",
                None,
                failure_log if failure_log and failure_log.exists() else None,
                failure_report if failure_report and failure_report.exists() else None,
                None,
                method,
                diagnosis,
                final_validation.diagnosis if final_validation else None,
                repeated_frames,
                dropped_frames,
                quarantine,
            )

        if status == "no_repair_needed":
            return RepairResult(
                True,
                False,
                "The complete decoded timeline is healthy; no repair output was needed or created.",
                None,
                final_log,
                final_report,
                None,
                "none",
                diagnosis,
                None,
                0,
                0,
            )
        return RepairResult(
            True,
            False,
            "Repair copy completed and passed final reopen/full-decode validation.",
            output,
            final_log,
            final_report,
            output_hash,
            method,
            diagnosis,
            final_validation.diagnosis if final_validation else None,
            repeated_frames,
            dropped_frames,
            None,
        )


def diagnosis_summary(diagnosis: TimelineDiagnosis) -> str:
    rate = _format_rate(diagnosis.nominal_rate) if diagnosis.nominal_rate else "unknown"
    lines = [
        f"Classification: {diagnosis.classification}",
        f"Decoded frames: {diagnosis.decoded_frames} at {rate} fps",
        f"Timestamped frames: {diagnosis.timestamped_frames}",
        f"Decoded CFR span: {diagnosis.decoded_cfr_span_seconds if diagnosis.decoded_cfr_span_seconds is not None else 'unknown'} s",
        f"Decoded PTS timeline: {diagnosis.pts_duration_seconds if diagnosis.pts_duration_seconds is not None else 'unknown'} s",
        f"Reported video duration: {diagnosis.reported_video_duration if diagnosis.reported_video_duration is not None else 'unknown'} s",
        f"Material gaps: {len(diagnosis.material_gaps)}; non-monotonic steps: {diagnosis.non_monotonic_steps}",
        f"Decoder/container warnings: {diagnosis.decoder_warning_count} ({diagnosis.severe_warning_count} severe)",
        f"Automatic action: {diagnosis.recommended_method}",
    ]
    if diagnosis.largest_gap:
        gap = diagnosis.largest_gap
        lines.append(
            f"Largest gap: {gap.previous_pts:.3f} -> {gap.current_pts:.3f} s "
            f"({gap.duration_seconds:.3f} s; about {gap.estimated_missing_frames} unavailable frame slots)"
        )
    return "\n".join(lines)
