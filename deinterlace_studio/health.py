from __future__ import annotations

import heapq
import math
import os
import re
import statistics
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, Iterable

from .models import MediaProbe, PacketTimelineGap, SourceHealthReport


CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0
CLOCK_DURATION = re.compile(r"^(\d+):(\d{2}):(\d{2}(?:\.\d+)?)$")
STRUCTURAL_WARNING_MARKERS = (
    "corrupt",
    "damaged",
    "decode error",
    "error while",
    "exceeds containing",
    "failed",
    "illegal",
    "invalid",
    "malformed",
    "missing",
    "non monoton",
    "non-monoton",
    "overread",
    "reference picture",
    "truncated",
)
HealthProgress = Callable[[int, float | None], None]


class SourceHealthCancelled(RuntimeError):
    pass


class SourceHealthError(RuntimeError):
    pass


def source_identity(path: Path) -> tuple[str, int, int]:
    stat = path.stat()
    return os.path.normcase(os.path.abspath(path)), stat.st_size, stat.st_mtime_ns


def health_matches_source(report: SourceHealthReport, path: Path) -> bool:
    try:
        normalized, size, mtime_ns = source_identity(path)
    except OSError:
        return False
    return (
        normalized == os.path.normcase(os.path.abspath(report.path))
        and size == report.source_size
        and mtime_ns == report.source_mtime_ns
    )


def video_timeline_duration(media: MediaProbe) -> float | None:
    if media.video.duration is not None:
        return media.video.duration
    for key in ("DURATION", "DURATION-eng"):
        value = media.video.tags.get(key)
        match = CLOCK_DURATION.fullmatch(value.strip()) if value else None
        if match:
            hours, minutes, seconds = match.groups()
            return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    return media.duration


def _has_video_specific_duration(media: MediaProbe) -> bool:
    if media.video.duration is not None:
        return True
    return any(
        bool(value and CLOCK_DURATION.fullmatch(value.strip()))
        for key in ("DURATION", "DURATION-eng")
        if (value := media.video.tags.get(key)) is not None
    )


def _is_structural_warning(line: str) -> bool:
    lowered = line.casefold()
    return any(marker in lowered for marker in STRUCTURAL_WARNING_MARKERS)


def _parse_compact_packet(line: str) -> tuple[float | None, int | None]:
    fields: dict[str, str] = {}
    for item in line.rstrip("\r\n").split("|"):
        if "=" in item:
            key, value = item.split("=", 1)
            fields[key] = value
    pts: float | None = None
    raw_pts = fields.get("pts_time")
    if raw_pts not in (None, "", "N/A"):
        try:
            candidate = float(raw_pts)
            if math.isfinite(candidate):
                pts = candidate
        except ValueError:
            pass
    position: int | None = None
    raw_position = fields.get("pos")
    if raw_position not in (None, "", "N/A"):
        try:
            candidate_position = int(raw_position)
            if candidate_position >= 0:
                position = candidate_position
        except ValueError:
            pass
    return pts, position


def assess_packet_timeline(
    media: MediaProbe,
    *,
    source_size: int,
    source_mtime_ns: int,
    packet_count: int,
    pts_values: Iterable[float],
    elapsed_seconds: float,
    warning_samples: Iterable[str] = (),
    demux_warning_count: int = 0,
    structural_warning_count: int = 0,
    ffprobe_returncode: int | None = 0,
    scan_error: str | None = None,
) -> SourceHealthReport:
    """Classify a complete compressed-packet timeline without counting packets as frames."""

    rate = media.video.avg_frame_rate or media.video.r_frame_rate
    nominal_step = 1.0 / float(rate) if rate and rate > 0 else None
    gap_threshold = max(0.5, min(5.0, (nominal_step or 0.0625) * 8.0))
    ordered_pts = sorted(value for value in pts_values if math.isfinite(value))
    timestamped_count = len(ordered_pts)
    first_pts = ordered_pts[0] if ordered_pts else None
    last_pts = ordered_pts[-1] if ordered_pts else None
    previous: float | None = None
    unique_count = 0
    normal_step_sample: list[float] = []
    largest_gap_heap: list[tuple[float, float, float]] = []
    material_gap_count = 0

    for current in ordered_pts:
        if previous is None:
            previous = current
            unique_count = 1
            continue
        step = current - previous
        if step <= 1e-9:
            continue
        unique_count += 1
        if step >= gap_threshold - 1e-9:
            material_gap_count += 1
            item = (step, previous, current)
            if len(largest_gap_heap) < 8:
                heapq.heappush(largest_gap_heap, item)
            elif step > largest_gap_heap[0][0]:
                heapq.heapreplace(largest_gap_heap, item)
        elif len(normal_step_sample) < 50_000:
            normal_step_sample.append(step)
        previous = current

    typical_step = statistics.median(normal_step_sample) if normal_step_sample else nominal_step
    packet_span = None
    if first_pts is not None and last_pts is not None:
        packet_span = max(0.0, last_pts - first_pts + (typical_step or nominal_step or 0.0))
    reported_duration = video_timeline_duration(media)
    duration_difference = None
    duration_tolerance = max(1.0, 4.0 * (typical_step or nominal_step or 0.04))
    if packet_span is not None and reported_duration is not None and reported_duration > 0:
        duration_difference = reported_duration - packet_span
    duration_is_video_specific = _has_video_specific_duration(media)
    duration_inconsistent = bool(
        ffprobe_returncode == 0
        and duration_is_video_specific
        and duration_difference is not None
        and abs(duration_difference) > duration_tolerance
    )
    format_duration_warning = bool(
        ffprobe_returncode == 0
        and not duration_is_video_specific
        and duration_difference is not None
        and abs(duration_difference) > duration_tolerance
    )
    largest_gaps = tuple(
        PacketTimelineGap(before_pts=before, after_pts=after, duration=gap)
        for gap, before, after in sorted(largest_gap_heap, reverse=True)
    )
    timestamp_fraction = timestamped_count / packet_count if packet_count else 0.0

    if material_gap_count:
        largest = largest_gaps[0]
        status = "repair_required"
        reason = (
            f"Fast packet scan found a {largest.duration:.3f}-second video timestamp hole at "
            f"{largest.before_pts:.3f}→{largest.after_pts:.3f} seconds."
        )
    elif duration_inconsistent:
        status = "repair_required"
        reason = (
            "Fast packet scan found that the video packet span and reported timeline differ by "
            f"{duration_difference:+.3f} seconds."
        )
    elif scan_error or ffprobe_returncode not in (0, None) or packet_count == 0 or unique_count < 2:
        status = "inconclusive"
        detail = scan_error or (
            f"FFprobe exited with code {ffprobe_returncode}"
            if ffprobe_returncode not in (0, None)
            else "too few timestamped video packets were available"
        )
        reason = f"Fast full-file source-health scan was inconclusive: {detail}."
    elif timestamp_fraction < 0.98:
        status = "warning"
        missing = packet_count - timestamped_count
        reason = f"Fast packet scan found {missing:,} video packets without usable presentation timestamps."
    elif structural_warning_count:
        status = "warning"
        reason = (
            f"Fast packet scan found no material timestamp hole, but FFprobe reported "
            f"{structural_warning_count:,} structural/demux warning(s)."
        )
    elif format_duration_warning:
        status = "warning"
        reason = (
            "The container-level duration and compressed video packet span differ by "
            f"{duration_difference:+.3f} seconds, but no video-stream-specific duration was available; "
            "the managed full decoded preflight will decide whether QTGMC can preserve the timeline."
        )
    else:
        status = "clear"
        reason = "Fast full-file packet/timestamp scan found no material gap or structural warning."

    return SourceHealthReport(
        path=media.path,
        source_size=source_size,
        source_mtime_ns=source_mtime_ns,
        status=status,
        reason=reason,
        elapsed_seconds=max(0.0, elapsed_seconds),
        packet_count=packet_count,
        timestamped_packet_count=timestamped_count,
        unique_timestamp_count=unique_count,
        first_pts=first_pts,
        last_pts=last_pts,
        typical_step_seconds=typical_step,
        packet_timeline_span_seconds=packet_span,
        reported_duration_seconds=reported_duration,
        duration_difference_seconds=duration_difference,
        gap_threshold_seconds=gap_threshold,
        material_gap_count=material_gap_count,
        largest_gaps=largest_gaps,
        demux_warning_count=demux_warning_count,
        structural_warning_count=structural_warning_count,
        warning_samples=tuple(dict.fromkeys(line.strip() for line in warning_samples if line.strip()))[:20],
        ffprobe_returncode=ffprobe_returncode,
        scan_error=scan_error,
    )


def health_headline(report: SourceHealthReport) -> str:
    elapsed = f"{report.elapsed_seconds:.1f}s"
    if report.status == "repair_required":
        if report.largest_gaps:
            gap = report.largest_gaps[0]
            return (
                "Source health: DAMAGE LIKELY — REPAIR NEEDED · "
                f"{gap.duration:.3f}s timestamp hole at {gap.before_pts:.3f}→{gap.after_pts:.3f}s · "
                f"fast full-file scan {elapsed}"
            )
        return f"Source health: SOURCE REPAIR NEEDED · {report.reason} · fast scan {elapsed}"
    if report.status == "warning":
        return f"Source health: WARNING · {report.reason} · scan {elapsed}"
    if report.status == "inconclusive":
        return f"Source health: INCONCLUSIVE · {report.reason}"
    return (
        "Source health: no obvious damage found by fast full-file scan "
        f"({report.packet_count:,} video packets in {elapsed}); QTGMC next establishes an exact frame contract."
    )


def health_details(report: SourceHealthReport) -> str:
    label = {
        "repair_required": "DAMAGE LIKELY — REPAIR NEEDED",
        "warning": "WARNING",
        "inconclusive": "INCONCLUSIVE",
        "clear": "NO OBVIOUS DAMAGE FOUND",
    }.get(report.status, report.status.upper())
    lines = [
        f"SOURCE HEALTH — {label}",
        (
            f"Fast full-file precheck: {report.packet_count:,} compressed video packets scanned in "
            f"{report.elapsed_seconds:.3f}s; {report.timestamped_packet_count:,} had usable PTS."
        ),
        report.reason,
    ]
    if report.first_pts is not None and report.last_pts is not None:
        lines.append(
            f"Packet PTS range: {report.first_pts:.3f}→{report.last_pts:.3f}s · "
            f"typical step {report.typical_step_seconds or 0.0:.6f}s · "
            f"material-gap threshold {report.gap_threshold_seconds:.3f}s."
        )
    if report.demux_warning_count:
        lines.append(
            f"FFprobe diagnostics: {report.demux_warning_count:,} warning line(s), "
            f"{report.structural_warning_count:,} classified as structural/demux warnings."
        )
        lines.extend(f"  • {line}" for line in report.warning_samples[:3])
    if report.repair_required:
        lines.append(
            "QTGMC Start is blocked for this unchanged source. With Automatic QTGMC recovery enabled, the app "
            "creates and validates a separate repair copy, re-analyzes it, then continues a valid QTGMC plan. "
            "Otherwise use Repair required… manually or choose a clean replacement."
        )
        lines.append(
            "Explicit FFmpeg BWDIF CPU/CUDA bypasses automatic repair and can process the original timestamp-aware "
            "stream directly. BWDIF is not a repair and cannot restore missing/corrupt pictures; inspect motion "
            "and audio/video sync around every reported damaged interval."
        )
    lines += [
        (
            "Scope: this low-overhead scan catches obvious timestamp/container damage without decoding every picture; "
            "a clear result is not a complete decoded-picture guarantee."
        ),
        (
            "Automatic QTGMC recovery is controlled by the main-window checkbox, applies only when the plan resolves "
            "to QTGMC, and never rewrites the selected source. Before encoding, QTGMC establishes an exact frame "
            "contract: unchanged packet-clean FFV1 sources can use a fast indexed VSPipe graph check; every other "
            "VapourSynth source uses a cancellable full decoded fallback with live progress."
        ),
    ]
    return "\n".join(lines)


def health_summary(report: SourceHealthReport) -> str:
    """Compact final-analysis text; the dedicated banner carries the headline."""

    label = {
        "repair_required": "DAMAGE LIKELY — REPAIR NEEDED",
        "warning": "WARNING",
        "inconclusive": "INCONCLUSIVE",
        "clear": "NO OBVIOUS DAMAGE FOUND",
    }.get(report.status, report.status.upper())
    if report.repair_required:
        gap = report.largest_gaps[0] if report.largest_gaps else None
        measured = (
            f"{gap.duration:.3f}s PTS hole at {gap.before_pts:.3f}→{gap.after_pts:.3f}s"
            if gap
            else report.reason
        )
        return (
            f"SOURCE HEALTH — {label}: {measured}; {report.packet_count:,} packets scanned in "
            f"{report.elapsed_seconds:.3f}s. QTGMC needs a validated repair copy; BWDIF runs the original directly "
            "but cannot restore missing pictures."
        )
    return (
        f"SOURCE HEALTH — {label}: {report.packet_count:,} packets scanned in {report.elapsed_seconds:.3f}s. "
        "Packet evidence is not a full decode guarantee; QTGMC still establishes an exact frame contract before encoding."
    )


def scan_source_health(
    ffprobe: Path,
    media: MediaProbe,
    *,
    cancel_event: threading.Event | None = None,
    progress: HealthProgress | None = None,
    timeout: float = 300.0,
) -> SourceHealthReport:
    """Scan every compressed video packet while keeping the GUI cancellable."""

    cancel_event = cancel_event or threading.Event()
    if cancel_event.is_set():
        raise SourceHealthCancelled("Source-health scan canceled")
    normalized, source_size, source_mtime_ns = source_identity(media.path)
    del normalized
    command = [
        str(ffprobe),
        "-hide_banner",
        "-v",
        "warning",
        "-select_streams",
        "v:0",
        "-show_packets",
        "-show_entries",
        "packet=pts_time,pos",
        "-of",
        "compact=p=0:nk=0",
        "--",
        str(media.path),
    ]
    environment = os.environ.copy()
    environment["AV_LOG_FORCE_NOCOLOR"] = "1"
    started = time.monotonic()
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=CREATE_NO_WINDOW,
            env=environment,
        )
    except OSError as exc:
        return assess_packet_timeline(
            media,
            source_size=source_size,
            source_mtime_ns=source_mtime_ns,
            packet_count=0,
            pts_values=(),
            elapsed_seconds=time.monotonic() - started,
            scan_error=f"could not start FFprobe: {exc}",
            ffprobe_returncode=None,
        )

    assert process.stdout is not None
    assert process.stderr is not None
    warning_samples: list[str] = []
    warning_counts = [0, 0]
    monitor_stop = threading.Event()
    timed_out = threading.Event()

    def drain_stderr() -> None:
        for raw in process.stderr:
            line = raw.rstrip()
            if not line:
                continue
            warning_counts[0] += 1
            if _is_structural_warning(line):
                warning_counts[1] += 1
            if len(warning_samples) < 20:
                warning_samples.append(line)

    def monitor() -> None:
        deadline = started + timeout
        while not monitor_stop.wait(0.05):
            if process.poll() is not None:
                return
            if cancel_event.is_set() or time.monotonic() >= deadline:
                if not cancel_event.is_set():
                    timed_out.set()
                try:
                    process.terminate()
                except OSError:
                    pass
                return

    stderr_thread = threading.Thread(target=drain_stderr, daemon=True)
    monitor_thread = threading.Thread(target=monitor, daemon=True)
    stderr_thread.start()
    monitor_thread.start()
    packet_count = 0
    pts_values: list[float] = []
    maximum_position = 0
    last_progress = started - 1.0
    returncode: int | None = None
    try:
        for line in process.stdout:
            packet_count += 1
            pts, position = _parse_compact_packet(line)
            if pts is not None:
                pts_values.append(pts)
            if position is not None:
                maximum_position = max(maximum_position, position)
            now = time.monotonic()
            if progress and now - last_progress >= 0.25:
                fraction = min(1.0, maximum_position / source_size) if source_size and maximum_position else None
                progress(packet_count, fraction)
                last_progress = now
        returncode = process.wait()
    except BaseException:
        if process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    process.kill()
                except OSError:
                    pass
        raise
    finally:
        monitor_stop.set()
        monitor_thread.join(timeout=2)
        stderr_thread.join(timeout=2)
        try:
            process.stdout.close()
            process.stderr.close()
        except OSError:
            pass

    if cancel_event.is_set():
        raise SourceHealthCancelled("Source-health scan canceled")
    elapsed = time.monotonic() - started
    if progress:
        progress(packet_count, 1.0)
    try:
        _, final_size, final_mtime_ns = source_identity(media.path)
    except OSError as exc:
        raise SourceHealthError(f"Source disappeared during its health scan: {exc}") from exc
    if final_size != source_size or final_mtime_ns != source_mtime_ns:
        raise SourceHealthError("Source changed during its health scan; analyze the current file again.")
    scan_error = None
    if timed_out.is_set():
        scan_error = f"FFprobe packet scan exceeded its {timeout:.0f}-second limit"
    elif returncode != 0:
        scan_error = f"FFprobe packet scan exited with code {returncode}"
    return assess_packet_timeline(
        media,
        source_size=source_size,
        source_mtime_ns=source_mtime_ns,
        packet_count=packet_count,
        pts_values=pts_values,
        elapsed_seconds=elapsed,
        warning_samples=warning_samples,
        demux_warning_count=warning_counts[0],
        structural_warning_count=warning_counts[1],
        ffprobe_returncode=returncode,
        scan_error=scan_error,
    )
