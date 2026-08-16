from __future__ import annotations

import os
import re
import subprocess
import threading
from pathlib import Path
from typing import Callable

from .models import IDetCounts, IDetReport, IDetSegment, MediaProbe


CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0


class AnalysisCancelled(RuntimeError):
    pass


class IDetError(RuntimeError):
    pass


_REPEATED = re.compile(r"Repeated Fields:\s*Neither:\s*(\d+)\s*Top:\s*(\d+)\s*Bottom:\s*(\d+)")
_SINGLE = re.compile(
    r"Single frame detection:\s*TFF:\s*(\d+)\s*BFF:\s*(\d+)\s*Progressive:\s*(\d+)\s*Undetermined:\s*(\d+)"
)
_MULTI = re.compile(
    r"Multi frame detection:\s*TFF:\s*(\d+)\s*BFF:\s*(\d+)\s*Progressive:\s*(\d+)\s*Undetermined:\s*(\d+)"
)


def _best_match(pattern: re.Pattern[str], text: str) -> tuple[int, ...] | None:
    candidates = [tuple(int(value) for value in match.groups()) for match in pattern.finditer(text)]
    if not candidates:
        return None
    return max(candidates, key=sum)


def parse_idet_output(text: str) -> IDetCounts:
    repeated = _best_match(_REPEATED, text)
    single = _best_match(_SINGLE, text)
    multi = _best_match(_MULTI, text)
    if not repeated or not single or not multi:
        raise IDetError("FFmpeg IDet summary was not present in the analysis output")
    return IDetCounts(
        repeated_neither=repeated[0],
        repeated_top=repeated[1],
        repeated_bottom=repeated[2],
        single_tff=single[0],
        single_bff=single[1],
        single_progressive=single[2],
        single_undetermined=single[3],
        multi_tff=multi[0],
        multi_bff=multi[1],
        multi_progressive=multi[2],
        multi_undetermined=multi[3],
    )


def classify_idet(counts: IDetCounts) -> tuple[str, str | None, float, str]:
    determined = counts.determined_total
    total = counts.multi_total
    if determined < 25:
        return "insufficient", None, 0.0, f"Only {determined} frames received a determined multi-frame classification."

    progressive_ratio = counts.multi_progressive / determined
    interlaced = counts.multi_tff + counts.multi_bff
    interlaced_ratio = interlaced / determined
    dominant_count = max(counts.multi_tff, counts.multi_bff)
    dominance = dominant_count / interlaced if interlaced else 0.0
    undetermined_ratio = counts.multi_undetermined / total if total else 0.0

    if progressive_ratio >= 0.95 and interlaced_ratio <= 0.05:
        confidence = progressive_ratio * (1.0 - min(undetermined_ratio, 0.5))
        return (
            "progressive",
            None,
            confidence,
            f"{counts.multi_progressive}/{determined} determined frames ({progressive_ratio:.1%}) are progressive; "
            f"TFF={counts.multi_tff}, BFF={counts.multi_bff}, undetermined={counts.multi_undetermined}.",
        )

    if interlaced_ratio >= 0.80 and dominance >= 0.90:
        order = "tff" if counts.multi_tff >= counts.multi_bff else "bff"
        confidence = interlaced_ratio * dominance * (1.0 - min(undetermined_ratio, 0.5))
        return (
            order,
            order,
            confidence,
            f"{interlaced}/{determined} determined frames ({interlaced_ratio:.1%}) are interlaced and "
            f"{dominance:.1%} of those favor {order.upper()}; progressive={counts.multi_progressive}, "
            f"undetermined={counts.multi_undetermined}.",
        )

    return (
        "mixed_or_ambiguous",
        None,
        max(progressive_ratio, interlaced_ratio * dominance),
        f"Mixed evidence: TFF={counts.multi_tff}, BFF={counts.multi_bff}, "
        f"progressive={counts.multi_progressive}, undetermined={counts.multi_undetermined}.",
    )


def sample_intervals(duration: float | None, count: int = 8, seconds: float = 12.0) -> list[tuple[float, float]]:
    if count <= 0 or seconds <= 0:
        raise ValueError("Sample count and duration must be positive")
    if not duration or duration <= seconds:
        return [(0.0, max(duration or seconds, 0.1))]
    usable = max(duration - seconds, 0.0)
    return [((index + 0.5) * usable / count, seconds) for index in range(count)]


def _run_segment(
    ffmpeg: Path,
    source: Path,
    offset: float,
    duration: float | None,
    cancel_event: threading.Event | None,
) -> tuple[IDetCounts, str]:
    command = [str(ffmpeg), "-hide_banner", "-nostdin", "-loglevel", "info"]
    if offset > 0:
        command += ["-ss", f"{offset:.6f}"]
    command += ["-i", str(source), "-map", "0:v:0"]
    if duration is not None:
        command += ["-t", f"{duration:.6f}"]
    command += ["-an", "-sn", "-dn", "-vf", "idet", "-f", "null", "NUL" if os.name == "nt" else "/dev/null"]
    env = os.environ.copy()
    env["AV_LOG_FORCE_NOCOLOR"] = "1"
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=CREATE_NO_WINDOW,
            env=env,
        )
    except OSError as exc:
        raise IDetError(f"Could not start FFmpeg IDet: {exc}") from exc

    lines: list[str] = []
    with process:
        assert process.stderr is not None
        while True:
            if cancel_event and cancel_event.is_set():
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                raise AnalysisCancelled("Interlace analysis canceled")
            line = process.stderr.readline()
            if line:
                lines.append(line)
                continue
            if process.poll() is not None:
                break
        return_code = process.wait()
    text = "".join(lines)
    if return_code != 0:
        raise IDetError(text[-4000:] or f"FFmpeg IDet exited with {return_code}")
    return parse_idet_output(text), subprocess.list2cmdline(command)


def scan_idet(
    ffmpeg: Path,
    media: MediaProbe,
    *,
    mode: str = "sampled",
    sample_count: int = 8,
    sample_seconds: float = 12.0,
    cancel_event: threading.Event | None = None,
    progress: Callable[[int, int, float], None] | None = None,
) -> IDetReport:
    if mode not in {"sampled", "full"}:
        raise ValueError("IDet mode must be 'sampled' or 'full'")
    intervals: list[tuple[float, float | None]]
    if mode == "full":
        intervals = [(0.0, None)]
    else:
        intervals = [(offset, length) for offset, length in sample_intervals(media.duration, sample_count, sample_seconds)]

    segments: list[IDetSegment] = []
    commands: list[str] = []
    aggregate = IDetCounts()
    for index, (offset, length) in enumerate(intervals, start=1):
        if progress:
            progress(index - 1, len(intervals), offset)
        counts, command = _run_segment(ffmpeg, media.path, offset, length, cancel_event)
        aggregate += counts
        segments.append(IDetSegment(offset=offset, duration=length, counts=counts))
        commands.append(command)
        if progress:
            progress(index, len(intervals), offset)

    classification, order, confidence, rationale = classify_idet(aggregate)
    return IDetReport(
        mode=mode,
        segments=tuple(segments),
        aggregate=aggregate,
        classification=classification,
        dominant_field_order=order,
        confidence=confidence,
        rationale=rationale,
        command_lines=tuple(commands),
    )
