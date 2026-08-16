from __future__ import annotations

import json
import os
import re
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any

from .models import MediaProbe, StreamInfo
from .rationals import derive_dar, parse_fraction


CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0


class ProbeError(RuntimeError):
    pass


def _optional_int(value: object) -> int | None:
    if value in (None, "", "N/A"):
        return None
    try:
        return int(str(value))
    except ValueError:
        return None


def _optional_float(value: object) -> float | None:
    if value in (None, "", "N/A"):
        return None
    try:
        return float(str(value))
    except ValueError:
        return None


def pixel_format_depth(pix_fmt: str | None) -> int | None:
    if not pix_fmt:
        return None
    if pix_fmt in {"yuv420p", "yuv422p", "yuv444p", "yuvj420p", "yuvj422p", "yuvj444p", "nv12"}:
        return 8
    if pix_fmt in {"rgb24", "bgr24", "rgba", "bgra"}:
        return 8
    match = re.search(r"(?:p|gray|rgb|gbrp|p0)(9|10|12|14|16)(?:le|be|msb|$)", pix_fmt)
    if match:
        return int(match.group(1))
    return None


def _parse_stream(raw: dict[str, Any]) -> StreamInfo:
    width = _optional_int(raw.get("width"))
    height = _optional_int(raw.get("height"))
    sar = parse_fraction(raw.get("sample_aspect_ratio"))
    dar = parse_fraction(raw.get("display_aspect_ratio"))
    if dar is None and width and height:
        dar = derive_dar(width, height, sar)
    bits = _optional_int(raw.get("bits_per_raw_sample")) or pixel_format_depth(raw.get("pix_fmt"))
    return StreamInfo(
        index=int(raw.get("index", 0)),
        codec_type=str(raw.get("codec_type", "unknown")),
        codec_name=str(raw.get("codec_name", "unknown")),
        profile=str(raw["profile"]) if raw.get("profile") is not None else None,
        width=width,
        height=height,
        pix_fmt=raw.get("pix_fmt"),
        bits_per_raw_sample=bits,
        sample_aspect_ratio=sar,
        display_aspect_ratio=dar,
        r_frame_rate=parse_fraction(raw.get("r_frame_rate")),
        avg_frame_rate=parse_fraction(raw.get("avg_frame_rate")),
        time_base=parse_fraction(raw.get("time_base")),
        field_order=raw.get("field_order"),
        start_time=_optional_float(raw.get("start_time")),
        duration=_optional_float(raw.get("duration")),
        nb_frames=_optional_int(raw.get("nb_frames")),
        color_range=raw.get("color_range"),
        color_space=raw.get("color_space"),
        color_transfer=raw.get("color_transfer"),
        color_primaries=raw.get("color_primaries"),
        tags={str(key): str(value) for key, value in raw.get("tags", {}).items()},
        disposition={str(key): int(value) for key, value in raw.get("disposition", {}).items()},
    )


def parse_probe_json(path: Path, payload: dict[str, Any], frame_flags: list[dict[str, Any]] | None = None) -> MediaProbe:
    raw_format = payload.get("format") or {}
    streams = tuple(_parse_stream(raw) for raw in payload.get("streams", []))
    if not any(stream.codec_type == "video" for stream in streams):
        raise ProbeError(f"No video stream found in {path}")

    interlaced = progressive = tff = bff = 0
    for frame in frame_flags or []:
        is_interlaced = _optional_int(frame.get("interlaced_frame")) == 1
        is_tff = _optional_int(frame.get("top_field_first")) == 1
        if is_interlaced:
            interlaced += 1
            if is_tff:
                tff += 1
            else:
                bff += 1
        else:
            progressive += 1

    return MediaProbe(
        path=path,
        format_name=str(raw_format.get("format_name", "unknown")),
        format_long_name=raw_format.get("format_long_name"),
        duration=_optional_float(raw_format.get("duration")),
        size=_optional_int(raw_format.get("size")),
        bit_rate=_optional_int(raw_format.get("bit_rate")),
        start_time=_optional_float(raw_format.get("start_time")),
        streams=streams,
        chapters=tuple(payload.get("chapters", [])),
        format_tags={str(key): str(value) for key, value in raw_format.get("tags", {}).items()},
        sampled_interlaced_frames=interlaced,
        sampled_progressive_frames=progressive,
        sampled_tff_frames=tff,
        sampled_bff_frames=bff,
    )


def _run_json(args: list[str], timeout: float) -> dict[str, Any]:
    env = os.environ.copy()
    env["AV_LOG_FORCE_NOCOLOR"] = "1"
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            creationflags=CREATE_NO_WINDOW,
            env=env,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ProbeError(f"Could not run ffprobe: {exc}") from exc
    if result.returncode != 0:
        raise ProbeError(result.stderr.strip() or f"ffprobe exited with {result.returncode}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ProbeError("ffprobe returned invalid JSON") from exc


def probe_media(ffprobe: Path, path: Path, *, sample_frames: int = 64, timeout: float = 90.0) -> MediaProbe:
    if not path.is_file():
        raise ProbeError(f"Input file does not exist: {path}")
    main = _run_json(
        [
            str(ffprobe),
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-show_chapters",
            "-of",
            "json",
            str(path),
        ],
        timeout,
    )
    frame_flags: list[dict[str, Any]] = []
    if sample_frames > 0:
        frames = _run_json(
            [
                str(ffprobe),
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-read_intervals",
                f"%+#{sample_frames}",
                "-show_entries",
                "frame=interlaced_frame,top_field_first",
                "-of",
                "json",
                str(path),
            ],
            timeout,
        )
        frame_flags = list(frames.get("frames", []))
    return parse_probe_json(path, main, frame_flags)


def probe_frame_samples(
    ffprobe: Path,
    path: Path,
    duration: float | None,
    *,
    samples: int = 5,
    frames_per_sample: int = 8,
    timeout: float = 90.0,
) -> list[dict[str, Any]]:
    """Read bounded frame evidence from positions distributed across a file."""

    if samples <= 0 or frames_per_sample <= 0:
        raise ValueError("Frame sample counts must be positive")
    offsets = [0.0]
    if duration and duration > 1:
        offsets = [max(0.0, duration * (index + 0.5) / samples) for index in range(samples)]
    evidence: list[dict[str, Any]] = []
    for offset in offsets:
        payload = _run_json(
            [
                str(ffprobe),
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-read_intervals",
                f"{offset:.6f}%+#{frames_per_sample}",
                "-show_entries",
                (
                    "frame=interlaced_frame,top_field_first,key_frame,pict_type,pkt_duration_time,"
                    "width,height,pix_fmt,color_range,color_space,color_transfer,color_primaries"
                ),
                "-of",
                "json",
                str(path),
            ],
            timeout,
        )
        evidence.extend(payload.get("frames", []))
    return evidence


def count_video_packets(ffprobe: Path, path: Path, *, timeout: float = 300.0) -> int | None:
    payload = _run_json(
        [
            str(ffprobe),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_packets",
            "-show_entries",
            "stream=nb_read_packets",
            "-of",
            "json",
            str(path),
        ],
        timeout,
    )
    streams = payload.get("streams", [])
    if not streams:
        return None
    return _optional_int(streams[0].get("nb_read_packets"))


def count_video_frames(ffprobe: Path, path: Path, *, timeout: float = 600.0) -> int | None:
    """Count decoded video frames, which can differ from compressed packets.

    Interlaced H.264 may store each field in its own packet. ``-count_packets``
    therefore overcounts the progressive frames produced by a source-rate
    deinterlacer, while ``-count_frames`` follows the decoder's frame output.
    """

    payload = _run_json(
        [
            str(ffprobe),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "json",
            str(path),
        ],
        timeout,
    )
    streams = payload.get("streams", [])
    if not streams:
        return None
    return _optional_int(streams[0].get("nb_read_frames"))
