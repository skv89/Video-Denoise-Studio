from __future__ import annotations

import os
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Protocol

from .models import OutputExpectation, StreamInfo, ValidationResult
from .probe import ProbeError, count_video_packets, probe_frame_samples, probe_media
from .rationals import derive_dar, fractions_close


CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0
UNSPECIFIED_FRAME_VALUES = {None, "", "unknown", "reserved", "unspecified"}


class OutputValidationSettings(Protocol):
    family: str


def _acceptable_pixel_formats(expected: str) -> set[str]:
    equivalents = {
        "p010le": {"p010le", "yuv420p10le"},
        "p012le": {"p012le", "yuv420p12le"},
        "yuv444p10le": {"yuv444p10le", "yuv444p12le"},
    }
    return equivalents.get(expected, {expected})


def _decoded_frame_consensus(frames: list[dict[str, object]], key: str) -> str | None:
    """Return one value only when every bounded decoded sample proves it."""

    if not frames:
        return None
    values: list[str] = []
    for frame in frames:
        value = frame.get(key)
        if value in UNSPECIFIED_FRAME_VALUES:
            return None
        values.append(str(value))
    return values[0] if len(set(values)) == 1 else None


def _compare_streams(kind: str, expected: tuple[StreamInfo, ...], actual: tuple[StreamInfo, ...], errors: list[str]) -> None:
    if len(expected) != len(actual):
        errors.append(f"{kind.title()} stream count changed: expected {len(expected)}, found {len(actual)}.")
        return
    for position, (before, after) in enumerate(zip(expected, actual), start=1):
        if before.codec_name != after.codec_name:
            errors.append(
                f"{kind.title()} stream {position} codec changed from {before.codec_name} to {after.codec_name}; direct copy was required."
            )
        for key in ("language", "title", "filename", "mimetype"):
            if key in before.tags and before.tags.get(key) != after.tags.get(key):
                errors.append(
                    f"{kind.title()} stream {position} metadata '{key}' changed from "
                    f"{before.tags.get(key)!r} to {after.tags.get(key)!r}."
                )
        for key, value in before.disposition.items():
            if value and after.disposition.get(key) != value:
                errors.append(f"{kind.title()} stream {position} lost disposition '{key}'.")


def _count_intra_packet_flags(ffprobe: Path, path: Path, timeout: float = 300.0) -> tuple[int, int] | None:
    command = [
        str(ffprobe),
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "packet=flags",
        "-of",
        "csv=p=0",
        str(path),
    ]
    env = os.environ.copy()
    env["AV_LOG_FORCE_NOCOLOR"] = "1"
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=CREATE_NO_WINDOW,
            env=env,
        )
        with process:
            assert process.stdout is not None
            total = key = 0
            for line in process.stdout:
                flags = line.strip()
                if not flags:
                    continue
                total += 1
                if "K" in flags:
                    key += 1
            try:
                return_code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
                return None
            if return_code != 0:
                return None
            return total, key
    except OSError:
        return None


def validate_output(
    ffprobe: Path,
    output_path: Path,
    expected: OutputExpectation,
    settings: OutputValidationSettings,
    *,
    thorough_packet_count: bool = True,
) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    if not output_path.is_file() or output_path.stat().st_size <= 0:
        return ValidationResult(False, ("Output file is missing or empty.",), (), None)

    try:
        output = probe_media(ffprobe, output_path, sample_frames=32, timeout=180)
    except (ProbeError, OSError) as exc:
        return ValidationResult(False, (f"FFprobe could not reopen the output: {exc}",), (), None)

    video = output.video
    if video.codec_name not in expected.codec_names:
        errors.append(f"Video codec is {video.codec_name}, expected one of {', '.join(expected.codec_names)}.")
    if (video.width, video.height) != (expected.width, expected.height):
        errors.append(
            f"Stored raster is {video.width}x{video.height}, expected {expected.width}x{expected.height}."
        )

    try:
        frame_evidence = probe_frame_samples(ffprobe, output_path, output.duration, samples=5, frames_per_sample=8)
    except ProbeError as exc:
        frame_evidence = []
        errors.append(f"Could not sample decoded output frame properties: {exc}")
    checked = len(frame_evidence)

    acceptable_formats = {
        candidate
        for pix_fmt in expected.pix_fmts
        for candidate in _acceptable_pixel_formats(pix_fmt)
    }
    if video.pix_fmt not in UNSPECIFIED_FRAME_VALUES:
        if video.pix_fmt not in acceptable_formats:
            errors.append(
                f"Pixel format is {video.pix_fmt}, expected {', '.join(expected.pix_fmts)} or an equivalent decoded form."
            )
    else:
        decoded_pix_fmt = _decoded_frame_consensus(frame_evidence, "pix_fmt")
        if decoded_pix_fmt in acceptable_formats:
            warnings.append(
                "FFprobe omitted the stream-level pixel format; "
                f"all {checked} bounded decoded frame samples independently reported {decoded_pix_fmt}."
            )
        else:
            errors.append(
                "Pixel format is unspecified at stream level and bounded decoded frame samples did not "
                f"consistently prove {', '.join(expected.pix_fmts)} or an equivalent decoded form"
                + (f" (sample consensus: {decoded_pix_fmt})." if decoded_pix_fmt else ".")
            )
    actual_depth = video.bits_per_raw_sample or 0
    if actual_depth and actual_depth < expected.bit_depth:
        errors.append(f"Decoded bit depth is {actual_depth}, below the planned {expected.bit_depth}-bit pipeline.")

    for label, wanted, actual, frame_key in (
        ("range", expected.color_range, video.color_range, "color_range"),
        ("matrix", expected.color_space, video.color_space, "color_space"),
        ("transfer", expected.color_transfer, video.color_transfer, "color_transfer"),
        ("primaries", expected.color_primaries, video.color_primaries, "color_primaries"),
    ):
        if not wanted or wanted in {"unknown", "reserved", "unspecified"}:
            continue
        if actual not in UNSPECIFIED_FRAME_VALUES:
            if actual != wanted:
                errors.append(f"Output color {label} is {actual}, expected preserved value {wanted}.")
            continue
        decoded_value = _decoded_frame_consensus(frame_evidence, frame_key)
        if decoded_value == wanted:
            warnings.append(
                f"FFprobe omitted stream-level color {label}; all {checked} bounded decoded frame samples "
                f"independently reported {wanted}."
            )
        else:
            errors.append(
                f"Output color {label} is unspecified at stream level and bounded decoded frame samples did not "
                f"consistently prove preserved value {wanted}"
                + (f" (sample consensus: {decoded_value})." if decoded_value else ".")
            )

    actual_sar = video.sample_aspect_ratio or Fraction(1, 1)
    actual_dar = video.display_aspect_ratio
    if actual_dar is None and video.width and video.height:
        actual_dar = derive_dar(video.width, video.height, actual_sar)
    if actual_sar != expected.sar:
        errors.append(f"Output SAR is {actual_sar}, expected {expected.sar}.")
    if actual_dar != expected.dar:
        errors.append(f"Output DAR is {actual_dar}, expected {expected.dar}.")

    actual_rate = video.avg_frame_rate or video.r_frame_rate
    if expected.frame_rate and not fractions_close(actual_rate, expected.frame_rate, tolerance=2e-4):
        errors.append(f"Output frame rate is {actual_rate}, expected {expected.frame_rate}.")

    interlaced = sum(1 for frame in frame_evidence if int(frame.get("interlaced_frame", 0) or 0) != 0)
    progressive = checked - interlaced
    if expected.progressive and interlaced:
        errors.append(f"{interlaced}/{checked} sampled output frames remain flagged interlaced.")
    if expected.progressive and video.field_order and video.field_order not in {"progressive", "unknown"}:
        errors.append(f"Output stream field order is {video.field_order}, expected progressive.")

    _compare_streams("audio", expected.expected_audio, output.streams_of_type("audio"), errors)
    _compare_streams("subtitle", expected.expected_subtitles, output.streams_of_type("subtitle"), errors)
    _compare_streams("attachment", expected.expected_attachments, output.streams_of_type("attachment"), errors)
    _compare_streams("data", expected.expected_data, output.streams_of_type("data"), errors)
    if len(output.chapters) != expected.expected_chapter_count:
        errors.append(
            f"Chapter count changed: expected {expected.expected_chapter_count}, found {len(output.chapters)}."
        )
    for key, value in expected.expected_format_tags.items():
        if output.format_tags.get(key) != value:
            errors.append(
                f"Container metadata '{key}' changed from {value!r} to {output.format_tags.get(key)!r}."
            )

    if expected.duration is not None and output.duration is not None:
        frame_tolerance = 2.0 / float(expected.frame_rate) if expected.frame_rate else 0.1
        tolerance = max(0.5, frame_tolerance)
        if abs(output.duration - expected.duration) > tolerance:
            errors.append(
                f"Output duration {output.duration:.6f}s differs from expected {expected.duration:.6f}s by more than {tolerance:.3f}s."
            )

    packet_count = None
    key_packet_count = None
    thorough_completed = False
    if thorough_packet_count:
        if expected.lossless and settings.family == "ffv1":
            intra_counts = _count_intra_packet_flags(ffprobe, output_path, timeout=600)
            if intra_counts is None:
                errors.append("Could not verify the FFV1 packet count and all-intra flags in one scan.")
            else:
                packet_count, key_packet_count = intra_counts
                thorough_completed = True
                if packet_count == 0 or key_packet_count != packet_count:
                    errors.append(
                        f"FFV1 intra verification found {key_packet_count} key packets out of "
                        f"{packet_count} video packets."
                    )
        else:
            try:
                packet_count = count_video_packets(ffprobe, output_path, timeout=600)
                thorough_completed = packet_count is not None
            except ProbeError as exc:
                warnings.append(f"Could not count output video packets: {exc}")
    if expected.frame_count is not None and packet_count is not None and packet_count != expected.frame_count:
        errors.append(f"Output contains {packet_count} video packets/frames, expected exactly {expected.frame_count}.")

    return ValidationResult(
        valid=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
        output_probe=output,
        checked_frame_count=checked,
        checked_progressive_frames=progressive,
        checked_interlaced_frames=interlaced,
        verified_packet_count=packet_count,
        verified_key_packet_count=key_packet_count,
        thorough_packet_scan_completed=thorough_completed,
    )
