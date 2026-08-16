from __future__ import annotations

from fractions import Fraction

from deinterlace_studio.models import MediaProbe


def source_fps(media: MediaProbe) -> float:
    rate: Fraction | None = media.video.avg_frame_rate or media.video.r_frame_rate
    if rate and rate.numerator > 0 and rate.denominator > 0:
        return float(rate)
    return 24.0


def _duration_tag_seconds(value: str | None) -> float | None:
    if not value:
        return None
    rendered = value.strip()
    try:
        if ":" not in rendered:
            seconds = float(rendered)
        else:
            parts = rendered.split(":")
            if len(parts) != 3:
                return None
            hours, minutes, seconds_part = parts
            seconds = int(hours) * 3600 + int(minutes) * 60 + float(seconds_part)
    except (TypeError, ValueError):
        return None
    return seconds if seconds > 0 else None


def source_video_duration(media: MediaProbe) -> float | None:
    """Prefer the video stream span over a longer audio/container span.

    Matroska commonly exposes per-stream duration only as a DURATION tag. Using
    the format duration can otherwise invent a final video frame when copied
    audio extends a few milliseconds beyond the video.
    """

    if media.video.duration is not None and media.video.duration > 0:
        return media.video.duration
    tags = {key.casefold(): value for key, value in media.video.tags.items()}
    tagged = _duration_tag_seconds(tags.get("duration"))
    if tagged is not None:
        return tagged
    return media.duration if media.duration is not None and media.duration > 0 else None


def source_frame_count(media: MediaProbe) -> int | None:
    if media.video.nb_frames is not None:
        return max(0, media.video.nb_frames)
    duration = source_video_duration(media)
    if duration is None:
        return None
    return max(1, round(duration * source_fps(media)))


def frame_from_timeline_position(pointer_x: float, widget_width: int, total_frames: int) -> int:
    """Map a timeline pointer coordinate to an absolute, clamped frame index."""

    if total_frames <= 1:
        return 0
    usable_width = max(1, int(widget_width) - 1)
    fraction = max(0.0, min(1.0, float(pointer_x) / usable_width))
    return max(0, min(total_frames - 1, round(fraction * (total_frames - 1))))


def timeline_render_delay_ms(frame_preview_enabled: bool, immediate: bool) -> int:
    """Return intentional UI debounce; source-only seeks have none."""

    if not frame_preview_enabled:
        return 0
    return 20 if immediate else 400
