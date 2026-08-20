from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from video_processing_core.media.models import MediaProbe, ValidationResult
from video_processing_core.media.validation import validate_output

from .models import DenoisePlan
from .planner import source_field_order, source_is_interlaced


def validate_denoise_output(
    ffprobe: Path,
    output_path: Path,
    plan: DenoisePlan,
    *,
    thorough_packet_count: bool,
) -> ValidationResult:
    """Validate the shared output contract plus denoise-only field preservation."""

    if plan.expected is None or plan.media is None:
        return ValidationResult(False, ("The denoise plan has no output expectation.",), (), None)
    base = validate_output(
        ffprobe,
        output_path,
        plan.expected,
        plan.settings,
        thorough_packet_count=thorough_packet_count,
    )
    errors = list(base.errors)
    warnings = list(base.warnings)
    output = base.output_probe
    source = plan.media
    if output is not None and source_is_interlaced(source):
        if base.checked_frame_count <= 0:
            errors.append("No decoded output frames were available to verify preserved interlacing.")
        elif base.checked_interlaced_frames != base.checked_frame_count:
            errors.append(
                f"Only {base.checked_interlaced_frames}/{base.checked_frame_count} sampled output frames remain "
                "flagged interlaced; denoise-only processing must preserve every stored frame's field state."
            )
        wanted = source_field_order(source)
        actual = source_field_order(output)
        if wanted and actual != wanted:
            sampled_match = (
                wanted == "tff"
                and output.sampled_tff_frames > 0
                and output.sampled_bff_frames == 0
            ) or (
                wanted == "bff"
                and output.sampled_bff_frames > 0
                and output.sampled_tff_frames == 0
            )
            if sampled_match:
                warnings.append(
                    "The output stream omitted a conclusive field-order label, but every bounded decoded "
                    f"interlaced sample independently preserved {wanted.upper()}."
                )
            else:
                errors.append(
                    f"Output field order is {actual or output.video.field_order or 'unknown'}, expected preserved "
                    f"{wanted.upper()}."
                )
    return replace(base, valid=not errors, errors=tuple(errors), warnings=tuple(dict.fromkeys(warnings)))
