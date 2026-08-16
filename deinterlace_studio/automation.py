from __future__ import annotations

import math
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from .models import (
    AutomaticRecoveryAudit,
    JobSettings,
    MediaProbe,
    ProcessingPlan,
    SourceHealthReport,
)


GIB = 1024**3
REPAIR_ARTIFACT_SUFFIXES = (".Repair.log", ".Repair.json")
DEINTERLACE_ARTIFACT_SUFFIXES = (
    ".Deinterlace.log",
    ".Deinterlace.json",
    ".Deinterlace.vpy",
)
AUTOMATIC_RECOVERY_BACKEND = "vapoursynth_qtgmc"


def automatic_recovery_applies_to_backend(selected_backend: str | None) -> bool:
    """Return whether a resolved plan needs the separate-copy recovery chain."""

    return selected_backend == AUTOMATIC_RECOVERY_BACKEND


def path_key(path: Path) -> str:
    return os.path.normcase(os.path.abspath(path))


def completed_artifacts(output: Path, suffixes: tuple[str, ...]) -> tuple[Path, ...]:
    return (output,) + tuple(output.with_name(output.name + suffix) for suffix in suffixes)


def choose_available_artifact_path(
    preferred: Path,
    suffixes: tuple[str, ...],
    *,
    reserved: tuple[Path, ...] = (),
    maximum_attempts: int = 10_000,
) -> Path:
    """Choose a deterministic non-existing completed-artifact set without deleting anything."""

    reserved_keys = {path_key(path) for path in reserved}
    for index in range(1, maximum_attempts + 1):
        candidate = (
            preferred
            if index == 1
            else preferred.with_name(f"{preferred.stem}.{index}{preferred.suffix}")
        )
        artifacts = completed_artifacts(candidate, suffixes)
        if any(path_key(path) in reserved_keys for path in artifacts):
            continue
        if any(path.exists() for path in artifacts):
            continue
        return candidate
    raise RuntimeError(
        f"Could not find an unused artifact name after {maximum_attempts:,} attempts near {preferred}."
    )


def _pixel_format_layout(pixel_format: str | None, fallback_depth: int | None) -> tuple[float, int]:
    value = (pixel_format or "").casefold()
    if "yuva420" in value:
        samples = 2.5
    elif "yuva422" in value:
        samples = 3.0
    elif "yuva444" in value or value.startswith(("rgba", "bgra", "argb", "abgr")):
        samples = 4.0
    elif "420" in value:
        samples = 1.5
    elif "422" in value or value.startswith(("yuyv", "uyvy")):
        samples = 2.0
    elif "444" in value or value.startswith(("gbrp", "rgb24", "bgr24")):
        samples = 3.0
    elif value.startswith("gray"):
        samples = 1.0
    else:
        # Unknown decoded layouts use a deliberately conservative four-component estimate.
        samples = 4.0
    match = re.search(r"(?:p|gray|gbrp)(9|10|12|14|16)(?:le|be)?$", value)
    depth = int(match.group(1)) if match else int(fallback_depth or 8)
    return samples, 2 if depth > 8 else 1


def _raw_video_bytes(
    *,
    width: int | None,
    height: int | None,
    pixel_format: str | None,
    bit_depth: int | None,
    frame_count: int | None,
) -> int | None:
    if not width or not height or not frame_count or frame_count <= 0:
        return None
    samples, bytes_per_sample = _pixel_format_layout(pixel_format, bit_depth)
    return math.ceil(width * height * samples * bytes_per_sample * frame_count)


def estimate_repair_bytes(media: MediaProbe) -> int:
    rate = media.video.avg_frame_rate or media.video.r_frame_rate
    frame_count = media.video.nb_frames
    if frame_count is None and rate and rate > 0 and media.duration and media.duration > 0:
        frame_count = math.ceil(float(rate) * media.duration)
    raw = _raw_video_bytes(
        width=media.video.width,
        height=media.video.height,
        pixel_format=media.video.pix_fmt,
        bit_depth=media.video.bits_per_raw_sample,
        frame_count=frame_count,
    )
    baseline = raw if raw is not None else max(media.path.stat().st_size * 4, 10 * GIB)
    return math.ceil(baseline * 1.20)


def estimate_output_bytes(plan: ProcessingPlan, media: MediaProbe) -> int:
    expected = plan.expected
    if expected is None:
        return max(media.path.stat().st_size * 4, 10 * GIB)
    frame_count = expected.frame_count
    if frame_count is None and expected.frame_rate and expected.duration and expected.duration > 0:
        frame_count = math.ceil(float(expected.frame_rate) * expected.duration)
    raw = _raw_video_bytes(
        width=expected.width,
        height=expected.height,
        pixel_format=expected.pix_fmts[0] if expected.pix_fmts else None,
        bit_depth=expected.bit_depth,
        frame_count=frame_count,
    )
    baseline = raw if raw is not None else max(media.path.stat().st_size * 4, 10 * GIB)
    return math.ceil(baseline * 1.20)


@dataclass(frozen=True)
class VolumeStorageCheck:
    volume: str
    probe_path: Path
    required_bytes: int
    free_bytes: int

    @property
    def sufficient(self) -> bool:
        return self.free_bytes >= self.required_bytes

    @property
    def summary(self) -> str:
        state = "PASS" if self.sufficient else "INSUFFICIENT"
        return (
            f"{state} {self.volume}: requires conservative {self.required_bytes / GIB:.1f} GiB; "
            f"{self.free_bytes / GIB:.1f} GiB free"
        )


def storage_preflight(
    media: MediaProbe,
    plan: ProcessingPlan,
    repair_output: Path,
    final_output: Path,
) -> tuple[VolumeStorageCheck, ...]:
    """Conservatively reserve decoded-raw upper bounds for the unattended chain."""

    repair_parent = repair_output.parent
    final_parent = final_output.parent
    for parent in (repair_parent, final_parent):
        if not parent.is_dir():
            raise FileNotFoundError(f"Automatic output directory does not exist: {parent}")
    requirements = (
        (repair_parent, estimate_repair_bytes(media)),
        (final_parent, estimate_output_bytes(plan, media)),
    )
    grouped: dict[str, tuple[Path, int]] = {}
    for parent, required in requirements:
        volume = Path(os.path.abspath(parent)).anchor.casefold() or path_key(parent)
        probe_path, accumulated = grouped.get(volume, (parent, 0))
        grouped[volume] = probe_path, accumulated + required
    checks: list[VolumeStorageCheck] = []
    for volume, (probe_path, required) in grouped.items():
        free = shutil.disk_usage(probe_path).free
        checks.append(
            VolumeStorageCheck(
                volume=volume.upper(),
                probe_path=probe_path,
                required_bytes=required + 2 * GIB,
                free_bytes=free,
            )
        )
    return tuple(sorted(checks, key=lambda item: item.volume))


def storage_summary(checks: tuple[VolumeStorageCheck, ...]) -> str:
    return "; ".join(check.summary for check in checks)


@dataclass
class AutomaticRecoveryWorkflow:
    original_source: Path
    trigger_health: SourceHealthReport
    requested_settings: JobSettings
    final_settings: JobSettings
    repair_output: Path
    analysis_mode: str
    storage_preflight_summary: str
    stage: str = "repairing"
    validated_repair_source: Path | None = None
    repair_method: str | None = None
    repair_output_sha256: str | None = None
    repair_log_path: Path | None = None
    repair_report_path: Path | None = None
    repeated_frames: int = 0
    dropped_frames: int = 0

    def audit(self) -> AutomaticRecoveryAudit | None:
        if self.repair_method is None:
            return None
        return AutomaticRecoveryAudit(
            original_source=self.original_source,
            trigger_health=self.trigger_health,
            requested_output=self.requested_settings.output_path,
            selected_output=self.final_settings.output_path,
            repair_output=self.validated_repair_source,
            repair_method=self.repair_method,
            repair_output_sha256=self.repair_output_sha256,
            repair_log_path=self.repair_log_path,
            repair_report_path=self.repair_report_path,
            repeated_frames=self.repeated_frames,
            dropped_frames=self.dropped_frames,
            storage_preflight=self.storage_preflight_summary,
        )
