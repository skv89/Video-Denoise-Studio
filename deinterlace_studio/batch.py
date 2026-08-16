from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable

from .automation import DEINTERLACE_ARTIFACT_SUFFIXES, choose_available_artifact_path
from .denoise import DENOISER_BY_ID
from .models import (
    CapabilityReport,
    IDetReport,
    JobSettings,
    MediaProbe,
    ProcessingPlan,
    SourceHealthReport,
)
from .planner import MOV_AUDIO_CODECS, MOV_SUBTITLE_CODECS, build_plan
from .presets import profile_capability_error, select_profile


MAX_BATCH_FILES = 99
SUPPORTED_VIDEO_EXTENSIONS = frozenset(
    {
        ".3gp",
        ".asf",
        ".avi",
        ".divx",
        ".flv",
        ".m2t",
        ".m2ts",
        ".m4v",
        ".mkv",
        ".mov",
        ".mp4",
        ".mpeg",
        ".mpg",
        ".mts",
        ".mxf",
        ".ogm",
        ".rm",
        ".rmvb",
        ".ts",
        ".vob",
        ".webm",
        ".wmv",
    }
)


class BatchCompatibilityError(RuntimeError):
    """A row cannot be processed safely without a user decision."""

    def __init__(self, message: str, *, needs_review: bool = False) -> None:
        super().__init__(message)
        self.needs_review = needs_review


def normalized_path_key(path: Path) -> str:
    return os.path.normcase(os.path.abspath(path))


@dataclass(frozen=True)
class BatchAddResult:
    added: tuple["BatchRecord", ...]
    duplicates: tuple[Path, ...] = ()
    unsupported: tuple[Path, ...] = ()
    missing: tuple[Path, ...] = ()
    capacity_rejected: tuple[Path, ...] = ()


@dataclass
class BatchRecord:
    source_path: Path
    identifier: str = field(default_factory=lambda: uuid.uuid4().hex)
    state: str = "Queued"
    analysis_text: str = "Pending"
    effective_text: str = "Pending"
    output_path: Path | None = None
    progress_text: str = "Waiting"
    progress_percent: float = 0.0
    fallback_notes: tuple[str, ...] = ()
    media: MediaProbe | None = None
    analysis: IDetReport | None = None
    source_health: SourceHealthReport | None = None
    source_identity: tuple[str, int, int] | None = None
    settings: JobSettings | None = None
    plan: ProcessingPlan | None = None
    result_output: Path | None = None
    error: str | None = None

    def reset_plan(self, *, retain_analysis: bool = True) -> None:
        self.state = "Queued"
        self.effective_text = "Pending"
        self.output_path = None
        self.progress_text = "Waiting"
        self.progress_percent = 0.0
        self.fallback_notes = ()
        self.settings = None
        self.plan = None
        self.result_output = None
        self.error = None
        if not retain_analysis:
            self.analysis_text = "Pending"
            self.media = None
            self.analysis = None
            self.source_health = None
            self.source_identity = None


class BatchQueue:
    """Ordered, duplicate-safe queue independent of Tk widgets."""

    def __init__(self, *, maximum: int = MAX_BATCH_FILES) -> None:
        self.maximum = maximum
        self.records: list[BatchRecord] = []

    def __len__(self) -> int:
        return len(self.records)

    def record(self, identifier: str) -> BatchRecord | None:
        return next((item for item in self.records if item.identifier == identifier), None)

    def add_paths(
        self,
        paths: Iterable[Path],
        *,
        include_subfolders: bool = False,
    ) -> BatchAddResult:
        existing = {normalized_path_key(record.source_path) for record in self.records}
        candidates: list[Path] = []
        missing: list[Path] = []
        unsupported: list[Path] = []
        duplicates: list[Path] = []
        capacity: list[Path] = []
        added: list[BatchRecord] = []

        for supplied in paths:
            path = Path(supplied)
            if path.is_dir():
                pattern = "**/*" if include_subfolders else "*"
                candidates.extend(
                    sorted(
                        (item for item in path.glob(pattern) if item.is_file()),
                        key=lambda item: str(item).casefold(),
                    )
                )
            else:
                candidates.append(path)

        for path in candidates:
            if not path.is_file():
                missing.append(path)
                continue
            if path.suffix.casefold() not in SUPPORTED_VIDEO_EXTENSIONS:
                unsupported.append(path)
                continue
            key = normalized_path_key(path)
            if key in existing:
                duplicates.append(path)
                continue
            if len(self.records) >= self.maximum:
                capacity.append(path)
                continue
            record = BatchRecord(path.resolve())
            self.records.append(record)
            existing.add(key)
            added.append(record)

        return BatchAddResult(
            added=tuple(added),
            duplicates=tuple(duplicates),
            unsupported=tuple(unsupported),
            missing=tuple(missing),
            capacity_rejected=tuple(capacity),
        )

    def remove(self, identifiers: Iterable[str]) -> tuple[BatchRecord, ...]:
        selected = set(identifiers)
        removed = tuple(record for record in self.records if record.identifier in selected)
        self.records[:] = [record for record in self.records if record.identifier not in selected]
        return removed

    def clear(self) -> tuple[BatchRecord, ...]:
        removed = tuple(self.records)
        self.records.clear()
        return removed

    def reorder(self, ordered_identifiers: Iterable[str]) -> None:
        identifiers = tuple(ordered_identifiers)
        if len(identifiers) != len(self.records) or len(set(identifiers)) != len(identifiers):
            raise ValueError("A batch reorder must contain every row identifier exactly once.")
        by_id = {record.identifier: record for record in self.records}
        if set(identifiers) != set(by_id):
            raise ValueError("A batch reorder cannot add or remove rows.")
        self.records[:] = [by_id[identifier] for identifier in identifiers]

    def move(self, identifiers: Iterable[str], direction: int) -> None:
        """Move selected rows one position while retaining their relative order."""

        if direction not in {-1, 1}:
            raise ValueError("Batch rows can move only one position up (-1) or down (+1).")
        selected = set(identifiers)
        if direction < 0:
            for index in range(1, len(self.records)):
                if self.records[index].identifier in selected and self.records[index - 1].identifier not in selected:
                    self.records[index - 1], self.records[index] = self.records[index], self.records[index - 1]
        else:
            for index in range(len(self.records) - 2, -1, -1):
                if self.records[index].identifier in selected and self.records[index + 1].identifier not in selected:
                    self.records[index], self.records[index + 1] = self.records[index + 1], self.records[index]


@dataclass(frozen=True)
class BatchResolution:
    settings: JobSettings
    plan: ProcessingPlan
    fallback_notes: tuple[str, ...]
    requires_repair: bool

    @property
    def effective_summary(self) -> str:
        backend = self.plan.selected_backend or self.settings.backend
        denoise = self.plan.selected_denoiser or ("off" if not self.settings.denoise_enabled else self.settings.denoiser)
        fallback = f" · {len(self.fallback_notes)} fallback(s)" if self.fallback_notes else ""
        return f"{backend} · {self.settings.family} {self.settings.bit_depth}-bit · denoise {denoise}{fallback}"


def preferred_output_path(source: Path, family: str, output_directory: Path | None) -> Path:
    extension = ".mov" if family in {"prores", "dnxhr"} else ".mkv"
    directory = output_directory or source.parent
    return directory / f"{source.stem}.deinterlaced{extension}"


def _mov_requires_preservation_fallback(settings: JobSettings, media: MediaProbe) -> str | None:
    reasons: list[str] = []
    if settings.copy_attachments and media.attachment_count:
        reasons.append("attachments")
    if settings.copy_data and media.data_count:
        reasons.append("data streams")
    if settings.copy_subtitles:
        unsupported = sorted(
            {
                stream.codec_name
                for stream in media.streams_of_type("subtitle")
                if stream.codec_name not in MOV_SUBTITLE_CODECS
            }
        )
        if unsupported:
            reasons.append("subtitle codec(s) " + ", ".join(unsupported))
    if settings.copy_audio:
        unsupported = sorted(
            {
                stream.codec_name
                for stream in media.streams_of_type("audio")
                if stream.codec_name not in MOV_AUDIO_CODECS
            }
        )
        if unsupported:
            reasons.append("audio codec(s) " + ", ".join(unsupported))
    return "; ".join(reasons) if reasons else None


def _profile_candidates(settings: JobSettings) -> tuple[JobSettings, ...]:
    """Return same-family choices from requested fidelity toward safe compatibility."""

    candidates: list[JobSettings] = []

    def append(candidate: JobSettings) -> None:
        signature = (
            candidate.family,
            candidate.bit_depth,
            candidate.hardware_encode,
            candidate.av1_software_encoder,
            candidate.ffv1_chroma_mode,
        )
        if all(
            signature
            != (
                existing.family,
                existing.bit_depth,
                existing.hardware_encode,
                existing.av1_software_encoder,
                existing.ffv1_chroma_mode,
            )
            for existing in candidates
        ):
            candidates.append(candidate)

    append(settings)
    if settings.hardware_encode:
        append(replace(settings, hardware_encode=False))

    if settings.family == "ffv1":
        append(replace(settings, bit_depth=16, hardware_encode=False))
        append(replace(settings, bit_depth=16, hardware_encode=False, ffv1_chroma_mode="444"))
    elif settings.family == "prores":
        append(replace(settings, bit_depth=10, hardware_encode=False))
    elif settings.family == "dnxhr":
        append(replace(settings, bit_depth=10, hardware_encode=False))
    elif settings.family in {"hevc", "av1"}:
        depths = (settings.bit_depth, 10, 12)
        encoders = (
            (settings.av1_software_encoder, "libaom", "svt")
            if settings.family == "av1"
            else (settings.av1_software_encoder,)
        )
        for hardware in (settings.hardware_encode, False):
            for encoder in encoders:
                for depth in depths:
                    append(
                        replace(
                            settings,
                            bit_depth=depth,
                            hardware_encode=hardware,
                            av1_software_encoder=encoder,
                        )
                    )
    return tuple(candidates)


def _choose_profile_settings(
    settings: JobSettings,
    media: MediaProbe,
    capabilities: CapabilityReport,
) -> tuple[JobSettings, tuple[str, ...]]:
    diagnostics: list[str] = []
    for candidate in _profile_candidates(settings):
        try:
            profile = select_profile(
                candidate.family,
                candidate.bit_depth,
                candidate.hardware_encode,
                candidate.av1_software_encoder,
                candidate.ffv1_chroma_mode,
                media.video.pix_fmt,
            )
        except ValueError as exc:
            diagnostics.append(str(exc))
            continue
        capability_error = profile_capability_error(profile, capabilities)
        if capability_error:
            diagnostics.append(capability_error)
            continue
        notes: list[str] = []
        if candidate.hardware_encode != settings.hardware_encode:
            notes.append("Requested hardware encoder was unavailable; used the same codec family in software.")
        if candidate.bit_depth != settings.bit_depth:
            notes.append(
                f"Requested {settings.bit_depth}-bit output was unsupported; used proven {candidate.bit_depth}-bit output."
            )
        if candidate.av1_software_encoder != settings.av1_software_encoder:
            notes.append(
                f"Requested AV1 implementation was unavailable; used {candidate.av1_software_encoder}."
            )
        if candidate.ffv1_chroma_mode != settings.ffv1_chroma_mode:
            notes.append(
                "Native FFV1 chroma could not be classified safely; used explicit 4:4:4 FFV1 mastering."
            )
        return candidate, tuple(notes)
    detail = diagnostics[-1] if diagnostics else "no same-family profile passed capability validation"
    raise BatchCompatibilityError(
        f"No compatible {settings.family.upper()} output profile is available for this row: {detail}"
    )


def _choose_backend_settings(
    settings: JobSettings,
    analysis: IDetReport,
    capabilities: CapabilityReport,
) -> tuple[JobSettings, tuple[str, ...]]:
    if analysis.classification not in {"tff", "bff", "progressive"}:
        raise BatchCompatibilityError(
            "Interlace evidence is mixed or insufficient. This row needs manual cadence/field-order review and was not guessed.",
            needs_review=True,
        )

    candidate = settings
    notes: list[str] = []
    if analysis.classification == "progressive":
        if settings.backend not in {"auto", "progressive"} and not settings.allow_progressive_override:
            candidate = replace(candidate, backend="progressive", field_order="auto")
            notes.append("Measured progressive video bypassed the requested deinterlacer to prevent detail loss.")
        if candidate.output_cadence != "frame_rate":
            candidate = replace(candidate, output_cadence="frame_rate")
            notes.append("Progressive input retained its nominal frame rate because it has no separate fields to bob.")
    elif settings.backend == "progressive":
        candidate = replace(candidate, backend="auto")
        notes.append("Measured interlaced video replaced incompatible progressive passthrough with automatic deinterlacing.")

    backend = candidate.backend
    cuda_ready = bool(
        "bwdif_cuda" in capabilities.filters
        and "cuda" in capabilities.hwaccels
        and capabilities.interlace_runtime_verified.get("bwdif_cuda", True)
    )
    bwdif_ready = "bwdif" in capabilities.filters
    if backend == "vapoursynth_qtgmc" and not capabilities.qtgmc_ready:
        if not bwdif_ready:
            raise BatchCompatibilityError("QTGMC and FFmpeg BWDIF are both unavailable for this interlaced row.")
        candidate = replace(candidate, backend="ffmpeg_bwdif", vulkan_nnedi3=False)
        notes.append("QTGMC was unavailable; used FFmpeg BWDIF CPU as the next compatible deinterlacer.")
    elif backend == "ffmpeg_bwdif_cuda" and not cuda_ready:
        if not bwdif_ready:
            raise BatchCompatibilityError("CUDA BWDIF and CPU BWDIF are both unavailable for this row.")
        candidate = replace(candidate, backend="ffmpeg_bwdif")
        notes.append("CUDA BWDIF was unavailable; used the identical CPU BWDIF algorithm.")
    elif backend == "ffmpeg_bwdif" and not bwdif_ready:
        if capabilities.qtgmc_ready:
            candidate = replace(candidate, backend="vapoursynth_qtgmc")
            notes.append("FFmpeg BWDIF was unavailable; used higher-quality QTGMC.")
        else:
            raise BatchCompatibilityError("Neither FFmpeg BWDIF nor QTGMC is available for this row.")

    resolved_qtgmc = candidate.backend == "vapoursynth_qtgmc" or (
        candidate.backend == "auto"
        and analysis.classification in {"tff", "bff"}
        and capabilities.qtgmc_ready
    )
    if candidate.vulkan_nnedi3 and (not resolved_qtgmc or not capabilities.vulkan_nnedi3_ready):
        candidate = replace(candidate, vulkan_nnedi3=False)
        notes.append("Vulkan NNEDI3 was unavailable or inapplicable; retained QTGMC's CPU NNEDI3 interpolation.")
    if resolved_qtgmc and candidate.hardware_decode == "cuda":
        candidate = replace(candidate, hardware_decode="auto")
        notes.append("Explicit CUDA decode is not accepted by BestSource; automatic decode resolved to its safe software path.")
    return candidate, tuple(notes)


def _choose_denoise_settings(
    settings: JobSettings,
    analysis: IDetReport,
    capabilities: CapabilityReport,
) -> tuple[JobSettings, tuple[str, ...]]:
    if not settings.denoise_enabled:
        return settings, ()
    resolved_backend = settings.backend
    if resolved_backend == "auto":
        resolved_backend = (
            "progressive"
            if analysis.classification == "progressive"
            else ("vapoursynth_qtgmc" if capabilities.qtgmc_ready else "ffmpeg_bwdif")
        )
    allowed = (
        ("ffmpeg_fftdnoiz", "ffmpeg_atadenoise")
        if resolved_backend in {"ffmpeg_bwdif", "ffmpeg_bwdif_cuda"}
        else (
            "vs_bm3d",
            "vs_dfttest",
            "vs_mvtools",
            "vs_nlmeans",
            "ffmpeg_fftdnoiz",
            "ffmpeg_atadenoise",
        )
    )
    available = tuple(
        identifier
        for identifier in allowed
        if identifier in DENOISER_BY_ID and capabilities.denoise_capabilities.get(identifier, False)
    )
    if settings.denoiser in available:
        return settings, ()
    if available:
        selected = available[0]
        return (
            replace(
                settings,
                denoiser=selected,
                denoise_temporal_radius=(1 if selected == "ffmpeg_fftdnoiz" else settings.denoise_temporal_radius),
            ),
            (
                f"Requested temporal denoiser was unavailable/incompatible; used {DENOISER_BY_ID[selected].label}.",
            ),
        )
    return (
        replace(settings, denoise_enabled=False),
        ("No compatible temporal denoiser passed capability checks; denoising was disabled for this row.",),
    )


def resolve_batch_plan(
    requested: JobSettings,
    source: Path,
    media: MediaProbe,
    analysis: IDetReport,
    source_health: SourceHealthReport,
    capabilities: CapabilityReport,
    *,
    output_directory: Path | None = None,
    reserved_outputs: tuple[Path, ...] = (),
) -> BatchResolution:
    """Resolve one row without changing the shared user-selected settings.

    Fallbacks remain within the requested codec family, prefer the requested
    deinterlacer/denoiser when valid, and are always returned as explicit audit
    notes. Mixed/ambiguous cadence evidence is never guessed.
    """

    if not source.is_file():
        raise BatchCompatibilityError(f"Source file no longer exists: {source}")
    candidate = replace(
        requested,
        input_path=source,
        output_path=preferred_output_path(source, requested.family, output_directory),
        quality=min(40, max(0, requested.quality)),
        denoise_strength=min(10, max(1, requested.denoise_strength)),
        denoise_temporal_radius=min(6, max(1, requested.denoise_temporal_radius)),
        overwrite_approved=False,
    )
    notes: list[str] = []
    if candidate.aspect_mode == "manual":
        value = candidate.manual_dar.strip()
        pieces = value.replace("/", ":").split(":")
        try:
            valid_manual = len(pieces) == 2 and int(pieces[0]) > 0 and int(pieces[1]) > 0
        except ValueError:
            valid_manual = False
        if not valid_manual:
            candidate = replace(candidate, aspect_mode="preserve")
            notes.append("Invalid manual DAR was replaced with exact source SAR/DAR preservation.")

    candidate, backend_notes = _choose_backend_settings(candidate, analysis, capabilities)
    notes.extend(backend_notes)
    candidate, denoise_notes = _choose_denoise_settings(candidate, analysis, capabilities)
    notes.extend(denoise_notes)
    candidate, profile_notes = _choose_profile_settings(candidate, media, capabilities)
    notes.extend(profile_notes)

    preferred = preferred_output_path(source, candidate.family, output_directory)
    if candidate.family in {"prores", "dnxhr"}:
        mov_issue = _mov_requires_preservation_fallback(candidate, media)
        if mov_issue:
            preferred = preferred.with_suffix(".mkv")
            notes.append(
                "MOV could not preserve all selected tracks ("
                + mov_issue
                + "); used Matroska while retaining the selected video codec and tracks."
            )
    output = choose_available_artifact_path(
        preferred,
        DEINTERLACE_ARTIFACT_SUFFIXES,
        reserved=(source, *reserved_outputs),
    )
    candidate = replace(candidate, output_path=output)

    preview = build_plan(candidate, media, analysis, capabilities, source_health=None)
    if not preview.valid:
        raise BatchCompatibilityError(
            "No safe compatible plan could be built after bounded fallbacks: " + "; ".join(preview.errors)
        )
    requires_repair = bool(
        source_health.repair_required and preview.selected_backend == "vapoursynth_qtgmc"
    )
    if not requires_repair:
        plan = build_plan(candidate, media, analysis, capabilities, source_health=source_health)
        if not plan.valid:
            raise BatchCompatibilityError(
                "The source-aware plan remained invalid after compatible fallbacks: " + "; ".join(plan.errors)
            )
    else:
        plan = preview
        notes.append("QTGMC requires a validated separate repair copy before final processing.")
    fallback_warnings = tuple(f"Batch compatibility decision: {note}" for note in notes)
    if fallback_warnings:
        plan = replace(
            plan,
            warnings=tuple(dict.fromkeys((*plan.warnings, *fallback_warnings))),
        )
    return BatchResolution(candidate, plan, tuple(notes), requires_repair)
