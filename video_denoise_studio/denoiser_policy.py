from __future__ import annotations

from dataclasses import dataclass

from deinterlace_studio.denoise import (
    MAX_DENOISE_STRENGTH,
    MAX_TEMPORAL_RADIUS,
    MIN_DENOISE_STRENGTH,
    MIN_TEMPORAL_RADIUS,
    denoiser_backend_display,
    denoiser_backend_has_gpu,
    denoiser_frame_window,
    denoiser_spec,
    resolve_denoiser_backend,
)
from deinterlace_studio.models import CapabilityReport


@dataclass(frozen=True)
class DenoiserControlPolicy:
    identifier: str
    radius_enabled: bool
    radius_minimum: int
    radius_maximum: int
    normalized_radius: int
    window_frames: int
    strength_help: str
    radius_help: str
    overview: str


@dataclass(frozen=True)
class DenoiserRanking:
    identifier: str
    quality_score: int
    speed_score: int
    quality_basis: str
    speed_basis: str


@dataclass(frozen=True)
class DenoiserBackendStatus:
    identifier: str
    available: bool
    backend: str | None
    display: str
    gpu_active: bool
    classification: str
    summary: str
    help_text: str


_OVERVIEWS = {
    "ffmpeg_fftdnoiz": (
        "Frequency-domain Wiener denoising in FFmpeg on the CPU. It always uses one previous, "
        "the target, and one next frame. It is fast, but its temporal span is not adjustable."
    ),
    "ffmpeg_atadenoise": (
        "Adaptive temporal averaging in FFmpeg on the CPU. It stops averaging across sufficiently large "
        "changes, which helps protect motion and cuts. FFmpeg requires an odd window of at least five frames."
    ),
    "vs_bm3d": (
        "Quality-first two-pass V-BM3D temporal block matching. It is the default for difficult "
        "noise. The app automatically prefers a capability-tested NVIDIA backend and otherwise uses a "
        "capability-tested CPU implementation."
    ),
    "vs_dfttest": (
        "Temporal frequency-domain denoising. The app automatically selects a capability-tested NVIDIA "
        "NVRTC/cuFFT implementation when available, with a tested CPU fallback. On interlaced TFF/BFF "
        "sources, it filters the two field parities independently and reweaves them without deinterlacing "
        "or changing stored-frame cadence. It is a strong general-purpose alternative to V-BM3D."
    ),
    "vs_mvtools": (
        "CPU motion-compensated temporal degraining. Neighboring frames are motion-matched before "
        "being combined, which can protect moving detail when the motion analysis succeeds."
    ),
    "vs_nlmeans": (
        "Non-local means searches for similar patches spatially and across neighboring frames. "
        "The app automatically prefers the capability-tested NVIDIA CUDA implementation and falls "
        "back to the tested ISPC CPU implementation. It can preserve repeated texture well, but may "
        "be slower on large rasters."
    ),
}


# These are intentionally selection-guidance scores, not claims of a universal
# mathematical ordering.  Quality is the expected potential on difficult mixed
# temporal noise when tuned carefully.  Speed is measured interactive one-frame
# preview latency with this app's automatic backend policy.  Source, radius,
# raster, motion, installed backend, and hardware can change the observed order;
# sustained full-file throughput can differ because process/JIT startup is then
# amortized over many frames.
_RANKINGS = {
    "ffmpeg_fftdnoiz": DenoiserRanking(
        "ffmpeg_fftdnoiz",
        quality_score=1,
        speed_score=5,
        quality_basis="Useful fast Wiener cleanup, but the fixed three-frame span and no motion compensation limit difficult-noise quality.",
        speed_basis="Fast CPU filter with a fixed, small temporal window.",
    ),
    "ffmpeg_atadenoise": DenoiserRanking(
        "ffmpeg_atadenoise",
        quality_score=2,
        speed_score=6,
        quality_basis="Adaptive averaging is effective on stable noise but can trade texture or motion detail for smoothing.",
        speed_basis="Usually the lightest temporal operation despite using an adjustable centered window.",
    ),
    "vs_bm3d": DenoiserRanking(
        "vs_bm3d",
        quality_score=6,
        speed_score=1,
        quality_basis="Highest quality potential from two-pass block matching and collaborative temporal reconstruction.",
        speed_basis="Two-pass quality-first reconstruction is normally the most computationally expensive choice, even with CUDA.",
    ),
    "vs_dfttest": DenoiserRanking(
        "vs_dfttest",
        quality_score=5,
        speed_score=2,
        quality_basis=(
            "Strong frequency-domain suppression with good detail retention when sigma and radius are tuned. "
            "The verified CPU/NVRTC comparison differed by at most one 8-bit display code value in the bounded sample."
        ),
        speed_basis=(
            "The verified NVIDIA NVRTC route has high sustained throughput, but a fresh one-frame preview pays "
            "process and NVRTC/JIT startup; it ranked fifth in two repeated preview-latency runs."
        ),
    ),
    "vs_mvtools": DenoiserRanking(
        "vs_mvtools",
        quality_score=4,
        speed_score=3,
        quality_basis="Motion compensation can preserve moving detail well, but bad vectors or occlusions can create artifacts.",
        speed_basis="CPU motion search remains costly, but its one-frame preview latency beat NVRTC DFTTest2 in the repeated local benchmark.",
    ),
    "vs_nlmeans": DenoiserRanking(
        "vs_nlmeans",
        quality_score=3,
        speed_score=4,
        quality_basis="Patch matching can preserve repeating texture, but strong settings may produce waxy texture or temporal inconsistency.",
        speed_basis="The verified CUDA route ranked third in two repeated one-frame preview-latency runs on the local runtime.",
    ),
}


def denoiser_ranking(identifier: str) -> DenoiserRanking:
    denoiser_spec(identifier)
    return _RANKINGS[identifier]


def denoiser_rankings_guide() -> str:
    lines = [
        "Selection scores (6 = highest/best)",
        "Quality is expected potential on difficult mixed temporal noise when tuned carefully. Speed is measured "
        "one-frame preview responsiveness with Automatic verified acceleration; sustained full-file throughput can "
        "differ when startup is amortized. These are guidance—not universal lab scores; noise, motion, texture, "
        "Strength, radius, resolution, and the selected backend can change the order.",
        "",
    ]
    for identifier in (
        "vs_bm3d",
        "vs_dfttest",
        "vs_mvtools",
        "vs_nlmeans",
        "ffmpeg_atadenoise",
        "ffmpeg_fftdnoiz",
    ):
        spec = denoiser_spec(identifier)
        ranking = denoiser_ranking(identifier)
        lines.extend(
            (
                f"{spec.label}",
                f"Quality {ranking.quality_score}/6 · Speed {ranking.speed_score}/6",
                f"Quality basis: {ranking.quality_basis}",
                f"Speed basis: {ranking.speed_basis}",
                "",
            )
        )
    lines.append("Use Frame preview on representative still and moving scenes before committing a full file.")
    return "\n".join(lines)


def denoiser_backend_status(
    identifier: str,
    capabilities: CapabilityReport | None,
    width: int | None = None,
    height: int | None = None,
) -> DenoiserBackendStatus:
    denoiser_spec(identifier)
    if capabilities is None:
        return DenoiserBackendStatus(
            identifier,
            False,
            None,
            "Capability scan pending",
            False,
            "pending",
            "Denoiser acceleration: scanning installed CPU/GPU backends…",
            "The app will run a bounded real graph before selecting a denoiser backend.",
        )
    if not capabilities.denoise_capabilities.get(identifier, False):
        diagnostic = capabilities.denoise_diagnostics.get(identifier, "No capability diagnostic was reported.")
        return DenoiserBackendStatus(
            identifier,
            False,
            None,
            "Unavailable",
            False,
            "unavailable",
            "Denoiser acceleration: unavailable — see help or Tools",
            f"This denoiser did not pass capability discovery. {diagnostic}",
        )
    backend = resolve_denoiser_backend(
        identifier,
        capabilities.denoise_backends.get(identifier),
        width,
        height,
    )
    if not backend:
        return DenoiserBackendStatus(
            identifier,
            False,
            None,
            "Unresolved",
            False,
            "unavailable",
            "Denoiser acceleration: unresolved — processing is disabled",
            "The denoiser was detected but no named implementation passed the bounded backend test.",
        )
    display = denoiser_backend_display(identifier, backend)
    gpu_active = denoiser_backend_has_gpu(backend)
    if gpu_active:
        classification = "NVIDIA GPU active"
    elif identifier in {"ffmpeg_fftdnoiz", "ffmpeg_atadenoise", "vs_mvtools"}:
        classification = "CPU only"
    else:
        classification = "optimized CPU fallback"
    summary = f"Denoiser acceleration: Automatic · {classification} — {display}"
    help_text = (
        f"Effective filter backend: {display}. Classification: {classification}.\n\n"
        "Denoiser acceleration is automatic and separate from the NVIDIA output-encoder checkbox. Capability "
        "discovery runs a bounded real graph and selects the highest-priority implementation that actually emits "
        "frames. V-BM3D, DFTTest2, and NLMeans prefer a verified NVIDIA implementation and otherwise use a "
        "verified CPU fallback. MVTools, FFTDNOIZ, and ATADENOISE are CPU filters in this app. This status does "
        "not imply GPU video decoding.\n\n"
        "There is no on/off switch because the capability report retains the verified effective route, "
        "not a complete user-selectable inventory of every fallback. Automatic selection avoids offering a route "
        "that was not proven on this installation."
    )
    return DenoiserBackendStatus(
        identifier,
        True,
        backend,
        display,
        gpu_active,
        classification,
        summary,
        help_text,
    )


def normalize_temporal_radius(identifier: str, value: int) -> int:
    """Return the nearest user-visible radius that the selected filter truly supports."""

    denoiser_spec(identifier)
    if identifier == "ffmpeg_fftdnoiz":
        return 1
    minimum = 2 if identifier == "ffmpeg_atadenoise" else MIN_TEMPORAL_RADIUS
    maximum = 3 if identifier == "vs_dfttest" else MAX_TEMPORAL_RADIUS
    return max(minimum, min(maximum, int(value)))


def denoiser_control_policy(identifier: str, strength: int, radius: int) -> DenoiserControlPolicy:
    denoiser_spec(identifier)
    normalized = normalize_temporal_radius(identifier, radius)
    if identifier == "ffmpeg_fftdnoiz":
        strength_help = (
            f"Strength {strength}/10 maps to FFmpeg sigma {strength * 0.5:.2f}. "
            "Higher values remove more noise and can soften fine texture."
        )
        radius_help = (
            "This filter has no adjustable temporal-radius parameter. It always uses exactly "
            "one frame before and one frame after the target (3 frames total)."
        )
    elif identifier == "ffmpeg_atadenoise":
        strength_help = (
            f"Strength {strength}/10 maps to adaptive thresholds A={strength * 0.005:.4f} and "
            f"B={strength * 0.01:.4f}. Higher values accept more neighboring pixels into the average."
        )
        radius_help = (
            f"Radius {normalized} uses {normalized} real frames before and after the target "
            f"({2 * normalized + 1} total). FFmpeg requires at least a 5-frame window, so the minimum is 2."
        )
    elif identifier == "vs_bm3d":
        strength_help = (
            f"Strength {strength}/10 maps to V-BM3D sigma {0.25 + strength * 0.25:.2f}. "
            "Higher values assume stronger noise and can remove subtle grain or texture."
        )
        radius_help = _centered_radius_help(normalized)
    elif identifier == "vs_dfttest":
        strength_help = (
            f"Strength {strength}/10 maps to DFTTest2 sigma {strength * 2.0:.2f}. "
            "Higher values suppress more frequency-domain energy."
        )
        radius_help = (
            _centered_radius_help(normalized)
            + " This app caps DFTTest2 at radius 3 (7 frames) because its optimized CPU and NVIDIA NVRTC "
            "implementations reject larger temporal radii. On interlaced TFF/BFF input, those seven stored-frame "
            "positions are evaluated separately within each field parity, then rewoven without deinterlacing."
        )
    elif identifier == "vs_mvtools":
        strength_help = (
            f"Strength {strength}/10 maps to MVTools thsad {100 + strength * 75}. "
            "Higher values permit stronger motion-compensated degraining and can accept poorer matches."
        )
        radius_help = _centered_radius_help(normalized)
    else:
        strength_help = (
            f"Strength {strength}/10 maps to NLMeans h={strength * 0.3:.2f}. "
            "Higher values combine less-similar patches and can smooth texture."
        )
        radius_help = _centered_radius_help(normalized)
    return DenoiserControlPolicy(
        identifier=identifier,
        radius_enabled=identifier != "ffmpeg_fftdnoiz",
        radius_minimum=2 if identifier == "ffmpeg_atadenoise" else MIN_TEMPORAL_RADIUS,
        radius_maximum=(
            MIN_TEMPORAL_RADIUS
            if identifier == "ffmpeg_fftdnoiz"
            else 3
            if identifier == "vs_dfttest"
            else MAX_TEMPORAL_RADIUS
        ),
        normalized_radius=normalized,
        window_frames=denoiser_frame_window(identifier, normalized),
        strength_help=strength_help,
        radius_help=radius_help,
        overview=_OVERVIEWS[identifier],
    )


def _centered_radius_help(radius: int) -> str:
    return (
        f"Radius {radius} uses {radius} real frames before and after the target "
        f"({2 * radius + 1} total). More context can improve stable noise estimates but costs time and memory."
    )


def validate_denoiser_controls(identifier: str, strength: int, radius: int) -> tuple[str, ...]:
    errors: list[str] = []
    try:
        denoiser_spec(identifier)
    except ValueError as exc:
        return (str(exc),)
    if not MIN_DENOISE_STRENGTH <= strength <= MAX_DENOISE_STRENGTH:
        errors.append(
            f"Temporal denoise strength must be a whole number from {MIN_DENOISE_STRENGTH} through "
            f"{MAX_DENOISE_STRENGTH}."
        )
    if identifier == "ffmpeg_fftdnoiz":
        if radius != 1:
            errors.append("FFmpeg fftdnoiz has a fixed temporal radius of 1 (3-frame window).")
    else:
        minimum = 2 if identifier == "ffmpeg_atadenoise" else MIN_TEMPORAL_RADIUS
        maximum = 3 if identifier == "vs_dfttest" else MAX_TEMPORAL_RADIUS
        if not minimum <= radius <= maximum:
            errors.append(
                f"Temporal radius for {denoiser_spec(identifier).label} must be {minimum} through {maximum}."
            )
    return tuple(errors)
