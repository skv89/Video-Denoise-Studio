from __future__ import annotations

from dataclasses import dataclass, replace

from .denoise import denoiser_backend_has_gpu, resolve_denoiser_backend
from .models import CapabilityReport, JobSettings
from .presets import profile_capability_error, select_profile


MAXIMUM_FIDELITY = "maximum_fidelity"
HYBRID_QTGMC_GPU = "hybrid_qtgmc_gpu"
FAST_EVERYDAY_GPU = "fast_everyday_gpu"

# Keep this deliberately conservative. Other codecs may be supported by a
# particular NVIDIA driver/build, but these are the direct NVDEC paths the app
# can select without turning an otherwise valid CUDA-filter job into a decoder
# startup failure. Unsupported codecs still use CUDA BWDIF after one software
# decode/upload step.
DIRECT_NVDEC_CODECS = frozenset({"h264", "hevc", "av1", "mpeg2video", "vc1", "vp9"})


@dataclass(frozen=True)
class SpeedMode:
    identifier: str
    label: str
    quality: str
    speed: str
    description: str


@dataclass(frozen=True)
class AppliedSpeedMode:
    mode: SpeedMode
    settings: JobSettings
    changes: tuple[str, ...]
    cautions: tuple[str, ...]


SPEED_MODES: tuple[SpeedMode, ...] = (
    SpeedMode(
        MAXIMUM_FIDELITY,
        "QTGMC maximum quality — CPU reference",
        "Maximum QTGMC graph",
        "Reference / slowest",
        (
            "Uses automatic routing: the full maximum-quality QTGMC graph for measured interlace and "
            "passthrough for measured progressive video. CPU NNEDI3 and software encoding are selected "
            "where applicable; the output family and denoiser are retained."
        ),
    ),
    SpeedMode(
        HYBRID_QTGMC_GPU,
        "QTGMC maximum quality — accelerated where beneficial",
        "Same QTGMC settings",
        "Fastest verified QTGMC path",
        (
            "Uses the same maximum-quality QTGMC parameters. Verified Vulkan NNEDI3 is enabled only when "
            "it helps; CPU NNEDI3 is retained when a CUDA denoiser would contend for the GPU. HEVC/AV1 "
            "hardware encoding is enabled when the selected output supports it."
        ),
    ),
    SpeedMode(
        FAST_EVERYDAY_GPU,
        "Fast GPU — BWDIF CUDA + HEVC NVENC",
        "Good; different deinterlacer, below QTGMC",
        "Largest gain",
        (
            "Uses CUDA BWDIF, uses NVDEC when the source codec has a safe direct path, switches output "
            "to high-quality 10-bit HEVC NVENC, and turns temporal denoise off."
        ),
    ),
)

SPEED_MODE_BY_ID = {mode.identifier: mode for mode in SPEED_MODES}


def _qtgmc_error(capabilities: CapabilityReport) -> str | None:
    if not capabilities.vspipe_path:
        return "VSPipe is unavailable."
    if not capabilities.qtgmc_ready:
        return "QTGMC is unavailable: " + capabilities.qtgmc_diagnostic
    return None


def _fast_gpu_error(capabilities: CapabilityReport) -> str | None:
    if not capabilities.ffmpeg_path or not capabilities.ffprobe_path:
        return "FFmpeg and FFprobe are unavailable."
    if "cuda" not in capabilities.hwaccels:
        return "The selected FFmpeg build exposes no CUDA hardware path."
    if "bwdif" not in capabilities.filters or "bwdif_cuda" not in capabilities.filters:
        return "The selected FFmpeg build does not expose CUDA BWDIF."
    if (
        "bwdif_cuda" in capabilities.interlace_runtime_verified
        and not capabilities.interlace_runtime_verified["bwdif_cuda"]
    ):
        return "CUDA BWDIF failed its bounded runtime test: " + capabilities.interlace_runtime_diagnostics.get(
            "bwdif_cuda", "no diagnostic was reported"
        )
    profile = select_profile("hevc", 10, True)
    profile_error = profile_capability_error(profile, capabilities)
    if profile_error:
        return profile_error
    return None


def speed_mode_unavailable_reason(
    identifier: str,
    capabilities: CapabilityReport,
    *,
    source_classification: str | None = None,
    settings: JobSettings | None = None,
) -> str | None:
    if identifier not in SPEED_MODE_BY_ID:
        return f"Unknown speed/quality mode: {identifier}"
    progressive = source_classification == "progressive"
    if identifier == MAXIMUM_FIDELITY:
        return None if progressive else _qtgmc_error(capabilities)
    if identifier == HYBRID_QTGMC_GPU:
        if progressive:
            return None
        qtgmc_error = _qtgmc_error(capabilities)
        if qtgmc_error:
            return qtgmc_error
        # Vulkan NNEDI3 is one optional acceleration stage, not a requirement
        # for maximum-quality QTGMC.  A failed or unstable Vulkan graph keeps
        # CPU NNEDI3 and may still use an independently verified CUDA denoiser
        # or HEVC/AV1 NVENC output.
        return None
    return _fast_gpu_error(capabilities)


def apply_speed_mode(
    settings: JobSettings,
    identifier: str,
    capabilities: CapabilityReport,
    *,
    source_classification: str | None = None,
    source_codec: str | None = None,
    source_width: int | None = None,
    source_height: int | None = None,
) -> AppliedSpeedMode:
    try:
        mode = SPEED_MODE_BY_ID[identifier]
    except KeyError as exc:
        raise ValueError(f"Unknown speed/quality mode: {identifier}") from exc
    unavailable = speed_mode_unavailable_reason(
        identifier,
        capabilities,
        source_classification=source_classification,
        settings=settings,
    )
    if unavailable:
        raise ValueError(unavailable)
    progressive = source_classification == "progressive"
    changes: list[str] = []
    cautions: list[str] = []

    if identifier == MAXIMUM_FIDELITY:
        updated = replace(
            settings,
            backend="auto",
            hardware_decode="off",
            vulkan_nnedi3=False,
            hardware_encode=False,
        )
        changes.extend(
            (
                "Backend: Automatic (QTGMC for measured interlace; passthrough for progressive)",
                "QTGMC interpolation: CPU NNEDI3",
                "Decode: software / BestSource-managed",
                "HEVC/AV1 encode: software; archival codec choice retained",
                "Temporal denoise: retained exactly as selected",
            )
        )
        cautions.append("This is intentionally the slowest mode and is intended for important masters.")
    elif identifier == HYBRID_QTGMC_GPU:
        hardware_encode = settings.family in {"hevc", "av1"}
        denoise_backend = resolve_denoiser_backend(
            settings.denoiser,
            capabilities.denoise_backends.get(settings.denoiser),
            source_width,
            source_height,
        )
        cuda_denoise_active = bool(
            settings.denoise_enabled
            and denoiser_backend_has_gpu(denoise_backend)
        )
        use_vulkan = (
            not progressive
            and capabilities.vulkan_nnedi3_ready
            and not cuda_denoise_active
        )
        updated = replace(
            settings,
            # Keep Automatic visible: it is the routing policy which selects
            # this exact QTGMC graph for measured interlace and passthrough for
            # measured progressive material.  Explicit QTGMC is a manual
            # override, not a higher-quality graph.
            backend="auto",
            hardware_decode="off",
            vulkan_nnedi3=use_vulkan,
            hardware_encode=hardware_encode,
        )
        if progressive:
            changes.append("Automatic routing: measured progressive source remains progressive.")
        else:
            changes.append("Automatic routing: maximum-quality VapourSynth QTGMC for measured interlace")
            if use_vulkan:
                changes.append("QTGMC NNEDI3 interpolation: verified Vulkan GPU")
            elif not capabilities.vulkan_nnedi3_ready:
                changes.append(
                    "QTGMC NNEDI3 interpolation: CPU fallback because the optional Vulkan graph is unavailable"
                )
            else:
                changes.append(
                    "QTGMC NNEDI3 interpolation: CPU retained to avoid measured Vulkan/CUDA denoiser contention"
                )
            changes.append("QTGMC MVTools motion analysis/source matching: CPU (unchanged)")
        if settings.denoise_enabled and denoise_backend:
            changes.append(f"Temporal denoise: retained; resolved implementation {denoise_backend}")
        elif settings.denoise_enabled:
            changes.append("Temporal denoise: retained; implementation will be capability-validated in the plan")
        else:
            changes.append("Temporal denoise: remains off")
        if hardware_encode:
            changes.append(f"{settings.family.upper()} encode: NVIDIA hardware encoder enabled")
        else:
            changes.append(
                f"{settings.family.upper()} output retained; this codec has no NVIDIA encoder in the app"
            )
        cautions.extend(
            (
                "When Vulkan is used, only NNEDI3 spatial interpolation moves to it; MVTools motion analysis, "
                "source matching, lossless restoration, and the QTGMC parameter values are unchanged.",
                "If the optional Vulkan graph does not pass its local runtime probe, the mode remains available "
                "and safely retains CPU NNEDI3 instead of weakening QTGMC or changing deinterlacers.",
                "If a CUDA temporal denoiser is active, this mode keeps NNEDI3 on the CPU because simultaneous "
                "Vulkan and CUDA stages can be slower through GPU contention.",
                "NVENC accelerates only HEVC/AV1 compression; it does not accelerate or simplify QTGMC.",
            )
        )
    else:
        direct_nvdec = (source_codec or "").casefold() in DIRECT_NVDEC_CODECS
        decode_setting = "cuda" if direct_nvdec else "off"
        updated = replace(
            settings,
            backend="progressive" if progressive else "ffmpeg_bwdif_cuda",
            family="hevc",
            bit_depth=10,
            hardware_encode=True,
            hardware_decode=decode_setting,
            vulkan_nnedi3=False,
            denoise_enabled=False,
        )
        changes.extend(
            (
                (
                    "Measured progressive source: deinterlacing remains off"
                    if progressive
                    else "Deinterlacer: FFmpeg BWDIF CUDA"
                ),
                (
                    f"Decode: NVIDIA CUDA/NVDEC ({source_codec})"
                    if direct_nvdec
                    else (
                        f"Decode: software for {source_codec or 'unknown codec'}, then upload to CUDA BWDIF "
                        "(avoids an unsupported NVDEC startup failure)"
                    )
                ),
                "Output: 10-bit HEVC NVIDIA P7/UHQ/full multipass",
                f"Quality target: CQ {settings.quality} (retained from the current Quality control)",
                "Temporal denoise: disabled for maximum throughput",
            )
        )
        cautions.extend(
            (
                "BWDIF is a simpler deinterlacer than QTGMC; fine diagonals, difficult motion, and line twitter can look worse.",
                "HEVC NVENC is visually lossy and is not an archival substitute for mathematically lossless FFV1.",
                "The app preserves cadence, duration, geometry, tracks, and metadata settings.",
            )
        )
    return AppliedSpeedMode(mode, updated, tuple(changes), tuple(cautions))
