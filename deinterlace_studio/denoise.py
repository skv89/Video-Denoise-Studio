from __future__ import annotations

from dataclasses import dataclass


MIN_DENOISE_STRENGTH = 1
MAX_DENOISE_STRENGTH = 10
MIN_TEMPORAL_RADIUS = 1
MAX_TEMPORAL_RADIUS = 6
DEFAULT_DENOISER = "vs_bm3d"
DFTTEST_ADAPTIVE_CPU_NVRTC = "dfttest_adaptive_cpu_nvrtc"
DFTTEST_ADAPTIVE_CPU_CUFFT = "dfttest_adaptive_cpu_cufft"
DFTTEST_ADAPTIVE_NVRTC_CUFFT = "dfttest_adaptive_nvrtc_cufft"
GPU_DENOISE_BACKENDS = frozenset(
    {
        "vszipcu",
        "bm3dcuda_rtc",
        "bm3dcuda",
        "dfttest_nvrtc",
        "dfttest_cufft",
        DFTTEST_ADAPTIVE_CPU_NVRTC,
        DFTTEST_ADAPTIVE_CPU_CUFFT,
        DFTTEST_ADAPTIVE_NVRTC_CUFFT,
    }
)


@dataclass(frozen=True)
class DenoiserSpec:
    identifier: str
    label: str
    engine: str
    quality_rank: int
    temporal_description: str


DENOISER_SPECS = (
    DenoiserSpec(
        "ffmpeg_fftdnoiz",
        "FFmpeg fftdnoiz — fixed 3-frame frequency-domain",
        "ffmpeg",
        1,
        "fixed one previous + current + one next frame",
    ),
    DenoiserSpec(
        "ffmpeg_atadenoise",
        "FFmpeg atadenoise — adaptive temporal averaging",
        "ffmpeg",
        2,
        "configurable centered temporal window",
    ),
    DenoiserSpec(
        "vs_bm3d",
        "VapourSynth V-BM3D — two-pass temporal reconstruction (quality-first)",
        "vapoursynth",
        1,
        "two-pass block matching over a centered temporal window",
    ),
    DenoiserSpec(
        "vs_dfttest",
        "VapourSynth DFTTest2 — temporal frequency-domain (adaptive CPU/NVIDIA)",
        "vapoursynth",
        2,
        "frequency-domain filtering over a centered temporal window",
    ),
    DenoiserSpec(
        "vs_mvtools",
        "VapourSynth MVTools degrain — motion-compensated temporal",
        "vapoursynth",
        3,
        "motion-compensated previous/current/next-frame analysis",
    ),
    DenoiserSpec(
        "vs_nlmeans",
        "VapourSynth temporal NLMeans — non-local spatial/temporal matching",
        "vapoursynth",
        4,
        "non-local spatial matching over a centered temporal window",
    ),
)
DENOISER_BY_ID = {spec.identifier: spec for spec in DENOISER_SPECS}
DENOISER_LABELS = {spec.label: spec.identifier for spec in DENOISER_SPECS}


def denoiser_spec(identifier: str) -> DenoiserSpec:
    try:
        return DENOISER_BY_ID[identifier]
    except KeyError as exc:
        raise ValueError(f"Unknown temporal denoiser: {identifier}") from exc


def denoiser_label(identifier: str) -> str:
    return denoiser_spec(identifier).label


def denoiser_is_vapoursynth(identifier: str) -> bool:
    return denoiser_spec(identifier).engine == "vapoursynth"


def denoiser_backend_has_gpu(backend: str | None) -> bool:
    return bool(backend and backend in GPU_DENOISE_BACKENDS)


def denoiser_frame_window(identifier: str, temporal_radius: int) -> int:
    denoiser_spec(identifier)
    if identifier == "ffmpeg_fftdnoiz":
        return 3
    return max(5, temporal_radius * 2 + 1) if identifier == "ffmpeg_atadenoise" else temporal_radius * 2 + 1


def validate_denoise_numbers(strength: int, temporal_radius: int) -> tuple[str, ...]:
    errors: list[str] = []
    if not MIN_DENOISE_STRENGTH <= strength <= MAX_DENOISE_STRENGTH:
        errors.append(
            f"Temporal denoise strength must be a whole number from {MIN_DENOISE_STRENGTH} through "
            f"{MAX_DENOISE_STRENGTH}."
        )
    if not MIN_TEMPORAL_RADIUS <= temporal_radius <= MAX_TEMPORAL_RADIUS:
        errors.append(
            f"Temporal radius must be a whole number from {MIN_TEMPORAL_RADIUS} through {MAX_TEMPORAL_RADIUS}."
        )
    return tuple(errors)


def ffmpeg_denoise_filter(identifier: str, strength: int, temporal_radius: int) -> str:
    """Return a conservative quality-first FFmpeg temporal filter."""

    if validate_denoise_numbers(strength, temporal_radius):
        raise ValueError("Invalid temporal denoise strength or radius")
    if identifier == "ffmpeg_fftdnoiz":
        sigma = strength * 0.5
        return (
            f"fftdnoiz=sigma={sigma:.2f}:amount=1:block=32:overlap=0.8:"
            "method=wiener:prev=1:next=1:planes=7"
        )
    if identifier == "ffmpeg_atadenoise":
        threshold_a = strength * 0.005
        threshold_b = strength * 0.01
        window = max(5, temporal_radius * 2 + 1)
        return (
            f"atadenoise=0a={threshold_a:.4f}:0b={threshold_b:.4f}:"
            f"1a={threshold_a:.4f}:1b={threshold_b:.4f}:"
            f"2a={threshold_a:.4f}:2b={threshold_b:.4f}:s={window}:p=7:a=p"
        )
    raise ValueError(f"{identifier} is not an FFmpeg denoiser")


def _bm3d_backend_enum(backend: str) -> str:
    mapping = {
        "vszipcu": "CUDA_ZIP",
        "bm3dcuda_rtc": "CUDA_RTC",
        "bm3dcuda": "CUDA",
        "bm3dcpu": "CPU",
    }
    try:
        return mapping[backend]
    except KeyError as exc:
        raise ValueError(f"Unsupported V-BM3D implementation: {backend}") from exc


def resolve_denoiser_backend(
    identifier: str,
    backend: str | None,
    width: int | None,
    height: int | None,
) -> str | None:
    """Resolve a capability-approved adaptive implementation for this raster.

    Clean five-repeat measurements on the target RTX PRO 6000 show NVRTC is
    faster than cuFFT and optimized CPU at SD, 720p, and 1080p while preserving
    the accepted pixel-quality contract.  The capability scan must first prove
    every implementation named by an adaptive token; this function never
    promotes an untested backend and retains CPU/cuFFT as fallbacks.
    """

    if identifier != "vs_dfttest" or not backend:
        return backend
    if backend == DFTTEST_ADAPTIVE_CPU_NVRTC:
        return "dfttest_nvrtc"
    if backend == DFTTEST_ADAPTIVE_CPU_CUFFT:
        return "dfttest_cufft"
    if backend == DFTTEST_ADAPTIVE_NVRTC_CUFFT:
        return "dfttest_nvrtc"
    return backend


def vapoursynth_import_lines(identifier: str) -> list[str]:
    if identifier == "vs_bm3d":
        return ["from vsdenoise import bm3d"]
    if identifier == "vs_dfttest":
        return ["from vsdenoise import DFTTest"]
    if identifier == "vs_mvtools":
        return ["from vsdenoise import MVToolsPreset, mc_degrain"]
    if identifier == "vs_nlmeans":
        return ["from vsdenoise import nl_means"]
    raise ValueError(f"{identifier} is not a VapourSynth denoiser")


def vapoursynth_denoise_lines(
    identifier: str,
    strength: int,
    temporal_radius: int,
    backend: str,
) -> list[str]:
    """Return graph lines for a resolved, explicitly named VS implementation."""

    if validate_denoise_numbers(strength, temporal_radius):
        raise ValueError("Invalid temporal denoise strength or radius")
    if identifier == "vs_bm3d":
        sigma = 0.25 + strength * 0.25
        backend_enum = _bm3d_backend_enum(backend)
        return [
            "clip = depth(clip, 32)",
            (
                f"clip = bm3d(clip, sigma={sigma:.2f}, tr={temporal_radius}, refine=1, "
                f"profile=bm3d.Profile.HIGH, backend=bm3d.Backend.{backend_enum})"
            ),
            "clip = depth(clip, 16)",
        ]
    if identifier == "vs_dfttest":
        backend_enum = {
            "dfttest_cpu": "CPU",
            "dfttest_nvrtc": "NVRTC",
            "dfttest_cufft": "cuFFT",
        }.get(backend)
        if not backend_enum:
            raise ValueError(f"Unsupported DFTTest2 implementation: {backend}")
        sigma = strength * 2.0
        return [
            (
                f"clip = DFTTest(backend=DFTTest.Backend.{backend_enum}).denoise("
                f"clip, tr={temporal_radius}, sigma={sigma:.2f})"
            )
        ]
    if identifier == "vs_mvtools":
        if backend != "mvtools":
            raise ValueError(f"Unsupported MVTools implementation: {backend}")
        thsad = 100 + strength * 75
        return [
            (
                f"clip = mc_degrain(clip, preset=MVToolsPreset.HQ_SAD, tr={temporal_radius}, "
                f"blksize=16, overlap_div=2, refine=2, thsad={thsad})"
            )
        ]
    if identifier == "vs_nlmeans":
        backend_enum = {"vszipcu": "CUDA", "nlm_ispc": "ISPC"}.get(backend)
        if not backend_enum:
            raise ValueError(f"Unsupported temporal NLMeans implementation: {backend}")
        h = strength * 0.3
        return [
            (
                f"clip = nl_means(clip, h={h:.2f}, tr={temporal_radius}, a=3, s=4, "
                f"backend=nl_means.Backend.{backend_enum})"
            )
        ]
    raise ValueError(f"{identifier} is not a VapourSynth denoiser")


def denoiser_backend_display(identifier: str, backend: str | None) -> str:
    labels = {
        ("ffmpeg_fftdnoiz", "ffmpeg"): "FFmpeg CPU fftdnoiz",
        ("ffmpeg_atadenoise", "ffmpeg"): "FFmpeg CPU atadenoise",
        ("vs_bm3d", "vszipcu"): "VapourSynth V-BM3D CUDA (vszipcu)",
        ("vs_bm3d", "bm3dcuda_rtc"): "VapourSynth V-BM3D CUDA RTC",
        ("vs_bm3d", "bm3dcuda"): "VapourSynth V-BM3D CUDA",
        ("vs_bm3d", "bm3dcpu"): "VapourSynth V-BM3D CPU (AVX2 where supported)",
        ("vs_dfttest", "dfttest_cpu"): "VapourSynth DFTTest2 optimized CPU",
        ("vs_dfttest", "dfttest_nvrtc"): "VapourSynth DFTTest2 NVIDIA NVRTC",
        ("vs_dfttest", "dfttest_cufft"): "VapourSynth DFTTest2 NVIDIA cuFFT",
        (
            "vs_dfttest",
            DFTTEST_ADAPTIVE_CPU_NVRTC,
        ): "VapourSynth DFTTest2 NVIDIA NVRTC (verified CPU fallback)",
        (
            "vs_dfttest",
            DFTTEST_ADAPTIVE_CPU_CUFFT,
        ): "VapourSynth DFTTest2 NVIDIA cuFFT (verified CPU fallback)",
        (
            "vs_dfttest",
            DFTTEST_ADAPTIVE_NVRTC_CUFFT,
        ): "VapourSynth DFTTest2 NVIDIA NVRTC (verified cuFFT fallback)",
        ("vs_mvtools", "mvtools"): "VapourSynth MVTools CPU motion compensation",
        ("vs_nlmeans", "vszipcu"): "VapourSynth temporal NLMeans CUDA (vszipcu)",
        ("vs_nlmeans", "nlm_ispc"): "VapourSynth temporal NLMeans ISPC CPU",
    }
    return labels.get((identifier, backend or ""), backend or "unresolved")
