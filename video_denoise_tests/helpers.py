from __future__ import annotations

from fractions import Fraction
from pathlib import Path

from deinterlace_studio.models import CapabilityReport, MediaProbe, StreamInfo


def fake_capabilities() -> CapabilityReport:
    denoisers = {
        "ffmpeg_fftdnoiz": True,
        "ffmpeg_atadenoise": True,
        "vs_bm3d": True,
        "vs_dfttest": True,
        "vs_mvtools": True,
        "vs_nlmeans": True,
    }
    return CapabilityReport(
        ffmpeg_path=Path(r"C:\tools\ffmpeg.exe"),
        ffprobe_path=Path(r"C:\tools\ffprobe.exe"),
        ffmpeg_version="ffmpeg version 9.0",
        ffmpeg_configuration="test",
        filters=frozenset({"fftdnoiz", "atadenoise"}),
        encoders=frozenset({"ffv1", "libx265", "hevc_nvenc", "libaom-av1", "libsvtav1", "av1_nvenc", "prores_ks", "dnxhd"}),
        encoder_pixel_formats={
            "ffv1": ("yuv420p16le", "yuv422p16le", "yuv444p16le"),
            "libx265": ("yuv420p10le", "yuv420p12le"),
            "hevc_nvenc": ("p010le", "p012le"),
            "libaom-av1": ("yuv420p10le", "yuv420p12le"),
            "libsvtav1": ("yuv420p10le", "yuv420p12le"),
            "av1_nvenc": ("p010le", "p012le"),
            "prores_ks": ("yuv444p10le",),
            "dnxhd": ("yuv444p10le",),
        },
        hwaccels=frozenset(),
        vspipe_path=Path(r"C:\tools\vspipe.exe"),
        vapoursynth_version="R79",
        qtgmc_ready=True,
        qtgmc_diagnostic="ready",
        qtgmc_install_command=None,
        encoder_verified_bit_depths={"hevc_nvenc": (10, 12), "av1_nvenc": (10, 12)},
        ffprobe_version="ffprobe version 9.0",
        denoise_capabilities=denoisers,
        denoise_backends={
            "ffmpeg_fftdnoiz": "ffmpeg",
            "ffmpeg_atadenoise": "ffmpeg",
            "vs_bm3d": "bm3dcpu",
            "vs_dfttest": "dfttest_cpu",
            "vs_mvtools": "mvtools",
            "vs_nlmeans": "nlm_ispc",
        },
        denoise_diagnostics={key: "ready" for key in denoisers},
    )


def fake_media(path: Path, *, field_order: str = "progressive", interlaced: int = 0) -> MediaProbe:
    video = StreamInfo(
        index=0,
        codec_type="video",
        codec_name="h264",
        width=1280,
        height=720,
        pix_fmt="yuv420p",
        bits_per_raw_sample=8,
        sample_aspect_ratio=Fraction(1, 1),
        display_aspect_ratio=Fraction(16, 9),
        r_frame_rate=Fraction(30000, 1001),
        avg_frame_rate=Fraction(30000, 1001),
        field_order=field_order,
        duration=10.01,
        nb_frames=300,
        color_range="tv",
        color_space="bt709",
        color_transfer="bt709",
        color_primaries="bt709",
    )
    audio = StreamInfo(index=1, codec_type="audio", codec_name="aac", tags={"language": "eng"})
    return MediaProbe(
        path=path,
        format_name="matroska",
        format_long_name="Matroska",
        duration=10.01,
        size=1000,
        bit_rate=1000000,
        start_time=0.0,
        streams=(video, audio),
        sampled_interlaced_frames=interlaced,
        sampled_progressive_frames=64 - interlaced,
        sampled_tff_frames=interlaced if field_order in {"tt", "tb", "tff"} else 0,
        sampled_bff_frames=interlaced if field_order in {"bb", "bt", "bff"} else 0,
    )
