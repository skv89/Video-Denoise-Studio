from __future__ import annotations

import unittest
from pathlib import Path

from deinterlace_studio.acceleration import (
    FAST_EVERYDAY_GPU,
    HYBRID_QTGMC_GPU,
    MAXIMUM_FIDELITY,
    apply_speed_mode,
    speed_mode_unavailable_reason,
)
from deinterlace_studio.models import CapabilityReport, JobSettings


def capabilities(*, vulkan: bool = True, bwdif_cuda: bool = True) -> CapabilityReport:
    return CapabilityReport(
        ffmpeg_path=Path("C:/runtime/ffmpeg.exe"),
        ffprobe_path=Path("C:/runtime/ffprobe.exe"),
        ffmpeg_version="9.0",
        ffmpeg_configuration="--enable-cuda-nvcc",
        filters=frozenset({"bwdif", "bwdif_cuda", "idet"}),
        encoders=frozenset({"ffv1", "libx265", "hevc_nvenc", "av1_nvenc"}),
        encoder_pixel_formats={
            "ffv1": ("yuv420p16le",),
            "libx265": ("yuv420p10le",),
            "hevc_nvenc": ("p010le",),
            "av1_nvenc": ("p010le",),
        },
        hwaccels=frozenset({"cuda"}),
        vspipe_path=Path("C:/runtime/vspipe.exe"),
        vapoursynth_version="R79",
        qtgmc_ready=True,
        qtgmc_diagnostic="QTGMC graph passed.",
        qtgmc_install_command=None,
        encoder_verified_bit_depths={"hevc_nvenc": (10,), "av1_nvenc": (10,)},
        encoder_runtime_diagnostics={
            "hevc_nvenc": "10-bit bounded encode passed.",
            "av1_nvenc": "10-bit bounded encode passed.",
        },
        interlace_runtime_verified={"bwdif_cuda": bwdif_cuda},
        interlace_runtime_diagnostics={"bwdif_cuda": "bounded CUDA graph passed"},
        denoise_capabilities={"vs_bm3d": True},
        denoise_backends={"vs_bm3d": "vszipcu"},
        vulkan_nnedi3_ready=vulkan,
        vulkan_nnedi3_diagnostic=(
            "Vulkan graph passed." if vulkan else "No verified Vulkan device."
        ),
        vulkan_nnedi3_package_version="1.0" if vulkan else None,
    )


def settings(**changes) -> JobSettings:
    values = dict(
        input_path=Path("C:/media/source.mkv"),
        output_path=Path("C:/media/output.mkv"),
        backend="auto",
        family="ffv1",
        bit_depth=16,
        quality=14,
        output_cadence="field_rate",
        aspect_mode="preserve",
        denoise_enabled=True,
        denoiser="vs_bm3d",
        denoise_strength=4,
        denoise_temporal_radius=3,
        copy_audio=True,
        copy_subtitles=True,
        copy_attachments=True,
        copy_chapters=True,
        copy_metadata=True,
    )
    values.update(changes)
    return JobSettings(**values)


class SpeedModeTests(unittest.TestCase):
    def test_maximum_fidelity_preserves_user_master_and_denoise_intent(self) -> None:
        original = settings(hardware_encode=True, hardware_decode="cuda", vulkan_nnedi3=True)
        applied = apply_speed_mode(original, MAXIMUM_FIDELITY, capabilities())
        self.assertEqual(applied.settings.backend, "auto")
        self.assertEqual(applied.settings.family, "ffv1")
        self.assertTrue(applied.settings.denoise_enabled)
        self.assertEqual(applied.settings.denoise_temporal_radius, 3)
        self.assertFalse(applied.settings.hardware_encode)
        self.assertEqual(applied.settings.hardware_decode, "off")
        self.assertFalse(applied.settings.vulkan_nnedi3)

    def test_hybrid_mode_keeps_qtgmc_and_uses_only_verified_gpu_stages(self) -> None:
        original = settings(family="hevc", bit_depth=10, hardware_encode=False)
        applied = apply_speed_mode(original, HYBRID_QTGMC_GPU, capabilities())
        self.assertEqual(applied.settings.backend, "auto")
        self.assertFalse(applied.settings.vulkan_nnedi3)
        self.assertTrue(applied.settings.hardware_encode)
        self.assertEqual(applied.settings.hardware_decode, "off")
        self.assertTrue(applied.settings.denoise_enabled)
        self.assertIn("vszipcu", " ".join(applied.changes))
        self.assertIn("parameter values are unchanged", " ".join(applied.cautions))
        self.assertIn("GPU contention", " ".join(applied.cautions))

    def test_hybrid_mode_uses_vulkan_when_no_cuda_denoiser_competes(self) -> None:
        original = settings(denoise_enabled=False)
        applied = apply_speed_mode(original, HYBRID_QTGMC_GPU, capabilities())
        self.assertTrue(applied.settings.vulkan_nnedi3)

    def test_hybrid_mode_keeps_cpu_nnedi3_when_optional_vulkan_is_unavailable(self) -> None:
        caps = capabilities(vulkan=False)
        no_denoise = settings(denoise_enabled=False)
        reason = speed_mode_unavailable_reason(HYBRID_QTGMC_GPU, caps, settings=no_denoise)
        self.assertIsNone(reason)
        applied = apply_speed_mode(no_denoise, HYBRID_QTGMC_GPU, caps)
        self.assertEqual(applied.settings.backend, "auto")
        self.assertFalse(applied.settings.vulkan_nnedi3)
        self.assertIn("CPU fallback", " ".join(applied.changes))
        self.assertIn("mode remains available", " ".join(applied.cautions))

    def test_fast_mode_explicitly_changes_algorithm_codec_and_denoise_only(self) -> None:
        original = settings()
        applied = apply_speed_mode(
            original, FAST_EVERYDAY_GPU, capabilities(), source_codec="h264"
        )
        updated = applied.settings
        self.assertEqual(updated.backend, "ffmpeg_bwdif_cuda")
        self.assertEqual(updated.hardware_decode, "cuda")
        self.assertEqual((updated.family, updated.bit_depth, updated.hardware_encode), ("hevc", 10, True))
        self.assertFalse(updated.denoise_enabled)
        self.assertFalse(updated.vulkan_nnedi3)
        self.assertEqual(updated.quality, original.quality)
        self.assertEqual(updated.output_cadence, original.output_cadence)
        self.assertEqual(updated.aspect_mode, original.aspect_mode)
        self.assertEqual(updated.output_path, original.output_path)
        self.assertEqual(updated.copy_subtitles, original.copy_subtitles)
        self.assertIn("not an archival substitute", " ".join(applied.cautions))

    def test_fast_mode_software_decodes_codec_without_safe_direct_nvdec_path(self) -> None:
        applied = apply_speed_mode(
            settings(), FAST_EVERYDAY_GPU, capabilities(), source_codec="ffv1"
        )
        self.assertEqual(applied.settings.backend, "ffmpeg_bwdif_cuda")
        self.assertEqual(applied.settings.hardware_decode, "off")
        self.assertIn("then upload to CUDA BWDIF", " ".join(applied.changes))

    def test_fast_mode_requires_runtime_verified_cuda_bwdif(self) -> None:
        caps = capabilities(bwdif_cuda=False)
        reason = speed_mode_unavailable_reason(FAST_EVERYDAY_GPU, caps)
        self.assertIn("failed its bounded runtime test", reason or "")

    def test_progressive_source_is_never_deinterlaced_by_a_speed_mode(self) -> None:
        caps = capabilities()
        hybrid = apply_speed_mode(
            settings(), HYBRID_QTGMC_GPU, caps, source_classification="progressive"
        )
        fast = apply_speed_mode(
            settings(), FAST_EVERYDAY_GPU, caps, source_classification="progressive"
        )
        self.assertEqual(hybrid.settings.backend, "auto")
        self.assertFalse(hybrid.settings.vulkan_nnedi3)
        self.assertEqual(fast.settings.backend, "progressive")

    def test_reference_and_accelerated_modes_keep_same_qtgmc_routing_and_user_quality_intent(self) -> None:
        original = settings(family="hevc", bit_depth=10, denoise_enabled=False)
        reference = apply_speed_mode(original, MAXIMUM_FIDELITY, capabilities())
        accelerated = apply_speed_mode(original, HYBRID_QTGMC_GPU, capabilities())
        self.assertEqual(reference.settings.backend, "auto")
        self.assertEqual(accelerated.settings.backend, "auto")
        self.assertEqual(reference.settings.output_cadence, accelerated.settings.output_cadence)
        self.assertEqual(reference.settings.aspect_mode, accelerated.settings.aspect_mode)
        self.assertEqual(reference.settings.denoise_enabled, accelerated.settings.denoise_enabled)
        self.assertEqual(reference.settings.denoise_strength, accelerated.settings.denoise_strength)
        self.assertEqual(reference.settings.denoise_temporal_radius, accelerated.settings.denoise_temporal_radius)
        self.assertFalse(reference.settings.vulkan_nnedi3)
        self.assertTrue(accelerated.settings.vulkan_nnedi3)


if __name__ == "__main__":
    unittest.main()
