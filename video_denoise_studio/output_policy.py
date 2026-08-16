from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from deinterlace_studio.models import MediaProbe
from deinterlace_studio.presets import OutputProfile, select_profile

if TYPE_CHECKING:
    from .models import DenoiseSettings


CONTAINER_LABELS = {
    "Automatic (recommended for codec + selected source tracks)": "auto",
    "Matroska MKV — preservation-first": "mkv",
    "MPEG-4 MP4 — broad playback delivery": "mp4",
    "QuickTime MOV — editing / ProRes": "mov",
}
CONTAINER_ID_LABELS = {value: key for key, value in CONTAINER_LABELS.items()}
CONTAINER_EXTENSIONS = {"mkv": ".mkv", "mp4": ".mp4", "mov": ".mov"}
FAMILY_CONTAINERS = {
    "ffv1": ("auto", "mkv"),
    "hevc": ("auto", "mkv", "mp4", "mov"),
    "av1": ("auto", "mkv", "mp4"),
    "prores": ("auto", "mov"),
    "dnxhr": ("auto", "mov"),
}

ISO_DIRECT_SUBTITLE_CODECS = {"mov_text", "eia_608", "eia_708"}
ISO_TEXT_CONVERTIBLE_SUBTITLE_CODECS = {"ass", "ssa", "subrip", "srt", "text", "webvtt"}
ISO_SUBTITLE_CODECS = ISO_DIRECT_SUBTITLE_CODECS | ISO_TEXT_CONVERTIBLE_SUBTITLE_CODECS
MOV_AUDIO_CODECS = {
    "aac",
    "alac",
    "ac3",
    "eac3",
    "mp2",
    "mp3",
    "dts",
    "pcm_s16le",
    "pcm_s24le",
    "pcm_s32le",
    "pcm_f32le",
    "pcm_f64le",
}
MP4_AUDIO_CODECS = {"aac", "alac", "ac3", "eac3", "mp3"}


@dataclass(frozen=True)
class ContainerResolution:
    identifier: str
    extension: str
    reason: str


@dataclass(frozen=True)
class EncoderControlPolicy:
    quality_enabled: bool
    quality_label: str
    quality_minimum: int
    quality_maximum: int
    tune_grain_enabled: bool
    hardware_enabled: bool
    encoder_summary: str
    codec_help: str


def valid_container_ids(family: str) -> tuple[str, ...]:
    try:
        return FAMILY_CONTAINERS[family]
    except KeyError as exc:
        raise ValueError(f"Unsupported output codec family: {family}") from exc


def container_labels_for_family(family: str) -> tuple[str, ...]:
    allowed = set(valid_container_ids(family))
    return tuple(label for label, identifier in CONTAINER_LABELS.items() if identifier in allowed)


def resolve_container(settings: DenoiseSettings, media: MediaProbe | None) -> ContainerResolution:
    allowed = valid_container_ids(settings.family)
    requested = settings.container
    if requested not in allowed:
        raise ValueError(
            f"{settings.family.upper()} cannot be written to {requested.upper()}; choose one of "
            + ", ".join(identifier.upper() for identifier in allowed if identifier != "auto")
            + "."
        )
    if requested != "auto":
        reason = {
            "mkv": "Explicit MKV selection: flexible preservation-first container.",
            "mp4": "Explicit MP4 selection: broad playback delivery with stricter stream compatibility.",
            "mov": "Explicit MOV selection: editing-oriented QuickTime container.",
        }[requested]
        return ContainerResolution(requested, CONTAINER_EXTENSIONS[requested], reason)

    if settings.family == "ffv1":
        identifier = "mkv"
        reason = "Automatic: FFV1 uses MKV for its lossless master and flexible track preservation."
    elif settings.family in {"prores", "dnxhr"}:
        identifier = "mov"
        reason = f"Automatic: {settings.family.upper()} uses MOV for editing interoperability."
    elif settings.family == "av1":
        identifier = "mkv"
        reason = "Automatic: AV1 uses MKV as the preservation-first and most flexible default."
    elif _selected_tracks_need_mkv(settings, media):
        identifier = "mkv"
        reason = "Automatic: MKV selected because the requested source tracks need flexible preservation."
    else:
        identifier = "mp4"
        reason = "Automatic: HEVC MP4 selected for broad playback because the requested source tracks are compatible."
    return ContainerResolution(identifier, CONTAINER_EXTENSIONS[identifier], reason)


def _selected_tracks_need_mkv(settings: DenoiseSettings, media: MediaProbe | None) -> bool:
    if media is None:
        return True
    if settings.copy_attachments and media.attachment_count:
        return True
    if settings.copy_data and media.data_count:
        return True
    if settings.copy_subtitles and media.subtitle_count:
        return True
    if settings.copy_audio:
        return any(stream.codec_name.casefold() not in MP4_AUDIO_CODECS for stream in media.streams_of_type("audio"))
    return False


def select_output_profile(settings: DenoiseSettings, media: MediaProbe) -> tuple[OutputProfile, ContainerResolution]:
    container = resolve_container(settings, media)
    profile = select_profile(
        settings.family,
        settings.bit_depth,
        settings.hardware_encode,
        settings.av1_software_encoder,
        settings.ffv1_chroma_mode,
        media.video.pix_fmt,
    )
    return replace(profile, default_extension=container.extension), container


def encoder_control_policy(profile: OutputProfile | None, family: str, hardware_requested: bool) -> EncoderControlPolicy:
    hardware_enabled = family in {"hevc", "av1"}
    if profile is None:
        return EncoderControlPolicy(
            quality_enabled=family in {"hevc", "av1"},
            quality_label="Encoder quality (lower = better)",
            quality_minimum=0,
            quality_maximum=63 if family == "av1" else 51,
            tune_grain_enabled=family == "hevc" and not hardware_requested,
            hardware_enabled=hardware_enabled,
            encoder_summary="Select a compatible bit depth to resolve the encoder.",
            codec_help=_codec_help(family),
        )
    encoder = profile.encoder
    if encoder == "ffv1":
        return EncoderControlPolicy(False, "Lossless (no quality value)", 0, 0, False, False, "FFV1 v3 · mathematically lossless · all-intra · slices + CRC", _codec_help(family))
    if encoder == "prores_ks":
        return EncoderControlPolicy(False, "Fixed 4444 XQ profile", 0, 0, False, False, "ProRes 4444 XQ · 10-bit 4:4:4 · fixed highest profile", _codec_help(family))
    if encoder == "dnxhd":
        return EncoderControlPolicy(False, "Fixed DNxHR 444 profile", 0, 0, False, False, "DNxHR 444 · 10-bit 4:4:4 · fixed highest profile", _codec_help(family))
    if encoder in {"hevc_nvenc", "av1_nvenc"}:
        maximum = 63 if encoder == "av1_nvenc" else 51
        codec = "AV1" if encoder == "av1_nvenc" else "HEVC"
        return EncoderControlPolicy(
            True,
            f"NVENC CQ 0–{maximum} (lower = better)",
            0,
            maximum,
            False,
            True,
            f"{codec} NVENC · P7 · UHQ · VBR constant quality · full-resolution multipass · "
            "temporal AQ · UHQ-managed lookahead",
            _codec_help(family),
        )
    if encoder == "libx265":
        return EncoderControlPolicy(True, "x265 CRF 0–51 (lower = better)", 0, 51, True, True, "HEVC x265 · placebo preset · optional grain tune", _codec_help(family))
    if encoder == "libaom-av1":
        return EncoderControlPolicy(True, "libaom CRF 0–63 (lower = better)", 0, 63, False, True, "AV1 libaom · good usage · cpu-used 0 · 48-frame lag", _codec_help(family))
    return EncoderControlPolicy(True, "SVT-AV1 CRF 0–63 (lower = better)", 0, 63, False, True, "AV1 SVT-AV1 · preset 0", _codec_help(family))


def _codec_help(family: str) -> str:
    return {
        "ffv1": (
            "FFV1 is a mathematically lossless archival/intermediate codec. It preserves the denoised pixels exactly "
            "but creates large files. MKV is required and recommended. The CQ/CRF and grain controls do not apply."
        ),
        "hevc": (
            "HEVC/H.265 gives efficient delivery files. NVIDIA uses P7/UHQ constant quality when its exact bit-depth "
            "route passes a real capability encode; software fallback is x265 placebo/CRF. MP4 is convenient for simple "
            "playback, while MKV is safer for subtitles, attachments, data, and unusual audio."
        ),
        "av1": (
            "AV1 can compress more efficiently but playback/editing support varies. NVIDIA AV1 uses the same P7/UHQ "
            "constant-quality contract when verified; libaom/SVT are very slow software fallbacks. MKV is recommended; "
            "MP4 is available for compatible delivery workflows."
        ),
        "prores": (
            "ProRes 4444 XQ is a high-bitrate 10-bit 4:4:4 editing intermediate. It uses a fixed profile rather than "
            "CRF/CQ and is written to MOV. It is not mathematically lossless."
        ),
        "dnxhr": (
            "DNxHR 444 is a high-bitrate 10-bit 4:4:4 editing intermediate. It uses a fixed profile rather than "
            "CRF/CQ and is written to MOV. It is not mathematically lossless."
        ),
    }[family]


def quality_validation_error(profile: OutputProfile, quality: int) -> str | None:
    policy = encoder_control_policy(profile, profile.family, profile.hardware)
    if policy.quality_enabled and not policy.quality_minimum <= quality <= policy.quality_maximum:
        return f"{policy.quality_label} must be a whole number from {policy.quality_minimum} through {policy.quality_maximum}."
    return None


def encoder_args(profile: OutputProfile, quality: int, tune_grain: bool) -> list[str]:
    if profile.encoder not in {"hevc_nvenc", "av1_nvenc"}:
        return profile.encoder_args(quality, tune_grain and profile.encoder == "libx265")
    return [
        "-c:v",
        profile.encoder,
        "-preset",
        "p7",
        "-tune",
        "uhq",
        "-rc",
        "vbr",
        "-cq",
        str(quality),
        "-b:v",
        "0",
        "-multipass",
        "fullres",
        "-temporal-aq",
        "1",
        "-b_ref_mode",
        "middle",
        "-pix_fmt",
        profile.pix_fmt,
    ]


def container_compatibility_errors(settings: DenoiseSettings, media: MediaProbe, container: str) -> tuple[str, ...]:
    errors: list[str] = []
    if container == "mkv":
        return ()
    audio_codecs = MP4_AUDIO_CODECS if container == "mp4" else MOV_AUDIO_CODECS
    if settings.copy_audio:
        unsupported = sorted({stream.codec_name for stream in media.streams_of_type("audio") if stream.codec_name.casefold() not in audio_codecs})
        if unsupported:
            errors.append(
                f"{container.upper()} cannot directly preserve the selected audio codec(s): "
                + ", ".join(unsupported)
                + ". Choose MKV or deselect Audio."
            )
    if settings.copy_subtitles:
        unsupported = sorted({stream.codec_name for stream in media.streams_of_type("subtitle") if stream.codec_name.casefold() not in ISO_SUBTITLE_CODECS})
        if unsupported:
            errors.append(
                f"{container.upper()} cannot preserve or safely convert the selected subtitle codec(s): "
                + ", ".join(unsupported)
                + ". Choose MKV or deselect Subtitles."
            )
    if settings.copy_attachments and media.attachment_count:
        errors.append(f"{container.upper()} cannot preserve attachment streams. Choose MKV or deselect Attachments.")
    if settings.copy_data and media.data_count:
        errors.append(f"{container.upper()} cannot preserve data streams. Choose MKV or deselect Data.")
    return tuple(errors)


def container_help_text() -> str:
    return (
        "Automatic chooses by codec and the selected source tracks.\n\n"
        "MKV — preservation-first; required for FFV1 and recommended when retaining subtitles, attachments, data, or "
        "unusual audio. Supports HEVC and AV1.\n\n"
        "MP4 — broad playback/delivery for HEVC or AV1 with compatible audio/text subtitles; attachments and data are "
        "not supported.\n\n"
        "MOV — required/recommended for ProRes editing and also available for HEVC; it has stricter stream rules than MKV."
    )
