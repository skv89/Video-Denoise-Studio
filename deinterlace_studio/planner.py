from __future__ import annotations

import os
import subprocess
import uuid
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

from . import __version__
from .acceleration import DIRECT_NVDEC_CODECS
from .denoise import (
    DENOISER_SPECS,
    denoiser_backend_display,
    denoiser_frame_window,
    denoiser_is_vapoursynth,
    denoiser_spec,
    ffmpeg_denoise_filter,
    resolve_denoiser_backend,
    validate_denoise_numbers,
    vapoursynth_denoise_lines,
    vapoursynth_import_lines,
)
from .models import (
    AutomaticRecoveryAudit,
    CapabilityReport,
    IDetReport,
    JobSettings,
    MediaProbe,
    OutputExpectation,
    ProcessingPlan,
    SourceHealthReport,
    StreamInfo,
)
from .health import health_matches_source
from .presets import OutputProfile, profile_capability_error, select_profile
from .rationals import RationalError, derive_dar, exact_square_pixel_raster, parse_fraction
from .scheduling import VapourSynthSchedule, choose_vapoursynth_schedule


MOV_DIRECT_SUBTITLE_CODECS = {"mov_text", "eia_608", "eia_708"}
MOV_TEXT_CONVERTIBLE_SUBTITLE_CODECS = {"ass", "ssa", "subrip", "srt", "text", "webvtt"}
MOV_SUBTITLE_CODECS = MOV_DIRECT_SUBTITLE_CODECS | MOV_TEXT_CONVERTIBLE_SUBTITLE_CODECS
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
MOV_LANGUAGE_ALIASES = {
    "zh": "chi",
    "zho": "chi",
    "cmn": "chi",
    "yue": "chi",
}
INTERLACED_FIELD_ORDERS = {
    "tt": "tff",
    "tb": "tff",
    "tff": "tff",
    "bb": "bff",
    "bt": "bff",
    "bff": "bff",
}
QTGMC_CORE_THREAD_CAP = 16
QTGMC_VSPIPE_REQUESTS = 24


def _same_path(left: Path, right: Path) -> bool:
    return os.path.normcase(os.path.abspath(left)) == os.path.normcase(os.path.abspath(right))


def _execution_vspipe_path(configured: Path) -> Path:
    """Bypass Python's console-script relay when its native VSPipe is present.

    The wheel-installed ``Scripts\\vspipe.exe`` starts a second native process.
    Running that native executable directly lets cancellation and exit-status
    handling target the process that owns the VapourSynth graph.
    """

    if configured.parent.name.casefold() == "scripts":
        native = configured.parent.parent / "Lib" / "site-packages" / "vapoursynth" / "vspipe.exe"
        if native.is_file():
            return native
    return configured


def _sidecar(output: Path, suffix: str) -> Path:
    return output.with_name(output.name + suffix)


def _source_geometry(media: MediaProbe) -> tuple[int, int, Fraction, Fraction]:
    video = media.video
    if not video.width or not video.height:
        raise ValueError("The source video has no usable stored dimensions")
    sar = video.sample_aspect_ratio
    dar = video.display_aspect_ratio
    if sar is None and dar is not None:
        sar = dar / Fraction(video.width, video.height)
    sar = sar or Fraction(1, 1)
    dar = dar or derive_dar(video.width, video.height, sar)
    return video.width, video.height, sar, dar


def _resolve_geometry(settings: JobSettings, media: MediaProbe) -> tuple[int, int, Fraction, Fraction, str]:
    source_width, source_height, source_sar, source_dar = _source_geometry(media)
    if settings.aspect_mode == "preserve":
        return source_width, source_height, source_sar, source_dar, "Preserve stored raster and exact source SAR/DAR"
    if settings.aspect_mode == "square":
        width, height = exact_square_pixel_raster(source_dar, source_width, source_height)
        return width, height, Fraction(1, 1), source_dar, "Square pixels at the smallest exact-DAR raster without downscaling"
    if settings.aspect_mode == "manual":
        dar = parse_fraction(settings.manual_dar)
        if dar is None:
            raise RationalError("Manual DAR is required")
        sar = dar / Fraction(source_width, source_height)
        return source_width, source_height, sar, dar, "Manual DAR metadata without scaling/cropping"
    raise ValueError(f"Unknown aspect mode: {settings.aspect_mode}")


def _resolve_backend(settings: JobSettings, analysis: IDetReport, capabilities: CapabilityReport) -> tuple[str, list[str]]:
    warnings: list[str] = []
    backend = settings.backend
    if backend == "auto":
        if analysis.classification == "progressive":
            backend = "progressive"
        elif analysis.classification in {"tff", "bff"}:
            if capabilities.qtgmc_ready:
                backend = "vapoursynth_qtgmc"
            else:
                backend = "ffmpeg_bwdif"
                warnings.append(
                    "QTGMC dependencies are unavailable, so Auto selected FFmpeg CPU BWDIF. "
                    "This is an explicit analyzed choice, not a silent runtime fallback."
                )
        else:
            backend = "unresolved"
    return backend, warnings


def _resolve_field_order(settings: JobSettings, media: MediaProbe, analysis: IDetReport) -> tuple[str | None, list[str]]:
    warnings: list[str] = []
    if settings.field_order in {"tff", "bff"}:
        selected = settings.field_order
    elif analysis.dominant_field_order:
        selected = analysis.dominant_field_order
    else:
        selected = INTERLACED_FIELD_ORDERS.get((media.video.field_order or "").lower())

    metadata_order = INTERLACED_FIELD_ORDERS.get((media.video.field_order or "").lower())
    if selected and metadata_order and selected != metadata_order:
        warnings.append(
            f"Measured IDet/selected order {selected.upper()} conflicts with metadata {metadata_order.upper()}; "
            f"the plan will use {selected.upper()}."
        )
    return selected, warnings


def _profile_from_settings(settings: JobSettings, media: MediaProbe | None) -> OutputProfile:
    return select_profile(
        settings.family,
        settings.bit_depth,
        settings.hardware_encode,
        settings.av1_software_encoder,
        settings.ffv1_chroma_mode,
        media.video.pix_fmt if media else None,
    )


def _ffmpeg_input_args(
    settings: JobSettings,
    backend: str,
    media: MediaProbe | None = None,
) -> tuple[list[str], bool]:
    """Return input options and whether a hardware-frame download is required."""

    if backend == "ffmpeg_bwdif_cuda":
        automatic_direct = bool(
            settings.hardware_decode == "auto"
            and media is not None
            and media.video.codec_name.casefold() in DIRECT_NVDEC_CODECS
        )
        if settings.hardware_decode == "cuda" or automatic_direct:
            return ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda", "-extra_hw_frames", "16"], True
        # Software-decoded frames are uploaded immediately before bwdif_cuda.
        # This keeps the GPU deinterlacer/encoder usable for FFV1, ProRes,
        # DNxHR, and other sources without a safe direct NVDEC path.
        return [], False
    if settings.hardware_decode == "cuda":
        return ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda", "-extra_hw_frames", "16"], True
    if settings.hardware_decode == "auto":
        return ["-hwaccel", "auto"], False
    return [], False


def _download_format(media: MediaProbe) -> str:
    depth = media.video.bits_per_raw_sample or 8
    if depth <= 8:
        return "nv12"
    if depth <= 10:
        return "p010le"
    return "p016le"


def _aspect_filters(
    settings: JobSettings,
    width: int,
    height: int,
    sar: Fraction,
    media: MediaProbe,
) -> list[str]:
    filters: list[str] = []
    source_width, source_height, _, _ = _source_geometry(media)
    if settings.aspect_mode == "square" and (width != source_width or height != source_height):
        filters.append(f"zscale=w={width}:h={height}:filter=spline36:dither=error_diffusion")
    filters.append(_setsar_filter(sar))
    return filters


def _setsar_filter(sar: Fraction) -> str:
    maximum = max(abs(sar.numerator), sar.denominator, 100)
    if maximum > 2_147_483_647:
        raise RationalError("The requested sample-aspect ratio exceeds FFmpeg's supported integer range")
    return f"setsar=sar={sar.numerator}/{sar.denominator}:max={maximum}"


def _ffmpeg_video_filters(
    settings: JobSettings,
    backend: str,
    field_order: str | None,
    profile: OutputProfile,
    media: MediaProbe,
    width: int,
    height: int,
    sar: Fraction,
    download_frames: bool,
) -> str:
    filters: list[str] = []
    if download_frames and backend != "ffmpeg_bwdif_cuda":
        filters += ["hwdownload", f"format={_download_format(media)}"]
    if backend in {"ffmpeg_bwdif", "ffmpeg_bwdif_cuda"}:
        mode = "send_field" if settings.output_cadence == "field_rate" else "send_frame"
        parity = field_order or "auto"
        filter_name = "bwdif_cuda" if backend == "ffmpeg_bwdif_cuda" else "bwdif"
        if backend == "ffmpeg_bwdif_cuda" and not download_frames:
            filters += [f"format={_download_format(media)}", "hwupload_cuda"]
        filters.append(f"{filter_name}=mode={mode}:parity={parity}:deint=all")
        if backend == "ffmpeg_bwdif_cuda":
            filters += ["hwdownload", f"format={_download_format(media)}"]
        filters.append("setfield=mode=prog")
    setparams = _setparams_filter(media)
    if setparams:
        filters.append(setparams)
    if settings.denoise_enabled and not denoiser_is_vapoursynth(settings.denoiser):
        filters.append(
            ffmpeg_denoise_filter(
                settings.denoiser,
                settings.denoise_strength,
                settings.denoise_temporal_radius,
            )
        )
    filters += _aspect_filters(settings, width, height, sar, media)
    filters.append(f"format={profile.pix_fmt}")
    return ",".join(filters)


def _vapoursynth_script(
    settings: JobSettings,
    media: MediaProbe,
    backend: str,
    field_order: str | None,
    denoise_backend: str | None,
    width: int,
    height: int,
    sar: Fraction,
    schedule: VapourSynthSchedule,
) -> str:
    source_width, source_height, _, _ = _source_geometry(media)
    deinterlace = backend == "vapoursynth_qtgmc"
    imports = [
        "import os",
        "import tempfile",
        "import vapoursynth as vs",
        "from vapoursynth import core",
        "from vstools import depth",
    ]
    if deinterlace:
        imports.append("from vsdeinterlace import QTempGaussMC")
        if settings.vulkan_nnedi3:
            imports.append("from vsaa import NNEDI3")
    if settings.denoise_enabled and denoiser_is_vapoursynth(settings.denoiser):
        imports.extend(vapoursynth_import_lines(settings.denoiser))
    lines = [
        f"# Generated by Deinterlace Studio {__version__}; do not edit while a job is running.",
        *imports,
        "",
        f"core.num_threads = min({schedule.core_threads}, max(1, core.num_threads))",
        f"SOURCE = {str(media.path)!r}",
        "CACHE_ROOT = os.path.join(tempfile.gettempdir(), 'Deinterlace Studio BestSource Cache')",
        "os.makedirs(CACHE_ROOT, exist_ok=True)",
        "clip = core.bs.VideoSource(source=SOURCE, cachemode=1, cachepath=CACHE_ROOT)",
        "if clip.format is None:",
        "    raise RuntimeError('BestSource returned a variable-format clip; normalize the source before processing')",
        "clip = depth(clip, 16)",
    ]
    if deinterlace:
        if field_order not in {"tff", "bff"}:
            raise ValueError("QTGMC requires a resolved TFF or BFF field order")
        lines += [
            f"clip = core.std.SetFieldBased(clip, value={2 if field_order == 'tff' else 1})",
            "qtgmc = QTempGaussMC(",
            "    analyze_force_tr=3,",
            "    analyze_blksize=16,",
            "    analyze_overlap=2,",
            "    analyze_refine=2,",
            "    prefilter_tr=2,",
            "    basic_tr=2,",
            *(["    basic_bobber=NNEDI3(nsize=1, gpu=True),"] if settings.vulkan_nnedi3 else []),
            "    final_tr=3,",
            ")",
            "qtgmc = qtgmc.source_match(tr=2, mode=QTempGaussMC.SourceMatchMode.TWICE_REFINED)",
            "qtgmc = qtgmc.lossless(mode=QTempGaussMC.LosslessMode.POSTSMOOTH, anti_comb=True)",
        ]
        if settings.output_cadence == "field_rate":
            lines.append(f"clip = qtgmc.bob(clip, tff={field_order == 'tff'})")
        else:
            lines += [
                "qtgmc = qtgmc.motion_blur(fps_divisor=2)",
                f"clip = qtgmc.deinterlace(clip, tff={field_order == 'tff'})",
            ]
    if settings.denoise_enabled and denoiser_is_vapoursynth(settings.denoiser):
        if not denoise_backend:
            raise ValueError("The selected VapourSynth denoiser has no resolved implementation")
        lines += [
            "# Temporal denoise is deliberately applied after deinterlacing.",
            *vapoursynth_denoise_lines(
                settings.denoiser,
                settings.denoise_strength,
                settings.denoise_temporal_radius,
                denoise_backend,
            ),
        ]
    if settings.aspect_mode == "square" and (width != source_width or height != source_height):
        lines.append(f"clip = core.resize.Spline36(clip, width={width}, height={height})")
    lines += [
        "clip = core.std.SetFieldBased(clip, value=0)",
        f"clip = core.std.SetFrameProps(clip, _SARNum={sar.numerator}, _SARDen={sar.denominator})",
        "clip.set_output()",
        "",
    ]
    return "\n".join(lines)


def _track_mapping(
    settings: JobSettings,
    source_index: int,
    *,
    media: MediaProbe | None = None,
    mov: bool = False,
) -> tuple[list[str], list[str]]:
    maps: list[str] = ["-map", "0:v:0"]
    codecs: list[str] = []
    if settings.copy_audio:
        maps += ["-map", f"{source_index}:a?"]
        codecs += ["-c:a", "copy"]
    if settings.copy_subtitles:
        maps += ["-map", f"{source_index}:s?"]
        if mov and media is not None:
            for position, stream in enumerate(media.streams_of_type("subtitle")):
                codec = (
                    "mov_text"
                    if stream.codec_name.casefold() in MOV_TEXT_CONVERTIBLE_SUBTITLE_CODECS
                    else "copy"
                )
                codecs += [f"-c:s:{position}", codec]
        else:
            codecs += ["-c:s", "copy"]
    if settings.copy_attachments:
        maps += ["-map", f"{source_index}:t?"]
        codecs += ["-c:t", "copy"]
    if settings.copy_data:
        maps += ["-map", f"{source_index}:d?"]
        codecs += ["-c:d", "copy"]
    return maps, codecs


def _metadata_args(settings: JobSettings, source_index: int) -> list[str]:
    args = ["-map_metadata", str(source_index) if settings.copy_metadata else "-1"]
    args += ["-map_chapters", str(source_index) if settings.copy_chapters else "-1"]
    return args


def _mov_language(value: str) -> str:
    normalized = value.strip().casefold().replace("_", "-")
    base = normalized.split("-", 1)[0]
    return MOV_LANGUAGE_ALIASES.get(normalized, MOV_LANGUAGE_ALIASES.get(base, value))


def _copied_stream_metadata_args(
    settings: JobSettings,
    media: MediaProbe,
    *,
    mov: bool = False,
) -> list[str]:
    """Make direct-copy track metadata explicit across container changes.

    FFmpeg normally propagates stream metadata with ``-map``, but MOV can drop
    a language tag while copying an AC-3/MP3 stream from Matroska.  Validation
    correctly treats that as a track-contract failure, so emit the portable
    metadata keys explicitly for every selected non-video stream.
    """

    args: list[str] = []
    selected = (
        ("audio", "a", settings.copy_audio),
        ("subtitle", "s", settings.copy_subtitles),
        ("attachment", "t", settings.copy_attachments),
        ("data", "d", settings.copy_data),
    )
    for codec_type, specifier, enabled in selected:
        if not enabled:
            continue
        for position, stream in enumerate(media.streams_of_type(codec_type)):
            for key in ("language", "title", "filename", "mimetype"):
                value = stream.tags.get(key)
                if value:
                    if mov and key == "language":
                        value = _mov_language(value)
                    args += [f"-metadata:s:{specifier}:{position}", f"{key}={value}"]
    return args


def _color_args(media: MediaProbe) -> list[str]:
    video = media.video
    args: list[str] = []
    for option, value in (
        ("-color_range", video.color_range),
        ("-colorspace", video.color_space),
        ("-color_trc", video.color_transfer),
        ("-color_primaries", video.color_primaries),
    ):
        if value and value not in {"unknown", "reserved", "unspecified"}:
            args += [option, value]
    return args


def _setparams_filter(media: MediaProbe) -> str | None:
    """Apply source color properties to frames, including across a Y4M pipe."""

    video = media.video
    range_value = {"pc": "full", "jpeg": "full", "tv": "limited", "mpeg": "limited"}.get(
        (video.color_range or "").lower(),
        video.color_range,
    )
    values: list[str] = []
    for key, value in (
        ("range", range_value),
        ("colorspace", video.color_space),
        ("color_trc", video.color_transfer),
        ("color_primaries", video.color_primaries),
    ):
        if value and value not in {"unknown", "reserved", "unspecified"}:
            values.append(f"{key}={value}")
    return "setparams=" + ":".join(values) if values else None


def _expected_tracks(
    settings: JobSettings,
    media: MediaProbe,
    codec_type: str,
    *,
    mov: bool = False,
) -> tuple[StreamInfo, ...]:
    enabled = {
        "audio": settings.copy_audio,
        "subtitle": settings.copy_subtitles,
        "attachment": settings.copy_attachments,
        "data": settings.copy_data,
    }.get(codec_type, False)
    streams = media.streams_of_type(codec_type) if enabled else ()
    if not mov:
        return streams
    normalized: list[StreamInfo] = []
    for stream in streams:
        tags = dict(stream.tags)
        if tags.get("language"):
            tags["language"] = _mov_language(tags["language"])
        codec_name = (
            "mov_text"
            if codec_type == "subtitle"
            and stream.codec_name.casefold() in MOV_TEXT_CONVERTIBLE_SUBTITLE_CODECS
            else stream.codec_name
        )
        normalized.append(replace(stream, codec_name=codec_name, tags=tags))
    return tuple(normalized)


def build_plan(
    settings: JobSettings,
    media: MediaProbe | None,
    analysis: IDetReport | None,
    capabilities: CapabilityReport,
    *,
    source_health: SourceHealthReport | None = None,
    automatic_recovery: AutomaticRecoveryAudit | None = None,
    run_id: str | None = None,
) -> ProcessingPlan:
    errors: list[str] = []
    warnings: list[str] = []
    selected_backend: str | None = None
    selected_field_order: str | None = None
    selected_denoiser: str | None = None
    selected_denoise_backend: str | None = None
    profile: OutputProfile | None = None
    expected: OutputExpectation | None = None
    schedule: VapourSynthSchedule | None = None
    ffmpeg_command: list[str] = []
    vspipe_command: list[str] | None = None
    script: str | None = None
    display_command = ""
    analysis_summary = "Analysis has not completed."
    output = settings.output_path
    token = run_id or uuid.uuid4().hex[:12]
    partial = output.with_name(f".{output.stem}.partial.{token}{output.suffix}") if output.name else None
    log_path = _sidecar(output, ".Deinterlace.log") if output.name else None
    report_path = _sidecar(output, ".Deinterlace.json") if output.name else None
    script_path = _sidecar(output, ".Deinterlace.vpy") if output.name else None
    temp_script = output.with_name(f".{output.stem}.Deinterlace.{token}.vpy") if output.name else None

    if not capabilities.ffmpeg_path:
        errors.append("FFmpeg was not found. Select a compatible FFmpeg build in Tools.")
    if not capabilities.ffprobe_path:
        errors.append("FFprobe was not found. Select a matching FFprobe binary in Tools.")
    if not settings.input_path.is_file():
        errors.append(f"Input file does not exist: {settings.input_path}")
    if not output.name:
        errors.append("Choose an output file.")
    elif _same_path(settings.input_path, output):
        errors.append("The output resolves to the input file. Source media can never be overwritten.")
    if settings.quality < 0 or settings.quality > 40:
        errors.append("Quality must be a whole number from 0 through 40.")
    if settings.denoise_enabled:
        errors.extend(validate_denoise_numbers(settings.denoise_strength, settings.denoise_temporal_radius))
        try:
            denoiser_spec(settings.denoiser)
        except ValueError as exc:
            errors.append(str(exc))
    if media and not _same_path(media.path, settings.input_path):
        errors.append("The analyzed media path no longer matches the selected input; analyze again.")
    health_is_current = False
    if source_health is not None:
        health_is_current = health_matches_source(source_health, settings.input_path)
        if not health_is_current:
            errors.append("The source changed after its fast health precheck; run Probe + IDet analysis again.")
    if media is None:
        errors.append("Probe the selected input before building a processing plan.")
    if analysis is None:
        errors.append("Run sampled or full IDet analysis before processing.")

    try:
        profile = _profile_from_settings(settings, media)
    except ValueError as exc:
        errors.append(str(exc))

    if profile:
        capability_error = profile_capability_error(profile, capabilities)
        if capability_error:
            errors.append(capability_error)
        if output.suffix.lower() not in {".mkv", ".mov"}:
            errors.append("Output must use .mkv or .mov so streams and mastering codecs are muxed predictably.")
        if profile.family in {"hevc", "av1", "ffv1"} and output.suffix.lower() != ".mkv":
            warnings.append(f"{profile.label} is normally stored in Matroska (.mkv).")
        if profile.encoder in {"prores_ks", "dnxhd"} and output.suffix.lower() == ".mkv":
            warnings.append(
                "ProRes/DNxHR video is legal in Matroska, but some DirectShow/MPC and editor paths can stall or "
                "reject that codec/container combination even when FFmpeg and VLC decode it cleanly. Native MOV "
                "is recommended. Tools → Create fast MOV compatibility copy can remux a completed MKV without "
                "rerunning deinterlacing or denoising."
            )

    if media and output.suffix.lower() == ".mov":
        if settings.copy_attachments and media.attachment_count:
            errors.append("MOV cannot preserve the selected attachments; choose MKV or turn off attachment copy.")
        incompatible_subtitles = [
            stream.codec_name
            for stream in media.streams_of_type("subtitle")
            if stream.codec_name not in MOV_SUBTITLE_CODECS
        ]
        if settings.copy_subtitles and incompatible_subtitles:
            errors.append(
                "MOV cannot preserve or convert these subtitle codecs: "
                + ", ".join(sorted(set(incompatible_subtitles)))
                + ". Choose MKV or omit subtitles."
            )
        incompatible_audio = [
            stream.codec_name
            for stream in media.streams_of_type("audio")
            if stream.codec_name not in MOV_AUDIO_CODECS
        ]
        if settings.copy_audio and incompatible_audio:
            errors.append(
                "MOV cannot reliably direct-copy these audio codecs: "
                + ", ".join(sorted(set(incompatible_audio)))
                + ". Choose MKV or omit audio."
            )
        if settings.copy_data and media.data_count:
            errors.append("MOV data-stream preservation is not guaranteed; choose MKV or omit data streams.")

    completed_artifacts = tuple(
        path
        for path in (output, log_path, report_path, script_path)
        if path is not None and path.exists()
    )
    if completed_artifacts:
        if settings.overwrite_approved:
            warnings.append(
                "Existing completed output/sidecar artifacts are approved for safe replacement only after the new partial validates."
            )
        else:
            warnings.append("Completed output/sidecar artifacts already exist; Start will require explicit replacement confirmation.")

    if media and analysis:
        analysis_summary = f"{analysis.mode.title()} IDet: {analysis.classification}; {analysis.rationale}"
        selected_backend, backend_warnings = _resolve_backend(settings, analysis, capabilities)
        warnings.extend(backend_warnings)
        selected_field_order, field_warnings = _resolve_field_order(settings, media, analysis)
        warnings.extend(field_warnings)
        requested_vapoursynth_denoiser = bool(
            settings.denoise_enabled
            and settings.denoiser in {spec.identifier for spec in DENOISER_SPECS}
            and denoiser_is_vapoursynth(settings.denoiser)
        )

        if health_is_current and source_health:
            analysis_summary += f" Fast source-health precheck: {source_health.status}; {source_health.reason}"
            if source_health.repair_required:
                if selected_backend == "vapoursynth_qtgmc" or (
                    selected_backend == "progressive" and requested_vapoursynth_denoiser
                ):
                    errors.append(
                        source_health.reason
                        + " The selected VapourSynth constant-rate graph is blocked before its expensive decoded "
                        "preflight or output encode. Click Repair required… to create a validated separate copy, "
                        "use a clean replacement, or choose an FFmpeg denoiser for a timestamp-aware direct path."
                    )
                else:
                    warnings.append(
                        source_health.reason
                        + " This non-QTGMC backend processes the original timestamp-aware stream directly without "
                        "automatic repair. It is not a repair and cannot restore missing/corrupt pictures; "
                        "inspect motion and audio/video sync around every reported damaged interval."
                    )
            elif source_health.status == "warning":
                warnings.append(
                    source_health.reason
                    + " QTGMC will use the managed full decoded preflight with live progress before encoding."
                )
            elif source_health.status == "inconclusive":
                warnings.append(
                    source_health.reason
                    + " The managed, cancellable full decoded preflight remains mandatory before QTGMC encoding."
                )

        if selected_backend == "unresolved":
            errors.append("IDet evidence is mixed or insufficient. Select a backend and field order explicitly after inspection.")
        if selected_backend == "progressive" and analysis.classification != "progressive":
            errors.append("Progressive passthrough cannot be used on measured interlaced/mixed material.")
        if selected_backend != "progressive" and analysis.classification == "progressive" and not settings.allow_progressive_override:
            errors.append(
                "Analysis classifies the input as progressive. Deinterlacing is blocked to prevent detail loss; "
                "enable the deliberate progressive override only after visual evidence of combing."
            )
        if selected_backend in {"ffmpeg_bwdif", "ffmpeg_bwdif_cuda"} and "bwdif" not in capabilities.filters:
            errors.append("The selected FFmpeg build does not provide the BWDIF filter.")
        if selected_backend == "ffmpeg_bwdif_cuda":
            if "bwdif_cuda" not in capabilities.filters or "cuda" not in capabilities.hwaccels:
                errors.append("CUDA BWDIF is unavailable in the selected FFmpeg build.")
        if selected_backend == "vapoursynth_qtgmc":
            if not capabilities.vspipe_path:
                errors.append("VSPipe was not found.")
            if not capabilities.qtgmc_ready:
                errors.append("QTGMC dependency check failed: " + capabilities.qtgmc_diagnostic)
            if settings.vulkan_nnedi3 and not capabilities.vulkan_nnedi3_ready:
                errors.append(
                    "Vulkan NNEDI3 was selected but its bounded QTGMC graph did not pass: "
                    + capabilities.vulkan_nnedi3_diagnostic
                )
        elif settings.vulkan_nnedi3:
            errors.append("Vulkan NNEDI3 interpolation applies only to the VapourSynth QTGMC backend.")
        if selected_backend != "progressive" and not selected_field_order:
            errors.append("Field order is unresolved. Select TFF or BFF explicitly.")

        uses_vapoursynth_denoiser = False
        if settings.denoise_enabled and settings.denoiser in capabilities.denoise_capabilities:
            selected_denoiser = settings.denoiser
            uses_vapoursynth_denoiser = denoiser_is_vapoursynth(settings.denoiser)
            if not capabilities.denoise_capabilities.get(settings.denoiser, False):
                diagnostic = capabilities.denoise_diagnostics.get(settings.denoiser, "capability check failed")
                errors.append(
                    f"The selected temporal denoiser is unavailable: {denoiser_spec(settings.denoiser).label}. "
                    f"Dependency Doctor reports: {diagnostic}"
                )
            else:
                selected_denoise_backend = resolve_denoiser_backend(
                    settings.denoiser,
                    capabilities.denoise_backends.get(settings.denoiser),
                    media.video.width or width,
                    media.video.height or height,
                )
                if not selected_denoise_backend:
                    errors.append("The selected temporal denoiser passed no named implementation/backend check.")
                else:
                    window = denoiser_frame_window(settings.denoiser, settings.denoise_temporal_radius)
                    warnings.append(
                        f"Temporal denoise is enabled after deinterlacing: {denoiser_spec(settings.denoiser).label}; "
                        f"resolved implementation: {denoiser_backend_display(settings.denoiser, selected_denoise_backend)}; "
                        f"strength {settings.denoise_strength}/10; {window}-frame temporal window."
                    )
                    warnings.append(
                        "Temporal denoising can remove real film grain and fine texture. Inspect representative motion "
                        "and low-light scenes before committing this shared setting to a library-wide batch."
                    )
                    if settings.denoiser == "ffmpeg_fftdnoiz" and settings.denoise_temporal_radius != 1:
                        warnings.append(
                            "FFmpeg fftdnoiz has a fixed temporal radius of one frame on each side; the radius control "
                            "does not expand its three-frame window."
                        )
                    if (
                        settings.denoiser == "vs_dfttest"
                        and selected_denoise_backend == "dfttest_cpu"
                        and capabilities.gpu_name
                    ):
                        warnings.append(
                            "DFTTest2 is using its verified optimized CPU graph because the faster optional "
                            "NVIDIA NVRTC graph is not installed or did not pass. Tools → Install/update "
                            "app-local dependencies can add it without changing system Python or PATH."
                        )
        elif settings.denoise_enabled and settings.denoiser in {spec.identifier for spec in DENOISER_SPECS}:
            selected_denoiser = settings.denoiser
            errors.append(
                f"The selected temporal denoiser was not capability-scanned: {denoiser_spec(settings.denoiser).label}. "
                "Refresh the capability scan or install/update the app-local tools in Dependency Doctor."
            )

        if uses_vapoursynth_denoiser and selected_backend in {"ffmpeg_bwdif", "ffmpeg_bwdif_cuda"}:
            errors.append(
                "A VapourSynth temporal denoiser cannot follow FFmpeg BWDIF in the same quality-preserving pipeline. "
                "Choose an FFmpeg denoiser, or choose VapourSynth QTGMC. The app will not denoise interlaced fields "
                "before BWDIF, silently change the deinterlacer, or create an unrequested intermediate."
            )
        uses_vspipe = selected_backend == "vapoursynth_qtgmc" or (
            selected_backend == "progressive" and uses_vapoursynth_denoiser
        )
        if uses_vspipe and not capabilities.vspipe_path:
            errors.append("VSPipe was not found for the selected VapourSynth processing graph.")
        if uses_vspipe and settings.hardware_decode == "auto":
            warnings.append(
                "Automatic hardware decode resolved to BestSource's verified software decode for this "
                "VapourSynth graph; the shared preference remains Automatic for other batch rows."
            )
        elif uses_vspipe and settings.hardware_decode != "off":
            errors.append(
                "Explicit CUDA decode applies only to direct FFmpeg paths; choose Automatic or Off so "
                "BestSource can control decoding in the selected VapourSynth graph."
            )

        if settings.hardware_decode == "cuda" and "cuda" not in capabilities.hwaccels:
            errors.append("CUDA hardware decoding is not exposed by the selected FFmpeg build.")

        try:
            width, height, sar, dar, geometry_note = _resolve_geometry(settings, media)
            for label, ratio in (("SAR", sar), ("DAR", dar)):
                if max(abs(ratio.numerator), ratio.denominator) > 2_147_483_647:
                    raise RationalError(f"{label} exceeds FFmpeg's supported integer range")
            if settings.aspect_mode == "square" and (width, height) != media.video.dimensions:
                warnings.append(f"Exact square-pixel DAR requires {width}x{height}; high-quality scaling will be applied.")
            if settings.aspect_mode == "manual":
                warnings.append("Manual DAR changes display geometry metadata; it does not crop the stored picture.")
        except (ValueError, RationalError) as exc:
            errors.append(str(exc))
            width = media.video.width or 0
            height = media.video.height or 0
            sar = Fraction(1, 1)
            dar = Fraction(width or 1, height or 1)
            geometry_note = "Invalid geometry"

        input_rate = media.video.avg_frame_rate or media.video.r_frame_rate
        output_rate = input_rate
        if selected_backend != "progressive" and settings.output_cadence == "field_rate" and input_rate:
            output_rate = input_rate * 2

        if profile:
            mov_output = output.suffix.lower() == ".mov"
            if mov_output:
                converted_subtitles = [
                    stream.codec_name
                    for stream in media.streams_of_type("subtitle")
                    if settings.copy_subtitles
                    and stream.codec_name.casefold() in MOV_TEXT_CONVERTIBLE_SUBTITLE_CODECS
                ]
                if converted_subtitles:
                    warnings.append(
                        "MOV stores the selected text subtitle track(s) as native mov_text instead of Matroska "
                        f"{', '.join(sorted(set(converted_subtitles)))}. Video and audio handling are unchanged."
                    )
                language_aliases = sorted(
                    {
                        (stream.tags["language"], _mov_language(stream.tags["language"]))
                        for codec_type, enabled in (
                            ("audio", settings.copy_audio),
                            ("subtitle", settings.copy_subtitles),
                            ("attachment", settings.copy_attachments),
                            ("data", settings.copy_data),
                        )
                        if enabled
                        for stream in media.streams_of_type(codec_type)
                        if stream.tags.get("language")
                        and _mov_language(stream.tags["language"]) != stream.tags["language"]
                    }
                )
                if language_aliases:
                    rendered_aliases = ", ".join(
                        f"{source!r}→{target!r}" for source, target in language_aliases
                    )
                    warnings.append(
                        "MOV/QuickTime cannot retain these extended language codes directly; the editor master "
                        f"uses compatible legacy codes: {rendered_aliases}. The source remains unchanged."
                    )
            if profile.lossless:
                warnings.append("FFV1 Intra 16-bit is lossless but can require several times the source storage.")
                warnings.append(
                    "FFV1 is an archival master, not the recommended DaVinci Resolve interchange format. "
                    "High-bit-depth FFV1 variants can appear Media Offline in Resolve; use the GUI's Resolve "
                    "editor preset for a DNxHR 444 10-bit MOV while retaining the FFV1 file as an archive."
                )
                if settings.ffv1_chroma_mode == "native":
                    warnings.append(
                        f"Native-chroma FFV1 preserves the graph's {profile.chroma} chroma sampling at 16-bit; "
                        "it avoids derived chroma upsampling and is the recommended archival choice for this source."
                    )
                else:
                    warnings.append(
                        "Explicit FFV1 4:4:4 mastering upsamples lower-subsampled chroma for compositing workflows; "
                        "it does not create additional source chroma detail and substantially increases storage."
                    )
            elif profile.encoder in {"libaom-av1", "libx265"}:
                warnings.append("The selected software quality preset is intentionally extremely slow.")
            elif profile.hardware:
                warnings.append("Hardware encoding is faster, but software encoders usually compress more efficiently at equal visual quality.")
            if profile.id == "prores_4444_xq":
                warnings.append(
                    "FFmpeg prores_ks accepts 10-bit 4:4:4 input for 4444 XQ; selecting XQ does not create missing 12-bit source precision."
                )
            if profile.bit_depth > (media.video.bits_per_raw_sample or 8):
                warnings.append(
                    f"The {profile.bit_depth}-bit pipeline prevents additional rounding but cannot recover precision absent from the "
                    f"{media.video.bits_per_raw_sample or 8}-bit source."
                )

            expected = OutputExpectation(
                codec_names=profile.codec_names,
                pix_fmts=(profile.pix_fmt,),
                width=width,
                height=height,
                sar=sar,
                dar=dar,
                frame_rate=output_rate,
                progressive=True,
                lossless=profile.lossless,
                bit_depth=profile.bit_depth,
                expected_audio=_expected_tracks(settings, media, "audio", mov=mov_output),
                expected_subtitles=_expected_tracks(settings, media, "subtitle", mov=mov_output),
                expected_attachments=_expected_tracks(settings, media, "attachment", mov=mov_output),
                duration=media.duration,
                frame_count=(
                    media.video.nb_frames * (2 if selected_backend != "progressive" and settings.output_cadence == "field_rate" else 1)
                    if media.video.nb_frames is not None
                    else None
                ),
                expected_data=_expected_tracks(settings, media, "data", mov=mov_output),
                expected_chapter_count=len(media.chapters) if settings.copy_chapters else 0,
                expected_format_tags=(
                    {
                        key: value
                        for key, value in media.format_tags.items()
                        if key.upper() not in {"ENCODER", "DURATION"}
                    }
                    if settings.copy_metadata
                    else {}
                ),
                color_range=media.video.color_range,
                color_space=media.video.color_space,
                color_transfer=media.video.color_transfer,
                color_primaries=media.video.color_primaries,
            )
            warnings.append(geometry_note + ".")

            denoise_plan_ready = not settings.denoise_enabled or (
                selected_denoiser is not None and selected_denoise_backend is not None
                and not validate_denoise_numbers(
                    settings.denoise_strength,
                    settings.denoise_temporal_radius,
                )
            )
            if capabilities.ffmpeg_path and partial and denoise_plan_ready:
                command = [str(capabilities.ffmpeg_path), "-hide_banner", "-nostdin", "-y"]
                source_index = 0
                uses_vapoursynth_denoiser = bool(
                    settings.denoise_enabled
                    and selected_denoiser
                    and denoiser_is_vapoursynth(selected_denoiser)
                )
                uses_vspipe = selected_backend == "vapoursynth_qtgmc" or (
                    selected_backend == "progressive" and uses_vapoursynth_denoiser
                )
                if uses_vspipe:
                    assert capabilities.vspipe_path and temp_script
                    schedule = choose_vapoursynth_schedule(
                        width,
                        height,
                        media.video.pix_fmt,
                        temporal_denoise=uses_vapoursynth_denoiser,
                        vulkan_nnedi3=bool(settings.vulkan_nnedi3 and selected_backend == "vapoursynth_qtgmc"),
                    )
                    warnings.append(schedule.rationale)
                    if settings.vulkan_nnedi3:
                        warnings.append(
                            "Vulkan NNEDI3 accelerates QTGMC's spatial interpolation only; MVTools motion analysis "
                            "and degrain remain CPU work. CPU NNEDI3 remains the default maximum-fidelity path."
                        )
                    script = _vapoursynth_script(
                        settings,
                        media,
                        selected_backend or "progressive",
                        selected_field_order,
                        selected_denoise_backend,
                        width,
                        height,
                        sar,
                        schedule,
                    )
                    vspipe_command = [
                        str(_execution_vspipe_path(capabilities.vspipe_path)),
                        "--requests",
                        str(schedule.requests),
                        "--container",
                        "y4m",
                        "--progress",
                        str(temp_script),
                        "-",
                    ]
                    command += ["-f", "yuv4mpegpipe", "-i", "pipe:0", "-i", str(media.path)]
                    source_index = 1
                    post_filters: list[str] = []
                    setparams = _setparams_filter(media)
                    if setparams:
                        post_filters.append(setparams)
                    if settings.denoise_enabled and not uses_vapoursynth_denoiser:
                        post_filters.append(
                            ffmpeg_denoise_filter(
                                settings.denoiser,
                                settings.denoise_strength,
                                settings.denoise_temporal_radius,
                            )
                        )
                    post_filters += [_setsar_filter(sar), f"format={profile.pix_fmt}"]
                    video_filters = ",".join(post_filters)
                else:
                    input_args, download_frames = _ffmpeg_input_args(
                        settings,
                        selected_backend or "progressive",
                        media,
                    )
                    command += input_args + ["-i", str(media.path)]
                    video_filters = _ffmpeg_video_filters(
                        settings,
                        selected_backend or "progressive",
                        selected_field_order,
                        profile,
                        media,
                        width,
                        height,
                        sar,
                        download_frames,
                    )
                maps, stream_codecs = _track_mapping(
                    settings,
                    source_index,
                    media=media,
                    mov=output.suffix.lower() == ".mov",
                )
                command += maps
                if video_filters:
                    command += ["-vf", video_filters]
                command += profile.encoder_args(settings.quality, settings.tune_grain)
                command += stream_codecs
                command += _metadata_args(settings, source_index)
                command += _copied_stream_metadata_args(
                    settings,
                    media,
                    mov=output.suffix.lower() == ".mov",
                )
                command += _color_args(media)
                command += [
                    "-fps_mode",
                    "passthrough",
                    "-max_muxing_queue_size",
                    "4096",
                ]
                if output.suffix.lower() == ".mov":
                    command += ["-movflags", "+write_colr+use_metadata_tags"]
                command += ["-progress", "pipe:1", "-nostats", str(partial)]
                ffmpeg_command = command
                if vspipe_command:
                    display_command = subprocess.list2cmdline(vspipe_command) + " | " + subprocess.list2cmdline(command)
                else:
                    display_command = subprocess.list2cmdline(command)

    return ProcessingPlan(
        valid=not errors,
        errors=tuple(errors),
        warnings=tuple(dict.fromkeys(warnings)),
        settings=settings,
        profile_id=profile.id if profile else None,
        profile_label=profile.label if profile else None,
        selected_backend=selected_backend,
        selected_field_order=selected_field_order,
        selected_denoiser=selected_denoiser,
        selected_denoise_backend=selected_denoise_backend,
        output_path=output,
        partial_path=partial,
        log_path=log_path,
        report_path=report_path,
        script_path=script_path if vspipe_command else None,
        temporary_script_path=temp_script if vspipe_command else None,
        ffmpeg_command=tuple(ffmpeg_command),
        vspipe_command=tuple(vspipe_command) if vspipe_command else None,
        vapoursynth_script=script,
        display_command=display_command,
        expected=expected,
        analysis_summary=analysis_summary,
        source_health=source_health if health_is_current else None,
        automatic_recovery=automatic_recovery,
        vapoursynth_threads=schedule.core_threads if schedule else None,
        vspipe_requests=schedule.requests if schedule else None,
        vapoursynth_schedule_note=(
            f"{schedule.rationale} Applied schedule: "
            f"core threads={schedule.core_threads}; VSPipe requests={schedule.requests}."
            if schedule
            else None
        ),
        vulkan_nnedi3_active=bool(schedule and settings.vulkan_nnedi3),
    )
