from __future__ import annotations

import os
import subprocess
import uuid
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

from deinterlace_studio.denoise import (
    DENOISER_SPECS,
    denoiser_backend_display,
    denoiser_is_vapoursynth,
    denoiser_spec,
    ffmpeg_denoise_filter,
    resolve_denoiser_backend,
    vapoursynth_import_lines,
)
from deinterlace_studio.models import CapabilityReport, MediaProbe, OutputExpectation, StreamInfo
from deinterlace_studio.presets import OutputProfile, profile_capability_error
from deinterlace_studio.rationals import derive_dar
from deinterlace_studio.scheduling import choose_vapoursynth_schedule

from . import __version__
from .denoiser_policy import denoiser_control_policy, validate_denoiser_controls
from .models import DenoisePlan, DenoiseSettings
from .output_policy import (
    ISO_TEXT_CONVERTIBLE_SUBTITLE_CODECS,
    container_compatibility_errors,
    encoder_args as output_encoder_args,
    quality_validation_error,
    select_output_profile,
)
from .vapoursynth_fields import video_denoise_lines


MOV_TEXT_CONVERTIBLE_SUBTITLE_CODECS = ISO_TEXT_CONVERTIBLE_SUBTITLE_CODECS
MOV_LANGUAGE_ALIASES = {"zh": "chi", "zho": "chi", "cmn": "chi", "yue": "chi"}
INTERLACED_FIELD_ORDERS = {
    "tt": "tff",
    "tb": "tff",
    "tff": "tff",
    "bb": "bff",
    "bt": "bff",
    "bff": "bff",
}


def same_path(left: Path, right: Path) -> bool:
    return os.path.normcase(os.path.abspath(left)) == os.path.normcase(os.path.abspath(right))


def sidecar(output: Path, suffix: str) -> Path:
    return output.with_name(output.name + suffix)


def execution_vspipe_path(configured: Path) -> Path:
    if configured.parent.name.casefold() == "scripts":
        native = configured.parent.parent / "Lib" / "site-packages" / "vapoursynth" / "vspipe.exe"
        if native.is_file():
            return native
    return configured


def source_geometry(media: MediaProbe) -> tuple[int, int, Fraction, Fraction]:
    video = media.video
    if not video.width or not video.height:
        raise ValueError("The source video has no usable stored dimensions.")
    sar = video.sample_aspect_ratio
    dar = video.display_aspect_ratio
    if sar is None and dar is not None:
        sar = dar / Fraction(video.width, video.height)
    sar = sar or Fraction(1, 1)
    dar = dar or derive_dar(video.width, video.height, sar)
    return video.width, video.height, sar, dar


def source_field_order(media: MediaProbe) -> str | None:
    return INTERLACED_FIELD_ORDERS.get((media.video.field_order or "").casefold())


def source_is_interlaced(media: MediaProbe) -> bool:
    field = (media.video.field_order or "").casefold()
    return bool(
        media.sampled_interlaced_frames
        or field in INTERLACED_FIELD_ORDERS
        or (field and field not in {"progressive", "unknown", "unspecified"})
    )


def default_output_path(source: Path, profile: OutputProfile) -> Path:
    return source.with_name(f"{source.stem}.denoised{profile.default_extension}")


def unique_output_path(preferred: Path, reserved: tuple[Path, ...] = ()) -> Path:
    reserved_keys = {os.path.normcase(os.path.abspath(path)) for path in reserved}
    candidate = preferred
    counter = 2
    while candidate.exists() or os.path.normcase(os.path.abspath(candidate)) in reserved_keys:
        candidate = preferred.with_name(f"{preferred.stem}-{counter}{preferred.suffix}")
        counter += 1
    return candidate


def _setsar_filter(sar: Fraction) -> str:
    maximum = max(abs(sar.numerator), sar.denominator, 100)
    if maximum > 2_147_483_647:
        raise ValueError("The source sample-aspect ratio exceeds FFmpeg's supported integer range.")
    return f"setsar=sar={sar.numerator}/{sar.denominator}:max={maximum}"


def _setparams_filter(media: MediaProbe) -> str | None:
    video = media.video
    range_value = {"pc": "full", "jpeg": "full", "tv": "limited", "mpeg": "limited"}.get(
        (video.color_range or "").casefold(), video.color_range
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


def _color_args(media: MediaProbe) -> list[str]:
    args: list[str] = []
    for option, value in (
        ("-color_range", media.video.color_range),
        ("-colorspace", media.video.color_space),
        ("-color_trc", media.video.color_transfer),
        ("-color_primaries", media.video.color_primaries),
    ):
        if value and value not in {"unknown", "reserved", "unspecified"}:
            args += [option, value]
    return args


def _mov_language(value: str) -> str:
    normalized = value.strip().casefold().replace("_", "-")
    base = normalized.split("-", 1)[0]
    return MOV_LANGUAGE_ALIASES.get(normalized, MOV_LANGUAGE_ALIASES.get(base, value))


def _track_mapping(
    settings: DenoiseSettings,
    source_index: int,
    media: MediaProbe,
    *,
    mov: bool,
) -> tuple[list[str], list[str]]:
    maps: list[str] = ["-map", "0:v:0"]
    codecs: list[str] = []
    if settings.copy_audio:
        maps += ["-map", f"{source_index}:a?"]
        codecs += ["-c:a", "copy"]
    if settings.copy_subtitles:
        maps += ["-map", f"{source_index}:s?"]
        if mov:
            for position, stream in enumerate(media.streams_of_type("subtitle")):
                codec = "mov_text" if stream.codec_name.casefold() in MOV_TEXT_CONVERTIBLE_SUBTITLE_CODECS else "copy"
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


def _metadata_args(settings: DenoiseSettings, source_index: int) -> list[str]:
    return [
        "-map_metadata",
        str(source_index) if settings.copy_metadata else "-1",
        "-map_chapters",
        str(source_index) if settings.copy_chapters else "-1",
    ]


def _copied_stream_metadata_args(settings: DenoiseSettings, media: MediaProbe, *, mov: bool) -> list[str]:
    args: list[str] = []
    for codec_type, specifier, enabled in (
        ("audio", "a", settings.copy_audio),
        ("subtitle", "s", settings.copy_subtitles),
        ("attachment", "t", settings.copy_attachments),
        ("data", "d", settings.copy_data),
    ):
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


def _expected_tracks(
    settings: DenoiseSettings,
    media: MediaProbe,
    codec_type: str,
    *,
    mov: bool,
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
            if codec_type == "subtitle" and stream.codec_name.casefold() in MOV_TEXT_CONVERTIBLE_SUBTITLE_CODECS
            else stream.codec_name
        )
        normalized.append(replace(stream, codec_name=codec_name, tags=tags))
    return tuple(normalized)


def _vapoursynth_script(
    settings: DenoiseSettings,
    media: MediaProbe,
    denoise_backend: str,
    schedule_threads: int,
    field_order: str | None,
) -> str:
    imports = [
        "import os",
        "import tempfile",
        "import vapoursynth as vs",
        "from vapoursynth import core",
        "from vstools import depth",
        *vapoursynth_import_lines(settings.denoiser),
    ]
    lines = [
        f"# Generated by Video Denoise Studio {__version__}; do not edit while a job is running.",
        *imports,
        "",
        f"core.num_threads = min({schedule_threads}, max(1, core.num_threads))",
        f"SOURCE = {str(media.path)!r}",
        "CACHE_ROOT = os.path.join(tempfile.gettempdir(), 'Video Denoise Studio BestSource Cache')",
        "os.makedirs(CACHE_ROOT, exist_ok=True)",
        "clip = core.bs.VideoSource(source=SOURCE, cachemode=1, cachepath=CACHE_ROOT)",
        "if clip.format is None:",
        "    raise RuntimeError('BestSource returned a variable-format clip; normalize the source before processing')",
        "clip = depth(clip, 16)",
    ]
    if field_order:
        lines.append(f"clip = core.std.SetFieldBased(clip, value={2 if field_order == 'tff' else 1})")
    field_note = (
        "# Preserve interlacing: DFTTest2 temporarily separates same-parity fields and reweaves them; no deinterlacing occurs."
        if settings.denoiser == "vs_dfttest" and field_order
        else "# Denoise stored frames only; no field separation or deinterlacing is performed."
    )
    lines += [
        field_note,
        *video_denoise_lines(
            settings.denoiser,
            settings.denoise_strength,
            settings.denoise_temporal_radius,
            denoise_backend,
            field_order=field_order,
        ),
        "clip.set_output()",
        "",
    ]
    return "\n".join(lines)


def build_plan(
    settings: DenoiseSettings,
    media: MediaProbe | None,
    capabilities: CapabilityReport,
    *,
    run_id: str | None = None,
) -> DenoisePlan:
    errors: list[str] = []
    warnings: list[str] = []
    profile: OutputProfile | None = None
    container: str | None = None
    container_reason: str | None = None
    selected_backend: str | None = None
    expected: OutputExpectation | None = None
    schedule = None
    script: str | None = None
    ffmpeg_command: list[str] = []
    vspipe_command: list[str] | None = None
    display_command = ""
    output = settings.output_path
    token = run_id or uuid.uuid4().hex[:12]
    partial = output.with_name(f".{output.stem}.partial.{token}{output.suffix}") if output.name else None
    log_path = sidecar(output, ".Denoise.log") if output.name else None
    report_path = sidecar(output, ".Denoise.json") if output.name else None
    script_path = sidecar(output, ".Denoise.vpy") if output.name else None
    temp_script = output.with_name(f".{output.stem}.Denoise.{token}.vpy") if output.name else None

    if not settings.input_path.is_file():
        errors.append(f"Source video does not exist: {settings.input_path}")
    if not output.name:
        errors.append("An output filename is required.")
    elif same_path(settings.input_path, output):
        errors.append("The output path must not be the source path.")
    elif output.exists():
        errors.append("The output already exists. Choose a new filename; existing media is never silently overwritten.")
    elif not output.parent.is_dir():
        errors.append(f"Output directory does not exist: {output.parent}")
    if partial and partial.exists():
        errors.append(f"Unique partial path unexpectedly exists: {partial}")
    for artifact, label in (
        (log_path, "run log"),
        (report_path, "JSON report"),
        (script_path, "VapourSynth script"),
    ):
        if artifact and artifact.exists():
            errors.append(f"The planned {label} already exists; choose a new output filename: {artifact}")
    if not capabilities.ffmpeg_path or not capabilities.ffprobe_path:
        errors.append("FFmpeg and FFprobe must both pass capability discovery.")

    errors.extend(validate_denoiser_controls(settings.denoiser, settings.denoise_strength, settings.denoise_temporal_radius))
    try:
        denoiser_spec(settings.denoiser)
    except ValueError as exc:
        errors.append(str(exc))
    else:
        if not capabilities.denoise_capabilities.get(settings.denoiser, False):
            diagnostic = capabilities.denoise_diagnostics.get(settings.denoiser, "capability check failed")
            errors.append(f"The selected temporal denoiser is unavailable: {diagnostic}")

    if media is not None:
        try:
            width, height, sar, dar = source_geometry(media)
        except ValueError as exc:
            errors.append(str(exc))
            width = height = 0
            sar = dar = Fraction(1, 1)
        try:
            profile, container_resolution = select_output_profile(settings, media)
            container = container_resolution.identifier
            container_reason = container_resolution.reason
        except ValueError as exc:
            errors.append(str(exc))
        if profile:
            capability_error = profile_capability_error(profile, capabilities)
            if capability_error:
                errors.append(capability_error)
            if output.suffix.casefold() != profile.default_extension:
                errors.append(
                    f"{profile.label} requires a {profile.default_extension} output filename; received {output.suffix or 'no extension'}."
                )
            quality_error = quality_validation_error(profile, settings.quality)
            if quality_error:
                errors.append(quality_error)

        uses_vs = denoiser_is_vapoursynth(settings.denoiser) if settings.denoiser in {s.identifier for s in DENOISER_SPECS} else False
        if uses_vs and not capabilities.vspipe_path:
            errors.append("VSPipe is required for the selected VapourSynth denoiser.")
        if capabilities.denoise_capabilities.get(settings.denoiser, False):
            selected_backend = resolve_denoiser_backend(
                settings.denoiser,
                capabilities.denoise_backends.get(settings.denoiser),
                width,
                height,
            )
            if not selected_backend:
                errors.append("The selected denoiser passed no named backend capability check.")

        interlaced = source_is_interlaced(media)
        field_order = source_field_order(media)
        if interlaced and not field_order:
            errors.append(
                "The source contains interlaced frames but its TFF/BFF order is not provable. "
                "Denoise-only processing will not guess or silently change field order."
            )
        if interlaced:
            if settings.denoiser == "vs_dfttest":
                warnings.append(
                    "Denoise-only DFTTest2 preserves every stored interlaced frame and its field order. Internally, "
                    "the proven TFF/BFF parities are filtered as two independent progressive sequences and rewoven; "
                    "this is not deinterlacing and does not change cadence."
                )
            else:
                warnings.append(
                    "Denoise-only mode preserves the stored interlaced frames and field order. It does not separate fields, "
                    "deinterlace, or change cadence; temporal filtering operates on stored frames."
                )
            if profile and profile.encoder in {"libaom-av1", "libsvtav1", "av1_nvenc"}:
                errors.append("AV1 does not preserve an interlaced coding contract; choose FFV1, ProRes, DNxHR, or software HEVC.")
            if profile and profile.encoder == "hevc_nvenc":
                errors.append("The active HEVC NVENC path has no verified interlaced-field contract; use software HEVC or an intra master.")

        iso_bmff = container in {"mp4", "mov"}
        if container:
            errors.extend(container_compatibility_errors(settings, media, container))

        if profile:
            input_rate = media.video.avg_frame_rate or media.video.r_frame_rate
            expected = OutputExpectation(
                codec_names=profile.codec_names,
                pix_fmts=(profile.pix_fmt,),
                width=width,
                height=height,
                sar=sar,
                dar=dar,
                frame_rate=input_rate,
                progressive=not interlaced,
                lossless=profile.lossless,
                bit_depth=profile.bit_depth,
                expected_audio=_expected_tracks(settings, media, "audio", mov=iso_bmff),
                expected_subtitles=_expected_tracks(settings, media, "subtitle", mov=iso_bmff),
                expected_attachments=_expected_tracks(settings, media, "attachment", mov=iso_bmff),
                duration=media.duration,
                frame_count=media.video.nb_frames,
                expected_data=_expected_tracks(settings, media, "data", mov=iso_bmff),
                expected_chapter_count=len(media.chapters) if settings.copy_chapters else 0,
                expected_format_tags=(
                    {key: value for key, value in media.format_tags.items() if key.upper() not in {"ENCODER", "DURATION"}}
                    if settings.copy_metadata
                    else {}
                ),
                color_range=media.video.color_range,
                color_space=media.video.color_space,
                color_transfer=media.video.color_transfer,
                color_primaries=media.video.color_primaries,
            )
            if profile.lossless:
                warnings.append("FFV1 is mathematically lossless after denoising but can be several times larger than the source.")
            elif profile.encoder in {"libaom-av1", "libx265"}:
                warnings.append("The selected software quality profile is intentionally very slow.")
            if profile.bit_depth > (media.video.bits_per_raw_sample or 8):
                warnings.append(
                    f"The {profile.bit_depth}-bit pipeline prevents additional rounding but cannot restore precision absent from "
                    f"the {media.video.bits_per_raw_sample or 8}-bit source."
                )
            control_policy = denoiser_control_policy(
                settings.denoiser,
                settings.denoise_strength,
                settings.denoise_temporal_radius,
            )
            window = control_policy.window_frames
            warnings.append(
                f"Denoiser: {denoiser_spec(settings.denoiser).label}; backend: "
                f"{denoiser_backend_display(settings.denoiser, selected_backend)}; strength "
                f"{settings.denoise_strength}/10; exact {window}-frame temporal window."
            )
            if container_reason:
                warnings.append(container_reason)

        ready = bool(
            profile
            and expected
            and selected_backend
            and capabilities.ffmpeg_path
            and not errors
            and partial
        )
        if ready:
            command = [str(capabilities.ffmpeg_path), "-hide_banner", "-loglevel", "verbose", "-nostdin", "-y"]
            source_index = 0
            if uses_vs:
                assert capabilities.vspipe_path and temp_script
                schedule = choose_vapoursynth_schedule(
                    width,
                    height,
                    media.video.pix_fmt,
                    temporal_denoise=True,
                )
                warnings.append(schedule.rationale)
                script = _vapoursynth_script(settings, media, selected_backend, schedule.core_threads, field_order)
                vspipe_command = [
                    str(execution_vspipe_path(capabilities.vspipe_path)),
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
                filters: list[str] = []
            else:
                command += ["-i", str(media.path)]
                filters = [
                    ffmpeg_denoise_filter(
                        settings.denoiser,
                        settings.denoise_strength,
                        settings.denoise_temporal_radius,
                    )
                ]
            setparams = _setparams_filter(media)
            if setparams:
                filters.insert(0, setparams)
            filters.append(_setsar_filter(sar))
            filters.append(f"setfield=mode={field_order if interlaced else 'prog'}")
            filters.append(f"format={profile.pix_fmt}")
            maps, stream_codecs = _track_mapping(settings, source_index, media, mov=iso_bmff)
            command += maps + ["-vf", ",".join(filters)]
            encoder_args = output_encoder_args(profile, settings.quality, settings.tune_grain)
            if interlaced:
                if profile.encoder == "libx265":
                    encoder_args += ["-x265-params", f"interlace={field_order}"]
                encoder_args += [
                    "-flags",
                    "+ildct+ilme",
                    "-field_order",
                    "tt" if field_order == "tff" else "bb",
                ]
            command += encoder_args + stream_codecs
            command += _metadata_args(settings, source_index)
            command += _copied_stream_metadata_args(settings, media, mov=iso_bmff)
            command += _color_args(media)
            command += ["-fps_mode", "passthrough", "-max_muxing_queue_size", "4096"]
            if container in {"mp4", "mov"}:
                if profile.encoder in {"hevc_nvenc", "libx265"}:
                    command += ["-tag:v", "hvc1"]
                elif profile.encoder in {"av1_nvenc", "libaom-av1", "libsvtav1"}:
                    command += ["-tag:v", "av01"]
                movflags = "+write_colr+use_metadata_tags"
                if container == "mp4":
                    movflags += "+faststart"
                command += ["-movflags", movflags]
            command += ["-progress", "pipe:1", "-nostats", str(partial)]
            ffmpeg_command = command
            display_command = (
                subprocess.list2cmdline(vspipe_command) + " | " + subprocess.list2cmdline(command)
                if vspipe_command
                else subprocess.list2cmdline(command)
            )

    return DenoisePlan(
        valid=not errors,
        errors=tuple(errors),
        warnings=tuple(dict.fromkeys(warnings)),
        settings=settings,
        media=media,
        ffprobe_path=capabilities.ffprobe_path,
        profile=profile,
        container=container,
        selected_denoise_backend=selected_backend,
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
        schedule=schedule,
    )
