from __future__ import annotations

import re
from dataclasses import dataclass

from .models import CapabilityReport


def nvenc_maximum_quality_args(quality: int) -> list[str]:
    """Return the single capability-tested NVENC quality contract used by the app.

    ``uhq`` is intentionally used instead of ``hq`` when the selected FFmpeg
    build exposes it: on current FFmpeg 9/NVIDIA SDK 13 builds it enables the
    encoder's Ultra High Quality tools, including internally managed lookahead
    and temporal filtering.  We therefore do not force an explicit lookahead
    count. Capability discovery runs a real bounded encode with this exact
    list, so an advertised option is not treated as sufficient proof.
    """

    return [
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
    ]


@dataclass(frozen=True)
class OutputProfile:
    id: str
    label: str
    family: str
    bit_depth: int
    encoder: str
    pix_fmt: str
    codec_names: tuple[str, ...]
    default_extension: str
    chroma: str
    lossless: bool
    intra_only: bool
    hardware: bool
    description: str

    def encoder_args(self, quality: int, tune_grain: bool) -> list[str]:
        if self.encoder == "libx265":
            args = [
                "-c:v",
                "libx265",
                "-preset",
                "placebo",
                "-crf",
                str(quality),
                "-pix_fmt",
                self.pix_fmt,
            ]
            if tune_grain:
                args += ["-tune", "grain"]
            return args
        if self.encoder == "hevc_nvenc":
            return ["-c:v", "hevc_nvenc", *nvenc_maximum_quality_args(quality), "-pix_fmt", self.pix_fmt]
        if self.encoder == "libaom-av1":
            return [
                "-c:v",
                "libaom-av1",
                "-usage",
                "good",
                "-cpu-used",
                "0",
                "-crf",
                str(quality),
                "-b:v",
                "0",
                "-row-mt",
                "1",
                "-tiles",
                "1x1",
                "-lag-in-frames",
                "48",
                "-aq-mode",
                "1",
                "-pix_fmt",
                self.pix_fmt,
            ]
        if self.encoder == "libsvtav1":
            return [
                "-c:v",
                "libsvtav1",
                "-preset",
                "0",
                "-crf",
                str(quality),
                "-pix_fmt",
                self.pix_fmt,
            ]
        if self.encoder == "av1_nvenc":
            return ["-c:v", "av1_nvenc", *nvenc_maximum_quality_args(quality), "-pix_fmt", self.pix_fmt]
        if self.encoder == "ffv1":
            return [
                "-c:v",
                "ffv1",
                "-level",
                "3",
                "-coder",
                "2",
                "-context",
                "1",
                "-g",
                "1",
                "-slicecrc",
                "1",
                "-slices",
                "16",
                "-pix_fmt",
                self.pix_fmt,
            ]
        if self.encoder == "prores_ks":
            return [
                "-c:v",
                "prores_ks",
                "-profile:v",
                "4444xq",
                "-vendor",
                "apl0",
                "-bits_per_mb",
                "8000",
                "-alpha_bits",
                "0",
                "-pix_fmt",
                self.pix_fmt,
            ]
        if self.encoder == "dnxhd":
            return [
                "-c:v",
                "dnxhd",
                "-profile:v",
                "dnxhr_444",
                "-pix_fmt",
                self.pix_fmt,
            ]
        raise ValueError(f"Unsupported encoder profile: {self.id}")


PROFILES: dict[str, OutputProfile] = {
    "hevc_x265_10": OutputProfile(
        "hevc_x265_10",
        "HEVC x265 10-bit — placebo / quality-first",
        "hevc",
        10,
        "libx265",
        "yuv420p10le",
        ("hevc", "h265"),
        ".mkv",
        "4:2:0",
        False,
        False,
        False,
        "Software x265 at its slowest preset; excellent fidelity and compression.",
    ),
    "hevc_x265_12": OutputProfile(
        "hevc_x265_12",
        "HEVC x265 12-bit — placebo / quality-first",
        "hevc",
        12,
        "libx265",
        "yuv420p12le",
        ("hevc", "h265"),
        ".mkv",
        "4:2:0",
        False,
        False,
        False,
        "Twelve-bit software HEVC; source precision is retained but not invented.",
    ),
    "hevc_nvenc_10": OutputProfile(
        "hevc_nvenc_10",
        "HEVC NVIDIA 10-bit — P7 UHQ full multipass",
        "hevc",
        10,
        "hevc_nvenc",
        "p010le",
        ("hevc", "h265"),
        ".mkv",
        "4:2:0",
        False,
        False,
        True,
        "Fast NVIDIA path at the highest exposed NVENC quality settings.",
    ),
    "hevc_nvenc_12": OutputProfile(
        "hevc_nvenc_12",
        "HEVC NVIDIA 12-bit — P7 UHQ full multipass",
        "hevc",
        12,
        "hevc_nvenc",
        "p012le",
        ("hevc", "h265"),
        ".mkv",
        "4:2:0",
        False,
        False,
        True,
        "Capability-gated 12-bit NVIDIA path for supported Blackwell drivers/builds.",
    ),
    "av1_libaom_10": OutputProfile(
        "av1_libaom_10",
        "AV1 libaom 10-bit — cpu-used 0",
        "av1",
        10,
        "libaom-av1",
        "yuv420p10le",
        ("av1",),
        ".mkv",
        "4:2:0",
        False,
        False,
        False,
        "Reference-quality software AV1; extremely slow.",
    ),
    "av1_libaom_12": OutputProfile(
        "av1_libaom_12",
        "AV1 libaom 12-bit — cpu-used 0",
        "av1",
        12,
        "libaom-av1",
        "yuv420p12le",
        ("av1",),
        ".mkv",
        "4:2:0",
        False,
        False,
        False,
        "Twelve-bit reference-quality AV1; exceptionally slow.",
    ),
    "av1_svt_10": OutputProfile(
        "av1_svt_10",
        "AV1 SVT-AV1 10-bit — preset 0",
        "av1",
        10,
        "libsvtav1",
        "yuv420p10le",
        ("av1",),
        ".mkv",
        "4:2:0",
        False,
        False,
        False,
        "Very slow high-quality SVT-AV1 software encode.",
    ),
    "av1_svt_12": OutputProfile(
        "av1_svt_12",
        "AV1 SVT-AV1 12-bit — preset 0",
        "av1",
        12,
        "libsvtav1",
        "yuv420p12le",
        ("av1",),
        ".mkv",
        "4:2:0",
        False,
        False,
        False,
        "Capability-gated twelve-bit SVT-AV1 software encode.",
    ),
    "av1_nvenc_10": OutputProfile(
        "av1_nvenc_10",
        "AV1 NVIDIA 10-bit — P7 UHQ full multipass",
        "av1",
        10,
        "av1_nvenc",
        "p010le",
        ("av1",),
        ".mkv",
        "4:2:0",
        False,
        False,
        True,
        "Fast NVIDIA AV1 path at the highest exposed NVENC quality settings.",
    ),
    "av1_nvenc_12": OutputProfile(
        "av1_nvenc_12",
        "AV1 NVIDIA 12-bit — P7 UHQ full multipass",
        "av1",
        12,
        "av1_nvenc",
        "p012le",
        ("av1",),
        ".mkv",
        "4:2:0",
        False,
        False,
        True,
        "Capability-gated 12-bit NVIDIA AV1 path.",
    ),
    "ffv1_intra_16_native_420": OutputProfile(
        "ffv1_intra_16_native_420",
        "FFV1 v3 Intra 16-bit native 4:2:0 — mathematically lossless",
        "ffv1",
        16,
        "ffv1",
        "yuv420p16le",
        ("ffv1",),
        ".mkv",
        "4:2:0",
        True,
        True,
        False,
        "Recommended archival master for 4:2:0 sources; preserves native chroma samples without derived upsampling.",
    ),
    "ffv1_intra_16_native_422": OutputProfile(
        "ffv1_intra_16_native_422",
        "FFV1 v3 Intra 16-bit native 4:2:2 — mathematically lossless",
        "ffv1",
        16,
        "ffv1",
        "yuv422p16le",
        ("ffv1",),
        ".mkv",
        "4:2:2",
        True,
        True,
        False,
        "Recommended archival master for 4:2:2 sources; preserves native chroma samples without resampling.",
    ),
    "ffv1_intra_16_native_444": OutputProfile(
        "ffv1_intra_16_native_444",
        "FFV1 v3 Intra 16-bit native 4:4:4 — mathematically lossless",
        "ffv1",
        16,
        "ffv1",
        "yuv444p16le",
        ("ffv1",),
        ".mkv",
        "4:4:4",
        True,
        True,
        False,
        "Recommended archival master for native 4:4:4 sources; no chroma resampling is introduced.",
    ),
    "ffv1_intra_16": OutputProfile(
        "ffv1_intra_16",
        "FFV1 v3 Intra 16-bit 4:4:4 mastering — mathematically lossless",
        "ffv1",
        16,
        "ffv1",
        "yuv444p16le",
        ("ffv1",),
        ".mkv",
        "4:4:4",
        True,
        True,
        False,
        "Explicit 4:4:4 mastering/compositing intermediate with every frame intra-coded, slices and CRCs; very large.",
    ),
    "prores_4444_xq": OutputProfile(
        "prores_4444_xq",
        "Apple ProRes 4444 XQ — FFmpeg prores_ks",
        "prores",
        10,
        "prores_ks",
        "yuv444p10le",
        ("prores",),
        ".mov",
        "4:4:4",
        False,
        True,
        False,
        "XQ profile at FFmpeg prores_ks's highest accepted 4:4:4 input precision.",
    ),
    "dnxhr_444_10": OutputProfile(
        "dnxhr_444_10",
        "Avid DNxHR 444 10-bit — highest FFmpeg DNxHR profile",
        "dnxhr",
        10,
        "dnxhd",
        "yuv444p10le",
        ("dnxhd",),
        ".mov",
        "4:4:4",
        False,
        True,
        False,
        "DNxHR 444, the highest-quality DNxHR profile currently encodable by FFmpeg.",
    ),
    "dnxhr_444_12": OutputProfile(
        "dnxhr_444_12",
        "Avid DNxHR 444 12-bit — future capability gate",
        "dnxhr",
        12,
        "dnxhd",
        "yuv444p12le",
        ("dnxhd",),
        ".mov",
        "4:4:4",
        False,
        True,
        False,
        "Never enabled unless the selected FFmpeg encoder explicitly reports 12-bit support.",
    ),
}


_FFV1_PIXEL_FORMATS: dict[int, dict[str, str]] = {
    8: {"420": "yuv420p", "422": "yuv422p", "444": "yuv444p"},
    10: {"420": "yuv420p10le", "422": "yuv422p10le", "444": "yuv444p10le"},
    12: {"420": "yuv420p12le", "422": "yuv422p12le", "444": "yuv444p12le"},
    16: {"420": "yuv420p16le", "422": "yuv422p16le", "444": "yuv444p16le"},
}


def source_matched_ffv1_bit_depth(
    bits_per_raw_sample: int | None,
    pix_fmt: str | None,
) -> int:
    """Map source precision to an FFV1 storage depth supported by the app.

    FFprobe frequently omits ``bits_per_raw_sample`` for ordinary 8-bit YUV,
    so the pixel-format name is used as the secondary authority.  The result
    is deliberately a storage precision, not a claim that processing can
    recreate precision absent from the source.
    """

    explicit_depth = bits_per_raw_sample if bits_per_raw_sample and bits_per_raw_sample > 0 else None
    value = (pix_fmt or "").casefold()
    packed_depth = {"p010le": 10, "p012le": 12, "p016le": 16}.get(value)
    match = re.search(r"(9|10|12|14|16)(?:le|be)?$", value)
    inferred_depth = packed_depth or (int(match.group(1)) if match else (8 if value else None))
    # A populated FFprobe field and the pixel-format suffix should normally
    # agree.  If a demuxer reports conflicting metadata, using the larger
    # value avoids silently reducing the actual decoded sample precision.
    candidates = tuple(candidate for candidate in (explicit_depth, inferred_depth) if candidate)
    depth = max(candidates) if candidates else None
    depth = depth or 8
    if depth <= 8:
        return 8
    if depth <= 10:
        return 10
    if depth <= 12:
        return 12
    return 16


def _install_ffv1_profiles() -> None:
    chroma_labels = {"420": "4:2:0", "422": "4:2:2", "444": "4:4:4"}
    for depth, formats in _FFV1_PIXEL_FORMATS.items():
        for chroma_key, pix_fmt in formats.items():
            identifier = f"ffv1_intra_{depth}_native_{chroma_key}"
            chroma = chroma_labels[chroma_key]
            PROFILES[identifier] = OutputProfile(
                identifier,
                f"FFV1 v3 Intra {depth}-bit native {chroma} — mathematically lossless",
                "ffv1",
                depth,
                "ffv1",
                pix_fmt,
                ("ffv1",),
                ".mkv",
                chroma,
                True,
                True,
                False,
                f"Archival master preserving {chroma} chroma without derived upsampling; "
                f"the encoded {depth}-bit processed frames are stored exactly.",
            )
        mastering_id = f"ffv1_intra_{depth}"
        PROFILES[mastering_id] = OutputProfile(
            mastering_id,
            f"FFV1 v3 Intra {depth}-bit 4:4:4 mastering — mathematically lossless",
            "ffv1",
            depth,
            "ffv1",
            formats["444"],
            ("ffv1",),
            ".mkv",
            "4:4:4",
            True,
            True,
            False,
            "Explicit 4:4:4 mastering/compositing intermediate with every frame intra-coded, "
            "slices and CRCs; lower-subsampled sources gain no new chroma detail.",
        )


_install_ffv1_profiles()


def select_profile(
    family: str,
    bit_depth: int,
    hardware_encode: bool,
    av1_software_encoder: str = "libaom",
    ffv1_chroma_mode: str = "native",
    source_pix_fmt: str | None = None,
) -> OutputProfile:
    if family == "hevc":
        key = f"hevc_{'nvenc' if hardware_encode else 'x265'}_{bit_depth}"
    elif family == "av1":
        implementation = "nvenc" if hardware_encode else ("svt" if av1_software_encoder == "svt" else "libaom")
        key = f"av1_{implementation}_{bit_depth}"
    elif family == "ffv1":
        if bit_depth not in _FFV1_PIXEL_FORMATS:
            raise ValueError(
                f"Unsupported FFV1 storage precision: {bit_depth}-bit. "
                "Choose source-matched 8/10/12/16-bit output or explicit 16-bit promotion."
            )
        if ffv1_chroma_mode == "444":
            key = f"ffv1_intra_{bit_depth}"
        elif ffv1_chroma_mode == "native":
            value = (source_pix_fmt or "yuv420p").casefold()
            if "420" in value or value in {"nv12", "p010le", "p012le", "p016le"}:
                key = f"ffv1_intra_{bit_depth}_native_420"
            elif "422" in value:
                key = f"ffv1_intra_{bit_depth}_native_422"
            elif "444" in value or value.startswith("gbr"):
                key = f"ffv1_intra_{bit_depth}_native_444"
            else:
                raise ValueError(
                    f"Native-chroma FFV1 cannot safely classify source pixel format {source_pix_fmt!r}. "
                    "Choose explicit 4:4:4 mastering or use a YUV 4:2:0/4:2:2/4:4:4 source."
                )
        else:
            raise ValueError(f"Unknown FFV1 chroma mode: {ffv1_chroma_mode}")
    elif family == "prores":
        key = "prores_4444_xq"
    elif family == "dnxhr":
        key = f"dnxhr_444_{bit_depth}"
    else:
        raise ValueError(f"Unknown output family: {family}")
    try:
        return PROFILES[key]
    except KeyError as exc:
        raise ValueError(f"Unsupported output combination: {key}") from exc


def profile_capability_error(profile: OutputProfile, capabilities: CapabilityReport) -> str | None:
    if profile.encoder not in capabilities.encoders:
        return f"FFmpeg encoder '{profile.encoder}' is not available in the selected build."
    supported = capabilities.encoder_pixel_formats.get(profile.encoder, ())
    if profile.id == "dnxhr_444_12" and profile.pix_fmt not in supported:
        rendered = ", ".join(supported) if supported else "no proven pixel formats"
        return (
            "The selected FFmpeg DNxHR encoder does not support yuv444p12le. "
            "Current FFmpeg release and git-master encoders are limited to 10-bit DNxHR 444; "
            f"the app will not mislabel a 10-bit file as 12-bit. Reported formats: {rendered}."
        )
    if supported and profile.pix_fmt not in supported:
        return (
            f"FFmpeg encoder '{profile.encoder}' does not report the required pixel format "
            f"'{profile.pix_fmt}'. Supported formats: {', '.join(supported)}"
        )
    if profile.hardware and profile.encoder in capabilities.encoder_runtime_diagnostics:
        verified = capabilities.encoder_verified_bit_depths.get(profile.encoder, ())
        if profile.bit_depth not in verified:
            diagnostic = capabilities.encoder_runtime_diagnostics[profile.encoder]
            return (
                f"The selected {profile.encoder} runtime did not produce a true {profile.bit_depth}-bit stream in its "
                f"bounded startup test. Advertised input pixel formats are not accepted as proof of coded precision. "
                f"{diagnostic}"
            )
    return None


def selectable_bit_depths(
    family: str,
    capabilities: CapabilityReport | None,
    *,
    hardware_encode: bool = False,
    av1_software_encoder: str = "libaom",
) -> tuple[int, ...]:
    """Return only bit depths that the active encoder can actually produce.

    A conservative single depth remains visible before capability discovery or
    when an encoder is entirely unavailable, allowing the normal plan/dependency
    diagnostic to explain the missing tool.  Crucially, a merely hypothetical
    profile such as DNxHR 444 12-bit is never presented as a selectable dead end.
    A future FFmpeg build that really reports the required format will expose it
    automatically through the same function.
    """

    candidates = {
        "ffv1": (8, 10, 12, 16),
        "prores": (10,),
        "hevc": (10, 12),
        "av1": (10, 12),
        "dnxhr": (10, 12),
    }.get(family)
    if candidates is None:
        raise ValueError(f"Unknown output family: {family}")
    if capabilities is None:
        return candidates if family not in {"dnxhr"} else (10,)

    available: list[int] = []
    for depth in candidates:
        if (
            family == "dnxhr"
            and depth == 12
            and "yuv444p12le" not in capabilities.encoder_pixel_formats.get("dnxhd", ())
        ):
            # Twelve-bit DNxHR is a future capability gate, not an optimistic
            # default. An empty/unknown format list is not proof of support.
            continue
        try:
            profile = select_profile(
                family,
                depth,
                hardware_encode,
                av1_software_encoder,
            )
        except ValueError:
            continue
        if profile_capability_error(profile, capabilities) is None:
            available.append(depth)
    return tuple(available) or (candidates[0],)


def available_profiles(capabilities: CapabilityReport) -> list[tuple[OutputProfile, str | None]]:
    return [(profile, profile_capability_error(profile, capabilities)) for profile in PROFILES.values()]
