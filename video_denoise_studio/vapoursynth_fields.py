from __future__ import annotations

from video_processing_core.denoise.engine import vapoursynth_denoise_lines


INTERLACED_FIELD_ORDERS = frozenset({"tff", "bff"})


def _for_clip(lines: list[str], source: str, target: str) -> list[str]:
    """Rename the conventional graph variable in shared denoiser lines."""

    if source == target:
        return list(lines)
    return [line.replace(source, target) for line in lines]


def video_denoise_lines(
    identifier: str,
    strength: int,
    temporal_radius: int,
    backend: str,
    *,
    clip_variable: str = "clip",
    field_order: str | None = None,
) -> list[str]:
    """Return Video Denoise Studio's field-safe VapourSynth denoise graph.

    DFTTest2 accepts progressive nodes only.  A woven interlaced frame must not
    simply be relabeled progressive because its comb structure could be treated
    as noise.  For conclusive TFF/BFF input, split first- and second-field
    positions into independent progressive sequences, filter each parity over
    the same stored-frame radius, then restore the original weave and cadence.

    Other denoisers retain their established stored-frame graphs.  Progressive
    DFTTest2 also retains the shared one-node graph byte-for-byte apart from an
    optional graph-variable rename.
    """

    base = vapoursynth_denoise_lines(identifier, strength, temporal_radius, backend)
    if identifier != "vs_dfttest" or field_order is None:
        return _for_clip(base, "clip", clip_variable)
    if field_order not in INTERLACED_FIELD_ORDERS:
        raise ValueError(f"Interlaced DFTTest2 requires tff or bff; received {field_order!r}")

    tff = field_order == "tff"
    fields = f"{clip_variable}_fields"
    first = f"{clip_variable}_first_parity"
    second = f"{clip_variable}_second_parity"
    return [
        (
            "# DFTTest2 accepts progressive nodes only: process field parities independently, "
            "then restore the original interlaced frames."
        ),
        f"{fields} = core.std.SeparateFields({clip_variable}, tff={tff})",
        f"{first} = core.std.SelectEvery({fields}, cycle=2, offsets=0)",
        f"{second} = core.std.SelectEvery({fields}, cycle=2, offsets=1)",
        *_for_clip(base, "clip", first),
        *_for_clip(base, "clip", second),
        f"{fields} = core.std.Interleave([{first}, {second}], modify_duration=True)",
        f"{clip_variable} = core.std.DoubleWeave({fields}, tff={tff})",
        f"{clip_variable} = core.std.SelectEvery({clip_variable}, cycle=2, offsets=0)",
        f"{clip_variable} = core.std.SetFieldBased({clip_variable}, value={2 if tff else 1})",
    ]
