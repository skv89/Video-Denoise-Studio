from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping


_STABLE_FFMPEG_VERSION = re.compile(
    r"\b(?:ffmpeg|ffprobe)\s+version\s+n?(\d{1,3})\.(\d{1,3})(?:\.(\d{1,3}))?(?:[-+\s]|$)",
    flags=re.IGNORECASE,
)

_FFMPEG_VERSION_BANNER = re.compile(
    r"^\s*(?:ffmpeg|ffprobe)\s+version\s+([^\r\n]+)",
    flags=re.IGNORECASE | re.MULTILINE,
)
_GIT_REVISION_PATTERNS = (
    re.compile(r"(?:^|[-+])git-([0-9a-f]{7,40})(?=[-+\s]|$)", flags=re.IGNORECASE),
    re.compile(r"\bN-\d+-g([0-9a-f]{7,40})(?=[-+\s]|$)", flags=re.IGNORECASE),
    re.compile(r"(?:^|[-+])g([0-9a-f]{7,40})(?=[-+\s]|$)", flags=re.IGNORECASE),
)
_FFMPEG_LIBRARY_VERSION = re.compile(
    r"^\s*(libavutil|libavcodec|libavformat|libavfilter)\s+"
    r"(\d{1,3})\.\s*(\d{1,3})\.\s*(\d{1,3})(?:\s*/|\s*$)",
    flags=re.IGNORECASE | re.MULTILINE,
)

# Exact public-library versions reported by the official FFmpeg 9.0 release.
# A Git snapshot must meet every floor in both tools; a date alone is never
# treated as proof that the snapshot implements the FFmpeg 9 contract.
FFMPEG_9_LIBRARY_FLOOR: dict[str, tuple[int, int, int]] = {
    "libavutil": (61, 1, 100),
    "libavcodec": (63, 1, 100),
    "libavformat": (63, 1, 100),
    "libavfilter": (12, 1, 100),
}


@dataclass(frozen=True)
class FFmpegPairVersionAssessment:
    kind: str
    compatible: bool
    release: tuple[int, int, int] | None
    ffmpeg_git_revision: str | None
    ffprobe_git_revision: str | None
    ffmpeg_libraries: dict[str, tuple[int, int, int]]
    ffprobe_libraries: dict[str, tuple[int, int, int]]
    detail: str


def parse_stable_ffmpeg_version(text: str | None) -> tuple[int, int, int] | None:
    """Parse an explicitly labelled stable FFmpeg/FFprobe release banner.

    Date-stamped and commit-only Git snapshots intentionally return ``None``:
    their dates do not establish which stable release contract they implement.
    """

    if parse_ffmpeg_git_revision(text):
        return None
    match = _STABLE_FFMPEG_VERSION.search(text or "")
    if not match:
        return None
    return (
        int(match.group(1)),
        int(match.group(2)),
        int(match.group(3) or 0),
    )


def parse_ffmpeg_git_revision(text: str | None) -> str | None:
    """Return a revision only for recognized FFmpeg Git snapshot banners."""

    banner_match = _FFMPEG_VERSION_BANNER.search(text or "")
    if not banner_match:
        return None
    banner = banner_match.group(1)
    for pattern in _GIT_REVISION_PATTERNS:
        match = pattern.search(banner)
        if match:
            return match.group(1).lower()
    return None


def parse_ffmpeg_library_versions(text: str | None) -> dict[str, tuple[int, int, int]]:
    """Parse the four FFmpeg public libraries used to prove the 9.0 ABI floor."""

    return {
        match.group(1).lower(): (int(match.group(2)), int(match.group(3)), int(match.group(4)))
        for match in _FFMPEG_LIBRARY_VERSION.finditer(text or "")
    }


def _format_version(version: tuple[int, int, int]) -> str:
    return ".".join(map(str, version))


def _coerce_libraries(
    text: str | None,
    supplied: Mapping[str, tuple[int, int, int]] | None,
) -> dict[str, tuple[int, int, int]]:
    source = supplied if supplied is not None else parse_ffmpeg_library_versions(text)
    return {str(name).lower(): tuple(version) for name, version in source.items()}


def assess_ffmpeg_pair_versions(
    ffmpeg_text: str | None,
    ffprobe_text: str | None,
    *,
    ffmpeg_libraries: Mapping[str, tuple[int, int, int]] | None = None,
    ffprobe_libraries: Mapping[str, tuple[int, int, int]] | None = None,
) -> FFmpegPairVersionAssessment:
    """Classify a paired FFmpeg/FFprobe release without trusting snapshot dates.

    Stable releases are accepted when both banners name the same 9.0-or-newer
    release. Git snapshots additionally require the same recognized revision,
    identical required-library versions, and the exact FFmpeg 9.0 library floor.
    """

    ffmpeg_revision = parse_ffmpeg_git_revision(ffmpeg_text)
    ffprobe_revision = parse_ffmpeg_git_revision(ffprobe_text)
    ffmpeg_release = parse_stable_ffmpeg_version(ffmpeg_text) if not ffmpeg_revision else None
    ffprobe_release = parse_stable_ffmpeg_version(ffprobe_text) if not ffprobe_revision else None
    empty_libraries: dict[str, tuple[int, int, int]] = {}
    if ffmpeg_release or ffprobe_release:
        if not ffmpeg_release or not ffprobe_release:
            detail = "mismatched pair: only one tool reports an explicitly labelled stable release"
            release = None
        elif ffmpeg_release != ffprobe_release:
            detail = (
                "mismatched pair: "
                f"FFmpeg {_format_version(ffmpeg_release)}, FFprobe {_format_version(ffprobe_release)}"
            )
            release = None
        elif ffmpeg_release >= (9, 0, 0):
            detail = f"confirmed compatible stable release {_format_version(ffmpeg_release)}"
            return FFmpegPairVersionAssessment(
                kind="stable",
                compatible=True,
                release=ffmpeg_release,
                ffmpeg_git_revision=None,
                ffprobe_git_revision=None,
                ffmpeg_libraries=empty_libraries,
                ffprobe_libraries=empty_libraries,
                detail=detail,
            )
        else:
            release = ffmpeg_release
            detail = (
                f"confirmed stable release {_format_version(ffmpeg_release)}, older than required 9.0"
            )
        return FFmpegPairVersionAssessment(
            kind="unverified",
            compatible=False,
            release=release,
            ffmpeg_git_revision=None,
            ffprobe_git_revision=None,
            ffmpeg_libraries=empty_libraries,
            ffprobe_libraries=empty_libraries,
            detail=detail,
        )

    parsed_ffmpeg_libraries = _coerce_libraries(ffmpeg_text, ffmpeg_libraries)
    parsed_ffprobe_libraries = _coerce_libraries(ffprobe_text, ffprobe_libraries)

    if not ffmpeg_revision or not ffprobe_revision:
        if ffmpeg_revision or ffprobe_revision:
            detail = "unverified Git pair: only one tool reports a recognized Git revision"
        else:
            detail = (
                "unconfirmed Git/date-stamped or otherwise non-stable version banner; "
                "both tools must report the same recognized Git revision"
            )
    elif ffmpeg_revision != ffprobe_revision:
        detail = (
            "mismatched Git revisions: "
            f"FFmpeg {ffmpeg_revision}, FFprobe {ffprobe_revision}"
        )
    else:
        missing_ffmpeg = [name for name in FFMPEG_9_LIBRARY_FLOOR if name not in parsed_ffmpeg_libraries]
        missing_ffprobe = [name for name in FFMPEG_9_LIBRARY_FLOOR if name not in parsed_ffprobe_libraries]
        if missing_ffmpeg or missing_ffprobe:
            parts = []
            if missing_ffmpeg:
                parts.append("FFmpeg missing " + ", ".join(missing_ffmpeg))
            if missing_ffprobe:
                parts.append("FFprobe missing " + ", ".join(missing_ffprobe))
            detail = (
                f"Git revision {ffmpeg_revision} matches, but FFmpeg 9.0 library proof is incomplete: "
                + "; ".join(parts)
            )
        else:
            mismatched_libraries = [
                name
                for name in FFMPEG_9_LIBRARY_FLOOR
                if parsed_ffmpeg_libraries[name] != parsed_ffprobe_libraries[name]
            ]
            if mismatched_libraries:
                rendered = ", ".join(
                    f"{name} {_format_version(parsed_ffmpeg_libraries[name])}/"
                    f"{_format_version(parsed_ffprobe_libraries[name])}"
                    for name in mismatched_libraries
                )
                detail = f"Git revision {ffmpeg_revision} matches, but required library versions differ: {rendered}"
            else:
                below_floor = [
                    name
                    for name, floor in FFMPEG_9_LIBRARY_FLOOR.items()
                    if parsed_ffmpeg_libraries[name] < floor
                ]
                if below_floor:
                    rendered = ", ".join(
                        f"{name} {_format_version(parsed_ffmpeg_libraries[name])} < {_format_version(FFMPEG_9_LIBRARY_FLOOR[name])}"
                        for name in below_floor
                    )
                    detail = (
                        f"Git revision {ffmpeg_revision} matches, but predates the FFmpeg 9.0 library floor: {rendered}"
                    )
                else:
                    detail = (
                        f"verified Git snapshot revision {ffmpeg_revision}; FFmpeg/FFprobe revisions and required "
                        "library versions match, and the FFmpeg 9.0 library floor is met"
                    )
                    return FFmpegPairVersionAssessment(
                        kind="verified_git",
                        compatible=True,
                        release=None,
                        ffmpeg_git_revision=ffmpeg_revision,
                        ffprobe_git_revision=ffprobe_revision,
                        ffmpeg_libraries=parsed_ffmpeg_libraries,
                        ffprobe_libraries=parsed_ffprobe_libraries,
                        detail=detail,
                    )

    return FFmpegPairVersionAssessment(
        kind="unverified",
        compatible=False,
        release=None,
        ffmpeg_git_revision=ffmpeg_revision,
        ffprobe_git_revision=ffprobe_revision,
        ffmpeg_libraries=parsed_ffmpeg_libraries,
        ffprobe_libraries=parsed_ffprobe_libraries,
        detail=detail,
    )
