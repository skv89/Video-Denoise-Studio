from __future__ import annotations

import os
import json
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .denoise import (
    DFTTEST_ADAPTIVE_CPU_CUFFT,
    DFTTEST_ADAPTIVE_CPU_NVRTC,
    DFTTEST_ADAPTIVE_NVRTC_CUFFT,
    vapoursynth_denoise_lines,
    vapoursynth_import_lines,
)
from .dependencies import active_managed_binary, managed_runtime_environment
from .models import CapabilityReport
from .presets import nvenc_maximum_quality_args
from .tool_versions import assess_ffmpeg_pair_versions, parse_ffmpeg_library_versions


CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0
RELEVANT_ENCODERS = (
    "libx265",
    "hevc_nvenc",
    "libaom-av1",
    "libsvtav1",
    "av1_nvenc",
    "ffv1",
    "prores_ks",
    "dnxhd",
)

FFMPEG_9_INTERLACE_AUDIT_SUMMARY = (
    "FFmpeg 9.0 adds no new deinterlacing-quality algorithm over 8.1. Its interlace-related "
    "changes are scheduling, Vulkan plumbing, and edge-case maintenance; CPU/CUDA BWDIF's "
    "algorithm and the driver-defined D3D12 deinterlacer are unchanged."
)


@dataclass(frozen=True)
class _FFmpegToolchainCandidate:
    ffmpeg: Path
    ffprobe: Path
    source: str
    order: int
    ffmpeg_version: str | None
    ffprobe_version: str | None
    version_kind: str
    git_revision: str | None
    ffmpeg_libraries: dict[str, tuple[int, int, int]]
    ffprobe_libraries: dict[str, tuple[int, int, int]]
    release: tuple[int, int, int] | None
    compatible: bool
    detail: str


@dataclass(frozen=True)
class _ToolVersionEvidence:
    version: str | None
    text: str
    error: str | None


def _run(args: list[str], timeout: float = 30.0) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["AV_LOG_FORCE_NOCOLOR"] = "1"
    if args:
        env = managed_runtime_environment(args[0], env)
    return subprocess.run(
        args,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        creationflags=CREATE_NO_WINDOW,
        env=env,
        check=False,
    )


def find_binary(name: str, explicit: str | Path | None = None) -> Path | None:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    managed = active_managed_binary(name)
    if managed:
        candidates.append(managed)
    located = shutil.which(name)
    if located:
        candidates.append(Path(located))

    exe_name = name if name.lower().endswith(".exe") else f"{name}.exe"
    executable_dir = Path(sys.executable).resolve().parent
    candidates.extend(
        [
            executable_dir / exe_name,
            executable_dir / "bin" / exe_name,
            Path.cwd() / exe_name,
            Path.cwd() / "bin" / exe_name,
            Path(r"C:\Program Files (x86)\FFMPEG") / exe_name,
            Path(r"C:\Program Files\FFmpeg\bin") / exe_name,
            Path(r"C:\ffmpeg\bin") / exe_name,
        ]
    )
    seen: set[str] = set()
    for candidate in candidates:
        try:
            key = os.path.normcase(os.path.abspath(candidate))
        except OSError:
            continue
        if key in seen:
            continue
        seen.add(key)
        try:
            if candidate.is_file():
                return candidate.resolve()
        except OSError:
            continue
    return None


def _path_key(path: Path) -> str:
    return os.path.normcase(os.path.abspath(path))


def _as_executable(path: str | Path | None, name: str) -> Path | None:
    if not path:
        return None
    candidate = Path(path)
    try:
        if candidate.is_dir():
            candidate = candidate / (name if name.lower().endswith(".exe") else f"{name}.exe")
        return candidate.resolve() if candidate.is_file() else None
    except OSError:
        return None


def _split_path_value(value: str | None) -> list[Path]:
    entries: list[Path] = []
    expanded_value = os.path.expandvars(value or "")
    for raw in expanded_value.split(os.pathsep):
        expanded = raw.strip().strip('"')
        if expanded:
            entries.append(Path(expanded))
    return entries


def _windows_registry_path_entries() -> list[tuple[Path, str]]:
    """Read the current registry PATH without changing process or system state."""

    if os.name != "nt":
        return []
    try:
        import winreg
    except ImportError:
        return []

    locations = (
        (
            winreg.HKEY_LOCAL_MACHINE,
            r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment",
            "machine registry PATH",
        ),
        (winreg.HKEY_CURRENT_USER, r"Environment", "user registry PATH"),
    )
    result: list[tuple[Path, str]] = []
    for hive, key_name, label in locations:
        try:
            with winreg.OpenKey(hive, key_name) as key:
                value, _kind = winreg.QueryValueEx(key, "Path")
        except OSError:
            continue
        if not isinstance(value, str):
            continue
        for index, directory in enumerate(_split_path_value(value), start=1):
            result.append((directory, f"{label} entry {index}"))
    return result


def _automatic_ffmpeg_directories() -> list[tuple[Path, str]]:
    directories: list[tuple[Path, str]] = []
    directories.extend(
        (directory, f"process PATH entry {index}")
        for index, directory in enumerate(_split_path_value(os.environ.get("PATH")), start=1)
    )
    directories.extend(_windows_registry_path_entries())

    located = shutil.which("ffmpeg")
    if located:
        directories.append((Path(located).parent, "operating-system lookup"))

    executable_dir = Path(sys.executable).resolve().parent
    directories.extend(
        [
            (executable_dir, "application directory"),
            (executable_dir / "bin", "application bin directory"),
            (Path.cwd(), "working directory"),
            (Path.cwd() / "bin", "working bin directory"),
            (Path(r"C:\Program Files (x86)\FFMPEG"), "common FFmpeg location"),
            (Path(r"C:\Program Files\FFmpeg\bin"), "common FFmpeg location"),
            (Path(r"C:\ffmpeg\bin"), "common FFmpeg location"),
        ]
    )
    return directories


def _tool_version_evidence(executable: Path, tool_name: str) -> _ToolVersionEvidence:
    try:
        result = _run([str(executable), "-hide_banner", "-version"], timeout=5)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return _ToolVersionEvidence(None, "", f"version probe failed: {exc}")
    text = (result.stdout + "\n" + result.stderr).strip()
    if result.returncode != 0:
        tail = text[-500:] if text else f"exit code {result.returncode}"
        return _ToolVersionEvidence(None, text, f"version probe failed: {tail}")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    version = next((line for line in lines if line.lower().startswith(f"{tool_name} version ")), None)
    if not version:
        return _ToolVersionEvidence(None, text, f"version probe returned no {tool_name} banner")
    return _ToolVersionEvidence(version, text, None)


def _probe_ffmpeg_candidate(
    ffmpeg: Path,
    ffprobe: Path,
    source: str,
    order: int,
) -> _FFmpegToolchainCandidate:
    ffmpeg_evidence = _tool_version_evidence(ffmpeg, "ffmpeg")
    ffprobe_evidence = _tool_version_evidence(ffprobe, "ffprobe")
    assessment = assess_ffmpeg_pair_versions(ffmpeg_evidence.text, ffprobe_evidence.text)
    probe_errors = [error for error in (ffmpeg_evidence.error, ffprobe_evidence.error) if error]
    detail = "; ".join(probe_errors) if probe_errors else assessment.detail
    compatible = not probe_errors and assessment.compatible
    return _FFmpegToolchainCandidate(
        ffmpeg=ffmpeg,
        ffprobe=ffprobe,
        source=source,
        order=order,
        ffmpeg_version=ffmpeg_evidence.version,
        ffprobe_version=ffprobe_evidence.version,
        version_kind=assessment.kind if not probe_errors else "unverified",
        git_revision=assessment.ffmpeg_git_revision,
        ffmpeg_libraries=assessment.ffmpeg_libraries,
        ffprobe_libraries=assessment.ffprobe_libraries,
        release=assessment.release,
        compatible=compatible,
        detail=detail,
    )


def _candidate_diagnostic(candidate: _FFmpegToolchainCandidate, selected: bool) -> str:
    marker = "SELECTED" if selected else "NOT SELECTED"
    banner = candidate.ffmpeg_version or "FFmpeg version unavailable"
    return (
        f"{marker} [{candidate.source}] {candidate.ffmpeg.parent} — "
        f"{candidate.detail}; reported: {banner}"
    )


def discover_ffmpeg_toolchain(
    ffmpeg_path: str | Path | None = None,
    ffprobe_path: str | Path | None = None,
) -> tuple[Path | None, Path | None, str | None, tuple[str, ...]]:
    """Select a paired FFmpeg toolchain and retain auditable discovery evidence.

    Explicit user selection and an activated app-local runtime remain authoritative.
    Automatic discovery evaluates every paired candidate. It prefers the newest
    explicitly labelled stable 9.0-or-newer release; when none exists, the first
    verified Git pair wins. Unverified candidates preserve normal PATH order.
    """

    notes: list[str] = []
    order = 0

    if ffmpeg_path or ffprobe_path:
        requested_ffmpeg = _as_executable(ffmpeg_path, "ffmpeg")
        requested_ffprobe = _as_executable(ffprobe_path, "ffprobe")
        if requested_ffmpeg and not requested_ffprobe:
            requested_ffprobe = _as_executable(requested_ffmpeg.with_name("ffprobe.exe"), "ffprobe")
        if requested_ffprobe and not requested_ffmpeg:
            requested_ffmpeg = _as_executable(requested_ffprobe.with_name("ffmpeg.exe"), "ffmpeg")
        if requested_ffmpeg and requested_ffprobe:
            candidate = _probe_ffmpeg_candidate(requested_ffmpeg, requested_ffprobe, "explicit user selection", order)
            return (
                candidate.ffmpeg,
                candidate.ffprobe,
                candidate.source,
                (_candidate_diagnostic(candidate, True),),
            )
        notes.append(
            "IGNORED [explicit user selection] FFmpeg and FFprobe were not both readable; automatic discovery continued."
        )

    managed_ffmpeg = _as_executable(active_managed_binary("ffmpeg"), "ffmpeg")
    managed_ffprobe = _as_executable(active_managed_binary("ffprobe"), "ffprobe")
    if managed_ffmpeg or managed_ffprobe:
        if managed_ffmpeg and managed_ffprobe:
            candidate = _probe_ffmpeg_candidate(managed_ffmpeg, managed_ffprobe, "validated app-local runtime", order)
            return (
                candidate.ffmpeg,
                candidate.ffprobe,
                candidate.source,
                tuple(notes + [_candidate_diagnostic(candidate, True)]),
            )
        notes.append(
            "IGNORED [app-local runtime] its activation manifest did not resolve both FFmpeg and FFprobe; "
            "automatic discovery continued."
        )

    candidates: list[_FFmpegToolchainCandidate] = []
    seen: set[tuple[str, str]] = set()
    for directory, source in _automatic_ffmpeg_directories():
        ffmpeg = _as_executable(directory / "ffmpeg.exe", "ffmpeg")
        ffprobe = _as_executable(directory / "ffprobe.exe", "ffprobe")
        if not ffmpeg or not ffprobe:
            continue
        key = (_path_key(ffmpeg), _path_key(ffprobe))
        if key in seen:
            continue
        seen.add(key)
        order += 1
        candidates.append(_probe_ffmpeg_candidate(ffmpeg, ffprobe, source, order))

    if not candidates:
        notes.append("No directory in the active/registry PATH or supported common locations contained a readable pair.")
        return None, None, None, tuple(notes)

    stable_compatible = [
        candidate for candidate in candidates if candidate.compatible and candidate.version_kind == "stable"
    ]
    git_compatible = [
        candidate for candidate in candidates if candidate.compatible and candidate.version_kind == "verified_git"
    ]
    if stable_compatible:
        selected = max(stable_compatible, key=lambda candidate: (candidate.release or (0, 0, 0), -candidate.order))
    elif git_compatible:
        # Git hashes are identifiers, not sortable version numbers. Preserve
        # the user's discovery/PATH order after each pair proves the same strict
        # revision and FFmpeg-9-or-newer public-library contract.
        selected = min(git_compatible, key=lambda candidate: candidate.order)
    else:
        # Preserve normal PATH precedence when no candidate satisfies the
        # stable/Git compatibility floor. Every candidate remains visible in
        # diagnostics, but a parseable old release does not displace a user's
        # first PATH choice merely because its banner is easier to classify.
        selected = min(candidates, key=lambda candidate: candidate.order)

    diagnostics = notes + [
        _candidate_diagnostic(candidate, candidate == selected)
        for candidate in sorted(candidates, key=lambda candidate: candidate.order)
    ]
    return selected.ffmpeg, selected.ffprobe, selected.source, tuple(diagnostics)


def _parse_named_components(text: str) -> frozenset[str]:
    names: set[str] = set()
    for line in text.splitlines():
        match = re.match(r"^\s*[A-Z\.]{2,8}\s+([^\s=]+)", line)
        if match:
            names.add(match.group(1))
    return frozenset(names)


def _parse_pixel_formats(text: str) -> tuple[str, ...]:
    match = re.search(r"Supported pixel formats:\s*(.+)", text)
    if not match:
        return ()
    return tuple(part for part in match.group(1).strip().split() if part)


def _infer_vspipe_python(vspipe: Path | None) -> Path | None:
    if not vspipe:
        return None
    parent = vspipe.parent
    candidates = [parent.parent / "python.exe", parent / "python.exe"]
    candidates.extend(ancestor / "python.exe" for ancestor in vspipe.parents)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return find_binary("python")


def _inspect_vapoursynth(vspipe: Path | None) -> tuple[str | None, bool, str, str | None]:
    if not vspipe:
        return None, False, "VSPipe was not found.", None

    version_result = _run([str(vspipe), "--version"], timeout=15)
    version_text = (version_result.stdout + "\n" + version_result.stderr).strip()
    version_match = re.search(r"Core\s+(R\d+)", version_text)
    version = version_match.group(1) if version_match else (version_text.splitlines()[0] if version_text else None)

    diagnostic_script = r'''import vapoursynth as vs
from vapoursynth import core
from vsdeinterlace import QTempGaussMC
from vstools import depth

try:
    core.bs.VideoSource
except Exception as exc:
    raise RuntimeError("BestSource plugin namespace 'bs' is unavailable") from exc

clip = core.std.BlankClip(width=64, height=48, format=vs.YUV420P8, length=8, fpsnum=25, fpsden=1)
clip = core.std.SetFieldBased(clip, value=2)
clip = depth(clip, 16)
qtgmc = QTempGaussMC(analyze_force_tr=2, analyze_blksize=16, analyze_overlap=2, analyze_refine=2)
qtgmc = qtgmc.source_match(tr=2, mode=QTempGaussMC.SourceMatchMode.TWICE_REFINED)
qtgmc = qtgmc.lossless(mode=QTempGaussMC.LosslessMode.POSTSMOOTH, anti_comb=True)
clip = qtgmc.bob(clip, tff=True)
clip.set_output()
'''
    script_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".vpy", encoding="utf-8", delete=False) as handle:
            handle.write(diagnostic_script)
            script_path = Path(handle.name)
        result = _run([str(vspipe), "--info", str(script_path), "-"], timeout=60)
        combined = (result.stdout + "\n" + result.stderr).strip()
        ready = result.returncode == 0
        diagnostic = "QTGMC maximum-fidelity graph and BestSource are available." if ready else combined[-4000:]
    except (OSError, subprocess.TimeoutExpired) as exc:
        ready = False
        diagnostic = f"VapourSynth dependency check failed: {exc}"
    finally:
        if script_path:
            try:
                script_path.unlink(missing_ok=True)
            except OSError:
                pass

    python_path = _infer_vspipe_python(vspipe)
    install_command = None
    if python_path:
        install_command = subprocess.list2cmdline(
            [
                str(python_path),
                "-m",
                "pip",
                "install",
                "--upgrade",
                "vsjetpack[deinterlace]",
                "--extra-index-url",
                "https://jaded-encoding-thaumaturgy.github.io/vs-wheels/simple",
            ]
        )
    return version, ready, diagnostic, install_command


def _inspect_vulkan_nnedi3(vspipe: Path | None) -> tuple[bool, str, str | None]:
    """Render a bounded QTGMC graph through the optional Vulkan interpolator."""

    if not vspipe:
        return False, "VSPipe was not found.", None

    diagnostic_script = r'''import vapoursynth as vs
from vapoursynth import core
from vsaa import NNEDI3
from vsdeinterlace import QTempGaussMC
from vstools import depth

core.num_threads = min(4, max(1, core.num_threads))
clip = core.std.BlankClip(width=160, height=96, format=vs.YUV420P8, length=8, fpsnum=25, fpsden=1)
clip = depth(clip, 16)
clip = core.std.SetFieldBased(clip, value=2)
qtgmc = QTempGaussMC(
    analyze_force_tr=2,
    analyze_blksize=16,
    analyze_overlap=2,
    analyze_refine=2,
    basic_bobber=NNEDI3(nsize=1, gpu=True),
)
qtgmc = qtgmc.source_match(tr=2, mode=QTempGaussMC.SourceMatchMode.TWICE_REFINED)
qtgmc = qtgmc.lossless(mode=QTempGaussMC.LosslessMode.POSTSMOOTH, anti_comb=True)
clip = qtgmc.bob(clip, tff=True)
clip = core.std.SetFieldBased(clip, value=0)
clip.set_output()
'''
    script_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".vpy", encoding="utf-8", delete=False) as handle:
            handle.write(diagnostic_script)
            script_path = Path(handle.name)
        result = _run(
            [
                str(vspipe),
                "--requests",
                "2",
                "--end",
                "7",
                "--container",
                "y4m",
                str(script_path),
                os.devnull,
            ],
            timeout=90,
        )
        combined = (result.stdout + "\n" + result.stderr).strip()
        if result.returncode != 0:
            excerpt = combined[-3000:].replace("\x00", " ") or f"exit code {result.returncode}"
            return False, "Vulkan NNEDI3 QTGMC graph failed: " + excerpt, None
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"Vulkan NNEDI3 QTGMC graph could not complete: {type(exc).__name__}: {exc}", None
    finally:
        if script_path:
            try:
                script_path.unlink(missing_ok=True)
            except OSError:
                pass

    package_version = None
    python_path = _infer_vspipe_python(vspipe)
    if python_path:
        try:
            version_result = _run(
                [
                    str(python_path),
                    "-c",
                    "from importlib.metadata import version; print(version('vapoursynth-nnedi3vk'))",
                ],
                timeout=15,
            )
            if version_result.returncode == 0 and version_result.stdout.strip():
                package_version = version_result.stdout.strip().splitlines()[-1]
        except (OSError, subprocess.TimeoutExpired):
            pass
    version_note = f" package {package_version}" if package_version else ""
    return (
        True,
        "Vulkan NNEDI3" + version_note + " emitted 8 frames through the real QTGMC graph on the default Vulkan device.",
        package_version,
    )


VAPOURSYNTH_DENOISER_CANDIDATES: dict[str, tuple[str, ...]] = {
    "vs_bm3d": ("vszipcu", "bm3dcuda_rtc", "bm3dcuda", "bm3dcpu"),
    "vs_dfttest": ("dfttest_cpu", "dfttest_nvrtc", "dfttest_cufft"),
    "vs_mvtools": ("mvtools",),
    "vs_nlmeans": ("vszipcu", "nlm_ispc"),
}


def _vapoursynth_denoiser_script(identifier: str, backend: str) -> str:
    namespace = {
        "vszipcu": "vszipcu",
        "bm3dcuda_rtc": "bm3dcuda_rtc",
        "bm3dcuda": "bm3dcuda",
        "bm3dcpu": "bm3dcpu",
        "dfttest_cpu": "dfttest2_cpu",
        "dfttest_nvrtc": "dfttest2_nvrtc",
        "dfttest_cufft": "dfttest2_cuda",
        "mvtools": "mvu",
        "nlm_ispc": "nlm_ispc",
    }[backend]
    lines = [
        "import vapoursynth as vs",
        "from vapoursynth import core",
        "from vstools import depth",
        *vapoursynth_import_lines(identifier),
        "",
        f"if not hasattr(core, {namespace!r}):",
        f"    raise RuntimeError('Required VapourSynth plugin namespace {namespace} is unavailable')",
        "core.num_threads = min(4, max(1, core.num_threads))",
        "clip = core.std.BlankClip(width=64, height=48, format=vs.YUV420P8, length=12, fpsnum=25, fpsden=1)",
        "clip = depth(clip, 16)",
        *vapoursynth_denoise_lines(identifier, 4, 2, backend),
        "clip = core.std.SetFieldBased(clip, value=0)",
        "clip.set_output()",
        "",
    ]
    return "\n".join(lines)


def _inspect_vapoursynth_denoisers(
    vspipe: Path | None,
) -> tuple[dict[str, bool], dict[str, str], dict[str, str]]:
    """Evaluate bounded real graphs and report the exact implementation used."""

    ready = {identifier: False for identifier in VAPOURSYNTH_DENOISER_CANDIDATES}
    backends: dict[str, str] = {}
    diagnostics: dict[str, str] = {}
    if not vspipe:
        for identifier in ready:
            diagnostics[identifier] = "VSPipe was not found."
        return ready, backends, diagnostics

    for identifier, candidates in VAPOURSYNTH_DENOISER_CANDIDATES.items():
        attempts: list[str] = []
        passed_backends: list[str] = []
        for backend in candidates:
            script_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile("w", suffix=".vpy", encoding="utf-8", delete=False) as handle:
                    handle.write(_vapoursynth_denoiser_script(identifier, backend))
                    script_path = Path(handle.name)
                result = _run(
                    [
                        str(vspipe),
                        "--requests",
                        "1",
                        "--end",
                        "3",
                        "--container",
                        "y4m",
                        str(script_path),
                        os.devnull,
                    ],
                    timeout=90,
                )
                combined = (result.stdout + "\n" + result.stderr).strip()
                if result.returncode == 0:
                    passed_backends.append(backend)
                    attempts.append(f"{backend}: graph emitted 4 frames")
                    if identifier != "vs_dfttest":
                        ready[identifier] = True
                        backends[identifier] = backend
                        break
                    continue
                excerpt = combined[-1200:].replace("\x00", " ") or f"exit code {result.returncode}"
                attempts.append(f"{backend}: unavailable ({excerpt})")
            except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
                attempts.append(f"{backend}: {type(exc).__name__}: {exc}")
            finally:
                if script_path:
                    try:
                        script_path.unlink(missing_ok=True)
                    except OSError:
                        pass
        if identifier == "vs_dfttest" and passed_backends:
            passed = set(passed_backends)
            ready[identifier] = True
            if {"dfttest_cpu", "dfttest_nvrtc"} <= passed:
                backends[identifier] = DFTTEST_ADAPTIVE_CPU_NVRTC
            elif {"dfttest_cpu", "dfttest_cufft"} <= passed:
                backends[identifier] = DFTTEST_ADAPTIVE_CPU_CUFFT
            elif {"dfttest_nvrtc", "dfttest_cufft"} <= passed:
                backends[identifier] = DFTTEST_ADAPTIVE_NVRTC_CUFFT
            else:
                backends[identifier] = passed_backends[0]
        diagnostics[identifier] = "; ".join(attempts)[-4000:]
    return ready, backends, diagnostics


def _inspect_gpu() -> tuple[str | None, int | None, str | None]:
    nvidia_smi = find_binary("nvidia-smi")
    if not nvidia_smi:
        return None, None, None
    result = _run(
        [
            str(nvidia_smi),
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        timeout=15,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return None, None, None
    first = result.stdout.splitlines()[0]
    parts = [part.strip() for part in first.split(",")]
    if len(parts) < 3:
        return first.strip(), None, None
    try:
        memory = int(parts[1])
    except ValueError:
        memory = None
    return parts[0], memory, parts[2]


def _pixel_format_depth(pixel_format: str | None) -> int | None:
    value = (pixel_format or "").lower()
    for depth in (16, 14, 12, 10, 9):
        if f"p{depth}" in value or value.startswith(f"p0{depth}"):
            return depth
    return 8 if value else None


def _inspect_nvenc_coded_depths(
    ffmpeg: Path | None,
    ffprobe: Path | None,
    encoders: frozenset[str],
) -> tuple[dict[str, tuple[int, ...]], dict[str, str]]:
    """Prove coded precision; NVENC's accepted input formats can overstate output depth."""

    verified: dict[str, tuple[int, ...]] = {}
    diagnostics: dict[str, str] = {}
    if not ffmpeg or not ffprobe:
        return verified, diagnostics

    with tempfile.TemporaryDirectory(prefix="DeinterlaceStudio-NVENC-") as directory:
        root = Path(directory)
        for encoder in ("hevc_nvenc", "av1_nvenc"):
            if encoder not in encoders:
                continue
            depths: list[int] = []
            details: list[str] = []
            for requested, pixel_format in ((10, "p010le"), (12, "p012le")):
                output = root / f"{encoder}-{requested}.mkv"
                encode = _run(
                    [
                        str(ffmpeg),
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-y",
                        "-f",
                        "lavfi",
                        "-i",
                        "testsrc2=size=320x240:rate=10:duration=0.4",
                        "-vf",
                        f"format={pixel_format}",
                        "-frames:v",
                        "4",
                        "-c:v",
                        encoder,
                        *nvenc_maximum_quality_args(14),
                        "-pix_fmt",
                        pixel_format,
                        str(output),
                    ],
                    timeout=45,
                )
                if encode.returncode != 0 or not output.is_file():
                    error = (encode.stderr or encode.stdout).strip().splitlines()
                    details.append(f"{requested}-bit request failed: {(error[-1] if error else 'no diagnostic')[:300]}")
                    continue
                probe = _run(
                    [
                        str(ffprobe),
                        "-v",
                        "error",
                        "-select_streams",
                        "v:0",
                        "-show_entries",
                        "stream=pix_fmt,bits_per_raw_sample,profile",
                        "-of",
                        "json",
                        str(output),
                    ],
                    timeout=30,
                )
                stream: dict[str, object] = {}
                try:
                    stream = json.loads(probe.stdout)["streams"][0]
                    coded_depth = int(stream.get("bits_per_raw_sample") or 0) or _pixel_format_depth(stream.get("pix_fmt"))
                except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
                    coded_depth = None
                if coded_depth == requested:
                    depths.append(requested)
                    details.append(
                        f"{requested}-bit request verified as {stream.get('pix_fmt')} with the complete "
                        "P7/UHQ/full-resolution-multipass quality contract"
                    )
                else:
                    details.append(
                        f"{requested}-bit request decoded as {stream.get('pix_fmt', 'unknown')} "
                        f"({coded_depth or 'unknown'}-bit), so it is disabled"
                    )
            verified[encoder] = tuple(depths)
            diagnostics[encoder] = "; ".join(details)
    return verified, diagnostics


def _d3d12_error_excerpt(text: str) -> str:
    preferred = (
        "Failed to create video processor",
        "No deinterlacing methods supported by hardware",
        "Failed to configure processor",
        "Device creation failed",
        "No device available",
        "Conversion failed",
    )
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    selected: list[str] = []
    for marker in preferred:
        match = next((line for line in lines if marker.casefold() in line.casefold()), None)
        if match and match not in selected:
            selected.append(match)
        if len(selected) == 2:
            break
    if not selected and lines:
        selected.append(lines[-1])
    return "; ".join(selected)[:700] or "FFmpeg returned no diagnostic text."


def _format_process_exit_code(code: int) -> str:
    if code > 0x7FFFFFFF:
        return f"{code - 0x100000000} / Windows code 0x{code:08X}"
    return str(code)


def _probe_d3d12_deinterlace(ffmpeg: Path, method: str, pixel_format: str) -> tuple[bool, str]:
    """Run a tiny real filter graph; filter enumeration alone does not prove driver support."""

    try:
        result = _run(
            [
                str(ffmpeg),
                "-hide_banner",
                "-nostdin",
                "-loglevel",
                "verbose",
                "-init_hw_device",
                "d3d12va=d3d12",
                "-filter_hw_device",
                "d3d12",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=64x64:rate=25:duration=0.32",
                "-an",
                "-vf",
                (
                    f"format={pixel_format},setfield=tff,hwupload,"
                    f"deinterlace_d3d12=method={method}:mode=frame:deint=all,"
                    f"hwdownload,format={pixel_format}"
                ),
                "-frames:v",
                "4",
                "-f",
                "null",
                "NUL",
            ],
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"Runtime probe could not complete: {type(exc).__name__}: {exc}"

    combined = result.stdout + "\n" + result.stderr
    passed = (
        result.returncode == 0
        and "D3D12 deinterlace processor successfully configured" in combined
        and re.search(r"Output stream #0:0 \(video\):\s*4 frames encoded", combined) is not None
    )
    if passed:
        format_label = "NV12" if pixel_format == "nv12" else "P010"
        return True, (
            f"Runtime probe passed: D3D12 processor configured and emitted 4 progressive {format_label} frames."
        )
    return False, (
        f"Runtime probe failed (exit {_format_process_exit_code(result.returncode)}): "
        f"{_d3d12_error_excerpt(combined)}"
    )


def _inspect_ffmpeg_interlace_runtime(
    ffmpeg: Path | None,
    filters: frozenset[str],
    hwaccels: frozenset[str],
) -> tuple[dict[str, bool], dict[str, str]]:
    """Summarize FFmpeg 9 interlace paths and prove D3D12 methods when they are present."""

    verified: dict[str, bool] = {}
    diagnostics = {
        "ffmpeg9_source_audit": FFMPEG_9_INTERLACE_AUDIT_SUMMARY,
        "bwdif_cpu": (
            "Available. FFmpeg's highest-quality built-in software baseline; FFmpeg 9.0 did not "
            "change its deinterlacing algorithm."
            if "bwdif" in filters
            else "Unavailable in the selected FFmpeg build."
        ),
        "bwdif_cuda": (
            "Available as a GPU implementation of BWDIF; it is a speed path, not a new FFmpeg 9 quality mode."
            if "bwdif_cuda" in filters and "cuda" in hwaccels
            else "Unavailable in the selected FFmpeg build or CUDA hardware context."
        ),
    }

    if not ffmpeg or "deinterlace_d3d12" not in filters:
        for key in ("d3d12_custom", "d3d12_bob", "d3d12_custom_p010", "d3d12_bob_p010"):
            diagnostics[key] = "Not present in the selected FFmpeg build."
        return verified, diagnostics
    if os.name != "nt":
        for key in ("d3d12_custom", "d3d12_bob", "d3d12_custom_p010", "d3d12_bob_p010"):
            diagnostics[key] = "Present but not probed because D3D12 is Windows-only."
        return verified, diagnostics

    for pixel_format, suffix in (("nv12", ""), ("p010le", "_p010")):
        for method in ("custom", "bob"):
            passed, detail = _probe_d3d12_deinterlace(ffmpeg, method, pixel_format)
            key = f"d3d12_{method}{suffix}"
            verified[key] = passed
            if method == "custom":
                qualifier = (
                    "Driver-defined advanced processing is operational, but remains experimental and is never "
                    "selected automatically because its algorithm and quality depend on the display driver."
                    if passed
                    else "Driver-defined advanced processing is excluded from output backends on this runtime."
                )
            else:
                qualifier = (
                    "Basic bob interpolation works, but is intentionally excluded because BWDIF and QTGMC offer better quality."
                    if passed
                    else "Basic D3D12 bob is unavailable on this runtime."
                )
            diagnostics[key] = f"{detail.rstrip('. ')}. {qualifier}"
    return verified, diagnostics


def inspect_capabilities(
    ffmpeg_path: str | Path | None = None,
    ffprobe_path: str | Path | None = None,
    vspipe_path: str | Path | None = None,
) -> CapabilityReport:
    ffmpeg, ffprobe, ffmpeg_selection_source, ffmpeg_discovery_diagnostics = discover_ffmpeg_toolchain(
        ffmpeg_path,
        ffprobe_path,
    )
    vspipe = find_binary("vspipe", vspipe_path)

    version = None
    ffprobe_version = None
    configuration = None
    ffmpeg_version_text = ""
    ffprobe_version_text = ""
    ffmpeg_library_versions: dict[str, tuple[int, int, int]] = {}
    ffprobe_library_versions: dict[str, tuple[int, int, int]] = {}
    filters: frozenset[str] = frozenset()
    encoders: frozenset[str] = frozenset()
    pixel_formats: dict[str, tuple[str, ...]] = {}
    hwaccels: frozenset[str] = frozenset()

    if ffmpeg:
        version_result = _run([str(ffmpeg), "-hide_banner", "-version"])
        ffmpeg_version_text = (version_result.stdout + "\n" + version_result.stderr).strip()
        version_lines = ffmpeg_version_text.splitlines()
        if version_lines:
            version = next(
                (line.strip() for line in version_lines if line.strip().lower().startswith("ffmpeg version ")),
                version_lines[0].strip(),
            )
        for line in version_lines:
            if line.startswith("configuration:"):
                configuration = line.partition(":")[2].strip()
                break
        ffmpeg_library_versions = parse_ffmpeg_library_versions(ffmpeg_version_text)

        filter_result = _run([str(ffmpeg), "-hide_banner", "-filters"])
        filters = _parse_named_components(filter_result.stdout + filter_result.stderr)
        encoder_result = _run([str(ffmpeg), "-hide_banner", "-encoders"])
        encoders = _parse_named_components(encoder_result.stdout + encoder_result.stderr)
        hw_result = _run([str(ffmpeg), "-hide_banner", "-hwaccels"])
        hw_lines = (hw_result.stdout + hw_result.stderr).splitlines()
        hwaccels = frozenset(
            line.strip()
            for line in hw_lines
            if line.strip() and not line.lower().startswith("hardware acceleration")
        )
        for encoder in RELEVANT_ENCODERS:
            if encoder not in encoders:
                continue
            help_result = _run([str(ffmpeg), "-hide_banner", "-h", f"encoder={encoder}"])
            pixel_formats[encoder] = _parse_pixel_formats(help_result.stdout + help_result.stderr)

    if ffprobe:
        ffprobe_evidence = _tool_version_evidence(ffprobe, "ffprobe")
        ffprobe_version = ffprobe_evidence.version
        ffprobe_version_text = ffprobe_evidence.text
        ffprobe_library_versions = parse_ffmpeg_library_versions(ffprobe_version_text)

    version_assessment = assess_ffmpeg_pair_versions(
        ffmpeg_version_text or version,
        ffprobe_version_text or ffprobe_version,
        ffmpeg_libraries=ffmpeg_library_versions,
        ffprobe_libraries=ffprobe_library_versions,
    )

    vs_version, qtgmc_ready, qtgmc_diagnostic, install_command = _inspect_vapoursynth(vspipe)
    vulkan_nnedi3_ready, vulkan_nnedi3_diagnostic, vulkan_nnedi3_version = _inspect_vulkan_nnedi3(vspipe)
    vs_denoise_ready, vs_denoise_backends, vs_denoise_diagnostics = _inspect_vapoursynth_denoisers(vspipe)
    denoise_capabilities = {
        "ffmpeg_fftdnoiz": "fftdnoiz" in filters,
        "ffmpeg_atadenoise": "atadenoise" in filters,
        **vs_denoise_ready,
    }
    denoise_backends = {
        **({"ffmpeg_fftdnoiz": "ffmpeg"} if denoise_capabilities["ffmpeg_fftdnoiz"] else {}),
        **({"ffmpeg_atadenoise": "ffmpeg"} if denoise_capabilities["ffmpeg_atadenoise"] else {}),
        **vs_denoise_backends,
    }
    denoise_diagnostics = {
        "ffmpeg_fftdnoiz": (
            "FFmpeg fftdnoiz filter is available."
            if denoise_capabilities["ffmpeg_fftdnoiz"]
            else "The selected FFmpeg build does not expose fftdnoiz."
        ),
        "ffmpeg_atadenoise": (
            "FFmpeg atadenoise filter is available."
            if denoise_capabilities["ffmpeg_atadenoise"]
            else "The selected FFmpeg build does not expose atadenoise."
        ),
        **vs_denoise_diagnostics,
    }
    gpu_name, gpu_memory, gpu_driver = _inspect_gpu()
    verified_depths, encoder_diagnostics = _inspect_nvenc_coded_depths(ffmpeg, ffprobe, encoders)
    interlace_verified, interlace_diagnostics = _inspect_ffmpeg_interlace_runtime(ffmpeg, filters, hwaccels)
    return CapabilityReport(
        ffmpeg_path=ffmpeg,
        ffprobe_path=ffprobe,
        ffmpeg_version=version,
        ffmpeg_configuration=configuration,
        filters=filters,
        encoders=encoders,
        encoder_pixel_formats=pixel_formats,
        hwaccels=hwaccels,
        vspipe_path=vspipe,
        vapoursynth_version=vs_version,
        qtgmc_ready=qtgmc_ready,
        qtgmc_diagnostic=qtgmc_diagnostic,
        qtgmc_install_command=install_command,
        gpu_name=gpu_name,
        gpu_memory_mib=gpu_memory,
        gpu_driver=gpu_driver,
        encoder_verified_bit_depths=verified_depths,
        encoder_runtime_diagnostics=encoder_diagnostics,
        interlace_runtime_verified=interlace_verified,
        interlace_runtime_diagnostics=interlace_diagnostics,
        ffmpeg_selection_source=ffmpeg_selection_source,
        ffmpeg_discovery_diagnostics=ffmpeg_discovery_diagnostics,
        ffprobe_version=ffprobe_version,
        ffmpeg_version_kind=version_assessment.kind,
        ffmpeg_version_diagnostic=version_assessment.detail,
        ffmpeg_git_revision=version_assessment.ffmpeg_git_revision,
        ffprobe_git_revision=version_assessment.ffprobe_git_revision,
        ffmpeg_library_versions=ffmpeg_library_versions,
        ffprobe_library_versions=ffprobe_library_versions,
        denoise_capabilities=denoise_capabilities,
        denoise_backends=denoise_backends,
        denoise_diagnostics=denoise_diagnostics,
        vulkan_nnedi3_ready=vulkan_nnedi3_ready,
        vulkan_nnedi3_diagnostic=vulkan_nnedi3_diagnostic,
        vulkan_nnedi3_package_version=vulkan_nnedi3_version,
    )
