from __future__ import annotations

import hashlib
import gzip
import io
import json
import os
import queue
import re
import shutil
import stat
import subprocess
import sys
import threading
import time
import uuid
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Callable, Iterable
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .. import __version__
from ..media.models import CapabilityReport
from ..media.tool_versions import assess_ffmpeg_pair_versions


RUNTIME_DIRECTORY_NAME = "Video Processing Runtime"
LEGACY_RUNTIME_DIRECTORY_NAMES = (
    "Deinterlace Studio Runtime",
    "Video Repair Tool Runtime",
    "Video Denoise Studio Runtime",
)
GITHUB_FFMPEG_LATEST = "https://api.github.com/repos/GyanD/codexffmpeg/releases/latest"
GITHUB_VAPOURSYNTH_LATEST = "https://api.github.com/repos/vapoursynth/vapoursynth/releases/latest"
PYTHON_FTP_INDEX = "https://www.python.org/ftp/python/"
PYPI_PIP_JSON = "https://pypi.org/pypi/pip/json"
PYPI_VSJETPACK_JSON = "https://pypi.org/pypi/vsjetpack/json"
PYPI_NNEDI3VK_JSON = "https://pypi.org/pypi/vapoursynth-nnedi3vk/json"
VS_WHEELS_INDEX = "https://jaded-encoding-thaumaturgy.github.io/vs-wheels/simple"
PYTHON_MINOR = (3, 14)

MAX_METADATA_BYTES = 8 * 1024 * 1024
MAX_DOWNLOAD_BYTES = 1024 * 1024 * 1024
MAX_ARCHIVE_ENTRIES = 30_000
MAX_ARCHIVE_UNCOMPRESSED_BYTES = 8 * 1024 * 1024 * 1024
MAX_ARCHIVE_MEMBER_BYTES = 2 * 1024 * 1024 * 1024
CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0
CREATE_NEW_PROCESS_GROUP = 0x00000200 if os.name == "nt" else 0

ProgressCallback = Callable[[str, str, int | None, int | None], None]
LogCallback = Callable[[str], None]


class DependencyInstallError(RuntimeError):
    pass


class DependencyInstallCancelled(DependencyInstallError):
    pass


@dataclass(frozen=True)
class ReleaseAsset:
    name: str
    url: str
    sha256: str | None
    size: int | None


@dataclass(frozen=True)
class DependencyReleasePlan:
    ffmpeg_version: str
    ffmpeg_asset: ReleaseAsset
    vapoursynth_version: str
    vapoursynth_asset: ReleaseAsset
    python_version: str
    python_asset: ReleaseAsset
    pip_version: str
    pip_asset: ReleaseAsset
    vsjetpack_version: str
    nnedi3vk_version: str | None = None
    nnedi3vk_asset: ReleaseAsset | None = None


@dataclass(frozen=True)
class DependencyInstallResult:
    runtime_root: Path
    ffmpeg_path: Path | None
    ffprobe_path: Path | None
    vspipe_path: Path | None
    ffmpeg_version: str | None
    vapoursynth_version: str | None
    vsjetpack_version: str | None
    nnedi3vk_version: str | None
    manifest_path: Path


def application_directory() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    # ``dependencies.py`` lives one level deeper than the historical
    # application-local modules.  Source launches must still place/read the
    # portable runtime beside the entry points, not inside this package.
    return Path(__file__).resolve().parents[2]


def managed_runtime_root(app_directory: Path | None = None) -> Path:
    return (app_directory or application_directory()) / RUNTIME_DIRECTORY_NAME


def managed_runtime_roots(app_directory: Path | None = None) -> tuple[Path, ...]:
    """Return the neutral runtime followed by recognized legacy locations.

    New installs always target :data:`RUNTIME_DIRECTORY_NAME`.  Read-only
    discovery also recognizes each application's former app-local folder so a
    portable upgrade does not force users to download the same validated tools
    again.
    """

    app = app_directory or application_directory()
    return (managed_runtime_root(app),) + tuple(app / name for name in LEGACY_RUNTIME_DIRECTORY_NAMES)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _manifest_path(root: Path) -> Path:
    return root / "active.json"


def _read_active_manifest(root: Path) -> dict[str, object]:
    try:
        payload = json.loads(_manifest_path(root).read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict) or payload.get("schema") != 1:
        return {}
    components = payload.get("components")
    if not isinstance(components, dict):
        return {}
    return payload


def _resolve_manifest_file(root: Path, value: object) -> Path | None:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        return None
    candidate = (root / Path(value)).resolve()
    resolved_root = root.resolve()
    if not _is_relative_to(candidate, resolved_root) or not candidate.is_file():
        return None
    return candidate


def active_managed_binary(name: str, app_directory: Path | None = None) -> Path | None:
    key = name.lower().removesuffix(".exe")
    component_name = "vapoursynth" if key == "vspipe" else "ffmpeg"
    for root in managed_runtime_roots(app_directory):
        payload = _read_active_manifest(root)
        components = payload.get("components")
        if not isinstance(components, dict):
            continue
        component = components.get(component_name)
        if not isinstance(component, dict):
            continue
        binary = _resolve_manifest_file(root, component.get(key))
        if binary is not None:
            return binary
    return None


def managed_runtime_environment(executable: str | Path, base: dict[str, str] | None = None) -> dict[str, str]:
    """Return a child-only environment for an app-local portable VapourSynth."""

    env = dict(base or os.environ)
    path = Path(executable).resolve()
    root: Path | None = None
    for parent in path.parents:
        if (parent / "python.exe").is_file() and (parent / "python3.dll").is_file():
            root = parent
            break
    if root is None:
        return env
    package = root / "Lib" / "site-packages" / "vapoursynth"
    vsscript = package / "vsscript.dll"
    prefixes = [str(root), str(package)]
    env["PATH"] = os.pathsep.join(prefixes + ([env["PATH"]] if env.get("PATH") else []))
    env["PYTHONHOME"] = str(root)
    env["PYTHONPATH"] = str(root / "Lib" / "site-packages")
    if vsscript.is_file():
        env["VSSCRIPT_PATH"] = str(vsscript)
    return env


def dependency_issues(capabilities: CapabilityReport | None) -> dict[str, tuple[str, ...]]:
    issues: dict[str, list[str]] = {"ffmpeg": [], "vapoursynth": []}
    if not capabilities or not capabilities.ffmpeg_path or not capabilities.ffprobe_path:
        issues["ffmpeg"].append("FFmpeg and FFprobe were not both found")
    else:
        version_assessment = assess_ffmpeg_pair_versions(
            capabilities.ffmpeg_version,
            capabilities.ffprobe_version,
            ffmpeg_libraries=capabilities.ffmpeg_library_versions,
            ffprobe_libraries=capabilities.ffprobe_library_versions,
        )
        if version_assessment.release and version_assessment.release < (9, 0, 0):
            issues["ffmpeg"].append("FFmpeg 9.0 or newer is required")
        elif not version_assessment.compatible:
            issues["ffmpeg"].append(
                "the completed scan could not verify a matching FFmpeg 9.0-or-newer stable/Git pair: "
                + version_assessment.detail
            )
        missing_filters = {"idet", "bwdif", "fftdnoiz", "atadenoise"} - set(capabilities.filters)
        required_encoders = {"libx265", "libaom-av1", "libsvtav1", "ffv1", "prores_ks", "dnxhd"}
        missing_encoders = required_encoders - set(capabilities.encoders)
        if missing_filters:
            issues["ffmpeg"].append("missing filters: " + ", ".join(sorted(missing_filters)))
        if missing_encoders:
            issues["ffmpeg"].append("missing encoders: " + ", ".join(sorted(missing_encoders)))

    version_match = re.search(r"R(\d+)", capabilities.vapoursynth_version or "") if capabilities else None
    if not capabilities or not capabilities.vspipe_path:
        issues["vapoursynth"].append("VSPipe was not found")
    elif not version_match or int(version_match.group(1)) < 78:
        issues["vapoursynth"].append("VapourSynth R78 or newer is required")
    if not capabilities or not capabilities.qtgmc_ready:
        issues["vapoursynth"].append("BestSource and the maximum-fidelity QTGMC graph are not ready")
    required_denoisers = {
        "vs_bm3d": "two-pass temporal V-BM3D",
        "vs_dfttest": "temporal DFTTest2",
        "vs_mvtools": "motion-compensated MVTools degrain",
        "vs_nlmeans": "temporal NLMeans",
    }
    for identifier, label in required_denoisers.items():
        if not capabilities or not capabilities.denoise_capabilities.get(identifier, False):
            issues["vapoursynth"].append(f"{label} graph is not ready")
    return {name: tuple(values) for name, values in issues.items() if values}


def _request(url: str, *, method: str = "GET") -> Request:
    parsed = urlparse(url)
    if parsed.scheme.lower() != "https" or not parsed.hostname:
        raise DependencyInstallError(f"Refusing a non-HTTPS dependency URL: {url}")
    return Request(
        url,
        method=method,
        headers={
            "Accept": "application/vnd.github+json, application/json, text/html;q=0.9, */*;q=0.1",
            "User-Agent": f"Video-Processing-Core/{__version__}",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )


def _read_url(url: str, *, limit: int = MAX_METADATA_BYTES, timeout: float = 30.0) -> bytes:
    with urlopen(_request(url), timeout=timeout) as response:
        final = urlparse(response.geturl())
        if final.scheme.lower() != "https":
            raise DependencyInstallError(f"Dependency metadata redirected to a non-HTTPS URL: {response.geturl()}")
        declared = response.headers.get("Content-Length")
        if declared and int(declared) > limit:
            raise DependencyInstallError(f"Dependency metadata is unexpectedly large ({declared} bytes).")
        data = response.read(limit + 1)
    if len(data) > limit:
        raise DependencyInstallError("Dependency metadata exceeded its safety limit.")
    if data.startswith(b"\x1f\x8b"):
        try:
            with gzip.GzipFile(fileobj=io.BytesIO(data)) as compressed:
                data = compressed.read(limit + 1)
        except (OSError, EOFError) as exc:
            raise DependencyInstallError(f"Dependency metadata had invalid gzip encoding: {exc}") from exc
        if len(data) > limit:
            raise DependencyInstallError("Expanded dependency metadata exceeded its safety limit.")
    return data


def _read_json(url: str) -> dict[str, object]:
    try:
        payload = json.loads(_read_url(url).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, OSError) as exc:
        raise DependencyInstallError(f"Could not read release metadata from {url}: {exc}") from exc
    if not isinstance(payload, dict):
        raise DependencyInstallError(f"Release metadata from {url} was not a JSON object.")
    return payload


def _sha256_from_digest(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    match = re.fullmatch(r"sha256:([0-9a-fA-F]{64})", value.strip())
    return match.group(1).lower() if match else None


def _safe_download_name(name: str) -> str:
    parts = _safe_member_parts(name)
    if len(parts) != 1 or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,239}", parts[0]):
        raise DependencyInstallError(f"Unsafe dependency asset filename: {name!r}")
    return parts[0]


def _github_asset(payload: dict[str, object], exact_name: str) -> ReleaseAsset:
    if payload.get("draft") or payload.get("prerelease"):
        raise DependencyInstallError("The provider's latest endpoint returned a draft or prerelease.")
    assets = payload.get("assets")
    if not isinstance(assets, list):
        raise DependencyInstallError("Release metadata did not contain an asset list.")
    matches = [item for item in assets if isinstance(item, dict) and item.get("name") == exact_name]
    if len(matches) != 1:
        raise DependencyInstallError(f"Expected exactly one release asset named {exact_name!r}; found {len(matches)}.")
    item = matches[0]
    url = item.get("browser_download_url")
    size = item.get("size")
    digest = _sha256_from_digest(item.get("digest"))
    if not isinstance(url, str) or not isinstance(size, int) or size <= 0 or size > MAX_DOWNLOAD_BYTES:
        raise DependencyInstallError(f"Release asset {exact_name!r} has invalid URL or size metadata.")
    if not digest:
        raise DependencyInstallError(f"Release asset {exact_name!r} has no valid published SHA-256 digest.")
    _request(url)
    return ReleaseAsset(_safe_download_name(exact_name), url, digest, size)


def _pypi_wheel(payload: dict[str, object], package: str, suffix: str) -> tuple[str, ReleaseAsset]:
    info = payload.get("info")
    urls = payload.get("urls")
    if not isinstance(info, dict) or not isinstance(info.get("version"), str) or not isinstance(urls, list):
        raise DependencyInstallError(f"PyPI metadata for {package} is incomplete.")
    version = info["version"]
    matches = [
        item
        for item in urls
        if isinstance(item, dict)
        and item.get("packagetype") == "bdist_wheel"
        and isinstance(item.get("filename"), str)
        and item["filename"].endswith(suffix)
        and not item.get("yanked")
    ]
    if len(matches) != 1:
        raise DependencyInstallError(f"Expected one current {package} wheel ending in {suffix!r}; found {len(matches)}.")
    item = matches[0]
    digests = item.get("digests")
    sha256 = digests.get("sha256") if isinstance(digests, dict) else None
    url = item.get("url")
    size = item.get("size")
    if not isinstance(url, str) or not isinstance(size, int) or not isinstance(sha256, str):
        raise DependencyInstallError(f"PyPI wheel metadata for {package} is incomplete.")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", sha256) or size <= 0 or size > MAX_DOWNLOAD_BYTES:
        raise DependencyInstallError(f"PyPI wheel metadata for {package} is invalid.")
    _request(url)
    return version, ReleaseAsset(_safe_download_name(str(item["filename"])), url, sha256.lower(), size)


def _latest_python_embed() -> tuple[str, ReleaseAsset]:
    try:
        listing = _read_url(PYTHON_FTP_INDEX).decode("utf-8", errors="strict")
    except (UnicodeDecodeError, OSError) as exc:
        raise DependencyInstallError(f"Could not read the official Python release index: {exc}") from exc
    major, minor = PYTHON_MINOR
    pattern = rf'''href=["']{major}\.{minor}\.(\d+)/["']'''
    patches = {int(match.group(1)) for match in re.finditer(pattern, listing)}
    if not patches:
        # Python's directory listing format is simple, but fail closed if it changes.
        raise DependencyInstallError(f"No Python {major}.{minor}.x releases were found in the official index.")
    patch = max(patches)
    version = f"{major}.{minor}.{patch}"
    name = f"python-{version}-embed-amd64.zip"
    url = f"{PYTHON_FTP_INDEX}{version}/{name}"
    release_slug = version.replace(".", "")
    release_page = f"https://www.python.org/downloads/release/python-{release_slug}/"
    try:
        page = _read_url(release_page).decode("utf-8", errors="strict")
    except (UnicodeDecodeError, OSError) as exc:
        raise DependencyInstallError(f"Could not read the official Python {version} release page: {exc}") from exc
    anchor = re.search(rf'''(?is)href=["'][^"']*/{re.escape(name)}["']''', page)
    row_text = ""
    if anchor:
        folded = page.casefold()
        row_start = folded.rfind("<tr", 0, anchor.start())
        row_end = folded.find("</tr>", anchor.end())
        if row_start >= 0 and row_end >= 0:
            row_text = page[row_start : row_end + len("</tr>")]
    checksum_match = re.search(r'''(?is)<code\s+class=["']checksum["']>(.*?)</code>''', row_text)
    checksum = re.sub(r"<[^>]+>|\s+", "", checksum_match.group(1)) if checksum_match else ""
    if not re.fullmatch(r"[0-9a-fA-F]{64}", checksum):
        raise DependencyInstallError(f"The official Python {version} release page had no valid SHA-256 for {name}.")
    return version, ReleaseAsset(_safe_download_name(name), url, checksum.lower(), None)


def resolve_latest_releases() -> DependencyReleasePlan:
    ffmpeg_payload = _read_json(GITHUB_FFMPEG_LATEST)
    ffmpeg_version = ffmpeg_payload.get("tag_name")
    if not isinstance(ffmpeg_version, str) or not re.fullmatch(r"\d+(?:\.\d+){1,3}", ffmpeg_version):
        raise DependencyInstallError("The FFmpeg build provider returned an invalid stable tag.")
    ffmpeg_asset = _github_asset(ffmpeg_payload, f"ffmpeg-{ffmpeg_version}-full_build.zip")

    vs_payload = _read_json(GITHUB_VAPOURSYNTH_LATEST)
    vs_version = vs_payload.get("tag_name")
    if not isinstance(vs_version, str) or not re.fullmatch(r"R\d+", vs_version):
        raise DependencyInstallError("The VapourSynth provider returned an invalid stable tag.")
    vs_asset = _github_asset(vs_payload, f"VapourSynth64-Portable-{vs_version}.zip")

    python_version, python_asset = _latest_python_embed()
    pip_version, pip_asset = _pypi_wheel(_read_json(PYPI_PIP_JSON), "pip", "-py3-none-any.whl")
    jet_payload = _read_json(PYPI_VSJETPACK_JSON)
    jet_info = jet_payload.get("info")
    jet_version = jet_info.get("version") if isinstance(jet_info, dict) else None
    if not isinstance(jet_version, str) or not re.fullmatch(r"\d+(?:\.\d+){1,3}", jet_version):
        raise DependencyInstallError("PyPI returned an invalid current VSJetpack version.")
    nnedi3vk_version: str | None = None
    nnedi3vk_asset: ReleaseAsset | None = None
    try:
        nnedi3vk_version, nnedi3vk_asset = _pypi_wheel(
            _read_json(PYPI_NNEDI3VK_JSON),
            "vapoursynth-nnedi3vk",
            "-py3-none-win_amd64.whl",
        )
    except DependencyInstallError:
        # Vulkan NNEDI3 is optional.  Metadata or wheel unavailability must not
        # prevent a fully verified CPU QTGMC runtime from being installed.
        pass
    return DependencyReleasePlan(
        ffmpeg_version=ffmpeg_version,
        ffmpeg_asset=ffmpeg_asset,
        vapoursynth_version=vs_version,
        vapoursynth_asset=vs_asset,
        python_version=python_version,
        python_asset=python_asset,
        pip_version=pip_version,
        pip_asset=pip_asset,
        vsjetpack_version=jet_version,
        nnedi3vk_version=nnedi3vk_version,
        nnedi3vk_asset=nnedi3vk_asset,
    )


def _check_cancel(cancel_event: threading.Event) -> None:
    if cancel_event.is_set():
        raise DependencyInstallCancelled("Dependency installation was canceled.")


def _download(
    asset: ReleaseAsset,
    destination: Path,
    cancel_event: threading.Event,
    progress: ProgressCallback | None,
    stage: str,
) -> str:
    if destination.name != _safe_download_name(asset.name):
        raise DependencyInstallError(f"Download destination does not match the validated asset name: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    written = 0
    try:
        with urlopen(_request(asset.url), timeout=60) as response, destination.open("xb") as output:
            final = urlparse(response.geturl())
            if final.scheme.lower() != "https":
                raise DependencyInstallError(f"{asset.name} redirected to a non-HTTPS URL.")
            declared_text = response.headers.get("Content-Length")
            declared = int(declared_text) if declared_text and declared_text.isdigit() else None
            expected = asset.size or declared
            if declared is not None and declared > MAX_DOWNLOAD_BYTES:
                raise DependencyInstallError(f"{asset.name} exceeds the download safety limit.")
            while True:
                _check_cancel(cancel_event)
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > MAX_DOWNLOAD_BYTES:
                    raise DependencyInstallError(f"{asset.name} exceeded the download safety limit.")
                output.write(chunk)
                digest.update(chunk)
                if progress:
                    progress(stage, f"Downloading {asset.name}", written, expected)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    if asset.size is not None and written != asset.size:
        destination.unlink(missing_ok=True)
        raise DependencyInstallError(f"{asset.name} size mismatch: expected {asset.size}, received {written}.")
    actual = digest.hexdigest()
    if asset.sha256 and actual.lower() != asset.sha256.lower():
        destination.unlink(missing_ok=True)
        raise DependencyInstallError(
            f"{asset.name} SHA-256 mismatch: expected {asset.sha256}, received {actual}. Nothing was activated."
        )
    return actual


_WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{number}" for number in range(1, 10)),
    *(f"LPT{number}" for number in range(1, 10)),
}


def _safe_member_parts(name: str) -> tuple[str, ...]:
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    if not normalized or normalized.startswith("/") or path.is_absolute():
        raise DependencyInstallError(f"Unsafe absolute archive member: {name!r}")
    parts = tuple(part for part in path.parts if part not in {"", "."})
    if not parts or any(part == ".." for part in parts):
        raise DependencyInstallError(f"Unsafe traversal archive member: {name!r}")
    for part in parts:
        stem = part.rstrip(" .").split(".", 1)[0].upper()
        if part.endswith((" ", ".")) or ":" in part or stem in _WINDOWS_RESERVED_NAMES:
            raise DependencyInstallError(f"Unsafe Windows archive member: {name!r}")
        if len(part) > 240:
            raise DependencyInstallError(f"Archive member component is too long: {name!r}")
    return parts


def safe_extract_zip(
    archive: Path,
    destination: Path,
    cancel_event: threading.Event | None = None,
    progress: ProgressCallback | None = None,
    stage: str = "extract",
) -> None:
    cancel = cancel_event or threading.Event()
    destination.mkdir(parents=True, exist_ok=False)
    destination_root = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        entries = bundle.infolist()
        if len(entries) > MAX_ARCHIVE_ENTRIES:
            raise DependencyInstallError(f"Archive contains too many entries ({len(entries)}).")
        total = sum(info.file_size for info in entries)
        if total > MAX_ARCHIVE_UNCOMPRESSED_BYTES:
            raise DependencyInstallError("Archive's expanded size exceeds the safety limit.")
        seen: set[str] = set()
        completed = 0
        for info in entries:
            _check_cancel(cancel)
            if info.file_size > MAX_ARCHIVE_MEMBER_BYTES:
                raise DependencyInstallError(f"Archive member is too large: {info.filename!r}")
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise DependencyInstallError(f"Archive links are not allowed: {info.filename!r}")
            parts = _safe_member_parts(info.filename)
            key = "/".join(parts).casefold()
            if key in seen:
                raise DependencyInstallError(f"Archive contains a duplicate normalized path: {info.filename!r}")
            seen.add(key)
            target = destination.joinpath(*parts)
            resolved = target.resolve()
            if not _is_relative_to(resolved, destination_root):
                raise DependencyInstallError(f"Archive member escapes the staging directory: {info.filename!r}")
            is_directory = info.is_dir() or info.filename.endswith(("/", "\\"))
            if is_directory:
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                copied = 0
                with bundle.open(info, "r") as source, target.open("xb") as output:
                    while True:
                        _check_cancel(cancel)
                        chunk = source.read(1024 * 1024)
                        if not chunk:
                            break
                        copied += len(chunk)
                        if copied > info.file_size or copied > MAX_ARCHIVE_MEMBER_BYTES:
                            raise DependencyInstallError(f"Archive member expanded beyond its declared size: {info.filename!r}")
                        output.write(chunk)
                if copied != info.file_size:
                    raise DependencyInstallError(f"Archive member size mismatch: {info.filename!r}")
            completed += info.file_size
            if progress:
                progress(stage, f"Extracting {archive.name}", completed, total)


def _run_process(
    args: list[str],
    cancel_event: threading.Event,
    log: LogCallback | None,
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: float = 1800.0,
) -> str:
    _check_cancel(cancel_event)
    if log:
        log("Running: " + subprocess.list2cmdline(args))
    process = subprocess.Popen(
        args,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP,
        env=env,
    )
    lines: queue.Queue[str | None] = queue.Queue()

    def reader() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            lines.put(line.rstrip())
        lines.put(None)

    reader_thread = threading.Thread(target=reader, daemon=True)
    reader_thread.start()
    deadline = time.monotonic() + timeout
    finished_reader = False
    output: list[str] = []
    while process.poll() is None or not finished_reader:
        if cancel_event.is_set() or time.monotonic() > deadline:
            try:
                if os.name == "nt":
                    subprocess.run(
                        ["taskkill.exe", "/PID", str(process.pid), "/T", "/F"],
                        capture_output=True,
                        creationflags=CREATE_NO_WINDOW,
                        check=False,
                    )
                else:
                    process.kill()
            except OSError:
                process.kill()
            process.wait(timeout=10)
            if cancel_event.is_set():
                raise DependencyInstallCancelled("Dependency installation was canceled.")
            raise DependencyInstallError(f"Dependency command exceeded its {timeout:.0f}-second limit.")
        try:
            item = lines.get(timeout=0.1)
        except queue.Empty:
            continue
        if item is None:
            finished_reader = True
        else:
            output.append(item)
            if log:
                log(item)
    code = process.wait()
    reader_thread.join(timeout=5)
    if process.stdout:
        process.stdout.close()
    rendered = "\n".join(output)
    if code != 0:
        tail = "\n".join(output[-30:])
        raise DependencyInstallError(f"Dependency command exited with code {code}.\n{tail}")
    return rendered


def _powershell_signature(path: Path, cancel_event: threading.Event, log: LogCallback | None) -> None:
    if os.name != "nt":
        raise DependencyInstallError("The portable dependency installer is supported on Windows only.")
    escaped = str(path).replace("'", "''")
    command = (
        f"$s=Get-AuthenticodeSignature -LiteralPath '{escaped}'; "
        "[pscustomobject]@{Status=[string]$s.Status;Subject=$s.SignerCertificate.Subject} | ConvertTo-Json -Compress"
    )
    powershell_env = os.environ.copy()
    # A parent PowerShell 7 session can export a PSModulePath that prevents
    # Windows PowerShell 5.1 from loading its inbox Security module.
    windows_root = Path(powershell_env.get("SystemRoot", r"C:\Windows"))
    powershell_env["PSModulePath"] = str(
        windows_root / "System32" / "WindowsPowerShell" / "v1.0" / "Modules"
    )
    output = _run_process(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
        cancel_event,
        None,
        env=powershell_env,
        timeout=60,
    )
    try:
        payload = json.loads(output.splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise DependencyInstallError(f"Could not verify the Authenticode signature on {path.name}.") from exc
    if payload.get("Status") != "Valid" or "Python Software Foundation" not in str(payload.get("Subject")):
        raise DependencyInstallError(
            f"{path.name} did not have a valid Python Software Foundation Authenticode signature "
            f"(status={payload.get('Status')!r}, subject={payload.get('Subject')!r}). Nothing was activated."
        )
    if log:
        log(f"Verified Python Software Foundation signature: {path.name}")


def _find_exactly_one(root: Path, name: str) -> Path:
    matches = [path for path in root.rglob(name) if path.is_file()]
    if len(matches) != 1:
        raise DependencyInstallError(f"Expected exactly one {name} in the staged runtime; found {len(matches)}.")
    return matches[0]


def _validate_ffmpeg(ffmpeg: Path, ffprobe: Path, cancel_event: threading.Event, log: LogCallback | None) -> str:
    version = _run_process([str(ffmpeg), "-hide_banner", "-version"], cancel_event, log, timeout=60)
    filters = _run_process([str(ffmpeg), "-hide_banner", "-filters"], cancel_event, None, timeout=60)
    encoders = _run_process([str(ffmpeg), "-hide_banner", "-encoders"], cancel_event, None, timeout=60)
    _run_process([str(ffprobe), "-hide_banner", "-version"], cancel_event, None, timeout=60)
    from .capabilities import _parse_named_components

    filter_names = _parse_named_components(filters)
    encoder_names = _parse_named_components(encoders)
    missing_filters = {"idet", "bwdif", "fftdnoiz", "atadenoise"} - set(filter_names)
    missing_encoders = {"libx265", "libaom-av1", "libsvtav1", "ffv1", "prores_ks", "dnxhd"} - set(encoder_names)
    if missing_filters or missing_encoders:
        raise DependencyInstallError(
            "Staged FFmpeg is incomplete: "
            + (f"missing filters {sorted(missing_filters)}. " if missing_filters else "")
            + (f"missing encoders {sorted(missing_encoders)}." if missing_encoders else "")
        )
    return version.splitlines()[0] if version else "unknown"


def _normalized_distribution_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value.strip().casefold())


def _validate_pip_install_report(
    report: Path,
    *,
    expected_requested: frozenset[str] = frozenset(),
) -> None:
    report_payload = json.loads(report.read_text(encoding="utf-8"))
    install_items = report_payload.get("install", [])
    if not isinstance(install_items, list) or not install_items:
        raise DependencyInstallError("Pip's installation audit report contains no installed packages.")
    requested_names: set[str] = set()
    for item in install_items:
        download_info = item.get("download_info", {}) if isinstance(item, dict) else {}
        url = download_info.get("url") if isinstance(download_info, dict) else None
        archive_info = download_info.get("archive_info", {}) if isinstance(download_info, dict) else {}
        hashes_info = archive_info.get("hashes", {}) if isinstance(archive_info, dict) else {}
        if (
            not isinstance(url, str)
            or urlparse(url).scheme != "https"
            or not isinstance(hashes_info.get("sha256"), str)
        ):
            raise DependencyInstallError("A Python package lacked an HTTPS source or recorded SHA-256 hash.")
        metadata = item.get("metadata", {}) if isinstance(item, dict) else {}
        name = metadata.get("name") if isinstance(metadata, dict) else None
        if item.get("requested") is True and isinstance(name, str) and name.strip():
            requested_names.add(_normalized_distribution_name(name))
    expected_names = {_normalized_distribution_name(name) for name in expected_requested}
    missing = expected_names - requested_names
    if missing:
        raise DependencyInstallError(
            "Pip's installation audit did not prove the requested package(s): " + ", ".join(sorted(missing)) + "."
        )


def _nvidia_runtime_detected() -> bool:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return False
    try:
        result = subprocess.run(
            [executable, "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
            creationflags=CREATE_NO_WINDOW,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and bool(result.stdout.strip())


def _prepare_python(
    plan: DependencyReleasePlan,
    downloads: Path,
    candidate: Path,
    cancel_event: threading.Event,
    progress: ProgressCallback | None,
    log: LogCallback | None,
) -> tuple[Path, Path, dict[str, str]]:
    python_zip = downloads / plan.python_asset.name
    vs_zip = downloads / plan.vapoursynth_asset.name
    pip_wheel = downloads / plan.pip_asset.name
    hashes = {
        "python": _download(plan.python_asset, python_zip, cancel_event, progress, "python-download"),
        "vapoursynth": _download(plan.vapoursynth_asset, vs_zip, cancel_event, progress, "vapoursynth-download"),
        "pip": _download(plan.pip_asset, pip_wheel, cancel_event, progress, "pip-download"),
    }
    nnedi3vk_wheel: Path | None = None
    if plan.nnedi3vk_asset:
        nnedi3vk_wheel = downloads / plan.nnedi3vk_asset.name
        hashes["nnedi3vk"] = _download(
            plan.nnedi3vk_asset,
            nnedi3vk_wheel,
            cancel_event,
            progress,
            "nnedi3vk-download",
        )
    safe_extract_zip(python_zip, candidate, cancel_event, progress, "python-extract")
    # The destination already exists after Python; extract the trusted VS ZIP through
    # a sibling and merge only after both archives independently pass path validation.
    vs_extract = candidate.parent / f"vs-extract-{uuid.uuid4().hex}"
    safe_extract_zip(vs_zip, vs_extract, cancel_event, progress, "vapoursynth-extract")
    for child in vs_extract.iterdir():
        target = candidate / child.name
        if target.exists():
            raise DependencyInstallError(f"VapourSynth archive unexpectedly collides with Python path {child.name!r}.")
        os.replace(child, target)
    vs_extract.rmdir()

    python = candidate / "python.exe"
    python3_dll = candidate / "python3.dll"
    if not python.is_file() or not python3_dll.is_file():
        raise DependencyInstallError("The staged embedded Python runtime is incomplete.")
    if progress:
        progress("python-verify", "Verifying embedded Python hashes, signatures, and version", None, None)
    _powershell_signature(python, cancel_event, log)
    _powershell_signature(python3_dll, cancel_event, log)
    version_text = _run_process([str(python), "--version"], cancel_event, log, cwd=candidate, timeout=60)
    if plan.python_version not in version_text:
        raise DependencyInstallError(
            f"Embedded Python version mismatch: expected {plan.python_version}, received {version_text.strip()}."
        )

    pth_files = list(candidate.glob("python3*._pth"))
    if len(pth_files) != 1:
        raise DependencyInstallError(f"Expected one embedded-Python _pth file; found {len(pth_files)}.")
    pth = pth_files[0]
    lines = pth.read_text(encoding="utf-8").splitlines()
    if "Lib\\site-packages" not in lines:
        lines.append("Lib\\site-packages")
    pth.write_text("\n".join(lines) + "\n", encoding="utf-8")

    site_packages = candidate / "Lib" / "site-packages"
    safe_extract_zip(pip_wheel, site_packages, cancel_event, progress, "pip-bootstrap")
    _run_process([str(python), "-m", "pip", "--version"], cancel_event, log, cwd=candidate, timeout=60)
    wheel_prefix = f"vapoursynth-{plan.vapoursynth_version.removeprefix('R')}-"
    vs_wheels = [
        path
        for path in (candidate / "wheel").glob("*.whl")
        if path.is_file() and path.name.casefold().startswith(wheel_prefix.casefold())
    ]
    if len(vs_wheels) != 1:
        raise DependencyInstallError(
            f"Expected one VapourSynth {plan.vapoursynth_version} wheel; found {len(vs_wheels)}."
        )
    vs_wheel = vs_wheels[0]
    if progress:
        progress("vapoursynth-install", "Installing VapourSynth R78+ into the portable runtime", None, None)
    _run_process(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-warn-script-location",
            "--no-index",
            str(vs_wheel),
        ],
        cancel_event,
        log,
        cwd=candidate,
        timeout=300,
    )
    report = candidate.parent / "vsjetpack-install-report.json"
    if progress:
        progress("qtgmc-install", "Installing the current VSJetpack deinterlacing wheel set", None, None)
    _run_process(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-warn-script-location",
            "--only-binary=:all:",
            "--upgrade",
            "--index-url",
            "https://pypi.org/simple",
            "--extra-index-url",
            VS_WHEELS_INDEX,
            "--report",
            str(report),
            f"vsjetpack[deinterlace]=={plan.vsjetpack_version}",
        ],
        cancel_event,
        log,
        cwd=candidate,
        timeout=1800,
    )
    if not report.is_file():
        raise DependencyInstallError("Pip did not produce the required installation audit report.")
    _validate_pip_install_report(report, expected_requested=frozenset({"vsjetpack"}))

    vulkan_note = candidate / "optional-vulkan-nnedi3.txt"
    if nnedi3vk_wheel and plan.nnedi3vk_version:
        if progress:
            progress(
                "nnedi3vk-install",
                "Installing optional Vulkan NNEDI3 into the app-local runtime",
                None,
                None,
            )
        try:
            _run_process(
                [
                    str(python),
                    "-m",
                    "pip",
                    "install",
                    "--disable-pip-version-check",
                    "--no-warn-script-location",
                    "--no-index",
                    "--no-deps",
                    str(nnedi3vk_wheel),
                ],
                cancel_event,
                log,
                cwd=candidate,
                timeout=600,
            )
            vulkan_note.write_text(
                "Optional vapoursynth-nnedi3vk "
                f"{plan.nnedi3vk_version} installed from {plan.nnedi3vk_asset.name}; "
                f"published SHA-256 {hashes['nnedi3vk']}. It remains disabled unless a real Vulkan QTGMC graph passes.\n",
                encoding="utf-8",
            )
        except DependencyInstallCancelled:
            raise
        except (DependencyInstallError, OSError) as exc:
            vulkan_note.write_text(
                "Optional Vulkan NNEDI3 installation was unavailable; verified CPU QTGMC remains active. "
                f"Diagnostic: {exc}\n",
                encoding="utf-8",
            )
            if log:
                log(f"Optional Vulkan NNEDI3 was not installed; continuing with CPU QTGMC: {exc}")
    else:
        vulkan_note.write_text(
            "Official Windows vapoursynth-nnedi3vk release metadata was unavailable; verified CPU QTGMC remains active.\n",
            encoding="utf-8",
        )

    optional_note = candidate / "optional-nvidia-denoisers.txt"
    if _nvidia_runtime_detected():
        optional_report = candidate.parent / "nvidia-denoise-install-report.json"
        if progress:
            progress(
                "denoise-gpu-install",
                "Installing optional NVIDIA V-BM3D, DFTTest2, and temporal NLMeans plugins",
                None,
                None,
            )
        try:
            _run_process(
                [
                    str(python),
                    "-m",
                    "pip",
                    "install",
                    "--disable-pip-version-check",
                    "--no-warn-script-location",
                    "--only-binary=:all:",
                    "--upgrade",
                    "--index-url",
                    "https://pypi.org/simple",
                    "--extra-index-url",
                    VS_WHEELS_INDEX,
                    "--report",
                    str(optional_report),
                    "vapoursynth-bm3dcuda>=2.16",
                    "vapoursynth-dfttest2-nvrtc>=10.3",
                    "vapoursynth-vszipcu>=1.2.0",
                ],
                cancel_event,
                log,
                cwd=candidate,
                timeout=1800,
            )
            if not optional_report.is_file():
                raise DependencyInstallError("Pip did not produce the optional NVIDIA installation audit report.")
            _validate_pip_install_report(
                optional_report,
                expected_requested=frozenset(
                    {
                        "vapoursynth-bm3dcuda",
                        "vapoursynth-dfttest2-nvrtc",
                        "vapoursynth-vszipcu",
                    }
                ),
            )
            shutil.copy2(optional_report, candidate / "nvidia-denoise-installation-report.json")
            optional_note.write_text(
                "Optional NVIDIA temporal-denoise plugins installed and retained only if each exact graph validation passes.\n",
                encoding="utf-8",
            )
        except DependencyInstallCancelled:
            raise
        except (DependencyInstallError, OSError, json.JSONDecodeError) as exc:
            optional_note.write_text(
                "Optional NVIDIA temporal-denoise plugin installation was unavailable; the staged CPU graphs "
                f"remain required and will be validated. Diagnostic: {exc}\n",
                encoding="utf-8",
            )
            if log:
                log(
                    "Optional NVIDIA temporal-denoise plugins were not installed; continuing with the required "
                    f"CPU implementations: {exc}"
                )
    else:
        optional_note.write_text(
            "No NVIDIA runtime was detected during staging; verified CPU temporal-denoise implementations are used.\n",
            encoding="utf-8",
        )

    _run_process([str(python), "-m", "pip", "check"], cancel_event, log, cwd=candidate, timeout=300)
    freeze = _run_process([str(python), "-m", "pip", "freeze", "--all"], cancel_event, None, cwd=candidate, timeout=300)
    (candidate / "installed-packages.txt").write_text(freeze + "\n", encoding="utf-8")
    shutil.copy2(report, candidate / "installation-report.json")
    vspipe = _find_exactly_one(candidate / "Lib" / "site-packages" / "vapoursynth", "vspipe.exe")
    return python, vspipe, hashes


def _validate_vapoursynth(
    vspipe: Path,
    cancel_event: threading.Event,
    log: LogCallback | None,
) -> tuple[str, bool, str, str | None]:
    _check_cancel(cancel_event)
    from .capabilities import _inspect_vapoursynth, _inspect_vapoursynth_denoisers, _inspect_vulkan_nnedi3

    version, ready, diagnostic, _command = _inspect_vapoursynth(vspipe)
    if log:
        log(f"VapourSynth graph check: {diagnostic}")
    if not ready:
        raise DependencyInstallError(f"The staged VapourSynth/QTGMC graph failed validation:\n{diagnostic}")
    denoise_ready, denoise_backends, denoise_diagnostics = _inspect_vapoursynth_denoisers(vspipe)
    if log:
        for identifier in sorted(denoise_ready):
            log(
                f"VapourSynth denoiser graph {identifier}: ready={denoise_ready[identifier]}; "
                f"backend={denoise_backends.get(identifier, 'none')}; {denoise_diagnostics.get(identifier, '')}"
            )
    missing = [identifier for identifier, passed in denoise_ready.items() if not passed]
    if missing:
        details = "\n".join(f"{identifier}: {denoise_diagnostics.get(identifier, '')}" for identifier in missing)
        raise DependencyInstallError(
            "The staged VapourSynth temporal-denoise graph set failed validation:\n" + details
        )
    vulkan_ready, vulkan_diagnostic, vulkan_version = _inspect_vulkan_nnedi3(vspipe)
    if log:
        log(
            f"Optional Vulkan NNEDI3 graph: ready={vulkan_ready}; "
            f"version={vulkan_version or 'unavailable'}; {vulkan_diagnostic}"
        )
    return version or "unknown", vulkan_ready, vulkan_diagnostic, vulkan_version


def _slug(value: str) -> str:
    rendered = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-.")
    return rendered or "unknown"


def _move_validated_directory(source: Path, target: Path, cancel_event: threading.Event) -> None:
    """Atomically rename a candidate, tolerating short Windows scanner locks."""

    deadline = time.monotonic() + 15.0
    while True:
        _check_cancel(cancel_event)
        try:
            os.replace(source, target)
            return
        except PermissionError as exc:
            if time.monotonic() >= deadline:
                raise DependencyInstallError(
                    f"Windows kept the validated runtime locked for 15 seconds, so it was not activated: {source}"
                ) from exc
            time.sleep(0.25)


def _atomic_manifest(root: Path, payload: dict[str, object]) -> Path:
    target = _manifest_path(root)
    temporary = root / f".active-{uuid.uuid4().hex}.tmp"
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    return target


def _remove_staging(path: Path, staging_root: Path) -> None:
    try:
        resolved = path.resolve()
        root = staging_root.resolve()
    except OSError:
        return
    if resolved.parent == root and resolved.name.startswith("install-"):
        shutil.rmtree(resolved, ignore_errors=True)


def install_latest_dependencies(
    *,
    components: Iterable[str] = ("ffmpeg", "vapoursynth"),
    app_directory: Path | None = None,
    cancel_event: threading.Event | None = None,
    progress: ProgressCallback | None = None,
    log: LogCallback | None = None,
) -> DependencyInstallResult:
    selected = frozenset(components)
    if not selected or not selected <= {"ffmpeg", "vapoursynth"}:
        raise ValueError("components must contain ffmpeg and/or vapoursynth")
    cancel = cancel_event or threading.Event()
    root = managed_runtime_root(app_directory)
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise DependencyInstallError(
            f"The app-local runtime folder could not be created: {root}. Move the app to a writable folder and retry."
        ) from exc
    app_root = (app_directory or application_directory()).resolve()
    if root.resolve().parent != app_root:
        raise DependencyInstallError(
            f"The managed runtime path resolves outside the application folder, so installation was refused: {root}"
        )
    probe = root / f".write-test-{uuid.uuid4().hex}"
    try:
        probe.write_bytes(b"")
        probe.unlink()
    except OSError as exc:
        raise DependencyInstallError(
            f"The app-local runtime folder is not writable: {root}. Move the app to a writable folder and retry."
        ) from exc
    staging_root = root / ".staging"
    versions_root = root / "versions"
    try:
        staging_root.mkdir(exist_ok=True)
        versions_root.mkdir(exist_ok=True)
    except OSError as exc:
        raise DependencyInstallError(f"The managed runtime staging folders could not be created under {root}.") from exc
    if staging_root.resolve().parent != root.resolve() or versions_root.resolve().parent != root.resolve():
        raise DependencyInstallError("A managed-runtime subfolder resolves outside the app-local runtime.")
    staging = staging_root / f"install-{uuid.uuid4().hex}"
    staging.mkdir()
    downloads = staging / "downloads"
    downloads.mkdir()
    if log:
        log(f"App-local runtime root: {root}")
        log("No system PATH, registry, system Python, or existing tool installation will be modified.")
    ffmpeg_path: Path | None = None
    ffprobe_path: Path | None = None
    vspipe_path: Path | None = None
    ffmpeg_version: str | None = None
    vs_version: str | None = None
    nnedi3vk_version: str | None = None
    plan: DependencyReleasePlan | None = None
    moved: list[Path] = []
    activated = False
    try:
        if progress:
            progress("metadata", "Resolving current stable releases", None, None)
        plan = resolve_latest_releases()
        if log:
            log(
                f"Resolved FFmpeg {plan.ffmpeg_version}, VapourSynth {plan.vapoursynth_version}, "
                f"Python {plan.python_version}, pip {plan.pip_version}, VSJetpack {plan.vsjetpack_version}."
            )
        _check_cancel(cancel)
        component_records: dict[str, object] = {}
        previous = _read_active_manifest(root).get("components")
        if isinstance(previous, dict):
            component_records.update(previous)

        if "ffmpeg" in selected:
            archive = downloads / plan.ffmpeg_asset.name
            ffmpeg_hash = _download(plan.ffmpeg_asset, archive, cancel, progress, "ffmpeg-download")
            candidate = staging / "ffmpeg-candidate"
            safe_extract_zip(archive, candidate, cancel, progress, "ffmpeg-extract")
            ffmpeg_candidate = _find_exactly_one(candidate, "ffmpeg.exe")
            ffprobe_candidate = _find_exactly_one(candidate, "ffprobe.exe")
            if ffmpeg_candidate.parent != ffprobe_candidate.parent:
                raise DependencyInstallError("Staged FFmpeg and FFprobe were not in the same bin directory.")
            if progress:
                progress("ffmpeg-validate", "Validating FFmpeg filters, encoders, and FFprobe", None, None)
            ffmpeg_version = _validate_ffmpeg(ffmpeg_candidate, ffprobe_candidate, cancel, log)
            target = versions_root / (
                f"ffmpeg-{_slug(plan.ffmpeg_version)}-{ffmpeg_hash[:12]}-{uuid.uuid4().hex[:8]}"
            )
            _move_validated_directory(candidate, target, cancel)
            moved.append(target)
            ffmpeg_path = target / ffmpeg_candidate.relative_to(candidate)
            ffprobe_path = target / ffprobe_candidate.relative_to(candidate)
            component_records["ffmpeg"] = {
                "version": plan.ffmpeg_version,
                "reported_version": ffmpeg_version,
                "asset": plan.ffmpeg_asset.name,
                "sha256": ffmpeg_hash,
                "ffmpeg": str(ffmpeg_path.relative_to(root)),
                "ffprobe": str(ffprobe_path.relative_to(root)),
            }

        if "vapoursynth" in selected:
            candidate = staging / "vapoursynth-candidate"
            _python, vspipe_candidate, hashes = _prepare_python(
                plan, downloads, candidate, cancel, progress, log
            )
            if progress:
                progress("qtgmc-validate", "Validating BestSource and the maximum-fidelity QTGMC graph", None, None)
            vs_version, vulkan_ready, vulkan_diagnostic, nnedi3vk_version = _validate_vapoursynth(
                vspipe_candidate,
                cancel,
                log,
            )
            target = versions_root / (
                f"vapoursynth-{_slug(plan.vapoursynth_version)}-py{_slug(plan.python_version)}-"
                f"{hashes['vapoursynth'][:12]}-{uuid.uuid4().hex[:8]}"
            )
            _move_validated_directory(candidate, target, cancel)
            moved.append(target)
            vspipe_path = target / vspipe_candidate.relative_to(candidate)
            component_records["vapoursynth"] = {
                "version": plan.vapoursynth_version,
                "reported_version": vs_version,
                "python_version": plan.python_version,
                "pip_version": plan.pip_version,
                "vsjetpack_version": plan.vsjetpack_version,
                "nnedi3vk_version": nnedi3vk_version,
                "nnedi3vk_asset": plan.nnedi3vk_asset.name if plan.nnedi3vk_asset else None,
                "nnedi3vk_sha256": hashes.get("nnedi3vk"),
                "nnedi3vk_graph_ready": vulkan_ready,
                "nnedi3vk_graph_diagnostic": vulkan_diagnostic,
                "asset": plan.vapoursynth_asset.name,
                "asset_sha256": hashes["vapoursynth"],
                "python_archive_sha256": hashes["python"],
                "pip_sha256": hashes["pip"],
                "vspipe": str(vspipe_path.relative_to(root)),
            }

        manifest = {
            "schema": 1,
            "installed_at": datetime.now(timezone.utc).isoformat(),
            "application_version": __version__,
            "components": component_records,
        }
        if progress:
            progress("activate", "Atomically activating the validated app-local runtime", None, None)
        manifest_path = _atomic_manifest(root, manifest)
        activated = True
        if log:
            log(f"Activated the validated app-local runtime: {manifest_path}")
        return DependencyInstallResult(
            runtime_root=root,
            ffmpeg_path=ffmpeg_path or active_managed_binary("ffmpeg", app_directory),
            ffprobe_path=ffprobe_path or active_managed_binary("ffprobe", app_directory),
            vspipe_path=vspipe_path or active_managed_binary("vspipe", app_directory),
            ffmpeg_version=ffmpeg_version,
            vapoursynth_version=vs_version,
            vsjetpack_version=plan.vsjetpack_version,
            nnedi3vk_version=nnedi3vk_version,
            manifest_path=manifest_path,
        )
    except Exception:
        # active.json is written only after both candidates validate and move;
        # therefore failure leaves the previously active runtime selected.
        if not activated:
            for directory in reversed(moved):
                try:
                    resolved = directory.resolve()
                    if resolved.parent == versions_root.resolve() and resolved.name.startswith(
                        ("ffmpeg-", "vapoursynth-")
                    ):
                        shutil.rmtree(resolved)
                except OSError:
                    if log:
                        log(f"Could not remove inactive staged version directory: {directory}")
        raise
    finally:
        _remove_staging(staging, staging_root)
