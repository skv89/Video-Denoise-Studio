from __future__ import annotations

import hashlib
import gzip
import json
import shutil
import tempfile
import threading
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from deinterlace_studio.capabilities import (
    _ToolVersionEvidence,
    _automatic_ffmpeg_directories,
    discover_ffmpeg_toolchain,
    find_binary,
)
from deinterlace_studio.dependencies import (
    DependencyInstallError,
    DependencyReleasePlan,
    ReleaseAsset,
    _atomic_manifest,
    _download,
    _github_asset,
    _latest_python_embed,
    _pypi_wheel,
    _powershell_signature,
    _read_url,
    _safe_download_name,
    _validate_pip_install_report,
    _validate_vapoursynth,
    active_managed_binary,
    dependency_issues,
    install_latest_dependencies,
    managed_runtime_environment,
    managed_runtime_root,
    resolve_latest_releases,
    safe_extract_zip,
)
from deinterlace_studio.models import CapabilityReport
from deinterlace_studio.tool_versions import (
    FFMPEG_9_LIBRARY_FLOOR,
    assess_ffmpeg_pair_versions,
    parse_ffmpeg_git_revision,
    parse_ffmpeg_library_versions,
    parse_stable_ffmpeg_version,
)


CURRENT_GIT_LIBRARIES = {
    "libavutil": (61, 5, 100),
    "libavcodec": (63, 7, 100),
    "libavformat": (63, 5, 101),
    "libavfilter": (12, 3, 101),
}


def version_evidence(
    tool: str,
    banner: str,
    libraries: dict[str, tuple[int, int, int]] | None = None,
) -> _ToolVersionEvidence:
    version = f"{tool} version {banner}"
    library_text = "\n".join(
        f"{name:<16} {major}. {minor:2d}.{micro:3d} / {major}. {minor:2d}.{micro:3d}"
        for name, (major, minor, micro) in (libraries or {}).items()
    )
    return _ToolVersionEvidence(version, version + ("\n" + library_text if library_text else ""), None)


def ready_capabilities() -> CapabilityReport:
    return CapabilityReport(
        ffmpeg_path=Path("C:/tools/ffmpeg.exe"),
        ffprobe_path=Path("C:/tools/ffprobe.exe"),
        ffmpeg_version="ffmpeg version 9.0",
        ffmpeg_configuration="",
        filters=frozenset({"idet", "bwdif", "fftdnoiz", "atadenoise"}),
        encoders=frozenset({"libx265", "libaom-av1", "libsvtav1", "ffv1", "prores_ks", "dnxhd"}),
        encoder_pixel_formats={},
        hwaccels=frozenset(),
        vspipe_path=Path("C:/tools/vspipe.exe"),
        vapoursynth_version="R78",
        qtgmc_ready=True,
        qtgmc_diagnostic="ready",
        qtgmc_install_command=None,
        ffprobe_version="ffprobe version 9.0",
        denoise_capabilities={
            "ffmpeg_fftdnoiz": True,
            "ffmpeg_atadenoise": True,
            "vs_bm3d": True,
            "vs_dfttest": True,
            "vs_mvtools": True,
            "vs_nlmeans": True,
        },
        denoise_backends={
            "ffmpeg_fftdnoiz": "ffmpeg",
            "ffmpeg_atadenoise": "ffmpeg",
            "vs_bm3d": "bm3dcpu",
            "vs_dfttest": "dfttest_cpu",
            "vs_mvtools": "mvtools",
            "vs_nlmeans": "nlm_ispc",
        },
    )


class FakeResponse:
    def __init__(self, data: bytes, url: str = "https://example.test/file") -> None:
        self.data = data
        self.url = url
        self.offset = 0
        self.headers = {"Content-Length": str(len(data))}

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def geturl(self) -> str:
        return self.url

    def read(self, size: int = -1) -> bytes:
        if self.offset >= len(self.data):
            return b""
        if size < 0:
            size = len(self.data) - self.offset
        chunk = self.data[self.offset : self.offset + size]
        self.offset += len(chunk)
        return chunk


class FFmpegVersionEvidenceTests(unittest.TestCase):
    def test_git_revision_parser_recognizes_gyan_and_standard_snapshot_banners(self) -> None:
        self.assertEqual(
            parse_ffmpeg_git_revision("ffmpeg version 2026-08-06-git-95c43d7df7-full_build-www.gyan.dev"),
            "95c43d7df7",
        )
        self.assertEqual(
            parse_ffmpeg_git_revision("ffprobe version N-121328-ge05f8acabf-20251005"),
            "e05f8acabf",
        )
        self.assertEqual(
            parse_ffmpeg_git_revision("ffmpeg version n9.0-5-gabcdef1234"),
            "abcdef1234",
        )
        self.assertIsNone(parse_stable_ffmpeg_version("ffmpeg version n9.0-5-gabcdef1234"))
        self.assertIsNone(parse_ffmpeg_git_revision("ffmpeg version 2026-08-06-git-build"))

    def test_library_parser_handles_ffmpeg_aligned_version_columns(self) -> None:
        text = (
            "ffmpeg version test\n"
            "libavutil      61.  5.100 / 61.  5.100\n"
            "libavcodec     63.  7.100 / 63.  7.100\n"
            "libavformat    63.  5.101 / 63.  5.101\n"
            "libavfilter    12.  3.101 / 12.  3.101\n"
        )
        self.assertEqual(parse_ffmpeg_library_versions(text), CURRENT_GIT_LIBRARIES)

    def test_matching_git_revision_and_ffmpeg9_libraries_are_verified(self) -> None:
        ffmpeg = version_evidence(
            "ffmpeg", "2026-08-06-git-95c43d7df7-full_build-www.gyan.dev", CURRENT_GIT_LIBRARIES
        )
        ffprobe = version_evidence(
            "ffprobe", "2026-08-06-git-95c43d7df7-full_build-www.gyan.dev", CURRENT_GIT_LIBRARIES
        )
        assessment = assess_ffmpeg_pair_versions(ffmpeg.text, ffprobe.text)
        self.assertTrue(assessment.compatible)
        self.assertEqual(assessment.kind, "verified_git")
        self.assertEqual(assessment.ffmpeg_git_revision, "95c43d7df7")
        self.assertIn("FFmpeg 9.0 library floor is met", assessment.detail)

    def test_git_revision_mismatch_fails_closed(self) -> None:
        assessment = assess_ffmpeg_pair_versions(
            version_evidence("ffmpeg", "2026-08-06-git-95c43d7df7", CURRENT_GIT_LIBRARIES).text,
            version_evidence("ffprobe", "2026-08-06-git-a5c43d7df7", CURRENT_GIT_LIBRARIES).text,
        )
        self.assertFalse(assessment.compatible)
        self.assertIn("mismatched Git revisions", assessment.detail)

    def test_tag_descendant_git_revisions_cannot_be_misclassified_as_stable(self) -> None:
        assessment = assess_ffmpeg_pair_versions(
            version_evidence("ffmpeg", "n9.0-5-gabcdef1234", CURRENT_GIT_LIBRARIES).text,
            version_evidence("ffprobe", "n9.0-5-gbbcdef1234", CURRENT_GIT_LIBRARIES).text,
        )
        self.assertFalse(assessment.compatible)
        self.assertEqual(assessment.kind, "unverified")
        self.assertIn("mismatched Git revisions", assessment.detail)

    def test_missing_or_mismatched_git_libraries_fail_closed(self) -> None:
        incomplete = dict(CURRENT_GIT_LIBRARIES)
        incomplete.pop("libavfilter")
        missing = assess_ffmpeg_pair_versions(
            version_evidence("ffmpeg", "2026-08-06-git-95c43d7df7", incomplete).text,
            version_evidence("ffprobe", "2026-08-06-git-95c43d7df7", CURRENT_GIT_LIBRARIES).text,
        )
        mismatched_libraries = dict(CURRENT_GIT_LIBRARIES)
        mismatched_libraries["libavformat"] = (63, 5, 100)
        mismatched = assess_ffmpeg_pair_versions(
            version_evidence("ffmpeg", "2026-08-06-git-95c43d7df7", CURRENT_GIT_LIBRARIES).text,
            version_evidence("ffprobe", "2026-08-06-git-95c43d7df7", mismatched_libraries).text,
        )
        self.assertFalse(missing.compatible)
        self.assertIn("proof is incomplete", missing.detail)
        self.assertFalse(mismatched.compatible)
        self.assertIn("library versions differ", mismatched.detail)

    def test_matching_pre9_git_libraries_fail_closed(self) -> None:
        old_libraries = {
            "libavutil": (60, 13, 100),
            "libavcodec": (62, 16, 100),
            "libavformat": (62, 6, 100),
            "libavfilter": (11, 9, 100),
        }
        assessment = assess_ffmpeg_pair_versions(
            version_evidence("ffmpeg", "N-121328-ge05f8acabf-20251005", old_libraries).text,
            version_evidence("ffprobe", "N-121328-ge05f8acabf-20251005", old_libraries).text,
        )
        self.assertFalse(assessment.compatible)
        self.assertIn("predates the FFmpeg 9.0 library floor", assessment.detail)
        self.assertEqual(set(FFMPEG_9_LIBRARY_FLOOR), set(old_libraries))


class MetadataTests(unittest.TestCase):
    def test_asset_filename_cannot_escape_download_directory(self) -> None:
        for name in ("../tool.zip", "subdir/tool.zip", "C:tool.zip", "CON.zip"):
            with self.subTest(name=name), self.assertRaises(DependencyInstallError):
                _safe_download_name(name)

    def test_metadata_reader_boundedly_decodes_gzip(self) -> None:
        encoded = gzip.compress(b"release metadata")
        with patch("deinterlace_studio.dependencies.urlopen", return_value=FakeResponse(encoded)):
            self.assertEqual(_read_url("https://example.test/metadata"), b"release metadata")

    def test_github_asset_requires_exact_name_and_published_sha256(self) -> None:
        digest = "ab" * 32
        payload = {
            "draft": False,
            "prerelease": False,
            "assets": [
                {
                    "name": "tool.zip",
                    "browser_download_url": "https://github.com/example/tool.zip",
                    "digest": f"sha256:{digest}",
                    "size": 123,
                }
            ],
        }
        asset = _github_asset(payload, "tool.zip")
        self.assertEqual(asset.sha256, digest)
        with self.assertRaises(DependencyInstallError):
            _github_asset({**payload, "assets": [payload["assets"][0], payload["assets"][0]]}, "tool.zip")

    def test_pypi_wheel_uses_current_non_yanked_hash(self) -> None:
        digest = "cd" * 32
        payload = {
            "info": {"version": "25.2"},
            "urls": [
                {
                    "filename": "pip-25.2-py3-none-any.whl",
                    "packagetype": "bdist_wheel",
                    "url": "https://files.pythonhosted.org/pip.whl",
                    "size": 100,
                    "digests": {"sha256": digest},
                    "yanked": False,
                }
            ],
        }
        version, asset = _pypi_wheel(payload, "pip", "-py3-none-any.whl")
        self.assertEqual(version, "25.2")
        self.assertEqual(asset.sha256, digest)

    def test_release_resolver_records_the_official_optional_vulkan_wheel(self) -> None:
        ffmpeg = ReleaseAsset("ffmpeg.zip", "https://example.test/ffmpeg.zip", "1" * 64, 1)
        vapoursynth = ReleaseAsset("vapoursynth.zip", "https://example.test/vs.zip", "2" * 64, 1)
        python = ReleaseAsset("python.zip", "https://example.test/python.zip", "3" * 64, 1)
        pip = ReleaseAsset("pip.whl", "https://example.test/pip.whl", "4" * 64, 1)
        nnedi3vk = ReleaseAsset(
            "vapoursynth_nnedi3vk-1.0-py3-none-win_amd64.whl",
            "https://example.test/nnedi3vk.whl",
            "5" * 64,
            13_102_591,
        )
        payloads = (
            {"tag_name": "9.0"},
            {"tag_name": "R79"},
            {},
            {"info": {"version": "2.2.0"}},
            {},
        )
        with patch("deinterlace_studio.dependencies._read_json", side_effect=payloads), patch(
            "deinterlace_studio.dependencies._github_asset",
            side_effect=(ffmpeg, vapoursynth),
        ), patch(
            "deinterlace_studio.dependencies._latest_python_embed",
            return_value=("3.14.7", python),
        ), patch(
            "deinterlace_studio.dependencies._pypi_wheel",
            side_effect=(("26.2.1", pip), ("1.0", nnedi3vk)),
        ):
            plan = resolve_latest_releases()
        self.assertEqual(plan.nnedi3vk_version, "1.0")
        self.assertEqual(plan.nnedi3vk_asset, nnedi3vk)

    def test_optional_vulkan_metadata_failure_does_not_block_cpu_runtime_plan(self) -> None:
        ffmpeg = ReleaseAsset("ffmpeg.zip", "https://example.test/ffmpeg.zip", "1" * 64, 1)
        vapoursynth = ReleaseAsset("vapoursynth.zip", "https://example.test/vs.zip", "2" * 64, 1)
        python = ReleaseAsset("python.zip", "https://example.test/python.zip", "3" * 64, 1)
        pip = ReleaseAsset("pip.whl", "https://example.test/pip.whl", "4" * 64, 1)

        def wheel(_payload, package, _suffix):
            if package == "pip":
                return "26.2.1", pip
            raise DependencyInstallError("optional metadata unavailable")

        payloads = (
            {"tag_name": "9.0"},
            {"tag_name": "R79"},
            {},
            {"info": {"version": "2.2.0"}},
            {},
        )
        with patch("deinterlace_studio.dependencies._read_json", side_effect=payloads), patch(
            "deinterlace_studio.dependencies._github_asset",
            side_effect=(ffmpeg, vapoursynth),
        ), patch(
            "deinterlace_studio.dependencies._latest_python_embed",
            return_value=("3.14.7", python),
        ), patch("deinterlace_studio.dependencies._pypi_wheel", side_effect=wheel):
            plan = resolve_latest_releases()
        self.assertIsNone(plan.nnedi3vk_version)
        self.assertIsNone(plan.nnedi3vk_asset)

    def test_python_index_selects_highest_supported_patch(self) -> None:
        listing = b'<a href="3.14.0/">3.14.0</a><a href="3.14.3/">3.14.3</a><a href="3.13.9/">3.13.9</a>'
        digest = "ef" * 32
        preceding_row = f'<tr><td>other</td><td><code class="checksum">{"aa" * 32}</code></td></tr>'
        release_page = (preceding_row + (
            '<tr><td><a href="https://www.python.org/ftp/python/3.14.3/'
            'python-3.14.3-embed-amd64.zip">Windows embeddable package</a></td>'
            f'<td><code class="checksum"><span>{digest[:32]}</span><wbr><span>{digest[32:]}</span></code></td></tr>'
        )).encode()
        with patch("deinterlace_studio.dependencies._read_url", side_effect=(listing, release_page)):
            version, asset = _latest_python_embed()
        self.assertEqual(version, "3.14.3")
        self.assertTrue(asset.url.endswith("/3.14.3/python-3.14.3-embed-amd64.zip"))
        self.assertEqual(asset.sha256, digest)

    def test_pip_report_requires_hashed_https_and_each_requested_distribution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            valid_item = {
                "download_info": {
                    "url": "https://files.pythonhosted.org/package.whl",
                    "archive_info": {"hashes": {"sha256": "a" * 64}},
                },
                "requested": True,
                "metadata": {"name": "VapourSynth_BM3DCUDA"},
            }
            report.write_text(json.dumps({"install": [valid_item]}), encoding="utf-8")
            _validate_pip_install_report(
                report,
                expected_requested=frozenset({"vapoursynth-bm3dcuda"}),
            )

            dft_item = {
                "download_info": {
                    "url": "https://files.pythonhosted.org/dfttest.whl",
                    "archive_info": {"hashes": {"sha256": "b" * 64}},
                },
                "requested": True,
                "metadata": {"name": "VapourSynth_DFTTest2_NVRTC"},
            }
            report.write_text(json.dumps({"install": [valid_item, dft_item]}), encoding="utf-8")
            _validate_pip_install_report(
                report,
                expected_requested=frozenset(
                    {"vapoursynth-bm3dcuda", "vapoursynth-dfttest2-nvrtc"}
                ),
            )

            with self.assertRaisesRegex(DependencyInstallError, "requested package"):
                _validate_pip_install_report(
                    report,
                    expected_requested=frozenset({"vapoursynth-bm3dcuda", "vapoursynth-vszipcu"}),
                )

            report.write_text(json.dumps({"install": []}), encoding="utf-8")
            with self.assertRaisesRegex(DependencyInstallError, "no installed packages"):
                _validate_pip_install_report(report)

            invalid_item = {
                **valid_item,
                "download_info": {
                    "url": "http://files.pythonhosted.org/package.whl",
                    "archive_info": {"hashes": {}},
                },
            }
            report.write_text(json.dumps({"install": [invalid_item]}), encoding="utf-8")
            with self.assertRaisesRegex(DependencyInstallError, "HTTPS source"):
                _validate_pip_install_report(report)


class ArchiveAndDownloadSafetyTests(unittest.TestCase):
    def test_checksum_mismatch_removes_download(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "download.zip"
            asset = ReleaseAsset("download.zip", "https://example.test/file", "0" * 64, 7)
            with patch("deinterlace_studio.dependencies.urlopen", return_value=FakeResponse(b"payload")):
                with self.assertRaisesRegex(DependencyInstallError, "SHA-256 mismatch"):
                    _download(asset, destination, threading.Event(), None, "test")
            self.assertFalse(destination.exists())

    def test_cancel_removes_partial_download(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "download.zip"
            data = b"x" * (1024 * 1024 + 10)
            asset = ReleaseAsset("download.zip", "https://example.test/file", hashlib.sha256(data).hexdigest(), len(data))
            cancel = threading.Event()

            def progress(*_args) -> None:
                cancel.set()

            with patch("deinterlace_studio.dependencies.urlopen", return_value=FakeResponse(data)):
                with self.assertRaisesRegex(Exception, "canceled"):
                    _download(asset, destination, cancel, progress, "test")
            self.assertFalse(destination.exists())

    def test_traversal_archive_is_rejected_without_escape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "malicious.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("../escaped.txt", "bad")
            with self.assertRaisesRegex(DependencyInstallError, "traversal"):
                safe_extract_zip(archive, root / "extract")
            self.assertFalse((root / "escaped.txt").exists())

    def test_duplicate_casefolded_paths_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "duplicate.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("bin/tool.exe", "one")
                bundle.writestr("BIN/TOOL.EXE", "two")
            with self.assertRaisesRegex(DependencyInstallError, "duplicate normalized path"):
                safe_extract_zip(archive, root / "extract")

    def test_signature_check_uses_windows_inbox_security_module_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "python.exe"
            executable.write_bytes(b"fixture")
            captured: dict[str, str] = {}

            def fake_run(_args, _cancel, _log, **kwargs):
                captured.update(kwargs["env"])
                return '{"Status":"Valid","Subject":"CN=Python Software Foundation"}'

            with patch("deinterlace_studio.dependencies.os.name", "nt"), patch(
                "deinterlace_studio.dependencies._run_process", side_effect=fake_run
            ):
                _powershell_signature(executable, threading.Event(), None)
            self.assertTrue(captured["PSModulePath"].endswith("WindowsPowerShell\\v1.0\\Modules"))


class OptionalVulkanRuntimeTests(unittest.TestCase):
    def test_vulkan_graph_is_optional_but_its_evidence_is_returned_when_ready(self) -> None:
        required_denoisers = (
            {
                "vs_bm3d": True,
                "vs_dfttest": True,
                "vs_mvtools": True,
                "vs_nlmeans": True,
            },
            {
                "vs_bm3d": "bm3dcpu",
                "vs_dfttest": "dfttest_cpu",
                "vs_mvtools": "mvtools",
                "vs_nlmeans": "nlm_ispc",
            },
            {},
        )
        with patch(
            "deinterlace_studio.capabilities._inspect_vapoursynth",
            return_value=("R79", True, "CPU QTGMC passed", None),
        ), patch(
            "deinterlace_studio.capabilities._inspect_vapoursynth_denoisers",
            return_value=required_denoisers,
        ), patch(
            "deinterlace_studio.capabilities._inspect_vulkan_nnedi3",
            return_value=(False, "Vulkan 1.4 device unavailable", None),
        ):
            version, ready, diagnostic, package = _validate_vapoursynth(
                Path("vspipe.exe"), threading.Event(), None
            )
        self.assertEqual(version, "R79")
        self.assertFalse(ready)
        self.assertIn("unavailable", diagnostic)
        self.assertIsNone(package)

        with patch(
            "deinterlace_studio.capabilities._inspect_vapoursynth",
            return_value=("R79", True, "CPU QTGMC passed", None),
        ), patch(
            "deinterlace_studio.capabilities._inspect_vapoursynth_denoisers",
            return_value=required_denoisers,
        ), patch(
            "deinterlace_studio.capabilities._inspect_vulkan_nnedi3",
            return_value=(True, "real graph emitted eight frames", "1.0"),
        ):
            _version, ready, diagnostic, package = _validate_vapoursynth(
                Path("vspipe.exe"), threading.Event(), None
            )
        self.assertTrue(ready)
        self.assertEqual(package, "1.0")
        self.assertIn("eight frames", diagnostic)


class DiscoveryAndActivationTests(unittest.TestCase):
    def test_active_manifest_uses_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            app = Path(directory) / "app"
            root = managed_runtime_root(app)
            binary = root / "versions" / "ffmpeg-test" / "bin" / "ffmpeg.exe"
            binary.parent.mkdir(parents=True)
            binary.write_bytes(b"exe")
            _atomic_manifest(
                root,
                {
                    "schema": 1,
                    "components": {"ffmpeg": {"ffmpeg": str(binary.relative_to(root))}},
                },
            )
            self.assertEqual(active_managed_binary("ffmpeg", app), binary.resolve())
            payload = json.loads((root / "active.json").read_text(encoding="utf-8"))
            self.assertFalse(Path(payload["components"]["ffmpeg"]["ffmpeg"]).is_absolute())

    def test_explicit_binary_precedes_managed_and_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            explicit = root / "explicit.exe"
            managed = root / "managed.exe"
            on_path = root / "path.exe"
            for item in (explicit, managed, on_path):
                item.write_bytes(b"exe")
            with patch("deinterlace_studio.capabilities.active_managed_binary", return_value=managed), patch(
                "deinterlace_studio.capabilities.shutil.which", return_value=str(on_path)
            ):
                self.assertEqual(find_binary("ffmpeg", explicit), explicit.resolve())
                self.assertEqual(find_binary("ffmpeg"), managed.resolve())

    def test_automatic_discovery_merges_fresh_registry_path_entries(self) -> None:
        fresh = Path(r"C:\newly-installed\ffmpeg\bin")
        with patch.dict("deinterlace_studio.capabilities.os.environ", {"PATH": ""}), patch(
            "deinterlace_studio.capabilities._windows_registry_path_entries",
            return_value=[(fresh, "user registry PATH entry 1")],
        ), patch("deinterlace_studio.capabilities.shutil.which", return_value=None):
            directories = _automatic_ffmpeg_directories()
        self.assertIn((fresh, "user registry PATH entry 1"), directories)

    def test_auto_discovery_prefers_later_confirmed_stable_pair_over_first_git_pair(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            git_bin = root / "git" / "bin"
            stable_bin = root / "stable" / "bin"
            for folder in (git_bin, stable_bin):
                folder.mkdir(parents=True)
                for name in ("ffmpeg.exe", "ffprobe.exe"):
                    (folder / name).write_bytes(b"fixture")

            def version_probe(executable: Path, tool: str):
                if executable.parent == git_bin.resolve():
                    return version_evidence(
                        tool,
                        "2026-08-03-git-01a25f74cc-full_build",
                        CURRENT_GIT_LIBRARIES,
                    )
                return version_evidence(tool, "9.0-full_build")

            with patch("deinterlace_studio.capabilities.active_managed_binary", return_value=None), patch(
                "deinterlace_studio.capabilities._automatic_ffmpeg_directories",
                return_value=[(git_bin, "process PATH entry 1"), (stable_bin, "process PATH entry 9")],
            ), patch("deinterlace_studio.capabilities._tool_version_evidence", side_effect=version_probe):
                ffmpeg, ffprobe, source, diagnostics = discover_ffmpeg_toolchain()

            self.assertEqual(ffmpeg, (stable_bin / "ffmpeg.exe").resolve())
            self.assertEqual(ffprobe, (stable_bin / "ffprobe.exe").resolve())
            self.assertEqual(source, "process PATH entry 9")
            self.assertTrue(any(line.startswith("NOT SELECTED [process PATH entry 1]") for line in diagnostics))
            self.assertTrue(any(line.startswith("SELECTED [process PATH entry 9]") for line in diagnostics))

    def test_auto_discovery_prefers_the_newest_compatible_stable_release(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stable_nine = root / "stable-9"
            stable_ten = root / "stable-10"
            for folder in (stable_nine, stable_ten):
                folder.mkdir()
                for name in ("ffmpeg.exe", "ffprobe.exe"):
                    (folder / name).write_bytes(b"fixture")

            def version_probe(executable: Path, tool: str):
                version = "9.0" if executable.parent == stable_nine.resolve() else "10.1.2"
                return version_evidence(tool, version)

            with patch("deinterlace_studio.capabilities.active_managed_binary", return_value=None), patch(
                "deinterlace_studio.capabilities._automatic_ffmpeg_directories",
                return_value=[(stable_nine, "process PATH entry 1"), (stable_ten, "process PATH entry 2")],
            ), patch("deinterlace_studio.capabilities._tool_version_evidence", side_effect=version_probe):
                ffmpeg, ffprobe, source, _diagnostics = discover_ffmpeg_toolchain()

            self.assertEqual(ffmpeg, (stable_ten / "ffmpeg.exe").resolve())
            self.assertEqual(ffprobe, (stable_ten / "ffprobe.exe").resolve())
            self.assertEqual(source, "process PATH entry 2")

    def test_auto_discovery_selects_verified_git_when_no_compatible_stable_exists(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old_stable = root / "stable-8"
            verified_git = root / "git"
            for folder in (old_stable, verified_git):
                folder.mkdir()
                for name in ("ffmpeg.exe", "ffprobe.exe"):
                    (folder / name).write_bytes(b"fixture")

            def version_probe(executable: Path, tool: str):
                if executable.parent == old_stable.resolve():
                    return version_evidence(tool, "8.0-full_build")
                return version_evidence(
                    tool,
                    "2026-08-06-git-95c43d7df7-full_build-www.gyan.dev",
                    CURRENT_GIT_LIBRARIES,
                )

            with patch("deinterlace_studio.capabilities.active_managed_binary", return_value=None), patch(
                "deinterlace_studio.capabilities._automatic_ffmpeg_directories",
                return_value=[(old_stable, "process PATH entry 1"), (verified_git, "process PATH entry 2")],
            ), patch("deinterlace_studio.capabilities._tool_version_evidence", side_effect=version_probe):
                ffmpeg, ffprobe, source, diagnostics = discover_ffmpeg_toolchain()

            self.assertEqual(ffmpeg, (verified_git / "ffmpeg.exe").resolve())
            self.assertEqual(ffprobe, (verified_git / "ffprobe.exe").resolve())
            self.assertEqual(source, "process PATH entry 2")
            self.assertTrue(any("verified Git snapshot revision 95c43d7df7" in line for line in diagnostics))

    def test_mismatched_pair_cannot_outrank_a_matching_stable_pair(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mismatched = root / "mismatched"
            matching = root / "matching"
            for folder in (mismatched, matching):
                folder.mkdir()
                for name in ("ffmpeg.exe", "ffprobe.exe"):
                    (folder / name).write_bytes(b"fixture")

            def version_probe(executable: Path, tool: str):
                if executable.parent == mismatched.resolve():
                    return version_evidence(tool, "10.0" if tool == "ffmpeg" else "9.0")
                return version_evidence(tool, "9.0")

            with patch("deinterlace_studio.capabilities.active_managed_binary", return_value=None), patch(
                "deinterlace_studio.capabilities._automatic_ffmpeg_directories",
                return_value=[(mismatched, "process PATH entry 1"), (matching, "process PATH entry 2")],
            ), patch("deinterlace_studio.capabilities._tool_version_evidence", side_effect=version_probe):
                ffmpeg, ffprobe, source, diagnostics = discover_ffmpeg_toolchain()

            self.assertEqual(ffmpeg, (matching / "ffmpeg.exe").resolve())
            self.assertEqual(ffprobe, (matching / "ffprobe.exe").resolve())
            self.assertEqual(source, "process PATH entry 2")
            self.assertTrue(any("mismatched pair" in line for line in diagnostics))

    def test_auto_discovery_requires_ffmpeg_and_ffprobe_in_the_same_candidate_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            incomplete = root / "incomplete"
            paired = root / "paired"
            incomplete.mkdir()
            paired.mkdir()
            (incomplete / "ffmpeg.exe").write_bytes(b"fixture")
            for name in ("ffmpeg.exe", "ffprobe.exe"):
                (paired / name).write_bytes(b"fixture")

            with patch("deinterlace_studio.capabilities.active_managed_binary", return_value=None), patch(
                "deinterlace_studio.capabilities._automatic_ffmpeg_directories",
                return_value=[(incomplete, "process PATH entry 1"), (paired, "process PATH entry 2")],
            ), patch(
                "deinterlace_studio.capabilities._tool_version_evidence",
                side_effect=lambda _executable, tool: version_evidence(tool, "9.0"),
            ):
                ffmpeg, ffprobe, source, diagnostics = discover_ffmpeg_toolchain()

            self.assertEqual(ffmpeg, (paired / "ffmpeg.exe").resolve())
            self.assertEqual(ffprobe, (paired / "ffprobe.exe").resolve())
            self.assertEqual(source, "process PATH entry 2")
            self.assertEqual(sum(line.startswith("SELECTED [") for line in diagnostics), 1)

    def test_auto_discovery_preserves_path_precedence_when_no_pair_meets_the_release_floor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            git_bin = root / "git"
            old_stable_bin = root / "old-stable"
            for folder in (git_bin, old_stable_bin):
                folder.mkdir()
                for name in ("ffmpeg.exe", "ffprobe.exe"):
                    (folder / name).write_bytes(b"fixture")

            def version_probe(executable: Path, tool: str):
                banner = "2026-08-06-git-build" if executable.parent == git_bin.resolve() else "8.0-full_build"
                return version_evidence(tool, banner)

            with patch("deinterlace_studio.capabilities.active_managed_binary", return_value=None), patch(
                "deinterlace_studio.capabilities._automatic_ffmpeg_directories",
                return_value=[(git_bin, "process PATH entry 1"), (old_stable_bin, "process PATH entry 2")],
            ), patch("deinterlace_studio.capabilities._tool_version_evidence", side_effect=version_probe):
                ffmpeg, ffprobe, source, diagnostics = discover_ffmpeg_toolchain()

            self.assertEqual(ffmpeg, (git_bin / "ffmpeg.exe").resolve())
            self.assertEqual(ffprobe, (git_bin / "ffprobe.exe").resolve())
            self.assertEqual(source, "process PATH entry 1")
            self.assertTrue(any("confirmed stable release 8.0.0" in line for line in diagnostics))

    def test_explicit_ffmpeg_pair_remains_authoritative_and_auditable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            binary_dir = Path(directory)
            ffmpeg = binary_dir / "ffmpeg.exe"
            ffprobe = binary_dir / "ffprobe.exe"
            ffmpeg.write_bytes(b"fixture")
            ffprobe.write_bytes(b"fixture")
            with patch(
                "deinterlace_studio.capabilities._tool_version_evidence",
                side_effect=lambda _executable, tool: version_evidence(tool, "2026-08-03-git-build"),
            ), patch("deinterlace_studio.capabilities._automatic_ffmpeg_directories") as automatic:
                selected_ffmpeg, selected_ffprobe, source, diagnostics = discover_ffmpeg_toolchain(ffmpeg, ffprobe)

            automatic.assert_not_called()
            self.assertEqual(selected_ffmpeg, ffmpeg.resolve())
            self.assertEqual(selected_ffprobe, ffprobe.resolve())
            self.assertEqual(source, "explicit user selection")
            self.assertIn("unconfirmed Git/date-stamped", diagnostics[0])

    def test_portable_environment_is_child_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "Lib" / "site-packages" / "vapoursynth"
            package.mkdir(parents=True)
            for item in (root / "python.exe", root / "python3.dll", package / "vspipe.exe", package / "vsscript.dll"):
                item.write_bytes(b"test")
            base = {"PATH": "C:\\Windows"}
            env = managed_runtime_environment(package / "vspipe.exe", base)
            self.assertEqual(base, {"PATH": "C:\\Windows"})
            self.assertEqual(env["PYTHONHOME"], str(root.resolve()))
            self.assertEqual(env["VSSCRIPT_PATH"], str((package / "vsscript.dll").resolve()))

    def test_capability_assessment_accepts_ready_and_rejects_missing_graph(self) -> None:
        ready = ready_capabilities()
        self.assertEqual(dependency_issues(ready), {})
        broken = CapabilityReport(**{**ready.__dict__, "qtgmc_ready": False, "qtgmc_diagnostic": "missing"})
        self.assertIn("vapoursynth", dependency_issues(broken))
        missing_denoiser = CapabilityReport(
            **{
                **ready.__dict__,
                "denoise_capabilities": {**ready.denoise_capabilities, "vs_nlmeans": False},
            }
        )
        self.assertTrue(
            any("temporal NLMeans graph" in issue for issue in dependency_issues(missing_denoiser)["vapoursynth"])
        )
        missing_ffmpeg_filter = CapabilityReport(
            **{**ready.__dict__, "filters": ready.filters - {"fftdnoiz"}}
        )
        self.assertTrue(any("fftdnoiz" in issue for issue in dependency_issues(missing_ffmpeg_filter)["ffmpeg"]))

    def test_capability_assessment_requires_confirmed_ffmpeg_9_or_newer(self) -> None:
        ready = ready_capabilities()
        old = CapabilityReport(
            **{
                **ready.__dict__,
                "ffmpeg_version": "ffmpeg version 8.0-full_build",
                "ffprobe_version": "ffprobe version 8.0-full_build",
            }
        )
        unknown = CapabilityReport(
            **{**ready.__dict__, "ffmpeg_version": "ffmpeg version 2026-08-03-git-01a25f74cc-full_build"}
        )
        newer = CapabilityReport(
            **{
                **ready.__dict__,
                "ffmpeg_version": "ffmpeg version n10.1.2",
                "ffprobe_version": "ffprobe version n10.1.2",
            }
        )
        self.assertIn("FFmpeg 9.0 or newer is required", dependency_issues(old)["ffmpeg"])
        self.assertTrue(any("could not verify" in issue for issue in dependency_issues(unknown)["ffmpeg"]))
        self.assertNotIn("ffmpeg", dependency_issues(newer))

    def test_capability_assessment_accepts_only_fully_verified_git_pair(self) -> None:
        ready = ready_capabilities()
        git_fields = {
            **ready.__dict__,
            "ffmpeg_version": "ffmpeg version 2026-08-06-git-95c43d7df7-full_build-www.gyan.dev",
            "ffprobe_version": "ffprobe version 2026-08-06-git-95c43d7df7-full_build-www.gyan.dev",
            "ffmpeg_library_versions": CURRENT_GIT_LIBRARIES,
            "ffprobe_library_versions": CURRENT_GIT_LIBRARIES,
        }
        verified = CapabilityReport(**git_fields)
        mismatched_revision = CapabilityReport(
            **{**git_fields, "ffprobe_version": "ffprobe version 2026-08-06-git-a5c43d7df7-full_build"}
        )
        pre9_libraries = {
            "libavutil": (60, 13, 100),
            "libavcodec": (62, 16, 100),
            "libavformat": (62, 6, 100),
            "libavfilter": (11, 9, 100),
        }
        pre9 = CapabilityReport(
            **{
                **git_fields,
                "ffmpeg_version": "ffmpeg version N-121328-ge05f8acabf-20251005",
                "ffprobe_version": "ffprobe version N-121328-ge05f8acabf-20251005",
                "ffmpeg_library_versions": pre9_libraries,
                "ffprobe_library_versions": pre9_libraries,
            }
        )
        self.assertNotIn("ffmpeg", dependency_issues(verified))
        self.assertTrue(any("mismatched Git revisions" in issue for issue in dependency_issues(mismatched_revision)["ffmpeg"]))
        self.assertTrue(any("predates the FFmpeg 9.0 library floor" in issue for issue in dependency_issues(pre9)["ffmpeg"]))

    def test_failed_activation_preserves_previous_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            app = workspace / "app"
            root = managed_runtime_root(app)
            root.mkdir(parents=True)
            original = b'{"schema":1,"components":{}}\n'
            (root / "active.json").write_bytes(original)
            fixture = workspace / "ffmpeg.zip"
            with zipfile.ZipFile(fixture, "w") as bundle:
                bundle.writestr("ffmpeg-test/bin/ffmpeg.exe", "ffmpeg")
                bundle.writestr("ffmpeg-test/bin/ffprobe.exe", "ffprobe")
            digest = hashlib.sha256(fixture.read_bytes()).hexdigest()
            asset = ReleaseAsset("ffmpeg.zip", "https://example.test/ffmpeg.zip", digest, fixture.stat().st_size)
            plan = DependencyReleasePlan(
                ffmpeg_version="9.0",
                ffmpeg_asset=asset,
                vapoursynth_version="R78",
                vapoursynth_asset=ReleaseAsset("vs.zip", "https://example.test/vs.zip", "1" * 64, 1),
                python_version="3.14.3",
                python_asset=ReleaseAsset("python.zip", "https://example.test/python.zip", None, None),
                pip_version="25.2",
                pip_asset=ReleaseAsset("pip.whl", "https://example.test/pip.whl", "2" * 64, 1),
                vsjetpack_version="2.2.0",
            )

            def copy_fixture(_asset, destination, *_args):
                shutil.copy2(fixture, destination)
                return digest

            with patch("deinterlace_studio.dependencies.resolve_latest_releases", return_value=plan), patch(
                "deinterlace_studio.dependencies._download", side_effect=copy_fixture
            ), patch("deinterlace_studio.dependencies._validate_ffmpeg", return_value="ffmpeg version 9.0"), patch(
                "deinterlace_studio.dependencies._atomic_manifest", side_effect=OSError("simulated activation failure")
            ):
                with self.assertRaisesRegex(OSError, "simulated activation failure"):
                    install_latest_dependencies(components={"ffmpeg"}, app_directory=app)
            self.assertEqual((root / "active.json").read_bytes(), original)
            self.assertEqual(list((root / "versions").iterdir()), [])


if __name__ == "__main__":
    unittest.main()
