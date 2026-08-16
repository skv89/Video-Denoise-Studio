from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import traceback
from pathlib import Path


def _windows_extended_tcl_path(raw_path: str) -> str:
    normalized = raw_path.replace("\\", "/")
    if normalized.startswith("//?/"):
        return normalized
    if len(normalized) >= 3 and normalized[1:3] == ":/":
        return "//?/" + normalized
    return raw_path


def _stabilize_windows_tcl_library_paths() -> None:
    if os.name != "nt":
        return
    if getattr(sys, "frozen", False):
        runtime_root = Path(getattr(sys, "_MEIPASS", sys.prefix))
        defaults = {
            "TCL_LIBRARY": runtime_root / "_tcl_data",
            "TK_LIBRARY": runtime_root / "_tk_data",
        }
    else:
        runtime_root = Path(sys.prefix) / "tcl"
        defaults = {
            "TCL_LIBRARY": runtime_root / "tcl8.6",
            "TK_LIBRARY": runtime_root / "tk8.6",
        }
    for variable, default in defaults.items():
        value = os.environ.get(variable)
        if not value and default.is_dir():
            value = str(default)
        if value:
            os.environ[variable] = _windows_extended_tcl_path(value)


_stabilize_windows_tcl_library_paths()

from tkinter import Tk  # noqa: E402

from deinterlace_studio.capabilities import inspect_capabilities  # noqa: E402
from video_denoise_studio import __version__  # noqa: E402
from video_denoise_studio.gui import VideoDenoiseStudioApp  # noqa: E402
from video_denoise_studio.models import PreviewRequest  # noqa: E402
from video_denoise_studio.planner import source_field_order  # noqa: E402
from video_denoise_studio.preview import PreviewRenderer  # noqa: E402
from video_denoise_studio.probe import probe_media_cancelable  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Temporal video denoising with full-timeline frame comparison")
    parser.add_argument("--self-test", action="store_true", help="Run a bounded packaged-startup smoke test")
    parser.add_argument("--self-test-report", type=Path, help="Write self-test JSON to this path")
    parser.add_argument("--ffmpeg", type=Path, help="Use this FFmpeg binary for diagnostic/self-test discovery")
    parser.add_argument("--ffprobe", type=Path, help="Use this FFprobe binary for diagnostic/self-test discovery")
    parser.add_argument("--vspipe", type=Path, help="Use this VSPipe binary for diagnostic/self-test discovery")
    parser.add_argument(
        "--self-test-dft-preview-source",
        type=Path,
        action="append",
        default=[],
        help="During --self-test, render a real Strength-10/radius-3 DFTTest2 preview for this source",
    )
    parser.add_argument(
        "--self-test-dft-preview-target",
        type=int,
        default=20,
        help="Zero-based target for optional packaged DFTTest2 preview checks",
    )
    parser.add_argument("--version", action="version", version=f"Video Denoise Studio {__version__}")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.self_test:
        root = None
        app = None
        try:
            root = Tk()
            root.withdraw()
            capabilities = inspect_capabilities(args.ffmpeg, args.ffprobe, args.vspipe)
            app = VideoDenoiseStudioApp(root, initial_capabilities=capabilities)
            result = app.self_test()
            if args.self_test_dft_preview_source:
                preview_cases = []
                for raw_source in args.self_test_dft_preview_source:
                    source = raw_source.resolve(strict=True)
                    media = probe_media_cancelable(
                        capabilities.ffprobe_path,
                        source,
                        threading.Event(),
                        sample_frames=64,
                    )
                    renderer = PreviewRenderer()
                    frames = None
                    try:
                        frames = renderer.render(
                            PreviewRequest(
                                source=source,
                                media=media,
                                capabilities=capabilities,
                                denoiser="vs_dfttest",
                                strength=10,
                                temporal_radius=3,
                                target_frame=args.self_test_dft_preview_target,
                                width=media.video.width or 720,
                                height=media.video.height or 576,
                                include_processed=True,
                            )
                        )
                        script_path = frames.directory / "preview.vpy"
                        script_text = script_path.read_text(encoding="utf-8")
                        order = source_field_order(media)
                        expected_field_value = 2 if order == "tff" else 1
                        processed_hash = hashlib.sha256(frames.processed_frame.read_bytes()).hexdigest().upper()
                        passed = all(
                            (
                                order in {"tff", "bff"},
                                frames.selected_backend == "dfttest_nvrtc",
                                frames.strength == 10,
                                frames.temporal_radius == 3,
                                frames.window_frames == 7,
                                frames.leading_context == 3,
                                frames.trailing_context == 3,
                                script_text.count("DFTTest.Backend.NVRTC") == 2,
                                "SeparateFields(" in script_text,
                                "DoubleWeave(" in script_text,
                                f"SetFieldBased(processed, value={expected_field_value})" in script_text,
                                "tr=3, sigma=20.00" in script_text,
                            )
                        )
                        preview_cases.append(
                            {
                                "source": str(source),
                                "field_order": order,
                                "sampled_interlaced_frames": media.sampled_interlaced_frames,
                                "selected_backend": frames.selected_backend,
                                "strength": frames.strength,
                                "temporal_radius": frames.temporal_radius,
                                "window_frames": frames.window_frames,
                                "leading_context": frames.leading_context,
                                "trailing_context": frames.trailing_context,
                                "generated_nvrtc_calls": script_text.count("DFTTest.Backend.NVRTC"),
                                "has_field_parity_graph": "SeparateFields(" in script_text
                                and "DoubleWeave(" in script_text,
                                "processed_png_sha256": processed_hash,
                                "status": frames.status,
                                "passed": passed,
                            }
                        )
                    finally:
                        if frames is not None:
                            renderer.cleanup(frames)
                        renderer.close()
                result["dfttest_interlaced_preview_cases"] = preview_cases
        except Exception as exc:
            result = {
                "application_version": __version__,
                "self_test_error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        finally:
            if app is not None:
                try:
                    app.close(save_preferences=False)
                    root = None
                except Exception:
                    pass
            if root is not None:
                try:
                    root.destroy()
                except Exception:
                    pass
        rendered = json.dumps(result, ensure_ascii=False, indent=2)
        if args.self_test_report:
            args.self_test_report.parent.mkdir(parents=True, exist_ok=True)
            args.self_test_report.write_text(rendered, encoding="utf-8")
        elif sys.stdout:
            print(rendered)
        denoisers = result.get("denoise_capabilities") or {}
        preview_cases = result.get("dfttest_interlaced_preview_cases") or []
        previews_passed = not args.self_test_dft_preview_source or (
            len(preview_cases) == len(args.self_test_dft_preview_source)
            and all(case.get("passed") is True for case in preview_cases)
        )
        return (
            0
            if result.get("ffmpeg_found")
            and result.get("ffprobe_found")
            and any(denoisers.values())
            and previews_passed
            else 1
        )

    root = Tk()
    initial_capabilities = None
    if args.ffmpeg or args.ffprobe or args.vspipe:
        initial_capabilities = inspect_capabilities(args.ffmpeg, args.ffprobe, args.vspipe)
    VideoDenoiseStudioApp(root, initial_capabilities=initial_capabilities)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
