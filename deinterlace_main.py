from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path


def _windows_extended_tcl_path(raw_path: str) -> str:
    """Render a local Windows Tcl library path without the user-profile path fault."""

    normalized = raw_path.replace("\\", "/")
    if normalized.startswith("//?/"):
        return normalized
    if len(normalized) >= 3 and normalized[1:3] == ":/":
        return "//?/" + normalized
    return raw_path


def _stabilize_windows_tcl_library_paths() -> None:
    """Use extended Tcl/Tk library paths for Windows user-profile installations."""

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

from deinterlace_studio import __version__
from deinterlace_studio.capabilities import inspect_capabilities
from deinterlace_studio.gui import DeinterlaceStudioApp


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quality-first FFmpeg and VapourSynth deinterlacing GUI")
    parser.add_argument("--self-test", action="store_true", help="Run a bounded packaged-startup smoke test")
    parser.add_argument("--self-test-report", type=Path, help="Write self-test JSON to this path")
    parser.add_argument("--ffmpeg", type=Path, help="Use this FFmpeg binary for a diagnostic/self-test run")
    parser.add_argument("--ffprobe", type=Path, help="Use this FFprobe binary for a diagnostic/self-test run")
    parser.add_argument("--vspipe", type=Path, help="Use this VSPipe binary for a diagnostic/self-test run")
    parser.add_argument("--version", action="version", version=f"Deinterlace Studio {__version__}")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.self_test:
        root = None
        try:
            root = Tk()
            root.withdraw()
            capabilities = inspect_capabilities(args.ffmpeg, args.ffprobe, args.vspipe)
            app = DeinterlaceStudioApp(root, initial_capabilities=capabilities)
            result = app.self_test()
        except Exception as exc:
            result = {
                "application_version": __version__,
                "self_test_error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        finally:
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
        return 0 if result.get("ffmpeg_found") and result.get("ffprobe_found") else 1

    root = Tk()
    initial_capabilities = None
    if args.ffmpeg or args.ffprobe or args.vspipe:
        initial_capabilities = inspect_capabilities(args.ffmpeg, args.ffprobe, args.vspipe)
    DeinterlaceStudioApp(root, initial_capabilities=initial_capabilities)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
