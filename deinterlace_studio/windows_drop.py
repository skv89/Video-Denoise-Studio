from __future__ import annotations

import os
import platform
import struct
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Callable, Protocol

try:
    from tkinterdnd2 import COPY, DND_FILES, REFUSE_DROP, TkinterDnD
except ImportError as exc:  # Browse must remain usable in an incomplete source environment.
    COPY = "copy"
    DND_FILES = "DND_Files"
    REFUSE_DROP = "refuse_drop"
    TkinterDnD = None
    _TKDND_IMPORT_ERROR: Exception | None = exc
else:
    _TKDND_IMPORT_ERROR = None


MAX_DROPPED_PATHS = 99


class FileDropUnavailable(RuntimeError):
    """Raised when the maintained TkDND drop provider cannot be loaded safely."""


class _DropEvent(Protocol):
    data: str


def parse_tcl_file_list(splitlist: Callable[[str], tuple[str, ...]], raw_data: str) -> tuple[Path, ...]:
    """Decode TkDND's raw Tcl file list without splitting paths on spaces."""

    if not isinstance(raw_data, str):
        raise FileDropUnavailable("The native drop provider returned non-text file data.")
    try:
        values = tuple(str(value) for value in splitlist(raw_data))
    except Exception as exc:
        raise FileDropUnavailable(f"The native drop provider returned an invalid Tcl file list: {exc}") from exc
    if len(values) > MAX_DROPPED_PATHS:
        raise FileDropUnavailable(
            f"The drop contains {len(values)} paths; at most {MAX_DROPPED_PATHS} can be inspected safely."
        )
    paths: list[Path] = []
    for index, value in enumerate(values, start=1):
        if not value:
            raise FileDropUnavailable(f"Dropped path {index} is empty.")
        if "\0" in value:
            raise FileDropUnavailable(f"Dropped path {index} contains an embedded null character.")
        paths.append(Path(value))
    return tuple(paths)


def _windows_tkdnd_provider_path(root) -> str:
    """Return a Tcl-safe extended path to TkinterDnD2's native provider.

    Some Windows Tcl builds incorrectly collapse a normal absolute path below
    the current user's profile before ``package require`` inspects it.  The
    Win32 extended-path spelling keeps the exact package directory intact and
    remains accepted by Tcl's package loader.
    """

    if os.name != "nt" or TkinterDnD is None:
        raise FileDropUnavailable("The Windows TkDND provider fallback is not applicable on this platform.")

    machine = str(os.environ.get("PROCESSOR_ARCHITECTURE") or platform.machine()).upper()
    if machine in {"AMD64", "X86_64"}:
        platform_directory = "win-x64" if struct.calcsize("P") * 8 == 64 else "win-x86"
    elif machine in {"X86", "I386", "I686"}:
        platform_directory = "win-x86"
    elif machine in {"ARM64", "AARCH64"}:
        platform_directory = "win-arm64"
    else:
        raise FileDropUnavailable(f"TkDND does not provide a native package for {machine or 'this process'}.")

    package_root = Path(TkinterDnD.__file__).resolve().parent / "tkdnd"
    try:
        tcl_major = int(str(root.tk.call("info", "tclversion")).split(".", 1)[0])
    except (TypeError, ValueError) as exc:
        raise FileDropUnavailable(f"Could not determine the Tcl runtime version: {exc}") from exc
    tcl9_candidate = package_root / f"{platform_directory}-tcl9"
    provider_directory = tcl9_candidate if tcl_major >= 9 and tcl9_candidate.is_dir() else package_root / platform_directory
    if not provider_directory.is_dir():
        raise FileDropUnavailable(f"TkDND native provider directory is missing: {provider_directory}")

    provider_path = provider_directory.as_posix()
    if provider_path.startswith("//"):
        return provider_path
    return "//?/" + provider_path


def _require_tkdnd(root) -> str:
    """Load TkDND, retrying with a Tcl-safe Windows extended path."""

    if TkinterDnD is None:
        detail = f": {_TKDND_IMPORT_ERROR}" if _TKDND_IMPORT_ERROR else ""
        raise FileDropUnavailable(f"TkinterDnD2 is not installed{detail}")
    try:
        return str(TkinterDnD.require(root))
    except Exception as first_error:
        if os.name != "nt":
            raise FileDropUnavailable(f"TkDND could not load its native provider: {first_error}") from first_error
        try:
            provider_path = _windows_tkdnd_provider_path(root)
            root.tk.call("lappend", "auto_path", provider_path)
            return str(root.tk.call("package", "require", "tkdnd"))
        except Exception as fallback_error:
            raise FileDropUnavailable(
                "TkDND could not load its native provider through either the package loader or the "
                f"Windows extended-path fallback: {fallback_error}"
            ) from first_error


class WindowsFileDropTarget:
    """Receive Explorer file drops through TkDND's native Windows OLE target."""

    def __init__(
        self,
        root,
        callback: Callable[[tuple[Path, ...]], None],
        *,
        error_callback: Callable[[str], None] | None = None,
    ) -> None:
        self.root = root
        self.callback = callback
        self.error_callback = error_callback
        self.active = False
        self.last_error: str | None = None
        self.registrations: list[tuple[object, str]] = []
        self.registration_errors: tuple[str, ...] = ()
        self.provider_version: str | None = None
        try:
            self.package_version = version("tkinterdnd2")
        except PackageNotFoundError:
            self.package_version = None

    def install(self) -> None:
        if self.active:
            return
        try:
            self.provider_version = _require_tkdnd(self.root)
        except Exception as exc:
            self.provider_version = None
            if isinstance(exc, FileDropUnavailable):
                raise
            raise FileDropUnavailable(f"TkDND could not load its native provider: {exc}") from exc

        failures: list[str] = []
        for widget in self._candidate_widgets():
            try:
                widget.drop_target_register(DND_FILES)
                binding_id = widget.dnd_bind("<<Drop>>", self._on_drop, add="+")
                if not binding_id:
                    raise FileDropUnavailable("no binding identifier was returned")
            except Exception as exc:
                try:
                    widget.drop_target_unregister()
                except Exception:
                    pass
                failures.append(f"{widget}: {exc}")
            else:
                self.registrations.append((widget, str(binding_id)))
        self.registration_errors = tuple(failures)
        if not self.registrations:
            self.provider_version = None
            detail = failures[0] if failures else "the root contains no registrable widgets"
            raise FileDropUnavailable(f"TkDND could not register any application drop surface: {detail}")
        self.active = True

    def _candidate_widgets(self) -> tuple[object, ...]:
        candidates: list[object] = []
        pending = list(self.root.winfo_children())
        while pending:
            widget = pending.pop()
            try:
                pending.extend(widget.winfo_children())
            except Exception:
                pass
            if hasattr(widget, "drop_target_register") and hasattr(widget, "dnd_bind"):
                candidates.append(widget)
        return tuple(candidates)

    def _on_drop(self, event: _DropEvent) -> str:
        try:
            paths = parse_tcl_file_list(self.root.tk.splitlist, event.data)
        except Exception as exc:
            self._schedule_error(f"Could not read dropped files: {exc}")
            return REFUSE_DROP
        self.root.after_idle(self._deliver, paths)
        return COPY

    def _deliver(self, paths: tuple[Path, ...]) -> None:
        if not self.active:
            return
        try:
            self.callback(paths)
        except Exception as exc:
            self._schedule_error(f"The application could not accept the dropped file: {exc}")

    def _schedule_error(self, message: str) -> None:
        self.last_error = message
        if self.error_callback:
            self.root.after_idle(self.error_callback, message)

    def close(self) -> None:
        if not self.active:
            return
        self.active = False
        registrations = self.registrations
        self.registrations = []
        for widget, binding_id in reversed(registrations):
            try:
                widget.unbind("<<Drop>>", binding_id)
            except Exception:
                pass
            try:
                widget.drop_target_unregister()
            except Exception:
                pass
