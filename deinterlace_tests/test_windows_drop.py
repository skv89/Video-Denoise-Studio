from __future__ import annotations

import unittest
from pathlib import Path
from tkinter import Tcl, TclError, Tk, ttk
from types import SimpleNamespace

from deinterlace_studio.windows_drop import (
    COPY,
    MAX_DROPPED_PATHS,
    FileDropUnavailable,
    WindowsFileDropTarget,
    parse_tcl_file_list,
)


def _raw_tcl_list(interpreter, values: tuple[str, ...]) -> str:
    variable = "::deinterlace_drop_unit_data"
    interpreter.call("set", variable, values)
    try:
        return str(interpreter.eval(f"set {variable}"))
    finally:
        interpreter.call("unset", variable)


class TclDropParserTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tcl = Tcl()

    def test_unicode_spaces_braces_and_multiple_paths_round_trip(self) -> None:
        expected = (
            r"C:\Video Folder\拖放 測試.mkv",
            r"D:\Series\episode {final}.ts",
        )
        raw = _raw_tcl_list(self.tcl, expected)
        self.assertEqual(parse_tcl_file_list(self.tcl.splitlist, raw), tuple(map(Path, expected)))

    def test_malformed_tcl_list_fails_closed(self) -> None:
        with self.assertRaisesRegex(FileDropUnavailable, "invalid Tcl file list"):
            parse_tcl_file_list(self.tcl.splitlist, "{unterminated")

    def test_path_count_and_null_character_are_bounded(self) -> None:
        values = tuple(f"C:\\Video\\{index}.mkv" for index in range(MAX_DROPPED_PATHS + 1))
        with self.assertRaisesRegex(FileDropUnavailable, "at most"):
            parse_tcl_file_list(self.tcl.splitlist, _raw_tcl_list(self.tcl, values))
        with self.assertRaisesRegex(FileDropUnavailable, "null"):
            parse_tcl_file_list(lambda _raw: ("C:\\bad\0name.mkv",), "ignored")


class TkDndLifecycleTests(unittest.TestCase):
    def test_provider_load_drop_delivery_and_clean_close(self) -> None:
        try:
            root = Tk()
        except TclError as exc:
            self.skipTest(f"Tk unavailable in this test runtime: {exc}")
        received: list[tuple[Path, ...]] = []
        errors: list[str] = []
        target = WindowsFileDropTarget(root, received.append, error_callback=errors.append)
        root.withdraw()
        try:
            surface = ttk.Frame(root)
            surface.pack(fill="both", expand=True)
            target.install()
            dropped = Path(r"C:\Video Folder\拖放 測試.mkv")
            raw = _raw_tcl_list(root.tk, (str(dropped),))
            self.assertEqual(target._on_drop(SimpleNamespace(data=raw)), COPY)
            root.update()
            self.assertEqual(received, [(dropped,)])
            self.assertFalse(errors)
            self.assertTrue(target.active)
            self.assertGreaterEqual(len(target.registrations), 1)
            self.assertTrue(target.provider_version)
            self.assertEqual(target.package_version, "0.6.2")
            self.assertEqual(target.registration_errors, ())
            self.assertEqual(target._on_drop(SimpleNamespace(data="{unterminated")), "refuse_drop")
            root.update()
            self.assertEqual(len(errors), 1)
            self.assertIn("invalid Tcl file list", errors[0])
            target.close()
            self.assertFalse(target.active)
            self.assertEqual(target.registrations, [])
        finally:
            target.close()
            root.destroy()


if __name__ == "__main__":
    unittest.main()
