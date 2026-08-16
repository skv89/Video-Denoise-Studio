from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from video_denoise_studio.batch import BatchQueue


class BatchQueueTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def video(self, name: str) -> Path:
        path = self.root / name
        path.write_bytes(b"video")
        return path

    def test_add_duplicate_reorder_remove_and_capacity(self) -> None:
        first = self.video("first.mkv")
        second = self.video("second.mov")
        unsupported = self.video("notes.txt")
        queue = BatchQueue(maximum=2)
        result = queue.add_paths((first, second, unsupported, first))
        self.assertEqual(len(result.added), 2)
        self.assertEqual(result.unsupported, (unsupported,))
        self.assertEqual(result.duplicates, (first,))
        identifiers = [record.identifier for record in queue.records]
        queue.move((identifiers[1],), -1)
        self.assertEqual(queue.records[0].source_path, second.resolve())
        removed = queue.remove((identifiers[0],))
        self.assertEqual(removed[0].source_path, first.resolve())

    def test_folder_scan_obeys_subfolder_switch(self) -> None:
        self.video("top.mkv")
        nested = self.root / "nested"
        nested.mkdir()
        (nested / "inside.mp4").write_bytes(b"video")
        queue = BatchQueue()
        self.assertEqual(len(queue.add_paths((self.root,), include_subfolders=False).added), 1)
        queue.clear()
        self.assertEqual(len(queue.add_paths((self.root,), include_subfolders=True).added), 2)


if __name__ == "__main__":
    unittest.main()

