from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from deinterlace_studio.batch import (
    BatchCompatibilityError,
    BatchQueue,
    resolve_batch_plan,
)
from deinterlace_studio.batch_runner import BatchRunOptions, BatchRunner
from deinterlace_studio.health import assess_packet_timeline
from deinterlace_studio.models import (
    SOURCE_REPAIR_REQUIRED_FAILURE,
    JobSettings,
    ProcessingResult,
    StreamInfo,
)
from deinterlace_tests.test_core import capabilities, media, report


def clear_health(source: Path, source_media):
    stat = source.stat()
    points = [index / 25 for index in range(1500)]
    result = assess_packet_timeline(
        source_media,
        source_size=stat.st_size,
        source_mtime_ns=stat.st_mtime_ns,
        packet_count=len(points),
        pts_values=points,
        elapsed_seconds=0.1,
        warning_samples=(),
        demux_warning_count=0,
        structural_warning_count=0,
        ffprobe_returncode=0,
        scan_error=None,
    )
    if result.status != "clear":
        return replace(result, status="clear", reason="No obvious damage found by the test packet scan.")
    return result


class BatchQueueTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def make_file(self, name: str) -> Path:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"video")
        return path

    def test_add_rejects_duplicates_unsupported_and_capacity_without_reordering(self) -> None:
        first = self.make_file("一.mkv")
        second = self.make_file("two.ts")
        third = self.make_file("three.mov")
        unsupported = self.make_file("notes.txt")
        queue = BatchQueue(maximum=2)
        result = queue.add_paths((first, unsupported, second, first, third))
        self.assertEqual([item.source_path for item in queue.records], [first.resolve(), second.resolve()])
        self.assertEqual(result.duplicates, (first,))
        self.assertEqual(result.unsupported, (unsupported,))
        self.assertEqual(result.capacity_rejected, (third,))

    def test_folder_add_move_reorder_and_remove_are_stable(self) -> None:
        first = self.make_file("folder/a.mkv")
        second = self.make_file("folder/b.mkv")
        third = self.make_file("folder/nested/c.mkv")
        queue = BatchQueue()
        queue.add_paths((first.parent,), include_subfolders=True)
        self.assertEqual([record.source_path for record in queue.records], [first.resolve(), second.resolve(), third.resolve()])
        selected = [queue.records[1].identifier, queue.records[2].identifier]
        queue.move(selected, -1)
        self.assertEqual([record.source_path.name for record in queue.records], ["b.mkv", "c.mkv", "a.mkv"])
        queue.move(selected, 1)
        self.assertEqual([record.source_path.name for record in queue.records], ["a.mkv", "b.mkv", "c.mkv"])
        queue.reorder(tuple(reversed([record.identifier for record in queue.records])))
        self.assertEqual([record.source_path.name for record in queue.records], ["c.mkv", "b.mkv", "a.mkv"])
        removed = queue.remove((queue.records[1].identifier,))
        self.assertEqual(removed[0].source_path.name, "b.mkv")


class BatchResolutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "節目 source.mkv"
        self.source.write_bytes(b"source")
        self.media = media(self.source)
        self.health = clear_health(self.source, self.media)

    def requested(self, **changes) -> JobSettings:
        values = dict(
            input_path=self.source,
            output_path=self.root / "unused.mkv",
            backend="auto",
            field_order="auto",
            output_cadence="frame_rate",
            family="ffv1",
            bit_depth=16,
            ffv1_chroma_mode="native",
            hardware_decode="auto",
            denoise_enabled=True,
            denoiser="vs_bm3d",
            denoise_strength=4,
            denoise_temporal_radius=3,
        )
        values.update(changes)
        return JobSettings(**values)

    def test_requested_defaults_build_max_quality_qtgmc_without_decode_block(self) -> None:
        resolution = resolve_batch_plan(
            self.requested(),
            self.source,
            self.media,
            report("tff"),
            self.health,
            capabilities(),
        )
        self.assertTrue(resolution.plan.valid, resolution.plan.errors)
        self.assertEqual(resolution.plan.selected_backend, "vapoursynth_qtgmc")
        self.assertEqual(resolution.settings.output_cadence, "frame_rate")
        self.assertEqual(resolution.settings.hardware_decode, "auto")
        self.assertEqual(resolution.plan.selected_denoiser, "vs_bm3d")
        self.assertEqual(resolution.settings.denoise_temporal_radius, 3)
        self.assertFalse(resolution.requires_repair)

    def test_qtgmc_missing_falls_back_to_cpu_bwdif_and_ffmpeg_temporal_denoise(self) -> None:
        resolution = resolve_batch_plan(
            self.requested(backend="vapoursynth_qtgmc"),
            self.source,
            self.media,
            report("tff"),
            self.health,
            capabilities(qtgmc=False),
        )
        self.assertEqual(resolution.plan.selected_backend, "ffmpeg_bwdif")
        self.assertEqual(resolution.settings.denoiser, "ffmpeg_fftdnoiz")
        self.assertEqual(resolution.settings.denoise_temporal_radius, 1)
        self.assertTrue(any("QTGMC was unavailable" in note for note in resolution.fallback_notes))
        self.assertTrue(any("denoiser" in note for note in resolution.fallback_notes))

    def test_dnxhr_12_falls_back_to_proven_10_bit_and_preserves_family(self) -> None:
        resolution = resolve_batch_plan(
            self.requested(family="dnxhr", bit_depth=12, denoise_enabled=False),
            self.source,
            self.media,
            report("tff"),
            self.health,
            capabilities(dnx12=False),
        )
        self.assertEqual(resolution.settings.family, "dnxhr")
        self.assertEqual(resolution.settings.bit_depth, 10)
        self.assertEqual(resolution.settings.output_path.suffix, ".mov")
        self.assertTrue(any("12-bit" in note and "10-bit" in note for note in resolution.fallback_notes))

    def test_mov_profile_uses_mkv_when_selected_tracks_are_not_mov_compatible(self) -> None:
        with_attachment = replace(
            self.media,
            streams=self.media.streams
            + (StreamInfo(index=2, codec_type="attachment", codec_name="ttf"),),
        )
        resolution = resolve_batch_plan(
            self.requested(family="prores", bit_depth=10, denoise_enabled=False),
            self.source,
            with_attachment,
            report("tff"),
            self.health,
            capabilities(),
        )
        self.assertEqual(resolution.settings.family, "prores")
        self.assertEqual(resolution.settings.output_path.suffix, ".mkv")
        self.assertTrue(any("attachments" in note for note in resolution.fallback_notes))

    def test_progressive_row_bypasses_forced_deinterlacer_and_field_rate(self) -> None:
        progressive_media = media(self.source, progressive=True)
        health = clear_health(self.source, progressive_media)
        resolution = resolve_batch_plan(
            self.requested(backend="vapoursynth_qtgmc", output_cadence="field_rate"),
            self.source,
            progressive_media,
            report("progressive"),
            health,
            capabilities(),
        )
        self.assertEqual(resolution.plan.selected_backend, "progressive")
        self.assertEqual(resolution.settings.output_cadence, "frame_rate")
        self.assertTrue(any("Measured progressive" in note for note in resolution.fallback_notes))

    def test_mixed_evidence_is_held_for_review_instead_of_guessed(self) -> None:
        with self.assertRaises(BatchCompatibilityError) as raised:
            resolve_batch_plan(
                self.requested(),
                self.source,
                self.media,
                report("mixed_or_ambiguous"),
                self.health,
                capabilities(),
            )
        self.assertTrue(raised.exception.needs_review)

    def test_damaged_qtgmc_row_is_marked_for_separate_repair(self) -> None:
        damaged = replace(self.health, status="repair_required", reason="Measured material timeline gap.")
        resolution = resolve_batch_plan(
            self.requested(),
            self.source,
            self.media,
            report("tff"),
            damaged,
            capabilities(),
        )
        self.assertTrue(resolution.requires_repair)
        self.assertTrue(resolution.plan.valid)
        self.assertTrue(any("repair copy" in note for note in resolution.fallback_notes))


class BatchRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.sources = (self.root / "one.mkv", self.root / "two.mkv")
        for source in self.sources:
            source.write_bytes(b"source")
        self.queue = BatchQueue()
        self.queue.add_paths(self.sources)
        self.requested = JobSettings(
            self.sources[0],
            self.root / "unused.mkv",
            backend="auto",
            output_cadence="frame_rate",
            hardware_decode="auto",
            denoise_enabled=False,
        )
        self.capabilities = capabilities()

    def patch_analysis(self, order: list[str]):
        def fake_probe(_ffprobe, source, sample_frames=64):
            del sample_frames
            order.append("probe:" + Path(source).name)
            return media(Path(source))

        def fake_health(_ffprobe, source_media, **_kwargs):
            order.append("health:" + source_media.path.name)
            return clear_health(source_media.path, source_media)

        def fake_idet(_ffmpeg, source_media, **_kwargs):
            order.append("idet:" + source_media.path.name)
            return report("tff")

        return (
            patch("deinterlace_studio.batch_runner.probe_media", side_effect=fake_probe),
            patch("deinterlace_studio.batch_runner.scan_source_health", side_effect=fake_health),
            patch("deinterlace_studio.batch_runner.scan_idet", side_effect=fake_idet),
        )

    def test_all_rows_preflight_before_first_sequential_encode(self) -> None:
        order: list[str] = []

        class FakeProcessor:
            def cancel(self):
                return None

            def run(self, plan, *_args, **_kwargs):
                order.append("encode:" + plan.settings.input_path.name)
                return ProcessingResult(True, False, "done", plan.output_path, None, None, None, "A" * 64, None)

        patches = self.patch_analysis(order)
        with patches[0], patches[1], patches[2], patch(
            "deinterlace_studio.batch_runner.JobProcessor", FakeProcessor
        ):
            summary = BatchRunner().run(
                self.queue,
                self.requested,
                self.capabilities,
                BatchRunOptions(output_directory=self.root),
            )
        self.assertEqual(summary.completed, 2)
        first_encode = next(index for index, value in enumerate(order) if value.startswith("encode:"))
        self.assertEqual(sum(value.startswith("idet:") for value in order[:first_encode]), 2)
        self.assertEqual([record.state for record in self.queue.records], ["Completed", "Completed"])

    def test_row_failure_continues_to_later_compatible_rows_by_default(self) -> None:
        order: list[str] = []
        calls = 0

        class FakeProcessor:
            def cancel(self):
                return None

            def run(self, plan, *_args, **_kwargs):
                nonlocal calls
                calls += 1
                order.append("encode:" + plan.settings.input_path.name)
                if calls == 1:
                    return ProcessingResult(False, False, "test failure", None, None, None, None, None, None)
                return ProcessingResult(True, False, "done", plan.output_path, None, None, None, "B" * 64, None)

        patches = self.patch_analysis(order)
        with patches[0], patches[1], patches[2], patch(
            "deinterlace_studio.batch_runner.JobProcessor", FakeProcessor
        ):
            summary = BatchRunner().run(
                self.queue,
                self.requested,
                self.capabilities,
                BatchRunOptions(output_directory=self.root, continue_after_error=True),
            )
        self.assertEqual(summary.failed, 1)
        self.assertEqual(summary.completed, 1)
        self.assertEqual([record.state for record in self.queue.records], ["Failed", "Completed"])

    def test_cancel_marks_active_and_unstarted_rows_without_deleting_sources(self) -> None:
        order: list[str] = []
        runner = BatchRunner()

        class FakeProcessor:
            def cancel(self):
                return None

            def run(self, _plan, *_args, **_kwargs):
                runner.cancel()
                return ProcessingResult(False, True, "canceled", None, None, None, None, None, None)

        patches = self.patch_analysis(order)
        with patches[0], patches[1], patches[2], patch(
            "deinterlace_studio.batch_runner.JobProcessor", FakeProcessor
        ):
            summary = runner.run(
                self.queue,
                self.requested,
                self.capabilities,
                BatchRunOptions(output_directory=self.root),
            )
        self.assertEqual(summary.canceled, 2)
        self.assertTrue(all(source.is_file() for source in self.sources))

    def test_unexpected_preflight_exception_is_isolated_to_its_row(self) -> None:
        def fake_probe(_ffprobe, source, sample_frames=64):
            del sample_frames
            if Path(source).name == "one.mkv":
                raise RuntimeError("synthetic corrupt container")
            return media(Path(source))

        class FakeProcessor:
            def cancel(self):
                return None

            def run(self, plan, *_args, **_kwargs):
                return ProcessingResult(True, False, "done", plan.output_path, None, None, None, "C" * 64, None)

        with patch("deinterlace_studio.batch_runner.probe_media", side_effect=fake_probe), patch(
            "deinterlace_studio.batch_runner.scan_source_health",
            side_effect=lambda _ffprobe, source_media, **_kwargs: clear_health(source_media.path, source_media),
        ), patch(
            "deinterlace_studio.batch_runner.scan_idet",
            side_effect=lambda _ffmpeg, _source_media, **_kwargs: report("tff"),
        ), patch("deinterlace_studio.batch_runner.JobProcessor", FakeProcessor):
            summary = BatchRunner().run(
                self.queue,
                self.requested,
                self.capabilities,
                BatchRunOptions(output_directory=self.root, continue_after_error=True),
            )
        self.assertEqual(summary.failed, 1)
        self.assertEqual(summary.completed, 1)
        self.assertEqual(self.queue.records[0].state, "Preflight failed")
        self.assertIn("synthetic corrupt container", self.queue.records[0].error or "")
        self.assertEqual(self.queue.records[1].state, "Completed")

    def test_bwdif_failure_never_enters_qtgmc_repair_retry(self) -> None:
        requested = replace(
            self.requested,
            backend="ffmpeg_bwdif",
            denoise_enabled=False,
        )

        class FakeProcessor:
            def cancel(self):
                return None

            def run(self, _plan, *_args, **_kwargs):
                return ProcessingResult(
                    success=False,
                    canceled=False,
                    message="synthetic BWDIF failure",
                    output_path=None,
                    log_path=None,
                    report_path=None,
                    script_path=None,
                    output_sha256=None,
                    validation=None,
                    failure_code=SOURCE_REPAIR_REQUIRED_FAILURE,
                )

        patches = self.patch_analysis([])
        with patches[0], patches[1], patches[2], patch(
            "deinterlace_studio.batch_runner.JobProcessor", FakeProcessor
        ), patch.object(BatchRunner, "_repair_for_qtgmc") as repair:
            summary = BatchRunner().run(
                self.queue,
                requested,
                self.capabilities,
                BatchRunOptions(output_directory=self.root, auto_repair=True),
            )
        repair.assert_not_called()
        self.assertEqual(summary.failed, 2)
        self.assertTrue(all(record.state == "Failed" for record in self.queue.records))


if __name__ == "__main__":
    unittest.main()
