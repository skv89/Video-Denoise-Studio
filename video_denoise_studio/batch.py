from __future__ import annotations

import os
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

from deinterlace_studio.models import CapabilityReport

from .models import (
    BatchEventCallback,
    BatchRecord,
    BatchRunOptions,
    BatchRunSummary,
    DenoiseSettings,
)
from .planner import build_plan, unique_output_path
from .probe import ProbeCancelled, probe_media_cancelable
from .processor import DenoiseProcessor
from .output_policy import resolve_container, select_output_profile


MAX_BATCH_FILES = 99
SUPPORTED_VIDEO_EXTENSIONS = frozenset(
    {
        ".3gp",
        ".asf",
        ".avi",
        ".divx",
        ".flv",
        ".m2t",
        ".m2ts",
        ".m4v",
        ".mkv",
        ".mov",
        ".mp4",
        ".mpeg",
        ".mpg",
        ".mts",
        ".mxf",
        ".ogm",
        ".rm",
        ".rmvb",
        ".ts",
        ".vob",
        ".webm",
        ".wmv",
    }
)


def normalized_path_key(path: Path) -> str:
    return os.path.normcase(os.path.abspath(path))


@dataclass(frozen=True)
class BatchAddResult:
    added: tuple[BatchRecord, ...]
    duplicates: tuple[Path, ...] = ()
    unsupported: tuple[Path, ...] = ()
    missing: tuple[Path, ...] = ()
    capacity_rejected: tuple[Path, ...] = ()


class BatchQueue:
    def __init__(self, *, maximum: int = MAX_BATCH_FILES) -> None:
        self.maximum = maximum
        self.records: list[BatchRecord] = []

    def __len__(self) -> int:
        return len(self.records)

    def record(self, identifier: str) -> BatchRecord | None:
        return next((record for record in self.records if record.identifier == identifier), None)

    def add_paths(self, paths: Iterable[Path], *, include_subfolders: bool = False) -> BatchAddResult:
        existing = {normalized_path_key(record.source_path) for record in self.records}
        candidates: list[Path] = []
        missing: list[Path] = []
        unsupported: list[Path] = []
        duplicates: list[Path] = []
        capacity: list[Path] = []
        added: list[BatchRecord] = []
        for supplied in paths:
            path = Path(supplied)
            if path.is_dir():
                pattern = "**/*" if include_subfolders else "*"
                candidates.extend(sorted((item for item in path.glob(pattern) if item.is_file()), key=lambda item: str(item).casefold()))
            else:
                candidates.append(path)
        for path in candidates:
            if not path.is_file():
                missing.append(path)
                continue
            if path.suffix.casefold() not in SUPPORTED_VIDEO_EXTENSIONS:
                unsupported.append(path)
                continue
            key = normalized_path_key(path)
            if key in existing:
                duplicates.append(path)
                continue
            if len(self.records) >= self.maximum:
                capacity.append(path)
                continue
            record = BatchRecord(path.resolve())
            self.records.append(record)
            existing.add(key)
            added.append(record)
        return BatchAddResult(tuple(added), tuple(duplicates), tuple(unsupported), tuple(missing), tuple(capacity))

    def remove(self, identifiers: Iterable[str]) -> tuple[BatchRecord, ...]:
        selected = set(identifiers)
        removed = tuple(record for record in self.records if record.identifier in selected)
        self.records[:] = [record for record in self.records if record.identifier not in selected]
        return removed

    def clear(self) -> tuple[BatchRecord, ...]:
        removed = tuple(self.records)
        self.records.clear()
        return removed

    def move(self, identifiers: Iterable[str], direction: int) -> None:
        if direction not in {-1, 1}:
            raise ValueError("Batch rows can move only one position up or down.")
        selected = set(identifiers)
        if direction < 0:
            for index in range(1, len(self.records)):
                if self.records[index].identifier in selected and self.records[index - 1].identifier not in selected:
                    self.records[index - 1], self.records[index] = self.records[index], self.records[index - 1]
        else:
            for index in range(len(self.records) - 2, -1, -1):
                if self.records[index].identifier in selected and self.records[index + 1].identifier not in selected:
                    self.records[index], self.records[index + 1] = self.records[index + 1], self.records[index]


def preferred_output_path(source: Path, settings: DenoiseSettings, output_directory: Path | None) -> Path:
    container = resolve_container(settings, None)
    return (output_directory or source.parent) / f"{source.stem}.denoised{container.extension}"


class BatchRunner:
    def __init__(self, event_callback: BatchEventCallback | None = None) -> None:
        self.event_callback = event_callback
        self.cancel_event = threading.Event()
        self.active_processor: DenoiseProcessor | None = None

    def cancel(self) -> None:
        self.cancel_event.set()
        if self.active_processor:
            self.active_processor.cancel()

    def _check_canceled(self) -> None:
        if self.cancel_event.is_set():
            raise ProbeCancelled("Batch canceled.")

    def _emit(self, kind: str, record: BatchRecord | None = None, payload: object | None = None) -> None:
        if self.event_callback:
            self.event_callback(kind, record, payload)

    def _set_row(self, record: BatchRecord, **values: object) -> None:
        for key, value in values.items():
            setattr(record, key, value)
        self._emit("row", record)

    @staticmethod
    def _effective(plan, notes: tuple[str, ...]) -> str:
        profile = plan.profile.label if plan.profile else "unresolved output"
        backend = plan.selected_denoise_backend or "unresolved denoiser"
        container = plan.container.upper() if plan.container else "unresolved container"
        suffix = f" · {len(notes)} fallback(s)" if notes else ""
        return f"{backend} · {profile} · {container}{suffix}"

    def _resolve_row(
        self,
        record: BatchRecord,
        requested: DenoiseSettings,
        capabilities: CapabilityReport,
        output_directory: Path | None,
        reserved: tuple[Path, ...],
    ):
        self._check_canceled()
        assert capabilities.ffprobe_path
        self._set_row(record, state="Preflighting", progress_text="Reading media metadata", percent=0.0)
        media = probe_media_cancelable(capabilities.ffprobe_path, record.source_path, self.cancel_event, sample_frames=64)
        record.media = media
        attempts: list[tuple[DenoiseSettings, str | None]] = []
        try:
            profile, _container = select_output_profile(requested, media)
            output = unique_output_path(
                (output_directory or record.source_path.parent) / f"{record.source_path.stem}.denoised{profile.default_extension}",
                reserved,
            )
            attempts.append((replace(requested, input_path=record.source_path, output_path=output), None))
        except ValueError:
            output = unique_output_path(
                (output_directory or record.source_path.parent) / f"{record.source_path.stem}.denoised.mkv",
                reserved,
            )
            attempts.append((replace(requested, input_path=record.source_path, output_path=output), None))

        if requested.hardware_encode:
            software = replace(attempts[0][0], hardware_encode=False)
            attempts.append((software, "Hardware encoder was unavailable; used the matching software profile."))
        if requested.family != "ffv1":
            fallback_output = unique_output_path(
                (output_directory or record.source_path.parent) / f"{record.source_path.stem}.denoised.mkv",
                reserved,
            )
            attempts.append(
                (
                    replace(
                        requested,
                        input_path=record.source_path,
                        output_path=fallback_output,
                        family="ffv1",
                        container="mkv",
                        bit_depth=16,
                        hardware_encode=False,
                        ffv1_chroma_mode="native",
                    ),
                    "Requested container/profile could not preserve this row; used FFV1 native-chroma MKV.",
                )
            )

        failures: list[str] = []
        for candidate, note in attempts:
            plan = build_plan(candidate, media, capabilities)
            if plan.valid:
                notes = (note,) if note else ()
                return candidate, plan, notes
            failures.append("; ".join(plan.errors))
        raise RuntimeError("No safe batch plan is available. " + " | ".join(dict.fromkeys(failures)))

    def run(
        self,
        batch_queue: BatchQueue,
        requested: DenoiseSettings,
        capabilities: CapabilityReport,
        options: BatchRunOptions,
    ) -> BatchRunSummary:
        self.cancel_event.clear()
        if options.output_directory is not None and not options.output_directory.is_dir():
            raise ValueError(f"Batch output directory does not exist: {options.output_directory}")
        for record in batch_queue.records:
            record.reset()
        reserved: list[Path] = []
        ready: list[BatchRecord] = []
        stop = False
        self._emit("phase", payload="Preflighting every row before long processing starts")
        try:
            for record in batch_queue.records:
                self._check_canceled()
                try:
                    settings, plan, notes = self._resolve_row(
                        record,
                        requested,
                        capabilities,
                        options.output_directory,
                        tuple(reserved),
                    )
                    record.plan = plan
                    record.output_path = settings.output_path
                    record.fallback_notes = notes
                    record.effective_text = self._effective(plan, notes)
                    reserved.append(settings.output_path)
                    ready.append(record)
                    self._set_row(record, state="Ready", progress_text="Preflight passed", percent=100.0)
                except ProbeCancelled:
                    raise
                except Exception as exc:
                    self._set_row(
                        record,
                        state="Preflight failed",
                        progress_text="Not queued for processing",
                        error=f"{type(exc).__name__}: {exc}",
                    )
                    if not options.continue_after_error:
                        stop = True
                        break
            if stop:
                for record in ready:
                    self._set_row(record, state="Skipped", progress_text="Stopped after preflight error")
                ready.clear()
                for record in batch_queue.records:
                    if record.state == "Queued":
                        self._set_row(record, state="Skipped", progress_text="Stopped after preflight error")

            self._emit("phase", payload="Preflight complete; processing compatible rows sequentially")
            for index, record in enumerate(ready):
                self._check_canceled()
                assert record.plan is not None
                self._set_row(record, state="Processing", progress_text="Starting", percent=0.0)
                processor = DenoiseProcessor()
                self.active_processor = processor

                def progress(values: dict[str, str], *, current=record) -> None:
                    percent: float | None = current.percent
                    frame = values.get("frame")
                    expected = current.plan.expected.frame_count if current.plan and current.plan.expected else None
                    if frame and expected:
                        try:
                            percent = min(100.0, 100.0 * int(frame) / expected)
                        except (TypeError, ValueError, ZeroDivisionError):
                            pass
                    phase = values.get("phase", "processing").replace("_", " ")
                    self._set_row(current, progress_text=phase, percent=percent)

                try:
                    result = processor.run(record.plan, progress_callback=progress)
                finally:
                    self.active_processor = None
                record.result = result
                if result.canceled or self.cancel_event.is_set():
                    raise ProbeCancelled("Batch canceled during processing.")
                if result.success:
                    self._set_row(record, state="Completed", progress_text="Validated", percent=100.0)
                else:
                    self._set_row(record, state="Failed", progress_text="Processing failed", error=result.message)
                    if not options.continue_after_error:
                        for remaining in ready[index + 1 :]:
                            self._set_row(remaining, state="Skipped", progress_text="Stopped after row failure")
                        break
        except ProbeCancelled:
            for record in batch_queue.records:
                if record.state in {"Queued", "Preflighting", "Ready", "Processing"}:
                    self._set_row(record, state="Canceled", progress_text="Canceled safely")
        finally:
            self.active_processor = None

        summary = BatchRunSummary(
            total=len(batch_queue.records),
            completed=sum(record.state == "Completed" for record in batch_queue.records),
            failed=sum(record.state in {"Failed", "Preflight failed"} for record in batch_queue.records),
            canceled=sum(record.state == "Canceled" for record in batch_queue.records),
            skipped=sum(record.state == "Skipped" for record in batch_queue.records),
        )
        self._emit("complete", payload=summary)
        return summary
