from __future__ import annotations

import threading
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

from .automation import (
    REPAIR_ARTIFACT_SUFFIXES,
    choose_available_artifact_path,
    storage_preflight,
    storage_summary,
)
from .batch import (
    BatchCompatibilityError,
    BatchQueue,
    BatchRecord,
    BatchResolution,
    resolve_batch_plan,
)
from .health import health_matches_source, scan_source_health, source_identity
from .idet import scan_idet
from .models import (
    SOURCE_REPAIR_REQUIRED_FAILURE,
    AutomaticRecoveryAudit,
    CapabilityReport,
    JobSettings,
    MediaProbe,
    ProcessingPlan,
)
from .planner import build_plan
from .probe import probe_media
from .processor import JobProcessor
from .repair import RepairRequest, RepairResult, SourceRepairer


BatchEventCallback = Callable[[str, BatchRecord | None, Any], None]


class BatchCancelled(RuntimeError):
    pass


@dataclass(frozen=True)
class BatchRunOptions:
    output_directory: Path | None = None
    auto_repair: bool = True
    continue_after_error: bool = True
    analysis_mode: str = "sampled"


@dataclass(frozen=True)
class BatchRunSummary:
    total: int
    completed: int
    failed: int
    needs_review: int
    canceled: int
    skipped: int


@dataclass(frozen=True)
class _RepairOutcome:
    source_path: Path
    media: MediaProbe
    analysis: Any
    health: Any
    audit: AutomaticRecoveryAudit | None


class BatchRunner:
    """Two-phase, sequential batch coordinator using existing safe workers."""

    def __init__(self, event_callback: BatchEventCallback | None = None) -> None:
        self.event_callback = event_callback
        self.cancel_event = threading.Event()
        self.active_processor: JobProcessor | None = None
        self.active_repairer: SourceRepairer | None = None

    def cancel(self) -> None:
        self.cancel_event.set()
        if self.active_processor is not None:
            self.active_processor.cancel()
        if self.active_repairer is not None:
            self.active_repairer.cancel()

    def _emit(self, kind: str, record: BatchRecord | None = None, payload: Any = None) -> None:
        if self.event_callback:
            self.event_callback(kind, record, payload)

    def _check_canceled(self) -> None:
        if self.cancel_event.is_set():
            raise BatchCancelled("Batch processing canceled")

    def _set_row(
        self,
        record: BatchRecord,
        *,
        state: str | None = None,
        analysis: str | None = None,
        effective: str | None = None,
        progress: str | None = None,
        percent: float | None = None,
        error: str | None = None,
    ) -> None:
        if state is not None:
            record.state = state
        if analysis is not None:
            record.analysis_text = analysis
        if effective is not None:
            record.effective_text = effective
        if progress is not None:
            record.progress_text = progress
        if percent is not None:
            record.progress_percent = max(0.0, min(100.0, percent))
        if error is not None:
            record.error = error
        self._emit("row", record)

    def _log(self, record: BatchRecord, text: str) -> None:
        self._emit("log", record, text)

    @staticmethod
    def _annotate_plan(plan: ProcessingPlan, resolution: BatchResolution) -> ProcessingPlan:
        warnings = tuple(
            f"Batch compatibility decision: {note}" for note in resolution.fallback_notes
        )
        return replace(
            plan,
            warnings=tuple(dict.fromkeys((*plan.warnings, *warnings))),
        )

    def _analyze_record(
        self,
        record: BatchRecord,
        capabilities: CapabilityReport,
        mode: str,
    ) -> tuple[MediaProbe, Any, Any]:
        self._check_canceled()
        identity = source_identity(record.source_path)
        if (
            record.source_identity == identity
            and record.media is not None
            and record.analysis is not None
            and record.source_health is not None
            and health_matches_source(record.source_health, record.source_path)
        ):
            self._set_row(
                record,
                analysis=f"{record.analysis.classification.upper()} · cached",
                progress="Reused unchanged analysis",
            )
            return record.media, record.analysis, record.source_health

        if not capabilities.ffprobe_path or not capabilities.ffmpeg_path:
            raise BatchCompatibilityError("FFmpeg and FFprobe must pass capability discovery before batch preflight.")
        self._set_row(record, state="Preflighting", analysis="Probing", progress="Reading media metadata", percent=0)
        media = probe_media(capabilities.ffprobe_path, record.source_path, sample_frames=64)
        self._check_canceled()

        def health_progress(packets: int, fraction: float | None) -> None:
            self._check_canceled()
            percent = (fraction or 0.0) * 35.0
            self._set_row(
                record,
                analysis="Source-health scan",
                progress=f"{packets:,} packets",
                percent=percent,
            )

        health = scan_source_health(
            capabilities.ffprobe_path,
            media,
            cancel_event=self.cancel_event,
            progress=health_progress,
        )
        self._check_canceled()

        def idet_progress(done: int, total: int, offset: float) -> None:
            self._check_canceled()
            fraction = done / total if total else 0.0
            self._set_row(
                record,
                analysis=f"IDet {done}/{total}",
                progress=f"Analyzing near {offset:.1f}s",
                percent=35.0 + fraction * 65.0,
            )

        analysis = scan_idet(
            capabilities.ffmpeg_path,
            media,
            mode=mode,
            cancel_event=self.cancel_event,
            progress=idet_progress,
        )
        record.media = media
        record.analysis = analysis
        record.source_health = health
        record.source_identity = identity
        self._set_row(
            record,
            analysis=f"{analysis.classification.upper()} · {analysis.confidence:.0%}",
            progress="Preflight analyzed",
            percent=100,
        )
        return media, analysis, health

    def _repair_for_qtgmc(
        self,
        record: BatchRecord,
        resolution: BatchResolution,
        capabilities: CapabilityReport,
    ) -> _RepairOutcome:
        self._check_canceled()
        assert record.media is not None
        assert record.analysis is not None
        assert record.source_health is not None
        original_source = record.source_path
        preferred_repair = original_source.with_name(f"{original_source.stem}.qtgmc-repair.mkv")
        repair_output = choose_available_artifact_path(
            preferred_repair,
            REPAIR_ARTIFACT_SUFFIXES,
            reserved=(original_source, resolution.settings.output_path),
        )
        checks = storage_preflight(
            record.media,
            resolution.plan,
            repair_output,
            resolution.settings.output_path,
        )
        storage = storage_summary(checks)
        if any(not check.sufficient for check in checks):
            raise BatchCompatibilityError(
                "Automatic repair storage preflight failed: " + storage
            )
        self._set_row(record, state="Repairing", progress="Automatic repair 1/3", percent=0)
        self._log(record, "Automatic QTGMC repair started. " + storage)
        repairer = SourceRepairer()
        self.active_repairer = repairer

        def repair_progress(values: dict[str, str]) -> None:
            self._check_canceled()
            phase = values.get("phase", "repair")
            frame = values.get("frame")
            detail = f"{phase} · frame {frame}" if frame else phase
            self._set_row(record, progress="Repair 1/3 · " + detail)

        result: RepairResult
        try:
            result = repairer.run(
                RepairRequest(original_source, repair_output, mode="automatic", overwrite_approved=False),
                record.media,
                capabilities,
                log_callback=lambda line: self._log(record, line),
                progress_callback=repair_progress,
            )
        finally:
            self.active_repairer = None
        if result.canceled or self.cancel_event.is_set():
            raise BatchCancelled(result.message or "Batch repair canceled")
        if not result.success:
            raise BatchCompatibilityError(result.message)

        # A successful no-op diagnosis means the fast packet scan was a false
        # positive. The original is retained and JobProcessor's decoded source
        # contract remains the final authority.
        validated_source = result.output_path or original_source
        if result.output_path is None:
            self._log(record, "Deep repair diagnosis found no repairable defect; using the original for strict preflight.")
            return _RepairOutcome(original_source, record.media, record.analysis, record.source_health, None)

        self._set_row(record, state="Reanalyzing", progress="Repair re-analysis 2/3", percent=0)
        if not capabilities.ffprobe_path or not capabilities.ffmpeg_path:
            raise BatchCompatibilityError("FFmpeg/FFprobe became unavailable during repair re-analysis.")
        repaired_media = probe_media(capabilities.ffprobe_path, validated_source, sample_frames=64)
        repaired_health = scan_source_health(
            capabilities.ffprobe_path,
            repaired_media,
            cancel_event=self.cancel_event,
        )
        repaired_analysis = scan_idet(
            capabilities.ffmpeg_path,
            repaired_media,
            mode=record.analysis.mode,
            cancel_event=self.cancel_event,
        )
        if repaired_health.repair_required:
            raise BatchCompatibilityError("The validated repair copy still reports a repair-required timeline.")
        audit = AutomaticRecoveryAudit(
            original_source=original_source,
            trigger_health=record.source_health,
            requested_output=resolution.settings.output_path,
            selected_output=resolution.settings.output_path,
            repair_output=validated_source,
            repair_method=result.method or "automatic",
            repair_output_sha256=result.output_sha256,
            repair_log_path=result.log_path,
            repair_report_path=result.report_path,
            repeated_frames=result.repeated_frames,
            dropped_frames=result.dropped_frames,
            storage_preflight=storage,
        )
        return _RepairOutcome(validated_source, repaired_media, repaired_analysis, repaired_health, audit)

    def _executable_plan(
        self,
        record: BatchRecord,
        resolution: BatchResolution,
        capabilities: CapabilityReport,
        *,
        allow_repair: bool,
    ) -> tuple[ProcessingPlan, MediaProbe, Any, bool]:
        assert record.media is not None
        assert record.analysis is not None
        assert record.source_health is not None
        if not resolution.requires_repair:
            return resolution.plan, record.media, record.analysis, False
        if not allow_repair:
            raise BatchCompatibilityError(
                "QTGMC requires repair for this source, but automatic batch repair is disabled.",
                needs_review=True,
            )
        repaired = self._repair_for_qtgmc(record, resolution, capabilities)
        if repaired.source_path == record.source_path and repaired.audit is None:
            plan = build_plan(
                resolution.settings,
                record.media,
                record.analysis,
                capabilities,
                source_health=None,
            )
            if not plan.valid:
                raise BatchCompatibilityError("Strict original-source retry could not build: " + "; ".join(plan.errors))
            return self._annotate_plan(plan, resolution), record.media, record.analysis, True

        repaired_settings = replace(
            resolution.settings,
            input_path=repaired.source_path,
            output_path=resolution.settings.output_path,
        )
        plan = build_plan(
            repaired_settings,
            repaired.media,
            repaired.analysis,
            capabilities,
            source_health=repaired.health,
            automatic_recovery=repaired.audit,
        )
        if not plan.valid:
            raise BatchCompatibilityError("Repaired-source final plan is invalid: " + "; ".join(plan.errors))
        return self._annotate_plan(plan, resolution), repaired.media, repaired.analysis, True

    def _process_record(
        self,
        record: BatchRecord,
        resolution: BatchResolution,
        capabilities: CapabilityReport,
        options: BatchRunOptions,
    ) -> None:
        plan, source_media, analysis, repair_attempted = self._executable_plan(
            record,
            resolution,
            capabilities,
            allow_repair=options.auto_repair,
        )
        while True:
            self._check_canceled()
            self._set_row(record, state="Processing", progress="Final processing", percent=0)
            processor = JobProcessor()
            self.active_processor = processor

            def progress(values: dict[str, str]) -> None:
                self._check_canceled()
                phase = values.get("phase", "processing")
                percent: float | None = None
                if "percent" in values:
                    try:
                        percent = float(values["percent"])
                    except ValueError:
                        percent = None
                elif values.get("frame") and values.get("expected_frames"):
                    try:
                        percent = 100.0 * int(values["frame"]) / int(values["expected_frames"])
                    except (ValueError, ZeroDivisionError):
                        percent = None
                detail = phase.replace("_", " ")
                if values.get("frame"):
                    detail += f" · frame {values['frame']}"
                self._set_row(record, progress=detail, percent=percent)

            try:
                result = processor.run(
                    plan,
                    source_media,
                    analysis,
                    capabilities,
                    log_callback=lambda line: self._log(record, line),
                    progress_callback=progress,
                )
            finally:
                self.active_processor = None
            if result.canceled or self.cancel_event.is_set():
                raise BatchCancelled(result.message or "Batch job canceled")
            if result.success:
                record.result_output = result.output_path
                self._set_row(record, state="Completed", progress="Validated", percent=100)
                return
            if (
                result.failure_code == SOURCE_REPAIR_REQUIRED_FAILURE
                and options.auto_repair
                and not repair_attempted
                and plan.selected_backend == "vapoursynth_qtgmc"
            ):
                self._log(record, "Decoded source preflight requested automatic repair; starting one bounded retry.")
                repair_attempted = True
                repaired = self._repair_for_qtgmc(record, resolution, capabilities)
                if repaired.source_path == record.source_path and repaired.audit is None:
                    plan = build_plan(
                        resolution.settings,
                        record.media,
                        record.analysis,
                        capabilities,
                        source_health=None,
                    )
                    source_media = record.media
                    analysis = record.analysis
                else:
                    repaired_settings = replace(
                        resolution.settings,
                        input_path=repaired.source_path,
                        output_path=resolution.settings.output_path,
                    )
                    plan = build_plan(
                        repaired_settings,
                        repaired.media,
                        repaired.analysis,
                        capabilities,
                        source_health=repaired.health,
                        automatic_recovery=repaired.audit,
                    )
                    source_media = repaired.media
                    analysis = repaired.analysis
                if not plan.valid:
                    raise BatchCompatibilityError("Automatic repair retry plan is invalid: " + "; ".join(plan.errors))
                plan = self._annotate_plan(plan, resolution)
                continue
            raise BatchCompatibilityError(result.message)

    def run(
        self,
        batch_queue: BatchQueue,
        requested: JobSettings,
        capabilities: CapabilityReport,
        options: BatchRunOptions,
    ) -> BatchRunSummary:
        self.cancel_event.clear()
        if not batch_queue.records:
            return BatchRunSummary(0, 0, 0, 0, 0, 0)
        if options.output_directory is not None and not options.output_directory.is_dir():
            raise BatchCompatibilityError(f"Batch output directory does not exist: {options.output_directory}")

        reserved: list[Path] = []
        ready: list[tuple[BatchRecord, BatchResolution]] = []
        stop_after_error = False
        self._emit("phase", None, "Preflighting every row before any long encode starts")
        try:
            for index, record in enumerate(batch_queue.records, start=1):
                self._check_canceled()
                self._emit("overall", record, {"phase": "preflight", "index": index, "total": len(batch_queue.records)})
                record.reset_plan(retain_analysis=True)
                try:
                    media, analysis, health = self._analyze_record(record, capabilities, options.analysis_mode)
                    resolution = resolve_batch_plan(
                        requested,
                        record.source_path,
                        media,
                        analysis,
                        health,
                        capabilities,
                        output_directory=options.output_directory,
                        reserved_outputs=tuple(reserved),
                    )
                    record.settings = resolution.settings
                    record.plan = resolution.plan
                    record.output_path = resolution.settings.output_path
                    record.fallback_notes = resolution.fallback_notes
                    record.effective_text = resolution.effective_summary
                    record.state = "Ready"
                    record.progress_text = "Ready after preflight"
                    reserved.append(resolution.settings.output_path)
                    ready.append((record, resolution))
                    self._emit("row", record)
                    for note in resolution.fallback_notes:
                        self._log(record, "Compatibility fallback: " + note)
                except BatchCancelled:
                    raise
                except BatchCompatibilityError as exc:
                    state = "Needs review" if exc.needs_review else "Preflight failed"
                    self._set_row(record, state=state, progress="Not queued for encoding", error=str(exc))
                    self._log(record, f"{state}: {exc}")
                    if not options.continue_after_error:
                        stop_after_error = True
                        break
                except Exception as exc:
                    # A malformed/corrupt row must not terminate unrelated
                    # queue entries.  The exact exception is retained in the
                    # row and run log, while cancellation remains governed by
                    # the explicit BatchCancelled path above.
                    detail = f"{type(exc).__name__}: {exc}"
                    self._set_row(
                        record,
                        state="Preflight failed",
                        progress="Not queued for encoding",
                        error=detail,
                    )
                    self._log(record, "Preflight failed: " + detail)
                    if not options.continue_after_error:
                        stop_after_error = True
                        break

            if stop_after_error:
                for record in batch_queue.records:
                    if record.state == "Queued":
                        self._set_row(record, state="Skipped", progress="Stopped after preflight error")
                ready.clear()

            self._emit("phase", None, "Preflight complete; processing compatible rows sequentially")
            for index, (record, resolution) in enumerate(ready, start=1):
                self._check_canceled()
                self._emit("overall", record, {"phase": "processing", "index": index, "total": len(ready)})
                try:
                    self._process_record(record, resolution, capabilities, options)
                except BatchCancelled:
                    raise
                except BatchCompatibilityError as exc:
                    state = "Needs review" if exc.needs_review else "Failed"
                    self._set_row(record, state=state, progress="Processing stopped", error=str(exc))
                    self._log(record, f"{state}: {exc}")
                    if not options.continue_after_error:
                        for remaining, _resolution in ready[index:]:
                            self._set_row(remaining, state="Skipped", progress="Stopped after row failure")
                        break
                except Exception as exc:
                    detail = f"{type(exc).__name__}: {exc}"
                    self._set_row(
                        record,
                        state="Failed",
                        progress="Processing stopped",
                        error=detail,
                    )
                    self._log(record, "Failed: " + detail)
                    if not options.continue_after_error:
                        for remaining, _resolution in ready[index:]:
                            self._set_row(remaining, state="Skipped", progress="Stopped after row failure")
                        break
        except BatchCancelled:
            for record in batch_queue.records:
                if record.state in {"Queued", "Preflighting", "Ready", "Processing", "Repairing", "Reanalyzing"}:
                    self._set_row(record, state="Canceled", progress="Canceled safely")
        finally:
            self.active_processor = None
            self.active_repairer = None

        summary = BatchRunSummary(
            total=len(batch_queue.records),
            completed=sum(record.state == "Completed" for record in batch_queue.records),
            failed=sum(record.state in {"Failed", "Preflight failed"} for record in batch_queue.records),
            needs_review=sum(record.state == "Needs review" for record in batch_queue.records),
            canceled=sum(record.state == "Canceled" for record in batch_queue.records),
            skipped=sum(record.state == "Skipped" for record in batch_queue.records),
        )
        self._emit("complete", None, summary)
        return summary
