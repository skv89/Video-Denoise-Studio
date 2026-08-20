from __future__ import annotations

import hashlib
import json
import os
import queue
import subprocess
import threading
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from video_processing_core.runtime.dependencies import managed_runtime_environment
from video_processing_core.media.models import json_safe

from .models import DenoisePlan, DenoiseResult, LogCallback, ProgressCallback
from .validation import validate_denoise_output


CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0
CREATE_NEW_PROCESS_GROUP = 0x00000200 if os.name == "nt" else 0


class DenoiseCancelled(RuntimeError):
    pass


class DenoiseProcessingError(RuntimeError):
    pass


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _safe_unlink(path: Path | None) -> None:
    if path is None:
        return
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


class DenoiseProcessor:
    """Run one denoise plan with cancellation, validation, and atomic promotion."""

    def __init__(self) -> None:
        self.cancel_event = threading.Event()
        self._processes: set[subprocess.Popen] = set()
        self._lock = threading.Lock()

    def cancel(self) -> None:
        self.cancel_event.set()
        with self._lock:
            processes = tuple(self._processes)
        for process in processes:
            self._terminate(process)

    def _register(self, process: subprocess.Popen) -> None:
        with self._lock:
            self._processes.add(process)

    def _unregister(self, process: subprocess.Popen | None) -> None:
        if process is None:
            return
        with self._lock:
            self._processes.discard(process)

    @staticmethod
    def _terminate(process: subprocess.Popen) -> None:
        if process.poll() is not None:
            return
        try:
            process.terminate()
            process.wait(timeout=3)
        except (OSError, subprocess.TimeoutExpired):
            try:
                process.kill()
                process.wait(timeout=3)
            except (OSError, subprocess.TimeoutExpired):
                pass

    def _check_canceled(self) -> None:
        if self.cancel_event.is_set():
            raise DenoiseCancelled("Denoise processing canceled.")

    @staticmethod
    def _emit(callback: ProgressCallback | None, phase: str, **values: object) -> None:
        if callback:
            payload = {"phase": phase}
            payload.update({key: str(value) for key, value in values.items() if value is not None})
            callback(payload)

    def _run_pipeline(
        self,
        plan: DenoisePlan,
        write_log: LogCallback,
        progress_callback: ProgressCallback | None,
    ) -> None:
        vspipe_process: subprocess.Popen | None = None
        ffmpeg_process: subprocess.Popen | None = None
        stderr_queue: queue.Queue[tuple[str, str]] = queue.Queue()
        drain_threads: list[threading.Thread] = []

        def drain(stream, label: str) -> None:
            try:
                for line in stream:
                    rendered = line.rstrip("\r\n")
                    if rendered:
                        stderr_queue.put((label, rendered))
            finally:
                try:
                    stream.close()
                except OSError:
                    pass

        def flush_stderr() -> None:
            while True:
                try:
                    label, line = stderr_queue.get_nowait()
                except queue.Empty:
                    return
                write_log(f"{label}: {line}")

        creationflags = CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP
        try:
            ffmpeg_stdin = subprocess.DEVNULL
            if plan.vspipe_command:
                env = managed_runtime_environment(plan.vspipe_command[0], os.environ.copy())
                vspipe_process = subprocess.Popen(
                    list(plan.vspipe_command),
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    creationflags=creationflags,
                    env=env,
                )
                self._register(vspipe_process)
                assert vspipe_process.stdout is not None
                assert vspipe_process.stderr is not None
                vspipe_drain = threading.Thread(
                    target=drain,
                    args=(_BinaryLineReader(vspipe_process.stderr), "VSPipe"),
                    daemon=True,
                )
                vspipe_drain.start()
                drain_threads.append(vspipe_drain)
                ffmpeg_stdin = vspipe_process.stdout

            ffmpeg_process = subprocess.Popen(
                list(plan.ffmpeg_command),
                stdin=ffmpeg_stdin,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=creationflags,
                env=os.environ.copy(),
            )
            self._register(ffmpeg_process)
            if vspipe_process and vspipe_process.stdout:
                vspipe_process.stdout.close()
            assert ffmpeg_process.stdout is not None
            assert ffmpeg_process.stderr is not None
            ffmpeg_drain = threading.Thread(target=drain, args=(ffmpeg_process.stderr, "FFmpeg"), daemon=True)
            ffmpeg_drain.start()
            drain_threads.append(ffmpeg_drain)

            progress_state: dict[str, str] = {}
            while True:
                self._check_canceled()
                line = ffmpeg_process.stdout.readline()
                flush_stderr()
                if not line:
                    if ffmpeg_process.poll() is not None:
                        break
                    self.cancel_event.wait(0.02)
                    continue
                rendered = line.strip()
                if "=" not in rendered:
                    continue
                key, value = rendered.split("=", 1)
                progress_state[key] = value
                if key == "progress":
                    payload = dict(progress_state)
                    payload["phase"] = "encode_progress"
                    if progress_callback:
                        progress_callback(payload)
                    progress_state.clear()

            ffmpeg_code = ffmpeg_process.wait()
            flush_stderr()
            if vspipe_process:
                try:
                    vspipe_code = vspipe_process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    self._terminate(vspipe_process)
                    vspipe_code = vspipe_process.returncode
            else:
                vspipe_code = 0
            flush_stderr()
            self._check_canceled()
            if ffmpeg_code != 0:
                raise DenoiseProcessingError(f"FFmpeg exited with code {ffmpeg_code}. See the retained run log.")
            if vspipe_code not in {0, None}:
                raise DenoiseProcessingError(f"VSPipe exited with code {vspipe_code}. See the retained run log.")
        finally:
            for process in (ffmpeg_process, vspipe_process):
                if process is not None and process.poll() is None:
                    self._terminate(process)
                self._unregister(process)
            for thread in drain_threads:
                thread.join(timeout=2)
            for process in (ffmpeg_process, vspipe_process):
                if process is None:
                    continue
                for stream in (process.stdout, process.stderr):
                    try:
                        if stream is not None and not stream.closed:
                            stream.close()
                    except OSError:
                        pass
            flush_stderr()

    def run(
        self,
        plan: DenoisePlan,
        *,
        log_callback: LogCallback | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> DenoiseResult:
        self.cancel_event.clear()
        if (
            not plan.valid
            or not plan.media
            or not plan.partial_path
            or not plan.log_path
            or not plan.report_path
            or not plan.expected
        ):
            detail = "; ".join(plan.errors) or "The denoise plan is incomplete."
            return DenoiseResult(False, False, detail, None, None, None, None, None, None)

        if plan.output_path.exists():
            return DenoiseResult(False, False, "The output appeared after planning; processing was not started.", None, None, None, None, None, None)
        if plan.partial_path.exists():
            return DenoiseResult(False, False, f"Unique partial path already exists: {plan.partial_path}", None, None, None, None, None, None)

        started = datetime.now(timezone.utc)
        started_monotonic = time.monotonic()
        try:
            log_stream = plan.log_path.open("x", encoding="utf-8", errors="replace")
        except FileExistsError:
            return DenoiseResult(
                False,
                False,
                f"The run log appeared after planning; processing was not started: {plan.log_path}",
                None,
                None,
                None,
                None,
                None,
                None,
            )
        quarantine: Path | None = None
        validation = None
        output_hash: str | None = None
        final_script: Path | None = None
        status = "failed"
        message = "Denoise processing failed."

        def write_log(line: str) -> None:
            rendered = str(line).rstrip("\r\n")
            timestamp = datetime.now(timezone.utc).isoformat()
            log_stream.write(f"[{timestamp}] {rendered}\n")
            log_stream.flush()
            if log_callback:
                log_callback(rendered)

        try:
            write_log(f"Video Denoise Studio plan started: {plan.display_command}")
            for warning in plan.warnings:
                write_log("Plan warning: " + warning)
            if plan.vapoursynth_script and plan.temporary_script_path:
                plan.temporary_script_path.write_text(plan.vapoursynth_script, encoding="utf-8")
                write_log(f"Created unique VapourSynth script: {plan.temporary_script_path}")
            self._check_canceled()
            self._emit(progress_callback, "encode_start")
            self._run_pipeline(plan, write_log, progress_callback)
            self._check_canceled()
            self._emit(progress_callback, "validation_start")
            assert plan.media and plan.expected
            ffprobe = plan.ffprobe_path
            if not ffprobe or not ffprobe.is_file():
                raise DenoiseProcessingError(f"Capability-selected FFprobe disappeared before validation: {ffprobe}")
            validation = validate_denoise_output(
                ffprobe,
                plan.partial_path,
                plan,
                thorough_packet_count=True,
            )
            for warning in validation.warnings:
                write_log("Validation warning: " + warning)
            if not validation.valid:
                for error in validation.errors:
                    write_log("Validation error: " + error)
                raise DenoiseProcessingError("The partial output failed validation.")
            self._emit(progress_callback, "validation_complete")
            self._check_canceled()
            if plan.output_path.exists():
                raise DenoiseProcessingError("The output appeared before promotion; the validated partial was not promoted.")
            os.replace(plan.partial_path, plan.output_path)
            write_log("Validated partial promoted atomically to the final output path.")
            self._emit(progress_callback, "final_validation_start")
            final_validation = validate_denoise_output(
                ffprobe,
                plan.output_path,
                plan,
                thorough_packet_count=False,
            )
            if not final_validation.valid:
                validation = final_validation
                for error in final_validation.errors:
                    write_log("Final reopen validation error: " + error)
                raise DenoiseProcessingError("The promoted output failed final reopen validation.")
            partial_validation = validation
            validation = replace(
                final_validation,
                verified_packet_count=partial_validation.verified_packet_count,
                verified_key_packet_count=partial_validation.verified_key_packet_count,
                thorough_packet_scan_completed=partial_validation.thorough_packet_scan_completed,
                warnings=tuple(dict.fromkeys(partial_validation.warnings + final_validation.warnings)),
            )
            self._emit(progress_callback, "final_validation_complete")
            self._emit(progress_callback, "hash_start")
            output_hash = sha256_file(plan.output_path)
            self._emit(progress_callback, "hash_complete")
            status = "success"
            message = "Denoise output completed, validated, reopened, and hashed."
        except DenoiseCancelled as exc:
            status = "canceled"
            message = str(exc)
            _safe_unlink(plan.partial_path)
            write_log(message)
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            write_log(message)
            failed_path = plan.output_path if plan.output_path.exists() else plan.partial_path
            if failed_path.exists():
                quarantine = failed_path.with_name(f"{failed_path.stem}.rejected.{uuid_token()}{failed_path.suffix}")
                try:
                    os.replace(failed_path, quarantine)
                    write_log(f"Rejected output retained for diagnosis: {quarantine}")
                except OSError as move_error:
                    write_log(f"Could not quarantine rejected output: {move_error}")
                    quarantine = None
        finally:
            if plan.temporary_script_path and plan.temporary_script_path.exists() and plan.script_path:
                try:
                    if not plan.script_path.exists():
                        os.replace(plan.temporary_script_path, plan.script_path)
                        final_script = plan.script_path
                    else:
                        _safe_unlink(plan.temporary_script_path)
                except OSError as script_error:
                    write_log(f"Could not preserve generated script: {script_error}")
            elapsed = time.monotonic() - started_monotonic
            write_log(f"Run status: {status}; elapsed seconds: {elapsed:.3f}")
            report = {
                "schema": 1,
                "application": "Video Denoise Studio",
                "status": status,
                "started_utc": started.isoformat(),
                "finished_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": elapsed,
                "message": message,
                "plan": json_safe(plan),
                "validation": json_safe(validation),
                "output_sha256": output_hash,
                "quarantine_path": str(quarantine) if quarantine else None,
            }
            try:
                _atomic_json(plan.report_path, report)
            except OSError as report_error:
                write_log(f"Could not write JSON report: {report_error}")
            log_stream.close()

        self._emit(progress_callback, "job_complete", status=status)
        return DenoiseResult(
            success=status == "success",
            canceled=status == "canceled",
            message=message,
            output_path=plan.output_path if status == "success" else None,
            log_path=plan.log_path,
            report_path=plan.report_path if plan.report_path.exists() else None,
            script_path=final_script,
            output_sha256=output_hash,
            validation=validation,
            quarantine_path=quarantine,
        )


def uuid_token() -> str:
    import uuid

    return uuid.uuid4().hex[:12]


class _BinaryLineReader:
    """Yield UTF-8 replacement-decoded lines from a binary pipe."""

    def __init__(self, stream) -> None:
        self.stream = stream

    def __iter__(self):
        for line in self.stream:
            yield line.decode("utf-8", errors="replace")

    def close(self) -> None:
        self.stream.close()
