from __future__ import annotations

import hashlib
import json
import os
import queue
import re
import shutil
import subprocess
import threading
import time
import uuid
from collections import deque
from dataclasses import replace
from pathlib import Path
from typing import TextIO

from . import __version__
from .denoise import denoiser_backend_display
from .models import (
    CapabilityReport,
    IDetReport,
    LogCallback,
    MediaProbe,
    ProcessingPlan,
    ProcessingResult,
    ProgressCallback,
    SOURCE_REPAIR_REQUIRED_FAILURE,
    SourcePreflightEvidence,
    ValidationResult,
    json_safe,
)
from .validation import validate_output
from .dependencies import managed_runtime_environment
from .health import health_matches_source


CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0
CREATE_NEW_PROCESS_GROUP = 0x00000200 if os.name == "nt" else 0
PROCESS_FLAGS = CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP
FATAL_ZERO_EXIT_PATTERNS = (
    "file duration too long for timebase",
    "conversion failed",
    "error writing trailer",
    "error closing file",
)
VSPIPE_PROGRESS_TOTAL = re.compile(r"Frame:\s*\d+/(\d+)")
VSPIPE_INFO_FRAMES = re.compile(r"^Frames:\s*(\d+)\s*$", re.MULTILINE | re.IGNORECASE)
WINDOWS_ACCESS_VIOLATION = 0xC0000005
CLOCK_DURATION = re.compile(r"^(\d+):(\d{2}):(\d{2}(?:\.\d+)?)$")
DECODE_FAILURE_PATTERNS = (
    "corrupt decoded frame",
    "corrupt input packet",
    "decode error",
    "error while decoding",
    "invalid data found when processing input",
    "missing reference picture",
    "reference picture missing",
    "truncated input",
)


class ProcessingCancelled(RuntimeError):
    pass


class ProcessingError(RuntimeError):
    pass


class SourceRepairRequiredError(ProcessingError):
    """A decoded source fault that the separate-copy repair workflow can handle."""


class FastPreflightUnavailable(RuntimeError):
    """Request a safe full-decoded fallback without treating it as job failure."""

    pass


def describe_vspipe_exit(code: int) -> str:
    unsigned = code & 0xFFFFFFFF
    if unsigned == WINDOWS_ACCESS_VIOLATION:
        return (
            "VSPipe crashed with Windows access violation 0xC0000005 "
            f"({unsigned}). This is a native VapourSynth/plugin failure, not an FFmpeg encode error. "
            "The app used bounded frame-request concurrency; update/recheck VapourSynth in Dependency Doctor "
            "and retain the log if the same source repeats the crash."
        )
    if code < 0 or code > 0x7FFFFFFF:
        return f"VSPipe exited with Windows status 0x{unsigned:08X} ({unsigned}). See the retained run log."
    return f"VSPipe exited with code {code}. See the retained run log."


def _video_timeline_duration(media: MediaProbe) -> float | None:
    if media.video.duration is not None:
        return media.video.duration
    for key in ("DURATION", "DURATION-eng"):
        value = media.video.tags.get(key)
        match = CLOCK_DURATION.fullmatch(value.strip()) if value else None
        if match:
            hours, minutes, seconds = match.groups()
            return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    return media.duration


def qtgmc_timeline_integrity_error(media: MediaProbe, decoded_frames: int) -> str | None:
    """Explain a source timeline that a CFR Y4M QTGMC pipe cannot preserve."""

    rate = media.video.avg_frame_rate or media.video.r_frame_rate
    duration = _video_timeline_duration(media)
    if not rate or rate <= 0 or duration is None or duration <= 0:
        return None
    decoded_span = decoded_frames / float(rate)
    tolerance = max(1.0, 2.0 / float(rate))
    difference = duration - decoded_span
    if abs(difference) <= tolerance:
        return None
    return (
        "QTGMC source timeline integrity check failed before encoding: FFprobe decoded "
        f"{decoded_frames} frames at {rate.numerator}/{rate.denominator} fps "
        f"({decoded_span:.3f}s), but the source video timeline reports {duration:.3f}s "
        f"(difference {difference:+.3f}s). This usually indicates missing/corrupt packets, timestamp gaps, "
        "or incorrect duration metadata. A constant-rate VSPipe/Y4M QTGMC pipe cannot preserve that timeline "
        "safely. This unchanged file will be blocked every time QTGMC is selected; changing the output codec "
        "will not repair it. No partial encode was started. FFmpeg BWDIF CPU/CUDA is a timestamp-aware fallback, "
        "not a repair: it may process the file but cannot restore missing/corrupt pictures, so check sync around "
        "damaged areas. Use Repair source… to diagnose the complete decoded timeline and create a separate, "
        "fully validated copy: metadata-only faults may use lossless stream-copy remuxing; real gaps use an FFV1 "
        "lossless rescue that repeats whole interlaced frames across unavailable time to preserve duration and "
        "audio sync. Missing pictures are not reconstructed—a clean re-rip/replacement remains the only way to "
        "recover the real scene. See Help → QTGMC source timeline guide."
    )


def vapoursynth_timeline_integrity_error(media: MediaProbe, decoded_frames: int) -> str | None:
    """Explain an unsafe timeline for a progressive VS denoise/Y4M pipe."""

    rate = media.video.avg_frame_rate or media.video.r_frame_rate
    duration = _video_timeline_duration(media)
    if not rate or rate <= 0 or duration is None or duration <= 0:
        return None
    decoded_span = decoded_frames / float(rate)
    tolerance = max(1.0, 2.0 / float(rate))
    difference = duration - decoded_span
    if abs(difference) <= tolerance:
        return None
    return (
        "VapourSynth temporal-denoise source timeline integrity check failed before encoding: FFprobe decoded "
        f"{decoded_frames} frames at {rate.numerator}/{rate.denominator} fps ({decoded_span:.3f}s), but the "
        f"source video timeline reports {duration:.3f}s (difference {difference:+.3f}s). A constant-rate "
        "VSPipe/Y4M graph cannot preserve this timestamp discrepancy safely. No partial encode was started. "
        "Use Repair source… to create a validated separate copy, or choose one of the FFmpeg temporal denoisers "
        "so the job remains in FFmpeg's timestamp-aware pipeline."
    )


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _snapshot(path: Path) -> tuple[bool, int | None, int | None, int | None]:
    try:
        stat = path.stat()
        return True, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns
    except FileNotFoundError:
        return False, None, None, None


def _promotion_identity(path: Path) -> tuple[int, int, int, int]:
    """Identify the exact file object across a same-directory atomic rename."""

    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_dev, stat.st_ino


def _safe_unlink(path: Path | None) -> None:
    if path is None:
        return
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


class JobProcessor:
    """Run one plan with cancellation, partial validation, and rollback."""

    def __init__(self) -> None:
        self.cancel_event = threading.Event()
        self._processes: list[subprocess.Popen] = []
        self._process_lock = threading.Lock()

    def cancel(self) -> None:
        self.cancel_event.set()
        with self._process_lock:
            processes = list(self._processes)
        for process in processes:
            if process.poll() is None:
                try:
                    process.terminate()
                except OSError:
                    pass

    def _register(self, process: subprocess.Popen) -> None:
        with self._process_lock:
            self._processes.append(process)

    def _unregister(self, process: subprocess.Popen) -> None:
        with self._process_lock:
            try:
                self._processes.remove(process)
            except ValueError:
                pass

    def _unregister_all(self) -> None:
        with self._process_lock:
            self._processes.clear()

    def _terminate_all(self) -> None:
        with self._process_lock:
            processes = list(self._processes)
        for process in processes:
            if process.poll() is None:
                try:
                    process.terminate()
                except OSError:
                    pass
        deadline = time.monotonic() + 5
        for process in processes:
            if process.poll() is not None:
                continue
            remaining = max(0.0, deadline - time.monotonic())
            try:
                process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                try:
                    process.kill()
                except OSError:
                    pass

    @staticmethod
    def _emit_progress(
        callback: ProgressCallback | None,
        phase: str,
        **values: object,
    ) -> None:
        if callback is None:
            return
        payload = {"phase": phase}
        payload.update({key: str(value) for key, value in values.items() if value is not None})
        callback(payload)

    @staticmethod
    def _cadence_multiplier(plan: ProcessingPlan) -> int:
        return (
            2
            if plan.selected_backend != "progressive" and plan.settings.output_cadence == "field_rate"
            else 1
        )

    @staticmethod
    def _fast_preflight_ineligibility(plan: ProcessingPlan, source_probe: MediaProbe) -> str | None:
        if not plan.vspipe_command or not plan.temporary_script_path:
            return "the selected plan does not use a complete VSPipe graph"
        if source_probe.video.codec_name.lower() != "ffv1":
            return "the source video codec is not FFV1"
        health = plan.source_health
        if health is None:
            return "no current full-file source-health result is attached to the plan"
        if not health_matches_source(health, source_probe.path):
            return "the source changed after its full-file health scan"
        if health.status != "clear":
            return f"source-health status is {health.status}, not clear"
        if health.ffprobe_returncode != 0 or health.scan_error:
            return "the packet scan did not finish cleanly"
        if health.packet_count <= 0:
            return "the packet scan reported no video packets"
        if health.timestamped_packet_count != health.packet_count:
            return "not every video packet has a usable presentation timestamp"
        if health.unique_timestamp_count != health.packet_count:
            return "video packet timestamps are not one-to-one"
        if health.material_gap_count:
            return "the packet scan measured one or more material timestamp gaps"
        if health.demux_warning_count or health.structural_warning_count:
            return "the packet scan reported demux or structural warnings"
        return None

    def _run_vspipe_info(
        self,
        plan: ProcessingPlan,
        write_log,
        progress_callback: ProgressCallback | None,
        *,
        timeout: float = 120.0,
    ) -> int:
        assert plan.vspipe_command and plan.temporary_script_path
        command = [plan.vspipe_command[0], "--info", str(plan.temporary_script_path)]
        env = managed_runtime_environment(command[0], os.environ.copy())
        started = time.monotonic()
        self._emit_progress(progress_callback, "preflight_indexed_start", elapsed_seconds=0)
        write_log(
            "Beginning indexed VSPipe graph check for an unchanged, packet-clean FFV1 source; "
            "the graph is inspected without requesting decoded frames."
        )
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=PROCESS_FLAGS,
            env=env,
        )
        self._register(process)
        stdout = ""
        stderr = ""
        try:
            while True:
                if self.cancel_event.is_set():
                    raise ProcessingCancelled("Processing canceled during the indexed source preflight")
                remaining = timeout - (time.monotonic() - started)
                if remaining <= 0:
                    raise FastPreflightUnavailable(
                        f"VSPipe --info did not finish within {timeout:.0f} seconds"
                    )
                try:
                    stdout, stderr = process.communicate(timeout=min(0.20, remaining))
                    break
                except subprocess.TimeoutExpired:
                    continue
            if self.cancel_event.is_set():
                raise ProcessingCancelled("Processing canceled during the indexed source preflight")
            for line in stdout.splitlines():
                write_log(f"[VSPipe info] {line}")
            for line in stderr.splitlines():
                write_log(f"[VSPipe info] {line}")
            if process.returncode != 0:
                raise FastPreflightUnavailable(
                    f"VSPipe --info exited with code {process.returncode}"
                )
            match = VSPIPE_INFO_FRAMES.search(stdout + "\n" + stderr)
            if not match or int(match.group(1)) <= 0:
                raise FastPreflightUnavailable("VSPipe --info did not report a positive graph frame total")
            return int(match.group(1))
        finally:
            if process.poll() is None:
                try:
                    process.terminate()
                    process.wait(timeout=5)
                except (OSError, subprocess.TimeoutExpired):
                    try:
                        process.kill()
                    except OSError:
                        pass
            self._unregister(process)
            for stream in (process.stdout, process.stderr):
                if stream is not None:
                    try:
                        stream.close()
                    except (OSError, ValueError):
                        pass

    @staticmethod
    def _estimated_source_frames(plan: ProcessingPlan, source_probe: MediaProbe) -> int | None:
        if source_probe.video.nb_frames and source_probe.video.nb_frames > 0:
            return source_probe.video.nb_frames
        health = plan.source_health
        if health and health_matches_source(health, source_probe.path) and health.packet_count > 0:
            return health.packet_count
        duration = _video_timeline_duration(source_probe)
        rate = source_probe.video.avg_frame_rate or source_probe.video.r_frame_rate
        if duration and duration > 0 and rate and rate > 0:
            return max(1, round(duration * float(rate)))
        return None

    def _run_decoded_preflight(
        self,
        ffmpeg_path: Path,
        plan: ProcessingPlan,
        source_probe: MediaProbe,
        write_log,
        progress_callback: ProgressCallback | None,
    ) -> int:
        command = [
            str(ffmpeg_path),
            "-hide_banner",
            "-nostdin",
            "-v",
            "warning",
            "-i",
            str(source_probe.path),
            "-map",
            "0:v:0",
            "-an",
            "-sn",
            "-dn",
            "-fps_mode",
            "passthrough",
            "-progress",
            "pipe:1",
            "-nostats",
            "-f",
            "null",
            os.devnull,
        ]
        expected_frames = self._estimated_source_frames(plan, source_probe)
        started = time.monotonic()
        self._emit_progress(
            progress_callback,
            "preflight_full_start",
            expected_frames=expected_frames,
            elapsed_seconds=0,
        )
        write_log(
            "Beginning managed full decoded-frame/timeline fallback; no output encode has started. "
            "This process is cancellable and reports live progress."
        )
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=PROCESS_FLAGS,
            env=os.environ.copy(),
        )
        self._register(process)
        stdout_lines: queue.Queue[str | None] = queue.Queue()
        stderr_tail: deque[str] = deque(maxlen=4000)

        def read_stdout() -> None:
            assert process.stdout is not None
            try:
                for line in process.stdout:
                    stdout_lines.put(line)
            finally:
                stdout_lines.put(None)

        def drain_stderr() -> None:
            assert process.stderr is not None
            for line in process.stderr:
                rendered = line.rstrip()
                stderr_tail.append(rendered)
                write_log(f"[FFmpeg preflight] {rendered}")

        stdout_thread = threading.Thread(target=read_stdout, daemon=True)
        stderr_thread = threading.Thread(target=drain_stderr, daemon=True)
        stdout_thread.start()
        stderr_thread.start()
        progress_state: dict[str, str] = {}
        source_frames = 0
        stdout_done = False
        try:
            while not (stdout_done and process.poll() is not None):
                if self.cancel_event.is_set():
                    raise ProcessingCancelled("Processing canceled during the full decoded source preflight")
                try:
                    line = stdout_lines.get(timeout=0.10)
                except queue.Empty:
                    continue
                if line is None:
                    stdout_done = True
                    continue
                key, separator, value = line.strip().partition("=")
                if not separator:
                    continue
                progress_state[key] = value
                if key != "progress":
                    continue
                try:
                    source_frames = max(source_frames, int(progress_state.get("frame", "0").strip()))
                except ValueError:
                    pass
                elapsed = time.monotonic() - started
                percent = (
                    min(100.0, 100.0 * source_frames / expected_frames)
                    if expected_frames
                    else None
                )
                eta_seconds = (
                    elapsed * max(0, expected_frames - source_frames) / source_frames
                    if expected_frames and source_frames > 0
                    else None
                )
                self._emit_progress(
                    progress_callback,
                    "preflight_full_progress",
                    frame=source_frames,
                    expected_frames=expected_frames,
                    percent=f"{percent:.3f}" if percent is not None else None,
                    eta_seconds=f"{eta_seconds:.3f}" if eta_seconds is not None else None,
                    speed=progress_state.get("speed"),
                    out_time_us=progress_state.get("out_time_us", progress_state.get("out_time_ms")),
                    elapsed_seconds=f"{elapsed:.3f}",
                )
                progress_state.clear()
            returncode = process.wait()
            stdout_thread.join(timeout=5)
            stderr_thread.join(timeout=5)
            if self.cancel_event.is_set():
                raise ProcessingCancelled("Processing canceled during the full decoded source preflight")
            diagnostics = "\n".join(stderr_tail).lower()
            fatal_text = next((pattern for pattern in DECODE_FAILURE_PATTERNS if pattern in diagnostics), None)
            if fatal_text:
                raise SourceRepairRequiredError(
                    "Full decoded source preflight reported a decoder-integrity fault "
                    f"({fatal_text}). No output encode was started; see the retained run log."
                )
            if returncode != 0:
                raise ProcessingError(
                    f"Full decoded source preflight failed: FFmpeg exited with code {returncode}. "
                    "No output encode was started; see the retained run log."
                )
            if source_frames <= 0:
                raise ProcessingError(
                    "Full decoded source preflight completed without a positive frame count. "
                    "No output encode was started."
                )
            return source_frames
        finally:
            if process.poll() is None:
                try:
                    process.terminate()
                    process.wait(timeout=5)
                except (OSError, subprocess.TimeoutExpired):
                    try:
                        process.kill()
                    except OSError:
                        pass
            stdout_thread.join(timeout=5)
            stderr_thread.join(timeout=5)
            self._unregister(process)
            for stream in (process.stdout, process.stderr):
                if stream is not None:
                    try:
                        stream.close()
                    except (OSError, ValueError):
                        pass

    def _source_preflight(
        self,
        plan: ProcessingPlan,
        source_probe: MediaProbe,
        capabilities: CapabilityReport,
        write_log,
        progress_callback: ProgressCallback | None,
    ) -> SourcePreflightEvidence:
        started = time.monotonic()
        multiplier = self._cadence_multiplier(plan)
        if not plan.vspipe_command:
            evidence = SourcePreflightEvidence(
                method="not_required_timestamp_aware_ffmpeg",
                source_frames=None,
                expected_output_frames=plan.expected.frame_count if plan.expected else None,
                elapsed_seconds=time.monotonic() - started,
            )
            write_log(
                "Decoded preflight not required: the selected FFmpeg-only pipeline remains timestamp-aware and "
                "will decode the source once during the actual job."
            )
            self._emit_progress(
                progress_callback,
                "preflight_complete",
                method=evidence.method,
                expected_frames=evidence.expected_output_frames,
                elapsed_seconds=f"{evidence.elapsed_seconds:.3f}",
            )
            return evidence

        if not capabilities.ffmpeg_path:
            raise ProcessingError("FFmpeg is unavailable for the decoded source preflight fallback.")

        health = plan.source_health
        packet_count = health.packet_count if health else None
        ineligibility = self._fast_preflight_ineligibility(plan, source_probe)
        graph_frames: int | None = None
        fallback_reason = ineligibility
        fast_eligible = ineligibility is None
        if fast_eligible:
            try:
                graph_frames = self._run_vspipe_info(plan, write_log, progress_callback)
                if graph_frames % multiplier:
                    fallback_reason = (
                        f"VSPipe graph frame total {graph_frames} is not divisible by cadence multiplier {multiplier}"
                    )
                else:
                    source_frames = graph_frames // multiplier
                    if source_frames != packet_count:
                        fallback_reason = (
                            f"VSPipe graph implies {source_frames} source frames but the unchanged FFV1 packet "
                            f"scan counted {packet_count}"
                        )
                    else:
                        timeline_error = (
                            qtgmc_timeline_integrity_error(source_probe, source_frames)
                            if plan.selected_backend == "vapoursynth_qtgmc"
                            else vapoursynth_timeline_integrity_error(source_probe, source_frames)
                        )
                        if timeline_error:
                            raise SourceRepairRequiredError(timeline_error)
                        elapsed = time.monotonic() - started
                        evidence = SourcePreflightEvidence(
                            method="vspipe_info_ffv1_packet_contract",
                            source_frames=source_frames,
                            expected_output_frames=graph_frames,
                            elapsed_seconds=elapsed,
                            packet_count=packet_count,
                            graph_output_frames=graph_frames,
                            fast_path_eligible=True,
                        )
                        write_log(
                            f"Indexed preflight established an exact {source_frames}-source-frame / "
                            f"{graph_frames}-output-frame contract in {elapsed:.3f}s; unchanged FFV1 packet count "
                            "and VSPipe graph total agree one-to-one."
                        )
                        self._emit_progress(
                            progress_callback,
                            "preflight_complete",
                            method=evidence.method,
                            source_frames=source_frames,
                            expected_frames=graph_frames,
                            elapsed_seconds=f"{elapsed:.3f}",
                        )
                        return evidence
            except FastPreflightUnavailable as exc:
                fallback_reason = str(exc)

        write_log(
            "Indexed preflight is not authoritative for this source; using the managed full decoded fallback. "
            f"Reason: {fallback_reason or 'no eligible indexed contract was available'}."
        )
        source_frames = self._run_decoded_preflight(
            capabilities.ffmpeg_path,
            plan,
            source_probe,
            write_log,
            progress_callback,
        )
        expected_output_frames = source_frames * multiplier
        if graph_frames is not None and graph_frames != expected_output_frames:
            raise ProcessingError(
                f"VSPipe graph reported {graph_frames} output frames, but the full decoded preflight established "
                f"{expected_output_frames}; no output encode was started."
            )
        timeline_error = (
            qtgmc_timeline_integrity_error(source_probe, source_frames)
            if plan.selected_backend == "vapoursynth_qtgmc"
            else vapoursynth_timeline_integrity_error(source_probe, source_frames)
        )
        if timeline_error:
            raise SourceRepairRequiredError(timeline_error)
        elapsed = time.monotonic() - started
        evidence = SourcePreflightEvidence(
            method="managed_full_decode",
            source_frames=source_frames,
            expected_output_frames=expected_output_frames,
            elapsed_seconds=elapsed,
            packet_count=packet_count,
            graph_output_frames=graph_frames,
            fast_path_eligible=fast_eligible,
            fallback_reason=fallback_reason,
        )
        write_log(
            f"Full decoded preflight established {source_frames} source frames and exactly "
            f"{expected_output_frames} expected output frames in {elapsed:.3f}s."
        )
        self._emit_progress(
            progress_callback,
            "preflight_complete",
            method=evidence.method,
            source_frames=source_frames,
            expected_frames=expected_output_frames,
            elapsed_seconds=f"{elapsed:.3f}",
        )
        return evidence

    def _run_pipeline(
        self,
        plan: ProcessingPlan,
        log: TextIO,
        write_log,
        progress_callback: ProgressCallback | None,
        expected_frames: int | None = None,
    ) -> int | None:
        env = os.environ.copy()
        env["AV_LOG_FORCE_NOCOLOR"] = "1"
        if plan.vspipe_command:
            env = managed_runtime_environment(plan.vspipe_command[0], env)
        stderr_tail: deque[str] = deque(maxlen=4000)
        stderr_threads: list[threading.Thread] = []
        vspipe_total_frames: int | None = None

        def drain_stderr(stream, prefix: str) -> None:
            nonlocal vspipe_total_frames
            try:
                for line in iter(stream.readline, ""):
                    rendered = f"[{prefix}] {line.rstrip()}"
                    if prefix == "VSPipe":
                        totals = VSPIPE_PROGRESS_TOTAL.findall(line)
                        if totals:
                            vspipe_total_frames = int(totals[-1])
                    stderr_tail.append(rendered)
                    write_log(rendered)
            finally:
                try:
                    stream.close()
                except OSError:
                    pass

        vspipe_process: subprocess.Popen | None = None
        ffmpeg_process: subprocess.Popen | None = None
        try:
            if plan.vspipe_command:
                vspipe_process = subprocess.Popen(
                    list(plan.vspipe_command),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    creationflags=PROCESS_FLAGS,
                    env=env,
                )
                self._register(vspipe_process)
                assert vspipe_process.stdout is not None
                assert vspipe_process.stderr is not None
                thread = threading.Thread(
                    target=drain_stderr,
                    args=(
                        # Decode VSPipe bytes without blocking its video stdout.
                        _BinaryLineDecoder(vspipe_process.stderr),
                        "VSPipe",
                    ),
                    daemon=True,
                )
                thread.start()
                stderr_threads.append(thread)
                ffmpeg_stdin = vspipe_process.stdout
            else:
                ffmpeg_stdin = subprocess.DEVNULL

            ffmpeg_process = subprocess.Popen(
                list(plan.ffmpeg_command),
                stdin=ffmpeg_stdin,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=PROCESS_FLAGS,
                env=env,
            )
            self._register(ffmpeg_process)
            if vspipe_process and vspipe_process.stdout:
                vspipe_process.stdout.close()
            assert ffmpeg_process.stdout is not None
            assert ffmpeg_process.stderr is not None
            stderr_thread = threading.Thread(target=drain_stderr, args=(ffmpeg_process.stderr, "FFmpeg"), daemon=True)
            stderr_thread.start()
            stderr_threads.append(stderr_thread)

            progress_state: dict[str, str] = {}
            for line in ffmpeg_process.stdout:
                if self.cancel_event.is_set():
                    raise ProcessingCancelled("Processing canceled")
                key, separator, value = line.strip().partition("=")
                if separator:
                    progress_state[key] = value
                    if key == "progress" and progress_callback:
                        payload = dict(progress_state)
                        payload["phase"] = "encode_progress"
                        if expected_frames is not None:
                            payload["expected_frames"] = str(expected_frames)
                        progress_callback(payload)
                        progress_state.clear()
            ffmpeg_process.stdout.close()

            while ffmpeg_process.poll() is None:
                if self.cancel_event.wait(0.05):
                    raise ProcessingCancelled("Processing canceled")
            ffmpeg_code = ffmpeg_process.wait()
            if vspipe_process:
                try:
                    vspipe_code = vspipe_process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    vspipe_process.terminate()
                    vspipe_code = vspipe_process.wait(timeout=5)
            else:
                vspipe_code = 0
            for thread in stderr_threads:
                thread.join(timeout=5)

            combined_tail = "\n".join(stderr_tail).lower()
            fatal_text = next((pattern for pattern in FATAL_ZERO_EXIT_PATTERNS if pattern in combined_tail), None)
            if ffmpeg_code != 0:
                raise ProcessingError(f"FFmpeg exited with code {ffmpeg_code}. See the retained run log.")
            if vspipe_code != 0:
                raise ProcessingError(describe_vspipe_exit(vspipe_code))
            if fatal_text:
                raise ProcessingError(f"FFmpeg emitted a fatal diagnostic despite its exit code: {fatal_text}")
            return vspipe_total_frames
        finally:
            with self._process_lock:
                unfinished = any(process.poll() is None for process in self._processes)
            if unfinished:
                self._terminate_all()
            for thread in stderr_threads:
                thread.join(timeout=5)
            for process in (ffmpeg_process, vspipe_process):
                if process is None:
                    continue
                for stream in (process.stdout, process.stderr):
                    if stream is not None:
                        try:
                            stream.close()
                        except (OSError, ValueError):
                            pass
            self._unregister_all()

    def run(
        self,
        plan: ProcessingPlan,
        source_probe: MediaProbe,
        analysis: IDetReport,
        capabilities: CapabilityReport,
        *,
        log_callback: LogCallback | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> ProcessingResult:
        self.cancel_event.clear()
        if not plan.valid or not plan.partial_path or not plan.log_path or not plan.report_path or not plan.expected:
            return ProcessingResult(False, False, "The processing plan is invalid.", None, None, None, None, None, None)
        if not capabilities.ffprobe_path:
            return ProcessingResult(False, False, "FFprobe is unavailable.", None, None, None, None, None, None)

        output = plan.output_path
        output.parent.mkdir(parents=True, exist_ok=True)
        completed_artifacts = [output, plan.log_path, plan.report_path, _sidecar_for_output(output, ".Deinterlace.vpy")]
        snapshots = {path: _snapshot(path) for path in completed_artifacts}
        source_snapshot = _snapshot(source_probe.path)
        existing = [path for path, state in snapshots.items() if state[0]]
        if existing and not plan.settings.overwrite_approved:
            return ProcessingResult(
                False,
                False,
                "Existing completed artifacts require explicit replacement approval: " + ", ".join(str(path) for path in existing),
                None,
                None,
                None,
                None,
                None,
                None,
            )
        if plan.partial_path.exists():
            return ProcessingResult(False, False, f"Unique partial path unexpectedly exists: {plan.partial_path}", None, None, None, None, None, None)

        run_token = uuid.uuid4().hex[:12]
        run_log = output.with_name(f".{output.stem}.Deinterlace.{run_token}.run.log")
        report_temp = output.with_name(f".{output.stem}.Deinterlace.{run_token}.json")
        backups: dict[Path, Path] = {}
        quarantine: Path | None = None
        promoted = False
        validation: ValidationResult | None = None
        effective_expected = plan.expected
        source_preflight: SourcePreflightEvidence | None = None
        final_hash: str | None = None
        failure_code: str | None = None
        log_lock = threading.Lock()

        with run_log.open("w", encoding="utf-8", buffering=1) as log:
            def write_log(message: str) -> None:
                stamp = time.strftime("%Y-%m-%d %H:%M:%S")
                rendered = f"{stamp} {message}"
                with log_lock:
                    log.write(rendered + "\n")
                    log.flush()
                if log_callback:
                    log_callback(rendered)

            try:
                write_log("Deinterlace Studio job started")
                write_log(f"Input: {source_probe.path}")
                write_log(f"Output: {output}")
                write_log(f"Backend: {plan.selected_backend}; profile: {plan.profile_label}")
                if plan.vapoursynth_schedule_note:
                    write_log("VapourSynth schedule: " + plan.vapoursynth_schedule_note)
                    write_log(
                        "Vulkan NNEDI3 interpolation: "
                        + ("enabled" if plan.vulkan_nnedi3_active else "disabled (CPU NNEDI3)")
                    )
                if plan.selected_denoiser:
                    denoise_implementation = denoiser_backend_display(
                        plan.selected_denoiser,
                        plan.selected_denoise_backend,
                    )
                    write_log(
                        "Temporal denoise: enabled after deinterlacing; "
                        f"algorithm={plan.selected_denoiser}; implementation={denoise_implementation}; "
                        f"implementation_id={plan.selected_denoise_backend}; "
                        f"strength={plan.settings.denoise_strength}; "
                        f"temporal_radius={plan.settings.denoise_temporal_radius}"
                    )
                else:
                    write_log("Temporal denoise: disabled")
                if plan.automatic_recovery:
                    recovery = plan.automatic_recovery
                    write_log(
                        "Automatic recovery chain: "
                        f"original={recovery.original_source}; trigger={recovery.trigger_health.reason}"
                    )
                    write_log(
                        "Validated repair: "
                        f"method={recovery.repair_method}; copy={recovery.repair_output}; "
                        f"SHA-256={recovery.repair_output_sha256 or 'not created'}; "
                        f"report={recovery.repair_report_path}"
                    )
                    write_log(f"Automatic recovery storage preflight: {recovery.storage_preflight}")
                write_log(f"Command: {plan.display_command}")
                if plan.vapoursynth_script and plan.temporary_script_path:
                    plan.temporary_script_path.write_text(plan.vapoursynth_script, encoding="utf-8", newline="\n")
                    write_log(f"Generated temporary VapourSynth script: {plan.temporary_script_path}")

                source_preflight = self._source_preflight(
                    plan,
                    source_probe,
                    capabilities,
                    write_log,
                    progress_callback,
                )
                if source_preflight.expected_output_frames is not None:
                    effective_expected = replace(
                        effective_expected,
                        frame_count=source_preflight.expected_output_frames,
                    )
                if self.cancel_event.is_set():
                    raise ProcessingCancelled("Processing canceled")

                self._emit_progress(
                    progress_callback,
                    "encode_start",
                    expected_frames=effective_expected.frame_count,
                )
                write_log("Source preflight complete; starting the unique partial output encode.")
                vspipe_total_frames = self._run_pipeline(
                    plan,
                    log,
                    write_log,
                    progress_callback,
                    effective_expected.frame_count,
                )
                if self.cancel_event.is_set():
                    raise ProcessingCancelled("Processing canceled")
                self._emit_progress(
                    progress_callback,
                    "encode_complete",
                    expected_frames=effective_expected.frame_count,
                )
                if effective_expected.frame_count is None and plan.vspipe_command:
                    if vspipe_total_frames is None:
                        raise ProcessingError(
                            "VSPipe completed without reporting its decoded output-frame total; exact completeness "
                            "validation cannot proceed."
                        )
                    effective_expected = replace(effective_expected, frame_count=vspipe_total_frames)
                    write_log(f"VSPipe reported exactly {vspipe_total_frames} decoded output frames.")
                elif plan.vspipe_command and vspipe_total_frames is not None:
                    if vspipe_total_frames != effective_expected.frame_count:
                        raise ProcessingError(
                            f"VSPipe reported {vspipe_total_frames} output frames, but source preflight expected "
                            f"{effective_expected.frame_count}; promotion was blocked."
                        )
                    write_log(f"VSPipe confirmed the expected {vspipe_total_frames} decoded output frames.")
                write_log("Encoder exited successfully; validating partial output")
                self._emit_progress(progress_callback, "validation_start")
                validation = validate_output(
                    capabilities.ffprobe_path,
                    plan.partial_path,
                    effective_expected,
                    plan.settings,
                    thorough_packet_count=True,
                )
                for warning in validation.warnings:
                    write_log("Validation warning: " + warning)
                if not validation.valid:
                    for error in validation.errors:
                        write_log("Validation error: " + error)
                    raise ProcessingError("Partial output failed validation.")
                partial_validation = validation
                partial_identity = _promotion_identity(plan.partial_path)
                write_log(
                    "Thorough partial validation passed: "
                    f"packet_count={partial_validation.verified_packet_count}; "
                    f"key_packet_count={partial_validation.verified_key_packet_count}; "
                    f"single_packet_scan={partial_validation.thorough_packet_scan_completed}."
                )
                self._emit_progress(progress_callback, "validation_complete")

                if _snapshot(source_probe.path) != source_snapshot:
                    raise ProcessingError("The source file changed while processing; promotion was aborted.")

                for path, original_state in snapshots.items():
                    if _snapshot(path) != original_state:
                        raise ProcessingError(
                            f"Approved completed artifact changed while processing: {path}. Replacement was aborted."
                        )

                write_log("Partial output passed validation; promoting with rollback protection")
                for artifact in completed_artifacts:
                    if artifact.exists():
                        backup = artifact.with_name(f".{artifact.name}.backup.{run_token}")
                        if backup.exists():
                            raise ProcessingError(f"Backup collision: {backup}")
                        os.replace(artifact, backup)
                        backups[artifact] = backup
                os.replace(plan.partial_path, output)
                promoted = True
                if _promotion_identity(output) != partial_identity:
                    raise ProcessingError(
                        "Atomic promotion did not preserve the validated partial's exact file identity."
                    )
                write_log("Atomic promotion preserved the thoroughly validated partial's exact file identity.")

                self._emit_progress(progress_callback, "final_validation_start")
                final_reopen_validation = validate_output(
                    capabilities.ffprobe_path,
                    output,
                    effective_expected,
                    plan.settings,
                    thorough_packet_count=False,
                )
                if not final_reopen_validation.valid:
                    for error in final_reopen_validation.errors:
                        write_log("Final reopen validation error: " + error)
                    raise ProcessingError("The promoted output failed final reopen validation.")
                validation = replace(
                    final_reopen_validation,
                    verified_packet_count=partial_validation.verified_packet_count,
                    verified_key_packet_count=partial_validation.verified_key_packet_count,
                    thorough_packet_scan_completed=partial_validation.thorough_packet_scan_completed,
                    warnings=tuple(
                        dict.fromkeys(partial_validation.warnings + final_reopen_validation.warnings)
                    ),
                )
                self._emit_progress(progress_callback, "final_validation_complete")

                write_log(
                    "Bounded final reopen validation passed without repeating the full packet scan; "
                    "calculating SHA-256 once."
                )
                self._emit_progress(progress_callback, "hash_start")
                hash_snapshot = _snapshot(output)
                final_hash = sha256_file(output)
                if _snapshot(output) != hash_snapshot:
                    raise ProcessingError("The promoted output changed while its SHA-256 was being calculated.")
                write_log(f"Output SHA-256: {final_hash}")
                self._emit_progress(progress_callback, "hash_complete")
                report_payload = {
                    "application": "Deinterlace Studio",
                    "application_version": __version__,
                    "completed_at_local": time.strftime("%Y-%m-%d %H:%M:%S %z"),
                    "source_file": {
                        "path": str(source_probe.path),
                        "size": source_probe.path.stat().st_size,
                        "mtime_ns": source_probe.path.stat().st_mtime_ns,
                    },
                    "source_probe": json_safe(source_probe),
                    "idet_analysis": json_safe(analysis),
                    "capabilities": json_safe(capabilities),
                    "plan": json_safe(plan),
                    "source_preflight": json_safe(source_preflight),
                    "effective_output_expectation": json_safe(effective_expected),
                    "final_validation": json_safe(validation),
                    "validation_strategy": {
                        "partial_thorough_packet_scans": 1,
                        "promoted_full_packet_rescans": 0,
                        "same_file_atomic_promotion_verified": True,
                        "final_reopen": "bounded probe and decoded-frame samples",
                        "sha256_passes": 1,
                    },
                    "output": {"path": str(output), "size": output.stat().st_size, "sha256": final_hash},
                }
                report_temp.write_text(json.dumps(report_payload, ensure_ascii=False, indent=2), encoding="utf-8")
                write_log("Audit report prepared")

                # Close/flush happens when leaving this block; promotion of the
                # log itself is performed immediately afterward.
            except ProcessingCancelled as exc:
                self._terminate_all()
                _safe_unlink(plan.partial_path)
                _safe_unlink(plan.temporary_script_path)
                write_log(str(exc))
                canceled_log = output.with_name(f"{output.stem}.Deinterlace.canceled.{run_token}.log")
                # The open log is moved after the context closes below.
                pending_result = ("canceled", str(exc), canceled_log)
            except Exception as exc:
                if isinstance(exc, SourceRepairRequiredError):
                    failure_code = SOURCE_REPAIR_REQUIRED_FAILURE
                self._terminate_all()
                _safe_unlink(plan.partial_path)
                _safe_unlink(plan.temporary_script_path)
                _safe_unlink(report_temp)
                write_log(f"Job failed: {exc}")
                # If promotion already happened, quarantine the new candidate
                # and restore every prior artifact.
                if promoted and output.exists():
                    quarantine = output.with_name(f"{output.stem}.failed.{run_token}{output.suffix}")
                    os.replace(output, quarantine)
                for original, backup in backups.items():
                    if backup.exists():
                        os.replace(backup, original)
                failed_log = output.with_name(f"{output.stem}.Deinterlace.failed.{run_token}.log")
                pending_result = ("failed", str(exc), failed_log)
            else:
                pending_result = ("success", "Completed and validated.", plan.log_path)

        status, message, destination_log = pending_result
        if status != "success":
            try:
                os.replace(run_log, destination_log)
            except OSError:
                destination_log = run_log
            return ProcessingResult(
                False,
                status == "canceled",
                message,
                None,
                destination_log,
                None,
                None,
                None,
                validation,
                quarantine,
                failure_code,
            )

        # Complete the sidecar set. Any failure here rolls the promoted media
        # back, because a success without its audit evidence is incomplete.
        promoted_sidecars: list[Path] = []
        try:
            os.replace(run_log, plan.log_path)
            promoted_sidecars.append(plan.log_path)
            os.replace(report_temp, plan.report_path)
            promoted_sidecars.append(plan.report_path)
            if plan.temporary_script_path and plan.script_path:
                os.replace(plan.temporary_script_path, plan.script_path)
                promoted_sidecars.append(plan.script_path)
            for backup in backups.values():
                _safe_unlink(backup)
        except Exception as exc:
            if output.exists():
                quarantine = output.with_name(f"{output.stem}.failed-sidecars.{run_token}{output.suffix}")
                os.replace(output, quarantine)
            for sidecar in promoted_sidecars:
                _safe_unlink(sidecar)
            for original, backup in backups.items():
                if backup.exists():
                    os.replace(backup, original)
            return ProcessingResult(
                False,
                False,
                f"Final sidecar promotion failed and prior artifacts were restored: {exc}",
                None,
                run_log if run_log.exists() else None,
                None,
                None,
                None,
                validation,
                quarantine,
            )

        self._emit_progress(progress_callback, "job_complete")
        return ProcessingResult(
            True,
            False,
            "Completed and validated.",
            output,
            plan.log_path,
            plan.report_path,
            plan.script_path if plan.vapoursynth_script else None,
            final_hash,
            validation,
            None,
        )


class _BinaryLineDecoder:
    """Expose a binary pipe as the small text interface used by the log drainer."""

    def __init__(self, stream) -> None:
        self.stream = stream

    def readline(self) -> str:
        data = self.stream.readline()
        return data.decode("utf-8", errors="replace") if data else ""

    def close(self) -> None:
        self.stream.close()


def _sidecar_for_output(output: Path, suffix: str) -> Path:
    return output.with_name(output.name + suffix)
