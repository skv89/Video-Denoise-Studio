from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TextIO

from . import __version__
from .dependencies import managed_runtime_environment
from .probe import count_video_packets, probe_media


CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0
CREATE_NEW_PROCESS_GROUP = 0x00000200 if os.name == "nt" else 0
PROCESS_FLAGS = CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP

MOV_AUDIO_CODECS = {
    "aac",
    "alac",
    "ac3",
    "eac3",
    "mp2",
    "mp3",
    "dts",
    "pcm_s16le",
    "pcm_s24le",
    "pcm_s32le",
    "pcm_f32le",
    "pcm_f64le",
}
MOV_TEXT_SUBTITLE_CODECS = {"ass", "ssa", "subrip", "srt", "text", "webvtt", "mov_text"}
MOV_COMPATIBLE_VIDEO_CODECS = {"prores", "dnxhd"}

LogCallback = Callable[[str], None]
ProgressCallback = Callable[[dict[str, str]], None]


class CompatibilityCopyError(RuntimeError):
    pass


class CompatibilityCopyCancelled(RuntimeError):
    pass


@dataclass(frozen=True)
class CompatibilityCopyRequest:
    source_path: Path
    output_path: Path


@dataclass(frozen=True)
class CompatibilityCopyResult:
    success: bool
    canceled: bool
    message: str
    output_path: Path | None
    log_path: Path | None
    report_path: Path | None
    video_essence_sha256: str | None = None
    source_video_packets: int | None = None
    output_video_packets: int | None = None
    copied_audio_tracks: int = 0
    converted_subtitle_tracks: int = 0
    omitted_tracks: tuple[str, ...] = ()


def _same_path(left: Path, right: Path) -> bool:
    return os.path.normcase(os.path.abspath(left)) == os.path.normcase(os.path.abspath(right))


def _promotion_identity(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_dev, stat.st_ino


def _safe_unlink(path: Path | None) -> None:
    if path is None:
        return
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


class MOVCompatibilityCopier:
    """Stream-copy a completed ProRes/DNxHR MKV into native MOV safely.

    Video and compatible audio are never re-encoded. Text subtitles are changed
    to MOV's native ``mov_text`` representation; attachments, data streams, and
    unsupported subtitle types stay in the unchanged MKV and are reported.
    """

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

    @staticmethod
    def _emit_progress(callback: ProgressCallback | None, phase: str, **values: object) -> None:
        if callback is None:
            return
        payload = {"phase": phase}
        payload.update({key: str(value) for key, value in values.items() if value is not None})
        callback(payload)

    def _run_capture(
        self,
        command: list[str],
        write_log: LogCallback,
        *,
        label: str,
        timeout: float | None = None,
    ) -> tuple[str, str]:
        if self.cancel_event.is_set():
            raise CompatibilityCopyCancelled("Compatibility copy canceled")
        write_log(f"{label} command: {subprocess.list2cmdline(command)}")
        env = managed_runtime_environment(command[0], os.environ.copy())
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
        started = time.monotonic()
        try:
            while True:
                if self.cancel_event.is_set():
                    raise CompatibilityCopyCancelled("Compatibility copy canceled")
                if timeout is not None and time.monotonic() - started > timeout:
                    raise CompatibilityCopyError(f"{label} exceeded its {timeout:.0f}-second safety limit")
                try:
                    stdout, stderr = process.communicate(timeout=0.20)
                    break
                except subprocess.TimeoutExpired:
                    continue
            if self.cancel_event.is_set():
                raise CompatibilityCopyCancelled("Compatibility copy canceled")
            for line in stderr.splitlines():
                write_log(f"[{label}] {line}")
            if process.returncode != 0:
                tail = (stderr or stdout).strip().splitlines()
                raise CompatibilityCopyError(
                    f"{label} failed with code {process.returncode}: "
                    f"{tail[-1] if tail else 'no diagnostic was returned'}"
                )
            return stdout, stderr
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
            for stream in (process.stdout, process.stderr):
                try:
                    if stream:
                        stream.close()
                except OSError:
                    pass
            self._unregister(process)

    def _run_progress_command(
        self,
        command: list[str],
        write_log: LogCallback,
        progress_callback: ProgressCallback | None,
        *,
        phase: str,
        duration: float | None,
    ) -> None:
        if self.cancel_event.is_set():
            raise CompatibilityCopyCancelled("Compatibility copy canceled")
        write_log(f"{phase} command: {subprocess.list2cmdline(command)}")
        env = managed_runtime_environment(command[0], os.environ.copy())
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=PROCESS_FLAGS,
            env=env,
        )
        self._register(process)
        lines: queue.Queue[tuple[str, str | None]] = queue.Queue()

        def reader(stream: TextIO, name: str) -> None:
            try:
                for line in iter(stream.readline, ""):
                    lines.put((name, line.rstrip("\r\n")))
            finally:
                lines.put((name, None))

        assert process.stdout and process.stderr
        threading.Thread(target=reader, args=(process.stdout, "stdout"), daemon=True).start()
        threading.Thread(target=reader, args=(process.stderr, "stderr"), daemon=True).start()
        values: dict[str, str] = {}
        closed: set[str] = set()
        try:
            while process.poll() is None or len(closed) < 2 or not lines.empty():
                if self.cancel_event.is_set():
                    raise CompatibilityCopyCancelled("Compatibility copy canceled")
                try:
                    name, line = lines.get(timeout=0.10)
                except queue.Empty:
                    continue
                if line is None:
                    closed.add(name)
                    continue
                if name == "stderr":
                    write_log(f"[{phase}] {line}")
                    continue
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                values[key] = value
                if key == "progress":
                    payload = dict(values)
                    payload["phase"] = phase
                    if duration and duration > 0:
                        payload["duration_us"] = str(round(duration * 1_000_000))
                    if progress_callback:
                        progress_callback(payload)
                    values.clear()
            return_code = process.wait()
            if self.cancel_event.is_set():
                raise CompatibilityCopyCancelled("Compatibility copy canceled")
            if return_code != 0:
                raise CompatibilityCopyError(f"{phase} failed with FFmpeg exit code {return_code}; see the retained log")
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
            for stream in (process.stdout, process.stderr):
                try:
                    if stream:
                        stream.close()
                except OSError:
                    pass
            self._unregister(process)

    def _video_essence_hash(self, ffmpeg: Path, path: Path, write_log: LogCallback, label: str) -> str:
        stdout, _stderr = self._run_capture(
            [
                str(ffmpeg),
                "-hide_banner",
                "-nostdin",
                "-loglevel",
                "error",
                "-i",
                str(path),
                "-map",
                "0:v:0",
                "-c:v",
                "copy",
                "-f",
                "hash",
                "-hash",
                "sha256",
                "-",
            ],
            write_log,
            label=label,
        )
        for line in stdout.splitlines():
            if line.upper().startswith("SHA256="):
                return line.split("=", 1)[1].strip().upper()
        raise CompatibilityCopyError(f"{label} did not return a video-essence SHA-256")

    def run(
        self,
        request: CompatibilityCopyRequest,
        ffmpeg: Path,
        ffprobe: Path,
        *,
        log_callback: LogCallback | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> CompatibilityCopyResult:
        source = request.source_path.resolve()
        output = request.output_path.resolve()
        token = uuid.uuid4().hex[:12]
        partial = output.with_name(f".{output.stem}.Compatibility.partial.{token}.mov")
        log_path = output.with_name(output.name + ".Compatibility.log")
        report_path = output.with_name(output.name + ".Compatibility.json")
        failed_report = output.with_name(output.name + f".Compatibility.failed.{token}.json")
        log_handle: TextIO | None = None
        promoted = False
        report_payload: dict[str, object] = {
            "application": f"Deinterlace Studio {__version__}",
            "operation": "fast_mov_compatibility_copy",
            "source": str(source),
            "requested_output": str(output),
            "video_reencoded": False,
            "audio_reencoded": False,
            "subtitle_conversion": "supported text subtitles convert to mov_text",
        }

        def write_log(line: str) -> None:
            timestamped = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {line}"
            if log_handle:
                log_handle.write(timestamped + "\n")
                log_handle.flush()
            if log_callback:
                log_callback(timestamped)

        try:
            if not source.is_file():
                raise CompatibilityCopyError(f"Source file does not exist: {source}")
            if output.suffix.casefold() != ".mov":
                raise CompatibilityCopyError("Compatibility output must use the .mov extension")
            if _same_path(source, output):
                raise CompatibilityCopyError("The compatibility output resolves to the source; source media is never overwritten")
            if not output.parent.is_dir():
                raise CompatibilityCopyError(f"Output directory does not exist: {output.parent}")
            collisions = [path for path in (output, log_path, report_path) if path.exists()]
            if collisions:
                raise CompatibilityCopyError(
                    "Choose a new output name; compatibility output/sidecar files already exist: "
                    + ", ".join(str(path) for path in collisions)
                )

            log_handle = log_path.open("x", encoding="utf-8", newline="\n")
            write_log("Fast MOV compatibility copy started; the source will remain unchanged.")
            self._emit_progress(progress_callback, "compatibility_probe", percent=0)
            source_media = probe_media(ffprobe, source, sample_frames=32, timeout=180)
            if "matroska" not in source_media.format_name.casefold():
                raise CompatibilityCopyError(
                    f"This tool expects a completed Matroska/MKV master; source format is {source_media.format_name}."
                )
            if source_media.video.codec_name.casefold() not in MOV_COMPATIBLE_VIDEO_CODECS:
                raise CompatibilityCopyError(
                    "Fast MOV compatibility copy is limited to ProRes or DNxHR video; "
                    f"source video codec is {source_media.video.codec_name}."
                )

            incompatible_audio = [
                stream.codec_name
                for stream in source_media.streams_of_type("audio")
                if stream.codec_name.casefold() not in MOV_AUDIO_CODECS
            ]
            if incompatible_audio:
                raise CompatibilityCopyError(
                    "MOV cannot safely stream-copy every audio track ("
                    + ", ".join(sorted(set(incompatible_audio)))
                    + "). No file was created; retain MKV or create a deliberate audio conversion separately."
                )
            text_subtitles = [
                stream
                for stream in source_media.streams_of_type("subtitle")
                if stream.codec_name.casefold() in MOV_TEXT_SUBTITLE_CODECS
            ]
            unsupported_subtitles = [
                stream
                for stream in source_media.streams_of_type("subtitle")
                if stream.codec_name.casefold() not in MOV_TEXT_SUBTITLE_CODECS
            ]
            omitted_tracks: list[str] = []
            if unsupported_subtitles:
                omitted_tracks.append(
                    f"{len(unsupported_subtitles)} unsupported subtitle track(s): "
                    + ", ".join(sorted({stream.codec_name for stream in unsupported_subtitles}))
                )
            if source_media.attachment_count:
                omitted_tracks.append(f"{source_media.attachment_count} attachment(s) retained only in the original MKV")
            if source_media.data_count:
                omitted_tracks.append(f"{source_media.data_count} data stream(s) retained only in the original MKV")

            map_args = ["-map", "0:v:0"]
            for stream in source_media.streams_of_type("audio"):
                map_args += ["-map", f"0:{stream.index}"]
            for stream in text_subtitles:
                map_args += ["-map", f"0:{stream.index}"]

            command = [
                str(ffmpeg),
                "-hide_banner",
                "-nostdin",
                "-n",
                "-i",
                str(source),
                *map_args,
                "-map_metadata",
                "0",
                "-map_chapters",
                "0",
                "-c:v",
                "copy",
                "-c:a",
                "copy",
            ]
            if text_subtitles:
                command += ["-c:s", "mov_text"]
            command += [
                "-movflags",
                "+faststart",
                "-max_muxing_queue_size",
                "4096",
                "-progress",
                "pipe:1",
                "-nostats",
                str(partial),
            ]
            self._emit_progress(progress_callback, "compatibility_remux", percent=0)
            self._run_progress_command(
                command,
                write_log,
                progress_callback,
                phase="compatibility_remux",
                duration=source_media.duration,
            )
            if not partial.is_file() or partial.stat().st_size <= 0:
                raise CompatibilityCopyError("FFmpeg returned success but the MOV partial is missing or empty")

            self._emit_progress(progress_callback, "compatibility_validate", percent=0)
            output_media = probe_media(ffprobe, partial, sample_frames=32, timeout=180)
            errors: list[str] = []
            source_video = source_media.video
            output_video = output_media.video
            if output_video.codec_name != source_video.codec_name:
                errors.append(f"video codec changed from {source_video.codec_name} to {output_video.codec_name}")
            for label, expected, actual in (
                ("pixel format", source_video.pix_fmt, output_video.pix_fmt),
                ("width", source_video.width, output_video.width),
                ("height", source_video.height, output_video.height),
                ("sample aspect ratio", source_video.sample_aspect_ratio, output_video.sample_aspect_ratio),
                ("nominal frame rate", source_video.r_frame_rate, output_video.r_frame_rate),
                ("average frame rate", source_video.avg_frame_rate, output_video.avg_frame_rate),
            ):
                if expected is not None and actual != expected:
                    errors.append(f"{label} changed from {expected} to {actual}")
            if output_media.audio_count != source_media.audio_count:
                errors.append(
                    f"audio track count changed from {source_media.audio_count} to {output_media.audio_count}"
                )
            if output_media.subtitle_count != len(text_subtitles):
                errors.append(
                    f"converted subtitle count is {output_media.subtitle_count}, expected {len(text_subtitles)}"
                )
            if len(output_media.chapters) != len(source_media.chapters):
                errors.append(
                    f"chapter count changed from {len(source_media.chapters)} to {len(output_media.chapters)}"
                )

            source_packets = count_video_packets(ffprobe, source, timeout=900)
            output_packets = count_video_packets(ffprobe, partial, timeout=900)
            if source_packets is None or output_packets is None or source_packets != output_packets:
                errors.append(f"video packet count changed from {source_packets} to {output_packets}")
            if errors:
                raise CompatibilityCopyError("MOV structural validation failed: " + "; ".join(errors))

            self._emit_progress(progress_callback, "compatibility_hash", percent=55)
            source_essence_hash = self._video_essence_hash(ffmpeg, source, write_log, "source video essence hash")
            output_essence_hash = self._video_essence_hash(ffmpeg, partial, write_log, "MOV video essence hash")
            if output_essence_hash != source_essence_hash:
                raise CompatibilityCopyError(
                    "Video packet essence changed despite stream-copy; the partial was rejected"
                )

            decode_command = [
                str(ffmpeg),
                "-hide_banner",
                "-nostdin",
                "-v",
                "error",
                "-xerror",
                "-progress",
                "pipe:1",
                "-nostats",
                "-i",
                str(partial),
                "-map",
                "0:v:0",
                "-f",
                "null",
                "-",
            ]
            self._emit_progress(progress_callback, "compatibility_full_decode", percent=65)
            self._run_progress_command(
                decode_command,
                write_log,
                progress_callback,
                phase="compatibility_full_decode",
                duration=source_media.duration,
            )

            self._emit_progress(progress_callback, "compatibility_promote", percent=98)
            partial_identity = _promotion_identity(partial)
            if output.exists():
                raise CompatibilityCopyError(
                    "The requested output appeared while validation was running; it was not overwritten"
                )
            os.rename(partial, output)
            promoted = True
            if _promotion_identity(output) != partial_identity:
                raise CompatibilityCopyError("Atomic promotion did not retain the exact validated MOV file object")
            final_media = probe_media(ffprobe, output, sample_frames=8, timeout=180)
            if final_media.video.codec_name != source_video.codec_name:
                raise CompatibilityCopyError("Final-path reopen no longer reports the validated video codec")

            report_payload.update(
                {
                    "status": "validated",
                    "output": str(output),
                    "output_size": output.stat().st_size,
                    "video_essence_sha256": source_essence_hash,
                    "source_video_packets": source_packets,
                    "output_video_packets": output_packets,
                    "copied_audio_tracks": source_media.audio_count,
                    "converted_subtitle_tracks": len(text_subtitles),
                    "omitted_tracks": omitted_tracks,
                    "validation": {
                        "video_reencoded": False,
                        "audio_reencoded": False,
                        "video_packet_essence_equal": True,
                        "strict_full_decode": "pass",
                        "same_file_atomic_promotion": "pass",
                    },
                }
            )
            report_path.write_text(
                json.dumps(report_payload, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
            self._emit_progress(progress_callback, "compatibility_complete", percent=100)
            message = (
                "Validated native MOV compatibility copy created without re-encoding video or audio."
                + (f" {len(text_subtitles)} text subtitle track(s) were converted to mov_text." if text_subtitles else "")
                + (" Omitted tracks remain in the unchanged MKV: " + "; ".join(omitted_tracks) if omitted_tracks else "")
            )
            write_log(message)
            return CompatibilityCopyResult(
                True,
                False,
                message,
                output,
                log_path,
                report_path,
                source_essence_hash,
                source_packets,
                output_packets,
                source_media.audio_count,
                len(text_subtitles),
                tuple(omitted_tracks),
            )
        except CompatibilityCopyCancelled as exc:
            _safe_unlink(partial)
            if promoted:
                _safe_unlink(output)
            report_payload.update({"status": "canceled", "message": str(exc)})
            failed_report.write_text(json.dumps(report_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            return CompatibilityCopyResult(False, True, str(exc), None, log_path if log_path.exists() else None, failed_report)
        except Exception as exc:
            _safe_unlink(partial)
            if promoted:
                _safe_unlink(output)
            report_payload.update({"status": "failed", "message": str(exc)})
            failed_report.write_text(json.dumps(report_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            if log_handle:
                write_log(f"Compatibility copy failed: {exc}")
            return CompatibilityCopyResult(False, False, str(exc), None, log_path if log_path.exists() else None, failed_report)
        finally:
            if log_handle:
                log_handle.close()
