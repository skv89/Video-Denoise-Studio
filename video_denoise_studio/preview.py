from __future__ import annotations

import os
import queue
import shutil
import subprocess
import tempfile
import threading
import uuid
from pathlib import Path

from deinterlace_studio.denoise import (
    denoiser_backend_display,
    denoiser_is_vapoursynth,
    ffmpeg_denoise_filter,
    resolve_denoiser_backend,
    vapoursynth_import_lines,
)
from deinterlace_studio.dependencies import managed_runtime_environment
from deinterlace_studio.scheduling import choose_vapoursynth_schedule

from . import __version__
from .denoiser_policy import denoiser_control_policy, validate_denoiser_controls
from .models import PreviewFrames, PreviewRequest, ProgressCallback
from .planner import execution_vspipe_path, source_field_order
from .timeline import source_fps, source_frame_count
from .vapoursynth_fields import video_denoise_lines


CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0
CREATE_NEW_PROCESS_GROUP = 0x00000200 if os.name == "nt" else 0


class PreviewError(RuntimeError):
    pass


class PreviewCancelled(PreviewError):
    pass


def _native_preview_filter() -> str:
    """Keep source raster intact; the viewer alone performs display scaling."""

    return "format=rgb24"


def _source_fps(request: PreviewRequest) -> float:
    return source_fps(request.media)


def _source_frame_count(request: PreviewRequest, fps: float) -> int | None:
    del fps
    return source_frame_count(request.media)


class PreviewRenderer:
    """Render one source frame and, optionally, its exact temporal-denoise result."""

    def __init__(self) -> None:
        self.cancel_event = threading.Event()
        self._processes: set[subprocess.Popen] = set()
        self._lock = threading.Lock()
        self._owned_directories: set[Path] = set()

    def cancel(self) -> None:
        self.cancel_event.set()
        with self._lock:
            processes = tuple(self._processes)
        for process in processes:
            self._terminate(process)

    def close(self) -> None:
        self.cancel()
        for directory in tuple(self._owned_directories):
            self.cleanup_directory(directory)

    def cleanup(self, frames: PreviewFrames | None) -> None:
        if frames:
            self.cleanup_directory(frames.directory)

    def cleanup_directory(self, directory: Path) -> None:
        try:
            shutil.rmtree(directory)
        except FileNotFoundError:
            pass
        except OSError:
            return
        self._owned_directories.discard(directory)

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
            process.wait(timeout=2)
        except (OSError, subprocess.TimeoutExpired):
            try:
                process.kill()
                process.wait(timeout=2)
            except (OSError, subprocess.TimeoutExpired):
                pass

    def _check_canceled(self) -> None:
        if self.cancel_event.is_set():
            raise PreviewCancelled("Preview rendering canceled.")

    @staticmethod
    def _emit(callback: ProgressCallback | None, phase: str, **values: object) -> None:
        if callback:
            payload = {"phase": phase}
            payload.update({key: str(value) for key, value in values.items() if value is not None})
            callback(payload)

    @staticmethod
    def _context(request: PreviewRequest, fps: float) -> tuple[int, int, int, int, int, int | None]:
        total = _source_frame_count(request, fps)
        target = max(0, request.target_frame)
        if total is not None:
            target = min(target, max(0, total - 1))
        policy = denoiser_control_policy(request.denoiser, request.strength, request.temporal_radius)
        radius = (policy.window_frames - 1) // 2
        leading = min(radius, target)
        trailing = radius if total is None else min(radius, max(0, total - target - 1))
        context_start = target - leading
        context_count = leading + 1 + trailing
        return target, context_start, context_count, leading, trailing, total

    @staticmethod
    def _verify_preview_storage(request: PreviewRequest) -> None:
        """Refuse a preview that could consume the remaining temporary drive."""

        stream_count = 2 if request.include_processed else 1
        # PNGs are normally much smaller than RGB24, but RGB24 plus a fixed
        # working reserve is a deliberately conservative upper estimate.
        frame_bytes = request.width * request.height * 3
        required = frame_bytes * stream_count + (256 * 1024 * 1024)
        temporary_root = Path(tempfile.gettempdir())
        try:
            available = shutil.disk_usage(temporary_root).free
        except OSError as exc:
            raise PreviewError(f"Could not inspect free space for preview storage: {exc}") from exc
        if available < required:
            required_gib = required / (1024**3)
            available_gib = available / (1024**3)
            raise PreviewError(
                "Not enough temporary-drive space for this preview: "
                f"approximately {required_gib:.2f} GiB required, {available_gib:.2f} GiB available. "
                "Close other applications or free temporary-drive space."
            )

    def _original_command(
        self,
        request: PreviewRequest,
        directory: Path,
        target: int,
        fps: float,
    ) -> list[str]:
        ffmpeg = request.capabilities.ffmpeg_path
        if not ffmpeg:
            raise PreviewError("FFmpeg is unavailable for source preview rendering.")
        original = directory / "original" / "%06d.png"
        return [
            str(ffmpeg),
            "-hide_banner",
            "-nostdin",
            "-y",
            "-progress",
            "pipe:1",
            "-nostats",
            "-ss",
            f"{target / fps:.9f}",
            "-i",
            str(request.source),
            "-an",
            "-sn",
            "-dn",
            "-vf",
            _native_preview_filter(),
            "-frames:v",
            "1",
            "-fps_mode",
            "passthrough",
            "-start_number",
            "1",
            str(original),
        ]

    def _ffmpeg_comparison_command(
        self,
        request: PreviewRequest,
        directory: Path,
        context_start: int,
        context_count: int,
        leading: int,
        fps: float,
    ) -> list[str]:
        ffmpeg = request.capabilities.ffmpeg_path
        if not ffmpeg:
            raise PreviewError("FFmpeg is unavailable for comparison preview rendering.")
        original = directory / "original" / "%06d.png"
        processed = directory / "processed" / "%06d.png"
        denoise = ffmpeg_denoise_filter(request.denoiser, request.strength, request.temporal_radius)
        native = _native_preview_filter()
        graph = (
            f"[0:v:0]trim=start_frame=0:end_frame={context_count},"
            "setpts=PTS-STARTPTS,split=2[base][work];"
            f"[base]trim=start_frame={leading}:end_frame={leading + 1},setpts=PTS-STARTPTS,{native}[original];"
            f"[work]{denoise},trim=start_frame={leading}:end_frame={leading + 1},"
            f"setpts=PTS-STARTPTS,{native}[processed]"
        )
        return [
            str(ffmpeg),
            "-hide_banner",
            "-nostdin",
            "-y",
            "-progress",
            "pipe:1",
            "-nostats",
            "-ss",
            f"{context_start / fps:.9f}",
            "-i",
            str(request.source),
            "-filter_complex",
            graph,
            "-map",
            "[original]",
            "-fps_mode",
            "passthrough",
            "-start_number",
            "1",
            str(original),
            "-map",
            "[processed]",
            "-fps_mode",
            "passthrough",
            "-start_number",
            "1",
            str(processed),
        ]

    def _vspipe_comparison(
        self,
        request: PreviewRequest,
        directory: Path,
        context_start: int,
        context_count: int,
        leading: int,
        backend: str,
    ) -> tuple[list[str], list[str], Path]:
        ffmpeg = request.capabilities.ffmpeg_path
        vspipe = request.capabilities.vspipe_path
        if not ffmpeg or not vspipe:
            raise PreviewError("FFmpeg and VSPipe are required for this preview.")
        schedule = choose_vapoursynth_schedule(
            request.media.video.width or request.width,
            request.media.video.height or request.height,
            request.media.video.pix_fmt,
            temporal_denoise=True,
        )
        imports = [
            "import os",
            "import tempfile",
            "import vapoursynth as vs",
            "from vapoursynth import core",
            "from vstools import depth",
            *vapoursynth_import_lines(request.denoiser),
        ]
        lines = [
            f"# Generated by Video Denoise Studio {__version__} for an aligned temporal preview.",
            *imports,
            "",
            f"core.num_threads = min({schedule.core_threads}, max(1, core.num_threads))",
            f"SOURCE = {str(request.source)!r}",
            "CACHE_ROOT = os.path.join(tempfile.gettempdir(), 'Video Denoise Studio BestSource Cache')",
            "os.makedirs(CACHE_ROOT, exist_ok=True)",
            "source = core.bs.VideoSource(source=SOURCE, cachemode=1, cachepath=CACHE_ROOT)",
            "if source.format is None:",
            "    raise RuntimeError('BestSource returned a variable-format clip; normalize the source before previewing')",
            f"source = source[{context_start}:{context_start + context_count}]",
            "source = depth(source, 16)",
        ]
        field_order = source_field_order(request.media)
        if field_order:
            lines.append(f"source = core.std.SetFieldBased(source, value={2 if field_order == 'tff' else 1})")
        lines += [
            "processed = source",
            *video_denoise_lines(
                request.denoiser,
                request.strength,
                request.temporal_radius,
                backend,
                clip_variable="processed",
                field_order=field_order,
            ),
            f"source = source[{leading}:{leading + 1}]",
            f"processed = processed[{leading}:{leading + 1}]",
            "comparison = core.std.StackHorizontal([source, processed])",
            "comparison.set_output()",
            "",
        ]
        script = directory / "preview.vpy"
        script.write_text("\n".join(lines), encoding="utf-8")
        requests = min(4, schedule.requests)
        vspipe_command = [
            str(execution_vspipe_path(vspipe)),
            "--requests",
            str(requests),
            "--container",
            "y4m",
            "--progress",
            str(script),
            "-",
        ]
        original = directory / "original" / "%06d.png"
        processed = directory / "processed" / "%06d.png"
        native = _native_preview_filter()
        graph = (
            "[0:v:0]split=2[left][right];"
            f"[left]crop=w=iw/2:h=ih:x=0:y=0,{native}[original];"
            f"[right]crop=w=iw/2:h=ih:x=iw/2:y=0,{native}[processed]"
        )
        ffmpeg_command = [
            str(ffmpeg),
            "-hide_banner",
            "-nostdin",
            "-y",
            "-progress",
            "pipe:1",
            "-nostats",
            "-f",
            "yuv4mpegpipe",
            "-i",
            "pipe:0",
            "-filter_complex",
            graph,
            "-map",
            "[original]",
            "-fps_mode",
            "passthrough",
            "-start_number",
            "1",
            str(original),
            "-map",
            "[processed]",
            "-fps_mode",
            "passthrough",
            "-start_number",
            "1",
            str(processed),
        ]
        return vspipe_command, ffmpeg_command, script

    def _run(
        self,
        ffmpeg_command: list[str],
        progress_callback: ProgressCallback | None,
        *,
        vspipe_command: list[str] | None = None,
    ) -> None:
        vspipe_process: subprocess.Popen | None = None
        ffmpeg_process: subprocess.Popen | None = None
        stderr_lines: queue.Queue[str] = queue.Queue()
        drain_threads: list[threading.Thread] = []

        def drain(stream, label: str, binary: bool = False) -> None:
            for line in stream:
                if binary:
                    line = line.decode("utf-8", errors="replace")
                rendered = str(line).strip()
                if rendered:
                    stderr_lines.put(f"{label}: {rendered}")
            stream.close()

        creationflags = CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP
        try:
            ffmpeg_stdin = subprocess.DEVNULL
            if vspipe_command:
                env = managed_runtime_environment(vspipe_command[0], os.environ.copy())
                vspipe_process = subprocess.Popen(
                    vspipe_command,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    creationflags=creationflags,
                    env=env,
                )
                self._register(vspipe_process)
                assert vspipe_process.stdout is not None and vspipe_process.stderr is not None
                thread = threading.Thread(target=drain, args=(vspipe_process.stderr, "VSPipe", True), daemon=True)
                thread.start()
                drain_threads.append(thread)
                ffmpeg_stdin = vspipe_process.stdout

            ffmpeg_process = subprocess.Popen(
                ffmpeg_command,
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
            assert ffmpeg_process.stdout is not None and ffmpeg_process.stderr is not None
            thread = threading.Thread(target=drain, args=(ffmpeg_process.stderr, "FFmpeg"), daemon=True)
            thread.start()
            drain_threads.append(thread)
            progress: dict[str, str] = {}
            while True:
                self._check_canceled()
                line = ffmpeg_process.stdout.readline()
                if not line:
                    if ffmpeg_process.poll() is not None:
                        break
                    self.cancel_event.wait(0.02)
                    continue
                rendered = line.strip()
                if "=" not in rendered:
                    continue
                key, value = rendered.split("=", 1)
                progress[key] = value
                if key == "progress":
                    payload = dict(progress)
                    payload["phase"] = "preview_progress"
                    if progress_callback:
                        progress_callback(payload)
                    progress.clear()
            ffmpeg_code = ffmpeg_process.wait()
            if vspipe_process:
                try:
                    vspipe_code = vspipe_process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    self._terminate(vspipe_process)
                    vspipe_code = vspipe_process.returncode
            else:
                vspipe_code = 0
            for thread in drain_threads:
                thread.join(timeout=2)
            diagnostics: list[str] = []
            while not stderr_lines.empty():
                diagnostics.append(stderr_lines.get())
            tail = "\n".join(diagnostics[-20:])
            if ffmpeg_code != 0:
                raise PreviewError(f"FFmpeg preview render exited with code {ffmpeg_code}.\n{tail}")
            if vspipe_code not in {0, None}:
                raise PreviewError(f"VSPipe preview render exited with code {vspipe_code}.\n{tail}")
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

    def render(
        self,
        request: PreviewRequest,
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> PreviewFrames:
        self.cancel_event.clear()
        if request.target_frame < 0:
            raise PreviewError("Timeline frame must be zero or greater.")
        if request.width < 64 or request.height < 64:
            raise PreviewError("Preview raster is too small.")
        if not request.source.is_file():
            raise PreviewError(f"Preview source does not exist: {request.source}")
        if request.include_processed:
            number_errors = validate_denoiser_controls(request.denoiser, request.strength, request.temporal_radius)
            if number_errors:
                raise PreviewError(" ".join(number_errors))
            if not request.capabilities.denoise_capabilities.get(request.denoiser, False):
                raise PreviewError(
                    request.capabilities.denoise_diagnostics.get(request.denoiser, "Selected denoiser is unavailable.")
                )

        self._verify_preview_storage(request)

        token = uuid.uuid4().hex[:12]
        directory = Path(tempfile.mkdtemp(prefix=f"VideoDenoiseStudio-preview-{token}-"))
        self._owned_directories.add(directory)
        (directory / "original").mkdir()
        (directory / "processed").mkdir()
        fps = _source_fps(request)
        total = _source_frame_count(request, fps)
        target = request.target_frame if total is None else min(request.target_frame, max(0, total - 1))
        self._emit(progress_callback, "preview_start", frame=target)
        try:
            if request.include_processed:
                target, context_start, context_count, leading, trailing, total = self._context(request, fps)
                backend = resolve_denoiser_backend(
                    request.denoiser,
                    request.capabilities.denoise_backends.get(request.denoiser),
                    request.media.video.width,
                    request.media.video.height,
                )
                if not backend:
                    raise PreviewError("The selected denoiser has no resolved backend.")
                if denoiser_is_vapoursynth(request.denoiser):
                    vspipe, ffmpeg, _script = self._vspipe_comparison(
                        request,
                        directory,
                        context_start,
                        context_count,
                        leading,
                        backend,
                    )
                    self._run(ffmpeg, progress_callback, vspipe_command=vspipe)
                else:
                    command = self._ffmpeg_comparison_command(
                        request,
                        directory,
                        context_start,
                        context_count,
                        leading,
                        fps,
                    )
                    self._run(command, progress_callback)
                processed = tuple(sorted((directory / "processed").glob("*.png")))
            else:
                command = self._original_command(request, directory, target, fps)
                self._run(command, progress_callback)
                leading = trailing = 0
                backend = None
                processed = ()

            original = tuple(sorted((directory / "original").glob("*.png")))
            if not original:
                raise PreviewError("Preview rendering produced no source frames.")
            if len(original) != 1:
                raise PreviewError(f"Source preview produced {len(original)} frames; exactly one was required.")
            if request.include_processed and len(processed) != 1:
                raise PreviewError(
                    f"Denoised preview produced {len(processed)} frames; exactly one aligned frame was required."
                )
            policy = denoiser_control_policy(request.denoiser, request.strength, request.temporal_radius)
            field_note = (
                f" · interlaced {source_field_order(request.media).upper()} preserved by independent field-parity denoising and reweave"
                if request.include_processed
                and request.denoiser == "vs_dfttest"
                and source_field_order(request.media)
                else ""
            )
            status = (
                f"Frame {target + 1} denoised with Strength {request.strength}/10 · "
                f"Radius {policy.normalized_radius} · nominal {policy.window_frames}-frame window · "
                f"used {leading + 1 + trailing} real frame(s) ({leading} before + target + {trailing} after) · "
                f"backend: {denoiser_backend_display(request.denoiser, backend)}{field_note}. Hold left for Original."
                if request.include_processed
                else f"Frame {target + 1} source preview ready; Frame preview is off and no denoiser ran."
            )
            self._emit(progress_callback, "preview_complete", frame=target)
            return PreviewFrames(
                token=token,
                directory=directory,
                original_frame=original[0],
                processed_frame=processed[0] if processed else None,
                target_frame=target,
                total_frames=total,
                strength=request.strength,
                temporal_radius=request.temporal_radius,
                window_frames=policy.window_frames if request.include_processed else 1,
                leading_context=leading,
                trailing_context=trailing,
                fps=fps,
                selected_backend=backend,
                status=status,
            )
        except Exception:
            self.cleanup_directory(directory)
            raise
