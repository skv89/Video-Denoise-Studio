from __future__ import annotations

import json
import os
import subprocess
import threading
from pathlib import Path

from deinterlace_studio.models import MediaProbe
from deinterlace_studio.probe import ProbeError, parse_probe_json


CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0
CREATE_NEW_PROCESS_GROUP = 0x00000200 if os.name == "nt" else 0


class ProbeCancelled(ProbeError):
    pass


def _run_json(args: list[str], cancel_event: threading.Event, timeout: float) -> dict[str, object]:
    env = os.environ.copy()
    env["AV_LOG_FORCE_NOCOLOR"] = "1"
    try:
        process = subprocess.Popen(
            args,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP,
            env=env,
        )
    except OSError as exc:
        raise ProbeError(f"Could not run ffprobe: {exc}") from exc
    elapsed = 0.0
    while True:
        if cancel_event.is_set():
            try:
                process.terminate()
                process.wait(timeout=2)
            except (OSError, subprocess.TimeoutExpired):
                process.kill()
                process.wait(timeout=2)
            raise ProbeCancelled("Media probe canceled.")
        try:
            stdout, stderr = process.communicate(timeout=0.1)
            break
        except subprocess.TimeoutExpired:
            elapsed += 0.1
            if elapsed >= timeout:
                process.kill()
                process.wait(timeout=2)
                raise ProbeError(f"ffprobe exceeded the {timeout:.0f}-second timeout.")
    if process.returncode != 0:
        raise ProbeError(stderr.strip() or f"ffprobe exited with {process.returncode}")
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise ProbeError("ffprobe returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise ProbeError("ffprobe returned an unexpected JSON value")
    return payload


def probe_media_cancelable(
    ffprobe: Path,
    path: Path,
    cancel_event: threading.Event,
    *,
    sample_frames: int = 64,
    timeout: float = 90.0,
) -> MediaProbe:
    if not path.is_file():
        raise ProbeError(f"Input file does not exist: {path}")
    main = _run_json(
        [
            str(ffprobe),
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-show_chapters",
            "-of",
            "json",
            str(path),
        ],
        cancel_event,
        timeout,
    )
    frame_flags: list[dict[str, object]] = []
    if sample_frames > 0:
        frames = _run_json(
            [
                str(ffprobe),
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-read_intervals",
                f"%+#{sample_frames}",
                "-show_entries",
                "frame=interlaced_frame,top_field_first",
                "-of",
                "json",
                str(path),
            ],
            cancel_event,
            timeout,
        )
        raw_frames = frames.get("frames", [])
        if isinstance(raw_frames, list):
            frame_flags = [item for item in raw_frames if isinstance(item, dict)]
    return parse_probe_json(path, main, frame_flags)

