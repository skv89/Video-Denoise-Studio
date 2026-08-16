from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


DEFAULTS: dict[str, Any] = {
    "settings_schema_version": 2,
    "ffmpeg_path": "",
    "ffprobe_path": "",
    "vspipe_path": "",
    "last_input_dir": "",
    "last_output_dir": "",
    "window_geometry": "1460x960",
    "denoiser": "vs_bm3d",
    "denoise_strength": 4,
    "denoise_temporal_radius": 3,
    "frame_preview_enabled": False,
    "family": "ffv1",
    "container": "auto",
    "bit_depth": 16,
    "ffv1_chroma_mode": "native",
    "hardware_encode": False,
    "av1_software_encoder": "libaom",
    "quality": 14,
    "tune_grain": True,
    "copy_audio": True,
    "copy_subtitles": True,
    "copy_attachments": True,
    "copy_data": False,
    "copy_chapters": True,
    "copy_metadata": True,
    "batch_output_dir": "",
    "batch_include_subfolders": False,
    "batch_continue_after_error": True,
}

REFERENCE_TOOL_KEYS = ("ffmpeg_path", "ffprobe_path", "vspipe_path")


def default_settings_path() -> Path:
    base = Path(os.environ.get("APPDATA") or Path.home())
    return base / "VideoDenoiseStudio" / "settings.json"


def reference_settings_path() -> Path:
    """Return the read-only Deinterlace Studio settings used for first-run tool reuse."""

    base = Path(os.environ.get("APPDATA") or Path.home())
    return base / "DeinterlaceStudio" / "settings.json"


def load_settings(
    path: Path | None = None,
    *,
    reference_path: Path | None = None,
) -> dict[str, Any]:
    target = path or default_settings_path()
    values = dict(DEFAULTS)
    loaded_own_settings = False
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            loaded_own_settings = True
            for key, default in DEFAULTS.items():
                if key in payload and isinstance(payload[key], type(default)):
                    values[key] = payload[key]
            if payload.get("settings_schema_version", 1) < 2:
                legacy_preview = payload.get("live_preview_enabled")
                if isinstance(legacy_preview, bool):
                    values["frame_preview_enabled"] = legacy_preview
                values["settings_schema_version"] = 2
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        pass

    # The denoise app is intentionally independent, but it can reuse the exact
    # external toolchain already selected for the reference application.  This
    # is first-run-only: once our settings exist, even deliberately cleared
    # paths remain cleared rather than being silently repopulated.
    if not loaded_own_settings:
        reference = reference_path or reference_settings_path()
        try:
            reference_payload = json.loads(reference.read_text(encoding="utf-8"))
            if isinstance(reference_payload, dict):
                for key in REFERENCE_TOOL_KEYS:
                    candidate = reference_payload.get(key)
                    if isinstance(candidate, str):
                        values[key] = candidate
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            pass
    return values


def save_settings(values: dict[str, Any], path: Path | None = None) -> None:
    target = path or default_settings_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: values.get(key, default) for key, default in DEFAULTS.items()}
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, target)
