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
    "window_geometry": "1180x900",
    "automatic_repair_and_continue": True,
    "denoise_enabled": True,
    "denoiser": "vs_bm3d",
    "denoise_strength": 4,
    "denoise_temporal_radius": 3,
    "ffv1_chroma_mode": "native",
    "vulkan_nnedi3": False,
    "batch_output_dir": "",
    "batch_include_subfolders": False,
    "batch_continue_after_error": True,
}


def default_settings_path() -> Path:
    base = Path(os.environ.get("APPDATA") or Path.home())
    return base / "DeinterlaceStudio" / "settings.json"


def load_settings(path: Path | None = None) -> dict[str, Any]:
    target = path or default_settings_path()
    values = dict(DEFAULTS)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            schema_version = payload.get("settings_schema_version", 1)
            if not isinstance(schema_version, int):
                schema_version = 1
            for key in DEFAULTS:
                # Version 2 intentionally adopts the newly requested shared
                # defaults.  Older settings files recorded denoise disabled
                # and radius 2, which would otherwise make a fresh GUI
                # contradict the documented Single and Batch defaults.
                if schema_version < 2 and key in {
                    "denoise_enabled",
                    "denoise_temporal_radius",
                }:
                    continue
                if key in payload and isinstance(payload[key], type(DEFAULTS[key])):
                    values[key] = payload[key]
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
