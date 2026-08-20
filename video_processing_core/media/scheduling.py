from __future__ import annotations

import ctypes
import math
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class VapourSynthSchedule:
    core_threads: int
    requests: int
    logical_threads: int
    memory_mib: int | None
    estimated_request_mib: int
    memory_budget_mib: int | None
    rationale: str


def physical_memory_mib() -> int | None:
    """Return installed physical memory without adding a runtime dependency."""

    if os.name == "nt":
        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatusEx()
        status.dwLength = ctypes.sizeof(status)
        try:
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return max(1, int(status.ullTotalPhys // (1024 * 1024)))
        except (AttributeError, OSError):
            return None
        return None

    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return max(1, int(pages * page_size // (1024 * 1024)))
    except (AttributeError, OSError, ValueError):
        return None


def _working_bytes_per_pixel(pixel_format: str | None) -> int:
    value = (pixel_format or "").casefold()
    if "444" in value or value.startswith("gbr"):
        return 6
    if "422" in value:
        return 4
    return 3


def choose_vapoursynth_schedule(
    width: int,
    height: int,
    pixel_format: str | None,
    *,
    temporal_denoise: bool = False,
    vulkan_nnedi3: bool = False,
    logical_threads: int | None = None,
    memory_mib: int | None = None,
) -> VapourSynthSchedule:
    """Choose bounded graph concurrency without changing filter mathematics.

    The CPU caps are calibrated from the exact quality-first QTGMC graph.  A
    memory ceiling then reduces concurrency for large rasters and heavier
    temporal graphs.  Missing resource information chooses conservative values
    rather than blocking the job or guessing an unbounded request count.
    """

    logical = max(1, int(logical_threads or os.cpu_count() or 4))
    core_threads = min(16, logical)
    target = core_threads if vulkan_nnedi3 else math.ceil(core_threads * 1.5)
    target = min(24, max(1, target))

    pixels = max(1, int(width) * int(height))
    if pixels >= 3840 * 2160:
        target = min(target, 12)
    elif pixels >= 1920 * 1080:
        target = min(target, 16)
    if temporal_denoise:
        target = min(target, max(4, math.ceil(core_threads * 0.75)))

    frame_bytes = pixels * _working_bytes_per_pixel(pixel_format)
    temporal_multiplier = 96 if temporal_denoise else 64
    estimated_request_mib = max(
        96,
        math.ceil(
            (64 * 1024 * 1024 + frame_bytes * temporal_multiplier)
            / (1024 * 1024)
        ),
    )

    total_memory = memory_mib if memory_mib is not None else physical_memory_mib()
    memory_budget = None
    requests = target
    if total_memory is not None and total_memory > 0:
        memory_budget = max(512, min(24 * 1024, int(total_memory * 0.25)))
        requests = min(target, max(1, memory_budget // estimated_request_mib))
    elif pixels >= 1920 * 1080:
        requests = min(requests, 8)

    mode = "Vulkan NNEDI3" if vulkan_nnedi3 else "CPU NNEDI3"
    denoise_note = "; temporal-denoise memory guard active" if temporal_denoise else ""
    memory_note = (
        f"; {memory_budget} MiB bounded memory budget from {total_memory} MiB physical RAM"
        if memory_budget is not None
        else "; physical RAM unavailable, conservative resolution cap applied"
    )
    rationale = (
        f"Adaptive {mode}: {logical} logical CPU threads, {core_threads} VapourSynth core threads, "
        f"target {target} requests, selected {requests}; estimated {estimated_request_mib} MiB/request"
        f"{memory_note}{denoise_note}."
    )
    return VapourSynthSchedule(
        core_threads=core_threads,
        requests=requests,
        logical_threads=logical,
        memory_mib=total_memory,
        estimated_request_mib=estimated_request_mib,
        memory_budget_mib=memory_budget,
        rationale=rationale,
    )
