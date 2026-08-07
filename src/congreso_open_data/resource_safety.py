from __future__ import annotations

import ctypes
import os
import subprocess
import sys
from ctypes import wintypes
from dataclasses import asdict, dataclass
from typing import Any

GIB = 1024**3
DEFAULT_DUCKDB_MEMORY_LIMIT = "1GB"
DEFAULT_DUCKDB_MAX_TEMP_DIRECTORY_SIZE = "64GB"


class ResourceBudgetError(RuntimeError):
    """Raised before a bulk job can exhaust workstation memory."""


@dataclass(frozen=True)
class MemorySnapshot:
    total_bytes: int | None
    available_bytes: int | None
    process_rss_bytes: int | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def memory_snapshot() -> MemorySnapshot:
    if sys.platform == "win32":
        return _windows_memory_snapshot()
    return _posix_memory_snapshot()


def gpu_memory_snapshot() -> dict[str, object]:
    """Return bounded NVIDIA telemetry without importing a GPU framework."""

    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        first = completed.stdout.strip().splitlines()[0]
        name, total, used, free, utilization = [value.strip() for value in first.split(",")]
        return {
            "available": True,
            "name": name,
            "memory_total_mib": int(total),
            "memory_used_mib": int(used),
            "memory_free_mib": int(free),
            "utilization_percent": int(utilization),
        }
    except (FileNotFoundError, IndexError, subprocess.SubprocessError, ValueError):
        return {"available": False}


def assert_model_fully_cuda(model: Any, *, model_name: str = "model") -> None:
    """Fail when any part of a GPU-only model was offloaded to CPU or disk."""

    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, dict):
        invalid = []
        for device in device_map.values():
            normalized = str(device).casefold()
            if normalized == "0" or normalized.startswith("cuda"):
                continue
            invalid.append(str(device))
        if invalid:
            raise RuntimeError(
                f"{model_name} is not fully on CUDA; placements={sorted(set(invalid))}"
            )
        return
    try:
        device = str(next(model.parameters()).device)
    except (AttributeError, StopIteration) as exc:
        raise RuntimeError(f"Cannot verify {model_name} CUDA placement") from exc
    if not device.casefold().startswith("cuda"):
        raise RuntimeError(f"{model_name} is not on CUDA: {device}")


def require_memory_budget(
    *,
    estimated_additional_bytes: int,
    minimum_free_bytes: int = 3 * GIB,
    maximum_process_bytes: int = 4 * GIB,
    snapshot: MemorySnapshot | None = None,
) -> MemorySnapshot:
    """Fail before allocating when a conservative memory budget is unsafe."""

    if estimated_additional_bytes < 0:
        raise ValueError("estimated_additional_bytes must be non-negative")
    if minimum_free_bytes < 0 or maximum_process_bytes <= 0:
        raise ValueError("memory limits must be positive")
    observed = snapshot or memory_snapshot()
    if observed.available_bytes is None or observed.process_rss_bytes is None:
        raise ResourceBudgetError(
            "Memory budget cannot be verified on this runtime; refusing bulk allocation"
        )
    projected_free = observed.available_bytes - estimated_additional_bytes
    projected_process = observed.process_rss_bytes + estimated_additional_bytes
    if projected_free < minimum_free_bytes:
        raise ResourceBudgetError(
            "Insufficient memory headroom: "
            f"available={_gib(observed.available_bytes)}GiB, "
            f"estimated_additional={_gib(estimated_additional_bytes)}GiB, "
            f"required_remaining={_gib(minimum_free_bytes)}GiB"
        )
    if projected_process > maximum_process_bytes:
        raise ResourceBudgetError(
            "Projected process RSS exceeds the bulk-job limit: "
            f"rss={_gib(observed.process_rss_bytes)}GiB, "
            f"estimated_additional={_gib(estimated_additional_bytes)}GiB, "
            f"limit={_gib(maximum_process_bytes)}GiB"
        )
    return observed


def estimate_ann_peak_bytes(
    *,
    rows: int,
    dimensions: int,
    connectivity: int,
    query_batch_size: int,
) -> int:
    """Conservative peak for f16 memmap + ANN copy + graph + query buffers."""

    if rows < 0 or dimensions <= 0 or connectivity <= 0 or query_batch_size <= 0:
        raise ValueError("ANN sizing parameters must be positive")
    vector_bytes = rows * dimensions * 2
    graph_bytes = rows * connectivity * 16
    query_bytes = query_batch_size * dimensions * 8
    fixed_overhead = 256 * 1024**2
    return 2 * vector_bytes + graph_bytes + query_bytes + fixed_overhead


def configure_duckdb(
    connection: Any,
    *,
    memory_limit: str = DEFAULT_DUCKDB_MEMORY_LIMIT,
    threads: int = 1,
    max_temp_directory_size: str = DEFAULT_DUCKDB_MAX_TEMP_DIRECTORY_SIZE,
) -> None:
    """Apply the workstation-safe DuckDB envelope to a connection."""

    if threads <= 0:
        raise ValueError("DuckDB threads must be positive")
    connection.execute("SET memory_limit = ?", [memory_limit])
    connection.execute("SET threads = ?", [threads])
    connection.execute("SET preserve_insertion_order = false")
    connection.execute(
        "SET max_temp_directory_size = ?",
        [max_temp_directory_size],
    )


def _windows_memory_snapshot() -> MemorySnapshot:
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

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    status = MemoryStatusEx()
    status.dwLength = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("GlobalMemoryStatusEx failed")
    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(ProcessMemoryCounters),
        wintypes.DWORD,
    )
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    handle = kernel32.GetCurrentProcess()
    if not psapi.GetProcessMemoryInfo(
        handle,
        ctypes.byref(counters),
        counters.cb,
    ):
        raise OSError("GetProcessMemoryInfo failed")
    return MemorySnapshot(
        total_bytes=int(status.ullTotalPhys),
        available_bytes=int(status.ullAvailPhys),
        process_rss_bytes=int(counters.WorkingSetSize),
    )


def _posix_memory_snapshot() -> MemorySnapshot:
    total = available = rss = None
    if sys.platform.startswith("linux"):
        values: dict[str, int] = {}
        with open("/proc/meminfo", encoding="ascii") as handle:
            for line in handle:
                key, raw = line.split(":", 1)
                values[key] = int(raw.strip().split()[0]) * 1024
        total = values.get("MemTotal")
        available = values.get("MemAvailable")
        with open("/proc/self/statm", encoding="ascii") as handle:
            resident_pages = int(handle.read().split()[1])
        rss = resident_pages * os.sysconf("SC_PAGE_SIZE")
    else:
        try:
            import resource

            maximum = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            rss = int(maximum if sys.platform == "darwin" else maximum * 1024)
        except (ImportError, OSError, ValueError):
            pass
    return MemorySnapshot(
        total_bytes=total,
        available_bytes=available,
        process_rss_bytes=rss,
    )


def _gib(value: int) -> str:
    return f"{value / GIB:.2f}"
