"""
Lightweight helpers to collect system statistics for the dashboard.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

try:  # psutil is required for CPU, memory, and disk metrics.
    import psutil  # type: ignore[import]
except Exception:  # pragma: no cover - psutil should be available but guard anyway.
    psutil = None  # type: ignore[assignment]

try:  # GPUtil is optional; absence means GPU stats are skipped.
    import GPUtil  # type: ignore[import]
except Exception:  # pragma: no cover - GPU monitoring is optional.
    GPUtil = None  # type: ignore[assignment]

_MEDIA_CANDIDATES = [
    Path("/mnt/media"),
    Path("/mnt/viewers"),
    Path.cwd() / "media",
    Path(__file__).resolve().parent / "media",
]


def _detect_media_path() -> Optional[Path]:
    """Return the first existing media directory, or None if unavailable."""
    for candidate in _MEDIA_CANDIDATES:
        try:
            if candidate.is_dir():
                return candidate
        except OSError:
            continue
    return None


def _bytes_to_gb(value: float) -> float:
    """Convert bytes to gigabytes rounded to one decimal place."""
    return round(value / (1024**3), 1)


def get_system_stats() -> Dict[str, Any]:
    """
    Gather basic system statistics for the dashboard.

    Returns:
        dict: Metrics keyed by human-friendly names. Fields default to None when
        unavailable so the caller can handle missing data gracefully.
    """

    stats: Dict[str, Any] = {
        "cpu_percent": None,
        "memory_used": None,
        "memory_total": None,
        "memory_percent": None,
        "gpu_percent": None,
        "disk_used": None,
        "disk_total": None,
        "disk_percent": None,
        "disk_path": None,
    }

    if psutil is None:  # pragma: no cover - defensive fallback.
        return stats

    try:
        stats["cpu_percent"] = round(psutil.cpu_percent(interval=0.1), 1)
    except Exception:
        pass

    try:
        memory = psutil.virtual_memory()
        stats["memory_used"] = _bytes_to_gb(memory.used)
        stats["memory_total"] = _bytes_to_gb(memory.total)
        stats["memory_percent"] = round(memory.percent, 1)
    except Exception:
        pass

    media_path = _detect_media_path()
    if media_path:
        stats["disk_path"] = str(media_path)
        try:
            disk = psutil.disk_usage(str(media_path))
            stats["disk_used"] = _bytes_to_gb(disk.used)
            stats["disk_total"] = _bytes_to_gb(disk.total)
            stats["disk_percent"] = round(disk.percent, 1)
        except Exception:
            pass

    if GPUtil is not None:
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                avg_load = sum(gpu.load for gpu in gpus) / len(gpus)
                stats["gpu_percent"] = round(avg_load * 100.0, 1)
        except Exception:
            pass

    return stats
