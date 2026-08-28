"""Runtime configuration profiles for constrained deployments."""

from __future__ import annotations

from typing import Any, Dict


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


LOW_MEMORY_LIMITS = {
    "LIVE_HLS_MAX_WORKERS": 1,
    "HLS_CACHE_SIZE": 64,
    "HLS_JOB_CACHE_SIZE": 32,
    "MEDIA_CATALOG_CACHE_SIZE": 16,
    "BAD_MEDIA_CACHE_SIZE": 256,
    "RESIZED_IMAGE_LOCK_CACHE_SIZE": 128,
    "VIDEO_DURATION_CACHE_SIZE": 64,
    "YOUTUBE_OEMBED_CACHE_SIZE": 64,
    "YOUTUBE_LIVE_PROBE_CACHE_SIZE": 64,
    "YOUTUBE_PLAYLIST_CACHE_SIZE": 24,
    "MEDIA_PREVIEW_FRAMES": 3,
    "MEDIA_PREVIEW_WIDTH": 240,
    "MEDIA_PREVIEW_MAX_MB": 128,
    "MEDIA_THUMB_WIDTH": 240,
}


def apply_runtime_profile(config: Dict[str, Any]) -> Dict[str, Any]:
    """Return effective config with conservative caps for low-memory mode."""

    effective = dict(config)
    if not _as_bool(effective.get("LOW_MEMORY_MODE")):
        return effective

    for key, limit in LOW_MEMORY_LIMITS.items():
        try:
            current = int(effective.get(key, limit))
        except (TypeError, ValueError):
            current = limit
        effective[key] = min(current, limit)

    # Video preview decoding is the largest avoidable transient allocation.
    # It can still be explicitly re-enabled by turning the profile off.
    effective["MEDIA_PREVIEW_ENABLED"] = False
    return effective
