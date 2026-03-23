"""Live HLS runtime bootstrap and executor state."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class HLSRuntime:
    live_hls_async: bool
    hls_ttl_secs: int
    max_hls_workers: int
    hls_error_retry_secs: int
    hls_lock: threading.RLock
    hls_metrics: Dict[str, int]
    hls_cache: Any
    hls_jobs: Any
    hls_executor: Optional[ThreadPoolExecutor]
    hls_log_prefix: str = "live_hls"


def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if value is None:
        return default
    return bool(value)


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def build_hls_runtime(
    *,
    config: Dict[str, Any],
    cache_factory,
    live_hls_async: bool,
    hls_ttl_secs: int,
    max_hls_workers: int,
    hls_error_retry_secs: int,
) -> HLSRuntime:
    configured_live_hls_async = _coerce_bool(config.get("LIVE_HLS_ASYNC"), live_hls_async)
    configured_hls_ttl_secs = max(60, _coerce_int(config.get("LIVE_HLS_TTL_SECS"), hls_ttl_secs))
    configured_max_hls_workers = max(1, _coerce_int(config.get("LIVE_HLS_MAX_WORKERS"), max_hls_workers))
    configured_hls_error_retry_secs = max(
        5,
        _coerce_int(config.get("LIVE_HLS_ERROR_RETRY_SECS"), hls_error_retry_secs),
    )
    configured_hls_error_retry_secs = min(configured_hls_error_retry_secs, configured_hls_ttl_secs)

    hls_executor: Optional[ThreadPoolExecutor]
    if configured_live_hls_async:
        hls_executor = ThreadPoolExecutor(
            max_workers=configured_max_hls_workers,
            thread_name_prefix="live-hls",
        )
    else:
        hls_executor = None

    return HLSRuntime(
        live_hls_async=configured_live_hls_async,
        hls_ttl_secs=configured_hls_ttl_secs,
        max_hls_workers=configured_max_hls_workers,
        hls_error_retry_secs=configured_hls_error_retry_secs,
        hls_lock=threading.RLock(),
        hls_metrics={
            "hits": 0,
            "misses": 0,
            "stale": 0,
            "jobs_started": 0,
            "jobs_completed": 0,
            "errors": 0,
        },
        hls_cache=cache_factory(256),
        hls_jobs=cache_factory(128),
        hls_executor=hls_executor,
    )
