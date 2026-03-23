"""YouTube-related runtime caches and sync state."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, Set


@dataclass
class YouTubeRuntime:
    youtube_oembed_cache: Any
    youtube_oembed_cache_lock: threading.Lock
    youtube_live_probe_cache: Any
    youtube_live_probe_cache_lock: threading.Lock
    youtube_sync_state_lock: threading.Lock
    youtube_sync_state: Dict[str, Dict[str, Any]]
    youtube_sync_subscribers: Dict[str, Set[str]]
    youtube_sync_leaders: Dict[str, str]
    youtube_in_flight: Set[str]
    youtube_in_flight_lock: threading.Lock


def build_youtube_runtime(*, cache_factory: Callable[[int], Any]) -> YouTubeRuntime:
    return YouTubeRuntime(
        youtube_oembed_cache=cache_factory(256),
        youtube_oembed_cache_lock=threading.Lock(),
        youtube_live_probe_cache=cache_factory(256),
        youtube_live_probe_cache_lock=threading.Lock(),
        youtube_sync_state_lock=threading.Lock(),
        youtube_sync_state={},
        youtube_sync_subscribers={},
        youtube_sync_leaders={},
        youtube_in_flight=set(),
        youtube_in_flight_lock=threading.Lock(),
    )
