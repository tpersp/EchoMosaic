"""Stream runtime caches and mutable state."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict


@dataclass
class StreamRuntime:
    stream_runtime_state: Dict[str, Dict[str, Any]]
    stream_runtime_lock: threading.Lock
    resized_image_locks: Any
    resized_image_locks_guard: threading.Lock
    video_duration_cache: Any


def build_stream_runtime(*, cache_factory: Callable[[int], Any]) -> StreamRuntime:
    return StreamRuntime(
        stream_runtime_state={},
        stream_runtime_lock=threading.Lock(),
        resized_image_locks=cache_factory(512),
        resized_image_locks_guard=threading.Lock(),
        video_duration_cache=cache_factory(128),
    )
