"""Media cache runtime owners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class MediaCacheRuntime:
    image_cache: Any
    bad_media_log_cache: Any


def build_media_cache_runtime(
    *, cache_factory: Callable[[int], Any], image_cache_size: int = 64, bad_media_cache_size: int = 1024
) -> MediaCacheRuntime:
    return MediaCacheRuntime(
        image_cache=cache_factory(max(1, image_cache_size)),
        bad_media_log_cache=cache_factory(max(1, bad_media_cache_size)),
    )
