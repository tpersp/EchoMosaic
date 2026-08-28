"""Lazy access to heavyweight optional media dependencies."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Optional


@lru_cache(maxsize=1)
def load_cv2() -> Optional[Any]:
    try:
        import cv2  # type: ignore[import]
    except Exception:
        return None
    return cv2


class LazyYoutubeDL:
    """Construct yt-dlp only when a feature actually needs extraction."""

    def __new__(cls, *args: Any, **kwargs: Any):
        from yt_dlp import YoutubeDL  # type: ignore[import]

        return YoutubeDL(*args, **kwargs)
