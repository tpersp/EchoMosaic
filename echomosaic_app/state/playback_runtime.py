"""Playback runtime ownership and defaults."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class PlaybackRuntime:
    stream_playback_history_limit: int
    stream_update_event: str
    stream_init_event: str
    sync_time_event: str
    stream_sync_interval_seconds: float
    playback_manager: Optional[Any] = None


def build_playback_runtime() -> PlaybackRuntime:
    return PlaybackRuntime(
        stream_playback_history_limit=50,
        stream_update_event="stream_update",
        stream_init_event="stream_init",
        sync_time_event="sync_time",
        stream_sync_interval_seconds=3.0,
        playback_manager=None,
    )
