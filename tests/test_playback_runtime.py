from __future__ import annotations

from echomosaic_app.state.playback_runtime import build_playback_runtime


def test_build_playback_runtime_exposes_defaults_and_manager_slot() -> None:
    runtime = build_playback_runtime()

    assert runtime.stream_playback_history_limit == 50
    assert runtime.stream_random_recent_avoid_count == 10
    assert runtime.stream_update_event == "stream_update"
    assert runtime.stream_init_event == "stream_init"
    assert runtime.sync_time_event == "sync_time"
    assert runtime.stream_sync_interval_seconds == 3.0
    assert runtime.playback_manager is None
