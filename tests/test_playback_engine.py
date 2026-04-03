from __future__ import annotations

from echomosaic_app.services.playback_engine import StreamPlaybackManager


def test_playback_engine_can_bootstrap_and_emit_initial_media() -> None:
    emitted = []
    runtime_updates = []
    manager = StreamPlaybackManager(
        safe_emit=lambda event, payload, to=None: emitted.append((event, payload, to)),
        list_media=lambda folder, library="media": [{"path": "root/photo.jpg", "kind": "image"}],
        library_for_media_mode=lambda mode: "media",
        update_stream_runtime_state=lambda stream_id, **kwargs: runtime_updates.append((stream_id, kwargs)) or {"url": "thumb"},
        get_runtime_thumbnail_payload=lambda stream_id: {"url": "thumb"},
        get_sync_timer_config=lambda timer_id: None,
        compute_next_sync_tick=lambda now, interval, offset: None,
        coerce_float=lambda value, default: default,
        infer_media_mode=lambda conf: "image",
        resolve_media_path=lambda path: None,
        video_duration_cache={},
        cv2_module=None,
        media_mode_choices={"image", "video", "ai"},
        media_mode_image="image",
        media_mode_video="video",
        media_mode_ai="ai",
        ai_random_mode="ai-random",
        video_playback_modes={"duration", "loop", "until_end"},
        sync_config_key="_sync",
        sync_supported_media_modes={"image"},
        stream_playback_history_limit=50,
        stream_update_event="stream_update",
        sync_time_event="sync_time",
        stream_sync_interval_seconds=3.0,
        sync_switch_lead_seconds=0.25,
    )
    try:
        manager.bootstrap(
            {
                "stream1": {
                    "mode": "random",
                    "media_mode": "image",
                    "folder": "all",
                    "duration": 5,
                }
            }
        )

        payload = manager.ensure_started("stream1")

        assert payload is not None
        assert payload["status"] == "playing"
        assert runtime_updates[0][0] == "stream1"
        assert emitted[0][0] == "stream_update"
    finally:
        manager.stop()
