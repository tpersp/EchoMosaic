from __future__ import annotations

import threading

from echomosaic_app.services.stream_config import StreamConfigService


def _build_service(settings):
    ai_jobs = {}
    ai_job_controls = {}
    lock = threading.Lock()
    events = []

    service = StreamConfigService(
        settings=settings,
        ai_jobs=ai_jobs,
        ai_job_controls=ai_job_controls,
        ai_jobs_lock=lock,
        playback_manager=None,
        auto_scheduler=None,
        picsum_scheduler=None,
        cleanup_temp_outputs=lambda stream_id: None,
        save_settings_debounced=lambda: events.append("saved"),
        safe_emit=lambda event, payload: events.append((event, payload)),
        get_global_tags=lambda: ["tag-a"],
        ensure_ai_defaults=lambda conf: conf.setdefault("_ai_settings", {"prompt": ""}),
        ensure_picsum_defaults=lambda conf: conf.setdefault("_picsum_settings", {}),
        ensure_timer_defaults=lambda conf: conf.setdefault("_timer", {}),
        ensure_sync_defaults=lambda conf: conf.setdefault("_sync", {"timer_id": None, "offset": 0.0}),
        ensure_background_defaults=lambda conf: conf.setdefault("_bg_defaults", True),
        ensure_tag_defaults=lambda conf: conf.setdefault("_tags", []),
        reconcile_stale_ai_state=lambda stream_id, conf: False,
        update_stream_runtime_state=lambda *args, **kwargs: None,
        refresh_embed_metadata=lambda *args, **kwargs: None,
        sanitize_picsum_settings=lambda payload, defaults=None: dict(defaults or {}, **(payload or {})),
        default_picsum_settings=lambda: {},
        sanitize_sync_config=lambda payload, defaults=None, timers=None: payload or defaults or {"timer_id": None, "offset": 0.0},
        sanitize_ai_settings=lambda payload, current: dict(current, **payload),
        ai_settings_match_defaults=lambda candidate: False,
        default_ai_settings=lambda: {"prompt": ""},
        default_ai_state=lambda: {"status": "idle"},
        default_stream_config=lambda: {"label": "", "media_mode": "image", "mode": "random"},
        detect_media_kind=lambda value: "image",
        infer_media_mode=lambda conf: "image",
        coerce_bool=lambda value, default=False: bool(value) if value is not None else default,
        coerce_int=lambda value, default=0: int(value) if value is not None else default,
        slugify=lambda value: str(value).strip().lower().replace(" ", "-"),
        sanitize_stream_tags=lambda value: [str(item) for item in (value or [])],
        register_global_tags=lambda tags: None,
        media_mode_choices={"image", "video", "ai", "livestream"},
        media_mode_variants={"image": {"random", "specific"}, "video": {"random", "specific"}},
        media_mode_ai="ai",
        media_mode_livestream="livestream",
        media_mode_video="video",
        ai_modes={"ai", "ai_generate", "ai_random", "ai_specific"},
        ai_generate_mode="ai_generate",
        image_quality_choices={"auto", "thumb"},
        video_playback_modes={"duration", "loop"},
        tag_key="_tags",
        picsum_settings_key="_picsum_settings",
        sync_config_key="_sync",
        sync_timers_key="_sync_timers",
        ai_settings_key="_ai_settings",
        ai_state_key="_ai_state",
        stream_order_key="_stream_order",
        stream_runtime_lock=lock,
        stream_runtime_state={},
    )
    return service, events


def test_stream_config_service_creates_stream() -> None:
    service, events = _build_service({})

    stream_id = service.create_stream()

    assert stream_id == "stream1"
    assert "stream1" in service.settings
    assert ("streams_changed", {"action": "added", "stream_id": "stream1"}) in events


def test_stream_config_service_rejects_duplicate_label_slug() -> None:
    settings = {
        "stream1": {"label": "Main Wall", "_ai_settings": {"prompt": ""}, "_ai_state": {}, "_sync": {"timer_id": None, "offset": 0.0}},
        "stream2": {"label": "Secondary", "_ai_settings": {"prompt": ""}, "_ai_state": {}, "_sync": {"timer_id": None, "offset": 0.0}},
    }
    service, _ = _build_service(settings)

    try:
        service.update_stream("stream2", {"label": "Main Wall"})
    except ValueError as exc:
        assert "already uses this name" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_stream_config_service_returns_settings_payload() -> None:
    settings = {"stream1": {"label": "One"}}
    service, _ = _build_service(settings)

    payload = service.get_stream_settings_payload("stream1")

    assert payload["label"] == "One"
    assert "_ai_settings" in payload


def test_stream_config_service_tracks_stream_order_for_create_delete_and_reorder() -> None:
    settings = {
        "stream1": {"label": "One"},
        "stream2": {"label": "Two"},
        "stream10": {"label": "Ten"},
    }
    service, events = _build_service(settings)

    assert service._get_stream_order() == ["stream1", "stream2", "stream10"]

    reordered = service.reorder_streams(["stream10", "stream1", "stream2"])
    assert reordered == ["stream10", "stream1", "stream2"]
    assert settings["_stream_order"] == ["stream10", "stream1", "stream2"]
    assert ("streams_changed", {"action": "reordered", "stream_order": ["stream10", "stream1", "stream2"]}) in events

    new_stream_id = service.create_stream()
    assert new_stream_id == "stream3"
    assert settings["_stream_order"] == ["stream10", "stream1", "stream2", "stream3"]

    deleted = service.delete_stream("stream1")
    assert deleted is True
    assert settings["_stream_order"] == ["stream10", "stream2", "stream3"]
