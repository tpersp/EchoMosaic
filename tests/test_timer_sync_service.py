from __future__ import annotations

from echomosaic_app.services.timer_sync import TimerSyncService


def _build_service(settings):
    events = []
    service = TimerSyncService(
        settings=settings,
        config={},
        auto_scheduler=None,
        picsum_scheduler=None,
        playback_manager=None,
        ensure_ai_defaults=lambda conf: conf.setdefault("_ai_state", {}),
        ensure_picsum_defaults=lambda conf: conf.setdefault("_picsum_settings", {}),
        ensure_sync_defaults=lambda conf, timers=None: conf.setdefault("_sync", {"timer_id": None, "offset": 0.0}),
        save_settings_debounced=lambda: events.append("saved"),
        safe_emit=lambda event, payload, **kwargs: events.append((event, payload)),
        get_global_tags=lambda: [],
        sanitize_sync_timer_entry=lambda timer_id, payload: {
            "label": payload.get("label") or timer_id,
            "interval": float(payload.get("interval", 30)),
        },
        ai_state_key="_ai_state",
        picsum_settings_key="_picsum_settings",
        sync_timers_key="_sync_timers",
        sync_config_key="_sync",
        sync_timer_default_interval=30.0,
    )
    return service, events


def test_timer_sync_service_creates_timer() -> None:
    service, events = _build_service({"_sync_timers": {}})

    payload, status = service.create_sync_timer({"label": "Demo", "interval": 45})

    assert status == 201
    assert payload["timer"]["label"] == "Demo"
    assert payload["timers"][0]["interval"] == 45.0
    assert "saved" in events


def test_timer_sync_service_updates_timer() -> None:
    settings = {"_sync_timers": {"master": {"label": "Master", "interval": 30.0}}}
    service, _ = _build_service(settings)

    payload = service.update_sync_timer("master", {"label": "Primary", "interval": 60})

    assert payload["timer"]["label"] == "Primary"
    assert payload["timer"]["interval"] == 60.0


def test_timer_sync_service_deletes_timer() -> None:
    settings = {"_sync_timers": {"master": {"label": "Master", "interval": 30.0}}}
    service, _ = _build_service(settings)

    payload = service.delete_sync_timer("master")

    assert payload["status"] == "deleted"
    assert payload["timers"] == []
