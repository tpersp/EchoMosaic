from __future__ import annotations

from echomosaic_app.services.groups import GroupService


def _build_service(settings):
    events = []
    service = GroupService(
        settings=settings,
        save_settings_debounced=lambda: events.append("saved"),
        safe_emit=lambda event, payload: events.append((event, payload)),
        maybe_int=lambda value: int(value) if value not in (None, "") else None,
        clamp=lambda value, lower, upper: max(lower, min(upper, value)),
    )
    return service, events


def test_group_service_creates_group_and_emits_refresh() -> None:
    settings = {"stream1": {}, "stream2": {}, "_groups": {}}
    service, events = _build_service(settings)

    payload = service.create_group({"name": "Wall", "streams": ["stream1", "stream2", "missing"]})

    assert payload["status"] == "ok"
    assert settings["_groups"]["Wall"] == ["stream1", "stream2"]
    assert "saved" in events
    assert ("mosaic_refresh", {"group": "Wall"}) in events


def test_group_service_normalizes_focus_layout() -> None:
    service, _ = _build_service({})

    layout = service.normalize_group_layout({"layout": "focus", "focus_mode": "1-2", "focus_pos": "left", "cols": 20})

    assert layout["layout"] == "focus"
    assert layout["focus_mode"] == "1-2"
    assert layout["focus_pos"] == "left"
    assert layout["cols"] == 8


def test_group_service_builds_group_view_model() -> None:
    settings = {
        "stream1": {"label": "One"},
        "stream2": {"label": "Two"},
        "_groups": {"Wall": {"streams": ["stream2", "stream1", "stream2"], "layout": {"layout": "focus", "focus_main": "stream1"}}},
    }
    service, _ = _build_service(settings)

    model = service.build_group_view_model("Wall")

    assert model["stream_order"] == ["stream2", "stream1", "stream2"]
    assert model["unique_stream_ids"] == ["stream2", "stream1"]
    assert model["focus_order"][0] == "stream1"
