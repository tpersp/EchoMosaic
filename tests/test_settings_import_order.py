from __future__ import annotations

from echomosaic_app import server


def _stream_payload(label: str) -> dict:
    return {
        "label": label,
        "mode": "random",
        "media_mode": "image",
        "folder": "all",
    }


def test_prepare_settings_import_prefers_explicit_stream_order_metadata() -> None:
    payload = {
        "_stream_order": ["stream1", "stream2", "stream10"],
        "stream1": _stream_payload("One"),
        "stream10": _stream_payload("Ten"),
        "stream2": _stream_payload("Two"),
    }

    snapshot, warnings = server._prepare_settings_import(payload)

    assert warnings == []
    assert [key for key in snapshot.keys() if not key.startswith("_")] == ["stream1", "stream2", "stream10"]


def test_prepare_settings_import_natural_sorts_older_exports_without_order_metadata() -> None:
    payload = {
        "stream1": _stream_payload("One"),
        "stream10": _stream_payload("Ten"),
        "stream2": _stream_payload("Two"),
    }

    snapshot, warnings = server._prepare_settings_import(payload)

    assert warnings == []
    assert [key for key in snapshot.keys() if not key.startswith("_")] == ["stream1", "stream2", "stream10"]
