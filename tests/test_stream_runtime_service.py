from __future__ import annotations

import threading

from echomosaic_app.services.stream_runtime import StreamRuntimeService


def _build_service():
    state = {}
    refresh_calls = []
    service = StreamRuntimeService(
        stream_runtime_lock=threading.Lock(),
        stream_runtime_state=state,
        video_extensions={".mp4"},
        ai_mode="ai",
        ai_generate_mode="ai_generate",
        ai_random_mode="ai_random",
        ai_specific_mode="ai_specific",
        media_mode_ai="ai",
        media_mode_livestream="livestream",
        media_mode_picsum="picsum",
        media_mode_video="video",
        media_mode_image="image",
        refresh_stream_thumbnail=lambda stream_id, force=False: refresh_calls.append((stream_id, force)) or {"url": "thumb"},
    )
    return service, state, refresh_calls


def test_stream_runtime_service_infers_video_mode_from_selected_kind() -> None:
    service, _, _ = _build_service()

    result = service.infer_media_mode({"mode": "random", "selected_media_kind": "video"})

    assert result == "video"


def test_stream_runtime_service_updates_state_and_thumbnail() -> None:
    service, state, refresh_calls = _build_service()

    payload = service.update_stream_runtime_state(
        "stream1",
        path="root/clip.mp4",
        media_mode="video",
        stream_url=" https://example.com/live ",
        source="test",
    )

    assert payload == {"url": "thumb"}
    assert state["stream1"]["kind"] == "video"
    assert state["stream1"]["stream_url"] == "https://example.com/live"
    assert refresh_calls == [("stream1", False)]


def test_stream_runtime_service_formats_timestamp_as_utc() -> None:
    service, _, _ = _build_service()

    result = service.runtime_timestamp_to_iso(1700000000)

    assert result == "2023-11-14T22:13:20Z"
