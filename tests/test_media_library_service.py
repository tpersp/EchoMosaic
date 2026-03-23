from __future__ import annotations

from echomosaic_app.services.media_library import MediaLibraryService


def _build_service(settings=None):
    runtime_updates = []
    service = MediaLibraryService(
        settings=settings or {},
        parse_truthy=lambda value: str(value).lower() in {"1", "true", "yes", "on"},
        normalize_library_key=lambda value, default="media": value or default,
        list_images=lambda folder="all", hide_nsfw=False, library="media": ["a.jpg", "b.jpg", "c.jpg"],
        list_media=lambda folder="all", hide_nsfw=False, library="media": [
            {"path": "a.jpg", "kind": "image"},
            {"path": "b.mp4", "kind": "video"},
        ],
        infer_media_mode=lambda conf: "image",
        update_stream_runtime_state=lambda *args, **kwargs: runtime_updates.append((args, kwargs)),
        media_library_default="media",
    )
    return service, runtime_updates


def test_media_library_service_lists_images_with_pagination() -> None:
    service, _ = _build_service()

    payload = service.get_images_payload(offset=1, limit=1)

    assert payload == ["b.jpg"]


def test_media_library_service_filters_media_by_kind() -> None:
    service, _ = _build_service()

    payload = service.get_media_entries_payload(kind="video")

    assert payload == [{"path": "b.mp4", "kind": "video"}]


def test_media_library_service_updates_runtime_on_random_media() -> None:
    settings = {"stream1": {"media_mode": "image"}}
    service, runtime_updates = _build_service(settings=settings)

    payload = service.get_random_media_payload(stream_id="stream1")

    assert payload["kind"] in {"image", "video"}
    assert len(runtime_updates) == 1
