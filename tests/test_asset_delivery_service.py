from __future__ import annotations

from pathlib import Path

from echomosaic_app.services.asset_delivery import AssetDeliveryService


class _Guard:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _Response:
    def __init__(self) -> None:
        self.headers = {}


def _build_service(tmp_path: Path) -> AssetDeliveryService:
    file_path = tmp_path / "sample.jpg"
    file_path.write_bytes(b"image-bytes")

    sent = []

    def fake_send_file(path, **kwargs):
        sent.append((path, kwargs))
        return _Response()

    return AssetDeliveryService(
        send_file=fake_send_file,
        jsonify=lambda payload: payload,
        url_for=lambda endpoint, **kwargs: f"/{endpoint}/{kwargs.get('stream_id', '')}",
        request_args_get=lambda key, default=None: default,
        parse_truthy=lambda value: False,
        as_int=lambda value, default=0: default,
        generate_etag=lambda value: "etag",
        logger=type("Logger", (), {"warning": lambda *args, **kwargs: None, "debug": lambda *args, **kwargs: None})(),
        bad_media_log_cache={},
        bad_media_log_ttl=60.0,
        image_extensions={".jpg", ".jpeg", ".png"},
        video_extensions={".mp4"},
        max_image_dimension=2048,
        thumbnail_size_presets={},
        thumbnail_subdir="_thumbs",
        thumbnail_jpeg_quality=85,
        image_thumbnail_filter=object(),
        image_cache_timeout=3600,
        image_cache_control_max_age=3600,
        media_root_lookup={"media": type("Root", (), {"path": tmp_path})()},
        split_virtual_media_path=lambda value: ("media", "sample.jpg"),
        resolve_virtual_media_path=lambda virtual_path: file_path,
        ensure_thumbnail_dir=lambda: tmp_path,
        thumbnail_disk_path=lambda stream_id: tmp_path / f"{stream_id}.jpg",
        thumbnail_public_url=lambda stream_id: f"/thumbs/{stream_id}.jpg",
        public_thumbnail_payload=lambda record: record,
        compute_thumbnail_snapshot=lambda stream_id: {"stream_id": stream_id, "timestamp": 1.0, "media_mode": "image", "kind": "image", "path": "sample.jpg", "badge": "Image", "source": "test"},
        refresh_stream_thumbnail=lambda stream_id, snapshot=None, force=False: {"url": "/thumbs/sample.jpg"},
        get_runtime_thumbnail_payload=lambda stream_id: {"url": "/thumbs/sample.jpg"},
        runtime_timestamp_to_iso=lambda ts: "2025-01-01T00:00:00Z",
        render_thumbnail_image=lambda info: (object(), False),
        thumbnail_image_to_bytes=lambda image: b"bytes",
        resized_image_locks={},
        resized_image_locks_guard=_Guard(),
        stream_runtime_lock=_Guard(),
        stream_runtime_state={},
        Image=type("ImageModule", (), {"open": None}),
        ImageOps=type("ImageOpsModule", (), {"exif_transpose": lambda img: img}),
    )


def test_asset_delivery_service_serves_image_response(tmp_path: Path) -> None:
    service = _build_service(tmp_path)

    response = service.serve_image("media:/sample.jpg")

    assert isinstance(response, _Response)
    assert response.headers["Cache-Control"] == "public, max-age=3600"


def test_asset_delivery_service_builds_thumbnail_metadata_payload(tmp_path: Path) -> None:
    service = _build_service(tmp_path)

    payload = service.stream_thumbnail_metadata("stream1")

    assert payload["stream_id"] == "stream1"
    assert payload["thumbnail"]["url"] == "/thumbs/sample.jpg"
