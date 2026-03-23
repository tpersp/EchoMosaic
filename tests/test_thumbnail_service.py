from __future__ import annotations

from pathlib import Path

from echomosaic_app.services.thumbnailing import ThumbnailService


class _ImageObject:
    width = 64
    height = 64
    mode = "RGB"

    def convert(self, mode):
        return self

    def thumbnail(self, size, resample):
        return None

    def save(self, buffer, format=None, quality=None, optimize=None):
        buffer.write(b"jpeg-bytes")


class _ImageModule:
    @staticmethod
    def new(mode, size, color):
        return _ImageObject()


class _ImageDrawModule:
    class _Draw:
        def textbbox(self, pos, text, font=None):
            return (0, 0, 20, 10)

        def text(self, pos, text, fill=None, font=None):
            return None

    @staticmethod
    def Draw(image):
        return _ImageDrawModule._Draw()


class _ImageFontModule:
    @staticmethod
    def load_default():
        class _Font:
            def getsize(self, text):
                return (20, 10)

        return _Font()


class _ImageOpsModule:
    @staticmethod
    def exif_transpose(img):
        return img


class _Guard:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _build_service(tmp_path: Path) -> ThumbnailService:
    settings = {
        "stream1": {
            "media_mode": "image",
            "selected_image": "media:/image.jpg",
            "stream_url": None,
        }
    }
    runtime_state = {
        "stream1": {
            "path": "media:/image.jpg",
            "kind": "image",
            "media_mode": "image",
            "timestamp": 1.0,
            "source": "test",
        }
    }
    return ThumbnailService(
        Image=_ImageModule,
        ImageDraw=_ImageDrawModule,
        ImageFont=_ImageFontModule,
        ImageOps=_ImageOpsModule,
        cv2_module=None,
        requests_module=None,
        eventlet_module=type("Eventlet", (), {"spawn": staticmethod(lambda fn: fn())}),
        logger=type("Logger", (), {"debug": lambda *args, **kwargs: None})(),
        dashboard_thumbnail_size=(128, 72),
        image_thumbnail_filter=object(),
        thumbnail_jpeg_quality=60,
        settings=settings,
        get_stream_runtime_state=lambda stream_id: dict(runtime_state.get(stream_id, {})),
        detect_media_kind=lambda value: "image",
        infer_media_mode=lambda conf: "image",
        resolve_media_path=lambda rel_path: tmp_path / "image.jpg",
        ensure_thumbnail_dir=lambda: tmp_path,
        thumbnail_disk_path=lambda stream_id: tmp_path / f"{stream_id}.jpg",
        thumbnail_public_url=lambda stream_id: f"/thumbs/{stream_id}.jpg",
        public_thumbnail_payload=lambda record: {k: v for k, v in record.items() if not k.startswith("_")},
        runtime_timestamp_to_iso=lambda ts: "2025-01-01T00:00:00Z",
        stream_runtime_lock=_Guard(),
        stream_runtime_state=runtime_state,
        safe_emit=lambda *args, **kwargs: None,
        playback_manager_getter=lambda: None,
        media_mode_choices={"image", "video", "ai", "livestream", "picsum"},
        media_mode_livestream="livestream",
        media_mode_video="video",
        media_mode_ai="ai",
        media_mode_picsum="picsum",
    )


def test_thumbnail_service_computes_snapshot(tmp_path: Path) -> None:
    service = _build_service(tmp_path)

    snapshot = service.compute_thumbnail_snapshot("stream1")

    assert snapshot["stream_id"] == "stream1"
    assert snapshot["kind"] == "image"
    assert snapshot["media_mode"] == "image"


def test_thumbnail_service_refresh_returns_placeholder_payload(tmp_path: Path) -> None:
    (tmp_path / "image.jpg").write_bytes(b"img")
    service = _build_service(tmp_path)

    payload = service.refresh_stream_thumbnail("stream1", force=True)

    assert payload["badge"] == "Image"
