from __future__ import annotations

import logging
import threading
from pathlib import Path
from types import SimpleNamespace

from echomosaic_app.services.media_catalog import MediaCatalogService


def _build_service(tmp_path: Path) -> tuple[MediaCatalogService, dict[str, dict[str, object]]]:
    root_path = tmp_path / "media"
    root_path.mkdir()
    cache: dict[str, dict[str, object]] = {}
    root = SimpleNamespace(alias="root", path=root_path, library="media")
    service = MediaCatalogService(
        image_cache=cache,
        image_cache_lock=threading.Lock(),
        logger=logging.getLogger("test-media-catalog"),
        normalize_library_key=lambda value, default="media": value or default,
        split_virtual_media_path=lambda value: (value.split("/", 1)[0], value.split("/", 1)[1] if "/" in value else ""),
        build_virtual_media_path=lambda alias, relative="": f"{alias}/{relative}".rstrip("/"),
        resolve_virtual_media_path=lambda value: root_path / (value.split("/", 1)[1] if "/" in value else ""),
        library_roots=lambda library: [root],
        should_ignore_media_name=lambda name: name.startswith("."),
        media_root_lookup={"root": root},
        media_extensions={".jpg", ".png", ".mp4"},
        video_extensions={".mp4"},
        media_library_default="media",
        ai_media_library="ai",
    )
    return service, cache


def test_media_catalog_service_refreshes_and_lists_media(tmp_path: Path) -> None:
    service, cache = _build_service(tmp_path)
    (tmp_path / "media" / "photo.jpg").write_bytes(b"jpg")
    (tmp_path / "media" / "clip.mp4").write_bytes(b"mp4")

    images = service.refresh_image_cache("all", library="media")
    media = service.list_media("all", library="media")

    assert images == ["root/photo.jpg"]
    assert [item["path"] for item in media] == ["root/clip.mp4", "root/photo.jpg"]
    assert "media::all" in cache


def test_media_catalog_service_invalidate_refreshes_parent_folder(tmp_path: Path) -> None:
    service, cache = _build_service(tmp_path)
    nested = tmp_path / "media" / "nested"
    nested.mkdir()
    (nested / "photo.jpg").write_bytes(b"jpg")

    service.refresh_image_cache("root/nested", library="media")
    assert "media::root/nested" in cache

    (nested / "new.jpg").write_bytes(b"jpg")
    service.invalidate_media_cache("root/nested/new.jpg", library="media")

    refreshed = service.list_images("root/nested", library="media")
    assert refreshed == ["root/nested/new.jpg", "root/nested/photo.jpg"]


def test_media_catalog_service_builds_folder_inventory(tmp_path: Path) -> None:
    service, _ = _build_service(tmp_path)
    nested = tmp_path / "media" / "nested"
    nested.mkdir()
    (nested / "photo.jpg").write_bytes(b"jpg")
    (nested / "clip.mp4").write_bytes(b"mp4")

    inventory = service.get_folder_inventory(library="media")

    assert inventory == [
        {"name": "all", "display_name": "all", "has_images": True, "has_videos": True},
        {"name": "root/nested", "display_name": "nested", "has_images": True, "has_videos": True},
    ]
