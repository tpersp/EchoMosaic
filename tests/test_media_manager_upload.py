from __future__ import annotations

from io import BytesIO
from pathlib import Path

from werkzeug.datastructures import FileStorage

from config_manager import MediaRoot
from media_manager import MediaManager, MediaManagerError


def _manager(root_path: Path) -> MediaManager:
    return MediaManager(
        [MediaRoot(alias="media", path=root_path, display_name="Media", library="media")],
        allowed_exts=[".png", ".jpg", ".jpeg", ".webp", ".gif", ".mp4", ".webm", ".mkv"],
        max_upload_mb=50,
        preview_enabled=False,
    )


def test_media_manager_upload_preserves_relative_folder_structure(tmp_path: Path) -> None:
    root = tmp_path / "media"
    destination = root / "viewers"
    destination.mkdir(parents=True)
    manager = _manager(root)

    files = [
      FileStorage(stream=BytesIO(b"first"), filename="image1.png"),
      FileStorage(stream=BytesIO(b"second"), filename="image2.png"),
    ]

    saved = manager.upload(
        "media:/viewers",
        files,
        relative_paths=["collection/image1.png", "collection/nested/image2.png"],
    )

    assert (destination / "collection" / "image1.png").read_bytes() == b"first"
    assert (destination / "collection" / "nested" / "image2.png").read_bytes() == b"second"
    assert saved[0]["path"].endswith("viewers/collection/image1.png")
    assert saved[1]["path"].endswith("viewers/collection/nested/image2.png")


def test_media_manager_upload_rejects_escaping_relative_folder_paths(tmp_path: Path) -> None:
    root = tmp_path / "media"
    destination = root / "viewers"
    destination.mkdir(parents=True)
    manager = _manager(root)

    file = FileStorage(stream=BytesIO(b"oops"), filename="image.png")

    try:
        manager.upload("media:/viewers", [file], relative_paths=["../escape/image.png"])
    except MediaManagerError as exc:
        assert exc.code == "invalid_name"
    else:
        raise AssertionError("expected MediaManagerError")
