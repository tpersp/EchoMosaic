from __future__ import annotations

from pathlib import Path

import config_manager
from echomosaic_app.config_runtime import build_media_runtime


def test_build_media_runtime_dedupes_aliases_and_provides_primary_roots(tmp_path: Path) -> None:
    media_one = tmp_path / "left" / "photos"
    media_two = tmp_path / "right" / "photos"
    ai_media = tmp_path / "ai_media"
    media_one.mkdir(parents=True)
    media_two.mkdir(parents=True)
    ai_media.mkdir()

    runtime = build_media_runtime(
        config={
            "MEDIA_PATHS": [str(media_one), str(media_two)],
            "AI_MEDIA_PATHS": [str(ai_media)],
            "MEDIA_ALLOWED_EXTS": [".png"],
        },
        media_library_default="media",
        ai_media_library="ai",
        thumbnail_subdir="_thumbnails",
        internal_media_dirs={"_thumbnails", "_ai_temp"},
    )

    assert [root.alias for root in runtime.standard_media_roots] == ["photos", "photos-2"]
    assert runtime.primary_media_root.alias == "photos"
    assert runtime.primary_ai_media_root.alias == "ai-media"
    assert runtime.media_allowed_exts == [".png"]


def test_build_media_runtime_creates_default_roots_when_config_paths_missing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    runtime = build_media_runtime(
        config={"MEDIA_PATHS": [], "AI_MEDIA_PATHS": []},
        media_library_default="media",
        ai_media_library="ai",
        thumbnail_subdir="_thumbnails",
        internal_media_dirs={"_thumbnails", "_ai_temp"},
    )

    assert isinstance(runtime.primary_media_root, config_manager.MediaRoot)
    assert isinstance(runtime.primary_ai_media_root, config_manager.MediaRoot)
    assert runtime.primary_media_root.path.exists()
    assert runtime.primary_ai_media_root.path.exists()
