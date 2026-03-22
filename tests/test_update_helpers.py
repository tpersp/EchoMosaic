from __future__ import annotations

import json
from pathlib import Path

import update_helpers


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def test_backup_and_restore_preserves_user_files_and_repo_local_media(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    media_dir = repo / "media"
    ai_media_dir = repo / "ai_media"
    media_dir.mkdir()
    ai_media_dir.mkdir()
    (repo / "backups").mkdir()
    (repo / "restorepoints").mkdir()

    _write_json(
        repo / "config.default.json",
        {
            "MEDIA_PATHS": ["./media"],
            "AI_MEDIA_PATHS": ["./ai_media"],
            "logging": {"level": "INFO", "retention_days": 7},
        },
    )
    _write_json(
        repo / "config.json",
        {
            "MEDIA_PATHS": ["./media"],
            "AI_MEDIA_PATHS": ["./ai_media"],
            "SERVICE_NAME": "custom.service",
            "logging": {"level": "DEBUG"},
        },
    )
    _write_json(repo / "settings.json", {"stream1": {"label": "Original"}})
    _write_json(repo / "update_history.json", [{"from": "abc1234", "to": "def5678"}])
    _write_json(repo / "backups" / "keep.json", {"saved": True})
    _write_json(repo / "restorepoints" / "rp1" / "metadata.json", {"id": "rp1"})

    (media_dir / "photo.jpg").write_text("photo-v1", encoding="utf-8")
    (media_dir / "_thumbnails").mkdir()
    (media_dir / "_thumbnails" / "cached.jpg").write_text("ignore-me", encoding="utf-8")
    (ai_media_dir / "gen.png").write_text("ai-v1", encoding="utf-8")

    backup_dir = Path(update_helpers.backup_user_state(str(repo)))

    _write_json(repo / "config.json", {"SERVICE_NAME": "replacement.service"})
    _write_json(repo / "settings.json", {"stream1": {"label": "Changed"}})
    _write_json(repo / "update_history.json", [])
    _write_json(repo / "backups" / "changed.json", {"saved": False})
    (repo / "restorepoints" / "rp1" / "metadata.json").write_text("{}", encoding="utf-8")
    (media_dir / "photo.jpg").write_text("photo-v2", encoding="utf-8")
    (ai_media_dir / "gen.png").write_text("ai-v2", encoding="utf-8")

    update_helpers.restore_user_state(str(repo), str(backup_dir), cleanup=True)

    restored_config = json.loads((repo / "config.json").read_text(encoding="utf-8"))
    restored_settings = json.loads((repo / "settings.json").read_text(encoding="utf-8"))
    restored_history = json.loads((repo / "update_history.json").read_text(encoding="utf-8"))

    assert restored_config["SERVICE_NAME"] == "custom.service"
    assert restored_config["logging"]["level"] == "DEBUG"
    assert restored_settings["stream1"]["label"] == "Original"
    assert restored_history[0]["from"] == "abc1234"
    assert json.loads((repo / "backups" / "keep.json").read_text(encoding="utf-8"))["saved"] is True
    assert json.loads((repo / "restorepoints" / "rp1" / "metadata.json").read_text(encoding="utf-8"))["id"] == "rp1"
    assert (media_dir / "photo.jpg").read_text(encoding="utf-8") == "photo-v1"
    assert (ai_media_dir / "gen.png").read_text(encoding="utf-8") == "ai-v1"
    assert (media_dir / "_thumbnails" / "cached.jpg").read_text(encoding="utf-8") == "ignore-me"
    assert not backup_dir.exists()


def test_backup_ignores_media_paths_outside_repo(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    external = tmp_path / "external-media"
    repo.mkdir()
    external.mkdir()

    _write_json(
        repo / "config.default.json",
        {
            "MEDIA_PATHS": [str(external)],
            "AI_MEDIA_PATHS": ["./ai_media"],
        },
    )
    _write_json(
        repo / "config.json",
        {
            "MEDIA_PATHS": [str(external)],
            "AI_MEDIA_PATHS": ["./ai_media"],
        },
    )
    (repo / "ai_media").mkdir()
    (external / "outside.jpg").write_text("external", encoding="utf-8")

    backup_dir = Path(update_helpers.backup_user_state(str(repo)))
    manifest_path = backup_dir / update_helpers.MEDIA_MANIFEST_NAME

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = []

    assert all(item["repo_relative_path"] != external.name for item in manifest)

    update_helpers.restore_user_state(str(repo), str(backup_dir), cleanup=True)
