from __future__ import annotations

import json
import os
from pathlib import Path

import config_manager


def test_ensure_config_file_adds_missing_defaults_without_overwriting_existing_values(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    default_path = tmp_path / "config.default.json"

    config_path.write_text(
        json.dumps(
            {
                "SERVICE_NAME": "custom.service",
                "logging": {"level": "DEBUG"},
            }
        ),
        encoding="utf-8",
    )
    default_path.write_text(
        json.dumps(
            {
                "SERVICE_NAME": "default.service",
                "UPDATE_BRANCH": "main",
                "logging": {"level": "INFO", "retention_days": 7},
            }
        ),
        encoding="utf-8",
    )

    merged = config_manager.ensure_config_file(config_path=config_path, default_path=default_path)
    saved = json.loads(config_path.read_text(encoding="utf-8"))

    assert merged["SERVICE_NAME"] == "custom.service"
    assert saved["SERVICE_NAME"] == "custom.service"
    assert saved["UPDATE_BRANCH"] == "main"
    assert saved["logging"]["level"] == "DEBUG"
    assert saved["logging"]["retention_days"] == 7


def test_load_config_applies_environment_overrides_for_media_paths(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "config.json"
    default_path = tmp_path / "config.default.json"
    env_path = tmp_path / ".env"

    default_path.write_text(
        json.dumps(
                {
                    "MEDIA_PATHS": ["./media"],
                    "AI_MEDIA_PATHS": ["./ai_media"],
                    "API_KEY": "",
                    "SERVICE_NAME": "echomosaic.service",
                }
            ),
        encoding="utf-8",
    )
    config_path.write_text(json.dumps({"SERVICE_NAME": "custom.service"}), encoding="utf-8")
    env_path.write_text("API_KEY=from-env-file\n", encoding="utf-8")

    media_a = tmp_path / "media_a"
    media_b = tmp_path / "media_b"
    ai_media = tmp_path / "ai_media_override"
    media_a.mkdir()
    media_b.mkdir()
    ai_media.mkdir()

    monkeypatch.setenv(
        "ECHOMOSAIC_MEDIA_PATHS",
        f"{media_a}{os.pathsep}{media_b}{os.pathsep}{media_a}",
    )
    monkeypatch.setenv("ECHOMOSAIC_AI_MEDIA_PATHS", str(ai_media))

    config = config_manager.load_config(
        config_path=config_path,
        default_path=default_path,
        env_path=env_path,
    )

    assert config["SERVICE_NAME"] == "custom.service"
    assert config["API_KEY"] == "from-env-file"
    assert config["MEDIA_PATHS"] == [str(media_a.resolve()), str(media_b.resolve())]
    assert config["AI_MEDIA_PATHS"] == [str(ai_media.resolve())]


def test_build_media_roots_generates_unique_aliases_for_duplicate_folder_names(tmp_path: Path) -> None:
    left = tmp_path / "left" / "photos"
    right = tmp_path / "right" / "photos"
    left.mkdir(parents=True)
    right.mkdir(parents=True)

    roots = config_manager.build_media_roots([str(left), str(right)])

    assert len(roots) == 2
    assert roots[0].alias == "photos"
    assert roots[1].alias == "photos-2"
    assert roots[0].display_name == "photos"
    assert roots[1].display_name == "photos"
