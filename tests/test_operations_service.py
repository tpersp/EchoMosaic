from __future__ import annotations

import json
from pathlib import Path

from echomosaic_app.services.operations import OperationsService, UpdateAlreadyRunningError


def _service(**overrides):
    params = {
        "load_config": lambda: {},
        "backup_dirname": "_backup",
        "restore_point_dirname": "restore_points",
        "restore_point_metadata_file": "metadata.json",
        "max_restore_points": 5,
        "settings_file": "settings.json",
        "config_file": "config.json",
        "set_update_job_active": lambda active: True,
        "run_update_job": lambda repo_path, update_script, service_name, socket_id: None,
        "logger": type("L", (), {"warning": lambda *args, **kwargs: None})(),
    }
    params.update(overrides)
    return OperationsService(**params)


def test_operations_service_prefers_git_repo_from_config(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    service = _service(load_config=lambda: {"INSTALL_DIR": str(repo)})

    assert service.repo_path_from_config() == repo.as_posix()


def test_operations_service_reads_enriched_update_history(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    (repo / "update_history.json").write_text(
        json.dumps([{"timestamp": "2026-03-23T00:00:00Z", "branch": "main", "from": "abc1234", "to": "def5678"}]),
        encoding="utf-8",
    )
    service = _service(load_config=lambda: {"INSTALL_DIR": str(repo)})

    history = service.read_update_history()

    assert len(history) == 1
    assert history[0]["from"] == "abc1234"
    assert history[0]["to"] == "def5678"
    assert history[0]["from_desc"] == "abc1234"
    assert history[0]["to_desc"] == "def5678"


def test_operations_service_start_update_detects_conflict(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    (repo / "update.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    service = _service(
        load_config=lambda: {"INSTALL_DIR": str(repo)},
        set_update_job_active=lambda active: False,
    )

    try:
        service.start_update(service.load_config())
    except UpdateAlreadyRunningError as exc:
        assert "already in progress" in str(exc)
    else:
        raise AssertionError("expected UpdateAlreadyRunningError")
