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


def test_operations_service_reads_release_update_info(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    def fetch_json(url: str):
        if url.endswith("/releases/latest"):
            return {
                "tag_name": "v1.1.0",
                "html_url": "https://example.com/release/v1.1.0",
                "name": "v1.1.0",
                "published_at": "1970-01-01T00:10:00Z",
            }
        if url.endswith("/releases/tags/v1.0.0"):
            return {
                "tag_name": "v1.0.0",
                "name": "v1.0.0",
                "published_at": "1970-01-01T00:00:00Z",
            }
        raise AssertionError(f"Unexpected URL {url}")
    service = _service(
        load_config=lambda: {
            "INSTALL_DIR": str(repo),
            "UPDATE_CHANNEL": "release",
            "REPO_SLUG": "tpersp/EchoMosaic",
            "INSTALLED_VERSION": "v1.0.0",
            "INSTALLED_COMMIT": "abc1234",
            "RELEASE_CHECK_INTERVAL_SECS": 3600,
        },
        fetch_json=fetch_json,
        time_fn=lambda: 1000.0,
    )

    info = service.read_update_info()

    assert info["channel"] == "release"
    assert info["installed_version"] == "v1.0.0"
    assert info["latest_version"] == "v1.1.0"
    assert info["latest_release_url"] == "https://example.com/release/v1.1.0"
    assert info["current_desc"] == "v1.0.0 (16 minutes ago)"
    assert info["remote_desc"] == "v1.1.0 (6 minutes ago)"
    assert info["update_available"] is True


def test_operations_service_force_refresh_bypasses_release_cache(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    latest_releases = iter(
        [
            {"tag_name": "v1.1.0", "html_url": "https://example.com/release/v1.1.0", "name": "v1.1.0"},
            {"tag_name": "v1.2.0", "html_url": "https://example.com/release/v1.2.0", "name": "v1.2.0"},
        ]
    )

    def fetch_json(url: str):
        if url.endswith("/releases/latest"):
            return next(latest_releases)
        if url.endswith("/releases/tags/v1.0.0"):
            return {"tag_name": "v1.0.0", "name": "v1.0.0"}
        raise AssertionError(f"Unexpected URL {url}")

    service = _service(
        load_config=lambda: {
            "INSTALL_DIR": str(repo),
            "UPDATE_CHANNEL": "release",
            "REPO_SLUG": "tpersp/EchoMosaic",
            "INSTALLED_VERSION": "v1.0.0",
            "INSTALLED_COMMIT": "abc1234",
            "RELEASE_CHECK_INTERVAL_SECS": 3600,
        },
        fetch_json=fetch_json,
        time_fn=lambda: 1000.0,
    )

    cached = service.read_update_info()
    refreshed = service.read_update_info(force_refresh=True)

    assert cached["latest_version"] == "v1.1.0"
    assert refreshed["latest_version"] == "v1.2.0"


def test_operations_service_release_status_survives_missing_installed_release_metadata(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()

    def fetch_json(url: str):
        if url.endswith("/releases/latest"):
            return {
                "tag_name": "v1.1.0",
                "html_url": "https://example.com/release/v1.1.0",
                "name": "v1.1.0",
                "published_at": "1970-01-01T00:10:00Z",
            }
        raise RuntimeError("not found")

    service = _service(
        load_config=lambda: {
            "INSTALL_DIR": str(repo),
            "UPDATE_CHANNEL": "release",
            "REPO_SLUG": "tpersp/EchoMosaic",
            "INSTALLED_VERSION": "v1.0.0",
            "INSTALLED_COMMIT": "abc1234",
            "RELEASE_CHECK_INTERVAL_SECS": 3600,
        },
        fetch_json=fetch_json,
        time_fn=lambda: 1000.0,
    )

    info = service.read_update_info()

    assert info["current_desc"] == "v1.0.0"
    assert info["remote_desc"] == "v1.1.0 (6 minutes ago)"


def test_operations_service_normalizes_stale_dev_release_channel_to_branch(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    service = _service(
        fetch_json=lambda url: {"tag_name": "v9.9.9"},
    )
    service._git_safe = lambda repo_path, cmd: None  # type: ignore[method-assign]

    info = service.read_update_info(
        {
            "INSTALL_DIR": "/home/doden/.local/share/echomosaic-dev",
            "SERVICE_NAME": "echomosaic-dev.service",
            "UPDATE_CHANNEL": "release",
            "UPDATE_BRANCH": "dev",
        }
    )

    assert info["channel"] == "branch"


def test_operations_service_reads_previous_update_from_history(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    (repo / "update_history.json").write_text(
        json.dumps(
            [
                {
                    "timestamp": "2026-03-25T00:00:00Z",
                    "branch": "main",
                    "from": "abc1234",
                    "to": "def5678",
                }
            ]
        ),
        encoding="utf-8",
    )
    service = _service(
        load_config=lambda: {"INSTALL_DIR": str(repo), "UPDATE_CHANNEL": "branch", "UPDATE_BRANCH": "main"}
    )

    def git_safe(repo_path, cmd):
        joined = " ".join(cmd)
        if "rev-parse HEAD" in joined:
            return "def5678"
        if "rev-parse origin/main" in joined:
            return "def5678"
        if "git log -1 --pretty=%h %s (%cr)" in joined:
            return "def5678 Current commit (just now)"
        if "git log -1 origin/main --pretty=%h %s (%cr)" in joined:
            return "def5678 Current commit (just now)"
        if "git log -1 abc1234 --pretty=%h %s (%cr)" in joined:
            return "abc1234 Previous commit (1 day ago)"
        return None

    service._git_safe = git_safe  # type: ignore[method-assign]

    info = service.read_update_info()

    assert info["previous_commit"] == "abc1234"
    assert info["previous_desc"] == "abc1234 Previous commit (1 day ago)"


def test_operations_service_matches_previous_entry_to_current_commit(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    (repo / "update_history.json").write_text(
        json.dumps(
            [
                {
                    "timestamp": "2026-03-24T00:00:00Z",
                    "branch": "main",
                    "from": "abc1234",
                    "to": "def5678",
                },
                {
                    "timestamp": "2026-03-25T00:00:00Z",
                    "branch": "main",
                    "from": "def5678",
                    "to": "fedcba9",
                },
            ]
        ),
        encoding="utf-8",
    )
    service = _service(
        load_config=lambda: {
            "INSTALL_DIR": str(repo),
            "UPDATE_CHANNEL": "branch",
            "UPDATE_BRANCH": "main",
        }
    )

    def git_safe(repo_path, cmd):
        joined = " ".join(cmd)
        if "rev-parse HEAD" in joined:
            return "def5678"
        if "rev-parse origin/main" in joined:
            return "def5678"
        if "git log -1 --pretty=%h %s (%cr)" in joined:
            return "def5678 Current commit (just now)"
        if "git log -1 origin/main --pretty=%h %s (%cr)" in joined:
            return "def5678 Current commit (just now)"
        if "git log -1 abc1234 --pretty=%h %s (%cr)" in joined:
            return "abc1234 Previous commit (2 days ago)"
        return None

    service._git_safe = git_safe  # type: ignore[method-assign]

    info = service.read_update_info()

    assert info["previous_commit"] == "abc1234"
    assert info["previous_desc"] == "abc1234 Previous commit (2 days ago)"


def test_operations_service_falls_back_to_orig_head_for_previous_commit(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    service = _service(
        load_config=lambda: {
            "INSTALL_DIR": str(repo),
            "UPDATE_CHANNEL": "release",
            "INSTALLED_VERSION": "v1.0.1",
            "INSTALLED_COMMIT": "def5678",
            "REPO_SLUG": "tpersp/EchoMosaic",
        },
        fetch_json=lambda url: {"tag_name": "v1.0.1", "name": "v1.0.1", "published_at": "1970-01-01T00:00:00Z"},
        time_fn=lambda: 1200.0,
    )

    def git_safe(repo_path, cmd):
        joined = " ".join(cmd)
        if "rev-parse ORIG_HEAD" in joined:
            return "abc1234"
        if "git log -1 abc1234 --pretty=%h %s (%cr)" in joined:
            return "abc1234 Previous commit (1 day ago)"
        return None

    service._git_safe = git_safe  # type: ignore[method-assign]

    info = service.read_update_info()

    assert info["previous_commit"] == "abc1234"
    assert info["previous_desc"] == "abc1234 Previous commit (1 day ago)"
