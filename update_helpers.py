"""Utilities for preserving user-specific data during application updates."""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

PERSISTENT_FILES = ("settings.json", "config.json", "update_history.json")
PERSISTENT_DIRS = ("backups", "restorepoints")


def _resolve_repo_path(repo_path: str) -> Path:
    path = Path(repo_path).resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Repository path not found: {repo_path}")
    return path


def backup_user_state(repo_path: str) -> str:
    """Copy user-managed data out of the repository and return the temp path."""
    repo = _resolve_repo_path(repo_path)
    temp_dir = Path(tempfile.mkdtemp(prefix="echomosaic-update-"))

    for entry in PERSISTENT_FILES:
        source = repo / entry
        if not source.is_file():
            continue
        dest = temp_dir / entry
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)

    for entry in PERSISTENT_DIRS:
        source = repo / entry
        if not source.exists():
            continue
        dest = temp_dir / entry
        if source.is_dir():
            shutil.copytree(source, dest)
        else:
            # Should not happen, but be defensive.
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)

    return str(temp_dir)


def _deep_merge(base: Any, override: Any) -> Any:
    if isinstance(base, dict) and isinstance(override, dict):
        merged: Dict[Any, Any] = dict(base)
        for key, value in override.items():
            if key in merged:
                merged[key] = _deep_merge(merged[key], value)
            else:
                merged[key] = value
        return merged
    return override


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")


def _restore_settings(repo: Path, backup: Path) -> None:
    source = backup / "settings.json"
    if not source.is_file():
        return
    dest = repo / "settings.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)


def _restore_update_history(repo: Path, backup: Path) -> None:
    source = backup / "update_history.json"
    if not source.is_file():
        return
    dest = repo / "update_history.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)


def _restore_config(repo: Path, backup: Path) -> None:
    source = backup / "config.json"
    if not source.is_file():
        return
    dest = repo / "config.json"
    try:
        with source.open("r", encoding="utf-8") as handle:
            user_config = json.load(handle)
    except json.JSONDecodeError:
        logger.warning("User config backup is not valid JSON; keeping repository version")
        return
    if dest.is_file():
        try:
            with dest.open("r", encoding="utf-8") as handle:
                base_config = json.load(handle)
        except json.JSONDecodeError:
            base_config = {}
    else:
        base_config = {}
    if not isinstance(user_config, dict):
        logger.warning("User config backup is not a dict; keeping repository version")
        return
    merged = _deep_merge(base_config, user_config)
    _write_json(dest, merged)


def _restore_dirs(repo: Path, backup: Path) -> None:
    for entry in PERSISTENT_DIRS:
        source = backup / entry
        if not source.exists():
            continue
        dest = repo / entry
        if dest.exists():
            if dest.is_dir():
                shutil.rmtree(dest)
            else:
                dest.unlink()
        if source.is_dir():
            shutil.copytree(source, dest)
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)


def restore_user_state(repo_path: str, backup_dir: str, *, cleanup: bool = False) -> None:
    """Restore previously backed-up user data into the repository."""
    if not backup_dir:
        return
    backup = Path(backup_dir)
    if not backup.exists():
        return
    repo = _resolve_repo_path(repo_path)
    try:
        _restore_dirs(repo, backup)
        _restore_settings(repo, backup)
        _restore_update_history(repo, backup)
        _restore_config(repo, backup)
    finally:
        if cleanup:
            try:
                shutil.rmtree(backup, ignore_errors=True)
            except Exception:
                logger.debug("Cleanup of backup dir %s failed", backup)
