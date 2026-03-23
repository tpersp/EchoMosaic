"""Utilities for preserving user-specific data during application updates."""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

PERSISTENT_FILES = ("settings.json", "config.json", "update_history.json", ".env")
PERSISTENT_DIRS = ("backups", "restorepoints")
MEDIA_BACKUP_ROOT = "repo_media"
MEDIA_MANIFEST_NAME = "repo_media_manifest.json"
INTERNAL_MEDIA_DIRS = {"_thumbnails", "_ai_temp", "_thumbnails_cache"}


def _resolve_repo_path(repo_path: str) -> Path:
    path = Path(repo_path).resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Repository path not found: {repo_path}")
    return path


def _load_json_file(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning("Unable to parse JSON from %s", path)
        return {}
    return data if isinstance(data, dict) else {}


def _normalize_media_paths(paths: Any) -> List[str]:
    if isinstance(paths, str):
        items: Iterable[Any] = [paths]
    elif isinstance(paths, (list, tuple)):
        items = paths
    else:
        items = []

    normalized: List[str] = []
    seen: set[str] = set()
    for raw in items:
        text = str(raw).strip() if raw is not None else ""
        if not text or text in seen:
            continue
        seen.add(text)
        normalized.append(text)
    return normalized


def _load_library_paths(repo: Path, key: str, default: str) -> List[str]:
    config_data = _load_json_file(repo / "config.json")
    default_data = _load_json_file(repo / "config.default.json")
    paths = _normalize_media_paths(config_data.get(key))
    if paths:
        return paths
    return _normalize_media_paths(default_data.get(key)) or [default]


def _load_media_paths(repo: Path) -> List[str]:
    return _load_library_paths(repo, "MEDIA_PATHS", "./media")


def _load_ai_media_paths(repo: Path) -> List[str]:
    return _load_library_paths(repo, "AI_MEDIA_PATHS", "./ai_media")


def _resolve_media_path(repo: Path, raw_path: str) -> Optional[Path]:
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = repo / candidate
    try:
        return candidate.resolve(strict=False)
    except OSError:
        logger.warning("Unable to resolve media path %s", candidate)
        return None


def _is_within_repo(repo: Path, target: Path) -> bool:
    try:
        target.relative_to(repo)
        return True
    except ValueError:
        return False


def _ignore_internal_media_dirs(_dir: str, names: List[str]) -> List[str]:
    return [name for name in names if name in INTERNAL_MEDIA_DIRS]


def _backup_repo_media_dirs(repo: Path, temp_dir: Path) -> None:
    manifest: List[Dict[str, str]] = []
    media_root = temp_dir / MEDIA_BACKUP_ROOT

    all_paths = _load_media_paths(repo) + _load_ai_media_paths(repo)
    seen_paths: set[str] = set()
    for raw_path in all_paths:
        raw_key = str(raw_path).strip()
        if not raw_key or raw_key in seen_paths:
            continue
        seen_paths.add(raw_key)
        resolved = _resolve_media_path(repo, raw_path)
        if resolved is None or not resolved.exists() or not resolved.is_dir():
            continue
        if not _is_within_repo(repo, resolved):
            continue

        rel_path = resolved.relative_to(repo)
        dest = media_root / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            resolved,
            dest,
            dirs_exist_ok=True,
            ignore=_ignore_internal_media_dirs,
        )
        manifest.append({
            "raw_path": raw_path,
            "repo_relative_path": rel_path.as_posix(),
        })

    if manifest:
        _write_json(temp_dir / MEDIA_MANIFEST_NAME, manifest)


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

    _backup_repo_media_dirs(repo, temp_dir)

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


def _restore_env_file(repo: Path, backup: Path) -> None:
    source = backup / ".env"
    if not source.is_file():
        return
    dest = repo / ".env"
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


def _restore_repo_media_dirs(repo: Path, backup: Path) -> None:
    manifest_path = backup / MEDIA_MANIFEST_NAME
    media_root = backup / MEDIA_BACKUP_ROOT
    if not manifest_path.is_file() or not media_root.exists():
        return

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning("Media backup manifest %s is not valid JSON", manifest_path)
        return
    if not isinstance(manifest, list):
        logger.warning("Media backup manifest %s is not a list", manifest_path)
        return

    for item in manifest:
        if not isinstance(item, dict):
            continue
        relative_text = str(item.get("repo_relative_path") or "").strip()
        if not relative_text:
            continue
        source = media_root / relative_text
        dest = repo / relative_text
        if not source.exists():
            continue
        if source.is_dir():
            if dest.exists() and not dest.is_dir():
                dest.unlink()
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(
                source,
                dest,
                dirs_exist_ok=True,
                ignore=_ignore_internal_media_dirs,
            )
        else:
            if dest.exists() and dest.is_dir():
                shutil.rmtree(dest, ignore_errors=True)
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
        _restore_repo_media_dirs(repo, backup)
        _restore_settings(repo, backup)
        _restore_update_history(repo, backup)
        _restore_env_file(repo, backup)
        _restore_config(repo, backup)
    finally:
        if cleanup:
            try:
                shutil.rmtree(backup, ignore_errors=True)
            except Exception:
                logger.debug("Cleanup of backup dir %s failed", backup)
