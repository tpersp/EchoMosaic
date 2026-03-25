"""Update, rollback, and restore-point service logic."""

from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import config_manager


class UpdateAlreadyRunningError(RuntimeError):
    """Raised when an update is already active."""


class OperationsService:
    def __init__(
        self,
        *,
        load_config: Callable[[], Dict[str, Any]],
        backup_dirname: str,
        restore_point_dirname: str,
        restore_point_metadata_file: str,
        max_restore_points: int,
        settings_file: str,
        config_file: str,
        set_update_job_active: Callable[[bool], bool],
        run_update_job: Callable[[str, str, str, Optional[str]], None],
        logger,
        fetch_json: Optional[Callable[[str], Any]] = None,
        time_fn: Optional[Callable[[], float]] = None,
    ) -> None:
        self.load_config = load_config
        self.backup_dirname = backup_dirname
        self.restore_point_dirname = restore_point_dirname
        self.restore_point_metadata_file = restore_point_metadata_file
        self.max_restore_points = max_restore_points
        self.settings_file = settings_file
        self.config_file = config_file
        self.set_update_job_active = set_update_job_active
        self.run_update_job = run_update_job
        self.logger = logger
        self.fetch_json = fetch_json or self._fetch_json
        self.time_fn = time_fn or time.time
        self._release_status_cache: Dict[str, Dict[str, Any]] = {}

    def _fetch_json(self, url: str) -> Any:
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "application/vnd.github+json",
                "User-Agent": "EchoMosaic-UpdateChecker/1.0",
            },
        )
        with urllib.request.urlopen(request, timeout=8) as response:
            payload = response.read().decode("utf-8")
        return json.loads(payload)

    def _git_safe(self, repo_path: str, cmd: List[str]) -> Optional[str]:
        try:
            return subprocess.check_output(cmd, cwd=repo_path, stderr=subprocess.STDOUT).decode().strip()
        except Exception:
            return None

    @staticmethod
    def _normalize_version(value: Any) -> str:
        return str(value or "").strip()

    @staticmethod
    def _version_key(value: str) -> Tuple[int, ...]:
        matches = re.findall(r"\d+", value or "")
        if not matches:
            return tuple()
        return tuple(int(part) for part in matches)

    def _read_release_status(self, cfg: Dict[str, Any], repo_path: str, *, force_refresh: bool = False) -> Dict[str, Any]:
        repo_slug = str(cfg.get("REPO_SLUG") or "tpersp/EchoMosaic").strip()
        cache_ttl = max(60, int(cfg.get("RELEASE_CHECK_INTERVAL_SECS") or 3600))
        cached = self._release_status_cache.get(repo_slug)
        now = float(self.time_fn())
        if not force_refresh and cached and (now - cached.get("fetched_at", 0.0)) < cache_ttl:
            return dict(cached["payload"])

        installed_version = self._normalize_version(
            cfg.get("INSTALLED_VERSION")
            or self._git_safe(repo_path, ["git", "describe", "--tags", "--exact-match", "HEAD"])
            or self._git_safe(repo_path, ["git", "describe", "--tags", "--abbrev=0"])
        )
        installed_commit = self._normalize_version(
            cfg.get("INSTALLED_COMMIT")
            or self._git_safe(repo_path, ["git", "rev-parse", "HEAD"])
        )
        payload: Dict[str, Any] = {
            "channel": "release",
            "repo_slug": repo_slug,
            "installed_version": installed_version,
            "installed_commit": installed_commit,
            "current_desc": installed_version or (installed_commit[:7] if installed_commit else ""),
            "update_available": False,
            "latest_version": "",
            "latest_release_url": "",
            "release_check_ok": False,
        }
        api_url = f"https://api.github.com/repos/{repo_slug}/releases/latest"
        try:
            release = self.fetch_json(api_url) or {}
            latest_version = self._normalize_version(release.get("tag_name"))
            payload["latest_version"] = latest_version
            payload["latest_release_url"] = self._normalize_version(release.get("html_url"))
            payload["release_name"] = self._normalize_version(release.get("name")) or latest_version
            payload["remote_desc"] = payload["release_name"]
            payload["release_check_ok"] = True
            if latest_version:
                if installed_version:
                    payload["update_available"] = self._version_key(latest_version) > self._version_key(installed_version)
                else:
                    payload["update_available"] = True
        except Exception as exc:
            self.logger.warning("Unable to query latest release for %s: %s", repo_slug, exc)
            payload["release_error"] = str(exc)

        self._release_status_cache[repo_slug] = {"fetched_at": now, "payload": dict(payload)}
        return payload

    def repo_path_from_config(self, cfg: Optional[Dict[str, Any]] = None) -> str:
        cfg = cfg or self.load_config()
        configured = str(cfg.get("INSTALL_DIR") or "").strip()
        candidates: List[Path] = []

        if configured:
            candidates.append(Path(configured).expanduser())

        candidates.append(Path(__file__).resolve().parents[2])
        candidates.append(Path.cwd())

        for candidate in candidates:
            try:
                resolved = candidate.resolve(strict=False)
            except OSError:
                continue
            if resolved.is_dir() and (resolved / ".git").exists():
                return resolved.as_posix()

        if configured:
            return str(Path(configured).expanduser())
        return Path(__file__).resolve().parents[2].as_posix()

    def restart_configured_service(self, service_name: str) -> None:
        commands = [
            ["systemctl", "--user", "restart", service_name],
            ["sudo", "systemctl", "restart", service_name],
        ]
        for command in commands:
            try:
                subprocess.Popen(command)
                return
            except OSError:
                continue

    def _restore_points_root(self, repo_path: str) -> str:
        return os.path.join(repo_path, self.backup_dirname, self.restore_point_dirname)

    def _slugify_restore_label(self, label: str) -> str:
        cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "-", label.strip())
        cleaned = re.sub(r"-{2,}", "-", cleaned).strip("-_")
        if not cleaned:
            cleaned = f"restore-{secrets.token_hex(3)}"
        return cleaned.lower()[:60]

    def _load_restore_point_metadata(self, point_path: str) -> Optional[Dict[str, Any]]:
        metadata_path = os.path.join(point_path, self.restore_point_metadata_file)
        try:
            with open(metadata_path, "r") as metadata_file:
                return json.load(metadata_file)
        except Exception:
            return None

    def serialize_restore_point(self, point_id: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        commit = metadata.get("commit")
        return {
            "id": point_id,
            "label": metadata.get("label") or point_id,
            "commit": commit,
            "short_commit": (commit[:7] if isinstance(commit, str) else None),
            "created_at": metadata.get("created_at"),
            "last_restored_at": metadata.get("last_restored_at"),
            "branch": metadata.get("branch"),
        }

    def list_restore_points(self, repo_path: str) -> List[Dict[str, Any]]:
        root = self._restore_points_root(repo_path)
        items: List[Dict[str, Any]] = []
        if not os.path.isdir(root):
            return items
        for entry in os.scandir(root):
            if not entry.is_dir():
                continue
            metadata = self._load_restore_point_metadata(entry.path)
            if not metadata:
                continue
            items.append(self.serialize_restore_point(entry.name, metadata))
        items.sort(key=lambda item: item.get("created_at") or "", reverse=True)
        return items

    def _prune_restore_points(self, root: str) -> None:
        if self.max_restore_points <= 0 or not os.path.isdir(root):
            return
        entries: List[Tuple[str, str]] = []
        for item in os.scandir(root):
            if not item.is_dir():
                continue
            metadata = self._load_restore_point_metadata(item.path)
            if not metadata:
                continue
            created = metadata.get("created_at") or ""
            entries.append((item.path, created))
        if len(entries) <= self.max_restore_points:
            return
        entries.sort(key=lambda pair: pair[1] or "")
        for path_to_remove, _ in entries[:-self.max_restore_points]:
            try:
                shutil.rmtree(path_to_remove)
            except Exception as exc:
                self.logger.warning("Failed to prune restore point %s: %s", path_to_remove, exc)

    def _allocate_restore_point_dir(self, repo_path: str, label: str) -> Tuple[str, str]:
        root = self._restore_points_root(repo_path)
        os.makedirs(root, exist_ok=True)
        slug = self._slugify_restore_label(label)
        now = datetime.utcnow()
        dir_stamp = now.strftime("%Y%m%d-%H%M%S")
        candidate = f"{dir_stamp}-{slug}"
        while os.path.exists(os.path.join(root, candidate)):
            candidate = f"{dir_stamp}-{slug}-{secrets.token_hex(2)}"
        point_path = os.path.join(root, candidate)
        os.makedirs(point_path, exist_ok=True)
        return candidate, point_path

    def _save_restore_point_metadata(self, point_path: str, metadata: Dict[str, Any]) -> None:
        metadata_path = os.path.join(point_path, self.restore_point_metadata_file)
        with open(metadata_path, "w") as metadata_file:
            json.dump(metadata, metadata_file, indent=2)

    def create_restore_point(self, repo_path: str, label: str) -> Dict[str, Any]:
        now = datetime.utcnow().replace(microsecond=0)
        timestamp = now.isoformat() + "Z"
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_path,
                stderr=subprocess.STDOUT,
            ).decode().strip()
        except FileNotFoundError as exc:
            raise RuntimeError("Git executable not found") from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"Unable to determine current commit: {exc}") from exc
        try:
            branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=repo_path,
                stderr=subprocess.STDOUT,
            ).decode().strip()
        except Exception:
            branch = None
        point_id, point_path = self._allocate_restore_point_dir(repo_path, label)
        files: List[Dict[str, str]] = []
        for source_name in (self.settings_file, self.config_file):
            source_path = os.path.join(repo_path, source_name)
            if not os.path.isfile(source_path):
                continue
            dest_name = os.path.basename(source_name)
            dest_path = os.path.join(point_path, dest_name)
            try:
                shutil.copy2(source_path, dest_path)
                files.append({"filename": dest_name, "destination": source_name})
            except OSError as exc:
                self.logger.warning(
                    "Failed to copy %s into restore point %s: %s",
                    source_name,
                    point_id,
                    exc,
                )
        metadata = {
            "label": label,
            "commit": commit,
            "branch": branch,
            "created_at": timestamp,
            "files": files,
        }
        self._save_restore_point_metadata(point_path, metadata)
        self._prune_restore_points(self._restore_points_root(repo_path))
        return self.serialize_restore_point(point_id, metadata)

    def load_restore_point(self, repo_path: str, point_id: str) -> Tuple[str, Dict[str, Any]]:
        safe_id = os.path.basename(point_id.strip())
        if not safe_id:
            raise FileNotFoundError(point_id)
        point_path = os.path.join(self._restore_points_root(repo_path), safe_id)
        if not os.path.isdir(point_path):
            raise FileNotFoundError(point_id)
        metadata = self._load_restore_point_metadata(point_path)
        if not metadata:
            raise FileNotFoundError(point_id)
        metadata["id"] = safe_id
        return point_path, metadata

    def restore_files_from_metadata(self, repo_path: str, point_path: str, metadata: Dict[str, Any]) -> None:
        for entry in metadata.get("files") or []:
            if isinstance(entry, dict):
                filename = entry.get("filename")
                destination = entry.get("destination") or filename
            else:
                filename = str(entry)
                destination = filename
            if not filename or not destination:
                continue
            source = os.path.join(point_path, filename)
            if not os.path.isfile(source):
                continue
            dest = os.path.join(repo_path, destination)
            dest_dir = os.path.dirname(dest)
            if dest_dir and not os.path.isdir(dest_dir):
                os.makedirs(dest_dir, exist_ok=True)
            try:
                shutil.copy2(source, dest)
            except OSError as exc:
                self.logger.warning(
                    "Failed to restore %s from restore point %s: %s",
                    destination,
                    metadata.get("id"),
                    exc,
                )

    def delete_restore_point(self, repo_path: str, point_id: str) -> bool:
        safe_id = os.path.basename(point_id.strip())
        if not safe_id:
            return False
        point_path = os.path.join(self._restore_points_root(repo_path), safe_id)
        if not os.path.isdir(point_path):
            return False
        shutil.rmtree(point_path)
        return True

    def restore_point(self, repo_path: str, service_name: str, point_id: str) -> Dict[str, Any]:
        point_path, metadata = self.load_restore_point(repo_path, point_id)
        commit = metadata.get("commit")
        if not commit:
            raise ValueError("Restore point missing commit")
        try:
            subprocess.check_call(["git", "reset", "--hard", commit], cwd=repo_path)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"Git rollback failed: {exc}") from exc
        self.restore_files_from_metadata(repo_path, point_path, metadata)
        metadata["last_restored_at"] = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        try:
            self._save_restore_point_metadata(point_path, metadata)
        except Exception:
            pass
        self.restart_configured_service(service_name)
        return self.serialize_restore_point(metadata["id"], metadata)

    def start_update(self, cfg: Dict[str, Any], socket_id: Optional[str] = None) -> Dict[str, Any]:
        repo_path = self.repo_path_from_config(cfg)
        if not os.path.isdir(repo_path):
            raise ValueError(f"Repository path '{repo_path}' not found. Check INSTALL_DIR in config.json")
        update_script = os.path.join(repo_path, "update.sh")
        if not os.path.isfile(update_script):
            raise ValueError(f"Update script not found at '{update_script}'.")
        if not self.set_update_job_active(True):
            raise UpdateAlreadyRunningError("An update is already in progress.")
        service_name = str(cfg.get("SERVICE_NAME", "echomosaic.service"))
        worker = threading.Thread(
            target=self.run_update_job,
            args=(repo_path, update_script, service_name, socket_id or None),
            daemon=True,
            name="echomosaic-update-job",
        )
        worker.start()
        return {"status": "started", "service_name": service_name}

    def read_update_info(self, cfg: Optional[Dict[str, Any]] = None, *, force_refresh: bool = False) -> Dict[str, Any]:
        cfg = cfg or self.load_config()
        cfg, _ = config_manager.normalize_update_configuration(cfg)
        repo_path = self.repo_path_from_config(cfg)
        update_channel = str(cfg.get("UPDATE_CHANNEL") or "branch").strip().lower()
        branch = cfg.get("UPDATE_BRANCH", "main")
        info = {"branch": branch, "channel": update_channel}

        if update_channel == "release":
            info.update(self._read_release_status(cfg, repo_path, force_refresh=force_refresh))
        else:
            current = self._git_safe(repo_path, ["git", "rev-parse", "HEAD"]) or ""
            current_short = self._git_safe(repo_path, ["git", "rev-parse", "--short", "HEAD"]) or ""
            current_desc = self._git_safe(repo_path, ["git", "log", "-1", "--pretty=%h %s (%cr)"]) or current_short
            _ = self._git_safe(repo_path, ["git", "fetch", "--quiet"])
            remote = self._git_safe(repo_path, ["git", "rev-parse", f"origin/{branch}"]) or ""
            remote_short = remote[:7] if remote else ""
            remote_desc = self._git_safe(repo_path, ["git", "log", "-1", f"origin/{branch}", "--pretty=%h %s (%cr)"]) or remote_short
            info.update(
                {
                    "current_commit": current,
                    "current_desc": current_desc,
                    "remote_commit": remote,
                    "remote_desc": remote_desc,
                    "installed_commit": self._normalize_version(cfg.get("INSTALLED_COMMIT") or current),
                    "installed_version": self._normalize_version(cfg.get("INSTALLED_VERSION") or branch),
                    "latest_version": self._normalize_version(cfg.get("INSTALLED_VERSION") or branch),
                    "update_available": bool(current and remote and current != remote),
                }
            )

        history_path = os.path.join(repo_path, "update_history.json")
        if os.path.exists(history_path):
            try:
                with open(history_path, "r") as history_file:
                    history = json.load(history_file)
                if history:
                    last = history[-1]
                    info["previous_commit"] = last.get("from")
                    previous = last.get("from")
                    if previous:
                        prev_desc = self._git_safe(
                            repo_path,
                            ["git", "log", "-1", previous, "--pretty=%h %s (%cr)"],
                        ) or previous[:7]
                    else:
                        prev_desc = None
                    info["previous_desc"] = prev_desc
            except Exception:
                pass
        return info

    def read_update_history(self, cfg: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        cfg = cfg or self.load_config()
        repo_path = self.repo_path_from_config(cfg)
        history_path = os.path.join(repo_path, "update_history.json")
        history = []
        if os.path.exists(history_path):
            try:
                with open(history_path, "r") as history_file:
                    history = json.load(history_file)
            except Exception:
                history = []

        def srun(cmd):
            try:
                return subprocess.check_output(cmd, cwd=repo_path, stderr=subprocess.STDOUT).decode().strip()
            except Exception:
                return None

        enriched = []
        for entry in history:
            frm = entry.get("from")
            to = entry.get("to")
            frm_desc = srun(["git", "log", "-1", frm, "--pretty=%h %s (%cr)"]) if frm else None
            to_desc = srun(["git", "log", "-1", to, "--pretty=%h %s (%cr)"]) if to else None
            enriched.append(
                {
                    "timestamp": entry.get("timestamp"),
                    "branch": entry.get("branch"),
                    "from": frm,
                    "to": to,
                    "from_desc": frm_desc or (frm[:7] if frm else None),
                    "to_desc": to_desc or (to[:7] if to else None),
                }
            )
        return enriched

    def rollback_to_restore_point(self, repo_path: str, service_name: str, restore_point_id: str) -> str:
        point_path, metadata = self.load_restore_point(repo_path, restore_point_id)
        commit = metadata.get("commit")
        if not commit:
            raise ValueError("Restore point is missing commit information.")
        try:
            subprocess.check_call(["git", "reset", "--hard", commit], cwd=repo_path)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"Rollback failed: {exc}") from exc
        self.restore_files_from_metadata(repo_path, point_path, metadata)
        metadata["last_restored_at"] = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        try:
            self._save_restore_point_metadata(point_path, metadata)
        except Exception:
            pass
        self.restart_configured_service(service_name)
        label = metadata.get("label") or restore_point_id
        short_commit = metadata.get("short_commit") or (commit[:7] if isinstance(commit, str) else commit)
        return f"Rolled back to restore point '{label}' ({short_commit}). Restarting service..."

    def rollback_to_previous_history(self, repo_path: str, service_name: str) -> str:
        history_path = os.path.join(repo_path, "update_history.json")
        try:
            with open(history_path, "r") as history_file:
                history = json.load(history_file)
        except Exception as exc:
            raise ValueError("No previous version to roll back to.") from exc
        if not history:
            raise ValueError("No previous version to roll back to.")
        target = history[-1].get("from")
        if not target:
            raise ValueError("History does not include a valid commit.")
        try:
            subprocess.check_call(["git", "reset", "--hard", target], cwd=repo_path)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"Rollback failed: {exc}") from exc
        self.restart_configured_service(service_name)
        return f"Rolled back to {target[:7]}. Restarting service..."
