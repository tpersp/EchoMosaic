"""Media cache, folder inventory, and library catalog helpers."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union


class MediaCatalogService:
    def __init__(
        self,
        *,
        logger,
        image_cache,
        image_cache_lock,
        media_root_lookup,
        media_extensions,
        video_extensions,
        media_library_default: str,
        ai_media_library: str,
        library_roots,
        normalize_library_key,
        split_virtual_media_path,
        resolve_virtual_media_path,
        build_virtual_media_path,
        should_ignore_media_name,
        path_contains_nsfw,
    ) -> None:
        self.logger = logger
        self.image_cache = image_cache
        self.image_cache_lock = image_cache_lock
        self.media_root_lookup = media_root_lookup
        self.media_extensions = media_extensions
        self.video_extensions = video_extensions
        self.media_library_default = media_library_default
        self.ai_media_library = ai_media_library
        self.library_roots = library_roots
        self.normalize_library_key = normalize_library_key
        self.split_virtual_media_path = split_virtual_media_path
        self.resolve_virtual_media_path = resolve_virtual_media_path
        self.build_virtual_media_path = build_virtual_media_path
        self.should_ignore_media_name = should_ignore_media_name
        self.path_contains_nsfw = path_contains_nsfw

    def normalize_folder_key(self, folder: Optional[str]) -> str:
        if folder is None:
            return "all"
        normalized = str(folder).strip()
        if normalized in {"", "all", "."}:
            return "all"
        normalized = normalized.replace("\\", "/").strip("/")
        if ":/" in normalized:
            alias, remainder = normalized.split(":/", 1)
            if alias in self.media_root_lookup:
                remainder = remainder.strip("/")
                normalized = f"{alias}/{remainder}" if remainder else alias
        return normalized

    def cache_scope_key(self, folder_key: str, library: str) -> str:
        return f"{self.normalize_library_key(library)}::{folder_key}"

    def resolve_folder_path(self, folder_key: str, *, library: str) -> Optional[Tuple[Any, Path]]:
        if folder_key == "all":
            return None
        alias, relative = self.split_virtual_media_path(folder_key)
        root = self.media_root_lookup.get(alias)
        normalized_library = self.normalize_library_key(library)
        if root is None or root.library != normalized_library:
            return None
        target_dir = (root.path / relative).resolve()
        try:
            target_dir.relative_to(root.path.resolve())
        except ValueError:
            self.logger.debug("Rejected folder key '%s' because it escapes media root '%s'", folder_key, root.path)
            return None
        return root, target_dir

    def scan_root_for_cache(self, root, base_path: Path) -> Tuple[List[Dict[str, str]], Dict[str, Tuple[int, int]]]:
        dir_markers: Dict[str, Tuple[int, int]] = {}
        base_key = os.fspath(base_path)
        try:
            stat_info = os.stat(base_path)
            dir_markers[base_key] = (stat_info.st_mtime_ns, stat_info.st_ctime_ns)
        except FileNotFoundError:
            dir_markers[base_key] = (0, 0)
            return [], dir_markers
        except OSError:
            dir_markers[base_key] = (0, 0)
            return [], dir_markers

        media: List[Dict[str, str]] = []
        for walk_root, dirnames, files in os.walk(base_path):
            dirnames[:] = [name for name in dirnames if not self.should_ignore_media_name(name)]
            walk_path = Path(walk_root)
            walk_key = os.fspath(walk_path)
            if walk_path != base_path:
                try:
                    stat_info = os.stat(walk_path)
                    dir_markers[walk_key] = (stat_info.st_mtime_ns, stat_info.st_ctime_ns)
                except OSError:
                    dir_markers[walk_key] = (0, 0)
            for file_name in files:
                if self.should_ignore_media_name(file_name):
                    continue
                ext = os.path.splitext(file_name)[1].lower()
                if ext not in self.media_extensions:
                    continue
                candidate_path = walk_path / file_name
                try:
                    relative_path = candidate_path.resolve().relative_to(root.path.resolve())
                except Exception:
                    continue
                virtual_path = self.build_virtual_media_path(root.alias, relative_path.as_posix())
                folder_relative = relative_path.parent.as_posix()
                folder_key = self.build_virtual_media_path(root.alias, folder_relative)
                kind = "video" if ext in self.video_extensions else "image"
                media.append({"path": virtual_path, "folder": folder_key, "kind": kind, "extension": ext})
        media.sort(key=lambda item: item["path"].lower())
        return media, dir_markers

    def scan_folder_for_cache(self, folder_key: str, *, library: str) -> Tuple[List[Dict[str, str]], Dict[str, Tuple[int, int]]]:
        normalized_library = self.normalize_library_key(library)
        if folder_key == "all":
            combined_media: List[Dict[str, str]] = []
            combined_markers: Dict[str, Tuple[int, int]] = {}
            for root in self.library_roots(normalized_library):
                media_entries, dir_markers = self.scan_root_for_cache(root, root.path)
                combined_media.extend(media_entries)
                combined_markers.update(dir_markers)
            combined_media.sort(key=lambda item: item["path"].lower())
            return combined_media, combined_markers

        resolved = self.resolve_folder_path(folder_key, library=normalized_library)
        if resolved is None:
            return [], {}
        root, target_dir = resolved
        return self.scan_root_for_cache(root, target_dir)

    def directory_markers_changed(self, markers: Dict[str, Tuple[int, int]]) -> bool:
        for path, previous_marker in markers.items():
            if not isinstance(previous_marker, tuple):
                return True
            try:
                stat_info = os.stat(path)
            except (FileNotFoundError, OSError):
                return True
            current_marker = (stat_info.st_mtime_ns, stat_info.st_ctime_ns)
            if current_marker != previous_marker:
                return True
        return False

    def refresh_image_cache(self, folder: str = "all", hide_nsfw: bool = False, *, force: bool = False, library: str) -> List[str]:
        folder_key = self.normalize_folder_key(folder)
        normalized_library = self.normalize_library_key(library)
        cache_key = self.cache_scope_key(folder_key, normalized_library)

        with self.image_cache_lock:
            cached_entry = self.image_cache.get(cache_key)
            if cached_entry:
                cached_images = list(cached_entry.get("images", []))
                markers_snapshot = dict(cached_entry.get("dir_markers", {}))
            else:
                cached_images = []
                markers_snapshot = None

        needs_refresh = force or cached_entry is None
        if not needs_refresh and markers_snapshot is not None and self.directory_markers_changed(markers_snapshot):
            needs_refresh = True

        if not needs_refresh:
            return cached_images

        media, dir_markers = self.scan_folder_for_cache(folder_key, library=normalized_library)
        if hide_nsfw:
            media = [item for item in media if not self.path_contains_nsfw(item.get("path"))]
        images = [item["path"] for item in media if item.get("kind") == "image"]
        entry = {"images": images, "media": media, "dir_markers": dir_markers, "last_updated": time.time()}
        with self.image_cache_lock:
            self.image_cache[cache_key] = entry
        return list(images)

    def cache_folder_for_path(self, path: Optional[str]) -> Optional[str]:
        if not path:
            return None
        normalized = self.normalize_folder_key(path)
        if normalized == "all":
            return "all"
        alias, relative = self.split_virtual_media_path(normalized)
        if alias not in self.media_root_lookup:
            return normalized
        rel_path = Path(relative) if relative else Path()
        resolved = self.resolve_virtual_media_path(normalized)
        if resolved is not None and resolved.exists() and resolved.is_file():
            rel_path = rel_path.parent
        elif rel_path.suffix.lower() in self.media_extensions:
            rel_path = rel_path.parent
        rel_text = rel_path.as_posix() if str(rel_path) not in {"", "."} else ""
        return self.build_virtual_media_path(alias, rel_text)

    def invalidate_media_cache(self, path: Optional[str], *, library: Optional[str] = None) -> None:
        keys = ["all"]
        folder_key = self.cache_folder_for_path(path)
        if folder_key and folder_key not in keys:
            keys.append(folder_key)
        library_key = self.normalize_library_key(library) if library else None
        libraries = [library_key] if library_key else [self.media_library_default, self.ai_media_library]
        for key in keys:
            for current_library in libraries:
                self.refresh_image_cache(key, force=True, library=current_library)

    def initialize_image_cache(self) -> None:
        self.refresh_image_cache("all", force=True, library=self.media_library_default)
        self.refresh_image_cache("all", force=True, library=self.ai_media_library)

    def list_images(self, folder: str = "all", hide_nsfw: bool = False, *, library: str) -> List[str]:
        return self.refresh_image_cache(folder, hide_nsfw=hide_nsfw, library=library)

    def list_media(self, folder: str = "all", hide_nsfw: bool = False, *, library: str) -> List[Dict[str, Any]]:
        folder_key = self.normalize_folder_key(folder)
        normalized_library = self.normalize_library_key(library)
        self.refresh_image_cache(folder, hide_nsfw=hide_nsfw, library=normalized_library)
        cache_key = self.cache_scope_key(folder_key, normalized_library)
        with self.image_cache_lock:
            cached_entry = self.image_cache.get(cache_key)
            if cached_entry:
                media = [dict(item) for item in cached_entry.get("media", [])]
            else:
                media = []
        return media

    def get_subfolders(self, hide_nsfw: bool = False, *, library: str) -> List[str]:
        subfolders: List[str] = []
        seen: Set[str] = set()
        normalized_library = self.normalize_library_key(library)

        def _add(value: str) -> None:
            if value not in seen:
                seen.add(value)
                subfolders.append(value)

        _add("all")
        for root in self.library_roots(normalized_library):
            if hide_nsfw and self.path_contains_nsfw(root.alias):
                continue
            try:
                for walk_root, dirnames, _ in os.walk(root.path):
                    dirnames[:] = [name for name in dirnames if not self.should_ignore_media_name(name)]
                    base = Path(walk_root)
                    for dirname in dirnames:
                        try:
                            relative = (base / dirname).resolve().relative_to(root.path.resolve())
                        except ValueError:
                            continue
                        folder_key = self.build_virtual_media_path(root.alias, relative.as_posix())
                        if hide_nsfw and self.path_contains_nsfw(folder_key):
                            continue
                        _add(folder_key)
            except OSError:
                continue
        return subfolders

    def get_folder_inventory(self, hide_nsfw: bool = False, *, library: str) -> List[Dict[str, Any]]:
        inventory: List[Dict[str, Any]] = []
        normalized_library = self.normalize_library_key(library)
        for name in self.get_subfolders(hide_nsfw=hide_nsfw, library=normalized_library):
            media_entries = self.list_media(name, hide_nsfw=hide_nsfw, library=normalized_library)
            has_images = any(entry.get("kind") == "image" for entry in media_entries)
            has_videos = any(entry.get("kind") == "video" for entry in media_entries)
            display_name = name.split("/", 1)[1] if "/" in name else name
            inventory.append(
                {
                    "name": name,
                    "display_name": display_name,
                    "has_images": has_images,
                    "has_videos": has_videos,
                }
            )
        return inventory
