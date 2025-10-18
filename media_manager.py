from __future__ import annotations

import logging
import math
import mimetypes
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
from urllib.parse import quote

import tempfile

from PIL import Image, ImageDraw, ImageFont

from config_manager import MediaRoot
from werkzeug.datastructures import FileStorage

logger = logging.getLogger(__name__)

MEDIA_MANAGER_CACHE_SUBDIR = "_thumbnails_cache"

IMAGE_EXTENSIONS: set[str] = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".gif",
    ".tiff",
    ".avif",
}
VIDEO_EXTENSIONS: set[str] = {
    ".mp4",
    ".webm",
    ".mkv",
    ".mov",
    ".avi",
    ".m4v",
}

PLACEHOLDER_SIZE = (320, 180)
PLACEHOLDER_BG = (24, 24, 24)
PLACEHOLDER_FG = (215, 215, 215)
PLACEHOLDER_FONT_SIZE = 24

ListablePath = Union[str, Path, "VirtualPath"]

try:  # Pillow >= 9.1
    _RESAMPLING_FILTER = Image.Resampling.LANCZOS  # type: ignore[attr-defined]
except AttributeError:  # pragma: no cover - legacy Pillow
    _RESAMPLING_FILTER = Image.LANCZOS  # type: ignore[attr-defined]


class MediaManagerError(RuntimeError):
    """Structured exception raised for media management operations."""

    def __init__(self, message: str, *, code: str = "error", status: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.status = status


@dataclass(frozen=True)
class VirtualPath:
    alias: Optional[str]
    relative: Path

    @property
    def is_root(self) -> bool:
        return self.alias is None


def _safe_name(name: str) -> str:
    cleaned = name.strip().strip("/")
    if not cleaned or cleaned in {".", ".."}:
        raise MediaManagerError("Invalid name", code="invalid_name")
    if "/" in cleaned or "\\" in cleaned:
        raise MediaManagerError("Folder or file names cannot contain path separators", code="invalid_name")
    return cleaned


def _normalize_ext(value: str) -> str:
    text = value.strip().lower()
    return text if text.startswith(".") else f".{text}"


def _human_readable_size(size: int) -> str:
    if size <= 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB"]
    idx = int(min(len(units) - 1, math.floor(math.log(size, 1024))))
    scaled = size / (1024**idx)
    return f"{scaled:.1f} {units[idx]}"


def _generate_placeholder(label: str = "Preview") -> Image.Image:
    image = Image.new("RGB", PLACEHOLDER_SIZE, PLACEHOLDER_BG)
    drawer = ImageDraw.Draw(image)
    text = label.strip() or "Preview"
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", PLACEHOLDER_FONT_SIZE)
    except Exception:  # pragma: no cover - font availability varies
        font = ImageFont.load_default()
    text_width, text_height = drawer.textsize(text, font=font)
    x = (PLACEHOLDER_SIZE[0] - text_width) / 2
    y = (PLACEHOLDER_SIZE[1] - text_height) / 2
    drawer.text((x, y), text, fill=PLACEHOLDER_FG, font=font)
    return image


class MediaManager:
    """Utility encapsulating filesystem-safe media operations."""

    def __init__(
        self,
        roots: Sequence[MediaRoot],
        *,
        allowed_exts: Sequence[str],
        max_upload_mb: int,
        thumb_width: int = 320,
        nsfw_keyword: str = "nsfw",
        internal_dirs: Optional[Iterable[str]] = None,
    ) -> None:
        if not roots:
            raise ValueError("At least one media root is required")
        self._roots: Dict[str, MediaRoot] = {root.alias: root for root in roots}
        self._order: List[str] = [root.alias for root in roots]
        self._allowed_exts = {_normalize_ext(ext) for ext in allowed_exts}
        if not self._allowed_exts:
            self._allowed_exts = set(IMAGE_EXTENSIONS | VIDEO_EXTENSIONS)
        self._max_upload_bytes = max_upload_mb * 1024 * 1024
        self._thumb_width = thumb_width
        self._nsfw_keyword = nsfw_keyword.lower().strip() or "nsfw"
        self._internal_dirs = {d.strip() for d in internal_dirs or set()}
        self._cache_dirs: Dict[str, Path] = {}
        self._lock = threading.Lock()
        self._thumbnail_locks: Dict[Path, threading.Lock] = {}
        for alias, root in self._roots.items():
            cache_dir = root.path / MEDIA_MANAGER_CACHE_SUBDIR
            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
            except OSError as exc:  # pragma: no cover - depends on fs permissions
                logger.debug("Unable to ensure thumbnail cache %s: %s", cache_dir, exc)
            self._cache_dirs[alias] = cache_dir

    # ----------------------------------------------------------------------
    # Path helpers
    # ----------------------------------------------------------------------
    def _normalize_virtual_path(self, value: ListablePath) -> VirtualPath:
        if isinstance(value, VirtualPath):
            return value
        if isinstance(value, Path):
            text = value.as_posix()
        else:
            text = str(value or "").strip()
        if not text or text in {".", "./", "/"}:
            return VirtualPath(alias=None, relative=Path())
        normalized = text.replace("\\", "/").strip()
        if normalized.endswith(":"):
            alias = normalized[:-1]
            remainder = ""
        elif ":/" in normalized:
            alias, remainder = normalized.split(":/", 1)
        else:
            segments = normalized.lstrip("/").split("/", 1)
            alias = segments[0]
            remainder = segments[1] if len(segments) > 1 else ""
        alias = alias.strip()
        if not alias:
            return VirtualPath(alias=None, relative=Path(remainder.strip("/")))
        if alias not in self._roots:
            lowered = alias.lower()
            for key in self._roots.keys():
                if key.lower() == lowered:
                    alias = key
                    break
            else:
                raise MediaManagerError(f"Unknown media root '{alias}'", code="not_found", status=404)
        remainder = remainder.strip("/")
        remainder_path = Path(remainder)
        return VirtualPath(alias=alias, relative=remainder_path)

    def _resolve(self, virtual_path: ListablePath) -> Tuple[MediaRoot, Path]:
        vp = self._normalize_virtual_path(virtual_path)
        if vp.alias is None:
            raise MediaManagerError("Root path cannot be resolved without alias", code="invalid_path")
        root = self._roots[vp.alias]
        root_base = root.path.resolve()
        target = (root.path / vp.relative).resolve()
        try:
            target.relative_to(root_base)
        except ValueError:
            raise MediaManagerError("Path escapes the media root", code="invalid_path")
        return root, target

    def _virtualize(self, alias: str, abs_path: Path) -> str:
        root = self._roots[alias]
        try:
            rel = abs_path.resolve().relative_to(root.path.resolve())
        except ValueError:
            rel = Path()
        relative_text = rel.as_posix()
        return f"{alias}:/{relative_text}" if relative_text else f"{alias}:/"

    def _hidden(self, name: str) -> bool:
        if not name:
            return False
        if name in self._internal_dirs:
            return True
        if name.startswith(".") or name.startswith("_"):
            return True
        return False

    def _is_nsfw(self, candidate: str) -> bool:
        return self._nsfw_keyword and candidate and self._nsfw_keyword in candidate.lower()

    # ----------------------------------------------------------------------
    # Listing
    # ----------------------------------------------------------------------
    def list_directory(
        self,
        path: Optional[ListablePath],
        *,
        hide_nsfw: bool = True,
        page: int = 1,
        page_size: int = 100,
        sort: str = "name",
        order: str = "asc",
    ) -> Dict[str, Any]:
        vp = self._normalize_virtual_path(path or "")
        sort_key = sort.strip().lower() if sort else "name"
        order_value = order.strip().lower() if order else "asc"
        descending = order_value in {"desc", "descending", "z-a", "new"}

        if vp.alias is None:
            folders = []
            for alias in self._order:
                root = self._roots[alias]
                if hide_nsfw and self._is_nsfw(alias):
                    continue
                entry_path = self._virtualize(alias, root.path)
                folders.append(
                    {
                        "name": root.display_name or alias,
                        "path": entry_path,
                        "alias": alias,
                    }
                )
            return {
                "path": "",
                "folders": folders,
                "files": [],
                "page": 1,
                "total_pages": 1,
                "total_files": 0,
                "total_folders": len(folders),
            }

        root, abs_path = self._resolve(vp)
        if not abs_path.exists():
            raise MediaManagerError("Folder not found", code="not_found", status=404)
        if not abs_path.is_dir():
            raise MediaManagerError("Path is not a folder", code="invalid_path")

        folder_items: List[Dict[str, Any]] = []
        file_items: List[Dict[str, Any]] = []

        try:
            with os.scandir(abs_path) as scan:
                for entry in scan:
                    name = entry.name
                    if self._hidden(name):
                        continue
                    virtual_child = self._virtualize(root.alias, Path(entry.path))
                    if hide_nsfw and self._is_nsfw(virtual_child):
                        continue
                    try:
                        stat_info = entry.stat()
                    except OSError:
                        continue
                    if entry.is_dir():
                        try:
                            subcount = self._count_visible_entries(Path(entry.path), hide_nsfw=hide_nsfw)
                        except OSError:
                            subcount = 0
                        folder_items.append(
                            {
                                "name": name,
                                "path": virtual_child,
                                "count": subcount,
                            }
                        )
                    elif entry.is_file():
                        ext = Path(name).suffix.lower()
                        size = stat_info.st_size
                        file_items.append(
                            {
                                "name": name,
                                "path": virtual_child,
                                "size": size,
                                "size_text": _human_readable_size(size),
                                "mtime": stat_info.st_mtime,
                                "ext": ext,
                                "thumbUrl": self._thumbnail_url(virtual_child),
                                "isVideo": ext in VIDEO_EXTENSIONS,
                                "isImage": ext in IMAGE_EXTENSIONS,
                            }
                        )
        except FileNotFoundError:
            raise MediaManagerError("Folder not found", code="not_found", status=404)
        except OSError as exc:
            logger.debug("Error scanning directory %s: %s", abs_path, exc)
            raise MediaManagerError("Unable to read folder", code="scan_failed", status=500)

        if sort_key == "mtime":
            file_items.sort(key=lambda item: item.get("mtime") or 0, reverse=descending)
        else:
            file_items.sort(key=lambda item: item.get("name", "").lower(), reverse=descending)
        folder_items.sort(key=lambda item: item.get("name", "").lower())

        total_files = len(file_items)
        total_pages = max(1, math.ceil(total_files / max(1, page_size))) if total_files else 1
        current_page = max(1, min(int(page or 1), total_pages))
        start_index = (current_page - 1) * max(1, page_size)
        end_index = start_index + max(1, page_size)
        file_slice = file_items[start_index:end_index]

        return {
            "path": self._virtualize(root.alias, abs_path),
            "folders": folder_items,
            "files": file_slice,
            "page": current_page,
            "total_pages": total_pages,
            "page_size": page_size,
            "total_files": total_files,
            "total_folders": len(folder_items),
        }

    def _count_visible_entries(self, folder: Path, *, hide_nsfw: bool) -> int:
        count = 0
        try:
            with os.scandir(folder) as scan:
                for entry in scan:
                    if self._hidden(entry.name):
                        continue
                    candidate = (folder / entry.name).as_posix()
                    if hide_nsfw and self._is_nsfw(candidate):
                        continue
                    count += 1
        except OSError:
            return 0
        return count

    def _thumbnail_url(self, virtual_path: str) -> str:
        encoded = quote(virtual_path, safe="/:@")
        return f"/api/media/thumbnail?path={encoded}"

    # ----------------------------------------------------------------------
    # File operations
    # ----------------------------------------------------------------------
    def create_folder(self, parent: ListablePath, name: str) -> str:
        folder_name = _safe_name(name)
        if self._hidden(folder_name):
            raise MediaManagerError("Folder name is reserved", code="invalid_name")
        root, parent_abs = self._resolve(parent)
        if not parent_abs.exists():
            raise MediaManagerError("Parent folder does not exist", code="not_found", status=404)
        if not parent_abs.is_dir():
            raise MediaManagerError("Destination is not a folder", code="invalid_path")
        target = parent_abs / folder_name
        if target.exists():
            raise MediaManagerError("Folder already exists", code="exists")
        try:
            target.mkdir(parents=False, exist_ok=False)
        except FileExistsError:
            raise MediaManagerError("Folder already exists", code="exists")
        except OSError as exc:
            logger.debug("Failed to create folder %s: %s", target, exc)
            raise MediaManagerError("Unable to create folder", code="create_failed", status=500)
        return self._virtualize(root.alias, target)

    def rename(self, target_path: ListablePath, new_name: str) -> str:
        root, abs_path = self._resolve(target_path)
        if abs_path == root.path:
            raise MediaManagerError("Cannot rename a media root", code="forbidden", status=403)
        candidate_name = _safe_name(new_name)
        if self._hidden(candidate_name):
            raise MediaManagerError("Name is reserved", code="invalid_name")
        destination = abs_path.with_name(candidate_name)
        if destination.exists():
            raise MediaManagerError("Target name already exists", code="exists")
        try:
            abs_path.rename(destination)
        except OSError as exc:
            logger.debug("Rename failed for %s: %s", abs_path, exc)
            raise MediaManagerError("Unable to rename item", code="rename_failed", status=500)
        return self._virtualize(root.alias, destination)

    def delete(self, target_path: ListablePath) -> None:
        root, abs_path = self._resolve(target_path)
        if abs_path == root.path:
            raise MediaManagerError("Cannot delete a media root", code="forbidden", status=403)
        name = abs_path.name
        if name in self._internal_dirs:
            raise MediaManagerError("Deletion of protected folders is not allowed", code="forbidden", status=403)
        try:
            if abs_path.is_file():
                abs_path.unlink()
            elif abs_path.is_dir():
                shutil.rmtree(abs_path)
            else:
                raise MediaManagerError("Item not found", code="not_found", status=404)
        except FileNotFoundError:
            raise MediaManagerError("Item not found", code="not_found", status=404)
        except OSError as exc:
            logger.debug("Failed to delete %s: %s", abs_path, exc)
            raise MediaManagerError("Unable to delete item", code="delete_failed", status=500)

    def validate_upload(self, filename: str, size: int) -> None:
        if size > self._max_upload_bytes:
            raise MediaManagerError("File exceeds configured upload limit", code="too_large")
        ext = Path(filename).suffix.lower()
        if ext and ext not in self._allowed_exts:
            raise MediaManagerError("File type not allowed", code="invalid_extension")

    def upload(self, destination: ListablePath, files: Sequence[FileStorage]) -> List[Dict[str, Any]]:
        if not files:
            return []
        root, folder = self._resolve(destination)
        if not folder.exists() or not folder.is_dir():
            raise MediaManagerError("Destination folder not found", code="not_found", status=404)
        saved: List[Dict[str, Any]] = []
        for storage in files:
            filename = (storage.filename or "").strip()
            if not filename:
                continue
            safe_filename = _safe_name(Path(filename).name)
            ext = Path(safe_filename).suffix.lower()
            size = self._detect_size(storage)
            self.validate_upload(safe_filename, size if size is not None else 0)
            if size is not None and size > self._max_upload_bytes:
                raise MediaManagerError("File exceeds configured upload limit", code="too_large")
            target_path = folder / safe_filename
            if target_path.exists():
                raise MediaManagerError(f"'{safe_filename}' already exists", code="exists")
            try:
                self._save_upload(storage, target_path)
                stats = target_path.stat()
                if stats.st_size > self._max_upload_bytes:
                    if target_path.exists():
                        try:
                            target_path.unlink()
                        except OSError:
                            pass
                    raise MediaManagerError("File exceeds configured upload limit", code="too_large")
            except MediaManagerError:
                raise
            except OSError as exc:
                logger.debug("Failed to save upload %s: %s", target_path, exc)
                raise MediaManagerError("Failed to write uploaded file", code="upload_failed", status=500)
            entry = {
                "name": safe_filename,
                "path": self._virtualize(root.alias, target_path),
                "size": stats.st_size,
                "size_text": _human_readable_size(stats.st_size),
                "mtime": stats.st_mtime,
                "ext": ext,
                "thumbUrl": self._thumbnail_url(self._virtualize(root.alias, target_path)),
                "isVideo": ext in VIDEO_EXTENSIONS,
                "isImage": ext in IMAGE_EXTENSIONS,
            }
            saved.append(entry)
        return saved

    def _detect_size(self, storage: FileStorage) -> Optional[int]:
        if storage.content_length is not None:
            return storage.content_length
        stream = storage.stream
        tell = getattr(stream, "tell", None)
        seek = getattr(stream, "seek", None)
        if callable(tell) and callable(seek):
            try:
                pos = stream.tell()
                stream.seek(0, os.SEEK_END)
                size = stream.tell()
                stream.seek(pos, os.SEEK_SET)
                return size
            except OSError:
                return None
        return None

    def _save_upload(self, storage: FileStorage, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        stream = storage.stream
        seek = getattr(stream, "seek", None)
        if callable(seek):
            try:
                stream.seek(0)
            except OSError:
                pass
        storage.save(str(target))

    # ----------------------------------------------------------------------
    # Thumbnail generation
    # ----------------------------------------------------------------------
    def get_thumbnail(
        self,
        virtual_path: ListablePath,
        *,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> Tuple[Path, float, str]:
        root, abs_path = self._resolve(virtual_path)
        if not abs_path.exists() or not abs_path.is_file():
            raise MediaManagerError("File not found", code="not_found", status=404)
        stat_info = abs_path.stat()
        width = width or self._thumb_width
        height = height or int(width * 9 / 16)
        cache_key = self._cache_key(abs_path, width, height, stat_info.st_mtime)
        cache_dir = self._cache_dirs[root.alias]
        try:
            cache_path = (cache_dir / f"{cache_key}.jpg").resolve()
            cache_path.relative_to(cache_dir.resolve())
        except (ValueError, OSError):
            raise MediaManagerError("Invalid cache path", code="invalid_path", status=400)
        etag = f"W/\"{cache_key}\""

        if cache_path.exists():
            cache_mtime = cache_path.stat().st_mtime
            if cache_mtime >= stat_info.st_mtime:
                return cache_path, stat_info.st_mtime, etag

        lock = self._acquire_lock(cache_path)
        with lock:
            if cache_path.exists():
                cache_mtime = cache_path.stat().st_mtime
                if cache_mtime >= stat_info.st_mtime:
                    return cache_path, stat_info.st_mtime, etag
            ext = abs_path.suffix.lower()
            if ext in IMAGE_EXTENSIONS:
                image = self._render_image(abs_path, width, height)
            elif ext in VIDEO_EXTENSIONS:
                image = self._render_video(abs_path, width, height)
                if image is None:
                    image = _generate_placeholder("Video")
            else:
                image = _generate_placeholder(ext.upper() if ext else "File")
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(cache_path, "JPEG", quality=70)
        return cache_path, stat_info.st_mtime, etag

    def _cache_key(self, path: Path, width: int, height: int, mtime: float) -> str:
        digest = f"{path}:{mtime}:{width}:{height}"
        return str(abs(hash(digest)))

    def _acquire_lock(self, cache_path: Path) -> threading.Lock:
        with self._lock:
            lock = self._thumbnail_locks.get(cache_path)
            if lock is None:
                lock = threading.Lock()
                self._thumbnail_locks[cache_path] = lock
            return lock

    def _render_image(self, path: Path, width: int, height: int) -> Image.Image:
        with Image.open(path) as img:
            img = img.convert("RGB")
            img.thumbnail((width, height), _RESAMPLING_FILTER)
            background = Image.new("RGB", (width, height), PLACEHOLDER_BG)
            offset = ((width - img.width) // 2, (height - img.height) // 2)
            background.paste(img, offset)
            return background

    def _render_video(self, path: Path, width: int, height: int) -> Optional[Image.Image]:
        ffmpeg_path = self._which("ffmpeg")
        if not ffmpeg_path:
            logger.debug("ffmpeg not available; cannot render video thumbnail")
            return None
        timestamp = self._probe_video_timestamp(path)
        if timestamp is None:
            timestamp = 0.1
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        temp_output = Path(temp_file.name)
        temp_file.close()
        cmd = [
            ffmpeg_path,
            "-y",
            "-ss",
            str(timestamp),
            "-i",
            str(path),
            "-frames:v",
            "1",
            str(temp_output),
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # noqa: S603,S607
            with Image.open(temp_output) as img:
                img = img.convert("RGB")
                result = Image.new("RGB", (width, height), PLACEHOLDER_BG)
                img.thumbnail((width, height), _RESAMPLING_FILTER)
                offset = ((width - img.width) // 2, (height - img.height) // 2)
                result.paste(img, offset)
                return result
        except (subprocess.CalledProcessError, FileNotFoundError, OSError) as exc:
            logger.debug("ffmpeg thumbnail generation failed for %s: %s", path, exc)
            return None
        finally:
            try:
                temp_output.unlink()
            except OSError:
                pass

    def _probe_video_timestamp(self, path: Path) -> Optional[float]:
        ffprobe_path = self._which("ffprobe")
        if not ffprobe_path:
            return None
        cmd = [
            ffprobe_path,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)  # noqa: S603,S607
            duration = float(result.stdout.strip())
            if not math.isfinite(duration) or duration <= 0:
                return None
            return duration / 2
        except (ValueError, subprocess.CalledProcessError, FileNotFoundError):
            return None

    def _which(self, program: str) -> Optional[str]:
        for search_dir in os.getenv("PATH", "").split(os.pathsep):
            candidate = Path(search_dir) / program
            if candidate.exists() and os.access(candidate, os.X_OK):
                return str(candidate)
        return None

    def mime_type(self, path: ListablePath) -> Optional[str]:
        _, abs_path = self._resolve(path)
        mime, _ = mimetypes.guess_type(abs_path.name)
        return mime

    def allowed_extensions(self) -> List[str]:
        return sorted(self._allowed_exts)

    def max_upload_bytes(self) -> int:
        return self._max_upload_bytes
