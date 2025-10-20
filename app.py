import base64
import json
import atexit
import logging
import os
import re
import shutil
import subprocess
import threading
import time
import secrets
import random
import shlex
import io
import hashlib
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union, Set
from urllib.parse import quote, urlparse, parse_qs

import engineio.payload
import engineio.server

# --- Engine.IO packet limit patch (prevents "Too many packets in payload") ---
engineio.payload.Payload.max_decode_packets = 200
_ORIGINAL_ENGINEIO_HANDLE_REQUEST = engineio.server.Server.handle_request

from PIL import Image, ImageDraw, ImageFont

try:
    import cv2  # type: ignore[import]
except Exception:
    cv2 = None

from werkzeug.http import generate_etag, http_date, quote_etag

try:
    import requests
except Exception:
    requests = None

from flask import (
    Flask,
    Response,
    jsonify,
    send_file,
    request,
    render_template,
    redirect,
    url_for,
    stream_with_context,
    has_request_context,
)
from flask_socketio import SocketIO, join_room, leave_room
from stablehorde import StableHorde, StableHordeError, StableHordeCancelled
from picsum import (
    register_picsum_routes,
    assign_new_picsum_to_stream,
    configure_socketio,
)
from update_helpers import backup_user_state, restore_user_state
from system_monitor import get_system_stats
import config_manager
from media_manager import MediaManager, MediaManagerError, MEDIA_MANAGER_CACHE_SUBDIR
import timer_manager
import debug_manager
from job_manager import job_manager

try:
    from yt_dlp import YoutubeDL  # type: ignore[import]
except Exception:  # pragma: no cover - yt_dlp is optional at import-time
    YoutubeDL = None


def _configure_gunicorn_defaults() -> None:
    """Ensure Gunicorn uses resilient defaults for Eventlet workers."""

    required_pairs = {
        "--timeout": "120",
        "--graceful-timeout": "30",
        "--keep-alive": "5",
    }
    required_flags = {"--no-sendfile"}

    existing_cmd_args = os.environ.get("GUNICORN_CMD_ARGS", "")
    existing_tokens = shlex.split(existing_cmd_args) if existing_cmd_args else []

    filtered_tokens = []
    idx = 0
    while idx < len(existing_tokens):
        token = existing_tokens[idx]
        if token in required_pairs:
            idx += 2
            continue
        if any(token.startswith(f"{flag}=") for flag in required_pairs):
            idx += 1
            continue
        if token in required_flags:
            idx += 1
            continue
        filtered_tokens.append(token)
        idx += 1

    updated_tokens = filtered_tokens
    for flag, value in required_pairs.items():
        updated_tokens.extend([flag, value])
    updated_tokens.extend(sorted(required_flags))

    os.environ["GUNICORN_CMD_ARGS"] = " ".join(updated_tokens).strip()
    os.environ.setdefault("ENGINEIO_MAX_BUFFER_SIZE", "100000000")
    try:
        import eventlet.debug  # type: ignore[import]
    except Exception:  # pragma: no cover - eventlet optional in some environments
        pass
    else:
        eventlet.debug.hub_exceptions(False)


_configure_gunicorn_defaults()

app = Flask(__name__, static_folder="static", static_url_path="/static")

if not getattr(engineio.server.Server.handle_request, "__codex_overflow_patch__", False):
    def _quiet_engineio_handle_request(self, environ, start_response=None):
        try:
            return _ORIGINAL_ENGINEIO_HANDLE_REQUEST(self, environ, start_response)
        except ValueError as exc:
            if "Too many packets in payload" in str(exc):
                app.logger.warning("[SocketIO] Ignored Engine.IO packet overflow.")
                return []
            raise

    _quiet_engineio_handle_request.__codex_overflow_patch__ = True  # type: ignore[attr-defined]
    engineio.server.Server.handle_request = _quiet_engineio_handle_request

socketio = SocketIO(
    app,
    async_mode="eventlet",
    ping_timeout=120,
    ping_interval=25,
    cors_allowed_origins="*",
    max_http_buffer_size=100_000_000,
)
configure_socketio(socketio)
register_picsum_routes(app)

logger = logging.getLogger(__name__)

_emit_throttle_lock = threading.Lock()
_emit_min_interval = 0.05  # seconds
_last_emit_timestamp = 0.0


def safe_emit(
    event_name: str,
    data: Any,
    *,
    room: Optional[str] = None,
    to: Optional[str] = None,
    namespace: Optional[str] = None,
    skip_sid: Optional[str] = None,
    **kwargs: Any,
) -> None:
    """Emit a Socket.IO event defensively, ignoring disconnected clients."""

    global _last_emit_timestamp
    with _emit_throttle_lock:
        now = time.monotonic()
        wait_for = _emit_min_interval - (now - _last_emit_timestamp)
        if wait_for > 0:
            time.sleep(wait_for)
        _last_emit_timestamp = time.monotonic()
    try:
        socketio.emit(
            event_name,
            data,
            room=room,
            to=to,
            namespace=namespace,
            skip_sid=skip_sid,
            **kwargs,
        )
    except (OSError, BrokenPipeError) as exc:  # pragma: no cover - expected on disconnects
        app.logger.warning(
            "[SocketIO] Tried to emit to disconnected client for event '%s': %s",
            event_name,
            exc,
        )


SETTINGS_FILE = "settings.json"
CONFIG_FILE = config_manager.CONFIG_FILE.name

BACKUP_DIRNAME = "backups"
RESTORE_POINT_DIRNAME = "restorepoints"
RESTORE_POINT_METADATA_FILE = "metadata.json"
RESTORE_POINT_SETTINGS_FILE = "settings.json"
RESTORE_POINT_CONFIG_FILE = "config.json"
MAX_RESTORE_POINTS = 50

# Bounding boxes for optional thumbnail sizes requested via ?size=.
THUMBNAIL_SUBDIR = "_thumbnails"
THUMBNAIL_SIZE_PRESETS = {
    "thumb": (320, 320),
    "medium": (1024, 1024),
    "full": None,  # Alias for the original size
}
DASHBOARD_THUMBNAIL_SIZE = (128, 72)
THUMBNAIL_JPEG_QUALITY = 60
IMAGE_CACHE_TIMEOUT = 60 * 60 * 24 * 7  # One week default for conditional responses
IMAGE_CACHE_CONTROL_MAX_AGE = 31536000  # One year for browser Cache-Control headers

IMAGE_QUALITY_CHOICES = {"auto", "thumb", "medium", "full"}

AI_MODE = "ai"
AI_SETTINGS_KEY = "ai_settings"
AI_STATE_KEY = "ai_state"
AI_PRESETS_KEY = "_ai_presets"

TAG_KEY = "tags"
GLOBAL_TAGS_KEY = "_tags"
TAG_MAX_LENGTH = 48

AI_DEFAULT_MODEL = "stable_diffusion"
AI_DEFAULT_SAMPLER = "k_euler"
AI_DEFAULT_WIDTH = 512
AI_DEFAULT_HEIGHT = 512
AI_DEFAULT_STEPS = 30
AI_DEFAULT_CFG = 7.5
AI_DEFAULT_SAMPLES = 1

AI_OUTPUT_SUBDIR = "ai_generated"
AI_TEMP_SUBDIR = "_ai_temp"
AI_DEFAULT_PERSIST = True
AI_POLL_INTERVAL = 5.0
AI_TIMEOUT = 0.0
INTERNAL_MEDIA_DIRS = {THUMBNAIL_SUBDIR, AI_TEMP_SUBDIR, MEDIA_MANAGER_CACHE_SUBDIR}
IGNORED_MEDIA_PREFIX = "_"
STABLE_HORDE_LOG_PREFIX = "[StableHorde]"


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        stripped = value.strip().lower()
        if not stripped:
            return default
        return stripped in {"1", "true", "yes", "on"}
    if value is None:
        return default
    return bool(value)


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_extensions(value: Any) -> List[str]:
    if isinstance(value, (list, tuple, set)):
        raw_items = list(value)
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raw_items = []
        else:
            raw_items = [segment.strip() for segment in re.split(r"[,\s]+", stripped) if segment.strip()]
    else:
        raw_items = []
    result: List[str] = []
    for item in raw_items:
        text = str(item).strip().lower()
        if not text:
            continue
        if not text.startswith("."):
            text = f".{text}"
        if text not in result:
            result.append(text)
    if not result:
        fallback_exts = sorted(set(IMAGE_EXTENSIONS) | set(VIDEO_EXTENSIONS))
        result = list(fallback_exts)
    return result


def _virtual_leaf(virtual_path: str) -> str:
    if not virtual_path:
        return ""
    candidate = virtual_path.rstrip("/")
    if ":/" in candidate:
        _, remainder = candidate.split(":/", 1)
    else:
        parts = candidate.split("/", 1)
        remainder = parts[1] if len(parts) > 1 else parts[0]
    remainder = remainder.strip("/")
    if not remainder:
        return ""
    segments = remainder.split("/")
    return segments[-1]


def _should_ignore_media_name(name: Optional[str]) -> bool:
    """Return True when a media entry should be skipped (internal or leading underscore)."""
    if not name:
        return False
    normalized = str(name).strip()
    if not normalized:
        return False
    leaf = normalized.replace("\\", "/").rsplit("/", 1)[-1]
    if leaf in INTERNAL_MEDIA_DIRS:
        return True
    return leaf.startswith(IGNORED_MEDIA_PREFIX)

AUTO_GENERATE_MODES = {"off", "timer", "clock"}
AUTO_GENERATE_INTERVAL_UNITS = {"minutes": 60.0, "hours": 3600.0}
AUTO_GENERATE_MIN_INTERVAL_SECONDS = 60.0
AUTO_GENERATE_DEFAULT_INTERVAL_VALUE = 10.0

class AutoGenerationError(RuntimeError):
    """Raised when an AI generation queue request cannot be fulfilled."""


class AutoGenerationUnavailable(AutoGenerationError):
    """Raised when Stable Horde is unavailable for generation."""


class AutoGenerationPromptMissing(AutoGenerationError):
    """Raised when a required prompt is missing."""


class AutoGenerationBusy(AutoGenerationError):
    """Raised when a generation job is already active for a stream."""


STABLE_HORDE_POST_PROCESSORS = [
    "GFPGAN",
    "RealESRGAN_x4plus",
    "RealESRGAN_x2plus",
    "RealESRGAN_x4plus_anime_6B",
    "NMKD_Siax",
    "4x_AnimeSharp",
    "CodeFormers",
    "strip_background",
]
STABLE_HORDE_MAX_LORAS = 4
STABLE_HORDE_CLIP_SKIP_RANGE = (1, 12)
STABLE_HORDE_STRENGTH_RANGE = (0.0, 1.0)
STABLE_HORDE_DENOISE_RANGE = (0.01, 1.0)
try:
    IMAGE_THUMBNAIL_FILTER = Image.Resampling.LANCZOS  # type: ignore[attr-defined]
except AttributeError:  # Pillow < 9.1
    IMAGE_THUMBNAIL_FILTER = Image.LANCZOS

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".webp")
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v", ".mpg", ".mpeg")
MEDIA_EXTENSIONS = IMAGE_EXTENSIONS + VIDEO_EXTENSIONS
VIDEO_PLAYBACK_MODES = {"duration", "until_end", "loop"}

MEDIA_MODE_IMAGE = "image"
MEDIA_MODE_VIDEO = "video"
MEDIA_MODE_LIVESTREAM = "livestream"
MEDIA_MODE_AI = AI_MODE
MEDIA_MODE_PICSUM = "picsum"
MEDIA_MODE_CHOICES = {
    MEDIA_MODE_IMAGE,
    MEDIA_MODE_VIDEO,
    MEDIA_MODE_LIVESTREAM,
    MEDIA_MODE_AI,
    MEDIA_MODE_PICSUM,
}
MEDIA_MODE_VARIANTS = {
    MEDIA_MODE_IMAGE: {"random", "specific"},
    MEDIA_MODE_VIDEO: {"random", "specific"},
    MEDIA_MODE_LIVESTREAM: {"livestream"},
    MEDIA_MODE_AI: {AI_MODE},
    MEDIA_MODE_PICSUM: {MEDIA_MODE_PICSUM},
}

DEFAULT_MEDIA_ROOT_PATH = Path(os.path.abspath("./media")).resolve()

CONFIG: Dict[str, Any] = config_manager.load_config()
MEDIA_ROOTS = config_manager.build_media_roots(CONFIG.get("MEDIA_PATHS", []))
if not MEDIA_ROOTS:
    default_path = DEFAULT_MEDIA_ROOT_PATH
    default_alias = default_path.name or "media"
    MEDIA_ROOTS = [
        config_manager.MediaRoot(alias=default_alias, path=default_path, display_name=default_alias)
    ]

for root in MEDIA_ROOTS:
    try:
        if root.path.resolve(strict=False) == DEFAULT_MEDIA_ROOT_PATH:
            root.path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning("Unable to ensure media directory %s: %s", root.path, exc)

config_manager.validate_media_paths([root.path.as_posix() for root in MEDIA_ROOTS])

AVAILABLE_MEDIA_ROOTS: List[config_manager.MediaRoot] = []
for candidate_root in MEDIA_ROOTS:
    try:
        if candidate_root.path.exists() and candidate_root.path.is_dir() and os.access(candidate_root.path, os.R_OK):
            AVAILABLE_MEDIA_ROOTS.append(candidate_root)
    except OSError:
        continue

if not AVAILABLE_MEDIA_ROOTS:
    fallback_root = MEDIA_ROOTS[0]
    try:
        fallback_root.path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning("Unable to prepare fallback media directory %s: %s", fallback_root.path, exc)
    AVAILABLE_MEDIA_ROOTS = [fallback_root]

MEDIA_ROOT_LOOKUP = {root.alias: root for root in MEDIA_ROOTS}
PRIMARY_MEDIA_ROOT = AVAILABLE_MEDIA_ROOTS[0]
THUMBNAIL_CACHE_DIR = PRIMARY_MEDIA_ROOT.path / THUMBNAIL_SUBDIR

NSFW_KEYWORD = "nsfw"

MEDIA_MANAGEMENT_ALLOW_EDIT = _as_bool(CONFIG.get("MEDIA_MANAGEMENT_ALLOW_EDIT"), True)
MEDIA_UPLOAD_MAX_MB = max(1, _as_int(CONFIG.get("MEDIA_UPLOAD_MAX_MB"), 256))
MEDIA_ALLOWED_EXTS = _normalize_extensions(CONFIG.get("MEDIA_ALLOWED_EXTS"))
MEDIA_THUMB_WIDTH = max(64, _as_int(CONFIG.get("MEDIA_THUMB_WIDTH"), 320))
MEDIA_PREVIEW_ENABLED = _as_bool(CONFIG.get("MEDIA_PREVIEW_ENABLED"), True)
MEDIA_PREVIEW_FRAMES = max(1, _as_int(CONFIG.get("MEDIA_PREVIEW_FRAMES"), 8))
MEDIA_PREVIEW_WIDTH = max(32, _as_int(CONFIG.get("MEDIA_PREVIEW_WIDTH"), MEDIA_THUMB_WIDTH))
_preview_duration_raw = CONFIG.get("MEDIA_PREVIEW_MAX_DURATION", 300)
try:
    MEDIA_PREVIEW_MAX_DURATION = float(_preview_duration_raw)
except (TypeError, ValueError):
    MEDIA_PREVIEW_MAX_DURATION = 300.0
if MEDIA_PREVIEW_MAX_DURATION < 0:
    MEDIA_PREVIEW_MAX_DURATION = 0.0
MEDIA_PREVIEW_MAX_MB = max(0, _as_int(CONFIG.get("MEDIA_PREVIEW_MAX_MB"), 512))
MEDIA_PREVIEW_MAX_BYTES: Optional[int]
if MEDIA_PREVIEW_MAX_MB > 0:
    MEDIA_PREVIEW_MAX_BYTES = MEDIA_PREVIEW_MAX_MB * 1024 * 1024
else:
    MEDIA_PREVIEW_MAX_BYTES = None
MEDIA_MANAGER = MediaManager(
    roots=MEDIA_ROOTS,
    allowed_exts=MEDIA_ALLOWED_EXTS,
    max_upload_mb=MEDIA_UPLOAD_MAX_MB,
    thumb_width=MEDIA_THUMB_WIDTH,
    nsfw_keyword=NSFW_KEYWORD,
    internal_dirs=INTERNAL_MEDIA_DIRS,
    preview_enabled=MEDIA_PREVIEW_ENABLED,
    preview_frames=MEDIA_PREVIEW_FRAMES,
    preview_width=MEDIA_PREVIEW_WIDTH,
    preview_max_duration=MEDIA_PREVIEW_MAX_DURATION,
    preview_max_bytes=MEDIA_PREVIEW_MAX_BYTES,
)
MEDIA_UPLOAD_MAX_BYTES = MEDIA_MANAGER.max_upload_bytes()


def _media_error_response(exc: MediaManagerError):
    status = getattr(exc, "status", 400) or 400
    payload = {"error": exc.message, "code": exc.code}
    return jsonify(payload), status


def _require_media_edit() -> None:
    if not MEDIA_MANAGEMENT_ALLOW_EDIT:
        raise MediaManagerError("Media editing is disabled", code="forbidden", status=403)


# Cache image paths per folder so we can serve repeated requests without rescanning the disk.
IMAGE_CACHE: Dict[str, Dict[str, Any]] = {}


def _media_root_available(root: config_manager.MediaRoot) -> bool:
    try:
        return root.path.exists() and root.path.is_dir() and os.access(root.path, os.R_OK)
    except OSError:
        return False


def _build_virtual_media_path(alias: str, relative: Union[str, Path]) -> str:
    relative_text = str(relative).replace("\\", "/")
    relative_text = relative_text.lstrip("./")
    return f"{alias}/{relative_text}" if relative_text else alias


def _split_virtual_media_path(value: Union[str, Path]) -> Tuple[str, str]:
    if isinstance(value, Path):
        value = value.as_posix()
    if not value:
        return PRIMARY_MEDIA_ROOT.alias, ""
    text = str(value).strip()
    if not text:
        return PRIMARY_MEDIA_ROOT.alias, ""
    if os.path.isabs(text):
        resolved = Path(text).resolve()
        for root in MEDIA_ROOTS:
            try:
                rel_path = resolved.relative_to(root.path.resolve())
                return root.alias, rel_path.as_posix()
            except ValueError:
                continue
        return PRIMARY_MEDIA_ROOT.alias, resolved.as_posix()
    normalized = text.replace("\\", "/").strip("/")
    if not normalized:
        return PRIMARY_MEDIA_ROOT.alias, ""
    parts = normalized.split("/", 1)
    alias = parts[0]
    remainder = parts[1] if len(parts) > 1 else ""
    if alias in MEDIA_ROOT_LOOKUP:
        return alias, remainder
    return PRIMARY_MEDIA_ROOT.alias, normalized


def _resolve_virtual_media_path(virtual_path: Union[str, Path]) -> Optional[Path]:
    alias, relative = _split_virtual_media_path(virtual_path)
    root = MEDIA_ROOT_LOOKUP.get(alias)
    if root is None:
        return None
    candidate = (root.path / relative).resolve()
    try:
        candidate.relative_to(root.path.resolve())
    except ValueError:
        return None
    return candidate


def _virtualize_path(path: Union[str, Path]) -> str:
    alias, relative = _split_virtual_media_path(path)
    if relative and os.path.isabs(relative):
        return Path(relative).as_posix()
    if relative.startswith("../"):
        return Path(relative).as_posix()
    return _build_virtual_media_path(alias, relative)

def _ensure_thumbnail_dir() -> Optional[Path]:
    """Create the thumbnail cache directory if possible; return the path when ready."""
    try:
        THUMBNAIL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as exc:  # pragma: no cover - filesystem availability varies
        logger.debug("Unable to prepare thumbnail cache directory: %s", exc)
        return None
    return THUMBNAIL_CACHE_DIR

def _thumbnail_disk_path(stream_id: str) -> Path:
    """Return the filesystem target for a stream's cached thumbnail image."""
    digest = hashlib.sha1(stream_id.encode("utf-8")).hexdigest()[:10]
    safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", stream_id) or "stream"
    filename = f"{safe_name}-{digest}.jpg"
    return THUMBNAIL_CACHE_DIR / filename

def _thumbnail_public_url(stream_id: str) -> str:
    """Return the public URL a client can use to load the cached thumbnail."""
    return f"/thumbnails/{quote(stream_id, safe='')}.jpg"

def _public_thumbnail_payload(record: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Strip internal bookkeeping keys before sending thumbnail metadata to clients."""
    if not isinstance(record, dict):
        return None
    payload = {k: v for k, v in record.items() if not k.startswith("_")}
    return payload or None


def _sanitize_embed_metadata(raw: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    title = raw.get("title")
    if isinstance(title, str):
        title = title.strip() or None
    else:
        title = None
    content_type = raw.get("content_type")
    if isinstance(content_type, str):
        lowered = content_type.strip().lower()
        if lowered in {"video", "playlist", "live"}:
            content_type = lowered
        else:
            content_type = None
    else:
        content_type = None
    provider = raw.get("provider") or raw.get("provider_name")
    if isinstance(provider, str):
        provider = provider.strip() or None
    else:
        provider = None
    video_id = raw.get("video_id")
    if isinstance(video_id, str):
        video_id = video_id.strip() or None
    else:
        video_id = None
    playlist_id = raw.get("playlist_id")
    if isinstance(playlist_id, str):
        playlist_id = playlist_id.strip() or None
    else:
        playlist_id = None
    start_index = raw.get("start_index")
    if isinstance(start_index, int):
        index_value = start_index
    else:
        try:
            index_value = int(start_index)
        except (TypeError, ValueError):
            index_value = None
    is_live_raw = raw.get("is_live")
    is_live = bool(is_live_raw) if is_live_raw is not None else (content_type == "live")
    canonical_url = raw.get("canonical_url") or raw.get("url")
    if isinstance(canonical_url, str):
        canonical_url = canonical_url.strip() or None
    else:
        canonical_url = None
    thumbnail_url = raw.get("thumbnail_url")
    if isinstance(thumbnail_url, str):
        thumbnail_url = thumbnail_url.strip() or None
    else:
        thumbnail_url = None
    author_name = raw.get("author_name")
    if isinstance(author_name, str):
        author_name = author_name.strip() or None
    else:
        author_name = None
    fetched_at = raw.get("fetched_at")
    if isinstance(fetched_at, str):
        fetched_at = fetched_at.strip() or None
    else:
        fetched_at = None
    meta = {
        "title": title,
        "content_type": content_type,
        "provider": provider or "YouTube",
        "video_id": video_id,
        "playlist_id": playlist_id,
        "start_index": index_value,
        "is_live": bool(is_live),
    }
    if canonical_url:
        meta["canonical_url"] = canonical_url
    if thumbnail_url:
        meta["thumbnail_url"] = thumbnail_url
    if author_name:
        meta["author_name"] = author_name
    if fetched_at:
        meta["fetched_at"] = fetched_at
    return meta


def _is_youtube_host(host: str) -> bool:
    if not host:
        return False
    host = host.lower()
    if host in YOUTUBE_DOMAINS:
        return True
    return any(host.endswith(f".{domain}") for domain in YOUTUBE_DOMAINS)


def _is_youtube_url(url: Optional[str]) -> bool:
    if not url or not isinstance(url, str):
        return False
    candidate = url.strip()
    if not candidate:
        return False
    if "://" not in candidate:
        candidate = f"https://{candidate}"
    try:
        parsed = urlparse(candidate)
    except ValueError:
        return False
    return _is_youtube_host(parsed.netloc)


def _parse_youtube_url_details(url: str) -> Optional[Dict[str, Any]]:
    if not _is_youtube_url(url):
        return None
    candidate = url.strip()
    if "://" not in candidate:
        candidate = f"https://{candidate}"
    try:
        parsed = urlparse(candidate)
    except ValueError:
        return None
    query_map = parse_qs(parsed.query or "")
    segments = [segment for segment in parsed.path.split("/") if segment]
    video_id: Optional[str] = None
    playlist_id: Optional[str] = None
    start_index: Optional[int] = None
    start_seconds: Optional[int] = None

    # Video ID detection across watch, embed, short, live and short URL formats.
    if "v" in query_map and query_map["v"]:
        video_id = query_map["v"][0]
    elif parsed.netloc.endswith("youtu.be") and segments:
        video_id = segments[0]
    elif segments:
        if segments[0] in {"embed", "shorts", "live"} and len(segments) > 1:
            video_id = segments[1]
        elif segments[0] == "watch" and len(segments) > 1:
            video_id = segments[-1]

    playlist_candidates = query_map.get("list")
    if playlist_candidates:
        playlist_id = playlist_candidates[0] or None

    index_candidates = query_map.get("index")
    if index_candidates:
        try:
            start_index = int(index_candidates[0])
        except (TypeError, ValueError):
            start_index = None

    start_candidates = query_map.get("start")
    if start_candidates:
        try:
            start_seconds = int(start_candidates[0])
        except (TypeError, ValueError):
            start_seconds = None

    is_live = False
    if segments and segments[0] == "live":
        is_live = True
    else:
        live_candidate = query_map.get("live") or query_map.get("live_stream")
        if live_candidate:
            value = live_candidate[0]
            if isinstance(value, str):
                is_live = value.strip().lower() in {"1", "true", "yes", "live"}
            else:
                is_live = bool(value)

    canonical_url = None
    if playlist_id and not video_id:
        canonical_url = f"https://www.youtube.com/playlist?list={playlist_id}"
    else:
        params = []
        if video_id:
            params.append(f"v={video_id}")
        if playlist_id:
            params.append(f"list={playlist_id}")
        if start_index is not None:
            params.append(f"index={start_index}")
        if start_seconds is not None:
            params.append(f"start={start_seconds}")
        canonical_url = "https://www.youtube.com/watch"
        if params:
            canonical_url += f"?{'&'.join(params)}"

    embed_base: str
    if playlist_id:
        embed_base = "https://www.youtube-nocookie.com/embed/videoseries"
    elif video_id:
        embed_base = f"https://www.youtube-nocookie.com/embed/{video_id}"
    else:
        embed_base = "https://www.youtube-nocookie.com/embed/"

    return {
        "original_url": candidate,
        "video_id": video_id,
        "playlist_id": playlist_id,
        "start_index": start_index,
        "start_seconds": start_seconds,
        "is_live": bool(is_live),
        "canonical_url": canonical_url,
        "embed_base": embed_base,
        "host": parsed.netloc.lower(),
        "path": parsed.path,
        "query": query_map,
    }


def _youtube_cache_key(details: Dict[str, Any]) -> Tuple[str, ...]:
    playlist_id = details.get("playlist_id") or ""
    video_id = details.get("video_id") or ""
    if playlist_id and video_id:
        return ("playlist_video", playlist_id, video_id)
    if playlist_id:
        return ("playlist", playlist_id)
    if video_id:
        return ("video", video_id)
    canonical = details.get("canonical_url") or details.get("original_url") or ""
    return ("url", canonical)


def _youtube_oembed_html_says_live(oembed: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(oembed, dict):
        return False
    html = oembed.get("html")
    if isinstance(html, str) and "live_stream" in html.lower():
        return True
    live_broadcast = oembed.get("is_live") or oembed.get("live_broadcast")
    if isinstance(live_broadcast, str):
        return live_broadcast.strip().lower() in {"1", "true", "yes", "live"}
    if isinstance(live_broadcast, (int, float)):
        return bool(live_broadcast)
    if isinstance(live_broadcast, bool):
        return live_broadcast
    return False


def _youtube_page_looks_live(details: Dict[str, Any]) -> Optional[bool]:
    if requests is None:
        return None
    cache_key = _youtube_cache_key(details)
    now = time.time()
    with YOUTUBE_LIVE_PROBE_CACHE_LOCK:
        cached = YOUTUBE_LIVE_PROBE_CACHE.get(cache_key)
        if (
            cached
            and now - cached.get("timestamp", 0) < YOUTUBE_LIVE_PROBE_CACHE_TTL
        ):
            return cached.get("result")
    url_candidates: List[str] = []
    for key in ("canonical_url", "original_url", "embed_base"):
        candidate = details.get(key)
        if isinstance(candidate, str) and candidate and candidate not in url_candidates:
            url_candidates.append(candidate)
    if not url_candidates:
        return None
    headers = {
        "User-Agent": "EchoMosaic/1.0 (+https://github.com/dodenbear/EchoMosaic)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    result: Optional[bool] = None
    for url in url_candidates:
        try:
            resp = requests.get(url, headers=headers, timeout=5, stream=True)
            resp.raise_for_status()
        except Exception as exc:
            logger.debug("YouTube live probe failed for %s: %s", url, exc)
            continue
        encoding = resp.encoding or "utf-8"
        text_bytes = bytearray()
        try:
            for chunk in resp.iter_content(chunk_size=4096, decode_unicode=False):
                if not chunk:
                    continue
                text_bytes.extend(chunk)
                if len(text_bytes) >= YOUTUBE_LIVE_PROBE_MAX_BYTES:
                    break
        finally:
            resp.close()
        if not text_bytes:
            continue
        try:
            snippet_text = text_bytes.decode(encoding, "ignore")
        except Exception:
            snippet_text = text_bytes.decode("utf-8", "ignore")
        snippet = snippet_text.lower()
        if not snippet:
            continue
        if any(marker in snippet for marker in YOUTUBE_LIVE_HTML_MARKERS):
            result = True
            break
        if '"livebroadcastdetails"' in snippet and '"islive":true' in snippet:
            result = True
            break
        if '"livebroadcastdetails"' in snippet and '"islivenow":true' in snippet:
            result = True
            break
    if result is None:
        result = False
    with YOUTUBE_LIVE_PROBE_CACHE_LOCK:
        YOUTUBE_LIVE_PROBE_CACHE[cache_key] = {
            "timestamp": time.time(),
            "result": result,
        }
    return result


def _derive_youtube_content_type(
    details: Dict[str, Any], oembed: Optional[Dict[str, Any]] = None
) -> str:
    if details.get("playlist_id"):
        return "playlist"
    if details.get("is_live"):
        return "live"
    if isinstance(oembed, dict):
        raw_type = oembed.get("type")
        if isinstance(raw_type, str):
            lowered = raw_type.strip().lower()
            if lowered in {"video", "playlist", "live"}:
                if lowered == "playlist" and details.get("playlist_id"):
                    return "playlist"
                if lowered == "live":
                    return "live"
        if _youtube_oembed_html_says_live(oembed):
            return "live"
    title = ""
    if isinstance(oembed, dict):
        title_candidate = oembed.get("title")
        if isinstance(title_candidate, str):
            title = title_candidate.lower()
    if " live " in f" {title} " or title.startswith("live "):
        return "live"
    probe = _youtube_page_looks_live(details)
    if probe:
        return "live"
    return "video"


def _build_youtube_metadata(
    details: Dict[str, Any], oembed: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    metadata = {}
    if isinstance(oembed, dict):
        title = oembed.get("title")
        if isinstance(title, str) and title.strip():
            metadata["title"] = title.strip()
        provider = oembed.get("provider_name")
        if isinstance(provider, str) and provider.strip():
            metadata["provider"] = provider.strip()
        author = oembed.get("author_name")
        if isinstance(author, str) and author.strip():
            metadata["author_name"] = author.strip()
        thumbnail = oembed.get("thumbnail_url")
        if isinstance(thumbnail, str) and thumbnail.strip():
            metadata["thumbnail_url"] = thumbnail.strip()
        oembed_type = oembed.get("type")
        if isinstance(oembed_type, str) and oembed_type.strip():
            metadata["oembed_type"] = oembed_type.strip().lower()
    if "provider" not in metadata:
        metadata["provider"] = "YouTube"
    metadata["video_id"] = details.get("video_id")
    metadata["playlist_id"] = details.get("playlist_id")
    metadata["start_index"] = details.get("start_index")
    metadata["start_seconds"] = details.get("start_seconds")
    metadata["canonical_url"] = details.get("canonical_url")
    content_type = _derive_youtube_content_type(details, oembed)
    metadata["content_type"] = content_type
    metadata["is_live"] = content_type == "live"
    return metadata


def _youtube_oembed_lookup(
    url: str,
    details: Dict[str, Any],
    *,
    force: bool = False,
) -> Optional[Dict[str, Any]]:
    cache_key = _youtube_cache_key(details)
    now = time.time()
    with YOUTUBE_OEMBED_CACHE_LOCK:
        cached = YOUTUBE_OEMBED_CACHE.get(cache_key)
        if (
            cached
            and not force
            and now - cached.get("timestamp", 0) < YOUTUBE_OEMBED_CACHE_TTL
        ):
            return dict(cached.get("data", {}))
    if requests is None:
        return _build_youtube_metadata(details, None)
    try:
        response = requests.get(
            YOUTUBE_OEMBED_ENDPOINT,
            params={"url": url, "format": "json"},
            timeout=6,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        logger.debug("YouTube oEmbed lookup failed for %s: %s", url, exc)
        with YOUTUBE_OEMBED_CACHE_LOCK:
            cached = YOUTUBE_OEMBED_CACHE.get(cache_key)
            if cached:
                return dict(cached.get("data", {}))
        return _build_youtube_metadata(details, None)

    metadata = _build_youtube_metadata(details, payload)
    timestamp = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    metadata["fetched_at"] = timestamp
    with YOUTUBE_OEMBED_CACHE_LOCK:
        YOUTUBE_OEMBED_CACHE[cache_key] = {"data": dict(metadata), "timestamp": now}
    return metadata


def _set_runtime_embed_metadata(stream_id: str, metadata: Optional[Dict[str, Any]]) -> None:
    if not stream_id or stream_id.startswith("_"):
        return
    with STREAM_RUNTIME_LOCK:
        entry = STREAM_RUNTIME_STATE.setdefault(stream_id, {})
        if metadata is None:
            entry.pop("embed_metadata", None)
        else:
            entry["embed_metadata"] = dict(metadata)


def _refresh_embed_metadata(
    stream_id: str,
    conf: Dict[str, Any],
    *,
    force: bool = False,
) -> Optional[Dict[str, Any]]:
    if not isinstance(conf, dict):
        return None
    media_mode_raw = conf.get("media_mode") or conf.get("mode")
    media_mode = media_mode_raw.strip().lower() if isinstance(media_mode_raw, str) else ""
    stream_url = conf.get("stream_url")

    if media_mode not in {MEDIA_MODE_LIVESTREAM, "livestream"} or not stream_url:
        conf["embed_metadata"] = None
        _set_runtime_embed_metadata(stream_id, None)
        return None

    details = _parse_youtube_url_details(stream_url)
    if details is None:
        conf["embed_metadata"] = None
        _set_runtime_embed_metadata(stream_id, None)
        return None

    metadata = _youtube_oembed_lookup(stream_url, details, force=force)
    sanitized = _sanitize_embed_metadata(metadata or {})
    conf["embed_metadata"] = sanitized
    _set_runtime_embed_metadata(stream_id, sanitized)
    return sanitized
IMAGE_CACHE_LOCK = threading.Lock()
STREAM_RUNTIME_STATE: Dict[str, Dict[str, Any]] = {}
STREAM_RUNTIME_LOCK = threading.Lock()

YOUTUBE_DOMAINS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
    "youtu.be",
    "www.youtu.be",
    "youtube-nocookie.com",
    "www.youtube-nocookie.com",
}
YOUTUBE_OEMBED_ENDPOINT = "https://www.youtube.com/oembed"
YOUTUBE_OEMBED_CACHE_TTL = 20 * 60  # 20 minutes
YOUTUBE_OEMBED_CACHE: Dict[Tuple[str, ...], Dict[str, Any]] = {}
YOUTUBE_OEMBED_CACHE_LOCK = threading.Lock()
YOUTUBE_LIVE_PROBE_CACHE_TTL = 15 * 60  # 15 minutes
YOUTUBE_LIVE_PROBE_CACHE: Dict[Tuple[str, ...], Dict[str, Any]] = {}
YOUTUBE_LIVE_PROBE_CACHE_LOCK = threading.Lock()
YOUTUBE_LIVE_PROBE_MAX_BYTES = 30_000
YOUTUBE_LIVE_HTML_MARKERS = (
    '"islive":true',
    '"islive":1',
    '"islivecontent":true',
    '"islivebroadcast":true',
    '"broadcaststatus":"live"',
    '"playbackmode":"livestream"',
    '"playerstate":"live_stream"',
    '"livenow":true',
    '"livestreamabilityrenderer"',
    '"live_streamability"',
    '"thumbnailoverlaytimestatusrenderer":{"style":"live"',
    '"livebroadcastdetails":{"',
    'itemprop="islivebroadcast"',
)

STREAM_PLAYBACK_HISTORY_LIMIT = 50
STREAM_UPDATE_EVENT = "stream_update"
STREAM_INIT_EVENT = "stream_init"
SYNC_TIME_EVENT = "sync_time"
STREAM_SYNC_INTERVAL_SECONDS = 3.0

LIVE_HLS_ASYNC = True
HLS_TTL_SECS = 3600
MAX_HLS_WORKERS = 3
HLS_ERROR_RETRY_SECS = 30

playback_manager: Optional["StreamPlaybackManager"] = None


def _normalize_folder_key(folder: Optional[str]) -> str:
    """Normalize request folder values into the cache key used internally."""
    if folder is None:
        return "all"
    normalized = str(folder).strip()
    if normalized in {"", "all", "."}:
        return "all"
    return normalized.replace("\\", "/").strip("/")


def _resolve_folder_path(folder_key: str) -> Optional[Tuple[config_manager.MediaRoot, Path]]:
    """Return the media root and absolute filesystem path for a cache key."""
    if folder_key == "all":
        return None
    alias, relative = _split_virtual_media_path(folder_key)
    root = MEDIA_ROOT_LOOKUP.get(alias)
    if root is None:
        return None
    target_dir = (root.path / relative).resolve()
    try:
        target_dir.relative_to(root.path.resolve())
    except ValueError:
        logger.debug("Rejected folder key '%s' because it escapes media root '%s'", folder_key, root.path)
        return None
    return root, target_dir


def _scan_root_for_cache(
    root: config_manager.MediaRoot,
    base_path: Path,
) -> Tuple[List[Dict[str, str]], Dict[str, float]]:
    dir_markers: Dict[str, float] = {}
    base_key = os.fspath(base_path)
    try:
        dir_markers[base_key] = os.stat(base_path).st_mtime
    except FileNotFoundError:
        dir_markers[base_key] = 0.0
        return [], dir_markers
    except OSError:
        dir_markers[base_key] = 0.0
        return [], dir_markers

    media: List[Dict[str, str]] = []
    for walk_root, dirnames, files in os.walk(base_path):
        dirnames[:] = [name for name in dirnames if not _should_ignore_media_name(name)]
        walk_path = Path(walk_root)
        walk_key = os.fspath(walk_path)
        if walk_path != base_path:
            try:
                dir_markers[walk_key] = os.stat(walk_path).st_mtime
            except OSError:
                dir_markers[walk_key] = time.time()
        for file_name in files:
            if _should_ignore_media_name(file_name):
                continue
            ext = os.path.splitext(file_name)[1].lower()
            if ext not in MEDIA_EXTENSIONS:
                continue
            candidate_path = walk_path / file_name
            try:
                relative_path = candidate_path.resolve().relative_to(root.path.resolve())
            except Exception:
                continue
            virtual_path = _build_virtual_media_path(root.alias, relative_path.as_posix())
            kind = "video" if ext in VIDEO_EXTENSIONS else "image"
            media.append({
                "path": virtual_path,
                "kind": kind,
                "extension": ext,
            })
    media.sort(key=lambda item: item["path"].lower())
    return media, dir_markers


def _scan_folder_for_cache(folder_key: str) -> Tuple[List[Dict[str, str]], Dict[str, float]]:
    """Scan configured media directories and build the cached payload for a folder key."""
    if folder_key == "all":
        combined_media: List[Dict[str, str]] = []
        combined_markers: Dict[str, float] = {}
        for root in AVAILABLE_MEDIA_ROOTS:
            media_entries, dir_markers = _scan_root_for_cache(root, root.path)
            combined_media.extend(media_entries)
            combined_markers.update(dir_markers)
        combined_media.sort(key=lambda item: item["path"].lower())
        return combined_media, combined_markers

    resolved = _resolve_folder_path(folder_key)
    if resolved is None:
        return [], {}
    root, target_dir = resolved
    return _scan_root_for_cache(root, target_dir)


def _directory_markers_changed(markers: Dict[str, float]) -> bool:
    """Return True when any tracked directory timestamp has diverged."""
    for path, previous_mtime in markers.items():
        try:
            current_mtime = os.stat(path).st_mtime
        except FileNotFoundError:
            return True
        except OSError:
            return True
        if current_mtime != previous_mtime:
            return True
    return False


def refresh_image_cache(folder: str = "all", force: bool = False) -> List[str]:
    """Return the cached image list for a folder, refreshing if anything changed."""
    folder_key = _normalize_folder_key(folder)

    with IMAGE_CACHE_LOCK:
        cached_entry = IMAGE_CACHE.get(folder_key)
        if cached_entry:
            cached_images = list(cached_entry.get("images", []))
            markers_snapshot = dict(cached_entry.get("dir_markers", {}))
        else:
            cached_images = []
            markers_snapshot = None

    needs_refresh = force or cached_entry is None
    if not needs_refresh and markers_snapshot is not None:
        if _directory_markers_changed(markers_snapshot):
            needs_refresh = True

    if not needs_refresh:
        return cached_images

    media, dir_markers = _scan_folder_for_cache(folder_key)
    images = [item["path"] for item in media if item.get("kind") == "image"]
    entry = {
        "images": images,
        "media": media,
        "dir_markers": dir_markers,
        "last_updated": time.time(),
    }
    with IMAGE_CACHE_LOCK:
        IMAGE_CACHE[folder_key] = entry
    return list(images)


def initialize_image_cache() -> None:
    """Warm the cache for the root folder and any existing subfolders on startup."""
    refresh_image_cache("all", force=True)
    for root in AVAILABLE_MEDIA_ROOTS:
        try:
            with os.scandir(root.path) as scan:
                for entry in scan:
                    if not entry.is_dir():
                        continue
                    if _should_ignore_media_name(entry.name):
                        continue
                    folder_key = _build_virtual_media_path(root.alias, entry.name)
                    refresh_image_cache(folder_key, force=True)
        except OSError:
            continue


initialize_image_cache()

AI_FALLBACK_DEFAULTS: Dict[str, Any] = {
    "prompt": "",
    "negative_prompt": "",
    "model": AI_DEFAULT_MODEL,
    "sampler": AI_DEFAULT_SAMPLER,
    "steps": AI_DEFAULT_STEPS,
    "cfg_scale": AI_DEFAULT_CFG,
    "width": AI_DEFAULT_WIDTH,
    "height": AI_DEFAULT_HEIGHT,
    "samples": AI_DEFAULT_SAMPLES,
    "seed": "random",
    "nsfw": False,
    "censor_nsfw": False,
    "save_output": AI_DEFAULT_PERSIST,
    "post_processing": [],
    "hires_fix": False,
    "hires_fix_denoising_strength": None,
    "denoising_strength": None,
    "facefixer_strength": None,
    "clip_skip": None,
    "karras": False,
    "tiling": False,
    "transparent": False,
    "loras": [],
    "style": "",
    "trusted_workers": False,
    "validated_backends": True,
    "slow_workers": True,
    "extra_slow_workers": False,
    "disable_batching": False,
    "allow_downgrade": False,
    "timeout": AI_TIMEOUT,
    "auto_generate_mode": "off",
    "auto_generate_interval_value": AUTO_GENERATE_DEFAULT_INTERVAL_VALUE,
    "auto_generate_interval_unit": "minutes",
    "auto_generate_clock_time": None,
}


def default_ai_settings() -> Dict[str, Any]:
    """Return the current default AI settings (global overrides + fallbacks)."""

    overrides: Optional[Dict[str, Any]] = None
    settings_ref = globals().get("settings")
    if isinstance(settings_ref, dict):
        candidate = settings_ref.get("_ai_defaults")
        if isinstance(candidate, dict):
            overrides = candidate

    merged = deepcopy(AI_FALLBACK_DEFAULTS)
    if overrides:
        merged = _sanitize_ai_settings(overrides, merged, defaults=AI_FALLBACK_DEFAULTS)
    return deepcopy(merged)


def default_ai_state() -> Dict[str, Any]:
    return {
        "status": "idle",
        "job_id": None,
        "message": None,
        "queue_position": None,
        "wait_time": None,
        "images": [],
        "persisted": AI_DEFAULT_PERSIST,
        "last_updated": None,
        "error": None,
        "last_auto_trigger": None,
        "next_auto_trigger": None,
        "last_trigger_source": None,
        "last_auto_error": None,
    }


def ensure_ai_defaults(conf: Dict[str, Any]) -> None:
    ai_defaults = default_ai_settings()
    ai_settings = conf.get(AI_SETTINGS_KEY)
    if not isinstance(ai_settings, dict):
        conf[AI_SETTINGS_KEY] = deepcopy(ai_defaults)
    else:
        for key, value in ai_defaults.items():
            if key not in ai_settings:
                ai_settings[key] = deepcopy(value) if isinstance(value, (dict, list)) else value

    ai_state = conf.get(AI_STATE_KEY)
    if not isinstance(ai_state, dict):
        conf[AI_STATE_KEY] = default_ai_state()
    else:
        state_defaults = default_ai_state()
        for key, value in state_defaults.items():
            ai_state.setdefault(key, value)
    ai_state_ref = conf[AI_STATE_KEY]
    ai_state_ref["next_auto_trigger"] = _normalize_timer_label(ai_state_ref.get("next_auto_trigger"))
    ai_state_ref["last_auto_trigger"] = _normalize_timer_label(ai_state_ref.get("last_auto_trigger"))

    if "_ai_customized" not in conf:
        conf["_ai_customized"] = not _ai_settings_match_defaults(conf[AI_SETTINGS_KEY], defaults=ai_defaults)


def ensure_background_defaults(conf: Dict[str, Any]) -> None:
    if not isinstance(conf, dict):
        return
    enabled_raw = conf.get("background_blur_enabled")
    conf["background_blur_enabled"] = _coerce_bool(enabled_raw, False)
    amount_raw = conf.get("background_blur_amount", 50)
    amount_val = _coerce_int(amount_raw, 50)
    conf["background_blur_amount"] = max(0, min(100, amount_val))


PICSUM_SETTINGS_KEY = "picsum_settings"
PICSUM_DEFAULT_WIDTH = 1920
PICSUM_DEFAULT_HEIGHT = 1080
PICSUM_MIN_DIMENSION = 16
PICSUM_MAX_DIMENSION = 4096
PICSUM_MIN_BLUR = 0
PICSUM_MAX_BLUR = 10
PICSUM_DEFAULT_BLUR = 0
PICSUM_AUTO_MODES = {"off", "timer", "clock"}
PICSUM_DEFAULT_AUTO_MODE = "off"
PICSUM_DEFAULT_INTERVAL_VALUE = 10.0
PICSUM_DEFAULT_INTERVAL_UNIT = "minutes"


def default_picsum_settings() -> Dict[str, Any]:
    return {
        "width": PICSUM_DEFAULT_WIDTH,
        "height": PICSUM_DEFAULT_HEIGHT,
        "blur": PICSUM_DEFAULT_BLUR,
        "grayscale": False,
        "seed": None,
        "auto_mode": PICSUM_DEFAULT_AUTO_MODE,
        "auto_interval_value": PICSUM_DEFAULT_INTERVAL_VALUE,
        "auto_interval_unit": PICSUM_DEFAULT_INTERVAL_UNIT,
        "auto_clock_time": "",
        "next_auto_trigger": None,
        "last_auto_trigger": None,
    }


def _normalize_picsum_seed(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    return cleaned[:64]


def _sanitize_picsum_settings(
    value: Any, defaults: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    baseline = deepcopy(defaults) if defaults else default_picsum_settings()
    if not isinstance(value, dict):
        return baseline
    width_default = baseline.get("width", PICSUM_DEFAULT_WIDTH)
    height_default = baseline.get("height", PICSUM_DEFAULT_HEIGHT)
    blur_default = baseline.get("blur", PICSUM_DEFAULT_BLUR)
    grayscale_default = bool(baseline.get("grayscale", False))
    seed_default = baseline.get("seed")
    auto_mode_default = baseline.get("auto_mode", PICSUM_DEFAULT_AUTO_MODE)
    auto_interval_default = float(baseline.get("auto_interval_value", PICSUM_DEFAULT_INTERVAL_VALUE))
    auto_unit_default = baseline.get("auto_interval_unit", PICSUM_DEFAULT_INTERVAL_UNIT)
    auto_clock_default = baseline.get("auto_clock_time", "")
    next_auto_default = baseline.get("next_auto_trigger")
    last_auto_default = baseline.get("last_auto_trigger")

    width_val = _coerce_int(value.get("width"), width_default)
    height_val = _coerce_int(value.get("height"), height_default)
    blur_val = _coerce_int(value.get("blur"), blur_default)
    grayscale_val = _coerce_bool(value.get("grayscale"), grayscale_default)
    seed_in_payload = "seed" in value
    seed_val = _normalize_picsum_seed(value.get("seed"))
    result = {
        "width": max(PICSUM_MIN_DIMENSION, min(PICSUM_MAX_DIMENSION, width_val)),
        "height": max(PICSUM_MIN_DIMENSION, min(PICSUM_MAX_DIMENSION, height_val)),
        "blur": max(PICSUM_MIN_BLUR, min(PICSUM_MAX_BLUR, blur_val)),
        "grayscale": bool(grayscale_val),
        "seed": seed_val if seed_in_payload else seed_default,
    }
    auto_mode_raw = value.get("auto_mode")
    auto_mode = str(auto_mode_raw).strip().lower() if isinstance(auto_mode_raw, str) else auto_mode_default
    if auto_mode not in PICSUM_AUTO_MODES:
        auto_mode = PICSUM_DEFAULT_AUTO_MODE

    interval_value = _coerce_float(value.get("auto_interval_value"), auto_interval_default)
    if interval_value is None or interval_value <= 0:
        interval_value = PICSUM_DEFAULT_INTERVAL_VALUE

    interval_unit_raw = value.get("auto_interval_unit")
    interval_unit = str(interval_unit_raw).strip().lower() if isinstance(interval_unit_raw, str) else auto_unit_default
    if interval_unit not in AUTO_GENERATE_INTERVAL_UNITS:
        interval_unit = PICSUM_DEFAULT_INTERVAL_UNIT

    clock_raw = value.get("auto_clock_time")
    if clock_raw is None:
        clock_time = _normalize_clock_time(auto_clock_default)
    else:
        clock_time = _normalize_clock_time(clock_raw)
    if clock_time is None:
        clock_time = ""

    next_auto_raw = value.get("next_auto_trigger")
    next_auto = _normalize_timer_label(next_auto_raw)
    if not next_auto:
        next_auto = _normalize_timer_label(next_auto_default)

    last_auto_raw = value.get("last_auto_trigger")
    last_auto = _normalize_timer_label(last_auto_raw)
    if not last_auto:
        last_auto = _normalize_timer_label(last_auto_default)

    result.update(
        {
            "auto_mode": auto_mode,
            "auto_interval_value": float(interval_value),
            "auto_interval_unit": interval_unit,
            "auto_clock_time": clock_time,
            "next_auto_trigger": next_auto,
            "last_auto_trigger": last_auto,
        }
    )
    return result


def ensure_picsum_defaults(conf: Dict[str, Any]) -> None:
    defaults = default_picsum_settings()
    current = conf.get(PICSUM_SETTINGS_KEY)
    if isinstance(current, dict):
        conf[PICSUM_SETTINGS_KEY] = _sanitize_picsum_settings(current, defaults)
    else:
        conf[PICSUM_SETTINGS_KEY] = deepcopy(defaults)
    if "_picsum_seed_custom" not in conf:
        conf["_picsum_seed_custom"] = False
    else:
        conf["_picsum_seed_custom"] = bool(conf["_picsum_seed_custom"])


def _normalize_tag_name(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    cleaned = re.sub(r"\s+", " ", value.strip())
    if not cleaned:
        return None
    if len(cleaned) > TAG_MAX_LENGTH:
        cleaned = cleaned[:TAG_MAX_LENGTH].rstrip()
    return cleaned


def _sanitize_stream_tags(value: Any) -> List[str]:
    if not value:
        return []
    if isinstance(value, str):
        candidates: List[Any] = [value]
    elif isinstance(value, (list, tuple, set)):
        candidates = list(value)
    else:
        return []
    result: List[str] = []
    seen = set()
    for item in candidates:
        normalized = _normalize_tag_name(item)
        if not normalized:
            continue
        key = normalized.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(normalized)
    return result


def _normalize_tag_collection(values: Any) -> List[str]:
    if not isinstance(values, (list, tuple, set)):
        iterable: List[Any] = []
    else:
        iterable = list(values)
    cleaned = _sanitize_stream_tags(iterable)
    cleaned.sort(key=str.lower)
    return cleaned


def get_global_tags() -> List[str]:
    tags = settings.get(GLOBAL_TAGS_KEY)
    if not isinstance(tags, list):
        tags = []
        settings[GLOBAL_TAGS_KEY] = tags
    return tags


def register_global_tags(new_tags: List[str]) -> bool:
    if not new_tags:
        return False
    tags = get_global_tags()
    existing = {tag.casefold(): tag for tag in tags}
    changed = False
    for tag in new_tags:
        key = tag.casefold()
        if key not in existing:
            tags.append(tag)
            existing[key] = tag
            changed = True
    if changed:
        tags.sort(key=str.lower)
    return changed


def ensure_tag_defaults(conf: Dict[str, Any]) -> bool:
    if not isinstance(conf, dict):
        return False
    current = conf.get(TAG_KEY)
    normalized = _sanitize_stream_tags(current)
    if normalized != current:
        conf[TAG_KEY] = normalized
        changed = True
    else:
        conf.setdefault(TAG_KEY, [])
        changed = False
    if register_global_tags(conf.get(TAG_KEY, [])):
        changed = True
    return changed


def ensure_settings_integrity(data: Dict[str, Any]) -> bool:
    changed = False
    tags = data.get(GLOBAL_TAGS_KEY)
    normalized_tags = _normalize_tag_collection(tags)
    if tags != normalized_tags:
        data[GLOBAL_TAGS_KEY] = normalized_tags
        changed = True
    for key, conf in list(data.items()):
        if key.startswith("_"):
            continue
        if not isinstance(conf, dict):
            continue
        if ensure_tag_defaults(conf):
            changed = True
    return changed


def _ensure_dir(path: Path) -> Path:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning("Unable to ensure directory %s: %s", path, exc)
    return path


CLOCK_TIME_PATTERN = re.compile(r"^(?:[01]\d|2[0-3]):[0-5]\d$")


def _normalize_clock_time(value: Any) -> Optional[str]:
    if isinstance(value, str):
        trimmed = value.strip()
        if not trimmed:
            return None
        if CLOCK_TIME_PATTERN.match(trimmed):
            return trimmed
    return None


def _sanitize_ai_settings(
    payload: Dict[str, Any],
    base: Optional[Dict[str, Any]] = None,
    defaults: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if defaults is None:
        defaults = default_ai_settings()
    else:
        defaults = deepcopy(defaults)
    current = dict(base or defaults)
    bool_keys = {
        "save_output",
        "nsfw",
        "censor_nsfw",
        "hires_fix",
        "karras",
        "tiling",
        "transparent",
        "trusted_workers",
        "validated_backends",
        "slow_workers",
        "extra_slow_workers",
        "disable_batching",
        "allow_downgrade",
    }
    for key, value in payload.items():
        if key not in defaults:
            continue
        if key in {"width", "height"}:
            try:
                coerced = int(value)
            except (TypeError, ValueError):
                continue
            current[key] = max(64, coerced)
        elif key == "steps":
            maybe = _maybe_int(value)
            if maybe is None:
                continue
            current[key] = max(1, maybe)
        elif key == "samples":
            maybe = _maybe_int(value)
            if maybe is None:
                continue
            current[key] = max(1, min(10, maybe))
        elif key == "cfg_scale":
            maybe = _maybe_float(value)
            if maybe is None:
                continue
            current[key] = maybe
        elif key == "timeout":
            maybe = _maybe_float(value)
            if maybe is None:
                continue
            current[key] = max(0.0, maybe)
        elif key == "auto_generate_mode":
            normalized = value.strip().lower() if isinstance(value, str) else ""
            if normalized not in AUTO_GENERATE_MODES:
                fallback = defaults.get(key, "off")
                if isinstance(fallback, str):
                    fallback = fallback.strip().lower()
                else:
                    fallback = "off"
                if fallback not in AUTO_GENERATE_MODES:
                    fallback = "off"
                normalized = fallback
            current[key] = normalized
        elif key == "auto_generate_interval_unit":
            unit_value = value.strip().lower() if isinstance(value, str) else ""
            if unit_value not in AUTO_GENERATE_INTERVAL_UNITS:
                fallback = defaults.get(key, "minutes")
                if isinstance(fallback, str):
                    fallback = fallback.strip().lower()
                else:
                    fallback = "minutes"
                if fallback not in AUTO_GENERATE_INTERVAL_UNITS:
                    fallback = "minutes"
                unit_value = fallback
            current[key] = unit_value
        elif key == "auto_generate_interval_value":
            maybe = _maybe_float(value)
            if maybe is None or maybe <= 0:
                fallback = defaults.get(key, AUTO_GENERATE_DEFAULT_INTERVAL_VALUE)
                try:
                    maybe = float(fallback)
                except (TypeError, ValueError):
                    maybe = AUTO_GENERATE_DEFAULT_INTERVAL_VALUE
            current[key] = maybe
        elif key == "auto_generate_clock_time":
            normalized = _normalize_clock_time(value)
            if normalized is None:
                normalized = _normalize_clock_time(defaults.get(key))
            current[key] = normalized
        elif key in bool_keys:
            current[key] = _coerce_bool(value, defaults[key])
        elif key == "seed":
            current[key] = str(value).strip() if value not in (None, "") else "random"
        elif key == "post_processing":
            current[key] = _sanitize_post_processing(value)
        elif key == "loras":
            current[key] = _sanitize_loras(value)
        elif key == "clip_skip":
            if value in (None, "", "none", "auto", "default"):
                current[key] = None
                continue
            maybe = _maybe_int(value)
            if maybe is None:
                continue
            lower, upper = STABLE_HORDE_CLIP_SKIP_RANGE
            current[key] = int(_clamp(maybe, lower, upper))
        elif key in {"facefixer_strength"}:
            if value in (None, "", "none"):
                current[key] = None
                continue
            maybe = _maybe_float(value)
            if maybe is None:
                continue
            low, high = STABLE_HORDE_STRENGTH_RANGE
            current[key] = _clamp(maybe, low, high)
        elif key in {"denoising_strength", "hires_fix_denoising_strength"}:
            if value in (None, "", "none"):
                current[key] = None
                continue
            maybe = _maybe_float(value)
            if maybe is None:
                continue
            low, high = STABLE_HORDE_DENOISE_RANGE
            current[key] = _clamp(maybe, low, high)
        elif isinstance(value, str):
            current[key] = value.strip()
        else:
            current[key] = value
    return current

def _sanitize_ai_presets(raw: Any) -> Dict[str, Dict[str, Any]]:
    sanitized: Dict[str, Dict[str, Any]] = {}
    if isinstance(raw, dict):
        for name, payload in raw.items():
            if not isinstance(name, str):
                continue
            trimmed = name.strip()
            if not trimmed:
                continue
            if not isinstance(payload, dict):
                continue
            sanitized[trimmed] = _sanitize_ai_settings(payload, defaults=AI_FALLBACK_DEFAULTS)
    return sanitized


def _sorted_presets(presets: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return dict(sorted(presets.items(), key=lambda item: item[0].lower()))


def ensure_ai_presets_storage() -> Dict[str, Dict[str, Any]]:
    raw = settings.get(AI_PRESETS_KEY) if isinstance(settings, dict) else {}
    sanitized = _sanitize_ai_presets(raw)
    ordered = _sorted_presets(sanitized)
    settings[AI_PRESETS_KEY] = ordered
    return ordered


def _ai_settings_match_defaults(candidate: Dict[str, Any], defaults: Optional[Dict[str, Any]] = None) -> bool:
    if not isinstance(candidate, dict):
        return False
    baseline = defaults or default_ai_settings()
    for key in AI_FALLBACK_DEFAULTS.keys():
        if candidate.get(key) != baseline.get(key):
            return False
    return True



def default_stream_config():
    """Return the default configuration for a new stream."""
    return {
        "mode": "random",
        "media_mode": MEDIA_MODE_IMAGE,
        "folder": "all",
        "selected_image": None,
        "selected_media_kind": None,
        "duration": 5,
        "video_playback_mode": "duration",
        "video_volume": 1.0,
        "shuffle": True,
        "hide_nsfw": False,
        "stream_url": None,
        "yt_cc": False,
        "yt_mute": True,
        "yt_quality": "auto",
        "image_quality": "auto",
        "label": "",
        TAG_KEY: [],
        "background_blur_enabled": False,
        "background_blur_amount": 50,
        "embed_metadata": None,
        PICSUM_SETTINGS_KEY: default_picsum_settings(),
        "_picsum_seed_custom": False,
        AI_SETTINGS_KEY: default_ai_settings(),
        AI_STATE_KEY: default_ai_state(),
        "_ai_customized": False,
    }


def _sanitize_imported_stream_config(stream_id: str, raw_conf: Any) -> Dict[str, Any]:
    if not isinstance(raw_conf, dict):
        raise ValueError(f"Stream '{stream_id}' configuration must be an object.")
    conf = default_stream_config()
    for key, value in raw_conf.items():
        conf[key] = deepcopy(value) if isinstance(value, (dict, list)) else value
    mode_raw = conf.get('mode')
    mode = mode_raw.strip().lower() if isinstance(mode_raw, str) else conf['mode']
    if mode not in {'random', 'specific', 'livestream', AI_MODE, MEDIA_MODE_PICSUM}:
        mode = conf['mode']
    conf['mode'] = mode
    folder_raw = conf.get('folder')
    conf['folder'] = folder_raw.strip() if isinstance(folder_raw, str) and folder_raw.strip() else 'all'
    conf['duration'] = max(1, _coerce_int(conf.get('duration'), 5))
    conf['shuffle'] = _coerce_bool(conf.get('shuffle'), True)
    conf['hide_nsfw'] = _coerce_bool(conf.get('hide_nsfw'), False)
    conf['yt_cc'] = _coerce_bool(conf.get('yt_cc'), False)
    conf['yt_mute'] = _coerce_bool(conf.get('yt_mute'), True)
    conf['background_blur_enabled'] = _coerce_bool(conf.get('background_blur_enabled'), False)
    conf['background_blur_amount'] = max(0, min(100, _coerce_int(conf.get('background_blur_amount'), 50)))
    quality_raw = conf.get('image_quality')
    quality = quality_raw.strip().lower() if isinstance(quality_raw, str) else 'auto'
    if quality not in IMAGE_QUALITY_CHOICES:
        quality = 'auto'
    conf['image_quality'] = quality
    playback_raw = conf.get('video_playback_mode')
    playback = playback_raw.strip().lower() if isinstance(playback_raw, str) else 'duration'
    if playback not in VIDEO_PLAYBACK_MODES:
        playback = 'duration'
    conf['video_playback_mode'] = playback
    volume_raw = conf.get('video_volume')
    try:
        volume = float(volume_raw)
    except (TypeError, ValueError):
        volume = 1.0
    conf['video_volume'] = max(0.0, min(1.0, volume))
    stream_url_raw = conf.get('stream_url')
    if isinstance(stream_url_raw, str):
        trimmed = stream_url_raw.strip()
        conf['stream_url'] = trimmed or None
    else:
        conf['stream_url'] = None
    label_raw = conf.get('label')
    conf['label'] = label_raw.strip() if isinstance(label_raw, str) else ''
    conf['include_in_global'] = _coerce_bool(conf.get('include_in_global'), True)
    tags_payload = conf.get(TAG_KEY, [])
    conf[TAG_KEY] = _sanitize_stream_tags(tags_payload)
    selected_image = conf.get('selected_image')
    if isinstance(selected_image, str):
        selected_image = selected_image.strip() or None
    else:
        selected_image = None
    conf['selected_image'] = selected_image
    selected_kind_raw = conf.get('selected_media_kind')
    if isinstance(selected_kind_raw, str):
        selected_kind = selected_kind_raw.strip().lower()
    else:
        selected_kind = None
    if selected_kind not in {'image', 'video'}:
        selected_kind = _detect_media_kind(selected_image) if selected_image else None
    conf['selected_media_kind'] = selected_kind
    media_mode_raw = conf.get('media_mode')
    media_mode = media_mode_raw.strip().lower() if isinstance(media_mode_raw, str) else ''
    if media_mode not in MEDIA_MODE_CHOICES:
        media_mode = _infer_media_mode(conf)
    conf['media_mode'] = media_mode
    if media_mode == MEDIA_MODE_AI:
        conf['mode'] = AI_MODE
    elif media_mode == MEDIA_MODE_LIVESTREAM:
        conf['mode'] = 'livestream'
    else:
        allowed = MEDIA_MODE_VARIANTS.get(media_mode, {'random', 'specific'})
        if conf['mode'] not in allowed:
            conf['mode'] = 'random' if 'random' in allowed else next(iter(sorted(allowed)))
        if media_mode == MEDIA_MODE_VIDEO:
            if conf['mode'] == 'specific':
                conf['video_playback_mode'] = 'loop'
            elif conf['mode'] == 'random' and conf['video_playback_mode'] == 'loop':
                conf['video_playback_mode'] = 'duration'
    ai_settings_raw = conf.get(AI_SETTINGS_KEY)
    if isinstance(ai_settings_raw, dict):
        conf[AI_SETTINGS_KEY] = _sanitize_ai_settings(ai_settings_raw, defaults=AI_FALLBACK_DEFAULTS)
    else:
        conf[AI_SETTINGS_KEY] = deepcopy(AI_FALLBACK_DEFAULTS)
    ai_state_raw = conf.get(AI_STATE_KEY)
    state_defaults = default_ai_state()
    if isinstance(ai_state_raw, dict):
        for key in state_defaults.keys():
            if key in ai_state_raw:
                state_defaults[key] = ai_state_raw[key]
    conf[AI_STATE_KEY] = state_defaults
    conf['_ai_customized'] = not _ai_settings_match_defaults(conf[AI_SETTINGS_KEY], defaults=AI_FALLBACK_DEFAULTS)
    picsum_raw = conf.get(PICSUM_SETTINGS_KEY)
    conf[PICSUM_SETTINGS_KEY] = _sanitize_picsum_settings(picsum_raw, defaults=default_picsum_settings())
    conf['_picsum_seed_custom'] = bool(conf.get('_picsum_seed_custom', False))
    conf["embed_metadata"] = _sanitize_embed_metadata(conf.get("embed_metadata"))
    ensure_background_defaults(conf)
    return conf


def _sanitize_group_collection_for_import(raw_groups: Any, valid_streams: Set[str]) -> Dict[str, Any]:
    sanitized: Dict[str, Any] = {}
    if not isinstance(raw_groups, dict):
        return sanitized
    for name, payload in raw_groups.items():
        if not isinstance(name, str):
            continue
        trimmed = name.strip()
        if not trimmed:
            continue
        if isinstance(payload, dict):
            streams_raw = payload.get('streams')
            cleaned: List[str] = []
            if isinstance(streams_raw, (list, tuple)):
                for entry in streams_raw:
                    if not isinstance(entry, str):
                        continue
                    candidate = entry.strip()
                    if candidate in valid_streams:
                        cleaned.append(candidate)
            layout = _normalize_group_layout(payload.get('layout'))
            entry: Dict[str, Any] = {'streams': cleaned}
            if layout:
                entry['layout'] = layout
            sanitized[trimmed] = entry
        elif isinstance(payload, (list, tuple)):
            cleaned = []
            for entry in payload:
                if not isinstance(entry, str):
                    continue
                candidate = entry.strip()
                if candidate in valid_streams:
                    cleaned.append(candidate)
            sanitized[trimmed] = cleaned
    return sanitized


def _prepare_settings_import(data: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    if not isinstance(data, dict):
        raise ValueError('Import payload must be a JSON object.')
    sanitized: Dict[str, Any] = {}
    warnings: List[str] = []
    stream_ids: List[str] = []
    for key, value in data.items():
        if not isinstance(key, str) or key.startswith('_'):
            continue
        stream_id = key.strip()
        if not stream_id:
            warnings.append('Skipped stream with empty identifier.')
            continue
        sanitized[stream_id] = _sanitize_imported_stream_config(stream_id, value)
        stream_ids.append(stream_id)
    valid_streams = set(stream_ids)
    tags_raw = data.get(GLOBAL_TAGS_KEY)
    sanitized[GLOBAL_TAGS_KEY] = _normalize_tag_collection(tags_raw)
    notes_raw = data.get('_notes')
    sanitized['_notes'] = notes_raw if isinstance(notes_raw, str) else ''
    defaults_raw = data.get('_ai_defaults')
    if isinstance(defaults_raw, dict):
        sanitized['_ai_defaults'] = _sanitize_ai_settings(defaults_raw, defaults=AI_FALLBACK_DEFAULTS)
    else:
        sanitized['_ai_defaults'] = deepcopy(AI_FALLBACK_DEFAULTS)
    presets_raw = data.get(AI_PRESETS_KEY)
    sanitized[AI_PRESETS_KEY] = _sorted_presets(_sanitize_ai_presets(presets_raw))
    groups_raw = data.get('_groups')
    sanitized['_groups'] = _sanitize_group_collection_for_import(groups_raw, valid_streams)
    for key, value in data.items():
        if not isinstance(key, str) or not key.startswith('_'):
            continue
        if key in {GLOBAL_TAGS_KEY, '_notes', '_ai_defaults', '_groups', AI_PRESETS_KEY}:
            continue
        sanitized[key] = deepcopy(value)
    return sanitized, warnings


def load_settings():
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, "r") as f:
            return json.load(f)
    # Start with no streams; dashboard can add them dynamically.
    return {}

def save_settings(data):
    with open(SETTINGS_FILE, "w") as f:
        json.dump(data, f, indent=4)


@app.route("/settings/export", methods=["GET"])
def export_settings_download():
    """Return current settings as a downloadable JSON attachment."""
    snapshot = deepcopy(settings)
    buffer = io.BytesIO()
    buffer.write(json.dumps(snapshot, indent=2, sort_keys=True).encode("utf-8"))
    buffer.seek(0)
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    filename = f"echo-settings-{timestamp}.json"
    response = send_file(
        buffer,
        mimetype="application/json",
        as_attachment=True,
        download_name=filename,
        max_age=0,
    )
    response.headers["Cache-Control"] = "no-store, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Export-Filename"] = filename
    return response


@app.route("/settings/import", methods=["POST"])
def import_settings():
    upload = request.files.get("file")
    payload: Any
    if upload:
        raw_bytes = upload.read()
        if not raw_bytes:
            return jsonify({"error": "Uploaded file is empty."}), 400
        try:
            payload = json.loads(raw_bytes.decode("utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return jsonify({"error": "Uploaded file is not valid JSON."}), 400
    else:
        payload = request.get_json(silent=True)
        if payload is None:
            raw_body = request.get_data(cache=False, as_text=True)
            if raw_body:
                try:
                    payload = json.loads(raw_body)
                except json.JSONDecodeError:
                    payload = None
        if payload is None:
            return jsonify({"error": "No JSON payload received."}), 400

    try:
        snapshot, warnings = _prepare_settings_import(payload)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    new_streams = {key for key in snapshot.keys() if isinstance(key, str) and not key.startswith("_")}
    existing_streams = {key for key in settings.keys() if not key.startswith("_")}
    added = sorted(new_streams - existing_streams)
    removed = sorted(existing_streams - new_streams)
    updated = sorted(new_streams & existing_streams)

    settings.clear()
    settings.update(snapshot)
    settings.setdefault(GLOBAL_TAGS_KEY, [])
    settings.setdefault("_notes", "")
    settings.setdefault("_groups", {})
    settings.setdefault("_ai_defaults", deepcopy(AI_FALLBACK_DEFAULTS))
    settings.setdefault(AI_PRESETS_KEY, {})

    settings["_ai_defaults"] = _sanitize_ai_settings(settings.get("_ai_defaults", {}), defaults=AI_FALLBACK_DEFAULTS)
    settings[AI_PRESETS_KEY] = _sorted_presets(_sanitize_ai_presets(settings.get(AI_PRESETS_KEY, {})))
    settings["_groups"] = _sanitize_group_collection_for_import(settings.get("_groups", {}), new_streams)

    for stream_id in new_streams:
        conf = settings[stream_id]
        conf.pop('_ai_customized', None)
        ensure_background_defaults(conf)
        ensure_ai_defaults(conf)
        ensure_picsum_defaults(conf)

    ensure_settings_integrity(settings)
    save_settings(settings)

    with ai_jobs_lock:
        for stream_id in removed:
            ai_jobs.pop(stream_id, None)
            ai_job_controls.pop(stream_id, None)

    for stream_id in removed:
        _cleanup_temp_outputs(stream_id)
        with STREAM_RUNTIME_LOCK:
            STREAM_RUNTIME_STATE.pop(stream_id, None)
        if auto_scheduler is not None:
            auto_scheduler.remove(stream_id)
        if playback_manager is not None:
            playback_manager.remove_stream(stream_id)

    tags_snapshot = get_global_tags()
    for stream_id in new_streams:
        conf = settings[stream_id]
        if conf.get("selected_image") and not conf.get("selected_media_kind"):
            conf["selected_media_kind"] = _detect_media_kind(conf.get("selected_image"))
        _update_stream_runtime_state(
            stream_id,
            path=conf.get("selected_image"),
            kind=conf.get("selected_media_kind"),
            media_mode=conf.get("media_mode"),
            stream_url=conf.get("stream_url"),
            source="settings_import",
        )
        if playback_manager is not None:
            playback_manager.update_stream_config(stream_id, conf)
        if auto_scheduler is not None:
            auto_scheduler.reschedule(stream_id)

    safe_emit(
        "streams_changed",
        {
            "action": "import",
            "added": added,
            "removed": removed,
            "updated": updated,
        },
    )
    for stream_id in new_streams:
        safe_emit(
            "refresh",
            {"stream_id": stream_id, "config": settings[stream_id], "tags": tags_snapshot},
        )

    response = {
        "success": True,
        "streams": sorted(new_streams),
        "added": added,
        "removed": removed,
        "updated": updated,
        "warnings": warnings,
        "tags": tags_snapshot,
        "ai_defaults": settings.get("_ai_defaults", {}),
        "groups": settings.get("_groups", {}),
    }
    return jsonify(response)



def load_config() -> Dict[str, Any]:
    return config_manager.load_config()


def _coerce_float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_int(value, default):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _coerce_bool(value, default):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return default


def _timer_snap_enabled() -> bool:
    return _as_bool(CONFIG.get("TIMER_SNAP_ENABLED"), False)


def _format_timer_label(moment: Optional[datetime]) -> Optional[str]:
    if moment is None:
        return None
    return timer_manager.format_display_time(moment)


def _normalize_timer_label(value: Any) -> Optional[str]:
    normalized = timer_manager.normalize_display_label(value)
    return normalized


def _maybe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _maybe_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _clamp(value, lower=None, upper=None):
    if value is None:
        return None
    if lower is not None and value < lower:
        value = lower
    if upper is not None and value > upper:
        value = upper
    return value


def _sanitize_post_processing(value):
    if not value:
        return []
    if isinstance(value, str):
        candidates = [part.strip() for part in value.split(',')]
    elif isinstance(value, (list, tuple, set)):
        candidates = []
        for item in value:
            if isinstance(item, str):
                candidates.append(item.strip())
    else:
        return []
    seen = set()
    cleaned = []
    for name in candidates:
        if not name:
            continue
        if name not in STABLE_HORDE_POST_PROCESSORS:
            continue
        if name in seen:
            continue
        cleaned.append(name)
        seen.add(name)
    return cleaned


def _sanitize_loras(value):
    if not value:
        return []
    raw_list = value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except Exception:
            parsed = [part.strip() for part in value.split(',') if part.strip()]
        raw_list = parsed
    if not isinstance(raw_list, (list, tuple)):
        return []
    cleaned = []
    for item in raw_list:
        if isinstance(item, str):
            entry = {"name": item}
        elif isinstance(item, dict):
            entry = dict(item)
        else:
            continue
        name = str(entry.get("name") or '').strip()
        if not name:
            continue
        model_strength = _maybe_float(entry.get("model"))
        clip_strength = _maybe_float(entry.get("clip"))
        inject_trigger = entry.get("inject_trigger")
        if isinstance(inject_trigger, str):
            inject_trigger = inject_trigger.strip() or None
        if model_strength is None:
            model_strength = 1.0
        model_strength = _clamp(model_strength, -5.0, 5.0)
        if clip_strength is None:
            clip_strength = model_strength
        clip_strength = _clamp(clip_strength, -5.0, 5.0)
        cleaned.append({
            "name": name,
            "model": model_strength,
            "clip": clip_strength,
            "inject_trigger": inject_trigger,
            "is_version": bool(entry.get("is_version")),
        })
        if len(cleaned) >= STABLE_HORDE_MAX_LORAS:
            break
    return cleaned


AI_DEFAULT_MODEL = CONFIG.get("AI_DEFAULT_MODEL", AI_DEFAULT_MODEL) or AI_DEFAULT_MODEL
AI_DEFAULT_SAMPLER = CONFIG.get("AI_DEFAULT_SAMPLER", AI_DEFAULT_SAMPLER) or AI_DEFAULT_SAMPLER
AI_DEFAULT_WIDTH = _coerce_int(CONFIG.get("AI_DEFAULT_WIDTH"), AI_DEFAULT_WIDTH)
AI_DEFAULT_HEIGHT = _coerce_int(CONFIG.get("AI_DEFAULT_HEIGHT"), AI_DEFAULT_HEIGHT)
AI_DEFAULT_STEPS = _coerce_int(CONFIG.get("AI_DEFAULT_STEPS"), AI_DEFAULT_STEPS)
AI_DEFAULT_CFG = _coerce_float(CONFIG.get("AI_DEFAULT_CFG"), AI_DEFAULT_CFG)
AI_DEFAULT_SAMPLES = _coerce_int(CONFIG.get("AI_DEFAULT_SAMPLES"), AI_DEFAULT_SAMPLES)
AI_OUTPUT_SUBDIR = CONFIG.get("AI_OUTPUT_SUBDIR", AI_OUTPUT_SUBDIR) or AI_OUTPUT_SUBDIR
AI_TEMP_SUBDIR = CONFIG.get("AI_TEMP_SUBDIR", AI_TEMP_SUBDIR) or AI_TEMP_SUBDIR
AI_DEFAULT_PERSIST = _coerce_bool(CONFIG.get("AI_DEFAULT_PERSIST"), AI_DEFAULT_PERSIST)
AI_POLL_INTERVAL = _coerce_float(CONFIG.get("AI_POLL_INTERVAL"), AI_POLL_INTERVAL)
AI_TIMEOUT = _coerce_float(CONFIG.get("AI_TIMEOUT"), AI_TIMEOUT)

LIVE_HLS_ASYNC = _coerce_bool(CONFIG.get("LIVE_HLS_ASYNC"), LIVE_HLS_ASYNC)
HLS_TTL_SECS = max(60, _coerce_int(CONFIG.get("LIVE_HLS_TTL_SECS"), HLS_TTL_SECS))
MAX_HLS_WORKERS = max(1, _coerce_int(CONFIG.get("LIVE_HLS_MAX_WORKERS"), MAX_HLS_WORKERS))
HLS_ERROR_RETRY_SECS = max(5, _coerce_int(CONFIG.get("LIVE_HLS_ERROR_RETRY_SECS"), HLS_ERROR_RETRY_SECS))
HLS_ERROR_RETRY_SECS = min(HLS_ERROR_RETRY_SECS, HLS_TTL_SECS)


AI_OUTPUT_ROOT = _ensure_dir(PRIMARY_MEDIA_ROOT.path / AI_OUTPUT_SUBDIR)
AI_TEMP_ROOT = _ensure_dir(PRIMARY_MEDIA_ROOT.path / AI_TEMP_SUBDIR)

try:
    stable_horde_client = StableHorde(
        save_dir=AI_OUTPUT_ROOT,
        persist_images=AI_DEFAULT_PERSIST,
        default_poll_interval=AI_POLL_INTERVAL,
        default_timeout=AI_TIMEOUT,
        logger=app.logger,
    )
except Exception as exc:  # pragma: no cover - defensive during optional setup
    logger.warning("Stable Horde client unavailable: %s", exc)
    stable_horde_client = None

ai_jobs_lock = threading.Lock()
ai_jobs: Dict[str, Dict[str, Any]] = {}
ai_job_controls: Dict[str, Dict[str, Any]] = {}
ai_model_cache: Dict[str, Any] = {"timestamp": 0.0, "data": []}
auto_scheduler: Optional['AutoGenerateScheduler'] = None
picsum_scheduler: Optional['PicsumAutoScheduler'] = None


@dataclass
class HLSCacheEntry:
    url: Optional[str]
    extracted_at: float
    error: Optional[str] = None


HLS_CACHE: Dict[str, HLSCacheEntry] = {}
HLS_JOBS: Dict[str, Future] = {}
HLS_METRICS: Dict[str, int] = {
    "hits": 0,
    "misses": 0,
    "stale": 0,
    "jobs_started": 0,
    "jobs_completed": 0,
    "errors": 0,
}
HLS_LOCK = threading.RLock()
HLS_LOG_PREFIX = "live_hls"
HLS_EXECUTOR: Optional[ThreadPoolExecutor]
if LIVE_HLS_ASYNC:
    HLS_EXECUTOR = ThreadPoolExecutor(max_workers=MAX_HLS_WORKERS, thread_name_prefix="live-hls")
else:
    HLS_EXECUTOR = None


def _shutdown_hls_executor():
    executor = HLS_EXECUTOR
    if executor is None:
        return
    try:
        executor.shutdown(wait=False, cancel_futures=True)
    except TypeError:  # Python < 3.9 fallback
        executor.shutdown(wait=False)


atexit.register(_shutdown_hls_executor)


def _relative_image_path(path: Union[Path, str]) -> str:
    try:
        resolved = Path(path).resolve()
    except Exception:
        resolved = Path(path)
    virtual = _virtualize_path(resolved)
    return virtual


def _cleanup_temp_outputs(stream_id: str) -> None:
    temp_dir = AI_TEMP_ROOT / stream_id
    if temp_dir.exists():
        try:
            shutil.rmtree(temp_dir)
        except Exception as exc:
            logger.warning('Failed to remove temp outputs for %s: %s', stream_id, exc)

def _emit_ai_update(stream_id: str, state: Dict[str, Any], job: Optional[Dict[str, Any]] = None) -> None:
    payload: Dict[str, Any] = {"stream_id": stream_id, "state": state}
    if job is not None:
        payload['job'] = job
    if job_manager.should_emit(stream_id):
        safe_emit('ai_job_update', payload)


def _update_ai_state(stream_id: str, updates: Dict[str, Any], *, persist: bool = False) -> Dict[str, Any]:
    conf = settings.get(stream_id)
    if not conf:
        return {}
    ensure_ai_defaults(conf)
    ensure_picsum_defaults(conf)
    state = conf[AI_STATE_KEY]
    state.update(updates)
    if persist:
        save_settings(settings)
    _emit_ai_update(stream_id, state)
    return state


def _record_job_progress(stream_id: str, stage: str, payload: Dict[str, Any]) -> None:
    manager_id: Optional[str] = None
    with ai_jobs_lock:
        job = ai_jobs.get(stream_id)
        if not job:
            return
        job = dict(job)
        controls = ai_job_controls.get(stream_id, {})
        manager_id = job.get('manager_id') or controls.get('manager_id')
        if manager_id:
            job['manager_id'] = manager_id
        job['stage'] = stage
        if stage == 'accepted':
            job['job_id'] = payload.get('job_id')
            job['status'] = 'accepted'
            job.setdefault('started', time.time())
        elif stage == 'status':
            status_payload = payload.get('status') or {}
            job['status'] = 'running'
            job['queue_position'] = status_payload.get('queue_position')
            job['wait_time'] = status_payload.get('wait_time')
        elif stage == 'fault':
            job['status'] = 'error'
            job['message'] = str(payload.get('message') or payload)
        elif stage == 'timeout':
            job['status'] = 'timeout'
            job['message'] = 'Timed out waiting for Stable Horde'
        elif stage == 'cancelled':
            job['status'] = 'cancelled'
            job['message'] = str(payload.get('message') or 'Cancelled by user')
        elif stage == 'completed':
            job['status'] = 'completed'
        ai_jobs[stream_id] = job
    state_ref: Optional[Dict[str, Any]] = None
    conf_ref = settings.get(stream_id)
    if isinstance(conf_ref, dict):
        candidate_state = conf_ref.get(AI_STATE_KEY)
        if isinstance(candidate_state, dict):
            state_ref = candidate_state
    if state_ref is not None:
        if job.get('status'):
            state_ref['status'] = job['status']
        if stage == 'status':
            state_ref['queue_position'] = job.get('queue_position')
            state_ref['wait_time'] = job.get('wait_time')
        elif stage in ('fault', 'timeout', 'cancelled', 'completed'):
            state_ref['queue_position'] = job.get('queue_position')
            state_ref['wait_time'] = job.get('wait_time')
            if job.get('message'):
                state_ref['message'] = job['message']
    if manager_id:
        if stage == 'accepted':
            job_manager.set_stable_id(manager_id, payload.get('job_id'))
            job_manager.update_status(manager_id, status='running')
        elif stage == 'status':
            job_manager.touch(manager_id)
        elif stage == 'fault':
            job_manager.update_status(manager_id, status='error', error=str(payload.get('message') or payload))
        elif stage == 'timeout':
            job_manager.update_status(manager_id, status='timeout', error='Timed out waiting for Stable Horde')
        elif stage == 'cancelled':
            job_manager.update_status(manager_id, status='cancelled', error=str(payload.get('message') or 'Cancelled by user'))
        elif stage == 'completed':
            job_manager.update_status(manager_id, status='completed', result=payload, error=None)
    current_state = settings.get(stream_id, {}).get(AI_STATE_KEY, {})
    _emit_ai_update(stream_id, current_state, job=job)


def _run_ai_generation(
    stream_id: str,
    options: Dict[str, Any],
    cancel_event: Optional[threading.Event] = None,
    manager_id: Optional[str] = None,
) -> None:
    prompt = str(options.get('prompt') or '').strip()
    if not prompt:
        message = 'Prompt is required'
        job_manager.update_status(manager_id, status='error', error=message)
        _update_ai_state(stream_id, {
            'status': 'error',
            'message': message,
            'error': message,
        }, persist=True)
        with ai_jobs_lock:
            ai_jobs.pop(stream_id, None)
            ai_job_controls.pop(stream_id, None)
        return

    persist = bool(options.get('save_output', AI_DEFAULT_PERSIST))
    if cancel_event and cancel_event.is_set():
        message = 'Cancelled by user'
        job_manager.update_status(manager_id, status='cancelled', error=message)
        job_snapshot = None
        with ai_jobs_lock:
            current_job = ai_jobs.get(stream_id)
            if current_job:
                job_snapshot = dict(current_job)
                job_snapshot['status'] = 'cancelled'
                job_snapshot['message'] = message
                ai_jobs[stream_id] = job_snapshot
        state = _update_ai_state(
            stream_id,
            {
                'status': 'cancelled',
                'message': message,
                'error': None,
                'persisted': persist,
            },
            persist=True,
        )
        _emit_ai_update(stream_id, state, job_snapshot)
        with ai_jobs_lock:
            ai_jobs.pop(stream_id, None)
            ai_job_controls.pop(stream_id, None)
        return
    target_root = _ensure_dir((AI_OUTPUT_ROOT if persist else AI_TEMP_ROOT) / stream_id)

    def _status_callback(stage: str, payload: Dict[str, Any]) -> None:
        _record_job_progress(stream_id, stage, payload)

    models = [options['model']] if options.get('model') else None
    sampler = options.get('sampler') or AI_DEFAULT_SAMPLER
    negative_prompt = options.get('negative_prompt') or None
    seed_input = options.get('seed')
    if seed_input is None:
        seed_payload = str(secrets.randbelow(2**32))
    else:
        seed_str = str(seed_input).strip()
        if not seed_str or seed_str.lower() in {"random", "rand", "auto"}:
            seed_payload = str(secrets.randbelow(2**32))
        else:
            seed_payload = seed_str

    timeout_raw = options.get('timeout')
    if timeout_raw in (None, '', 'none'):
        timeout_value = AI_TIMEOUT
    else:
        try:
            timeout_value = max(0.0, float(timeout_raw))
        except (TypeError, ValueError):
            timeout_value = AI_TIMEOUT

    post_processing = [
        proc
        for proc in (options.get('post_processing') or [])
        if isinstance(proc, str) and proc in STABLE_HORDE_POST_PROCESSORS
    ]
    loras_payload: List[Dict[str, Any]] = []
    for entry in options.get('loras') or []:
        if not isinstance(entry, dict):
            continue
        name = entry.get('name')
        if not name:
            continue
        cleaned_entry: Dict[str, Any] = {'name': name}
        for attr in ('model', 'clip'):
            val = entry.get(attr)
            if isinstance(val, (int, float)):
                cleaned_entry[attr] = float(val)
        trigger = entry.get('inject_trigger')
        if isinstance(trigger, str) and trigger:
            cleaned_entry['inject_trigger'] = trigger
        if entry.get('is_version'):
            cleaned_entry['is_version'] = bool(entry.get('is_version'))
        loras_payload.append(cleaned_entry)
        if len(loras_payload) >= STABLE_HORDE_MAX_LORAS:
            break
    advanced_params: Dict[str, Any] = {}
    if loras_payload:
        advanced_params['loras'] = loras_payload
    for flag in ('hires_fix', 'karras', 'tiling', 'transparent'):
        if options.get(flag):
            advanced_params[flag] = True
    clip_skip_value = options.get('clip_skip')
    if isinstance(clip_skip_value, (int, float)):
        advanced_params['clip_skip'] = int(clip_skip_value)
    for float_key in ('facefixer_strength',):
        val = options.get(float_key)
        if isinstance(val, (int, float)):
            advanced_params[float_key] = float(val)
    for float_key in ('denoising_strength', 'hires_fix_denoising_strength'):
        val = options.get(float_key)
        if isinstance(val, (int, float)):
            advanced_params[float_key] = float(val)
    extras_payload: Dict[str, Any] = {}
    for flag in ('trusted_workers', 'validated_backends', 'slow_workers', 'extra_slow_workers', 'disable_batching', 'allow_downgrade'):
        if flag in options:
            extras_payload[flag] = bool(options.get(flag))
    style_value = str(options.get('style') or '').strip()
    if style_value:
        extras_payload['style'] = style_value

    try:
        result = stable_horde_client.generate_images(
            prompt,
            negative_prompt=negative_prompt,
            models=models,
            width=int(options.get('width', AI_DEFAULT_WIDTH)),
            height=int(options.get('height', AI_DEFAULT_HEIGHT)),
            steps=int(options.get('steps', AI_DEFAULT_STEPS)),
            cfg_scale=float(options.get('cfg_scale', AI_DEFAULT_CFG)),
            sampler_name=sampler,
            seed=seed_payload,
            samples=int(options.get('samples', AI_DEFAULT_SAMPLES)),
            nsfw=bool(options.get('nsfw')),
            censor_nsfw=bool(options.get('censor_nsfw')),
            post_processing=post_processing or None,
            params=advanced_params or None,
            extras=extras_payload or None,
            poll_interval=float(options.get('poll_interval', AI_POLL_INTERVAL)),
            timeout=timeout_value,
            persist=persist,
            output_dir=target_root if persist else None,
            status_callback=_status_callback,
            cancel_callback=(lambda: bool(cancel_event and cancel_event.is_set())),
        )
    except StableHordeCancelled as exc:
        logger.info('Stable Horde job for %s cancelled: %s', stream_id, exc)
        message = 'Cancelled by user'
        _record_job_progress(stream_id, 'cancelled', {'message': message})
        job_snapshot = None
        with ai_jobs_lock:
            current_job = ai_jobs.get(stream_id)
            if current_job:
                job_snapshot = dict(current_job)
                job_snapshot['status'] = 'cancelled'
                job_snapshot['message'] = message
                ai_jobs[stream_id] = job_snapshot
        state = _update_ai_state(
            stream_id,
            {
                'status': 'cancelled',
                'message': message,
                'error': None,
                'persisted': persist,
            },
            persist=True,
        )
        _emit_ai_update(stream_id, state, job_snapshot)
        if not persist:
            _cleanup_temp_outputs(stream_id)
        job_manager.update_status(manager_id, status='cancelled', error=str(exc))
        with ai_jobs_lock:
            ai_jobs.pop(stream_id, None)
            ai_job_controls.pop(stream_id, None)
        return
    except StableHordeError as exc:
        logger.warning('Stable Horde job for %s failed: %s', stream_id, exc)
        _record_job_progress(stream_id, 'fault', {'message': str(exc)})
        _update_ai_state(
            stream_id,
            {
                'status': 'error',
                'message': str(exc),
                'error': str(exc),
                'persisted': persist,
            },
            persist=True,
        )
        job_manager.update_status(manager_id, status='error', error=str(exc))
        with ai_jobs_lock:
            ai_jobs.pop(stream_id, None)
            ai_job_controls.pop(stream_id, None)
        return
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception('Unexpected Stable Horde failure for %s: %s', stream_id, exc)
        _record_job_progress(stream_id, 'fault', {'message': str(exc)})
        _update_ai_state(
            stream_id,
            {
                'status': 'error',
                'message': 'Generation failed',
                'error': str(exc),
                'persisted': persist,
            },
            persist=True,
        )
        job_manager.update_status(manager_id, status='error', error=str(exc))
        with ai_jobs_lock:
            ai_jobs.pop(stream_id, None)
            ai_job_controls.pop(stream_id, None)
        return

    images: List[Dict[str, Any]] = []
    if persist:
        stored_paths = [(generation.path, generation) for generation in result.generations]
    else:
        job_dir = _ensure_dir(target_root / result.job_id)
        stored_paths = []
        for generation in result.generations:
            final_path = job_dir / Path(generation.path).name
            try:
                shutil.copy2(generation.path, final_path)
            except Exception as exc:
                logger.warning('Failed to copy temp generation %s: %s', generation.path, exc)
                continue
            stored_paths.append((final_path, generation))
        result.cleanup()

    for final_path, generation in stored_paths:
        rel_path = _relative_image_path(final_path)
        images.append({
            'path': rel_path,
            'seed': generation.seed,
            'model': generation.model or options.get('model'),
            'worker': generation.worker,
            'url': generation.url,
            'persisted': persist,
        })

    updates = {
        'status': 'completed',
        'job_id': result.job_id,
        'queue_position': result.queue_position,
        'wait_time': result.wait_time,
        'images': images,
        'persisted': persist,
        'message': None,
        'error': None,
        'last_updated': datetime.utcnow().isoformat() + 'Z',
    }
    conf = settings.get(stream_id)
    if conf:
        ensure_ai_defaults(conf)
        ensure_picsum_defaults(conf)
        conf[AI_SETTINGS_KEY] = _sanitize_ai_settings(options, conf[AI_SETTINGS_KEY])
        conf[AI_SETTINGS_KEY]['save_output'] = persist
        conf['_ai_customized'] = not _ai_settings_match_defaults(conf[AI_SETTINGS_KEY])
        conf[AI_STATE_KEY].update(updates)
        if images:
            conf['selected_image'] = images[0]['path']
        conf['mode'] = AI_MODE
        conf['media_mode'] = MEDIA_MODE_AI
        _update_stream_runtime_state(
            stream_id,
            path=conf.get('selected_image'),
            kind='image',
            media_mode=MEDIA_MODE_AI,
            source='ai_generation',
        )
        save_settings(settings)
        _emit_ai_update(stream_id, conf[AI_STATE_KEY])
        if job_manager.should_emit(stream_id):
            safe_emit('refresh', {'stream_id': stream_id, 'config': conf})

    completion_payload = {
        'job_id': result.job_id,
        'images': images,
        'queue_position': result.queue_position,
        'wait_time': result.wait_time,
        'persisted': persist,
    }
    _record_job_progress(stream_id, 'completed', completion_payload)
    with ai_jobs_lock:
        ai_jobs.pop(stream_id, None)
        ai_job_controls.pop(stream_id, None)


settings = load_settings()
if ensure_settings_integrity(settings):
    save_settings(settings)
# Normalize embed metadata placeholders for existing streams
for stream_id, conf in list(settings.items()):
    if not isinstance(stream_id, str) or stream_id.startswith("_"):
        continue
    if not isinstance(conf, dict):
        continue
    conf["embed_metadata"] = _sanitize_embed_metadata(conf.get("embed_metadata"))

# Ensure global AI defaults exist and are sanitized
raw_ai_defaults = settings.get("_ai_defaults") if isinstance(settings.get("_ai_defaults"), dict) else None
settings["_ai_defaults"] = _sanitize_ai_settings(
    raw_ai_defaults or {},
    deepcopy(AI_FALLBACK_DEFAULTS),
    defaults=AI_FALLBACK_DEFAULTS,
)

def _detect_media_kind(value: Optional[str]) -> str:
    if not value:
        return "image"
    ext = os.path.splitext(str(value))[1].lower()
    if ext in VIDEO_EXTENSIONS:
        return "video"
    return "image"


def _infer_media_mode(conf: Dict[str, Any]) -> str:
    mode_raw = conf.get("mode")
    mode = mode_raw.strip().lower() if isinstance(mode_raw, str) else ""
    if mode == AI_MODE:
        return MEDIA_MODE_AI
    if mode == "livestream":
        return MEDIA_MODE_LIVESTREAM
    if mode == MEDIA_MODE_PICSUM:
        return MEDIA_MODE_PICSUM

    selected_kind_raw = conf.get("selected_media_kind")
    selected_kind = selected_kind_raw.strip().lower() if isinstance(selected_kind_raw, str) else ""
    if selected_kind == "video":
        return MEDIA_MODE_VIDEO

    playback_raw = conf.get("video_playback_mode")
    playback_mode = playback_raw.strip().lower() if isinstance(playback_raw, str) else ""
    if mode in ("random", "specific") and playback_mode in ("until_end", "loop"):
        return MEDIA_MODE_VIDEO

    return MEDIA_MODE_IMAGE


def _get_stream_runtime_state(stream_id: str) -> Dict[str, Any]:
    with STREAM_RUNTIME_LOCK:
        entry = STREAM_RUNTIME_STATE.get(stream_id)
        return dict(entry) if entry else {}

def _get_runtime_thumbnail_payload(stream_id: str) -> Optional[Dict[str, Any]]:
    """Return the thumbnail metadata for a stream suitable for client payloads."""
    with STREAM_RUNTIME_LOCK:
        entry = STREAM_RUNTIME_STATE.get(stream_id)
        if not entry:
            return None
        record = entry.get("thumbnail")
        if not isinstance(record, dict):
            return None
        payload = {k: v for k, v in record.items() if not k.startswith("_")}
        return payload or None


def _update_stream_runtime_state(
    stream_id: str,
    *,
    path: Optional[str] = None,
    kind: Optional[str] = None,
    media_mode: Optional[str] = None,
    stream_url: Optional[str] = None,
    source: str = "unknown",
    force_thumbnail: bool = False,
) -> Optional[Dict[str, Any]]:
    if not stream_id or stream_id.startswith("_"):
        return None
    normalized_mode = media_mode.strip().lower() if isinstance(media_mode, str) else None
    normalized_kind = kind.strip().lower() if isinstance(kind, str) else None
    resolved_path = path if path not in ("", None) else None
    if resolved_path and not normalized_kind:
        normalized_kind = _detect_media_kind(resolved_path)
    timestamp = time.time()
    changed = False
    existing_thumbnail: Optional[Dict[str, Any]] = None
    with STREAM_RUNTIME_LOCK:
        entry = STREAM_RUNTIME_STATE.setdefault(stream_id, {})
        previous_mode = entry.get("media_mode")
        previous_path = entry.get("path")
        previous_kind = entry.get("kind")
        previous_url = entry.get("stream_url")
        existing_thumbnail = dict(entry["thumbnail"]) if isinstance(entry.get("thumbnail"), dict) else None

        if normalized_mode is not None:
            if previous_mode != normalized_mode:
                changed = True
            entry["media_mode"] = normalized_mode
        elif "media_mode" not in entry:
            entry["media_mode"] = None

        if path is not None:
            if resolved_path is None:
                if previous_path is not None:
                    changed = True
                entry.pop("path", None)
                if "kind" in entry:
                    entry.pop("kind", None)
            else:
                if previous_path != resolved_path:
                    changed = True
                entry["path"] = resolved_path
                detected_kind = normalized_kind or _detect_media_kind(resolved_path)
                if previous_kind != detected_kind:
                    changed = True
                entry["kind"] = detected_kind
        elif normalized_kind is not None:
            if previous_kind != normalized_kind:
                changed = True
            entry["kind"] = normalized_kind

        if stream_url is not None:
            if isinstance(stream_url, str):
                candidate = stream_url.strip()
                normalized_url = candidate or None
            else:
                normalized_url = None
            if previous_url != normalized_url:
                changed = True
            entry["stream_url"] = normalized_url

        entry["timestamp"] = timestamp
        entry["source"] = source

    if not changed and not force_thumbnail:
        return _get_runtime_thumbnail_payload(stream_id)
    thumbnail_info = _refresh_stream_thumbnail(stream_id, force=force_thumbnail)
    if thumbnail_info is None and existing_thumbnail:
        return {k: v for k, v in existing_thumbnail.items() if not k.startswith("_")}
    return thumbnail_info

def _runtime_timestamp_to_iso(ts: Optional[float]) -> Optional[str]:
    if not ts:
        return None
    try:
        return datetime.utcfromtimestamp(ts).replace(microsecond=0).isoformat() + 'Z'
    except (ValueError, OSError, OverflowError):
        return None


def _resolve_media_path(rel_path: Optional[str]) -> Optional[Path]:
    if not rel_path:
        return None
    target = _resolve_virtual_media_path(rel_path)
    if target is None or not target.exists():
        return None
    return target


class StreamPlaybackState:
    """Track shared playback data for a single stream."""

    def __init__(self, stream_id: str):
        self.stream_id = stream_id
        self.mode: str = "random"
        self.media_mode: str = MEDIA_MODE_IMAGE
        self.folder: str = "all"
        self.hide_nsfw: bool = False
        self.shuffle: bool = True
        self.duration_setting: float = 5.0
        self.video_playback_mode: str = "duration"
        self.video_volume: float = 1.0
        self.current_media: Optional[Dict[str, Any]] = None
        self.started_at: Optional[float] = None
        self.position: float = 0.0
        self.duration: Optional[float] = None
        self.is_paused: bool = False
        self.next_auto_event: Optional[float] = None
        self.sequence_index: int = 0
        self.history: List[Dict[str, Any]] = []
        self.history_index: int = -1
        self.last_reason: str = "init"
        self.error: Optional[str] = None
        self.updated_at: float = time.time()
        self._source_signature: Optional[Tuple[Any, ...]] = None
        self._duration_signature: Optional[Tuple[Any, ...]] = None
        self.last_sync_emit: float = 0.0

    def apply_config(self, conf: Dict[str, Any]) -> Dict[str, bool]:
        previous_should_run = self.should_run()
        previous_source_signature = self._source_signature
        previous_duration_signature = self._duration_signature

        mode_raw = conf.get("mode")
        new_mode = mode_raw.strip().lower() if isinstance(mode_raw, str) else "random"

        media_mode_raw = conf.get("media_mode")
        new_media_mode = media_mode_raw.strip().lower() if isinstance(media_mode_raw, str) else ""
        if new_media_mode not in MEDIA_MODE_CHOICES:
            inferred = _infer_media_mode(conf)
            if inferred in MEDIA_MODE_CHOICES:
                new_media_mode = inferred
            else:
                new_media_mode = MEDIA_MODE_IMAGE

        folder_raw = conf.get("folder")
        new_folder = folder_raw.strip() if isinstance(folder_raw, str) and folder_raw.strip() else "all"

        new_hide_nsfw = bool(conf.get("hide_nsfw"))
        shuffle_value = conf.get("shuffle")
        new_shuffle = False if shuffle_value is False else True

        duration_raw = conf.get("duration")
        try:
            new_duration = float(duration_raw)
        except (TypeError, ValueError):
            new_duration = self.duration_setting
        if not (new_duration and new_duration > 0):
            new_duration = 5.0

        playback_raw = conf.get("video_playback_mode")
        new_playback_mode = playback_raw.strip().lower() if isinstance(playback_raw, str) else "duration"
        if new_playback_mode not in VIDEO_PLAYBACK_MODES:
            new_playback_mode = "duration"

        volume_raw = conf.get("video_volume")
        try:
            new_volume = float(volume_raw)
        except (TypeError, ValueError):
            new_volume = self.video_volume
        new_volume = max(0.0, min(1.0, new_volume))

        self.mode = new_mode
        self.media_mode = new_media_mode
        self.folder = new_folder
        self.hide_nsfw = new_hide_nsfw
        self.shuffle = new_shuffle
        self.duration_setting = new_duration
        self.video_playback_mode = new_playback_mode
        self.video_volume = new_volume

        new_source_signature = (self.mode, self.media_mode, self.folder, self.hide_nsfw, self.shuffle)
        new_duration_signature = (self.duration_setting, self.video_playback_mode)

        self._source_signature = new_source_signature
        self._duration_signature = new_duration_signature

        enabled_changed = previous_should_run != self.should_run()
        sources_changed = new_source_signature != previous_source_signature
        duration_changed = new_duration_signature != previous_duration_signature
        return {
            "enabled_changed": enabled_changed,
            "sources_changed": sources_changed,
            "duration_changed": duration_changed,
        }

    def should_run(self) -> bool:
        return self.mode == "random" and self.media_mode in (MEDIA_MODE_IMAGE, MEDIA_MODE_VIDEO)

    def reset_state(self) -> None:
        self.current_media = None
        self.started_at = None
        self.position = 0.0
        self.duration = None
        self.is_paused = False
        self.next_auto_event = None
        self.sequence_index = 0
        self.history = []
        self.history_index = -1
        self.error = None
        self.last_reason = "reset"
        self.updated_at = time.time()
        self.last_sync_emit = 0.0

    def set_error(self, code: str) -> None:
        self.current_media = None
        self.started_at = None
        self.position = 0.0
        self.duration = None
        self.is_paused = False
        self.next_auto_event = None
        self.error = code
        self.last_reason = code
        self.updated_at = time.time()
        self.last_sync_emit = 0.0

    def get_position(self, now: Optional[float] = None) -> float:
        if now is None:
            now = time.time()
        if self.current_media is None:
            return 0.0
        if self.is_paused or self.started_at is None:
            return max(0.0, self.position)
        return max(0.0, self.position + (now - self.started_at))

    def pause(self) -> bool:
        if self.is_paused or not self.current_media:
            return False
        now = time.time()
        self.position = self.get_position(now)
        self.is_paused = True
        self.started_at = now
        self.next_auto_event = None
        self.updated_at = now
        return True

    def resume(self) -> bool:
        if not self.current_media or not self.is_paused:
            return False
        now = time.time()
        self.is_paused = False
        self.started_at = now
        if self.duration is not None:
            remaining = max(0.0, self.duration - self.position)
            if remaining > 0:
                self.next_auto_event = now + remaining
            else:
                self.next_auto_event = now
        self.updated_at = now
        return True

    def _append_history_entry(self, media: Dict[str, Any], playback_mode: Optional[str]) -> None:
        entry = {
            "media": dict(media),
            "duration": self.duration,
            "playback_mode": playback_mode,
        }
        self.history.append(entry)
        if len(self.history) > STREAM_PLAYBACK_HISTORY_LIMIT:
            overflow = len(self.history) - STREAM_PLAYBACK_HISTORY_LIMIT
            self.history = self.history[overflow:]
        self.history_index = len(self.history) - 1

    def set_media(
        self,
        media: Dict[str, Any],
        *,
        duration: Optional[float],
        source: str,
        playback_mode: Optional[str],
        history_index: Optional[int] = None,
        add_to_history: bool = True,
    ) -> None:
        now = time.time()
        actual_duration = duration if duration is None or duration > 0 else None
        self.current_media = dict(media)
        self.duration = actual_duration
        self.started_at = now
        self.position = 0.0
        self.is_paused = False
        self.error = None
        self.last_reason = source
        self.updated_at = now
        if actual_duration is not None:
            self.next_auto_event = now + actual_duration
        else:
            self.next_auto_event = None

        if add_to_history:
            if self.history_index < len(self.history) - 1:
                self.history = self.history[: self.history_index + 1]
            self._append_history_entry(media, playback_mode)
        elif history_index is not None:
            self.history_index = history_index

    def get_history_entry(self, index: int) -> Optional[Dict[str, Any]]:
        if 0 <= index < len(self.history):
            return self.history[index]
        return None

    def status(self) -> str:
        if self.error:
            return "error"
        if not self.current_media:
            return "idle"
        if self.is_paused:
            return "paused"
        return "playing"

    def to_payload(self) -> Dict[str, Any]:
        now = time.time()
        position = self.get_position(now)
        if self.duration is not None:
            position = min(self.duration, position)
        payload = {
            "stream_id": self.stream_id,
            "mode": self.mode,
            "media_mode": self.media_mode,
            "status": self.status(),
            "media": dict(self.current_media) if self.current_media else None,
            "duration": self.duration,
            "position": position,
            "started_at": self.started_at,
            "is_paused": self.is_paused,
            "next_update_at": self.next_auto_event,
            "history_index": self.history_index,
            "history_length": len(self.history),
            "video_playback_mode": self.video_playback_mode if self.media_mode == MEDIA_MODE_VIDEO else None,
            "video_volume": self.video_volume,
            "error": self.error,
            "source": self.last_reason,
            "server_time": now,
            "thumbnail": _get_runtime_thumbnail_payload(self.stream_id),
        }
        return payload

    def to_sync_payload(self, now: Optional[float] = None) -> Dict[str, Any]:
        snapshot = now if now is not None else time.time()
        position = self.get_position(snapshot)
        if self.duration is not None:
            position = min(self.duration, position)
        payload = {
            "stream_id": self.stream_id,
            "media": dict(self.current_media) if self.current_media else None,
            "duration": self.duration,
            "position": position,
            "started_at": self.started_at,
            "is_paused": self.is_paused,
            "server_time": snapshot,
        }
        return payload


def _compute_video_duration_seconds(rel_path: Optional[str]) -> Optional[float]:
    if not rel_path:
        return None
    absolute = _resolve_media_path(rel_path)
    if not absolute:
        return None
    if cv2 is None:
        return None
    capture = cv2.VideoCapture(str(absolute))
    if not capture.isOpened():
        capture.release()
        return None
    duration = None
    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        if fps > 0 and frame_count > 0:
            duration = frame_count / fps
        else:
            milliseconds = capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0
            if milliseconds > 0:
                duration = milliseconds / 1000.0
    except Exception:  # pragma: no cover - best effort metadata
        duration = None
    finally:
        capture.release()
    if duration is None or duration <= 0:
        return None
    return float(duration)


class StreamPlaybackManager:
    """Coordinate synchronized playback across connected viewers."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._states: Dict[str, StreamPlaybackState] = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._loop,
            name="StreamPlaybackManager",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def emit_state(self, payload: Dict[str, Any], *, room: Optional[str] = None, event: str = STREAM_UPDATE_EVENT) -> None:
        self._emit_state(payload, room=room, event=event)

    def bootstrap(self, stream_settings: Dict[str, Any]) -> None:
        for stream_id, conf in stream_settings.items():
            if stream_id.startswith("_"):
                continue
            if isinstance(conf, dict):
                self.update_stream_config(stream_id, conf)

    def update_stream_config(self, stream_id: str, conf: Dict[str, Any]) -> None:
        payload_to_emit: Optional[Dict[str, Any]] = None
        needs_refresh = False
        with self._lock:
            state = self._states.get(stream_id)
            if state is None:
                state = StreamPlaybackState(stream_id)
                self._states[stream_id] = state
            apply_result = state.apply_config(conf)
            if not state.should_run():
                if state.current_media or state.error:
                    state.reset_state()
                    payload_to_emit = state.to_payload()
                return
            if state.current_media is None:
                needs_refresh = True
            elif apply_result.get("enabled_changed") or apply_result.get("sources_changed") or apply_result.get("duration_changed"):
                state.reset_state()
                needs_refresh = True
        if payload_to_emit:
            self._emit_state(payload_to_emit, room=stream_id)
        if needs_refresh:
            payload = self._advance_stream(stream_id, reason="config")
            if payload:
                self._emit_state(payload, room=stream_id)

    def remove_stream(self, stream_id: str) -> None:
        with self._lock:
            self._states.pop(stream_id, None)

    def get_state(self, stream_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            state = self._states.get(stream_id)
            if not state:
                return None
            return state.to_payload()

    def ensure_started(self, stream_id: str) -> Optional[Dict[str, Any]]:
        payload = self.get_state(stream_id)
        if payload and payload.get("status") != "idle":
            return payload
        payload = self._advance_stream(stream_id, reason="initial")
        if payload:
            self._emit_state(payload, room=stream_id)
        return payload

    def skip_previous(self, stream_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            state = self._states.get(stream_id)
            if not state or not state.should_run():
                return None
            target_index = state.history_index - 1
            entry = state.get_history_entry(target_index)
            if entry is None:
                return None
            media = entry.get("media")
            duration = entry.get("duration")
            playback_mode = entry.get("playback_mode")
            if not media:
                return None
            media_copy = dict(media)
            state.set_media(media_copy, duration=duration, source="history_prev", playback_mode=playback_mode, history_index=target_index, add_to_history=False)
            payload = state.to_payload()
            media_mode = state.media_mode
        runtime_args = {
            "path": media_copy.get("path"),
            "kind": media_copy.get("kind"),
            "media_mode": media_mode,
            "stream_url": media_copy.get("stream_url"),
            "source": "history_prev",
        }
        thumbnail_info = _update_stream_runtime_state(stream_id, **runtime_args)
        if payload and thumbnail_info is not None:
            payload["thumbnail"] = thumbnail_info
        self._emit_state(payload, room=stream_id)
        return payload

    def skip_next(self, stream_id: str) -> Optional[Dict[str, Any]]:
        media_copy: Optional[Dict[str, Any]] = None
        media_mode: Optional[str] = None
        with self._lock:
            state = self._states.get(stream_id)
            if not state or not state.should_run():
                state_to_emit = state.to_payload() if state else None
                media_mode = state.media_mode if state else None
            else:
                target_index = state.history_index + 1
                entry = state.get_history_entry(target_index)
                if entry:
                    media = entry.get("media")
                    duration = entry.get("duration")
                    playback_mode = entry.get("playback_mode")
                    if media:
                        media_copy = dict(media)
                        state.set_media(media_copy, duration=duration, source="history_next", playback_mode=playback_mode, history_index=target_index, add_to_history=False)
                        state_to_emit = state.to_payload()
                    else:
                        state_to_emit = None
                        media_copy = None
                else:
                    state_to_emit = None
                    media_copy = None
                media_mode = state.media_mode
        if state_to_emit:
            if media_copy is not None:
                runtime_args = {
                    "path": media_copy.get("path"),
                    "kind": media_copy.get("kind"),
                    "media_mode": media_mode,
                    "stream_url": media_copy.get("stream_url"),
                    "source": "history_next",
                }
                thumbnail_info = _update_stream_runtime_state(stream_id, **runtime_args)
                if thumbnail_info is not None:
                    state_to_emit["thumbnail"] = thumbnail_info
            self._emit_state(state_to_emit, room=stream_id)
            return state_to_emit
        payload = self._advance_stream(stream_id, reason="manual")
        if payload:
            self._emit_state(payload, room=stream_id)
        return payload

    def pause(self, stream_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            state = self._states.get(stream_id)
            if not state or not state.pause():
                return None
            payload = state.to_payload()
        self._emit_state(payload, room=stream_id)
        return payload

    def resume(self, stream_id: str) -> Optional[Dict[str, Any]]:
        payload: Optional[Dict[str, Any]] = None
        need_new_media = False
        with self._lock:
            state = self._states.get(stream_id)
            if not state:
                return None
            if state.current_media is None:
                need_new_media = True
            else:
                if not state.resume():
                    return None
                payload = state.to_payload()
        if need_new_media:
            payload = self._advance_stream(stream_id, reason="resume")
            if payload:
                self._emit_state(payload, room=stream_id)
            return payload
        if payload:
            self._emit_state(payload, room=stream_id)
        return payload

    def set_volume(self, stream_id: str, volume: float) -> Optional[Dict[str, Any]]:
        clamped = max(0.0, min(1.0, float(volume)))
        with self._lock:
            state = self._states.get(stream_id)
            if not state:
                return None
            state.video_volume = clamped
            payload = state.to_payload()
        self._emit_state(payload, room=stream_id)
        return payload

    def toggle(self, stream_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            state = self._states.get(stream_id)
            if not state:
                return None
            paused = state.is_paused
        if paused:
            return self.resume(stream_id)
        return self.pause(stream_id)

    def _mark_sync_sent(self, stream_id: Optional[str], server_time: Optional[float]) -> None:
        if not stream_id:
            return
        with self._lock:
            state = self._states.get(stream_id)
            if not state:
                return
            if isinstance(server_time, (int, float)):
                state.last_sync_emit = float(server_time)
            else:
                state.last_sync_emit = time.time()

    def _emit_state(self, payload: Dict[str, Any], *, room: Optional[str] = None, event: str = STREAM_UPDATE_EVENT) -> None:
        stream_id = payload.get("stream_id")
        server_time = payload.get("server_time")
        if stream_id:
            self._mark_sync_sent(stream_id, server_time if isinstance(server_time, (int, float)) else None)
        target_room = room or stream_id
        if target_room:
            safe_emit(event, payload, to=target_room)
        else:
            safe_emit(event, payload)

    def _next_media(self, state: StreamPlaybackState) -> Optional[Dict[str, Any]]:
        entries = list_media(state.folder, hide_nsfw=state.hide_nsfw)
        if state.media_mode == MEDIA_MODE_IMAGE:
            entries = [item for item in entries if item.get("kind") == "image"]
        elif state.media_mode == MEDIA_MODE_VIDEO:
            entries = [item for item in entries if item.get("kind") == "video"]
        if not entries:
            return None
        if state.shuffle:
            pool = entries
            if state.current_media and len(entries) > 1:
                current_path = state.current_media.get("path")
                filtered = [item for item in entries if item.get("path") != current_path]
                if filtered:
                    pool = filtered
            choice = random.choice(pool)
        else:
            index = state.sequence_index % len(entries)
            choice = entries[index]
            state.sequence_index = (index + 1) % len(entries)
        return dict(choice)

    def _compute_duration(self, state: StreamPlaybackState, media: Dict[str, Any]) -> Optional[float]:
        kind = media.get("kind")
        if kind == "image":
            return max(1.0, float(state.duration_setting))
        if kind == "video":
            playback_mode = state.video_playback_mode
            if playback_mode == "loop":
                return None
            if playback_mode == "duration":
                return max(1.0, float(state.duration_setting))
            if playback_mode == "until_end":
                duration = _compute_video_duration_seconds(media.get("path"))
                if duration is None:
                    return max(1.0, float(state.duration_setting))
                return duration
        return max(1.0, float(state.duration_setting))

    def _advance_stream(self, stream_id: str, *, reason: str) -> Optional[Dict[str, Any]]:
        payload: Optional[Dict[str, Any]]
        runtime_args: Dict[str, Any]
        with self._lock:
            state = self._states.get(stream_id)
            if not state or not state.should_run():
                return None
            media = self._next_media(state)
            media_mode = state.media_mode
            if media is None:
                state.set_error("no_media")
                payload = state.to_payload()
                runtime_args = {
                    "path": None,
                    "kind": None,
                    "media_mode": media_mode,
                    "stream_url": None,
                    "source": f"playback_{reason}",
                    "force_thumbnail": True,
                }
            else:
                duration = self._compute_duration(state, media)
                playback_mode = state.video_playback_mode if media.get("kind") == "video" else None
                state.set_media(media, duration=duration, source=reason, playback_mode=playback_mode)
                payload = state.to_payload()
                runtime_args = {
                    "path": media.get("path"),
                    "kind": media.get("kind"),
                    "media_mode": media_mode,
                    "stream_url": media.get("stream_url"),
                    "source": f"playback_{reason}",
                }
        thumbnail_info = _update_stream_runtime_state(stream_id, **runtime_args)
        if payload and thumbnail_info is not None:
            payload["thumbnail"] = thumbnail_info
        return payload

    def _loop(self) -> None:
        while not self._stop.is_set():
            now = time.time()
            next_deadline: Optional[float] = None
            due_streams: List[str] = []
            sync_payloads: List[Dict[str, Any]] = []
            with self._lock:
                for stream_id, state in self._states.items():
                    if not state.should_run():
                        continue
                    is_due = False
                    if not state.is_paused:
                        deadline = state.next_auto_event
                        if deadline is not None:
                            if deadline <= now:
                                due_streams.append(stream_id)
                                is_due = True
                            else:
                                if next_deadline is None or deadline < next_deadline:
                                    next_deadline = deadline
                    else:
                        deadline = None
                    if (
                        state.current_media
                        and not state.is_paused
                        and not is_due
                    ):
                        last_sync = state.last_sync_emit if isinstance(state.last_sync_emit, (int, float)) else 0.0
                        if last_sync <= 0.0 or (now - last_sync) >= STREAM_SYNC_INTERVAL_SECONDS:
                            sync_payloads.append(state.to_sync_payload(now))
            for stream_id in due_streams:
                payload = self._advance_stream(stream_id, reason="auto")
                if payload:
                    self._emit_state(payload, room=stream_id)
            for payload in sync_payloads:
                self._emit_state(payload, event=SYNC_TIME_EVENT)
            if next_deadline is None:
                sleep_for = 1.0
            else:
                sleep_for = max(0.1, min(1.0, next_deadline - time.time()))
            self._stop.wait(sleep_for)

def _generate_placeholder_thumbnail(label: str) -> Image.Image:
    background = Image.new('RGB', DASHBOARD_THUMBNAIL_SIZE, (32, 34, 46))
    draw = ImageDraw.Draw(background)
    font = ImageFont.load_default()
    text = (label or '').strip() or 'No Preview'
    text = text.upper()
    bbox = None
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
    except AttributeError:
        if hasattr(font, 'getbbox'):
            bbox = font.getbbox(text)
    if bbox:
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
    else:
        # Final fallback for very old Pillow where bbox helpers are unavailable.
        text_width, text_height = font.getsize(text)  # type: ignore[attr-defined]
    x = max((DASHBOARD_THUMBNAIL_SIZE[0] - text_width) // 2, 4)
    y = max((DASHBOARD_THUMBNAIL_SIZE[1] - text_height) // 2, 4)
    draw.text((x, y), text, fill=(210, 210, 210), font=font)
    return background


def _compose_thumbnail(frame: Image.Image) -> Image.Image:
    background = Image.new('RGB', DASHBOARD_THUMBNAIL_SIZE, (20, 20, 24))
    prepared = frame.convert('RGB')
    prepared.thumbnail(DASHBOARD_THUMBNAIL_SIZE, IMAGE_THUMBNAIL_FILTER)
    offset_x = max((DASHBOARD_THUMBNAIL_SIZE[0] - prepared.width) // 2, 0)
    offset_y = max((DASHBOARD_THUMBNAIL_SIZE[1] - prepared.height) // 2, 0)
    background.paste(prepared, (offset_x, offset_y))
    return background


def _create_thumbnail_image(media_path: Path) -> Image.Image:
    try:
        with Image.open(media_path) as src:
            if getattr(src, 'is_animated', False):
                try:
                    src.seek(0)
                except EOFError:
                    pass
            return _compose_thumbnail(src)
    except Exception as exc:  # pragma: no cover - defensive, thumbnail best-effort
        logger.debug('Failed to render image thumbnail for %s: %s', media_path, exc)
    return _generate_placeholder_thumbnail('Image')


def _create_video_thumbnail(media_path: Path) -> Optional[Image.Image]:
    if cv2 is None:
        return None
    capture = cv2.VideoCapture(str(media_path))
    if not capture.isOpened():
        return None
    try:
        frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        target_frame: Optional[int] = None
        if frame_count and frame_count > 0:
            start = int(max(0, frame_count * 0.15))
            end = int(max(start + 1, frame_count * 0.85))
            if end <= start:
                end = start + 1
            try:
                target_frame = random.randint(start, max(start + 1, end - 1))
                capture.set(cv2.CAP_PROP_POS_FRAMES, float(target_frame))
            except Exception:
                target_frame = None
        if target_frame is None:
            try:
                duration_ms = capture.get(cv2.CAP_PROP_POS_MSEC) or 0
                if duration_ms > 0:
                    capture.set(cv2.CAP_PROP_POS_MSEC, duration_ms * 0.5)
            except Exception:
                pass
        success, frame = capture.read()
        if (not success or frame is None) and frame_count and frame_count > 0:
            try:
                capture.set(cv2.CAP_PROP_POS_FRAMES, max(0.0, float(frame_count) * 0.5))
                success, frame = capture.read()
            except Exception:
                pass
    finally:
        capture.release()
    if not success or frame is None:
        return None
    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    except Exception:
        rgb_frame = frame[:, :, ::-1]
    image = Image.fromarray(rgb_frame)
    return _compose_thumbnail(image)


def _load_remote_image(url: str) -> Optional[Image.Image]:
    if not url or requests is None:
        return None
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                      '(KHTML, like Gecko) Chrome/120.0 Safari/537.36'
    }
    try:
        resp = requests.get(url, headers=headers, timeout=6)
        if resp.status_code != 200:
            return None
        content_type = resp.headers.get('Content-Type', '').lower()
        if content_type and 'image' not in content_type:
            return None
        buffer = io.BytesIO(resp.content)
        image = Image.open(buffer)
        image.load()
        return image
    except Exception as exc:
        logger.debug('Remote thumbnail fetch failed for %s: %s', url, exc)
        return None


def _create_livestream_thumbnail(stream_url: Optional[str]) -> Optional[Image.Image]:
    if not stream_url:
        return None
    url = stream_url.strip()
    if not url:
        return None
    lower = url.lower()
    remote_image: Optional[Image.Image] = None
    if 'youtube.com' in lower or 'youtu.be/' in lower:
        video_id = None
        if 'watch?v=' in url:
            video_id = url.split('watch?v=')[1].split('&')[0].split('#')[0]
        elif 'youtu.be/' in url:
            video_id = url.split('youtu.be/')[1].split('?')[0].split('&')[0]
        if video_id:
            for variant in ('maxresdefault', 'sddefault', 'hqdefault', 'mqdefault', 'default'):
                remote_image = _load_remote_image(f'https://img.youtube.com/vi/{video_id}/{variant}.jpg')
                if remote_image:
                    break
    elif 'twitch.tv' in lower:
        try:
            channel = url.split('twitch.tv/')[1].split('/')[0]
        except Exception:
            channel = ''
        if channel:
            remote_image = _load_remote_image(f'https://static-cdn.jtvnw.net/previews-ttv/live_user_{channel}-192x108.jpg')
    if remote_image is None:
        remote_image = _load_remote_image(url)
    if remote_image is None:
        return None
    try:
        return _compose_thumbnail(remote_image)
    except Exception as exc:
        logger.debug('Livestream thumbnail compose failed for %s: %s', stream_url, exc)
        return None

def _render_thumbnail_image(snapshot: Dict[str, Any]) -> Tuple[Image.Image, bool]:
    """Return a composed thumbnail image and placeholder flag for the snapshot data."""
    kind = snapshot.get("kind")
    path = snapshot.get("path")
    badge = snapshot.get("badge") or None
    image_obj: Optional[Image.Image] = None
    placeholder = False
    if kind == "image" and path:
        if isinstance(path, str) and path.startswith(("http://", "https://")):
            remote_image = _load_remote_image(path)
            if remote_image is not None:
                try:
                    image_obj = _compose_thumbnail(remote_image)
                except Exception as exc:  # pragma: no cover - best effort rendering
                    logger.debug("Remote thumbnail compose failed for %s: %s", path, exc)
                    image_obj = None
        else:
            media_path = _resolve_media_path(path)
            if media_path is not None:
                image_obj = _create_thumbnail_image(media_path)
    elif kind == "video" and path:
        media_path = _resolve_media_path(path)
        if media_path is not None:
            image_obj = _create_video_thumbnail(media_path)
    elif kind == "livestream":
        image_obj = _create_livestream_thumbnail(snapshot.get("stream_url"))
    if image_obj is None:
        placeholder = True
        if kind == "video":
            badge_text = badge or "Video"
        elif kind == "livestream":
            badge_text = badge or "Live"
        else:
            badge_text = badge or "Image"
        image_obj = _generate_placeholder_thumbnail(badge_text)
    return image_obj, placeholder


def _thumbnail_image_to_bytes(image: Image.Image) -> io.BytesIO:
    buffer = io.BytesIO()
    image.convert('RGB').save(buffer, format='JPEG', quality=THUMBNAIL_JPEG_QUALITY, optimize=True)
    buffer.seek(0)
    return buffer
    buffer = io.BytesIO()
    image.convert('RGB').save(buffer, format='JPEG', quality=THUMBNAIL_JPEG_QUALITY, optimize=True)
    buffer.seek(0)
    return buffer

def _compute_thumbnail_snapshot(stream_id: str) -> Optional[Dict[str, Any]]:
    conf = settings.get(stream_id)
    if not isinstance(conf, dict):
        return None
    runtime = _get_stream_runtime_state(stream_id)
    media_mode = runtime.get('media_mode')
    if media_mode not in MEDIA_MODE_CHOICES:
        media_mode_value = conf.get('media_mode')
        if isinstance(media_mode_value, str):
            candidate = media_mode_value.strip().lower()
            media_mode = candidate if candidate in MEDIA_MODE_CHOICES else None
        if media_mode not in MEDIA_MODE_CHOICES:
            media_mode = _infer_media_mode(conf)
    path = runtime.get('path')
    if path is None:
        path = conf.get('selected_image')
    stream_url = runtime.get('stream_url')
    if stream_url is None:
        stream_url = conf.get('stream_url')
    timestamp = runtime.get('timestamp') or time.time()
    kind = runtime.get('kind')
    if not kind and path:
        kind = _detect_media_kind(path)
    if media_mode == MEDIA_MODE_LIVESTREAM:
        kind = 'livestream'
    elif media_mode == MEDIA_MODE_VIDEO:
        kind = 'video'
    elif media_mode == MEDIA_MODE_AI:
        kind = 'image'
    elif media_mode == MEDIA_MODE_PICSUM:
        kind = 'image'
    else:
        kind = kind or 'image'
    placeholder = False
    if not path and media_mode != MEDIA_MODE_LIVESTREAM:
        placeholder = True
    badge_map = {
        MEDIA_MODE_LIVESTREAM: 'Live',
        MEDIA_MODE_VIDEO: 'Video',
        MEDIA_MODE_AI: 'AI',
        MEDIA_MODE_PICSUM: 'Picsum',
    }
    badge = badge_map.get(media_mode, 'Image')
    return {
        'stream_id': stream_id,
        'media_mode': media_mode,
        'path': path,
        'kind': kind,
        'stream_url': stream_url,
        'timestamp': timestamp,
        'badge': badge,
        'placeholder': placeholder,
        'source': runtime.get('source'),
    }

def _thumbnail_signature(snapshot: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        snapshot.get("media_mode"),
        snapshot.get("kind"),
        snapshot.get("path"),
        snapshot.get("stream_url"),
    )

def _refresh_stream_thumbnail(stream_id: str, snapshot: Optional[Dict[str, Any]] = None, *, force: bool = False) -> Optional[Dict[str, Any]]:
    """Ensure a cached thumbnail exists for the stream and return client metadata."""
    info = snapshot if snapshot is not None else _compute_thumbnail_snapshot(stream_id)
    if info is None:
        return None
    signature = _thumbnail_signature(info)
    with STREAM_RUNTIME_LOCK:
        entry = STREAM_RUNTIME_STATE.setdefault(stream_id, {})
        existing = entry.get("thumbnail")
        if not force and isinstance(existing, dict) and existing.get("_signature") == signature:
            return _public_thumbnail_payload(existing)
    image_obj, placeholder = _render_thumbnail_image(info)
    if image_obj is None:
        return None
    buffer = _thumbnail_image_to_bytes(image_obj)
    binary = buffer.getvalue()
    updated_ts = time.time()
    record: Dict[str, Any] = {
        "url": None,
        "placeholder": placeholder,
        "badge": info.get("badge"),
        "updated_at": _runtime_timestamp_to_iso(updated_ts),
        "_signature": signature,
        "_updated_ts": updated_ts,
    }
    saved_path: Optional[Path] = None
    data_url: Optional[str] = None

    cache_dir = _ensure_thumbnail_dir()
    if cache_dir is not None:
        target_path = _thumbnail_disk_path(stream_id)
        temp_path = target_path.with_suffix(".tmp")
        try:
            with open(temp_path, "wb") as fh:
                fh.write(binary)
            os.replace(temp_path, target_path)
            saved_path = target_path
            record["url"] = _thumbnail_public_url(stream_id)
        except OSError as exc:  # pragma: no cover - filesystem differences best effort
            logger.debug("Failed to persist thumbnail for %s: %s", stream_id, exc)
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except OSError:
                pass

    if saved_path is None:
        encoded = base64.b64encode(binary).decode("ascii")
        data_url = f"data:image/jpeg;base64,{encoded}"
        record["url"] = data_url

    if saved_path is not None:
        record["_path"] = str(saved_path)
    if data_url is not None:
        record["_data_url"] = data_url

    with STREAM_RUNTIME_LOCK:
        entry = STREAM_RUNTIME_STATE.setdefault(stream_id, {})
        entry["thumbnail"] = record

    payload = _public_thumbnail_payload(record)
    if payload:
        safe_emit(
            "thumbnail_update",
            {
                "stream": stream_id,
                "url": record["url"],
                "placeholder": placeholder,
                "badge": info.get("badge"),
                "updated_at": payload.get("updated_at"),
            },
        )
    return payload
ensure_ai_presets_storage()
# Backfill defaults for existing stream entries
for k, v in list(settings.items()):
    if not k.startswith("_") and isinstance(v, dict):
        v.setdefault("label", k.capitalize())
        v.setdefault("shuffle", True)
        v.setdefault("hide_nsfw", False)
        if v.get("image_quality") not in IMAGE_QUALITY_CHOICES:
            v["image_quality"] = "auto"
        ensure_ai_defaults(v)
        ensure_picsum_defaults(v)
        ensure_background_defaults(v)
        ensure_tag_defaults(v)
        mode_value = v.get("video_playback_mode")
        if not isinstance(mode_value, str) or mode_value.lower().strip() not in VIDEO_PLAYBACK_MODES:
            v["video_playback_mode"] = "duration"
        else:
            v["video_playback_mode"] = mode_value.lower().strip()
        if "video_volume" not in v or not isinstance(v.get("video_volume"), (int, float)):
            v["video_volume"] = 1.0
        else:
            try:
                vol = float(v["video_volume"])
            except (TypeError, ValueError):
                vol = 1.0
            v["video_volume"] = max(0.0, min(1.0, vol))
        if "selected_media_kind" not in v or not isinstance(v.get("selected_media_kind"), str):
            v["selected_media_kind"] = _detect_media_kind(v.get("selected_image"))
        else:
            kind = v["selected_media_kind"].strip().lower()
            if kind not in ("image", "video"):
                v["selected_media_kind"] = _detect_media_kind(v.get("selected_image"))
            else:
                v["selected_media_kind"] = kind
        media_mode_raw = v.get("media_mode")
        media_mode = media_mode_raw.strip().lower() if isinstance(media_mode_raw, str) else ""
        if media_mode not in MEDIA_MODE_CHOICES:
            media_mode = _infer_media_mode(v)
        if media_mode == MEDIA_MODE_AI:
            desired_mode = AI_MODE
        elif media_mode == MEDIA_MODE_LIVESTREAM:
            desired_mode = "livestream"
        else:
            current_mode_raw = v.get("mode")
            current_mode = current_mode_raw.strip().lower() if isinstance(current_mode_raw, str) else ""
            allowed = MEDIA_MODE_VARIANTS.get(media_mode, {"random", "specific"})
            if current_mode in allowed:
                desired_mode = current_mode
            else:
                desired_mode = "random" if "random" in allowed else next(iter(sorted(allowed)))
        v["media_mode"] = media_mode
        v["mode"] = desired_mode
        if media_mode == MEDIA_MODE_VIDEO and desired_mode == "specific":
            v["video_playback_mode"] = "loop"
        _update_stream_runtime_state(
            k,
            path=v.get("selected_image"),
            kind=v.get("selected_media_kind"),
            media_mode=v.get("media_mode"),
            stream_url=v.get("stream_url"),
            source="startup",
        )
        _set_runtime_embed_metadata(k, v.get("embed_metadata"))

# Ensure notes key exists
settings.setdefault("_notes", "")

# Ensure groups key exists
settings.setdefault("_groups", {})

def _path_contains_nsfw(value: Optional[str]) -> bool:
    return bool(value and NSFW_KEYWORD in value.lower())


def _filter_nsfw_images(paths: List[str], hide_nsfw: bool) -> List[str]:
    if not hide_nsfw:
        return paths
    return [p for p in paths if not _path_contains_nsfw(p)]


def _parse_truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def get_subfolders(hide_nsfw: bool = False) -> List[str]:
    subfolders: List[str] = []
    seen: Set[str] = set()

    def _add(value: str) -> None:
        if value not in seen:
            seen.add(value)
            subfolders.append(value)

    _add("all")
    for root in AVAILABLE_MEDIA_ROOTS:
        if hide_nsfw and _path_contains_nsfw(root.alias):
            continue
        try:
            with os.scandir(root.path) as scan:
                for entry in scan:
                    if not entry.is_dir():
                        continue
                    if _should_ignore_media_name(entry.name):
                        continue
                    folder_key = _build_virtual_media_path(root.alias, entry.name)
                    if hide_nsfw and _path_contains_nsfw(folder_key):
                        continue
                    _add(folder_key)
        except OSError:
            continue
    return subfolders


def get_folder_inventory(hide_nsfw: bool = False) -> List[Dict[str, Any]]:
    inventory: List[Dict[str, Any]] = []
    for name in get_subfolders(hide_nsfw=hide_nsfw):
        media_entries = list_media(name, hide_nsfw=hide_nsfw)
        has_images = any(entry.get("kind") == "image" for entry in media_entries)
        has_videos = any(entry.get("kind") == "video" for entry in media_entries)
        if "/" in name:
            display_name = name.split("/", 1)[1]
        else:
            display_name = name
        inventory.append({
            "name": name,
            "display_name": display_name,
            "has_images": has_images,
            "has_videos": has_videos,
        })
    return inventory


@app.route("/api/media/list", methods=["GET"])
def api_media_list():
    path = request.args.get("path", "")
    page = max(1, _as_int(request.args.get("page"), 1))
    page_size = _as_int(request.args.get("page_size"), 100)
    page_size = max(1, min(page_size, 500))
    sort = request.args.get("sort", "name")
    order = request.args.get("order") or request.args.get("direction") or "asc"
    hide_nsfw_raw = request.args.get("hide_nsfw")
    hide_nsfw = True if hide_nsfw_raw is None else _parse_truthy(hide_nsfw_raw)
    try:
        payload = MEDIA_MANAGER.list_directory(
            path,
            hide_nsfw=hide_nsfw,
            page=page,
            page_size=page_size,
            sort=sort,
            order=order or "asc",
        )
    except MediaManagerError as exc:
        return _media_error_response(exc)
    return jsonify(payload)


@app.route("/api/media/create_folder", methods=["POST"])
def api_media_create_folder():
    payload = request.get_json(silent=True) or {}
    try:
        _require_media_edit()
        parent = payload.get("path") or ""
        name = payload.get("name")
        if not name or not isinstance(name, str):
            raise MediaManagerError("Folder name is required", code="invalid_name")
        new_path = MEDIA_MANAGER.create_folder(parent, name)
    except MediaManagerError as exc:
        return _media_error_response(exc)
    logger.info("media.create_folder parent=%s name=%s", parent or "", name)
    return jsonify({"path": new_path, "name": _virtual_leaf(new_path)})


@app.route("/api/media/rename", methods=["POST"])
def api_media_rename():
    payload = request.get_json(silent=True) or {}
    try:
        _require_media_edit()
        target_path = payload.get("path")
        new_name = payload.get("new_name") or payload.get("name")
        if not target_path or not isinstance(target_path, str):
            raise MediaManagerError("Path is required", code="invalid_request")
        if not new_name or not isinstance(new_name, str):
            raise MediaManagerError("New name is required", code="invalid_name")
        updated = MEDIA_MANAGER.rename(target_path, new_name)
    except MediaManagerError as exc:
        return _media_error_response(exc)
    logger.info("media.rename path=%s new_name=%s", target_path, new_name)
    return jsonify({"path": updated, "name": _virtual_leaf(updated)})


@app.route("/api/media/delete", methods=["DELETE"])
def api_media_delete():
    payload = request.get_json(silent=True) or {}
    target = payload.get("path") if isinstance(payload, dict) else None
    if not target:
        target = request.args.get("path")
    try:
        _require_media_edit()
        if not target or not isinstance(target, str):
            raise MediaManagerError("Path is required", code="invalid_request")
        MEDIA_MANAGER.delete(target)
    except MediaManagerError as exc:
        return _media_error_response(exc)
    logger.info("media.delete path=%s", target)
    return jsonify({"ok": True})


@app.route("/api/media/upload", methods=["POST"])
def api_media_upload():
    try:
        _require_media_edit()
        destination = request.form.get("path") or ""
        files = request.files.getlist("files")
        if not files:
            raise MediaManagerError("No files were provided", code="invalid_request")
        saved = MEDIA_MANAGER.upload(destination, files)
    except MediaManagerError as exc:
        return _media_error_response(exc)
    logger.info("media.upload path=%s count=%d", destination, len(saved))
    return jsonify({"uploaded": saved, "count": len(saved)})


@app.route("/api/media/thumbnail", methods=["GET"])
def api_media_thumbnail():
    path = request.args.get("path")
    if not path:
        return jsonify({"error": "Path is required", "code": "invalid_request"}), 400
    width_raw = request.args.get("w")
    height_raw = request.args.get("h")
    width = _as_int(width_raw, MEDIA_THUMB_WIDTH) if width_raw else MEDIA_THUMB_WIDTH
    if width <= 0:
        width = MEDIA_THUMB_WIDTH
    height = _as_int(height_raw, 0) if height_raw else None
    if height is not None and height <= 0:
        height = None
    if request.args.get("meta"):
        try:
            metadata = MEDIA_MANAGER.get_thumbnail_metadata(path, width=width, height=height)
        except MediaManagerError as exc:
            return _media_error_response(exc)
        return jsonify(metadata)
    try:
        thumb_path, source_mtime, etag = MEDIA_MANAGER.get_thumbnail(path, width=width, height=height)
    except MediaManagerError as exc:
        return _media_error_response(exc)
    etag_value = str(etag)
    weak = False
    if etag_value.startswith("W/"):
        weak = True
        etag_value = etag_value[2:]
    etag_value = etag_value.strip('"')
    quoted_etag = quote_etag(etag_value, weak=weak)
    incoming_etag = request.headers.get("If-None-Match")
    if incoming_etag and incoming_etag == quoted_etag:
        response = app.response_class(status=304)
        response.headers["ETag"] = quoted_etag
        response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        return response
    response = send_file(
        thumb_path,
        mimetype="image/jpeg",
        conditional=True,
    )
    response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
    response.headers["ETag"] = quoted_etag
    response.headers["Last-Modified"] = http_date(source_mtime)
    return response


@app.route("/api/media/preview_frame", methods=["GET"])
def api_media_preview_frame():
    path = request.args.get("path")
    if not path:
        return jsonify({"error": "Path is required", "code": "invalid_request"}), 400
    index_raw = request.args.get("i") or request.args.get("index") or "1"
    try:
        index = int(index_raw)
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid frame index", "code": "invalid_request"}), 400
    try:
        frame_path, source_mtime, etag, frame_count = MEDIA_MANAGER.get_preview_frame(path, index)
    except MediaManagerError as exc:
        fallback_codes = {"preview_failed", "preview_skipped", "preview_disabled", "unsupported_media", "not_found"}
        if exc.code in fallback_codes:
            try:
                thumb_path, source_mtime, thumb_etag = MEDIA_MANAGER.get_thumbnail(path)
            except MediaManagerError:
                return _media_error_response(exc)
            etag_value = str(thumb_etag)
            weak = False
            if etag_value.startswith("W/"):
                weak = True
                etag_value = etag_value[2:]
            etag_value = etag_value.strip('"')
            quoted_etag = quote_etag(etag_value, weak=weak)
            incoming_etag = request.headers.get("If-None-Match")
            if incoming_etag and incoming_etag == quoted_etag:
                response = app.response_class(status=304)
                response.headers["ETag"] = quoted_etag
                response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
                response.headers["X-Preview-Frame-Count"] = "0"
                response.headers["X-Preview-Fallback"] = "thumbnail"
                return response
            response = send_file(
                thumb_path,
                mimetype="image/jpeg",
                conditional=True,
            )
            response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
            response.headers["ETag"] = quoted_etag
            response.headers["Last-Modified"] = http_date(source_mtime)
            response.headers["X-Preview-Frame-Count"] = "0"
            response.headers["X-Preview-Fallback"] = "thumbnail"
            return response
        return _media_error_response(exc)
    etag_value = str(etag)
    weak = False
    if etag_value.startswith("W/"):
        weak = True
        etag_value = etag_value[2:]
    etag_value = etag_value.strip('"')
    quoted_etag = quote_etag(etag_value, weak=weak)
    incoming_etag = request.headers.get("If-None-Match")
    if incoming_etag and incoming_etag == quoted_etag:
        response = app.response_class(status=304)
        response.headers["ETag"] = quoted_etag
        response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        return response
    response = send_file(
        frame_path,
        mimetype="image/webp",
        conditional=True,
    )
    response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
    response.headers["ETag"] = quoted_etag
    response.headers["Last-Modified"] = http_date(source_mtime)
    response.headers["X-Preview-Frame-Count"] = str(frame_count)
    return response


@app.route("/folders", methods=["GET"])
def folders_collection():
    hide_nsfw = _parse_truthy(request.args.get("hide_nsfw"))
    inventory = get_folder_inventory(hide_nsfw=hide_nsfw)
    return jsonify(inventory)


def list_images(folder="all", hide_nsfw: bool = False):
    """Return cached image paths for the folder, refreshing when necessary."""
    images = refresh_image_cache(folder)
    return _filter_nsfw_images(images, hide_nsfw)

def list_media(folder="all", hide_nsfw: bool = False) -> List[Dict[str, Any]]:
    """Return cached media entries (images and videos) for the folder."""
    folder_key = _normalize_folder_key(folder)
    refresh_image_cache(folder)
    with IMAGE_CACHE_LOCK:
        cached_entry = IMAGE_CACHE.get(folder_key)
        if cached_entry:
            media = [dict(item) for item in cached_entry.get("media", [])]
        else:
            media = []
    if hide_nsfw:
        media = [item for item in media if not _path_contains_nsfw(item.get("path"))]
    return media

if playback_manager is None:
    playback_manager = StreamPlaybackManager()
    playback_manager.bootstrap(settings)
    atexit.register(playback_manager.stop)


def _hls_url_fingerprint(original_url: str) -> str:
    if not original_url:
        return "none"
    try:
        return hashlib.sha1(original_url.encode("utf-8")).hexdigest()[:10]
    except Exception:
        return "error"


def _log_hls_event(event: str, stream_id: str, original_url: str, **extra: Any) -> None:
    if not LIVE_HLS_ASYNC:
        return
    details = " ".join(f"{k}={v}" for k, v in sorted(extra.items())) if extra else ""
    logger.info(
        "%s.%s stream=%s url=%s%s",
        HLS_LOG_PREFIX,
        event,
        stream_id or "-",
        _hls_url_fingerprint(original_url),
        f" {details}" if details else "",
    )


def _record_hls_metric(name: str, delta: int = 1) -> None:
    if not LIVE_HLS_ASYNC:
        return
    with HLS_LOCK:
        HLS_METRICS[name] = HLS_METRICS.get(name, 0) + delta


def _live_hls_cache_key(stream_id: Optional[str], original_url: str) -> str:
    sid = (stream_id or "").strip() or "unknown"
    return f"live:{sid}:{original_url}"


def _is_manifest_url(candidate: Optional[str]) -> bool:
    if not isinstance(candidate, str):
        return False
    lower = candidate.lower()
    return any(marker in lower for marker in (".m3u8", ".mpd", "manifest.mpd", "format=m3u8"))


def _extract_hls_candidate(info: Dict[str, Any]) -> Optional[str]:
    if not isinstance(info, dict):
        return None

    for key in ("url", "manifest_url", "hls_manifest_url"):
        candidate = info.get(key)
        if isinstance(candidate, str) and _is_manifest_url(candidate):
            return candidate

    for formats_key in ("formats", "requested_formats"):
        formats = info.get(formats_key)
        if not isinstance(formats, list):
            continue
        for fmt in formats:
            if not isinstance(fmt, dict):
                continue
            candidate = fmt.get("url") or fmt.get("manifest_url")
            if not isinstance(candidate, str):
                continue
            protocol = str(fmt.get("protocol") or "").lower()
            ext = str(fmt.get("ext") or "").lower()
            if _is_manifest_url(candidate) or "m3u8" in protocol or "dash" in protocol or ext in {"m3u8", "mpd"}:
                return candidate
        for fmt in formats:
            if not isinstance(fmt, dict):
                continue
            manifest_url = fmt.get("manifest_url")
            if isinstance(manifest_url, str) and _is_manifest_url(manifest_url):
                return manifest_url

    entries = info.get("entries")
    if isinstance(entries, list):
        for entry in entries:
            candidate = _extract_hls_candidate(entry)
            if candidate:
                return candidate

    return None


def _detect_hls_stream_url(original_url: str) -> Optional[str]:
    if not original_url:
        return None
    if YoutubeDL is None:
        raise RuntimeError("yt_dlp module is not available")
    ydl_opts = {
        "quiet": True,
        "nocheckcertificate": True,
        "skip_download": True,
        "noplaylist": True,
        "extract_flat": False,
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(original_url, download=False)
    if not info:
        return None
    return _extract_hls_candidate(info)


def _get_hls_cache_entry(key: str) -> Optional[HLSCacheEntry]:
    with HLS_LOCK:
        return HLS_CACHE.get(key)


def _cancel_hls_job(key: str) -> bool:
    with HLS_LOCK:
        future = HLS_JOBS.get(key)
        if not future:
            return False
        cancelled = future.cancel()
        if cancelled:
            HLS_JOBS.pop(key, None)
        return cancelled


def schedule_hls_detection(stream_id: str, original_url: str) -> None:
    if not LIVE_HLS_ASYNC or not original_url or HLS_EXECUTOR is None:
        return
    key = _live_hls_cache_key(stream_id, original_url)
    with HLS_LOCK:
        future = HLS_JOBS.get(key)
        if future and not future.done():
            return
        future = HLS_EXECUTOR.submit(_run_hls_detection_job, key, stream_id, original_url)
        HLS_JOBS[key] = future
        in_flight = len(HLS_JOBS)
    _record_hls_metric("jobs_started")
    _log_hls_event("job_start", stream_id, original_url, inflight=in_flight)


def _run_hls_detection_job(key: str, stream_id: str, original_url: str) -> None:
    started_at = time.time()
    _log_hls_event("job_run", stream_id, original_url)
    try:
        url = _detect_hls_stream_url(original_url)
        entry = HLSCacheEntry(url=url, extracted_at=time.time(), error=None)
        success = bool(url)
        error_text = None
    except Exception as exc:
        entry = HLSCacheEntry(url=None, extracted_at=time.time(), error=str(exc))
        success = False
        error_text = entry.error
        _record_hls_metric("errors")
    finally:
        _record_hls_metric("jobs_completed")
    with HLS_LOCK:
        HLS_CACHE[key] = entry
        HLS_JOBS.pop(key, None)
        in_flight = len(HLS_JOBS)
    duration_ms = int((time.time() - started_at) * 1000)
    _log_hls_event(
        "job_done",
        stream_id,
        original_url,
        success=success,
        inflight=in_flight,
        duration_ms=duration_ms,
        error=error_text or "none",
    )
    if entry.url:
        with app.app_context():
            safe_emit(
                "live_hls_ready",
                {"stream_id": stream_id, "cache_key": key, "hls_url": entry.url},
            )


def try_get_hls(original_url):
    """Legacy synchronous helper retained for compatibility."""
    try:
        return _detect_hls_stream_url(original_url)
    except Exception:
        return None


@app.route("/media/manage")
def media_management_page():
    roots_payload = [
        {
            "alias": root.alias,
            "display_name": root.display_name or root.alias,
            "path": f"{root.alias}:/",
        }
        for root in AVAILABLE_MEDIA_ROOTS
    ]
    return render_template(
        "media_manage.html",
        media_roots=roots_payload,
        media_allow_edit=MEDIA_MANAGEMENT_ALLOW_EDIT,
        media_allowed_exts=MEDIA_ALLOWED_EXTS,
        media_upload_max_mb=MEDIA_UPLOAD_MAX_MB,
        media_thumb_width=MEDIA_THUMB_WIDTH,
        media_preview_enabled=MEDIA_PREVIEW_ENABLED,
        media_preview_frames=MEDIA_PREVIEW_FRAMES,
        media_preview_width=MEDIA_PREVIEW_WIDTH,
        media_preview_max_duration=MEDIA_PREVIEW_MAX_DURATION,
    )

@app.route("/")
def dashboard():
    folder_inventory = get_folder_inventory()
    subfolders = [item["name"] for item in folder_inventory]
    streams = {k: v for k, v in settings.items() if not k.startswith("_")}
    for stream_id, conf in streams.items():
        if not isinstance(conf, dict):
            continue
        quality = conf.get("image_quality")
        if not isinstance(quality, str) or quality.strip().lower() not in IMAGE_QUALITY_CHOICES:
            conf["image_quality"] = "auto"
        else:
            conf["image_quality"] = quality.strip().lower()
        ensure_background_defaults(conf)
        ensure_tag_defaults(conf)
        _refresh_embed_metadata(stream_id, conf)
    groups = sorted(list(settings.get("_groups", {}).keys()))
    return render_template(
        "index.html",
        subfolders=subfolders,
        folder_inventory=folder_inventory,
        stream_settings=streams,
        groups=groups,
        global_tags=get_global_tags(),
        post_processors=STABLE_HORDE_POST_PROCESSORS,
        max_loras=STABLE_HORDE_MAX_LORAS,
        clip_skip_range=STABLE_HORDE_CLIP_SKIP_RANGE,
        strength_range=STABLE_HORDE_STRENGTH_RANGE,
        denoise_range=STABLE_HORDE_DENOISE_RANGE,
    )


@app.route("/debug")
def debug_page() -> str:
    cfg = load_config()
    logging_config = cfg.get("logging") or {}
    service_name = cfg.get("SERVICE_NAME", "echomosaic.service")
    return render_template(
        "debug.html",
        service_name=service_name,
        logging_config=logging_config,
        default_initial_lines=debug_manager.DEFAULT_INITIAL_LINES,
    )


@app.route("/api/debug/stream")
def debug_stream() -> Response:
    include_initial = request.args.get("initial", "1").strip().lower() not in {"0", "false", "no"}
    initial_limit = debug_manager.DEFAULT_INITIAL_LINES if include_initial else 0
    stream_generator = stream_with_context(debug_manager.stream_journal_follow(initial_limit=initial_limit))
    response = Response(stream_generator, mimetype="text/event-stream")
    response.headers["Cache-Control"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"
    return response


@app.route("/api/debug/download")
def debug_download() -> Response:
    limit = request.args.get("limit", type=int) or debug_manager.DEFAULT_DOWNLOAD_LINES
    try:
        log_data = debug_manager.get_recent_log_lines(limit=limit)
    except debug_manager.JournalAccessError as exc:
        return Response(str(exc), status=503, mimetype="text/plain")

    cfg = load_config()
    service_name = cfg.get("SERVICE_NAME", "echomosaic.service")
    safe_name = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in service_name)
    filename = f"{safe_name or 'echomosaic'}.log"

    response = Response(log_data, mimetype="text/plain")
    response.headers["Content-Disposition"] = f'attachment; filename="{filename}"'
    response.headers["Cache-Control"] = "no-store"
    return response


@app.route("/stream")
def mosaic_streams():
    # Dynamic global view: include all streams ("online" assumed as configured)
    streams = {k: v for k, v in settings.items() if not k.startswith("_")}
    for stream_id, conf in streams.items():
        if not isinstance(conf, dict):
            continue
        ensure_background_defaults(conf)
        ensure_tag_defaults(conf)
        _refresh_embed_metadata(stream_id, conf)
    return render_template("streams.html", stream_settings=streams)

def _slugify(name: str) -> str:
    name = (name or "").strip().lower()
    name = re.sub(r"[^a-z0-9]+", "-", name)
    name = re.sub(r"-+", "-", name).strip("-")
    return name or ""

@app.template_filter('slugify')
def jinja_slugify(s):
    return _slugify(s)

@app.route("/stream/<name>")
def render_stream(name):
    # Accept either stream id or slugified label
    key = None
    if name in settings:
        key = name
    else:
        wanted = _slugify(name)
        for sid, conf in settings.items():
            if sid.startswith("_"):
                continue
            label = conf.get("label") or sid
            if _slugify(label) == wanted:
                key = sid
                break
    if not key or key not in settings:
        return f"No stream '{name}'", 404
    conf = settings[key]
    config_quality_raw = conf.get("image_quality", "auto")
    if isinstance(config_quality_raw, str):
        config_quality = config_quality_raw.strip().lower()
    else:
        config_quality = "auto"
    if config_quality not in IMAGE_QUALITY_CHOICES:
        config_quality = "auto"
    conf["image_quality"] = config_quality
    ensure_background_defaults(conf)
    ensure_tag_defaults(conf)
    _refresh_embed_metadata(key, conf)
    images = list_images(conf.get("folder", "all"), hide_nsfw=conf.get("hide_nsfw", False))
    requested_quality = (request.args.get("size") or "").strip().lower()
    if requested_quality and requested_quality not in IMAGE_QUALITY_CHOICES:
        requested_quality = ""
    default_quality = requested_quality or config_quality
    return render_template(
        "single_stream.html",
        stream_id=key,
        config=conf,
        images=images,
        default_quality=default_quality,
    )


@app.route("/streams", methods=["POST"])
def add_stream():
    """Create a new stream configuration and return its ID."""
    idx = 1
    while True:
        new_id = f"stream{idx}"
        if new_id not in settings:
            settings[new_id] = default_stream_config()
            settings[new_id]["label"] = new_id.capitalize()
            _update_stream_runtime_state(
                new_id,
                path=settings[new_id].get("selected_image"),
                kind=settings[new_id].get("selected_media_kind"),
                media_mode=settings[new_id].get("media_mode"),
                stream_url=settings[new_id].get("stream_url"),
                source="stream_created",
            )
            if playback_manager is not None:
                playback_manager.update_stream_config(new_id, settings[new_id])
            save_settings(settings)
            if auto_scheduler is not None:
                auto_scheduler.reschedule(new_id)
            safe_emit("streams_changed", {"action": "added", "stream_id": new_id})
            return jsonify({"stream_id": new_id})
        idx += 1


@app.route("/streams/<stream_id>", methods=["DELETE"])
def delete_stream(stream_id):
    if stream_id in settings:
        with ai_jobs_lock:
            ai_jobs.pop(stream_id, None)
            ai_job_controls.pop(stream_id, None)
        _cleanup_temp_outputs(stream_id)
        settings.pop(stream_id)
        with STREAM_RUNTIME_LOCK:
            STREAM_RUNTIME_STATE.pop(stream_id, None)
        if playback_manager is not None:
            playback_manager.remove_stream(stream_id)
        save_settings(settings)
        if auto_scheduler is not None:
            auto_scheduler.remove(stream_id)
        if picsum_scheduler is not None:
            picsum_scheduler.remove(stream_id)
        safe_emit("streams_changed", {"action": "deleted", "stream_id": stream_id})
        return jsonify({"status": "deleted"})
    return jsonify({"error": "not found"}), 404

@app.route("/get-settings/<stream_id>", methods=["GET"])
def get_stream_settings(stream_id):
    if stream_id not in settings:
        return jsonify({"error": f"No stream '{stream_id}' found"}), 404
    conf = settings[stream_id]
    ensure_ai_defaults(conf)
    ensure_picsum_defaults(conf)
    ensure_background_defaults(conf)
    ensure_tag_defaults(conf)
    return jsonify(conf)


@app.route("/stream/state/<stream_id>", methods=["GET"])
def get_stream_playback_state(stream_id):
    if playback_manager is None:
        return jsonify({"error": "Playback manager unavailable"}), 503
    conf = settings.get(stream_id)
    if conf is None or not isinstance(conf, dict):
        return jsonify({"error": f"No stream '{stream_id}' found"}), 404
    state = playback_manager.get_state(stream_id)
    if state is None:
        playback_manager.update_stream_config(stream_id, conf)
        state = playback_manager.ensure_started(stream_id)
    if state is None:
        return jsonify({"error": "No playback state"}), 404
    return jsonify(state)


@app.route("/settings/<stream_id>", methods=["POST"])
def update_stream_settings(stream_id):
    if stream_id not in settings:
        return jsonify({"error": f"No stream '{stream_id}' found"}), 404

    data = request.get_json(silent=True) or {}
    conf = settings[stream_id]
    ensure_ai_defaults(conf)
    ensure_picsum_defaults(conf)
    previous_mode = conf.get("mode")

    stream_url_changed = False
    media_mode_changed = False

    # We'll add new keys for YouTube: "yt_cc", "yt_mute", "yt_quality"
    for key in ["mode", "folder", "selected_image", "duration", "shuffle", "stream_url",
                "image_quality", "yt_cc", "yt_mute", "yt_quality", "label", "hide_nsfw",
                "background_blur_enabled", "background_blur_amount", "video_playback_mode",
                "video_volume", "selected_media_kind", "media_mode", TAG_KEY]:
        if key in data:
            val = data[key]
            if key == "stream_url":
                normalized_url = val.strip() if isinstance(val, str) else ""
                conf[key] = normalized_url if normalized_url and normalized_url.lower() != "none" else None
                stream_url_changed = True
            elif key == "mode":
                conf[key] = val
                if isinstance(val, str) and val.strip().lower() == "livestream":
                    media_mode_changed = True
            elif key == "label":
                # Enforce unique label slug across streams (ignoring case/spacing)
                new_label = (val or "").strip()
                new_slug = _slugify(new_label)
                if new_slug:
                    for other_id, other_conf in settings.items():
                        if other_id.startswith("_") or other_id == stream_id:
                            continue
                        other_label = (other_conf.get("label") or other_id)
                        if _slugify(other_label) == new_slug:
                            return jsonify({"error": "Another stream already uses this name"}), 400
                conf[key] = new_label
            elif key == TAG_KEY:
                tags_payload = _sanitize_stream_tags(val)
                conf[TAG_KEY] = tags_payload
                register_global_tags(tags_payload)
            elif key == "hide_nsfw":
                conf[key] = bool(val)
            elif key == "background_blur_enabled":
                conf[key] = _coerce_bool(val, conf.get(key, False))
            elif key == "background_blur_amount":
                amount = _coerce_int(val, conf.get(key, 50))
                conf[key] = max(0, min(100, amount))
            elif key == "video_playback_mode":
                normalized = (val or "").strip().lower() if isinstance(val, str) else ""
                if normalized not in VIDEO_PLAYBACK_MODES:
                    normalized = "duration"
                conf[key] = normalized
            elif key == "video_volume":
                try:
                    volume = float(val)
                except (TypeError, ValueError):
                    volume = conf.get(key, 1.0)
                conf[key] = max(0.0, min(1.0, volume))
            elif key == "selected_media_kind":
                if isinstance(val, str):
                    kind = val.strip().lower()
                else:
                    kind = ""
                if kind not in ("image", "video"):
                    kind = _detect_media_kind(conf.get("selected_image"))
                conf[key] = kind
            elif key == "media_mode":
                if isinstance(val, str):
                    media_mode = val.strip().lower()
                else:
                    media_mode = ""
                if media_mode in MEDIA_MODE_CHOICES:
                    conf["media_mode"] = media_mode
                    if media_mode == MEDIA_MODE_AI:
                        conf["mode"] = AI_MODE
                    elif media_mode == MEDIA_MODE_LIVESTREAM:
                        conf["mode"] = "livestream"
                    media_mode_changed = True
            elif key == "selected_image":
                conf[key] = val
                if "selected_media_kind" not in data:
                    conf["selected_media_kind"] = _detect_media_kind(val)
            elif key == "image_quality":
                normalized = (val or "").strip().lower() if isinstance(val, str) else ""
                if normalized not in IMAGE_QUALITY_CHOICES:
                    normalized = "auto"
                conf[key] = normalized
            else:
                conf[key] = val

    if PICSUM_SETTINGS_KEY in data:
        incoming_picsum = data.get(PICSUM_SETTINGS_KEY)
        conf[PICSUM_SETTINGS_KEY] = _sanitize_picsum_settings(
            incoming_picsum,
            defaults=conf.get(PICSUM_SETTINGS_KEY),
        )
        if isinstance(incoming_picsum, dict):
            seed_flag: Optional[bool] = None
            if "seed_custom" in incoming_picsum:
                seed_flag = bool(incoming_picsum.get("seed_custom"))
            elif "seed" in incoming_picsum:
                seed_candidate = incoming_picsum.get("seed")
                if isinstance(seed_candidate, str):
                    seed_flag = bool(seed_candidate.strip())
                else:
                    seed_flag = bool(seed_candidate)
            if seed_flag is not None:
                conf["_picsum_seed_custom"] = seed_flag
        if "_picsum_seed_custom" not in conf:
            conf["_picsum_seed_custom"] = False
    else:
        conf[PICSUM_SETTINGS_KEY] = _sanitize_picsum_settings(
            conf.get(PICSUM_SETTINGS_KEY),
            defaults=default_picsum_settings(),
        )

    media_mode = conf.get("media_mode")
    if isinstance(media_mode, str):
        media_mode = media_mode.strip().lower()
    else:
        media_mode = ""
    if media_mode not in MEDIA_MODE_CHOICES:
        media_mode = _infer_media_mode(conf)
    conf["media_mode"] = media_mode

    current_mode_raw = conf.get("mode")
    current_mode = current_mode_raw.strip().lower() if isinstance(current_mode_raw, str) else ""
    if media_mode == MEDIA_MODE_AI:
        conf["mode"] = AI_MODE
    elif media_mode == MEDIA_MODE_LIVESTREAM:
        conf["mode"] = "livestream"
    else:
        allowed = MEDIA_MODE_VARIANTS.get(media_mode, {"random", "specific"})
        if current_mode not in allowed:
            fallback_mode = "random" if "random" in allowed else next(iter(sorted(allowed)))
            conf["mode"] = fallback_mode
        else:
            conf["mode"] = current_mode
        if media_mode == MEDIA_MODE_VIDEO:
            if conf["mode"] == "specific":
                conf["video_playback_mode"] = "loop"
            elif conf["mode"] == "random" and conf.get("video_playback_mode") == "loop":
                conf["video_playback_mode"] = "duration"

    _refresh_embed_metadata(stream_id, conf, force=stream_url_changed or media_mode_changed)

    mode_requested = data.get("mode")
    if (
        mode_requested == AI_MODE
        and previous_mode != AI_MODE
        and not conf.get("_ai_customized", False)
    ):
        conf[AI_SETTINGS_KEY] = default_ai_settings()
        conf[AI_STATE_KEY] = default_ai_state()
        conf["_ai_customized"] = False

    if isinstance(data.get("ai_settings"), dict):
        conf[AI_SETTINGS_KEY] = _sanitize_ai_settings(data["ai_settings"], conf[AI_SETTINGS_KEY])
        conf["_ai_customized"] = not _ai_settings_match_defaults(conf[AI_SETTINGS_KEY])

    ensure_background_defaults(conf)
    ensure_tag_defaults(conf)
    _update_stream_runtime_state(
        stream_id,
        path=conf.get("selected_image"),
        kind=conf.get("selected_media_kind"),
        media_mode=conf.get("media_mode"),
        stream_url=conf.get("stream_url"),
        source="settings_update",
    )
    if playback_manager is not None:
        playback_manager.update_stream_config(stream_id, conf)
    save_settings(settings)
    if auto_scheduler is not None:
        auto_scheduler.reschedule(stream_id)
    if picsum_scheduler is not None:
        picsum_scheduler.reschedule(stream_id)
    safe_emit("refresh", {"stream_id": stream_id, "config": conf, "tags": get_global_tags()})
    return jsonify({"status": "success", "new_config": conf, "tags": get_global_tags()})


def _refresh_picsum_stream(
    stream_id: str, incoming_settings: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    conf = settings.get(stream_id)
    if not isinstance(conf, dict):
        return None

    ensure_picsum_defaults(conf)
    defaults = conf.get(PICSUM_SETTINGS_KEY) or default_picsum_settings()
    source_settings = incoming_settings if isinstance(incoming_settings, dict) else defaults
    sanitized = _sanitize_picsum_settings(source_settings, defaults=defaults)
    result_settings = deepcopy(sanitized)

    assignment = assign_new_picsum_to_stream(
        stream_id,
        {
            "width": sanitized["width"],
            "height": sanitized["height"],
            "blur": sanitized["blur"],
            "grayscale": bool(sanitized.get("grayscale")),
        },
    )
    normalized_params = assignment["params"]
    seed_value = assignment["seed"]
    seed_was_custom = bool(assignment["seed_custom"])
    image_url = assignment["url"]

    sanitized.update(
        {
            "width": normalized_params["width"],
            "height": normalized_params["height"],
            "blur": normalized_params["blur"],
            "grayscale": normalized_params["grayscale"],
            "seed": seed_value,
            "seed_custom": seed_was_custom,
        }
    )
    result_settings.update(
        {
            "width": normalized_params["width"],
            "height": normalized_params["height"],
            "blur": normalized_params["blur"],
            "grayscale": normalized_params["grayscale"],
            "seed": seed_value,
            "seed_custom": seed_was_custom,
        }
    )

    conf[PICSUM_SETTINGS_KEY] = deepcopy(sanitized)
    conf[PICSUM_SETTINGS_KEY]["seed_custom"] = seed_was_custom
    conf["_picsum_seed_custom"] = seed_was_custom
    conf["selected_image"] = image_url
    conf["selected_media_kind"] = "image"
    conf["media_mode"] = MEDIA_MODE_PICSUM
    conf["mode"] = MEDIA_MODE_PICSUM

    runtime_thumbnail = _update_stream_runtime_state(
        stream_id,
        path=image_url,
        kind="image",
        media_mode=MEDIA_MODE_PICSUM,
        source="picsum_refresh",
    )
    with STREAM_RUNTIME_LOCK:
        entry = STREAM_RUNTIME_STATE.setdefault(stream_id, {})
        entry["picsum_seed"] = seed_value
        entry["picsum_seed_custom"] = seed_was_custom

    return {
        "url": image_url,
        "settings": result_settings,
        "seed_custom": seed_was_custom,
        "thumbnail": runtime_thumbnail,
    }


def _broadcast_picsum_update(
    stream_id: str,
    conf: Dict[str, Any],
    image_url: str,
    seed_value: Optional[str],
    seed_custom: bool,
    thumbnail: Optional[Dict[str, Any]],
) -> None:
    refresh_payload = {"stream_id": stream_id, "config": conf, "tags": get_global_tags()}
    safe_emit("refresh", refresh_payload)

    stream_update_payload = {
        "stream_id": stream_id,
        "mode": MEDIA_MODE_PICSUM,
        "media_mode": MEDIA_MODE_PICSUM,
        "status": "playing",
        "seed": seed_value,
        "seed_custom": seed_custom,
        "media": {
            "path": image_url,
            "kind": "image",
            "stream_url": None,
            "seed": seed_value,
            "seed_custom": seed_custom,
        },
        "duration": None,
        "position": 0.0,
        "started_at": None,
        "is_paused": True,
        "server_time": time.time(),
        "thumbnail": thumbnail or _get_runtime_thumbnail_payload(stream_id),
    }
    safe_emit(STREAM_UPDATE_EVENT, stream_update_payload)


@app.route("/picsum/refresh", methods=["POST"])
def refresh_picsum_image():
    payload = request.get_json(silent=True) or {}
    stream_id_raw = payload.get("stream_id")
    stream_id = str(stream_id_raw).strip() if stream_id_raw is not None else ""
    if not stream_id:
        return jsonify({"error": "stream_id is required"}), 400

    conf = settings.get(stream_id)
    if not isinstance(conf, dict):
        return jsonify({"error": f"No stream '{stream_id}' found"}), 404

    ensure_ai_defaults(conf)
    ensure_picsum_defaults(conf)

    incoming_settings_raw = payload.get("settings")
    if not isinstance(incoming_settings_raw, dict):
        incoming_settings_raw = payload.get(PICSUM_SETTINGS_KEY)
        if not isinstance(incoming_settings_raw, dict):
            incoming_settings_raw = None
    incoming_settings = dict(incoming_settings_raw) if incoming_settings_raw is not None else None

    result = _refresh_picsum_stream(stream_id, incoming_settings=incoming_settings)
    if result is None:
        return jsonify({"error": f"No stream '{stream_id}' found"}), 404

    if picsum_scheduler is not None:
        picsum_scheduler.reschedule(stream_id, base_time=time.time())

    save_settings(settings)

    conf = settings.get(stream_id, conf)
    sanitized = result["settings"]
    picsum_conf = conf.get(PICSUM_SETTINGS_KEY, {})
    sanitized["next_auto_trigger"] = picsum_conf.get("next_auto_trigger")
    sanitized["last_auto_trigger"] = picsum_conf.get("last_auto_trigger")

    _broadcast_picsum_update(
        stream_id,
        conf,
        result["url"],
        sanitized.get("seed"),
        bool(result.get("seed_custom")),
        result.get("thumbnail"),
    )

    response = {
        "stream_id": stream_id,
        "url": result["url"],
        "settings": sanitized,
        "seed": sanitized.get("seed"),
        "seed_custom": bool(result.get("seed_custom")),
    }
    return jsonify(response)


@app.route("/settings/ai-defaults", methods=["GET"])
def get_ai_defaults():
    return jsonify({
        "defaults": default_ai_settings(),
        "fallback": deepcopy(AI_FALLBACK_DEFAULTS),
    })


@app.route("/settings/ai-defaults", methods=["POST"])
def update_ai_defaults():
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        return jsonify({"error": "Invalid payload"}), 400

    current_defaults = default_ai_settings()
    sanitized = _sanitize_ai_settings(
        payload,
        current_defaults,
        defaults=AI_FALLBACK_DEFAULTS,
    )
    settings["_ai_defaults"] = sanitized
    new_defaults = default_ai_settings()

    for stream_id, conf in settings.items():
        if stream_id.startswith("_") or not isinstance(conf, dict):
            continue
        ensure_ai_defaults(conf)
        ensure_picsum_defaults(conf)
        if not conf.get("_ai_customized", False):
            conf[AI_SETTINGS_KEY] = deepcopy(new_defaults)
            conf["_ai_customized"] = False

    current_presets = ensure_ai_presets_storage()
    refreshed_presets: Dict[str, Dict[str, Any]] = {}
    for name, preset in current_presets.items():
        refreshed_presets[name] = _sanitize_ai_settings(preset, defaults=AI_FALLBACK_DEFAULTS)
    settings[AI_PRESETS_KEY] = _sorted_presets(refreshed_presets)

    save_settings(settings)
    return jsonify({"status": "success", "defaults": new_defaults})

@app.route("/ai/presets", methods=["GET"])
def list_ai_presets():
    presets = ensure_ai_presets_storage()
    payload = [
        {"name": name, "settings": deepcopy(config)}
        for name, config in presets.items()
    ]
    return jsonify({"presets": payload})


@app.route("/ai/presets", methods=["POST"])
def create_ai_preset():
    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        return jsonify({"error": "Invalid payload"}), 400
    name = str(data.get("name") or "").strip()
    if not name:
        return jsonify({"error": "Preset name is required"}), 400
    settings_payload = data.get("settings")
    if not isinstance(settings_payload, dict):
        return jsonify({"error": "Preset settings must be an object"}), 400
    overwrite = bool(data.get("overwrite"))
    presets = ensure_ai_presets_storage()
    if name in presets and not overwrite:
        return jsonify({"error": "Preset already exists", "status": "exists"}), 409
    sanitized = _sanitize_ai_settings(settings_payload, defaults=AI_FALLBACK_DEFAULTS)
    updated = dict(presets)
    updated[name] = sanitized
    settings[AI_PRESETS_KEY] = _sorted_presets(updated)
    save_settings(settings)
    return jsonify({"status": "saved", "preset": {"name": name, "settings": deepcopy(sanitized)}})


@app.route("/ai/presets/<preset_name>", methods=["PATCH"])
def update_ai_preset(preset_name: str):
    name = (preset_name or "").strip()
    if not name:
        return jsonify({"error": "Preset not found"}), 404
    presets = ensure_ai_presets_storage()
    if name not in presets:
        return jsonify({"error": "Preset not found"}), 404
    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        return jsonify({"error": "Invalid payload"}), 400
    new_name = data.get("name")
    new_settings = data.get("settings")
    target_name = name
    if new_name is not None:
        candidate = str(new_name).strip()
        if not candidate:
            return jsonify({"error": "Preset name cannot be empty"}), 400
        if candidate != name and candidate in presets:
            return jsonify({"error": "Preset with that name already exists"}), 409
        target_name = candidate
    if new_settings is not None and not isinstance(new_settings, dict):
        return jsonify({"error": "Preset settings must be an object"}), 400
    sanitized = _sanitize_ai_settings(new_settings or presets[name], defaults=AI_FALLBACK_DEFAULTS)
    updated = dict(presets)
    updated.pop(name, None)
    updated[target_name] = sanitized
    settings[AI_PRESETS_KEY] = _sorted_presets(updated)
    save_settings(settings)
    return jsonify({"status": "updated", "preset": {"name": target_name, "settings": deepcopy(sanitized)}})


@app.route("/ai/presets/<preset_name>", methods=["DELETE"])
def delete_ai_preset(preset_name: str):
    name = (preset_name or "").strip()
    presets = ensure_ai_presets_storage()
    if name not in presets:
        return jsonify({"error": "Preset not found"}), 404
    updated = dict(presets)
    updated.pop(name, None)
    settings[AI_PRESETS_KEY] = _sorted_presets(updated)
    save_settings(settings)
    return jsonify({"status": "deleted"})

@app.route('/ai/loras')
def ai_loras():
    if requests is None:
        return jsonify({'error': 'LoRA search unavailable'}), 503
    query = (request.args.get('q') or request.args.get('query') or '').strip()
    try:
        limit = int(request.args.get('limit', 20))
    except (TypeError, ValueError):
        limit = 20
    limit = max(1, min(limit, 40))
    params = {
        'types': 'LORA',
        'limit': limit,
    }
    if query:
        params['query'] = query
    try:
        resp = requests.get('https://civitai.com/api/v1/models', params=params, timeout=10)
        resp.raise_for_status()
    except Exception as exc:
        logger.warning('LoRA search failed for %s: %s', query or 'default', exc)
        return jsonify({'error': 'LoRA lookup failed'}), 502
    try:
        payload = resp.json()
    except ValueError as exc:
        logger.warning('LoRA search returned invalid JSON: %s', exc)
        return jsonify({'error': 'LoRA lookup invalid response'}), 502
    items = payload.get('items') or []
    results = []
    for item in items:
        name = item.get('name') or ''
        versions = item.get('modelVersions') or []
        for version in versions:
            version_id = version.get('id')
            if version_id is None:
                continue
            entry = {
                'modelName': name,
                'versionId': version_id,
                'versionName': version.get('name'),
                'triggerWords': version.get('trainedWords') or [],
            }
            results.append(entry)
            if len(results) >= limit:
                break
        if len(results) >= limit:
            break
    return jsonify({
        'results': results,
        'query': query,
        'limit': limit,
        'source': 'civitai',
        'browseUrl': 'https://civitai.com/models?types=LORA&sort=Highest%20Rated',
    })

@app.route("/tags", methods=["GET"])
def list_tags():
    return jsonify({"tags": get_global_tags()})


@app.route("/tags", methods=["POST"])
def create_tag():
    payload = request.get_json(silent=True) or {}
    raw_name = payload.get("name") or payload.get("tag")
    normalized = _normalize_tag_name(raw_name)
    if not normalized:
        return jsonify({"error": "Invalid tag name"}), 400
    tags = get_global_tags()
    lookup = {t.casefold(): t for t in tags}
    existing = lookup.get(normalized.casefold())
    if existing is None:
        tags.append(normalized)
        tags.sort(key=str.lower)
        save_settings(settings)
        canonical = normalized
    else:
        canonical = existing
    return jsonify({"tag": canonical, "tags": tags})


@app.route("/tags/<path:tag_name>", methods=["DELETE"])
def delete_tag(tag_name: str):
    normalized = _normalize_tag_name(tag_name)
    if not normalized:
        return jsonify({"error": "Invalid tag name"}), 400
    tags = get_global_tags()
    lookup = {t.casefold(): (idx, t) for idx, t in enumerate(tags)}
    entry = lookup.get(normalized.casefold())
    if entry is None:
        return jsonify({"error": "Tag not found"}), 404
    idx, canonical = entry
    canon_key = canonical.casefold()
    for sid, conf in settings.items():
        if sid.startswith("_") or not isinstance(conf, dict):
            continue
        stream_tags = conf.get(TAG_KEY) or []
        for tag in stream_tags:
            if isinstance(tag, str) and tag.casefold() == canon_key:
                return jsonify({"error": "Tag is in use"}), 409
    tags.pop(idx)
    save_settings(settings)
    return jsonify({"status": "deleted", "tags": tags})


def _queue_ai_generation(stream_id: str, ai_settings: Dict[str, Any], *, trigger_source: str = "manual") -> Dict[str, Any]:
    if stable_horde_client is None:
        raise AutoGenerationUnavailable('Stable Horde client is not configured')

    conf = settings.get(stream_id)
    if not conf:
        raise AutoGenerationError(f"No stream '{stream_id}' found")

    ensure_ai_defaults(conf)
    ensure_picsum_defaults(conf)
    sanitized = _sanitize_ai_settings(ai_settings, conf[AI_SETTINGS_KEY])
    prompt = str(sanitized.get('prompt') or '').strip()
    if not prompt:
        raise AutoGenerationPromptMissing('Prompt is required')

    conf[AI_SETTINGS_KEY] = sanitized
    conf['_ai_customized'] = not _ai_settings_match_defaults(conf[AI_SETTINGS_KEY])
    persist = bool(sanitized.get('save_output', AI_DEFAULT_PERSIST))

    previous_state = conf.get(AI_STATE_KEY) or {}
    previous_images = list(previous_state.get('images') or [])
    previous_selected = conf.get('selected_image')

    cancel_event = threading.Event()
    socket_sid: Optional[str] = None
    if has_request_context():
        header_sid = request.headers.get('X-Socket-ID')
        if header_sid:
            socket_sid = header_sid.strip() or None
    manager_id = job_manager.create_job(stream_id, trigger=trigger_source, sid=socket_sid)
    with ai_jobs_lock:
        if stream_id in ai_jobs:
            raise AutoGenerationBusy('Generation already in progress')
        ai_jobs[stream_id] = {
            'status': 'queued',
            'job_id': None,
            'started': time.time(),
            'persisted': persist,
            'cancel_requested': False,
            'trigger': trigger_source,
            'manager_id': manager_id,
            'queue_position': None,
            'wait_time': None,
        }
        ai_job_controls[stream_id] = {
            'cancel_event': cancel_event,
            'socket_sid': socket_sid,
            'manager_id': manager_id,
        }
    if not persist:
        _cleanup_temp_outputs(stream_id)

    conf['mode'] = AI_MODE
    conf['media_mode'] = MEDIA_MODE_AI
    if previous_selected:
        conf['selected_image'] = previous_selected

    conf[AI_STATE_KEY] = default_ai_state()
    queued_state = conf[AI_STATE_KEY]
    queued_state.update({
        'status': 'queued',
        'message': 'Awaiting workers',
        'persisted': persist,
        'images': previous_images,
        'error': None,
        'last_trigger_source': trigger_source,
    })
    if trigger_source != 'manual':
        queued_state['last_auto_trigger'] = _format_timer_label(datetime.now())
    queued_state['last_auto_error'] = None

    save_settings(settings)
    _emit_ai_update(stream_id, queued_state, job=ai_jobs[stream_id])
    job_manager.update_status(manager_id, status='queued')
    if job_manager.should_emit(stream_id):
        safe_emit('refresh', {'stream_id': stream_id, 'config': conf, 'tags': get_global_tags()})

    job_options = dict(sanitized)
    job_options['prompt'] = prompt
    worker = threading.Thread(
        target=_run_ai_generation,
        args=(stream_id, job_options, cancel_event, manager_id),
        daemon=True,
    )
    with ai_jobs_lock:
        controls = ai_job_controls.get(stream_id)
        if controls is not None:
            controls['thread'] = worker
            controls['cancel_event'] = cancel_event
    worker.start()
    return {'status': 'queued', 'state': conf[AI_STATE_KEY], 'job': ai_jobs[stream_id]}


class AutoGenerateScheduler:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._next_run: Dict[str, float] = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name='AutoGenerateScheduler', daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def reschedule_all(self) -> None:
        for stream_id in list(settings.keys()):
            if stream_id.startswith('_'):
                continue
            self.reschedule(stream_id)

    def remove(self, stream_id: str) -> None:
        with self._lock:
            self._next_run.pop(stream_id, None)
        self._update_state(stream_id, next_auto_trigger=None, last_auto_error=None)

    def reschedule(self, stream_id: str, *, base_time: Optional[float] = None) -> None:
        conf = settings.get(stream_id)
        if not isinstance(conf, dict):
            self.remove(stream_id)
            return
        ensure_ai_defaults(conf)
        ensure_picsum_defaults(conf)
        ai_settings = conf[AI_SETTINGS_KEY]
        mode_raw = ai_settings.get('auto_generate_mode')
        mode_value = mode_raw.strip().lower() if isinstance(mode_raw, str) else 'off'
        if conf.get('mode') != AI_MODE or mode_value not in AUTO_GENERATE_MODES or mode_value == 'off':
            with self._lock:
                self._next_run.pop(stream_id, None)
            self._update_state(stream_id, next_auto_trigger=None, last_auto_error=None)
            return
        next_dt = self._compute_next_datetime(conf, mode_value, base_time=base_time)
        if next_dt is None:
            with self._lock:
                self._next_run.pop(stream_id, None)
            self._update_state(stream_id, next_auto_trigger=None, last_auto_error=None)
            return
        with self._lock:
            self._next_run[stream_id] = next_dt.timestamp()
        self._update_state(stream_id, next_auto_trigger=_format_timer_label(next_dt))

    def _compute_next_datetime(self, conf: Dict[str, Any], mode_value: str, *, base_time: Optional[float]) -> Optional[datetime]:
        ai_settings = conf[AI_SETTINGS_KEY]
        reference_ts = base_time if base_time is not None else time.time()
        reference_dt = datetime.fromtimestamp(reference_ts)
        if mode_value == 'timer':
            interval_value = ai_settings.get('auto_generate_interval_value')
            try:
                numeric = float(interval_value)
            except (TypeError, ValueError):
                numeric = AUTO_GENERATE_DEFAULT_INTERVAL_VALUE
            if numeric <= 0:
                numeric = AUTO_GENERATE_DEFAULT_INTERVAL_VALUE
            unit_raw = ai_settings.get('auto_generate_interval_unit')
            unit_value = unit_raw.strip().lower() if isinstance(unit_raw, str) else 'minutes'
            multiplier = AUTO_GENERATE_INTERVAL_UNITS.get(unit_value, 60.0)
            interval_seconds = max(AUTO_GENERATE_MIN_INTERVAL_SECONDS, numeric * multiplier)
            return timer_manager.compute_next_trigger(
                interval_seconds,
                reference=reference_dt,
                snap_to_increment=_timer_snap_enabled(),
            )
        if mode_value == 'clock':
            clock_value = _normalize_clock_time(ai_settings.get('auto_generate_clock_time'))
            if not clock_value:
                return None
            hour, minute = map(int, clock_value.split(':'))
            base_dt = reference_dt
            target = base_dt.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if target <= base_dt:
                target += timedelta(days=1)
            return target
        return None

    def _update_state(self, stream_id: str, **updates: Any) -> None:
        conf = settings.get(stream_id)
        if not isinstance(conf, dict):
            return
        ensure_ai_defaults(conf)
        ensure_picsum_defaults(conf)
        state = conf[AI_STATE_KEY]
        changed = False
        for key, value in updates.items():
            if key in {"next_auto_trigger", "last_auto_trigger"}:
                normalized_value = _normalize_timer_label(value)
            else:
                normalized_value = value
            if state.get(key) != normalized_value:
                state[key] = normalized_value
                changed = True
        if changed:
            _emit_ai_update(stream_id, state)

    def _loop(self) -> None:
        while not self._stop.is_set():
            now = time.time()
            due: List[str] = []
            with self._lock:
                for stream_id, next_ts in list(self._next_run.items()):
                    if next_ts <= now:
                        due.append(stream_id)
            for stream_id in due:
                self._trigger_stream(stream_id)
            self._stop.wait(5.0)

    def _trigger_stream(self, stream_id: str) -> None:
        conf = settings.get(stream_id)
        if not isinstance(conf, dict):
            self.remove(stream_id)
            return
        ensure_ai_defaults(conf)
        ensure_picsum_defaults(conf)
        ai_settings = conf[AI_SETTINGS_KEY]
        mode_raw = ai_settings.get('auto_generate_mode')
        mode_value = mode_raw.strip().lower() if isinstance(mode_raw, str) else 'off'
        if conf.get('mode') != AI_MODE or mode_value not in AUTO_GENERATE_MODES or mode_value == 'off':
            self.reschedule(stream_id)
            return
        prompt = str(ai_settings.get('prompt') or '').strip()
        if not prompt:
            self._update_state(stream_id, last_auto_error='Prompt is required for auto-generation')
            self.reschedule(stream_id)
            return
        try:
            _queue_ai_generation(stream_id, ai_settings, trigger_source='auto')
        except AutoGenerationBusy:
            self._update_state(stream_id, last_auto_error=None)
            self.reschedule(stream_id, base_time=time.time())
        except AutoGenerationPromptMissing as exc:
            self._update_state(stream_id, last_auto_error=str(exc))
            self.reschedule(stream_id)
        except AutoGenerationUnavailable as exc:
            self._update_state(stream_id, last_auto_error=str(exc))
            self.reschedule(stream_id)
        except AutoGenerationError as exc:
            self._update_state(stream_id, last_auto_error=str(exc))
            self.reschedule(stream_id)
        else:
            self._update_state(stream_id, last_auto_error=None)
            self.reschedule(stream_id, base_time=time.time())


if auto_scheduler is None:
    auto_scheduler = AutoGenerateScheduler()
    auto_scheduler.reschedule_all()
    atexit.register(auto_scheduler.stop)

# Picsum auto scheduler ------------------------------------------------------


class PicsumAutoScheduler:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._next_run: Dict[str, float] = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._loop,
            name="PicsumAutoScheduler",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def reschedule_all(self) -> None:
        for stream_id, conf in settings.items():
            if stream_id.startswith("_") or not isinstance(conf, dict):
                continue
            self.reschedule(stream_id)

    def remove(self, stream_id: str) -> None:
        with self._lock:
            self._next_run.pop(stream_id, None)
        conf = settings.get(stream_id)
        if isinstance(conf, dict):
            picsum = conf.get(PICSUM_SETTINGS_KEY)
            if isinstance(picsum, dict):
                picsum["next_auto_trigger"] = None

    def reschedule(self, stream_id: str, *, base_time: Optional[float] = None) -> None:
        conf = settings.get(stream_id)
        if not isinstance(conf, dict):
            self.remove(stream_id)
            return
        ensure_picsum_defaults(conf)
        picsum = conf.get(PICSUM_SETTINGS_KEY) or {}
        mode_raw = picsum.get("auto_mode")
        mode = str(mode_raw).strip().lower() if isinstance(mode_raw, str) else PICSUM_DEFAULT_AUTO_MODE
        if conf.get("media_mode") != MEDIA_MODE_PICSUM or mode not in PICSUM_AUTO_MODES or mode == "off":
            self.remove(stream_id)
            return
        next_dt = self._compute_next_datetime(conf, mode, base_time=base_time)
        if next_dt is None:
            self.remove(stream_id)
            return
        with self._lock:
            self._next_run[stream_id] = next_dt.timestamp()
        picsum["next_auto_trigger"] = _format_timer_label(next_dt)

    def _compute_next_datetime(
        self,
        conf: Dict[str, Any],
        mode: str,
        *,
        base_time: Optional[float],
    ) -> Optional[datetime]:
        picsum = conf.get(PICSUM_SETTINGS_KEY) or {}
        reference_ts = base_time if base_time is not None else time.time()
        reference_dt = datetime.fromtimestamp(reference_ts)
        if mode == "timer":
            interval_value = _coerce_float(picsum.get("auto_interval_value"), PICSUM_DEFAULT_INTERVAL_VALUE)
            if interval_value is None or interval_value <= 0:
                interval_value = PICSUM_DEFAULT_INTERVAL_VALUE
            unit_raw = picsum.get("auto_interval_unit")
            unit = str(unit_raw).strip().lower() if isinstance(unit_raw, str) else PICSUM_DEFAULT_INTERVAL_UNIT
            unit_seconds = AUTO_GENERATE_INTERVAL_UNITS.get(unit, AUTO_GENERATE_INTERVAL_UNITS[PICSUM_DEFAULT_INTERVAL_UNIT])
            seconds = max(60.0, interval_value * unit_seconds)
            return timer_manager.compute_next_trigger(
                seconds,
                reference=reference_dt,
                snap_to_increment=_timer_snap_enabled(),
            )
        if mode == "clock":
            clock_value = _normalize_clock_time(picsum.get("auto_clock_time"))
            if not clock_value:
                return None
            hour, minute = map(int, clock_value.split(":"))
            base_dt = reference_dt
            target = base_dt.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if target <= base_dt:
                target += timedelta(days=1)
            return target
        return None

    def _loop(self) -> None:
        while not self._stop.is_set():
            now = time.time()
            due: List[str] = []
            with self._lock:
                for stream_id, next_ts in list(self._next_run.items()):
                    if next_ts <= now:
                        due.append(stream_id)
            for stream_id in due:
                self._trigger_stream(stream_id)
            self._stop.wait(5.0)

    def _trigger_stream(self, stream_id: str) -> None:
        conf = settings.get(stream_id)
        if not isinstance(conf, dict):
            self.remove(stream_id)
            return
        ensure_picsum_defaults(conf)
        result = _refresh_picsum_stream(stream_id)
        if result is None:
            self.remove(stream_id)
            return
        picsum_conf = conf.get(PICSUM_SETTINGS_KEY) or {}
        picsum_conf["last_auto_trigger"] = _format_timer_label(datetime.now())
        self.reschedule(stream_id, base_time=time.time())
        save_settings(settings)
        _broadcast_picsum_update(
            stream_id,
            conf,
            result["url"],
            result["settings"].get("seed"),
            bool(result.get("seed_custom")),
            result.get("thumbnail"),
        )


if picsum_scheduler is None:
    picsum_scheduler = PicsumAutoScheduler()
    picsum_scheduler.reschedule_all()
    atexit.register(picsum_scheduler.stop)


@app.route('/ai/models')
def list_ai_models():
    if stable_horde_client is None:
        return jsonify({'error': 'Stable Horde client is not configured'}), 503
    now = time.time()
    cache_ttl = 300
    if (now - ai_model_cache['timestamp']) > cache_ttl or not ai_model_cache['data']:
        try:
            models = stable_horde_client.list_models()
        except StableHordeError as exc:
            logger.warning('Model fetch failed: %s', exc)
            return jsonify({'error': str(exc)}), 502
        ai_model_cache['data'] = [
            {
                'name': m.get('name'),
                'performance': m.get('performance'),
                'queued': m.get('queued'),
                'jobs': m.get('jobs'),
                'type': m.get('type'),
            }
            for m in models
            if isinstance(m, dict) and m.get('type') == 'image'
        ]
        ai_model_cache['timestamp'] = now
    return jsonify({'models': ai_model_cache['data']})


@app.route('/ai/status/<stream_id>')
def ai_status(stream_id: str):
    conf = settings.get(stream_id)
    if not conf:
        return jsonify({'error': f"No stream '{stream_id}' found"}), 404
    ensure_ai_defaults(conf)
    with ai_jobs_lock:
        job_snapshot = dict(ai_jobs.get(stream_id, {}))
    return jsonify({
        'state': conf[AI_STATE_KEY],
        'settings': conf[AI_SETTINGS_KEY],
        'job': job_snapshot,
    })


@app.route('/api/jobs/<stream_id>/latest')
def latest_job(stream_id: str):
    conf = settings.get(stream_id)
    state_payload: Optional[Dict[str, Any]] = None
    if conf:
        ensure_ai_defaults(conf)
        ensure_picsum_defaults(conf)
        state_payload = conf.get(AI_STATE_KEY)
    managed = job_manager.get_latest(stream_id)
    job_payload = managed.to_dict() if managed else None
    active_job: Optional[Dict[str, Any]] = None
    with ai_jobs_lock:
        current = ai_jobs.get(stream_id)
        if current:
            active_job = dict(current)
    return jsonify({
        'job': job_payload,
        'active_job': active_job,
        'state': state_payload,
    })


@app.route('/ai/generate/<stream_id>', methods=['POST'])
def ai_generate(stream_id: str):
    conf = settings.get(stream_id)
    if not conf:
        return jsonify({'error': f"No stream '{stream_id}' found"}), 404
    ensure_ai_defaults(conf)
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        payload = {}
    try:
        result = _queue_ai_generation(stream_id, payload, trigger_source='manual')
    except AutoGenerationPromptMissing as exc:
        return jsonify({'error': str(exc)}), 400
    except AutoGenerationBusy as exc:
        return jsonify({'error': str(exc)}), 409
    except AutoGenerationUnavailable as exc:
        return jsonify({'error': str(exc)}), 503
    except AutoGenerationError as exc:
        return jsonify({'error': str(exc)}), 400
    if auto_scheduler is not None:
        auto_scheduler.reschedule(stream_id, base_time=time.time())
    return jsonify(result)


@app.route('/ai/cancel/<stream_id>', methods=['POST'])
def ai_cancel(stream_id: str):
    if stable_horde_client is None:
        return jsonify({'error': 'Stable Horde client is not configured'}), 503
    conf = settings.get(stream_id)
    if not conf:
        return jsonify({'error': f"No stream '{stream_id}' found"}), 404
    ensure_ai_defaults(conf)
    manager_id: Optional[str] = None
    with ai_jobs_lock:
        job = ai_jobs.get(stream_id)
        controls = ai_job_controls.get(stream_id)
        if not job:
            return jsonify({'error': 'No active AI generation to cancel'}), 404
        status = (job.get('status') or '').lower()
        if status in {'completed', 'error', 'timeout', 'cancelled'}:
            return jsonify({'error': 'Job already finished'}), 409
        job = dict(job)
        job['cancel_requested'] = True
        job['status'] = 'cancelling'
        job['message'] = 'Cancellation requested'
        ai_jobs[stream_id] = job
        cancel_event = controls.get('cancel_event') if controls else None
        if controls:
            manager_id = controls.get('manager_id')
    if cancel_event:
        cancel_event.set()
    state = _update_ai_state(
        stream_id,
        {
            'status': 'cancelling',
            'message': 'Cancellation requested',
            'error': None,
            'persisted': bool(job.get('persisted')),
        },
        persist=True,
    )
    _emit_ai_update(stream_id, state, job)
    if manager_id:
        job_manager.update_status(manager_id, status='cancelling')
    warning = None
    job_id = job.get('job_id')
    if job_id:
        try:
            stable_horde_client.cancel_job(job_id)
        except StableHordeError as exc:
            logger.warning('Stable Horde cancel for %s failed: %s', stream_id, exc)
            warning = str(exc)
    response = {'status': 'cancelling'}
    if warning:
        response['warning'] = warning
    return jsonify(response)

@app.route("/settings")
def app_settings():
    cfg = load_config()
    return render_template(
        "settings.html",
        config=cfg,
        ai_defaults=default_ai_settings(),
        ai_fallback_defaults=AI_FALLBACK_DEFAULTS,
        post_processors=STABLE_HORDE_POST_PROCESSORS,
        max_loras=STABLE_HORDE_MAX_LORAS,
    )



def _repo_path_from_config(cfg: Optional[Dict[str, Any]] = None) -> str:
    cfg = cfg or load_config()
    return cfg.get("INSTALL_DIR") or os.getcwd()


def _restore_points_root(repo_path: str) -> str:
    return os.path.join(repo_path, BACKUP_DIRNAME, RESTORE_POINT_DIRNAME)


def _slugify_restore_label(label: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "-", label.strip())
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip("-_")
    if not cleaned:
        cleaned = f"restore-{secrets.token_hex(3)}"
    return cleaned.lower()[:60]


def _load_restore_point_metadata(point_path: str) -> Optional[Dict[str, Any]]:
    metadata_path = os.path.join(point_path, RESTORE_POINT_METADATA_FILE)
    try:
        with open(metadata_path, "r") as mp:
            return json.load(mp)
    except Exception:
        return None


def _serialize_restore_point(point_id: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
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


def _list_restore_points(repo_path: str) -> List[Dict[str, Any]]:
    root = _restore_points_root(repo_path)
    items: List[Dict[str, Any]] = []
    if not os.path.isdir(root):
        return items
    for entry in os.scandir(root):
        if not entry.is_dir():
            continue
        metadata = _load_restore_point_metadata(entry.path)
        if not metadata:
            continue
        items.append(_serialize_restore_point(entry.name, metadata))
    items.sort(key=lambda item: item.get("created_at") or "", reverse=True)
    return items


def _prune_restore_points(root: str) -> None:
    if MAX_RESTORE_POINTS <= 0 or not os.path.isdir(root):
        return
    entries: List[Tuple[str, str]] = []
    for item in os.scandir(root):
        if not item.is_dir():
            continue
        metadata = _load_restore_point_metadata(item.path)
        if not metadata:
            continue
        created = metadata.get("created_at") or ""
        entries.append((item.path, created))
    if len(entries) <= MAX_RESTORE_POINTS:
        return
    entries.sort(key=lambda pair: pair[1] or "")
    for path_to_remove, _ in entries[:-MAX_RESTORE_POINTS]:
        try:
            shutil.rmtree(path_to_remove)
        except Exception as exc:
            logger.warning("Failed to prune restore point %s: %s", path_to_remove, exc)


def _allocate_restore_point_dir(repo_path: str, label: str) -> Tuple[str, str]:
    root = _restore_points_root(repo_path)
    os.makedirs(root, exist_ok=True)
    slug = _slugify_restore_label(label)
    now = datetime.utcnow()
    dir_stamp = now.strftime("%Y%m%d-%H%M%S")
    candidate = f"{dir_stamp}-{slug}"
    while os.path.exists(os.path.join(root, candidate)):
        candidate = f"{dir_stamp}-{slug}-{secrets.token_hex(2)}"
    point_path = os.path.join(root, candidate)
    os.makedirs(point_path, exist_ok=True)
    return candidate, point_path


def _save_restore_point_metadata(point_path: str, metadata: Dict[str, Any]) -> None:
    metadata_path = os.path.join(point_path, RESTORE_POINT_METADATA_FILE)
    with open(metadata_path, "w") as mp:
        json.dump(metadata, mp, indent=2)


def _create_restore_point(repo_path: str, label: str) -> Dict[str, Any]:
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
    point_id, point_path = _allocate_restore_point_dir(repo_path, label)
    files: List[Dict[str, str]] = []
    for source_name in (SETTINGS_FILE, CONFIG_FILE):
        source_path = os.path.join(repo_path, source_name)
        if not os.path.isfile(source_path):
            continue
        dest_name = os.path.basename(source_name)
        dest_path = os.path.join(point_path, dest_name)
        try:
            shutil.copy2(source_path, dest_path)
            files.append({"filename": dest_name, "destination": source_name})
        except OSError as exc:
            logger.warning(
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
    _save_restore_point_metadata(point_path, metadata)
    _prune_restore_points(_restore_points_root(repo_path))
    return _serialize_restore_point(point_id, metadata)


def _load_restore_point(repo_path: str, point_id: str) -> Tuple[str, Dict[str, Any]]:
    safe_id = os.path.basename(point_id.strip())
    if not safe_id:
        raise FileNotFoundError(point_id)
    point_path = os.path.join(_restore_points_root(repo_path), safe_id)
    if not os.path.isdir(point_path):
        raise FileNotFoundError(point_id)
    metadata = _load_restore_point_metadata(point_path)
    if not metadata:
        raise FileNotFoundError(point_id)
    metadata["id"] = safe_id
    return point_path, metadata


def _restore_files_from_metadata(repo_path: str, point_path: str, metadata: Dict[str, Any]) -> None:
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
            logger.warning(
                "Failed to restore %s from restore point %s: %s",
                destination,
                metadata.get("id"),
                exc,
            )


def _delete_restore_point(repo_path: str, point_id: str) -> bool:
    safe_id = os.path.basename(point_id.strip())
    if not safe_id:
        return False
    point_path = os.path.join(_restore_points_root(repo_path), safe_id)
    if not os.path.isdir(point_path):
        return False
    shutil.rmtree(point_path)
    return True


def _apply_timer_snap_setting(enabled: bool) -> Dict[str, Dict[str, Optional[str]]]:
    CONFIG["TIMER_SNAP_ENABLED"] = enabled
    if auto_scheduler is not None:
        auto_scheduler.reschedule_all()
    if picsum_scheduler is not None:
        picsum_scheduler.reschedule_all()

    summary: Dict[str, Dict[str, Optional[str]]] = {}
    for stream_id, conf in settings.items():
        if not isinstance(conf, dict):
            continue
        ensure_ai_defaults(conf)
        ensure_picsum_defaults(conf)
        ai_state = conf.get(AI_STATE_KEY)
        picsum_conf = conf.get(PICSUM_SETTINGS_KEY)
        summary[stream_id] = {
            "ai_next": ai_state.get("next_auto_trigger") if isinstance(ai_state, dict) else None,
            "picsum_next": picsum_conf.get("next_auto_trigger") if isinstance(picsum_conf, dict) else None,
        }

    save_settings(settings)
    return summary


@app.route("/api/settings/timers", methods=["GET", "POST"])
def api_timer_settings():
    cfg = load_config()
    snap_enabled = _as_bool(cfg.get("TIMER_SNAP_ENABLED"), False)
    if request.method == "GET":
        return jsonify({"timer_snap_enabled": snap_enabled})

    api_key = cfg.get("API_KEY")
    if api_key and request.headers.get("X-API-Key") != api_key:
        return jsonify({"error": "Unauthorized"}), 401

    payload = request.get_json(silent=True) or {}
    requested = _as_bool(payload.get("timer_snap_enabled"), False)
    cfg["TIMER_SNAP_ENABLED"] = requested
    try:
        config_manager.save_config(cfg)
    except Exception:
        logger.exception("Failed to persist timer snap setting")
        return jsonify({"error": "Unable to save setting"}), 500

    stream_summary = _apply_timer_snap_setting(requested)
    return jsonify({"timer_snap_enabled": requested, "streams": stream_summary})


@app.route("/restore_points", methods=["GET", "POST"])
def restore_points_collection():
    cfg = load_config()
    api_key = cfg.get("API_KEY")
    if api_key and request.headers.get("X-API-Key") != api_key:
        return jsonify({"error": "Unauthorized"}), 401
    repo_path = _repo_path_from_config(cfg)
    if request.method == "GET":
        return jsonify({"restore_points": _list_restore_points(repo_path)})
    data = request.get_json(silent=True) or {}
    label = (data.get("label") or "").strip()
    if not label:
        return jsonify({"error": "Label required"}), 400
    try:
        point = _create_restore_point(repo_path, label)
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 500
    except Exception:
        logger.exception("Failed to create restore point")
        return jsonify({"error": "Failed to create restore point"}), 500
    return jsonify({"restore_point": point}), 201


@app.route("/restore_points/<point_id>", methods=["DELETE"])
def restore_points_delete(point_id):
    cfg = load_config()
    api_key = cfg.get("API_KEY")
    if api_key and request.headers.get("X-API-Key") != api_key:
        return jsonify({"error": "Unauthorized"}), 401
    repo_path = _repo_path_from_config(cfg)
    try:
        deleted = _delete_restore_point(repo_path, point_id)
    except Exception:
        logger.exception("Failed to delete restore point %s", point_id)
        return jsonify({"error": "Failed to delete restore point"}), 500
    if not deleted:
        return jsonify({"error": "Restore point not found"}), 404
    return jsonify({"status": "deleted"})


@app.route("/restore_points/<point_id>/restore", methods=["POST"])
def restore_points_restore(point_id):
    cfg = load_config()
    api_key = cfg.get("API_KEY")
    if api_key and request.headers.get("X-API-Key") != api_key:
        return jsonify({"error": "Unauthorized"}), 401
    repo_path = _repo_path_from_config(cfg)
    service_name = cfg.get("SERVICE_NAME", "echomosaic.service")
    try:
        point_path, metadata = _load_restore_point(repo_path, point_id)
    except FileNotFoundError:
        return jsonify({"error": "Restore point not found"}), 404
    commit = metadata.get("commit")
    if not commit:
        return jsonify({"error": "Restore point missing commit"}), 400
    try:
        subprocess.check_call(["git", "reset", "--hard", commit], cwd=repo_path)
    except subprocess.CalledProcessError as exc:
        return jsonify({"error": f"Git rollback failed: {exc}"}), 500
    _restore_files_from_metadata(repo_path, point_path, metadata)
    metadata["last_restored_at"] = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    try:
        _save_restore_point_metadata(point_path, metadata)
    except Exception:
        pass
    try:
        subprocess.Popen(["sudo", "systemctl", "restart", service_name])
    except OSError:
        pass
    return jsonify({"status": "ok", "restore_point": _serialize_restore_point(metadata["id"], metadata)})


@app.route("/update_app", methods=["POST"])
def update_app():
    cfg = load_config()
    api_key = cfg.get("API_KEY")
    if api_key and request.headers.get("X-API-Key") != api_key:
        return "Unauthorized", 401
    repo_path = cfg.get("INSTALL_DIR") or os.getcwd()
    branch = cfg.get("UPDATE_BRANCH", "main")
    service_name = cfg.get("SERVICE_NAME", "echomosaic.service")
    if not os.path.isdir(repo_path):
        return render_template(
            "update_status.html",
            message=f"Repository path '{repo_path}' not found. Check INSTALL_DIR in config.json",
        )
    # capture current commit before update
    def git_cmd(args, cwd=repo_path):
        return subprocess.check_output(["git", *args], cwd=cwd, stderr=subprocess.STDOUT).decode().strip()
    try:
        current_commit = git_cmd(["rev-parse", "HEAD"])
    except Exception:
        current_commit = None
    try:
        backup_dir = backup_user_state(repo_path)
    except Exception:
        logger.exception("Failed to back up user data before update")
        return render_template(
            "update_status.html",
            message="Unable to back up user data; update aborted.",
        )
    restore_error = None
    try:
        subprocess.check_call(["git", "fetch"], cwd=repo_path)
        subprocess.check_call(["git", "checkout", branch], cwd=repo_path)
        subprocess.check_call(["git", "reset", "--hard", f"origin/{branch}"], cwd=repo_path)
    except FileNotFoundError:
        update_error = "Git executable not found. Please install Git to update the application."
        return render_template("update_status.html", message=update_error)
    except subprocess.CalledProcessError as e:
        update_error = f"Git update failed: {e}"
        return render_template("update_status.html", message=update_error)
    finally:
        try:
            restore_user_state(repo_path, backup_dir, cleanup=True)
        except Exception:
            restore_error = "Failed to restore user data after update."
            logger.exception("Failed to restore user data after update")
    if restore_error:
        return render_template("update_status.html", message=restore_error)
    # record update history
    try:
        new_commit = git_cmd(["rev-parse", "HEAD"]) if 'git_cmd' in locals() else None
        history_path = os.path.join(repo_path, "update_history.json")
        history = []
        if os.path.exists(history_path):
            with open(history_path, "r") as hf:
                history = json.load(hf)
        history.append({
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "from": current_commit,
            "to": new_commit,
            "branch": branch,
        })
        with open(history_path, "w") as hf:
            json.dump(history[-50:], hf, indent=2)
    except Exception:
        pass
    try:
        subprocess.check_call([
            os.path.join(repo_path, "venv", "bin", "pip"),
            "install",
            "--upgrade",
            "-r",
            "requirements.txt",
        ], cwd=repo_path)
    except (subprocess.CalledProcessError, OSError):
        pass
    try:
        subprocess.Popen(["sudo", "systemctl", "restart", service_name])
    except OSError:
        pass
    return render_template(
        "update_status.html", message="Soft update complete. Restarting service..."
    )


def read_update_info():
    cfg = load_config()
    repo_path = cfg.get("INSTALL_DIR") or os.getcwd()
    branch = cfg.get("UPDATE_BRANCH", "main")
    info = {"branch": branch}
    def safe(cmd):
        try:
            return subprocess.check_output(cmd, cwd=repo_path, stderr=subprocess.STDOUT).decode().strip()
        except Exception:
            return None
    current = safe(["git", "rev-parse", "HEAD"]) or ""
    current_short = safe(["git", "rev-parse", "--short", "HEAD"]) or ""
    current_desc = safe(["git", "log", "-1", "--pretty=%h %s (%cr)"]) or current_short
    # fetch remote to learn about latest without changing local state
    _ = safe(["git", "fetch", "--quiet"])  # ignore errors silently
    remote = safe(["git", "rev-parse", f"origin/{branch}"]) or ""
    remote_short = remote[:7] if remote else ""
    remote_desc = safe(["git", "log", "-1", f"origin/{branch}", "--pretty=%h %s (%cr)"]) or remote_short
    info.update({
        "current_commit": current,
        "current_desc": current_desc,
        "remote_commit": remote,
        "remote_desc": remote_desc,
        "update_available": (current and remote and current != remote)
    })
    # previous commit from history
    history_path = os.path.join(repo_path, "update_history.json")
    if os.path.exists(history_path):
        try:
            with open(history_path, "r") as hf:
                history = json.load(hf)
            if history:
                last = history[-1]
                info["previous_commit"] = last.get("from")
                # resolve desc for previous
                prev = last.get("from")
                if prev:
                    prev_desc = safe(["git", "log", "-1", prev, "--pretty=%h %s (%cr)"]) or (prev[:7])
                else:
                    prev_desc = None
                info["previous_desc"] = prev_desc
        except Exception:
            pass
    return info


@app.route("/update_info", methods=["GET"])
def update_info():
    cfg = load_config()
    api_key = cfg.get("API_KEY")
    if api_key and request.headers.get("X-API-Key") != api_key:
        return jsonify({"error": "Unauthorized"}), 401
    return jsonify(read_update_info())


@app.route("/update_history", methods=["GET"])
def update_history():
    cfg = load_config()
    api_key = cfg.get("API_KEY")
    if api_key and request.headers.get("X-API-Key") != api_key:
        return jsonify({"error": "Unauthorized"}), 401
    repo_path = cfg.get("INSTALL_DIR") or os.getcwd()
    history_path = os.path.join(repo_path, "update_history.json")
    history = []
    if os.path.exists(history_path):
        try:
            with open(history_path, "r") as hf:
                history = json.load(hf)
        except Exception:
            history = []
    def srun(cmd):
        try:
            return subprocess.check_output(cmd, cwd=repo_path, stderr=subprocess.STDOUT).decode().strip()
        except Exception:
            return None
    enriched = []
    for ent in history:
        frm = ent.get("from")
        to = ent.get("to")
        frm_desc = srun(["git", "log", "-1", frm, "--pretty=%h %s (%cr)"]) if frm else None
        to_desc = srun(["git", "log", "-1", to, "--pretty=%h %s (%cr)"]) if to else None
        enriched.append({
            "timestamp": ent.get("timestamp"),
            "branch": ent.get("branch"),
            "from": frm,
            "to": to,
            "from_desc": frm_desc or (frm[:7] if frm else None),
            "to_desc": to_desc or (to[:7] if to else None),
        })
    return jsonify({"history": enriched})


@app.route("/rollback_app", methods=["POST"])
def rollback_app():
    cfg = load_config()
    api_key = cfg.get("API_KEY")
    if api_key and request.headers.get("X-API-Key") != api_key:
        return "Unauthorized", 401
    repo_path = _repo_path_from_config(cfg)
    service_name = cfg.get("SERVICE_NAME", "echomosaic.service")
    payload = request.get_json(silent=True)
    restore_point_id = ""
    if isinstance(payload, dict):
        restore_point_id = str(payload.get("restore_point_id") or payload.get("restore_point") or "").strip()
    if not restore_point_id:
        form_value = request.form.get("restore_point_id") if hasattr(request, "form") else None
        restore_point_id = str(
            form_value
            or request.args.get("restore_point_id")
            or request.args.get("restore_point")
            or ""
        ).strip()
    if restore_point_id:
        try:
            point_path, metadata = _load_restore_point(repo_path, restore_point_id)
        except FileNotFoundError:
            return render_template("update_status.html", message="Restore point not found."), 404
        commit = metadata.get("commit")
        if not commit:
            return render_template(
                "update_status.html",
                message="Restore point is missing commit information.",
            ), 400
        try:
            subprocess.check_call(["git", "reset", "--hard", commit], cwd=repo_path)
        except subprocess.CalledProcessError as exc:
            return render_template("update_status.html", message=f"Rollback failed: {exc}"), 500
        _restore_files_from_metadata(repo_path, point_path, metadata)
        metadata["last_restored_at"] = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        try:
            _save_restore_point_metadata(point_path, metadata)
        except Exception:
            pass
        try:
            subprocess.Popen(["sudo", "systemctl", "restart", service_name])
        except OSError:
            pass
        label = metadata.get("label") or restore_point_id
        short_commit = metadata.get("short_commit") or (
            commit[:7] if isinstance(commit, str) else commit
        )
        message = (
            f"Rolled back to restore point '{label}' ({short_commit}). Restarting service..."
        )
        return render_template("update_status.html", message=message)
    history_path = os.path.join(repo_path, "update_history.json")
    try:
        with open(history_path, "r") as hf:
            history = json.load(hf)
    except Exception:
        return render_template("update_status.html", message="No previous version to roll back to."), 400
    if not history:
        return render_template("update_status.html", message="No previous version to roll back to."), 400
    target = history[-1].get("from")
    if not target:
        return render_template("update_status.html", message="History does not include a valid commit."), 400
    try:
        subprocess.check_call(["git", "reset", "--hard", target], cwd=repo_path)
    except subprocess.CalledProcessError as exc:
        return render_template("update_status.html", message=f"Rollback failed: {exc}")
    try:
        subprocess.Popen(["sudo", "systemctl", "restart", service_name])
    except OSError:
        pass
    return render_template(
        "update_status.html",
        message=f"Rolled back to {target[:7]}. Restarting service...",
    )


@app.route("/update")
def update_view():
    # A simple progress UI that will kick off the update via fetch
    cfg = load_config()
    return render_template("update_progress.html", api_key=cfg.get("API_KEY", ""))


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/api/system_stats", methods=["GET"])
def system_stats():
    stats = get_system_stats()
    return jsonify(stats)


# --- Stream groups and metadata ---
@app.route("/streams_meta", methods=["GET"])
def streams_meta():
    meta = {}
    for k, v in settings.items():
        if k.startswith("_"):
            continue
        meta[k] = {
            "label": v.get("label", k),
            "include_in_global": v.get("include_in_global", True),
        }
    return jsonify(meta)


@app.route("/groups", methods=["GET", "POST"])
def groups_collection():
    if request.method == "GET":
        return jsonify(settings.get("_groups", {}))
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    streams = data.get("streams") or []
    layout = data.get("layout") or None
    if not name:
        return jsonify({"error": "Name required"}), 400
    # Prevent reserved name and duplicates (case-insensitive)
    if name.lower() == "default":
        return jsonify({"error": "'default' is a reserved group name"}), 400
    settings.setdefault("_groups", {})
    for existing in list(settings["_groups"].keys()):
        if existing.lower() == name.lower() and existing != name:
            return jsonify({"error": "Group name already exists (case-insensitive)"}), 409
    cleaned = [s for s in streams if s in settings]
    # Store as object with streams + optional layout
    if layout and isinstance(layout, dict):
        settings["_groups"][name] = {"streams": cleaned, "layout": layout}
    else:
        settings["_groups"][name] = cleaned
    save_settings(settings)
    safe_emit("mosaic_refresh", {"group": name})
    return jsonify({"status": "ok", "group": {name: settings["_groups"][name]}})


@app.route("/groups/<name>", methods=["DELETE"])
def groups_delete(name):
    if "_groups" in settings and name in settings["_groups"]:
        del settings["_groups"][name]
        save_settings(settings)
        return jsonify({"status": "deleted"})
    return jsonify({"error": "not found"}), 404


@app.route("/stream/live")
def stream_live():
    stream_id = (request.args.get("stream_id") or "").strip()
    if not stream_id:
        return jsonify({"error": "stream_id required"}), 400

    stream_conf = settings.get(stream_id)
    if not stream_conf:
        return jsonify({"error": f"No stream '{stream_id}' found"}), 404

    override_url = request.args.get("url")
    stream_url_raw = override_url if override_url is not None else stream_conf.get("stream_url", "")
    stream_url = (stream_url_raw or "").strip()
    if not stream_url:
        return jsonify({"error": "No live stream URL configured"}), 404

    lowered = stream_url.lower()
    youtube_details = _parse_youtube_url_details(stream_url)
    if youtube_details:
        if override_url is not None:
            metadata = _youtube_oembed_lookup(stream_url, youtube_details, force=True)
            sanitized_meta = _sanitize_embed_metadata(metadata or {})
        else:
            force_refresh = stream_conf.get("stream_url") != stream_url
            sanitized_meta = _refresh_embed_metadata(stream_id, stream_conf, force=force_refresh)
            if sanitized_meta is None:
                metadata = _youtube_oembed_lookup(stream_url, youtube_details, force=force_refresh)
                sanitized_meta = _sanitize_embed_metadata(metadata or {})
                if sanitized_meta:
                    stream_conf["embed_metadata"] = sanitized_meta
                    _set_runtime_embed_metadata(stream_id, sanitized_meta)
        content_type = (sanitized_meta or {}).get("content_type")
        if not content_type:
            if youtube_details.get("playlist_id"):
                content_type = "playlist"
            elif youtube_details.get("is_live"):
                content_type = "live"
            else:
                content_type = "video"
        response_payload: Dict[str, Any] = {
            "embed_type": "youtube",
            "embed_id": youtube_details.get("video_id"),
            "video_id": youtube_details.get("video_id"),
            "playlist_id": youtube_details.get("playlist_id"),
            "content_type": content_type,
            "start_index": youtube_details.get("start_index"),
            "start_seconds": youtube_details.get("start_seconds"),
            "is_live": bool((sanitized_meta or {}).get("is_live") or youtube_details.get("is_live")),
            "embed_base": youtube_details.get("embed_base"),
            "hls_url": None,
            "original_url": stream_url,
        }
        if sanitized_meta:
            response_payload["metadata"] = sanitized_meta
        return jsonify(response_payload)

    if "twitch.tv" in lowered:
        try:
            embed_id = stream_url.split("twitch.tv/")[1].split("/")[0]
        except Exception:
            embed_id = ""
        return jsonify({
            "embed_type": "twitch",
            "embed_id": embed_id,
            "hls_url": None,
            "original_url": stream_url,
        })

    if _is_manifest_url(stream_url):
        _log_hls_event("direct_manifest", stream_id, stream_url)
        return jsonify({
            "embed_type": "hls",
            "embed_id": None,
            "hls_url": stream_url,
            "original_url": stream_url,
        })

    hls_url = None
    if LIVE_HLS_ASYNC and HLS_EXECUTOR is not None:
        now = time.time()
        cache_key = _live_hls_cache_key(stream_id, stream_url)
        entry = _get_hls_cache_entry(cache_key)
        schedule_needed = False

        if entry is None:
            _record_hls_metric("misses")
            _log_hls_event("cache_miss", stream_id, stream_url)
            schedule_needed = True
        else:
            age = now - entry.extracted_at
            age_ms = int(age * 1000)
            if entry.url and age < HLS_TTL_SECS:
                hls_url = entry.url
                _record_hls_metric("hits")
                _log_hls_event("cache_hit", stream_id, stream_url, age_ms=age_ms)
            else:
                _record_hls_metric("stale")
                _log_hls_event(
                    "cache_stale",
                    stream_id,
                    stream_url,
                    age_ms=age_ms,
                    had_url=bool(entry.url),
                    error=entry.error or "none",
                )
                if entry.url:
                    schedule_needed = True
                else:
                    schedule_needed = age >= HLS_ERROR_RETRY_SECS
                    if not schedule_needed:
                        wait_ms = max(0, int((HLS_ERROR_RETRY_SECS - age) * 1000))
                        _log_hls_event(
                            "retry_pending",
                            stream_id,
                            stream_url,
                            retry_in_ms=wait_ms,
                            error=entry.error or "none",
                        )
        if schedule_needed:
            schedule_hls_detection(stream_id, stream_url)
    else:
        try:
            hls_url = _detect_hls_stream_url(stream_url)
        except Exception as exc:
            logger.debug("Synchronous HLS detection failed for %s: %s", stream_id, exc)

    if hls_url:
        return jsonify({
            "embed_type": "hls",
            "embed_id": None,
            "hls_url": hls_url,
            "original_url": stream_url,
        })

    return jsonify({
        "embed_type": "iframe",
        "embed_id": None,
        "hls_url": None,
        "original_url": stream_url,
    })


@app.route("/stream/live/invalidate", methods=["POST"])
def stream_live_invalidate():
    payload = request.get_json(silent=True) or {}
    stream_id = str(payload.get("stream_id") or "").strip()
    if not stream_id:
        return jsonify({"error": "stream_id required"}), 400

    stream_conf = settings.get(stream_id)
    if not stream_conf:
        return jsonify({"error": f"No stream '{stream_id}' found"}), 404

    requested_url = payload.get("url")
    target_url_raw = requested_url if requested_url is not None else stream_conf.get("stream_url", "")
    target_url = (target_url_raw or "").strip()
    if not target_url:
        return jsonify({"error": "No live stream URL provided"}), 400

    prefix = f"live:{stream_id}:"
    with HLS_LOCK:
        cache_keys = [key for key in list(HLS_CACHE.keys()) if key.startswith(prefix)]
        job_keys = [key for key in list(HLS_JOBS.keys()) if key.startswith(prefix)]
        removed = 0
        for key in cache_keys:
            if HLS_CACHE.pop(key, None) is not None:
                removed += 1
    cancelled = 0
    for key in job_keys:
        if _cancel_hls_job(key):
            cancelled += 1

    _log_hls_event("invalidate", stream_id, target_url, removed=removed, cancelled=cancelled)

    rescheduled = False
    if LIVE_HLS_ASYNC and HLS_EXECUTOR is not None:
        schedule_hls_detection(stream_id, target_url)
        rescheduled = True

    return jsonify({
        "status": "ok",
        "removed": removed,
        "jobs_cancelled": cancelled,
        "rescheduled": rescheduled,
    })


@app.route("/live")
def legacy_stream_live():
    """Backward compatible alias for /stream/live"""
    return stream_live()


def _classify_embed_target(url: str):
    """Return (kind, test_url) for a given input URL matching how we embed."""
    u = (url or "").strip()
    lu = u.lower()
    if "youtube.com" in lu or "youtu.be/" in lu:
        vid = None
        if "watch?v=" in u:
            try:
                vid = u.split("watch?v=")[1].split("&")[0].split("#")[0]
            except Exception:
                vid = None
        elif "youtu.be/" in u:
            try:
                vid = u.split("youtu.be/")[1].split("?")[0].split("&")[0]
            except Exception:
                vid = None
        if vid:
            return "youtube", f"https://www.youtube-nocookie.com/embed/{vid}"
        return "youtube", "https://www.youtube-nocookie.com/embed/"
    if "twitch.tv" in lu:
        try:
            channel = u.split("twitch.tv/")[1].split("/")[0]
        except Exception:
            channel = ""
        return "twitch", f"https://player.twitch.tv/?channel={channel}"
    if lu.endswith(".m3u8") or lu.endswith(".mpd"):
        return "hls", u
    return "website", u


@app.route("/test_embed", methods=["POST"])
def test_embed():
    data = request.get_json(silent=True) or {}
    url = (data.get("url") or "").strip()
    try:
        parsed = urlparse(url)
    except Exception:
        return jsonify({"status": "not_valid", "note": "Not valid"})
    if not parsed.scheme or not parsed.netloc or parsed.scheme not in ("http", "https"):
        return jsonify({"status": "not_valid", "note": "Not valid"})

    kind, target = _classify_embed_target(url)
    if kind in ("youtube", "twitch"):
        return jsonify({"status": "ok", "note": "OK", "final_url": target, "kind": kind})

    if requests is None:
        return jsonify({"status": "ok", "note": "OK", "final_url": target})

    def check_headers(headers):
        xf = (headers.get("x-frame-options") or headers.get("X-Frame-Options") or "").lower()
        if xf in ("deny", "sameorigin"):
            return False, "X-Frame-Options"
        csp = headers.get("content-security-policy") or headers.get("Content-Security-Policy")
        if csp:
            csp_l = csp.lower()
            if "frame-ancestors" in csp_l:
                fa = csp_l.split("frame-ancestors", 1)[1]
                fa = fa.split(";", 1)[0]
                if "'none'" in fa or ("*" not in fa and "http" not in fa and "https" not in fa and "'self'" not in fa):
                    return False, "CSP frame-ancestors"
        return True, None

    try:
        resp = requests.head(target, allow_redirects=True, timeout=6)
        if resp.status_code >= 400 or resp.status_code in (405, 501):
            resp = requests.get(target, allow_redirects=True, timeout=8, stream=True)
        ok_headers, _ = check_headers(resp.headers or {})
        if resp.status_code >= 400:
            return jsonify({"status": "unreachable", "http_status": resp.status_code, "note": "Unreachable"})
        if not ok_headers:
            return jsonify({"status": "not_showable", "http_status": resp.status_code, "note": "Not showable"})
        return jsonify({"status": "ok", "http_status": resp.status_code, "note": "OK", "final_url": str(resp.url)})
    except requests.exceptions.SSLError as exc:
        return jsonify({"status": "ok", "note": "SSL warning", "final_url": target, "warning": str(exc)})
    except requests.exceptions.RequestException:
        return jsonify({"status": "unreachable", "note": "Unreachable"})


@app.route("/images", methods=["GET"])
def get_images():
    folder = request.args.get("folder", "all")
    limit_arg = request.args.get("limit")
    offset_arg = request.args.get("offset")
    try:
        offset = int(offset_arg) if offset_arg is not None else 0
        limit = int(limit_arg) if limit_arg is not None else None
    except (TypeError, ValueError):
        return jsonify({"error": "limit and offset must be integers"}), 400
    if offset < 0 or (limit is not None and limit < 0):
        return jsonify({"error": "limit and offset must be non-negative"}), 400
    hide_nsfw = _parse_truthy(request.args.get("hide_nsfw"))
    images = list_images(folder, hide_nsfw=hide_nsfw)
    if limit_arg is not None or offset_arg is not None:
        end = offset + limit if limit is not None else None
        images = images[offset:end]
    return jsonify(images)


@app.route("/images/random", methods=["GET"])
def get_random_image():
    folder = request.args.get("folder", "all")
    hide_nsfw = _parse_truthy(request.args.get("hide_nsfw"))
    images = list_images(folder, hide_nsfw=hide_nsfw)
    if not images:
        return jsonify({"error": "No images found"}), 404
    return jsonify({"path": random.choice(images)})


@app.route("/media", methods=["GET"])
def get_media_entries():
    folder = request.args.get("folder", "all")
    limit_arg = request.args.get("limit")
    offset_arg = request.args.get("offset")
    try:
        offset = int(offset_arg) if offset_arg is not None else 0
        limit = int(limit_arg) if limit_arg is not None else None
    except (TypeError, ValueError):
        return jsonify({"error": "limit and offset must be integers"}), 400
    if offset < 0 or (limit is not None and limit < 0):
        return jsonify({"error": "limit and offset must be non-negative"}), 400
    hide_nsfw = _parse_truthy(request.args.get("hide_nsfw"))
    kind_filter = (request.args.get("kind") or "").strip().lower()
    media = list_media(folder, hide_nsfw=hide_nsfw)
    if kind_filter in ("image", "video"):
        media = [item for item in media if item.get("kind") == kind_filter]
    if limit_arg is not None or offset_arg is not None:
        end = offset + limit if limit is not None else None
        media = media[offset:end]
    return jsonify(media)


@app.route("/media/random", methods=["GET"])
def get_random_media():
    folder = request.args.get("folder", "all")
    hide_nsfw = _parse_truthy(request.args.get("hide_nsfw"))
    kind_filter = (request.args.get("kind") or "").strip().lower()
    stream_id = (request.args.get("stream_id") or "").strip()
    entries = list_media(folder, hide_nsfw=hide_nsfw)
    if kind_filter in ("image", "video"):
        entries = [item for item in entries if item.get("kind") == kind_filter]
    if not entries:
        return jsonify({"error": "No media found"}), 404
    choice = dict(random.choice(entries))
    if stream_id and stream_id in settings and isinstance(settings.get(stream_id), dict):
        conf = settings.get(stream_id) or {}
        media_mode_value = conf.get("media_mode")
        normalized_mode = media_mode_value.strip().lower() if isinstance(media_mode_value, str) else None
        if not normalized_mode and isinstance(conf, dict):
            normalized_mode = _infer_media_mode(conf)
        _update_stream_runtime_state(
            stream_id,
            path=choice.get("path"),
            kind=choice.get("kind"),
            media_mode=normalized_mode,
            source="random_media",
        )
    return jsonify(choice)


@app.route("/notes", methods=["GET", "POST"])
def notes():
    if request.method == "GET":
        return jsonify({"notes": settings.get("_notes", "")})
    data = request.get_json(silent=True) or {}
    settings["_notes"] = data.get("notes", "")
    save_settings(settings)
    return jsonify({"status": "saved"})


def _send_image_response(path: Union[str, Path]):
    """Return an image response with consistent caching headers."""
    abs_path = os.fspath(path)
    stat = os.stat(abs_path)
    etag_source = f"{stat.st_mtime_ns}-{stat.st_size}".encode("utf-8")
    etag_value = generate_etag(etag_source)
    response = send_file(
        abs_path,
        conditional=True,
        max_age=IMAGE_CACHE_TIMEOUT,
        etag=etag_value,
    )
    # Long-lived cache headers let browsers reuse thumbnails/originals without redownloading.
    # max_age mirrors Flask's cache_timeout value for forward compatibility.
    response.headers["Cache-Control"] = f"public, max-age={IMAGE_CACHE_CONTROL_MAX_AGE}"
    return response


def _send_video_response(path: Union[str, Path]):
    """Support streaming video files with caching and range requests."""
    abs_path = os.fspath(path)
    response = send_file(
        abs_path,
        conditional=True,
    )
    response.headers.setdefault("Cache-Control", f"public, max-age={IMAGE_CACHE_TIMEOUT}")
    response.headers.setdefault("Accept-Ranges", "bytes")
    return response


@app.route("/stream/image/<path:image_path>")
def serve_image(image_path):
    full_path = _resolve_virtual_media_path(image_path)
    if full_path is None or not full_path.exists():
        return "Not found", 404
    return _send_image_response(full_path)


@app.route("/stream/video/<path:video_path>")
def serve_video(video_path):
    target_path = _resolve_virtual_media_path(video_path)
    if target_path is None:
        return "Invalid path", 400
    if not target_path.exists() or not target_path.is_file():
        return "Not found", 404
    if target_path.suffix.lower() not in VIDEO_EXTENSIONS:
        return "Unsupported media type", 415
    return _send_video_response(target_path)


@app.route("/stream/thumbnail/<stream_id>", methods=["GET"])
def stream_thumbnail_metadata(stream_id):
    info = _compute_thumbnail_snapshot(stream_id)
    if info is None:
        return jsonify({"error": f"No stream '{stream_id}' found"}), 404
    force_refresh = _parse_truthy(request.args.get("force"))
    if force_refresh:
        thumbnail_info = _refresh_stream_thumbnail(stream_id, info, force=True)
    else:
        thumbnail_info = _get_runtime_thumbnail_payload(stream_id)
        if thumbnail_info is None:
            thumbnail_info = _refresh_stream_thumbnail(stream_id, info)
    timestamp = info.get('timestamp')
    cache_key = None
    if isinstance(timestamp, (int, float)):
        try:
            cache_key = str(int(timestamp))
        except (TypeError, ValueError):
            cache_key = None
    raw_url = thumbnail_info.get("url") if isinstance(thumbnail_info, dict) else None
    if raw_url and raw_url.startswith("data:"):
        image_url = raw_url
    else:
        image_url = raw_url or url_for('stream_thumbnail_image', stream_id=stream_id)
        if cache_key:
            image_url = f"{image_url}?v={cache_key}"
    payload = {
        'stream_id': stream_id,
        'media_mode': info.get('media_mode'),
        'kind': info.get('kind'),
        'path': info.get('path'),
        'image_url': image_url,
        'thumbnail': thumbnail_info,
        'placeholder': bool(info.get('placeholder')),
        'badge': info.get('badge'),
        'updated_at': _runtime_timestamp_to_iso(info.get('timestamp')),
        'source': info.get('source'),
    }
    stream_url = info.get('stream_url')
    if stream_url:
        payload['stream_url'] = stream_url
    return jsonify(payload)


@app.route("/stream/thumbnail/<stream_id>/image", methods=["GET"])
def stream_thumbnail_image(stream_id):
    info = _compute_thumbnail_snapshot(stream_id)
    if info is None:
        return 'Not found', 404
    _refresh_stream_thumbnail(stream_id, info)
    target_path = _thumbnail_disk_path(stream_id)
    raw_record: Optional[Dict[str, Any]] = None
    with STREAM_RUNTIME_LOCK:
        entry = STREAM_RUNTIME_STATE.get(stream_id)
        if entry and isinstance(entry.get("thumbnail"), dict):
            raw_record = dict(entry["thumbnail"])
    if target_path.exists():
        response = send_file(str(target_path), mimetype='image/jpeg')
    else:
        data_url = raw_record.get("_data_url") if isinstance(raw_record, dict) else None
        if isinstance(data_url, str) and "," in data_url:
            encoded = data_url.split(",", 1)[1]
            try:
                binary = base64.b64decode(encoded)
            except Exception:  # pragma: no cover - defensive decode
                binary = b""
            buffer = io.BytesIO(binary)
            response = send_file(buffer, mimetype='image/jpeg')
        else:
            image_obj, _ = _render_thumbnail_image(info)
            buffer = _thumbnail_image_to_bytes(image_obj)
            response = send_file(buffer, mimetype='image/jpeg')
    response.headers['Cache-Control'] = 'no-store, max-age=0'
    return response


@app.route("/thumbnails/<stream_id>.jpg", methods=["GET"])
def cached_stream_thumbnail(stream_id: str):
    """Serve cached dashboard thumbnails via the simplified public path."""
    return stream_thumbnail_image(stream_id)


def _normalize_group_layout(layout: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Return a sanitized layout configuration for group mosaics."""
    if not isinstance(layout, dict):
        return None
    allowed_layouts = {"grid", "focus", "pip"}
    layout_value_raw = layout.get("layout", "grid")
    layout_value = str(layout_value_raw).strip().lower() if isinstance(layout_value_raw, str) else "grid"
    if layout_value not in allowed_layouts:
        layout_value = "grid"

    def _bounded_int(value: Any, lower: int, upper: int) -> Optional[int]:
        maybe = _maybe_int(value)
        if maybe is None:
            return None
        bounded = int(_clamp(maybe, lower, upper))
        return bounded

    sanitized: Dict[str, Any] = {"layout": layout_value}

    cols = _bounded_int(layout.get("cols"), 1, 8)
    rows = _bounded_int(layout.get("rows"), 1, 8)
    sanitized["cols"] = cols
    sanitized["rows"] = rows

    focus_mode = layout.get("focus_mode") if isinstance(layout.get("focus_mode"), str) else None
    if focus_mode not in {"1-2", "1-3", "1-5"}:
        focus_mode = "1-5"
    focus_pos = layout.get("focus_pos") if isinstance(layout.get("focus_pos"), str) else None
    allowed_focus_pos = {"left", "right", "top", "bottom", "top-left", "top-right", "bottom-left", "bottom-right"}
    if focus_pos not in allowed_focus_pos:
        focus_pos = "bottom-right" if focus_mode == "1-5" else ("right" if focus_mode == "1-2" else "bottom")
    focus_main = layout.get("focus_main") if isinstance(layout.get("focus_main"), str) else None
    sanitized["focus_mode"] = focus_mode
    sanitized["focus_pos"] = focus_pos
    sanitized["focus_main"] = focus_main

    pip_main = layout.get("pip_main") if isinstance(layout.get("pip_main"), str) else None
    pip_pip = layout.get("pip_pip") if isinstance(layout.get("pip_pip"), str) else None
    pip_corner = layout.get("pip_corner") if isinstance(layout.get("pip_corner"), str) else None
    if pip_corner not in {"top-left", "top-right", "bottom-left", "bottom-right"}:
        pip_corner = "bottom-right"
    pip_size = _bounded_int(layout.get("pip_size"), 10, 50) or 25

    sanitized["pip_main"] = pip_main
    sanitized["pip_pip"] = pip_pip
    sanitized["pip_corner"] = pip_corner
    sanitized["pip_size"] = pip_size

    return sanitized

@app.route("/stream/group/<name>")
def stream_group(name):
    groups = settings.get("_groups", {})
    group_def = groups.get(name)
    if not group_def and name.lower() == "default":
        # Dynamic default group = all configured streams
        group_def = [k for k in settings.keys() if not k.startswith("_")]
    if not group_def:
        return f"No group '{name}'", 404
    # Support both legacy list and new object
    layout_conf: Optional[Dict[str, Any]] = None
    if isinstance(group_def, dict):
        members = group_def.get("streams", [])
        layout_conf = _normalize_group_layout(group_def.get("layout"))
    else:
        members = list(group_def)
    member_ids = [m for m in members if m in settings]
    unique_ids: List[str] = []
    seen_ids: Set[str] = set()
    for stream_id in member_ids:
        if stream_id not in seen_ids:
            seen_ids.add(stream_id)
            unique_ids.append(stream_id)
    stream_lookup = {stream_id: settings[stream_id] for stream_id in unique_ids}
    stream_entries = [(stream_id, stream_lookup[stream_id]) for stream_id in member_ids]
    focus_order = list(member_ids)
    if layout_conf and layout_conf.get("layout") == "focus":
        focus_main_id = layout_conf.get("focus_main")
        if focus_main_id in stream_lookup:
            reordered: List[str] = [focus_main_id]
            skipped = False
            for stream_id in member_ids:
                if stream_id == focus_main_id and not skipped:
                    skipped = True
                    continue
                reordered.append(stream_id)
            focus_order = reordered
    return render_template(
        "streams.html",
        stream_settings=stream_lookup,
        stream_entries=stream_entries,
        stream_order=member_ids,
        unique_stream_ids=unique_ids,
        focus_order=focus_order,
        mosaic_settings=layout_conf,
    )


@socketio.on("disconnect")
def handle_socket_disconnect():
    sid = request.sid
    detached_jobs = job_manager.detach_listener(sid)
    if not detached_jobs:
        logger.info('%s Client %s disconnected; no active Stable Horde jobs linked.', STABLE_HORDE_LOG_PREFIX, sid)
        return
    for job in detached_jobs:
        if job.stable_id:
            logger.info(
                '%s Client %s disconnected; continuing job %s (%s) in background',
                STABLE_HORDE_LOG_PREFIX,
                sid,
                job.stable_id,
                job.stream_id,
            )
        else:
            logger.info(
                '%s Client %s disconnected; continuing job for %s in background',
                STABLE_HORDE_LOG_PREFIX,
                sid,
                job.stream_id,
            )


@socketio.on('ai_watch')
def handle_ai_watch(payload):
    stream_id = ''
    if isinstance(payload, dict):
        stream_id = str(payload.get('stream_id') or '').strip()
    elif isinstance(payload, str):
        stream_id = payload.strip()
    if not stream_id:
        return
    job_manager.attach_listener(stream_id, request.sid)


@socketio.on('ai_unwatch')
def handle_ai_unwatch(payload):
    stream_id = ''
    if isinstance(payload, dict):
        stream_id = str(payload.get('stream_id') or '').strip()
    elif isinstance(payload, str):
        stream_id = payload.strip()
    if not stream_id:
        return
    job_manager.detach_listener(request.sid, stream_id)


@socketio.on("stream_subscribe")
def handle_stream_subscribe(payload):
    stream_id = ""
    if isinstance(payload, dict):
        stream_id = str(payload.get("stream_id") or "").strip()
    elif isinstance(payload, str):
        stream_id = payload.strip()
    if not stream_id:
        return
    join_room(stream_id)
    job_manager.attach_listener(stream_id, request.sid)
    if playback_manager is None:
        return
    state = playback_manager.ensure_started(stream_id)
    if not state:
        state = playback_manager.get_state(stream_id)
    if state:
        playback_manager.emit_state(state, room=request.sid, event=STREAM_INIT_EVENT)


@socketio.on("stream_unsubscribe")
def handle_stream_unsubscribe(payload):
    stream_id = ""
    if isinstance(payload, dict):
        stream_id = str(payload.get("stream_id") or "").strip()
    elif isinstance(payload, str):
        stream_id = payload.strip()
    if not stream_id:
        return
    leave_room(stream_id)
    job_manager.detach_listener(request.sid, stream_id)


@socketio.on('video_control')
def handle_video_control(payload):
    if not isinstance(payload, dict):
        return
    stream_id = str(payload.get('stream_id') or '').strip()
    if not stream_id:
        return
    action = str(payload.get('action') or '').strip().lower()
    allowed_actions = {"play", "pause", "toggle", "skip_next", "skip_prev", "set_volume"}
    if action not in allowed_actions:
        return
    volume_value: Optional[float] = None
    if action == 'set_volume':
        try:
            volume_value = float(payload.get('volume'))
        except (TypeError, ValueError):
            return
    if playback_manager is not None:
        if action == 'skip_next':
            playback_manager.skip_next(stream_id)
        elif action == 'skip_prev':
            playback_manager.skip_previous(stream_id)
        elif action == 'pause':
            playback_manager.pause(stream_id)
        elif action == 'play':
            state = playback_manager.get_state(stream_id)
            if state and state.get("status") == "paused":
                playback_manager.resume(stream_id)
            else:
                playback_manager.ensure_started(stream_id)
        elif action == 'toggle':
            playback_manager.toggle(stream_id)
        elif action == 'set_volume' and volume_value is not None:
            playback_manager.set_volume(stream_id, volume_value)
    message: Dict[str, Any] = {
        "stream_id": stream_id,
        "action": action,
        "timestamp": time.time(),
    }
    if volume_value is not None:
        message['volume'] = max(0.0, min(1.0, volume_value))
    safe_emit('video_control', message)


if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
