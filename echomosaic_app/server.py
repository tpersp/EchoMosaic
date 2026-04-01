import eventlet
eventlet.monkey_patch()

import base64
import errno
import json
import atexit
import logging
import math
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
import socket
import sys
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple, Union, Set
from urllib.parse import quote, urlparse, parse_qs
from collections import OrderedDict

class LRUCache:
    def __init__(self, maxsize: int) -> None:
        self.cache: OrderedDict[Any, Any] = OrderedDict()
        self.maxsize = maxsize
        self.lock = threading.Lock()

    def get(self, key: Any, default: Any = None) -> Any:
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
            return default

    def __setitem__(self, key: Any, value: Any) -> None:
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            self.cache[key] = value
            if len(self.cache) > self.maxsize:
                self.cache.popitem(last=False)

    def pop(self, key: Any, *args: Any) -> Any:
        with self.lock:
            return self.cache.pop(key, *args)

    def keys(self):
        with self.lock:
            return list(self.cache.keys())

    def __len__(self) -> int:
        with self.lock:
            return len(self.cache)

try:
    import engineio.payload  # type: ignore[import]
    import engineio.server  # type: ignore[import]
except ModuleNotFoundError:
    engineio = None  # type: ignore[assignment]
    _ORIGINAL_ENGINEIO_HANDLE_REQUEST = None
    logging.getLogger(__name__).warning(
        "engineio package not available; live socket optimisations disabled."
    )
else:
    engineio = sys.modules.get("engineio")
    # --- Engine.IO packet limit patch (prevents "Too many packets in payload") ---
    engineio.payload.Payload.max_decode_packets = 200
    _ORIGINAL_ENGINEIO_HANDLE_REQUEST = engineio.server.Server.handle_request

try:
    from PIL import Image, ImageDraw, ImageFont, ImageOps
except ModuleNotFoundError:  # pragma: no cover - optional dependency handling
    class _MissingPillowModule:
        def __getattr__(self, item: str) -> Any:
            raise RuntimeError(
                "Pillow is required for image processing features. "
                "Install it via 'pip install Pillow'."
            )

    Image = ImageDraw = ImageFont = ImageOps = _MissingPillowModule()  # type: ignore[assignment]

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
from flask_socketio import SocketIO
from stablehorde import StableHorde, StableHordeError, StableHordeCancelled
from picsum import (
    register_picsum_routes,
    assign_new_picsum_to_stream,
    configure_socketio,
)
from echomosaic_app.bootstrap import (
    create_flask_app,
    initialize_socketio,
    register_bootstrap_features,
    register_blueprint_with_legacy_aliases,
    run_dev_server,
)
from echomosaic_app.config_runtime import build_media_runtime
from echomosaic_app.routes.ai import create_ai_blueprint
from echomosaic_app.routes.assets import create_assets_blueprint
from echomosaic_app.routes.dashboard import create_dashboard_blueprint
from echomosaic_app.extensions import socketio
from echomosaic_app.routes.diagnostics import create_diagnostics_blueprint
from echomosaic_app.routes.library import create_library_blueprint
from echomosaic_app.routes.live import create_live_blueprint
from echomosaic_app.routes.media import create_media_blueprint
from echomosaic_app.routes.settings_operations import create_settings_operations_blueprint
from echomosaic_app.services.ai_orchestration import AIOrchestrationService
from echomosaic_app.services.ai_execution import AIExecutionService
from echomosaic_app.services.asset_delivery import AssetDeliveryService
from echomosaic_app.services.auto_schedulers import build_auto_schedulers
from echomosaic_app.services.groups import GroupService
from echomosaic_app.services.live_hls import HLSCacheEntry, LiveHLSService
from echomosaic_app.services.media_catalog import MediaCatalogService
from echomosaic_app.services.media_library import MediaLibraryService
from echomosaic_app.services.operations import OperationsService
from echomosaic_app.services.playback import PlaybackService
from echomosaic_app.services.playback_engine import StreamPlaybackManager
from echomosaic_app.services.stream_config import StreamConfigService
from echomosaic_app.services.stream_runtime import StreamRuntimeService
from echomosaic_app.services.timer_sync import TimerSyncService
from echomosaic_app.services.thumbnailing import ThumbnailService
from echomosaic_app.services.youtube_embed import YouTubeEmbedService
from echomosaic_app.sockets.streams import register_stream_socket_handlers
from echomosaic_app.sockets.youtube_sync import register_youtube_sync_socket_handlers
from echomosaic_app.state.ai_runtime import build_ai_runtime
from echomosaic_app.state.hls_runtime import build_hls_runtime
from echomosaic_app.state.media_runtime import build_media_cache_runtime
from echomosaic_app.state.playback_runtime import build_playback_runtime
from echomosaic_app.state.settings_store import SettingsStore
from echomosaic_app.state.stream_runtime import build_stream_runtime
from echomosaic_app.state.youtube_runtime import build_youtube_runtime
from system_monitor import get_system_stats
import config_manager
from media_manager import MediaManager, MediaManagerError, MEDIA_MANAGER_CACHE_SUBDIR
import timer_manager
from timer_manager import TimerManager, default_timer_config, ensure_timer_config
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

_SOCKET_NOISE_KEYWORDS = (
    "bad file descriptor",
    "connection timed out",
    "broken pipe",
    "timed out",
)
_SOCKET_NOISE_ERRNOS = {9, 32, 54, 60, 104, 110}


class SocketNoiseFilter(logging.Filter):
    """Downgrade noisy disconnect errors to DEBUG without tracebacks."""

    def _is_socket_noise(self, record: logging.LogRecord) -> bool:
        if record.exc_info:
            exception = record.exc_info[1]
            if isinstance(exception, (OSError, socket.error, TimeoutError)):
                errno = getattr(exception, "errno", None)
                if errno in _SOCKET_NOISE_ERRNOS:
                    return True
                text = str(exception)
                if text:
                    lowered = text.lower()
                    return any(keyword in lowered for keyword in _SOCKET_NOISE_KEYWORDS)
        try:
            message = record.getMessage()
        except Exception:
            message = record.msg
        if not message:
            return False
        lowered_message = str(message).lower()
        return any(keyword in lowered_message for keyword in _SOCKET_NOISE_KEYWORDS)

    def filter(self, record: logging.LogRecord) -> bool:
        if self._is_socket_noise(record):
            record.levelno = logging.DEBUG
            record.levelname = logging.getLevelName(logging.DEBUG)
            record.exc_info = None
            record.exc_text = None
            record.stack_info = None
        return True


_socket_noise_filter = SocketNoiseFilter()


def _install_socket_noise_filter() -> None:
    root_logger = logging.getLogger()
    if _socket_noise_filter not in root_logger.filters:
        root_logger.addFilter(_socket_noise_filter)

    noisy_logger_names = (
        "eventlet.wsgi.server",
        "gunicorn.error",
        "gunicorn.access",
        "engineio.server",
    )
    for name in noisy_logger_names:
        target_logger = logging.getLogger(name)
        if _socket_noise_filter not in target_logger.filters:
            target_logger.addFilter(_socket_noise_filter)

    logging.getLogger("eventlet.wsgi.server").setLevel(logging.ERROR)


_install_socket_noise_filter()

app = create_flask_app()

if engineio is not None and _ORIGINAL_ENGINEIO_HANDLE_REQUEST is not None:
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

initialize_socketio(app)
register_bootstrap_features(
    app,
    configure_socketio=configure_socketio,
    register_picsum_routes=register_picsum_routes,
)

logger = logging.getLogger(__name__)

_emit_throttle_lock = threading.Lock()
_emit_min_interval = 0.05  # seconds
_last_emit_timestamp = 0.0
UPDATE_PROGRESS_EVENT = "update_progress"
UPDATE_STAGE_SEQUENCE = ("fetch", "checkout", "apply", "dependencies", "restart", "wait_restart")
_update_job_lock = threading.Lock()
_update_job_active = False


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
    except (OSError, BrokenPipeError) as exc:  # pragma: no cover
        app.logger.warning(
            "[SocketIO] Tried to emit to disconnected client for event '%s': %s",
            event_name,
            exc,
        )
    except Exception as exc:
        app.logger.error(
            "[SocketIO] Failed to emit event '%s': %s",
            event_name,
            exc,
        )


def _set_update_job_active(active: bool) -> bool:
    global _update_job_active
    with _update_job_lock:
        if active and _update_job_active:
            return False
        _update_job_active = active
        return True


def _emit_update_progress(
    socket_id: Optional[str],
    *,
    stage: Optional[str] = None,
    message: Optional[str] = None,
    line: Optional[str] = None,
    state: str = "info",
    complete: bool = False,
    failed: bool = False,
) -> None:
    if not socket_id:
        return
    payload: Dict[str, Any] = {
        "state": state,
        "complete": bool(complete),
        "failed": bool(failed),
    }
    if stage:
        payload["stage"] = stage
    if message:
        payload["message"] = message
    if line is not None:
        payload["line"] = line
    safe_emit(UPDATE_PROGRESS_EVENT, payload, to=socket_id)


def _update_stage_from_line(line: str, current_stage: str) -> str:
    lowered = line.strip().lower()
    if "fetching latest changes" in lowered:
        return "fetch"
    if "already on '" in lowered or "head is now at" in lowered or "checking out" in lowered:
        return "checkout"
    if "restoring" in lowered or "applying" in lowered:
        return "apply"
    if "updating python dependencies" in lowered:
        return "dependencies"
    if lowered.startswith("restarting ") or "restarting echomosaic" in lowered:
        return "restart"
    if "update complete" in lowered:
        return "wait_restart"
    return current_stage


def _run_update_job(repo_path: str, update_script: str, service_name: str, socket_id: Optional[str]) -> None:
    stage = "fetch"
    _emit_update_progress(socket_id, stage=stage, message="Starting update…", state="info")
    process: Optional[subprocess.Popen[str]] = None
    try:
        process = subprocess.Popen(
            ["bash", update_script],
            cwd=repo_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for raw_line in process.stdout:
            clean_line = raw_line.rstrip("\n")
            if not clean_line:
                continue
            next_stage = _update_stage_from_line(clean_line, stage)
            stage = next_stage
            _emit_update_progress(
                socket_id,
                stage=stage,
                message=clean_line,
                line=clean_line,
                state="info",
            )
        return_code = process.wait()
        if return_code == 0:
            _emit_update_progress(
                socket_id,
                stage="wait_restart",
                message=f"{service_name} restarting… waiting for service to come back.",
                state="info",
                complete=True,
            )
        elif return_code == -15 and stage in {"restart", "wait_restart"}:
            _emit_update_progress(
                socket_id,
                stage="wait_restart",
                message=f"{service_name} restart interrupted the updater connection as expected. Waiting for service to come back.",
                line=f"{service_name} restart interrupted the updater connection as expected.",
                state="info",
                complete=True,
            )
        else:
            _emit_update_progress(
                socket_id,
                stage=stage,
                message=f"Update failed with exit code {return_code}.",
                state="error",
                failed=True,
            )
    except Exception as exc:
        logger.exception("In-app update failed")
        _emit_update_progress(
            socket_id,
            stage=stage,
            message=f"Update failed: {exc}",
            line=f"ERROR: {exc}",
            state="error",
            failed=True,
        )
    finally:
        _set_update_job_active(False)
        if process is not None and process.poll() is None:
            try:
                process.kill()
            except Exception:
                pass


SETTINGS_FILE = "settings.json"
CONFIG_FILE = config_manager.CONFIG_FILE.name
SETTINGS_STORE = SettingsStore(SETTINGS_FILE, debounce_seconds=2.0)

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
MAX_IMAGE_DIMENSION = 8192
DASHBOARD_THUMBNAIL_SIZE = (128, 72)
THUMBNAIL_JPEG_QUALITY = 60
IMAGE_CACHE_TIMEOUT = 60 * 60 * 24 * 7  # One week default for conditional responses
IMAGE_CACHE_CONTROL_MAX_AGE = 31536000  # One year for browser Cache-Control headers
BAD_MEDIA_LOG_TTL = 60 * 10

IMAGE_QUALITY_CHOICES = {"auto", "thumb", "medium", "full"}

AI_MODE = "ai"
AI_GENERATE_MODE = "generate"
AI_RANDOM_MODE = "random"
AI_SPECIFIC_MODE = "specific"
AI_SETTINGS_KEY = "ai_settings"
AI_STATE_KEY = "ai_state"
AI_PRESETS_KEY = "_ai_presets"
STREAM_ORDER_KEY = "_stream_order"

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
MEDIA_LIBRARY_DEFAULT = "media"
AI_MEDIA_LIBRARY = "ai"
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

AUTO_GENERATE_INTERVAL_UNITS = {"minutes": 60.0, "hours": 3600.0}

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
    MEDIA_MODE_AI: {AI_GENERATE_MODE, AI_RANDOM_MODE, AI_SPECIFIC_MODE, AI_MODE},
    MEDIA_MODE_PICSUM: {MEDIA_MODE_PICSUM},
}
TIMER_SUPPORTED_MODES = {AI_GENERATE_MODE, MEDIA_MODE_PICSUM, AI_MODE}
TIMER_MODE_LABELS = {
    AI_GENERATE_MODE: "AI Images",
    AI_MODE: "AI Images",
    MEDIA_MODE_PICSUM: "Picsum",
}

SYNC_CONFIG_KEY = "sync"
SYNC_TIMERS_KEY = "_sync_timers"
SYNC_TIMER_DEFAULT_INTERVAL = 10.0
SYNC_TIMER_DEFAULT_LABEL = "Master"
SYNC_TIMER_MIN_INTERVAL = 1.0
SYNC_TIMER_MAX_INTERVAL = 24 * 60 * 60.0
SYNC_TIMER_MIN_OFFSET = 0.0
SYNC_SWITCH_LEAD_SECONDS = 0.25
SYNC_SUPPORTED_MEDIA_MODES = {MEDIA_MODE_IMAGE}

CONFIG: Dict[str, Any] = config_manager.load_config()
NSFW_KEYWORD = "nsfw"
MEDIA_RUNTIME = build_media_runtime(
    config=CONFIG,
    media_library_default=MEDIA_LIBRARY_DEFAULT,
    ai_media_library=AI_MEDIA_LIBRARY,
    thumbnail_subdir=THUMBNAIL_SUBDIR,
    internal_media_dirs=INTERNAL_MEDIA_DIRS,
    nsfw_keyword=NSFW_KEYWORD,
)
STANDARD_MEDIA_ROOTS = MEDIA_RUNTIME.standard_media_roots
AI_MEDIA_ROOTS = MEDIA_RUNTIME.ai_media_roots
MEDIA_ROOTS = MEDIA_RUNTIME.media_roots
AVAILABLE_MEDIA_ROOTS = MEDIA_RUNTIME.available_media_roots
AVAILABLE_MEDIA_ROOTS_BY_LIBRARY = MEDIA_RUNTIME.available_media_roots_by_library
MEDIA_ROOT_LOOKUP = MEDIA_RUNTIME.media_root_lookup
PRIMARY_MEDIA_ROOT = MEDIA_RUNTIME.primary_media_root
PRIMARY_AI_MEDIA_ROOT = MEDIA_RUNTIME.primary_ai_media_root
THUMBNAIL_CACHE_DIR = MEDIA_RUNTIME.thumbnail_cache_dir
MEDIA_MANAGEMENT_ALLOW_EDIT = MEDIA_RUNTIME.media_management_allow_edit
MEDIA_UPLOAD_MAX_MB = MEDIA_RUNTIME.media_upload_max_mb
MEDIA_ALLOWED_EXTS = MEDIA_RUNTIME.media_allowed_exts
MEDIA_THUMB_WIDTH = MEDIA_RUNTIME.media_thumb_width
MEDIA_PREVIEW_ENABLED = MEDIA_RUNTIME.media_preview_enabled
MEDIA_PREVIEW_FRAMES = MEDIA_RUNTIME.media_preview_frames
MEDIA_PREVIEW_WIDTH = MEDIA_RUNTIME.media_preview_width
MEDIA_PREVIEW_MAX_DURATION = MEDIA_RUNTIME.media_preview_max_duration
MEDIA_PREVIEW_MAX_MB = MEDIA_RUNTIME.media_preview_max_mb
MEDIA_PREVIEW_MAX_BYTES = MEDIA_RUNTIME.media_preview_max_bytes
MEDIA_MANAGER = MEDIA_RUNTIME.media_manager
MEDIA_UPLOAD_MAX_BYTES = MEDIA_RUNTIME.media_upload_max_bytes


def _media_error_response(exc: MediaManagerError):
    status = getattr(exc, "status", 400) or 400
    payload = {"error": exc.message, "code": exc.code}
    return jsonify(payload), status


def _require_media_edit() -> None:
    if not MEDIA_MANAGEMENT_ALLOW_EDIT:
        raise MediaManagerError("Media editing is disabled", code="forbidden", status=403)


MEDIA_CACHE_RUNTIME = build_media_cache_runtime(cache_factory=lambda maxsize: LRUCache(maxsize=maxsize))
_BAD_MEDIA_LOG_CACHE = MEDIA_CACHE_RUNTIME.bad_media_log_cache

# Cache image paths per folder so we can serve repeated requests without rescanning the disk.
IMAGE_CACHE = MEDIA_CACHE_RUNTIME.image_cache


def _media_root_available(root: config_manager.MediaRoot) -> bool:
    try:
        return root.path.exists() and root.path.is_dir() and os.access(root.path, os.R_OK)
    except OSError:
        return False


def _normalize_library_key(value: Any, default: str = MEDIA_LIBRARY_DEFAULT) -> str:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {MEDIA_LIBRARY_DEFAULT, AI_MEDIA_LIBRARY}:
            return normalized
    return default


def _library_roots(library: str) -> List[config_manager.MediaRoot]:
    return list(AVAILABLE_MEDIA_ROOTS_BY_LIBRARY.get(_normalize_library_key(library), []))


def _library_for_media_mode(media_mode: Optional[str]) -> str:
    normalized = media_mode.strip().lower() if isinstance(media_mode, str) else ""
    return AI_MEDIA_LIBRARY if normalized == MEDIA_MODE_AI else MEDIA_LIBRARY_DEFAULT


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
    if ":/" in text:
        alias, remainder = text.split(":/", 1)
        alias = alias.strip()
        if alias in MEDIA_ROOT_LOOKUP:
            return alias, remainder.strip("/")
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
    return YOUTUBE_EMBED_SERVICE.sanitize_embed_metadata(raw)


def _is_youtube_host(host: str) -> bool:
    return YOUTUBE_EMBED_SERVICE.is_youtube_host(host)


def _is_youtube_url(url: Optional[str]) -> bool:
    return YOUTUBE_EMBED_SERVICE.is_youtube_url(url)


def _parse_youtube_url_details(url: str) -> Optional[Dict[str, Any]]:
    return YOUTUBE_EMBED_SERVICE.parse_youtube_url_details(url)


def _youtube_cache_key(details: Dict[str, Any]) -> Tuple[str, ...]:
    return YOUTUBE_EMBED_SERVICE.youtube_cache_key(details)


def _youtube_oembed_html_says_live(oembed: Optional[Dict[str, Any]]) -> bool:
    return YOUTUBE_EMBED_SERVICE.youtube_oembed_html_says_live(oembed)


def _youtube_page_looks_live(details: Dict[str, Any]) -> Optional[bool]:
    return YOUTUBE_EMBED_SERVICE.youtube_page_looks_live(details)


def _derive_youtube_content_type(
    details: Dict[str, Any], oembed: Optional[Dict[str, Any]] = None
) -> str:
    return YOUTUBE_EMBED_SERVICE.derive_youtube_content_type(details, oembed)


def _build_youtube_metadata(
    details: Dict[str, Any], oembed: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    return YOUTUBE_EMBED_SERVICE.build_youtube_metadata(details, oembed)


def _youtube_oembed_lookup(
    url: str,
    details: Dict[str, Any],
    *,
    force: bool = False,
) -> Optional[Dict[str, Any]]:
    return YOUTUBE_EMBED_SERVICE.youtube_oembed_lookup(url, details, force=force)


def _set_runtime_embed_metadata(stream_id: str, metadata: Optional[Dict[str, Any]]) -> None:
    YOUTUBE_EMBED_SERVICE.set_runtime_embed_metadata(stream_id, metadata)


def _refresh_embed_metadata(
    stream_id: str,
    conf: Dict[str, Any],
    *,
    force: bool = False,
) -> Optional[Dict[str, Any]]:
    return YOUTUBE_EMBED_SERVICE.refresh_embed_metadata(stream_id, conf, force=force)
IMAGE_CACHE_LOCK = threading.Lock()  # Kept for external callers; LRUCache also has internal lock
STREAM_RUNTIME = build_stream_runtime(cache_factory=lambda maxsize: LRUCache(maxsize=maxsize))
RESIZED_IMAGE_LOCKS = STREAM_RUNTIME.resized_image_locks
RESIZED_IMAGE_LOCKS_GUARD = STREAM_RUNTIME.resized_image_locks_guard
STREAM_RUNTIME_STATE = STREAM_RUNTIME.stream_runtime_state
STREAM_RUNTIME_LOCK = STREAM_RUNTIME.stream_runtime_lock
_VIDEO_DURATION_CACHE = STREAM_RUNTIME.video_duration_cache  # Cache video durations to avoid cv2 re-probes

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
YOUTUBE_LIVE_PROBE_CACHE_TTL = 15 * 60  # 15 minutes
YOUTUBE_LIVE_PROBE_MAX_BYTES = 900_000
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

PLAYBACK_RUNTIME = build_playback_runtime()
STREAM_PLAYBACK_HISTORY_LIMIT = PLAYBACK_RUNTIME.stream_playback_history_limit
STREAM_UPDATE_EVENT = PLAYBACK_RUNTIME.stream_update_event
STREAM_INIT_EVENT = PLAYBACK_RUNTIME.stream_init_event
SYNC_TIME_EVENT = PLAYBACK_RUNTIME.sync_time_event
STREAM_SYNC_INTERVAL_SECONDS = PLAYBACK_RUNTIME.stream_sync_interval_seconds
YOUTUBE_SYNC_EVENT = "youtube_sync"
YOUTUBE_SYNC_ROLE_EVENT = "youtube_sync_role"
YOUTUBE_SYNC_MAX_AGE_SECONDS = 20.0

LIVE_HLS_ASYNC = True
HLS_TTL_SECS = 3600
MAX_HLS_WORKERS = 3
HLS_ERROR_RETRY_SECS = 30

playback_manager: Optional["StreamPlaybackManager"] = PLAYBACK_RUNTIME.playback_manager
YOUTUBE_RUNTIME = build_youtube_runtime(cache_factory=lambda maxsize: LRUCache(maxsize=maxsize))
YOUTUBE_OEMBED_CACHE = YOUTUBE_RUNTIME.youtube_oembed_cache
YOUTUBE_OEMBED_CACHE_LOCK = YOUTUBE_RUNTIME.youtube_oembed_cache_lock
YOUTUBE_LIVE_PROBE_CACHE = YOUTUBE_RUNTIME.youtube_live_probe_cache
YOUTUBE_LIVE_PROBE_CACHE_LOCK = YOUTUBE_RUNTIME.youtube_live_probe_cache_lock
YOUTUBE_SYNC_STATE_LOCK = YOUTUBE_RUNTIME.youtube_sync_state_lock
YOUTUBE_SYNC_STATE = YOUTUBE_RUNTIME.youtube_sync_state
YOUTUBE_SYNC_SUBSCRIBERS = YOUTUBE_RUNTIME.youtube_sync_subscribers
YOUTUBE_SYNC_LEADERS = YOUTUBE_RUNTIME.youtube_sync_leaders
_YOUTUBE_IN_FLIGHT = YOUTUBE_RUNTIME.youtube_in_flight
_YOUTUBE_IN_FLIGHT_LOCK = YOUTUBE_RUNTIME.youtube_in_flight_lock

YOUTUBE_EMBED_SERVICE = YouTubeEmbedService(
    requests_module=requests,
    eventlet_module=eventlet,
    logger=logger,
    youtube_domains=YOUTUBE_DOMAINS,
    youtube_oembed_endpoint=YOUTUBE_OEMBED_ENDPOINT,
    youtube_oembed_cache_ttl=YOUTUBE_OEMBED_CACHE_TTL,
    youtube_live_probe_cache_ttl=YOUTUBE_LIVE_PROBE_CACHE_TTL,
    youtube_live_probe_max_bytes=YOUTUBE_LIVE_PROBE_MAX_BYTES,
    youtube_live_html_markers=YOUTUBE_LIVE_HTML_MARKERS,
    youtube_oembed_cache=YOUTUBE_OEMBED_CACHE,
    youtube_oembed_cache_lock=YOUTUBE_OEMBED_CACHE_LOCK,
    youtube_live_probe_cache=YOUTUBE_LIVE_PROBE_CACHE,
    youtube_live_probe_cache_lock=YOUTUBE_LIVE_PROBE_CACHE_LOCK,
    youtube_sync_state_lock=YOUTUBE_SYNC_STATE_LOCK,
    youtube_sync_state=YOUTUBE_SYNC_STATE,
    youtube_sync_subscribers=YOUTUBE_SYNC_SUBSCRIBERS,
    youtube_sync_leaders=YOUTUBE_SYNC_LEADERS,
    youtube_in_flight=_YOUTUBE_IN_FLIGHT,
    youtube_in_flight_lock=_YOUTUBE_IN_FLIGHT_LOCK,
    stream_runtime_lock=STREAM_RUNTIME_LOCK,
    stream_runtime_state=STREAM_RUNTIME_STATE,
    safe_emit=safe_emit,
    youtube_sync_role_event=YOUTUBE_SYNC_ROLE_EVENT,
    youtube_sync_max_age_seconds=YOUTUBE_SYNC_MAX_AGE_SECONDS,
    media_mode_livestream=MEDIA_MODE_LIVESTREAM,
)


def _youtube_sync_source_signature(details: Optional[Dict[str, Any]]) -> Tuple[str, str, str]:
    return YOUTUBE_EMBED_SERVICE.youtube_sync_source_signature(details)


def _get_youtube_sync_state(
    stream_id: str,
    details: Optional[Dict[str, Any]],
    *,
    max_age: float = YOUTUBE_SYNC_MAX_AGE_SECONDS,
) -> Optional[Dict[str, Any]]:
    return YOUTUBE_EMBED_SERVICE.get_youtube_sync_state(stream_id, details, max_age=max_age)


def _store_youtube_sync_state(
    stream_id: str,
    details: Dict[str, Any],
    *,
    position: float,
    playlist_index: Optional[int] = None,
    video_id: Optional[str] = None,
    reporter_sid: Optional[str] = None,
) -> Dict[str, Any]:
    return YOUTUBE_EMBED_SERVICE.store_youtube_sync_state(
        stream_id,
        details,
        position=position,
        playlist_index=playlist_index,
        video_id=video_id,
        reporter_sid=reporter_sid,
    )


def _youtube_subscriber_ids(stream_id: str) -> List[str]:
    return YOUTUBE_EMBED_SERVICE.youtube_subscriber_ids(stream_id)


def _assign_youtube_sync_leader(stream_id: str, sid: str) -> None:
    YOUTUBE_EMBED_SERVICE.assign_youtube_sync_leader(stream_id, sid)


def _promote_youtube_sync_leader(stream_id: str) -> None:
    YOUTUBE_EMBED_SERVICE.promote_youtube_sync_leader(stream_id)


def _remove_youtube_sync_subscriber(sid: str, stream_id: Optional[str] = None) -> None:
    YOUTUBE_EMBED_SERVICE.remove_youtube_sync_subscriber(sid, stream_id)


def _youtube_sync_role_for_sid(stream_id: str, sid: str) -> bool:
    return YOUTUBE_EMBED_SERVICE.youtube_sync_role_for_sid(stream_id, sid)


def _normalize_folder_key(folder: Optional[str]) -> str:
    """Normalize request folder values into the cache key used internally."""
    return MEDIA_CATALOG_SERVICE.normalize_folder_key(folder)


def _cache_scope_key(folder_key: str, library: str) -> str:
    return MEDIA_CATALOG_SERVICE.cache_scope_key(folder_key, library)


def _resolve_folder_path(folder_key: str, *, library: str = MEDIA_LIBRARY_DEFAULT) -> Optional[Tuple[config_manager.MediaRoot, Path]]:
    """Return the media root and absolute filesystem path for a cache key."""
    return MEDIA_CATALOG_SERVICE.resolve_folder_path(folder_key, library=library)


def _scan_root_for_cache(
    root: config_manager.MediaRoot,
    base_path: Path,
) -> Tuple[List[Dict[str, str]], Dict[str, Tuple[int, int]]]:
    return MEDIA_CATALOG_SERVICE.scan_root_for_cache(root, base_path)


def _scan_folder_for_cache(folder_key: str, *, library: str = MEDIA_LIBRARY_DEFAULT) -> Tuple[List[Dict[str, str]], Dict[str, Tuple[int, int]]]:
    """Scan configured media directories and build the cached payload for a folder key."""
    return MEDIA_CATALOG_SERVICE.scan_folder_for_cache(folder_key, library=library)


def _directory_markers_changed(markers: Dict[str, Tuple[int, int]]) -> bool:
    """Return True when any tracked directory timestamp has diverged."""
    return MEDIA_CATALOG_SERVICE.directory_markers_changed(markers)


def refresh_image_cache(folder: str = "all", hide_nsfw: bool = False, *, force: bool = False, library: str = MEDIA_LIBRARY_DEFAULT) -> List[str]:
    """Return the cached image list for a folder, refreshing if anything changed."""
    return MEDIA_CATALOG_SERVICE.refresh_image_cache(
        folder,
        hide_nsfw=hide_nsfw,
        force=force,
        library=library,
    )


def _cache_folder_for_path(path: Optional[str]) -> Optional[str]:
    return MEDIA_CATALOG_SERVICE.cache_folder_for_path(path)


def _invalidate_media_cache(path: Optional[str], *, library: Optional[str] = None) -> None:
    MEDIA_CATALOG_SERVICE.invalidate_media_cache(path, library=library)


def initialize_image_cache() -> None:
    """Warm the cache for the root folder on startup."""
    MEDIA_CATALOG_SERVICE.initialize_image_cache()

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

def default_picsum_settings() -> Dict[str, Any]:
    return {
        "width": PICSUM_DEFAULT_WIDTH,
        "height": PICSUM_DEFAULT_HEIGHT,
        "blur": PICSUM_DEFAULT_BLUR,
        "grayscale": False,
        "seed": None,
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


def ensure_timer_defaults(conf: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure the general timer configuration exists for the stream."""

    timer_conf = ensure_timer_config(conf)
    conf["timer"] = timer_conf
    return timer_conf


def default_sync_config() -> Dict[str, Any]:
    return {
        "timer_id": None,
        "offset": 0.0,
    }


def _sanitize_sync_offset(value: Any, *, interval: Optional[float] = None) -> float:
    offset = _coerce_float(value, 0.0)
    if offset is None or isinstance(offset, bool):
        offset = 0.0
    try:
        offset = float(offset)
    except (TypeError, ValueError):
        offset = 0.0
    if math.isnan(offset) or math.isinf(offset):
        offset = 0.0
    offset = max(SYNC_TIMER_MIN_OFFSET, offset)
    if interval is not None and interval > 0:
        offset = offset % float(interval)
    return round(offset, 3)


def sanitize_sync_config(
    payload: Any,
    *,
    defaults: Optional[Dict[str, Any]] = None,
    timers: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    base = dict(defaults or default_sync_config())
    incoming = dict(payload) if isinstance(payload, dict) else {}
    timer_raw = incoming.get("timer_id", base.get("timer_id"))
    timer_id = timer_raw.strip() if isinstance(timer_raw, str) else None
    if not timer_id:
        timer_id = None

    interval = None
    if timers and timer_id and timer_id in timers:
        interval = _coerce_float(timers[timer_id].get("interval"), None)
    elif timer_id and timers is not None and timer_id not in timers:
        timer_id = None

    offset_raw = incoming.get("offset", base.get("offset", 0.0))
    offset = _sanitize_sync_offset(offset_raw, interval=interval)

    return {
        "timer_id": timer_id,
        "offset": offset,
    }


def ensure_sync_defaults(conf: Dict[str, Any], *, timers: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
    if timers is None:
        timers = settings.get(SYNC_TIMERS_KEY, {}) if isinstance(settings, dict) else {}
    sanitized = sanitize_sync_config(conf.get(SYNC_CONFIG_KEY), defaults=conf.get(SYNC_CONFIG_KEY), timers=timers)
    conf[SYNC_CONFIG_KEY] = sanitized
    return sanitized


def default_sync_timer_entry(
    *,
    label: str = SYNC_TIMER_DEFAULT_LABEL,
    interval: float = SYNC_TIMER_DEFAULT_INTERVAL,
) -> Dict[str, Any]:
    return {
        "label": label,
        "interval": interval,
    }


def default_sync_timers() -> Dict[str, Dict[str, Any]]:
    return {
        "master": default_sync_timer_entry(),
    }


def _sanitize_sync_timer_entry(timer_id: str, payload: Any) -> Dict[str, Any]:
    base = default_sync_timer_entry()
    incoming = dict(payload) if isinstance(payload, dict) else {}
    label_raw = incoming.get("label", timer_id or base["label"])
    label = str(label_raw).strip() if label_raw is not None else base["label"]
    if not label:
        label = base["label"]

    interval = _coerce_float(incoming.get("interval", base["interval"]), base["interval"])
    try:
        interval = float(interval)
    except (TypeError, ValueError):
        interval = base["interval"]
    if math.isnan(interval) or math.isinf(interval) or interval <= 0:
        interval = base["interval"]
    interval = max(SYNC_TIMER_MIN_INTERVAL, min(SYNC_TIMER_MAX_INTERVAL, interval))

    return {
        "label": label,
        "interval": round(interval, 3),
    }


def sanitize_sync_timers_collection(raw: Any) -> Dict[str, Dict[str, Any]]:
    sanitized: Dict[str, Dict[str, Any]] = {}
    def _sync_slugify(value: str) -> str:
        text = (value or "").strip().lower()
        text = re.sub(r"[^a-z0-9]+", "-", text)
        text = re.sub(r"-+", "-", text).strip("-")
        return text

    if isinstance(raw, dict):
        items = raw.items()
    elif isinstance(raw, list):
        items = []
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            timer_id = entry.get("id")
            items.append((timer_id, entry))
    else:
        return sanitized

    for timer_id, payload in items:
        if not isinstance(timer_id, str):
            continue
        slug = _sync_slugify(timer_id)
        if not slug:
            continue
        sanitized[slug] = _sanitize_sync_timer_entry(slug, payload)
    return sanitized


def ensure_sync_timers(data: Dict[str, Any]) -> bool:
    raw = data.get(SYNC_TIMERS_KEY)
    if raw is None:
        data[SYNC_TIMERS_KEY] = default_sync_timers()
        return True
    sanitized = sanitize_sync_timers_collection(raw)
    if sanitized != raw:
        data[SYNC_TIMERS_KEY] = sanitized
        return True
    return False


def get_sync_timer_config(timer_id: Optional[str]) -> Optional[Dict[str, Any]]:
    return TIMER_SYNC_SERVICE.get_sync_timer_config(timer_id)


def get_sync_timers_snapshot() -> List[Dict[str, Any]]:
    return TIMER_SYNC_SERVICE.get_sync_timers_snapshot()


def compute_next_sync_tick(now_ts: float, interval: float, offset: float) -> Optional[float]:
    if interval <= 0:
        return None
    interval = float(interval)
    offset = max(SYNC_TIMER_MIN_OFFSET, float(offset))
    if interval > 0:
        offset = offset % interval
    if now_ts < offset:
        return offset
    cycles = math.floor((now_ts - offset) / interval) + 1
    return offset + (cycles * interval)


def _clock_time_to_offset_minutes(value: Any) -> Optional[float]:
    normalized = _normalize_clock_time(value)
    if not normalized:
        return None
    hour, minute = map(int, normalized.split(":"))
    return float(hour * 60 + minute)


def migrate_legacy_timer_config(stream_id: str, conf: Dict[str, Any]) -> bool:
    """Translate legacy per-mode timer fields into the unified timer structure."""

    if not isinstance(conf, dict):
        return False

    changed = False
    ensure_timer_defaults(conf)
    timer_mode = _canonical_timer_mode(conf)

    derived_enabled: Optional[bool] = None
    derived_interval: Optional[float] = None
    derived_offset: Optional[float] = None

    def pop_legacy(container: Optional[Dict[str, Any]], key: str) -> Any:
        nonlocal changed
        if isinstance(container, dict) and key in container:
            changed = True
            return container.pop(key)
        return None

    def update_enabled(flag: Optional[bool], *, force: bool = False) -> None:
        nonlocal derived_enabled
        if flag is None:
            return
        value = bool(flag)
        if force or derived_enabled is None:
            derived_enabled = value
        else:
            derived_enabled = derived_enabled or value

    def update_interval(value: Any, *, force: bool = False) -> None:
        nonlocal derived_interval
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return
        if numeric <= 0:
            return
        if force or derived_interval is None:
            derived_interval = numeric

    def update_offset(value: Any, *, force: bool = False) -> None:
        nonlocal derived_offset
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return
        if numeric < 0:
            return
        if force or derived_offset is None:
            derived_offset = numeric

    # Stream-level legacy fields
    legacy_every = pop_legacy(conf, "every_min")
    if legacy_every is not None:
        minutes = _coerce_float(legacy_every, None)
        if minutes and minutes > 0:
            update_enabled(True)
            update_interval(minutes)

    legacy_hourly = pop_legacy(conf, "hourly_fire")
    if legacy_hourly:
        update_enabled(True, force=True)
        update_interval(60.0, force=True)

    legacy_daily = pop_legacy(conf, "daily_trigger")
    if legacy_daily:
        offset = _clock_time_to_offset_minutes(legacy_daily)
        if offset is not None:
            update_enabled(True, force=True)
            update_interval(24 * 60.0, force=True)
            update_offset(offset, force=True)

    legacy_fire_at = pop_legacy(conf, "fire_at")
    if legacy_fire_at:
        offset = _clock_time_to_offset_minutes(legacy_fire_at)
        if offset is not None:
            update_enabled(True, force=True)
            update_interval(24 * 60.0, force=True)
            update_offset(offset, force=True)

    legacy_offset = pop_legacy(conf, "timer_offset")
    if legacy_offset is None:
        legacy_offset = pop_legacy(conf, "timer_offset_min")
    if legacy_offset is not None:
        offset_value = _coerce_float(legacy_offset, None)
        if offset_value is not None:
            update_offset(offset_value)

    legacy_interval = pop_legacy(conf, "timer_interval")
    if legacy_interval is not None:
        interval_value = _coerce_float(legacy_interval, None)
        if interval_value and interval_value > 0:
            update_interval(interval_value)

    # Mode-specific legacy fields
    ai_settings = conf.get(AI_SETTINGS_KEY)
    legacy_ai_mode = pop_legacy(ai_settings, "auto_generate_mode")
    legacy_ai_interval = pop_legacy(ai_settings, "auto_generate_interval_value")
    legacy_ai_unit = pop_legacy(ai_settings, "auto_generate_interval_unit")
    legacy_ai_clock = pop_legacy(ai_settings, "auto_generate_clock_time")
    legacy_ai_clock_alt = pop_legacy(ai_settings, "auto_generate_clock")

    picsum_settings = conf.get(PICSUM_SETTINGS_KEY)
    legacy_picsum_mode = pop_legacy(picsum_settings, "auto_mode")
    legacy_picsum_interval = pop_legacy(picsum_settings, "auto_interval_value")
    legacy_picsum_unit = pop_legacy(picsum_settings, "auto_interval_unit")
    legacy_picsum_clock = pop_legacy(picsum_settings, "auto_clock_time")

    def _convert_interval(value: Any, unit: Any) -> Optional[float]:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if numeric <= 0:
            return None
        unit_key = str(unit).strip().lower() if isinstance(unit, str) else "minutes"
        seconds = AUTO_GENERATE_INTERVAL_UNITS.get(unit_key, 60.0)
        minutes = numeric * (seconds / 60.0)
        if minutes <= 0:
            return None
        return minutes

    if timer_mode in {AI_MODE, AI_GENERATE_MODE}:
        mode_value = str(legacy_ai_mode).strip().lower() if isinstance(legacy_ai_mode, str) else ""
        if mode_value == "timer":
            minutes = _convert_interval(legacy_ai_interval, legacy_ai_unit)
            if minutes:
                update_enabled(True, force=True)
                update_interval(minutes, force=True)
        elif mode_value == "clock":
            offset = _clock_time_to_offset_minutes(legacy_ai_clock or legacy_ai_clock_alt)
            if offset is not None:
                update_enabled(True, force=True)
                update_interval(24 * 60.0, force=True)
                update_offset(offset, force=True)
        elif mode_value == "off":
            update_enabled(False, force=True)
    elif timer_mode == MEDIA_MODE_PICSUM:
        mode_value = str(legacy_picsum_mode).strip().lower() if isinstance(legacy_picsum_mode, str) else ""
        if mode_value == "timer":
            minutes = _convert_interval(legacy_picsum_interval, legacy_picsum_unit)
            if minutes:
                update_enabled(True, force=True)
                update_interval(minutes, force=True)
        elif mode_value == "clock":
            offset = _clock_time_to_offset_minutes(legacy_picsum_clock)
            if offset is not None:
                update_enabled(True, force=True)
                update_interval(24 * 60.0, force=True)
                update_offset(offset, force=True)
        elif mode_value == "off":
            update_enabled(False, force=True)

    updates: Dict[str, Any] = {}
    if derived_enabled is not None:
        updates["enabled"] = bool(derived_enabled)
    if derived_interval is not None:
        updates["interval"] = float(derived_interval)
    if derived_offset is not None:
        updates["offset"] = max(0.0, float(derived_offset))

    applied = False
    if updates:
        timer_manager = TimerManager(
            mode=timer_mode or conf.get("mode") or "",
            stream_id=stream_id,
            config_owner=conf,
            snap_provider=_timer_snap_enabled,
        )
        before = dict(timer_manager.config)
        timer_manager.apply_updates(updates)
        if not timer_manager.is_enabled():
            timer_manager.update_next(None)
        after = dict(timer_manager.config)
        applied = before != after
        changed = changed or applied

    return changed


def _canonical_timer_mode(conf: Dict[str, Any]) -> Optional[str]:
    """Return the canonical timer-capable mode for the stream, if any."""

    for candidate in (
        (conf.get("media_mode") or ""),
        (conf.get("mode") or ""),
    ):
        if not isinstance(candidate, str):
            continue
        normalized = candidate.strip().lower()
        if normalized in TIMER_SUPPORTED_MODES:
            return normalized
    return None


def _timer_mode_label(mode: Optional[str]) -> str:
    if not mode:
        return "Unknown"
    return TIMER_MODE_LABELS.get(mode, mode.capitalize())


def _log_timer_schedule(stream_id: str, mode: Optional[str], moment: datetime, offset_minutes: float) -> None:
    offset_note = ""
    if offset_minutes:
        offset_note = f" (offset +{offset_minutes:g}m)"
    logger.info(
        "[Timer] %s (%s) next run at %s%s",
        stream_id,
        _timer_mode_label(mode),
        timer_manager.format_display_time(moment),
        offset_note,
    )


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
    if ensure_sync_timers(data):
        changed = True
    timers = data.get(SYNC_TIMERS_KEY, {}) if isinstance(data.get(SYNC_TIMERS_KEY), dict) else {}
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
        before_conf = deepcopy(conf)
        ensure_tag_defaults(conf)
        ensure_ai_defaults(conf)
        ensure_picsum_defaults(conf)
        ensure_timer_defaults(conf)
        ensure_sync_defaults(conf, timers=timers)
        if migrate_legacy_timer_config(key, conf):
            changed = True
        if conf != before_conf:
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
        "timer": default_timer_config(),
        SYNC_CONFIG_KEY: default_sync_config(),
    }


def _sanitize_imported_stream_config(stream_id: str, raw_conf: Any) -> Dict[str, Any]:
    if not isinstance(raw_conf, dict):
        raise ValueError(f"Stream '{stream_id}' configuration must be an object.")
    conf = default_stream_config()
    for key, value in raw_conf.items():
        conf[key] = deepcopy(value) if isinstance(value, (dict, list)) else value
    mode_raw = conf.get('mode')
    mode = mode_raw.strip().lower() if isinstance(mode_raw, str) else conf['mode']
    if mode not in {'random', 'specific', 'livestream', AI_MODE, AI_GENERATE_MODE, AI_RANDOM_MODE, AI_SPECIFIC_MODE, MEDIA_MODE_PICSUM}:
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
        if conf['mode'] not in {AI_GENERATE_MODE, AI_RANDOM_MODE, AI_SPECIFIC_MODE}:
            conf['mode'] = AI_GENERATE_MODE
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
    ensure_timer_defaults(conf)
    ensure_sync_defaults(conf)
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
            layout = GROUP_SERVICE.normalize_group_layout(payload.get('layout'))
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


def _natural_stream_order_key(value: str) -> Tuple[Any, ...]:
    parts = re.split(r"(\d+)", value.strip().lower())
    normalized: List[Any] = []
    for part in parts:
        if not part:
            continue
        normalized.append(int(part) if part.isdigit() else part)
    return tuple(normalized)


def _prepare_settings_import(data: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    if not isinstance(data, dict):
        raise ValueError('Import payload must be a JSON object.')
    sanitized: Dict[str, Any] = {}
    warnings: List[str] = []
    stream_ids: List[str] = []
    stream_payloads: Dict[str, Dict[str, Any]] = {}
    for key, value in data.items():
        if not isinstance(key, str) or key.startswith('_'):
            continue
        stream_id = key.strip()
        if not stream_id:
            warnings.append('Skipped stream with empty identifier.')
            continue
        stream_payloads[stream_id] = _sanitize_imported_stream_config(stream_id, value)
        stream_ids.append(stream_id)
    ordered_stream_ids: List[str] = []
    seen_stream_ids: Set[str] = set()
    order_raw = data.get(STREAM_ORDER_KEY)
    if isinstance(order_raw, (list, tuple)):
        for entry in order_raw:
            if not isinstance(entry, str):
                continue
            stream_id = entry.strip()
            if stream_id in stream_payloads and stream_id not in seen_stream_ids:
                ordered_stream_ids.append(stream_id)
                seen_stream_ids.add(stream_id)
    remaining_stream_ids = [stream_id for stream_id in stream_ids if stream_id not in seen_stream_ids]
    if ordered_stream_ids:
        ordered_stream_ids.extend(remaining_stream_ids)
    else:
        ordered_stream_ids = sorted(remaining_stream_ids, key=_natural_stream_order_key)
    for stream_id in ordered_stream_ids:
        sanitized[stream_id] = stream_payloads[stream_id]
    sanitized[STREAM_ORDER_KEY] = list(ordered_stream_ids)
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
    sync_timers_raw = data.get(SYNC_TIMERS_KEY)
    sanitized[SYNC_TIMERS_KEY] = sanitize_sync_timers_collection(sync_timers_raw)
    groups_raw = data.get('_groups')
    sanitized['_groups'] = _sanitize_group_collection_for_import(groups_raw, valid_streams)
    for key, value in data.items():
        if not isinstance(key, str) or not key.startswith('_'):
            continue
        if key in {GLOBAL_TAGS_KEY, '_notes', '_ai_defaults', '_groups', AI_PRESETS_KEY, SYNC_TIMERS_KEY, STREAM_ORDER_KEY}:
            continue
        sanitized[key] = deepcopy(value)
    return sanitized, warnings


def build_settings_export_payload() -> Dict[str, Any]:
    snapshot = deepcopy(settings)
    snapshot[STREAM_ORDER_KEY] = get_stream_order_snapshot()
    return snapshot


def get_stream_order_snapshot() -> List[str]:
    stream_ids = [key for key in settings.keys() if isinstance(key, str) and not key.startswith("_")]
    available = set(stream_ids)
    order_raw = settings.get(STREAM_ORDER_KEY)
    ordered: List[str] = []
    seen: Set[str] = set()
    if isinstance(order_raw, (list, tuple)):
        for entry in order_raw:
            if not isinstance(entry, str):
                continue
            stream_id = entry.strip()
            if stream_id in available and stream_id not in seen:
                ordered.append(stream_id)
                seen.add(stream_id)
    if ordered:
        for stream_id in stream_ids:
            if stream_id not in seen:
                ordered.append(stream_id)
        return ordered
    return sorted(stream_ids, key=_natural_stream_order_key)


def get_natural_stream_order_snapshot() -> List[str]:
    stream_ids = [key for key in settings.keys() if isinstance(key, str) and not key.startswith("_")]
    return sorted(stream_ids, key=_natural_stream_order_key)


def iter_ordered_stream_items() -> List[Tuple[str, Dict[str, Any]]]:
    ordered_streams: List[Tuple[str, Dict[str, Any]]] = []
    for stream_id in get_stream_order_snapshot():
        conf = settings.get(stream_id)
        if isinstance(conf, dict):
            ordered_streams.append((stream_id, conf))
    return ordered_streams


def iter_natural_stream_items() -> List[Tuple[str, Dict[str, Any]]]:
    ordered_streams: List[Tuple[str, Dict[str, Any]]] = []
    for stream_id in get_natural_stream_order_snapshot():
        conf = settings.get(stream_id)
        if isinstance(conf, dict):
            ordered_streams.append((stream_id, conf))
    return ordered_streams


def load_settings():
    return SETTINGS_STORE.load()

def save_settings(data):
    try:
        SETTINGS_STORE.save(data)
    except Exception as exc:
        app.logger.error("Failed to save settings: %s", exc)
        raise


def save_settings_debounced():
    """Queue a settings save with a debounce delay to reduce disk I/O."""
    SETTINGS_STORE.save_debounced(lambda: settings)


def flush_settings_save():
    """Immediately perform any pending settings save before shutdown."""
    SETTINGS_STORE.flush_pending(lambda: settings)


atexit.register(flush_settings_save)

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
    save_settings_debounced()

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
        "streams": [stream_id for stream_id in get_stream_order_snapshot() if stream_id in new_streams],
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


register_blueprint_with_legacy_aliases(
    app,
    create_diagnostics_blueprint(
        load_config=load_config,
        get_system_stats=get_system_stats,
    ),
    {
        "debug_page": "diagnostics.debug_page",
        "debug_stream": "diagnostics.debug_stream",
        "debug_download": "diagnostics.debug_download",
        "health": "diagnostics.health",
        "system_stats": "diagnostics.system_stats",
    },
)
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


AI_RUNTIME = build_ai_runtime(
    config=CONFIG,
    default_model=AI_DEFAULT_MODEL,
    default_sampler=AI_DEFAULT_SAMPLER,
    default_width=AI_DEFAULT_WIDTH,
    default_height=AI_DEFAULT_HEIGHT,
    default_steps=AI_DEFAULT_STEPS,
    default_cfg=AI_DEFAULT_CFG,
    default_samples=AI_DEFAULT_SAMPLES,
    output_subdir=AI_OUTPUT_SUBDIR,
    temp_subdir=AI_TEMP_SUBDIR,
    default_persist=AI_DEFAULT_PERSIST,
    poll_interval=AI_POLL_INTERVAL,
    timeout=AI_TIMEOUT,
    primary_ai_root=PRIMARY_AI_MEDIA_ROOT.path,
    ensure_dir=_ensure_dir,
    logger=app.logger,
)
AI_DEFAULT_MODEL = AI_RUNTIME.ai_default_model
AI_DEFAULT_SAMPLER = AI_RUNTIME.ai_default_sampler
AI_DEFAULT_WIDTH = AI_RUNTIME.ai_default_width
AI_DEFAULT_HEIGHT = AI_RUNTIME.ai_default_height
AI_DEFAULT_STEPS = AI_RUNTIME.ai_default_steps
AI_DEFAULT_CFG = AI_RUNTIME.ai_default_cfg
AI_DEFAULT_SAMPLES = AI_RUNTIME.ai_default_samples
AI_OUTPUT_SUBDIR = AI_RUNTIME.ai_output_subdir
AI_TEMP_SUBDIR = AI_RUNTIME.ai_temp_subdir
AI_DEFAULT_PERSIST = AI_RUNTIME.ai_default_persist
AI_POLL_INTERVAL = AI_RUNTIME.ai_poll_interval
AI_TIMEOUT = AI_RUNTIME.ai_timeout
AI_OUTPUT_ROOT = AI_RUNTIME.ai_output_root
AI_TEMP_ROOT = AI_RUNTIME.ai_temp_root
stable_horde_client = AI_RUNTIME.stable_horde_client
ai_jobs_lock = AI_RUNTIME.ai_jobs_lock
ai_jobs = AI_RUNTIME.ai_jobs
ai_job_controls = AI_RUNTIME.ai_job_controls
ai_model_cache = AI_RUNTIME.ai_model_cache
auto_scheduler: Optional['AutoGenerateScheduler'] = None
picsum_scheduler: Optional['PicsumAutoScheduler'] = None


HLS_RUNTIME = build_hls_runtime(
    config=CONFIG,
    cache_factory=lambda maxsize: LRUCache(maxsize=maxsize),
    live_hls_async=LIVE_HLS_ASYNC,
    hls_ttl_secs=HLS_TTL_SECS,
    max_hls_workers=MAX_HLS_WORKERS,
    hls_error_retry_secs=HLS_ERROR_RETRY_SECS,
)
LIVE_HLS_ASYNC = HLS_RUNTIME.live_hls_async
HLS_TTL_SECS = HLS_RUNTIME.hls_ttl_secs
MAX_HLS_WORKERS = HLS_RUNTIME.max_hls_workers
HLS_ERROR_RETRY_SECS = HLS_RUNTIME.hls_error_retry_secs
HLS_METRICS = HLS_RUNTIME.hls_metrics
HLS_LOCK = HLS_RUNTIME.hls_lock
HLS_LOG_PREFIX = HLS_RUNTIME.hls_log_prefix
HLS_CACHE = HLS_RUNTIME.hls_cache
HLS_JOBS = HLS_RUNTIME.hls_jobs
HLS_EXECUTOR = HLS_RUNTIME.hls_executor


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
    AI_EXECUTION_SERVICE.cleanup_temp_outputs(stream_id)

def _emit_ai_update(stream_id: str, state: Dict[str, Any], job: Optional[Dict[str, Any]] = None) -> None:
    AI_EXECUTION_SERVICE.emit_ai_update(stream_id, state, job)


def _update_ai_state(stream_id: str, updates: Dict[str, Any], *, persist: bool = False) -> Dict[str, Any]:
    return AI_EXECUTION_SERVICE.update_ai_state(stream_id, updates, persist=persist)


def _reconcile_stale_ai_state(stream_id: str, conf: Dict[str, Any]) -> bool:
    return AI_EXECUTION_SERVICE.reconcile_stale_ai_state(stream_id, conf)


def _record_job_progress(stream_id: str, stage: str, payload: Dict[str, Any]) -> None:
    AI_EXECUTION_SERVICE.record_job_progress(stream_id, stage, payload)


def _run_ai_generation(
    stream_id: str,
    options: Dict[str, Any],
    cancel_event: Optional[threading.Event] = None,
    manager_id: Optional[str] = None,
) -> None:
    AI_EXECUTION_SERVICE.run_generation(stream_id, options, cancel_event, manager_id)


settings = load_settings()
SETTINGS_INTEGRITY_CHANGED_ON_BOOT = ensure_settings_integrity(settings)
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
    return STREAM_RUNTIME_SERVICE.detect_media_kind(value)


def _infer_media_mode(conf: Dict[str, Any]) -> str:
    return STREAM_RUNTIME_SERVICE.infer_media_mode(conf)


def _get_stream_runtime_state(stream_id: str) -> Dict[str, Any]:
    return STREAM_RUNTIME_SERVICE.get_stream_runtime_state(stream_id)

def _get_runtime_thumbnail_payload(stream_id: str) -> Optional[Dict[str, Any]]:
    """Return the thumbnail metadata for a stream suitable for client payloads."""
    return STREAM_RUNTIME_SERVICE.get_runtime_thumbnail_payload(stream_id)


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
    return STREAM_RUNTIME_SERVICE.update_stream_runtime_state(
        stream_id,
        path=path,
        kind=kind,
        media_mode=media_mode,
        stream_url=stream_url,
        source=source,
        force_thumbnail=force_thumbnail,
    )

def _runtime_timestamp_to_iso(ts: Optional[float]) -> Optional[str]:
    return STREAM_RUNTIME_SERVICE.runtime_timestamp_to_iso(ts)


def _resolve_media_path(rel_path: Optional[str]) -> Optional[Path]:
    if not rel_path:
        return None
    target = _resolve_virtual_media_path(rel_path)
    if target is None or not target.exists():
        return None
    return target

THUMBNAIL_SERVICE = ThumbnailService(
    Image=Image,
    ImageDraw=ImageDraw,
    ImageFont=ImageFont,
    ImageOps=ImageOps,
    cv2_module=cv2,
    requests_module=requests,
    eventlet_module=eventlet,
    logger=logger,
    dashboard_thumbnail_size=DASHBOARD_THUMBNAIL_SIZE,
    image_thumbnail_filter=IMAGE_THUMBNAIL_FILTER,
    thumbnail_jpeg_quality=THUMBNAIL_JPEG_QUALITY,
    settings=settings,
    get_stream_runtime_state=_get_stream_runtime_state,
    detect_media_kind=_detect_media_kind,
    infer_media_mode=_infer_media_mode,
    resolve_media_path=_resolve_media_path,
    ensure_thumbnail_dir=_ensure_thumbnail_dir,
    thumbnail_disk_path=_thumbnail_disk_path,
    thumbnail_public_url=_thumbnail_public_url,
    public_thumbnail_payload=_public_thumbnail_payload,
    runtime_timestamp_to_iso=_runtime_timestamp_to_iso,
    stream_runtime_lock=STREAM_RUNTIME_LOCK,
    stream_runtime_state=STREAM_RUNTIME_STATE,
    safe_emit=safe_emit,
    playback_manager_getter=lambda: playback_manager,
    media_mode_choices=MEDIA_MODE_CHOICES,
    media_mode_livestream=MEDIA_MODE_LIVESTREAM,
    media_mode_video=MEDIA_MODE_VIDEO,
    media_mode_ai=MEDIA_MODE_AI,
    media_mode_picsum=MEDIA_MODE_PICSUM,
)

def _generate_placeholder_thumbnail(label: str) -> Image.Image:
    return THUMBNAIL_SERVICE.generate_placeholder_thumbnail(label)


def _compose_thumbnail(frame: Image.Image) -> Image.Image:
    return THUMBNAIL_SERVICE.compose_thumbnail(frame)


def _create_thumbnail_image(media_path: Path) -> Image.Image:
    return THUMBNAIL_SERVICE.create_thumbnail_image(media_path)


def _create_video_thumbnail(media_path: Path) -> Optional[Image.Image]:
    return THUMBNAIL_SERVICE.create_video_thumbnail(media_path)


def _load_remote_image(url: str) -> Optional[Image.Image]:
    return THUMBNAIL_SERVICE.load_remote_image(url)


def _create_livestream_thumbnail(stream_url: Optional[str]) -> Optional[Image.Image]:
    return THUMBNAIL_SERVICE.create_livestream_thumbnail(stream_url)

def _render_thumbnail_image(snapshot: Dict[str, Any]) -> Tuple[Image.Image, bool]:
    return THUMBNAIL_SERVICE.render_thumbnail_image(snapshot)


def _thumbnail_image_to_bytes(image: Image.Image) -> io.BytesIO:
    return THUMBNAIL_SERVICE.thumbnail_image_to_bytes(image)

def _compute_thumbnail_snapshot(stream_id: str) -> Optional[Dict[str, Any]]:
    return THUMBNAIL_SERVICE.compute_thumbnail_snapshot(stream_id)

def _thumbnail_signature(snapshot: Dict[str, Any]) -> Tuple[Any, ...]:
    return THUMBNAIL_SERVICE.thumbnail_signature(snapshot)


def _refresh_stream_thumbnail(stream_id: str, snapshot: Optional[Dict[str, Any]] = None, *, force: bool = False) -> Optional[Dict[str, Any]]:
    return THUMBNAIL_SERVICE.refresh_stream_thumbnail(stream_id, snapshot, force=force)


STREAM_RUNTIME_SERVICE = StreamRuntimeService(
    stream_runtime_lock=STREAM_RUNTIME_LOCK,
    stream_runtime_state=STREAM_RUNTIME_STATE,
    video_extensions=VIDEO_EXTENSIONS,
    ai_mode=AI_MODE,
    ai_generate_mode=AI_GENERATE_MODE,
    ai_random_mode=AI_RANDOM_MODE,
    ai_specific_mode=AI_SPECIFIC_MODE,
    media_mode_ai=MEDIA_MODE_AI,
    media_mode_livestream=MEDIA_MODE_LIVESTREAM,
    media_mode_picsum=MEDIA_MODE_PICSUM,
    media_mode_video=MEDIA_MODE_VIDEO,
    media_mode_image=MEDIA_MODE_IMAGE,
    refresh_stream_thumbnail=_refresh_stream_thumbnail,
)

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
        ensure_sync_defaults(v)
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
            current_mode_raw = v.get("mode")
            current_mode = current_mode_raw.strip().lower() if isinstance(current_mode_raw, str) else ""
            if current_mode in {AI_RANDOM_MODE, AI_SPECIFIC_MODE, AI_GENERATE_MODE}:
                desired_mode = current_mode
            else:
                desired_mode = AI_GENERATE_MODE
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


def get_subfolders(hide_nsfw: bool = False, *, library: str = MEDIA_LIBRARY_DEFAULT) -> List[str]:
    return MEDIA_CATALOG_SERVICE.get_subfolders(hide_nsfw=hide_nsfw, library=library)


def get_folder_inventory(hide_nsfw: bool = False, *, library: str = MEDIA_LIBRARY_DEFAULT) -> List[Dict[str, Any]]:
    return MEDIA_CATALOG_SERVICE.get_folder_inventory(hide_nsfw=hide_nsfw, library=library)


_media_blueprint = create_media_blueprint(
    media_manager=MEDIA_MANAGER,
    media_error_type=MediaManagerError,
    media_thumb_width=MEDIA_THUMB_WIDTH,
    media_allow_edit=MEDIA_MANAGEMENT_ALLOW_EDIT,
    media_allowed_exts=MEDIA_ALLOWED_EXTS,
    media_upload_max_mb=MEDIA_UPLOAD_MAX_MB,
    media_upload_max_mb_getter=lambda: max(1, _as_int(load_config().get("MEDIA_UPLOAD_MAX_MB"), MEDIA_UPLOAD_MAX_MB)),
    media_preview_enabled=MEDIA_PREVIEW_ENABLED,
    media_preview_frames=MEDIA_PREVIEW_FRAMES,
    media_preview_width=MEDIA_PREVIEW_WIDTH,
    media_preview_max_duration=MEDIA_PREVIEW_MAX_DURATION,
    available_media_roots=AVAILABLE_MEDIA_ROOTS,
    media_library_default=MEDIA_LIBRARY_DEFAULT,
    media_error_response=_media_error_response,
    require_media_edit=_require_media_edit,
    invalidate_media_cache=_invalidate_media_cache,
    parse_truthy=_parse_truthy,
    normalize_library_key=_normalize_library_key,
    get_folder_inventory=get_folder_inventory,
    as_int=_as_int,
    virtual_leaf=_virtual_leaf,
    logger=logger,
    app_response_class=app.response_class,
)
register_blueprint_with_legacy_aliases(app, _media_blueprint, {
    "api_media_list": "media_routes.api_media_list",
    "api_media_create_folder": "media_routes.api_media_create_folder",
    "api_media_rename": "media_routes.api_media_rename",
    "api_media_delete": "media_routes.api_media_delete",
    "api_media_upload": "media_routes.api_media_upload",
    "api_media_thumbnail": "media_routes.api_media_thumbnail",
    "api_media_preview_frame": "media_routes.api_media_preview_frame",
    "folders_collection": "media_routes.folders_collection",
    "media_management_page": "media_routes.media_management_page",
})


def list_images(folder="all", hide_nsfw: bool = False, *, library: str = MEDIA_LIBRARY_DEFAULT):
    """Return cached image paths for the folder, refreshing when necessary."""
    return MEDIA_CATALOG_SERVICE.list_images(folder, hide_nsfw=hide_nsfw, library=library)

def list_media(folder="all", hide_nsfw: bool = False, *, library: str = MEDIA_LIBRARY_DEFAULT) -> List[Dict[str, Any]]:
    """Return cached media entries (images and videos) for the folder."""
    return MEDIA_CATALOG_SERVICE.list_media(folder, hide_nsfw=hide_nsfw, library=library)

if playback_manager is None:
    playback_manager = StreamPlaybackManager(
        safe_emit=safe_emit,
        list_media=list_media,
        library_for_media_mode=_library_for_media_mode,
        update_stream_runtime_state=_update_stream_runtime_state,
        get_runtime_thumbnail_payload=_get_runtime_thumbnail_payload,
        get_sync_timer_config=get_sync_timer_config,
        compute_next_sync_tick=compute_next_sync_tick,
        coerce_float=_coerce_float,
        infer_media_mode=_infer_media_mode,
        resolve_media_path=_resolve_media_path,
        video_duration_cache=_VIDEO_DURATION_CACHE,
        cv2_module=cv2,
        media_mode_choices=MEDIA_MODE_CHOICES,
        media_mode_image=MEDIA_MODE_IMAGE,
        media_mode_video=MEDIA_MODE_VIDEO,
        media_mode_ai=MEDIA_MODE_AI,
        ai_random_mode=AI_RANDOM_MODE,
        video_playback_modes=VIDEO_PLAYBACK_MODES,
        sync_config_key=SYNC_CONFIG_KEY,
        sync_supported_media_modes=SYNC_SUPPORTED_MEDIA_MODES,
        stream_playback_history_limit=STREAM_PLAYBACK_HISTORY_LIMIT,
        stream_update_event=STREAM_UPDATE_EVENT,
        sync_time_event=SYNC_TIME_EVENT,
        stream_sync_interval_seconds=STREAM_SYNC_INTERVAL_SECONDS,
        sync_switch_lead_seconds=SYNC_SWITCH_LEAD_SECONDS,
    )
    PLAYBACK_RUNTIME.playback_manager = playback_manager


def _hls_url_fingerprint(original_url: str) -> str:
    return LIVE_HLS_SERVICE.hls_url_fingerprint(original_url)


def _log_hls_event(event: str, stream_id: str, original_url: str, **extra: Any) -> None:
    LIVE_HLS_SERVICE.log_hls_event(event, stream_id, original_url, **extra)


def _record_hls_metric(name: str, delta: int = 1) -> None:
    LIVE_HLS_SERVICE.record_hls_metric(name, delta)


def _live_hls_cache_key(stream_id: Optional[str], original_url: str) -> str:
    return LIVE_HLS_SERVICE.live_hls_cache_key(stream_id, original_url)


def _is_manifest_url(candidate: Optional[str]) -> bool:
    return LIVE_HLS_SERVICE.is_manifest_url(candidate)


def _extract_hls_candidate(info: Dict[str, Any]) -> Optional[str]:
    return LIVE_HLS_SERVICE.extract_hls_candidate(info)


def _detect_hls_stream_url(original_url: str) -> Optional[str]:
    return LIVE_HLS_SERVICE.detect_hls_stream_url(original_url)


def _get_hls_cache_entry(key: str) -> Optional[HLSCacheEntry]:
    return LIVE_HLS_SERVICE.get_hls_cache_entry(key)


def _cancel_hls_job(key: str) -> bool:
    return LIVE_HLS_SERVICE.cancel_hls_job(key)


def schedule_hls_detection(stream_id: str, original_url: str) -> None:
    LIVE_HLS_SERVICE.schedule_hls_detection(stream_id, original_url)


def _run_hls_detection_job(key: str, stream_id: str, original_url: str) -> None:
    LIVE_HLS_SERVICE._run_hls_detection_job(key, stream_id, original_url)


def try_get_hls(original_url):
    """Legacy synchronous helper retained for compatibility."""
    return LIVE_HLS_SERVICE.try_get_hls(original_url)


def dashboard():
    folder_inventory = get_folder_inventory(library=MEDIA_LIBRARY_DEFAULT)
    ai_folder_inventory = get_folder_inventory(library=AI_MEDIA_LIBRARY)
    subfolders = [item["name"] for item in folder_inventory]
    streams = dict(iter_natural_stream_items())
    for stream_id, conf in streams.items():
        if not isinstance(conf, dict):
            continue
        quality = conf.get("image_quality")
        if not isinstance(quality, str) or quality.strip().lower() not in IMAGE_QUALITY_CHOICES:
            conf["image_quality"] = "auto"
        else:
            conf["image_quality"] = quality.strip().lower()
        ensure_timer_defaults(conf)
        ensure_sync_defaults(conf)
        ensure_background_defaults(conf)
        ensure_tag_defaults(conf)
        _refresh_embed_metadata(stream_id, conf)
    groups = sorted(list(settings.get("_groups", {}).keys()))
    sync_timers = get_sync_timers_snapshot()
    return render_template(
        "dashboard/index.html",
        subfolders=subfolders,
        folder_inventory=folder_inventory,
        ai_folder_inventory=ai_folder_inventory,
        stream_settings=streams,
        custom_stream_order=get_stream_order_snapshot(),
        groups=groups,
        global_tags=get_global_tags(),
        sync_timers=sync_timers,
        post_processors=STABLE_HORDE_POST_PROCESSORS,
        max_loras=STABLE_HORDE_MAX_LORAS,
        clip_skip_range=STABLE_HORDE_CLIP_SKIP_RANGE,
        strength_range=STABLE_HORDE_STRENGTH_RANGE,
        denoise_range=STABLE_HORDE_DENOISE_RANGE,
    )


def mosaic_streams():
    # Dynamic global view: include all streams ("online" assumed as configured)
    streams = dict(iter_natural_stream_items())
    for stream_id, conf in streams.items():
        if not isinstance(conf, dict):
            continue
        ensure_timer_defaults(conf)
        ensure_sync_defaults(conf)
        ensure_background_defaults(conf)
        ensure_tag_defaults(conf)
        _refresh_embed_metadata(stream_id, conf)
    return render_template("streams/streams.html", stream_settings=streams)

def _slugify(name: str) -> str:
    name = (name or "").strip().lower()
    name = re.sub(r"[^a-z0-9]+", "-", name)
    name = re.sub(r"-+", "-", name).strip("-")
    return name or ""

@app.template_filter('slugify')
def jinja_slugify(s):
    return _slugify(s)

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
    ensure_timer_defaults(conf)
    ensure_background_defaults(conf)
    ensure_tag_defaults(conf)
    _refresh_embed_metadata(key, conf)
    images = list_images(
        conf.get("folder", "all"),
        hide_nsfw=conf.get("hide_nsfw", False),
        library=_library_for_media_mode(conf.get("media_mode")),
    )
    requested_quality = (request.args.get("size") or "").strip().lower()
    if requested_quality and requested_quality not in IMAGE_QUALITY_CHOICES:
        requested_quality = ""
    default_quality = requested_quality or config_quality
    return render_template(
        "streams/single_stream.html",
        stream_id=key,
        config=conf,
        images=images,
        default_quality=default_quality,
    )


def add_stream():
    """Create a new stream configuration and return its ID."""
    return jsonify({"stream_id": STREAM_CONFIG_SERVICE.create_stream()})


def delete_stream(stream_id):
    if STREAM_CONFIG_SERVICE.delete_stream(stream_id):
        return jsonify({"status": "deleted"})
    return jsonify({"error": "not found"}), 404


def reorder_streams():
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        return jsonify({"error": "Invalid payload"}), 400
    order = payload.get("stream_order")
    if not isinstance(order, list):
        return jsonify({"error": "stream_order must be a list"}), 400
    updated_order = STREAM_CONFIG_SERVICE.reorder_streams(order)
    return jsonify({"status": "ok", "stream_order": updated_order})


def get_stream_settings(stream_id):
    try:
        return jsonify(STREAM_CONFIG_SERVICE.get_stream_settings_payload(stream_id))
    except KeyError:
        return jsonify({"error": f"No stream '{stream_id}' found"}), 404


def get_stream_playback_state(stream_id):
    try:
        return jsonify(PLAYBACK_SERVICE.get_stream_playback_state_payload(stream_id))
    except RuntimeError:
        return jsonify({"error": "Playback manager unavailable"}), 503
    except KeyError:
        return jsonify({"error": f"No stream '{stream_id}' found"}), 404
    except LookupError:
        return jsonify({"error": "No playback state"}), 404


def update_stream_settings(stream_id):
    try:
        return jsonify(STREAM_CONFIG_SERVICE.update_stream(stream_id, request.get_json(silent=True) or {}))
    except KeyError:
        return jsonify({"error": f"No stream '{stream_id}' found"}), 404
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


def update_stream_timer(stream_id: str):
    conf = settings.get(stream_id)
    if not isinstance(conf, dict):
        return jsonify({"error": f"No stream '{stream_id}' found"}), 404

    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        return jsonify({"error": "Invalid payload"}), 400

    ensure_ai_defaults(conf)
    ensure_picsum_defaults(conf)
    ensure_timer_defaults(conf)

    timer_mode = _canonical_timer_mode(conf)
    if timer_mode not in TIMER_SUPPORTED_MODES:
        return jsonify({"error": "Timer controls are not available for this stream."}), 400

    timer = TimerManager(
        mode=timer_mode or conf.get("mode") or "",
        stream_id=stream_id,
        config_owner=conf,
        snap_provider=_timer_snap_enabled,
    )
    allowed_updates = {key: payload[key] for key in ("enabled", "interval", "offset") if key in payload}
    updated_config = timer.apply_updates(allowed_updates)

    if updated_config["enabled"]:
        initial_next = timer.compute_next()
        timer.update_next(initial_next)
    else:
        timer.update_next(None)

    scheduler_ts = time.time()
    if timer_mode in {AI_MODE, AI_GENERATE_MODE} and auto_scheduler is not None:
        auto_scheduler.reschedule(stream_id, base_time=scheduler_ts)
    elif timer_mode == MEDIA_MODE_PICSUM and picsum_scheduler is not None:
        picsum_scheduler.reschedule(stream_id, base_time=scheduler_ts)

    refreshed_config = timer.refresh()
    next_label = refreshed_config.get("next_run")
    last_label = refreshed_config.get("last_run")

    if timer_mode in {AI_MODE, AI_GENERATE_MODE}:
        ai_state = conf[AI_STATE_KEY]
        ai_state["next_auto_trigger"] = next_label
        if last_label is not None:
            ai_state["last_auto_trigger"] = last_label
    elif timer_mode == MEDIA_MODE_PICSUM:
        picsum_state = conf.get(PICSUM_SETTINGS_KEY) or {}
        picsum_state["next_auto_trigger"] = next_label
        if last_label is not None:
            picsum_state["last_auto_trigger"] = last_label
        conf[PICSUM_SETTINGS_KEY] = picsum_state

    save_settings_debounced()
    return jsonify({
        "status": "success",
        "timer": refreshed_config,
        "mode": timer_mode,
        "next_auto_trigger": next_label,
        "last_auto_trigger": last_label,
    })


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

    save_settings_debounced()

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


OPERATIONS_SERVICE = OperationsService(
    load_config=load_config,
    backup_dirname=BACKUP_DIRNAME,
    restore_point_dirname=RESTORE_POINT_DIRNAME,
    restore_point_metadata_file=RESTORE_POINT_METADATA_FILE,
    max_restore_points=MAX_RESTORE_POINTS,
    settings_file=SETTINGS_FILE,
    config_file=CONFIG_FILE,
    set_update_job_active=_set_update_job_active,
    run_update_job=_run_update_job,
    logger=logger,
)


def dashboard_update_status():
    cfg = load_config()
    force_refresh = str(request.args.get("refresh") or "").strip().lower() in {"1", "true", "yes", "on"}
    info = OPERATIONS_SERVICE.read_update_info(cfg, force_refresh=force_refresh)
    return jsonify(
        {
            "channel": info.get("channel"),
            "branch": info.get("branch"),
            "installed_version": info.get("installed_version"),
            "latest_version": info.get("latest_version"),
            "update_available": bool(info.get("update_available")),
            "release_check_ok": info.get("release_check_ok", True),
        }
    )


_dashboard_blueprint = create_dashboard_blueprint(
    dashboard_handler=dashboard,
    update_status_handler=dashboard_update_status,
    mosaic_streams_handler=mosaic_streams,
    render_stream_handler=render_stream,
    add_stream_handler=add_stream,
    delete_stream_handler=delete_stream,
    get_stream_settings_handler=get_stream_settings,
    get_stream_playback_state_handler=get_stream_playback_state,
    update_stream_settings_handler=update_stream_settings,
    update_stream_timer_handler=update_stream_timer,
    refresh_picsum_image_handler=refresh_picsum_image,
    reorder_streams_handler=reorder_streams,
)
register_blueprint_with_legacy_aliases(app, _dashboard_blueprint, {
    "dashboard": "dashboard_routes.dashboard",
    "mosaic_streams": "dashboard_routes.mosaic_streams",
    "render_stream": "dashboard_routes.render_stream",
    "add_stream": "dashboard_routes.add_stream",
    "delete_stream": "dashboard_routes.delete_stream",
    "get_stream_settings": "dashboard_routes.get_stream_settings",
    "get_stream_playback_state": "dashboard_routes.get_stream_playback_state",
    "update_stream_settings": "dashboard_routes.update_stream_settings",
    "update_stream_timer": "dashboard_routes.update_stream_timer",
    "refresh_picsum_image": "dashboard_routes.refresh_picsum_image",
    "reorder_streams": "dashboard_routes.reorder_streams",
})


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

    save_settings_debounced()
    return jsonify({"status": "success", "defaults": new_defaults})


_settings_operations_blueprint = create_settings_operations_blueprint(
    settings=settings,
    load_config=load_config,
    media_settings_handler=lambda: api_media_settings(),
    default_ai_settings=default_ai_settings,
    ai_fallback_defaults=AI_FALLBACK_DEFAULTS,
    post_processors=STABLE_HORDE_POST_PROCESSORS,
    max_loras=STABLE_HORDE_MAX_LORAS,
    settings_export_payload=build_settings_export_payload,
    import_settings_handler=import_settings,
    update_ai_defaults_handler=update_ai_defaults,
    operations_service=OPERATIONS_SERVICE,
    logger=logger,
)
register_blueprint_with_legacy_aliases(app, _settings_operations_blueprint, {
    "export_settings_download": "settings_operations.export_settings_download",
    "import_settings": "settings_operations.import_settings",
    "get_ai_defaults": "settings_operations.get_ai_defaults",
    "update_ai_defaults": "settings_operations.update_ai_defaults",
    "app_settings": "settings_operations.app_settings",
    "restore_points_collection": "settings_operations.restore_points_collection",
    "restore_points_delete": "settings_operations.restore_points_delete",
    "restore_points_restore": "settings_operations.restore_points_restore",
    "update_app": "settings_operations.update_app",
    "update_info": "settings_operations.update_info",
    "update_history": "settings_operations.update_history",
    "rollback_app": "settings_operations.rollback_app",
    "update_view": "settings_operations.update_view",
})

def list_ai_presets():
    presets = ensure_ai_presets_storage()
    payload = [
        {"name": name, "settings": deepcopy(config)}
        for name, config in presets.items()
    ]
    return jsonify({"presets": payload})


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
    save_settings_debounced()
    return jsonify({"status": "saved", "preset": {"name": name, "settings": deepcopy(sanitized)}})


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
    save_settings_debounced()
    return jsonify({"status": "updated", "preset": {"name": target_name, "settings": deepcopy(sanitized)}})


def delete_ai_preset(preset_name: str):
    name = (preset_name or "").strip()
    presets = ensure_ai_presets_storage()
    if name not in presets:
        return jsonify({"error": "Preset not found"}), 404
    updated = dict(presets)
    updated.pop(name, None)
    settings[AI_PRESETS_KEY] = _sorted_presets(updated)
    save_settings_debounced()
    return jsonify({"status": "deleted"})

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


_ai_blueprint = create_ai_blueprint(
    list_ai_presets_handler=list_ai_presets,
    create_ai_preset_handler=create_ai_preset,
    update_ai_preset_handler=update_ai_preset,
    delete_ai_preset_handler=delete_ai_preset,
    ai_loras_handler=ai_loras,
    list_ai_models_handler=lambda: list_ai_models(),
    ai_status_handler=lambda stream_id: ai_status(stream_id),
    latest_job_handler=lambda stream_id: latest_job(stream_id),
    ai_generate_handler=lambda stream_id: ai_generate(stream_id),
    ai_cancel_handler=lambda stream_id: ai_cancel(stream_id),
)
register_blueprint_with_legacy_aliases(app, _ai_blueprint, {
    "list_ai_presets": "ai_routes.list_ai_presets",
    "create_ai_preset": "ai_routes.create_ai_preset",
    "update_ai_preset": "ai_routes.update_ai_preset",
    "delete_ai_preset": "ai_routes.delete_ai_preset",
    "ai_loras": "ai_routes.ai_loras",
    "list_ai_models": "ai_routes.list_ai_models",
    "ai_status": "ai_routes.ai_status",
    "latest_job": "ai_routes.latest_job",
    "ai_generate": "ai_routes.ai_generate",
    "ai_cancel": "ai_routes.ai_cancel",
})

def list_tags():
    return jsonify({"tags": get_global_tags()})


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
        save_settings_debounced()
        canonical = normalized
    else:
        canonical = existing
    return jsonify({"tag": canonical, "tags": tags})


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

    # Create a snapshot to avoid concurrent modification issues
    current_settings = list(settings.items())
    for sid, conf in current_settings:
        if sid.startswith("_") or not isinstance(conf, dict):
            continue
        stream_tags = conf.get(TAG_KEY) or []
        for tag in stream_tags:
            if isinstance(tag, str) and tag.casefold() == canon_key:
                return jsonify({"error": "Tag is in use"}), 409

    tags.pop(idx)
    save_settings_debounced()
    return jsonify({"status": "deleted", "tags": tags})


AI_EXECUTION_SERVICE = AIExecutionService(
    settings=settings,
    ai_jobs=ai_jobs,
    ai_job_controls=ai_job_controls,
    ai_jobs_lock=ai_jobs_lock,
    job_manager=job_manager,
    stable_horde_client=stable_horde_client,
    logger=logger,
    ensure_ai_defaults=ensure_ai_defaults,
    ensure_picsum_defaults=ensure_picsum_defaults,
    sanitize_ai_settings=_sanitize_ai_settings,
    ai_settings_match_defaults=_ai_settings_match_defaults,
    save_settings_debounced=save_settings_debounced,
    safe_emit=safe_emit,
    emit_ai_update_callback=_emit_ai_update,
    update_stream_runtime_state=_update_stream_runtime_state,
    invalidate_media_cache=_invalidate_media_cache,
    relative_image_path=_relative_image_path,
    ensure_dir=_ensure_dir,
    playback_manager=playback_manager,
    ai_output_root=AI_OUTPUT_ROOT,
    ai_temp_root=AI_TEMP_ROOT,
    ai_default_persist=AI_DEFAULT_PERSIST,
    ai_default_sampler=AI_DEFAULT_SAMPLER,
    ai_default_width=AI_DEFAULT_WIDTH,
    ai_default_height=AI_DEFAULT_HEIGHT,
    ai_default_steps=AI_DEFAULT_STEPS,
    ai_default_cfg=AI_DEFAULT_CFG,
    ai_default_samples=AI_DEFAULT_SAMPLES,
    ai_poll_interval=AI_POLL_INTERVAL,
    ai_timeout=AI_TIMEOUT,
    stable_horde_post_processors=STABLE_HORDE_POST_PROCESSORS,
    stable_horde_max_loras=STABLE_HORDE_MAX_LORAS,
    ai_settings_key=AI_SETTINGS_KEY,
    ai_state_key=AI_STATE_KEY,
    ai_generate_mode=AI_GENERATE_MODE,
    ai_random_mode=AI_RANDOM_MODE,
    ai_specific_mode=AI_SPECIFIC_MODE,
    media_mode_ai=MEDIA_MODE_AI,
    ai_media_library=AI_MEDIA_LIBRARY,
    stable_horde_cancelled_cls=StableHordeCancelled,
    stable_horde_error_cls=StableHordeError,
)

AI_ORCHESTRATION_SERVICE = AIOrchestrationService(
    settings=settings,
    stable_horde_client=stable_horde_client,
    ai_model_cache=ai_model_cache,
    ai_jobs=ai_jobs,
    ai_job_controls=ai_job_controls,
    ai_jobs_lock=ai_jobs_lock,
    job_manager=job_manager,
    ensure_ai_defaults=ensure_ai_defaults,
    ensure_picsum_defaults=ensure_picsum_defaults,
    sanitize_ai_settings=_sanitize_ai_settings,
    ai_settings_match_defaults=_ai_settings_match_defaults,
    default_ai_state=default_ai_state,
    cleanup_temp_outputs=_cleanup_temp_outputs,
    save_settings_debounced=save_settings_debounced,
    emit_ai_update=_emit_ai_update,
    update_ai_state=_update_ai_state,
    reconcile_stale_ai_state=_reconcile_stale_ai_state,
    safe_emit=safe_emit,
    get_global_tags=get_global_tags,
    run_ai_generation=_run_ai_generation,
    logger=logger,
    stable_horde_error_cls=StableHordeError,
    format_auto_trigger=_format_timer_label,
    ai_settings_key=AI_SETTINGS_KEY,
    ai_state_key=AI_STATE_KEY,
    ai_default_persist=AI_DEFAULT_PERSIST,
    ai_generate_mode=AI_GENERATE_MODE,
    media_mode_ai=MEDIA_MODE_AI,
    auto_generation_error_cls=AutoGenerationError,
    auto_generation_unavailable_cls=AutoGenerationUnavailable,
    auto_generation_busy_cls=AutoGenerationBusy,
    auto_generation_prompt_missing_cls=AutoGenerationPromptMissing,
)


def _queue_ai_generation(stream_id: str, ai_settings: Dict[str, Any], *, trigger_source: str = "manual") -> Dict[str, Any]:
    return AI_ORCHESTRATION_SERVICE.queue_generation(stream_id, ai_settings, trigger_source=trigger_source)


if auto_scheduler is None or picsum_scheduler is None:
    auto_scheduler, picsum_scheduler = build_auto_schedulers(
        settings=settings,
        timer_manager_cls=TimerManager,
        canonical_timer_mode=_canonical_timer_mode,
        timer_snap_enabled=_timer_snap_enabled,
        ensure_ai_defaults=ensure_ai_defaults,
        ensure_picsum_defaults=ensure_picsum_defaults,
        ensure_timer_defaults=ensure_timer_defaults,
        queue_ai_generation=_queue_ai_generation,
        emit_ai_update=_emit_ai_update,
        refresh_picsum_stream=_refresh_picsum_stream,
        broadcast_picsum_update=_broadcast_picsum_update,
        save_settings_debounced=save_settings_debounced,
        format_timer_label=_format_timer_label,
        normalize_timer_label=_normalize_timer_label,
        log_timer_schedule=_log_timer_schedule,
        ai_state_key=AI_STATE_KEY,
        ai_settings_key=AI_SETTINGS_KEY,
        picsum_settings_key=PICSUM_SETTINGS_KEY,
        ai_modes={AI_MODE, AI_GENERATE_MODE},
        picsum_mode=MEDIA_MODE_PICSUM,
        auto_generation_busy_cls=AutoGenerationBusy,
        auto_generation_prompt_missing_cls=AutoGenerationPromptMissing,
        auto_generation_unavailable_cls=AutoGenerationUnavailable,
        auto_generation_error_cls=AutoGenerationError,
    )
    atexit.register(auto_scheduler.stop)
    atexit.register(picsum_scheduler.stop)


TIMER_SYNC_SERVICE = TimerSyncService(
    settings=settings,
    config=CONFIG,
    auto_scheduler=auto_scheduler,
    picsum_scheduler=picsum_scheduler,
    playback_manager=playback_manager,
    ensure_ai_defaults=ensure_ai_defaults,
    ensure_picsum_defaults=ensure_picsum_defaults,
    ensure_sync_defaults=ensure_sync_defaults,
    save_settings_debounced=save_settings_debounced,
    safe_emit=safe_emit,
    get_global_tags=get_global_tags,
    sanitize_sync_timer_entry=_sanitize_sync_timer_entry,
    ai_state_key=AI_STATE_KEY,
    picsum_settings_key=PICSUM_SETTINGS_KEY,
    sync_timers_key=SYNC_TIMERS_KEY,
    sync_config_key=SYNC_CONFIG_KEY,
    sync_timer_default_interval=SYNC_TIMER_DEFAULT_INTERVAL,
)

GROUP_SERVICE = GroupService(
    settings=settings,
    save_settings_debounced=save_settings_debounced,
    safe_emit=safe_emit,
    maybe_int=_maybe_int,
    clamp=_clamp,
)

STREAM_CONFIG_SERVICE = StreamConfigService(
    settings=settings,
    ai_jobs=ai_jobs,
    ai_job_controls=ai_job_controls,
    ai_jobs_lock=ai_jobs_lock,
    playback_manager=playback_manager,
    auto_scheduler=auto_scheduler,
    picsum_scheduler=picsum_scheduler,
    cleanup_temp_outputs=_cleanup_temp_outputs,
    save_settings_debounced=save_settings_debounced,
    safe_emit=safe_emit,
    get_global_tags=get_global_tags,
    ensure_ai_defaults=ensure_ai_defaults,
    ensure_picsum_defaults=ensure_picsum_defaults,
    ensure_timer_defaults=ensure_timer_defaults,
    ensure_sync_defaults=ensure_sync_defaults,
    ensure_background_defaults=ensure_background_defaults,
    ensure_tag_defaults=ensure_tag_defaults,
    reconcile_stale_ai_state=_reconcile_stale_ai_state,
    update_stream_runtime_state=_update_stream_runtime_state,
    refresh_embed_metadata=_refresh_embed_metadata,
    sanitize_picsum_settings=_sanitize_picsum_settings,
    default_picsum_settings=default_picsum_settings,
    sanitize_sync_config=sanitize_sync_config,
    sanitize_ai_settings=_sanitize_ai_settings,
    ai_settings_match_defaults=_ai_settings_match_defaults,
    default_ai_settings=default_ai_settings,
    default_ai_state=default_ai_state,
    default_stream_config=default_stream_config,
    detect_media_kind=_detect_media_kind,
    infer_media_mode=_infer_media_mode,
    coerce_bool=_coerce_bool,
    coerce_int=_coerce_int,
    slugify=_slugify,
    sanitize_stream_tags=_sanitize_stream_tags,
    register_global_tags=register_global_tags,
    media_mode_choices=MEDIA_MODE_CHOICES,
    media_mode_variants=MEDIA_MODE_VARIANTS,
    media_mode_ai=MEDIA_MODE_AI,
    media_mode_livestream=MEDIA_MODE_LIVESTREAM,
    media_mode_video=MEDIA_MODE_VIDEO,
    ai_modes={AI_MODE, AI_GENERATE_MODE, AI_RANDOM_MODE, AI_SPECIFIC_MODE},
    ai_generate_mode=AI_GENERATE_MODE,
    image_quality_choices=IMAGE_QUALITY_CHOICES,
    video_playback_modes=VIDEO_PLAYBACK_MODES,
    tag_key=TAG_KEY,
    picsum_settings_key=PICSUM_SETTINGS_KEY,
    sync_config_key=SYNC_CONFIG_KEY,
    sync_timers_key=SYNC_TIMERS_KEY,
    ai_settings_key=AI_SETTINGS_KEY,
    ai_state_key=AI_STATE_KEY,
    stream_order_key=STREAM_ORDER_KEY,
    stream_runtime_lock=STREAM_RUNTIME_LOCK,
    stream_runtime_state=STREAM_RUNTIME_STATE,
)

PLAYBACK_SERVICE = PlaybackService(
    settings=settings,
    playback_manager=playback_manager,
    ensure_sync_defaults=ensure_sync_defaults,
    safe_emit=safe_emit,
)

MEDIA_CATALOG_SERVICE = MediaCatalogService(
    image_cache=IMAGE_CACHE,
    image_cache_lock=IMAGE_CACHE_LOCK,
    logger=logger,
    normalize_library_key=_normalize_library_key,
    split_virtual_media_path=_split_virtual_media_path,
    build_virtual_media_path=_build_virtual_media_path,
    resolve_virtual_media_path=_resolve_virtual_media_path,
    library_roots=_library_roots,
    path_contains_nsfw=_path_contains_nsfw,
    should_ignore_media_name=_should_ignore_media_name,
    media_root_lookup=MEDIA_ROOT_LOOKUP,
    media_extensions=MEDIA_EXTENSIONS,
    video_extensions=VIDEO_EXTENSIONS,
    media_library_default=MEDIA_LIBRARY_DEFAULT,
    ai_media_library=AI_MEDIA_LIBRARY,
)
initialize_image_cache()

if PLAYBACK_RUNTIME.playback_manager is playback_manager:
    playback_manager.bootstrap(settings)
    atexit.register(playback_manager.stop)

MEDIA_LIBRARY_SERVICE = MediaLibraryService(
    settings=settings,
    parse_truthy=_parse_truthy,
    normalize_library_key=_normalize_library_key,
    list_images=list_images,
    list_media=list_media,
    infer_media_mode=_infer_media_mode,
    update_stream_runtime_state=_update_stream_runtime_state,
    media_library_default=MEDIA_LIBRARY_DEFAULT,
)

LIVE_HLS_SERVICE = LiveHLSService(
    live_hls_async=LIVE_HLS_ASYNC,
    hls_ttl_secs=HLS_TTL_SECS,
    hls_error_retry_secs=HLS_ERROR_RETRY_SECS,
    hls_metrics=HLS_METRICS,
    hls_lock=HLS_LOCK,
    hls_log_prefix=HLS_LOG_PREFIX,
    hls_executor=HLS_EXECUTOR,
    hls_cache=HLS_CACHE,
    hls_jobs=HLS_JOBS,
    youtube_dl_cls=YoutubeDL,
    logger=logger,
    app_context_factory=app.app_context,
    safe_emit=safe_emit,
)

ASSET_DELIVERY_SERVICE = AssetDeliveryService(
    send_file=send_file,
    jsonify=jsonify,
    url_for=url_for,
    request_args_get=lambda key, default=None: request.args.get(key, default),
    parse_truthy=_parse_truthy,
    as_int=_as_int,
    generate_etag=generate_etag,
    logger=logger,
    bad_media_log_cache=_BAD_MEDIA_LOG_CACHE,
    bad_media_log_ttl=BAD_MEDIA_LOG_TTL,
    image_extensions=IMAGE_EXTENSIONS,
    video_extensions=VIDEO_EXTENSIONS,
    max_image_dimension=MAX_IMAGE_DIMENSION,
    thumbnail_size_presets=THUMBNAIL_SIZE_PRESETS,
    thumbnail_subdir=THUMBNAIL_SUBDIR,
    thumbnail_jpeg_quality=THUMBNAIL_JPEG_QUALITY,
    image_thumbnail_filter=IMAGE_THUMBNAIL_FILTER,
    image_cache_timeout=IMAGE_CACHE_TIMEOUT,
    image_cache_control_max_age=IMAGE_CACHE_CONTROL_MAX_AGE,
    media_root_lookup=MEDIA_ROOT_LOOKUP,
    split_virtual_media_path=_split_virtual_media_path,
    resolve_virtual_media_path=_resolve_virtual_media_path,
    ensure_thumbnail_dir=_ensure_thumbnail_dir,
    thumbnail_disk_path=_thumbnail_disk_path,
    thumbnail_public_url=_thumbnail_public_url,
    public_thumbnail_payload=_public_thumbnail_payload,
    compute_thumbnail_snapshot=_compute_thumbnail_snapshot,
    refresh_stream_thumbnail=_refresh_stream_thumbnail,
    get_runtime_thumbnail_payload=_get_runtime_thumbnail_payload,
    runtime_timestamp_to_iso=_runtime_timestamp_to_iso,
    render_thumbnail_image=_render_thumbnail_image,
    thumbnail_image_to_bytes=_thumbnail_image_to_bytes,
    resized_image_locks=RESIZED_IMAGE_LOCKS,
    resized_image_locks_guard=RESIZED_IMAGE_LOCKS_GUARD,
    stream_runtime_lock=STREAM_RUNTIME_LOCK,
    stream_runtime_state=STREAM_RUNTIME_STATE,
    Image=Image,
    ImageOps=ImageOps,
)


def list_ai_models():
    try:
        return jsonify(AI_ORCHESTRATION_SERVICE.list_models_payload())
    except AutoGenerationUnavailable as exc:
        return jsonify({'error': str(exc)}), 503
    except StableHordeError as exc:
        return jsonify({'error': str(exc)}), 502


def ai_status(stream_id: str):
    try:
        return jsonify(AI_ORCHESTRATION_SERVICE.status_payload(stream_id))
    except AutoGenerationError as exc:
        return jsonify({'error': str(exc)}), 404


def latest_job(stream_id: str):
    return jsonify(AI_ORCHESTRATION_SERVICE.latest_job_payload(stream_id))


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


def ai_cancel(stream_id: str):
    try:
        return jsonify(AI_ORCHESTRATION_SERVICE.cancel_generation(stream_id))
    except AutoGenerationUnavailable as exc:
        return jsonify({'error': str(exc)}), 503
    except AutoGenerationBusy as exc:
        return jsonify({'error': str(exc)}), 409
    except AutoGenerationError as exc:
        message = str(exc)
        status_code = 404 if "No stream" in message or "No active AI generation" in message else 400
        return jsonify({'error': message}), status_code

def _apply_timer_snap_setting(enabled: bool) -> Dict[str, Dict[str, Optional[str]]]:
    return TIMER_SYNC_SERVICE.apply_timer_snap_setting(enabled)


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


def api_media_settings():
    global CONFIG, MEDIA_UPLOAD_MAX_MB, MEDIA_UPLOAD_MAX_BYTES

    cfg = load_config()
    upload_limit_mb = max(1, _as_int(cfg.get("MEDIA_UPLOAD_MAX_MB"), 2048))
    if request.method == "GET":
        return jsonify(
            {
                "media_upload_max_mb": upload_limit_mb,
                "media_upload_max_bytes": upload_limit_mb * 1024 * 1024,
            }
        )

    api_key = cfg.get("API_KEY")
    if api_key and request.headers.get("X-API-Key") != api_key:
        return jsonify({"error": "Unauthorized"}), 401

    payload = request.get_json(silent=True) or {}
    requested = max(1, _as_int(payload.get("media_upload_max_mb"), upload_limit_mb))
    cfg["MEDIA_UPLOAD_MAX_MB"] = requested
    try:
        config_manager.save_config(cfg)
    except Exception:
        logger.exception("Failed to persist media upload setting")
        return jsonify({"error": "Unable to save setting"}), 500

    CONFIG["MEDIA_UPLOAD_MAX_MB"] = requested
    MEDIA_UPLOAD_MAX_MB = requested
    MEDIA_UPLOAD_MAX_BYTES = requested * 1024 * 1024
    try:
        MEDIA_MANAGER.set_max_upload_mb(requested)
    except Exception:
        logger.exception("Failed to apply media upload setting to runtime")
        return jsonify({"error": "Unable to apply setting"}), 500

    return jsonify(
        {
            "media_upload_max_mb": requested,
            "media_upload_max_bytes": MEDIA_UPLOAD_MAX_BYTES,
        }
    )


def _sync_timer_payload(timer_id: str, entry: Dict[str, Any]) -> Dict[str, Any]:
    return TIMER_SYNC_SERVICE.sync_timer_payload(timer_id, entry)


def _emit_sync_timer_update() -> None:
    TIMER_SYNC_SERVICE.emit_sync_timer_update()


def sync_timers_collection():
    if request.method == "GET":
        return jsonify({"timers": get_sync_timers_snapshot()})
    payload, status = TIMER_SYNC_SERVICE.create_sync_timer(request.get_json(silent=True) or {})
    return jsonify(payload), status


def sync_timer_item(timer_id: str):
    if not timer_id:
        return jsonify({"error": "Timer id required"}), 400
    if TIMER_SYNC_SERVICE.get_sync_timer_config(timer_id) is None:
        return jsonify({"error": "Timer not found"}), 404

    if request.method == "DELETE":
        return jsonify(TIMER_SYNC_SERVICE.delete_sync_timer(timer_id))

    return jsonify(TIMER_SYNC_SERVICE.update_sync_timer(timer_id, request.get_json(silent=True) or {}))


# --- Stream groups and metadata ---
def streams_meta():
    return jsonify(GROUP_SERVICE.streams_meta())


def groups_collection():
    if request.method == "GET":
        return jsonify(GROUP_SERVICE.groups_payload())
    data = request.get_json(silent=True) or {}
    try:
        return jsonify(GROUP_SERVICE.create_group(data))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 400
    except FileExistsError as exc:
        return jsonify({"error": str(exc)}), 409


def groups_delete(name):
    if GROUP_SERVICE.delete_group(name):
        return jsonify({"status": "deleted"})
    return jsonify({"error": "not found"}), 404


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
        if content_type in {"video", "playlist"}:
            response_payload["sync_enabled"] = True
            sync_state = _get_youtube_sync_state(
                stream_id,
                {
                    "playlist_id": youtube_details.get("playlist_id"),
                    "video_id": youtube_details.get("video_id"),
                    "content_type": content_type,
                },
            )
            if sync_state:
                if sync_state.get("playlist_index") is not None:
                    response_payload["start_index"] = sync_state.get("playlist_index")
                if sync_state.get("start_seconds") is not None:
                    response_payload["start_seconds"] = sync_state.get("start_seconds")
                if sync_state.get("video_id"):
                    response_payload["video_id"] = sync_state.get("video_id")
                response_payload["server_time"] = sync_state.get("server_time")
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

    try:
        hls_url = LIVE_HLS_SERVICE.resolve_hls_url(stream_id, stream_url)
    except Exception as exc:
        logger.debug("Synchronous HLS detection failed for %s: %s", stream_id, exc)
        hls_url = None

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

    return jsonify(LIVE_HLS_SERVICE.invalidate_stream(stream_id, target_url))


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
    return jsonify(
        MEDIA_LIBRARY_SERVICE.get_images_payload(
            folder=folder,
            hide_nsfw=request.args.get("hide_nsfw"),
            library=request.args.get("library"),
            offset=offset,
            limit=limit,
        )
    )


def get_random_image():
    try:
        return jsonify(
            MEDIA_LIBRARY_SERVICE.get_random_image_payload(
                folder=request.args.get("folder", "all"),
                hide_nsfw=request.args.get("hide_nsfw"),
                library=request.args.get("library"),
            )
        )
    except LookupError:
        return jsonify({"error": "No images found"}), 404


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
    return jsonify(
        MEDIA_LIBRARY_SERVICE.get_media_entries_payload(
            folder=folder,
            hide_nsfw=request.args.get("hide_nsfw"),
            kind=request.args.get("kind"),
            library=request.args.get("library"),
            offset=offset,
            limit=limit,
        )
    )


def get_random_media():
    try:
        return jsonify(
            MEDIA_LIBRARY_SERVICE.get_random_media_payload(
                folder=request.args.get("folder", "all"),
                hide_nsfw=request.args.get("hide_nsfw"),
                kind=request.args.get("kind"),
                library=request.args.get("library"),
                stream_id=(request.args.get("stream_id") or "").strip(),
            )
        )
    except LookupError:
        return jsonify({"error": "No media found"}), 404


def notes():
    if request.method == "GET":
        return jsonify({"notes": settings.get("_notes", "")})
    data = request.get_json(silent=True) or {}
    settings["_notes"] = data.get("notes", "")
    save_settings_debounced()
    return jsonify({"status": "saved"})


def _acquire_resized_image_lock(path: Path) -> threading.Lock:
    return ASSET_DELIVERY_SERVICE.acquire_resized_image_lock(path)


def _parse_image_resize_request() -> Optional[Tuple[int, int]]:
    return ASSET_DELIVERY_SERVICE.parse_image_resize_request()


def _get_resized_image_path(
    virtual_path: str,
    source_path: Path,
    bounds: Tuple[int, int],
) -> Optional[Path]:
    return ASSET_DELIVERY_SERVICE.get_resized_image_path(virtual_path, source_path, bounds)


def _send_image_response(path: Union[str, Path]):
    return ASSET_DELIVERY_SERVICE.send_image_response(path)


def _send_video_response(path: Union[str, Path]):
    return ASSET_DELIVERY_SERVICE.send_video_response(path)


def _log_bad_media_once(path: str, exc: BaseException) -> None:
    ASSET_DELIVERY_SERVICE.log_bad_media_once(path, exc)


def _media_unavailable_response(path: str):
    return ASSET_DELIVERY_SERVICE.media_unavailable_response(path)


def serve_image(image_path):
    return ASSET_DELIVERY_SERVICE.serve_image(image_path)


def serve_video(video_path):
    return ASSET_DELIVERY_SERVICE.serve_video(video_path)


def stream_thumbnail_metadata(stream_id):
    return ASSET_DELIVERY_SERVICE.stream_thumbnail_metadata(stream_id)


def stream_thumbnail_image(stream_id):
    return ASSET_DELIVERY_SERVICE.stream_thumbnail_image(stream_id)


def cached_stream_thumbnail(stream_id: str):
    return ASSET_DELIVERY_SERVICE.cached_stream_thumbnail(stream_id)


def stream_group(name):
    try:
        view_model = GROUP_SERVICE.build_group_view_model(name)
    except KeyError:
        return f"No group '{name}'", 404
    return render_template("streams/streams.html", **view_model)


_library_blueprint = create_library_blueprint(
    list_tags_handler=list_tags,
    create_tag_handler=create_tag,
    delete_tag_handler=delete_tag,
    timer_settings_handler=api_timer_settings,
    sync_timers_collection_handler=sync_timers_collection,
    sync_timer_item_handler=sync_timer_item,
    streams_meta_handler=streams_meta,
    groups_collection_handler=groups_collection,
    groups_delete_handler=groups_delete,
    notes_handler=notes,
    stream_group_handler=stream_group,
)
register_blueprint_with_legacy_aliases(app, _library_blueprint, {
    "list_tags": "library_routes.list_tags",
    "create_tag": "library_routes.create_tag",
    "delete_tag": "library_routes.delete_tag",
    "api_timer_settings": "library_routes.api_timer_settings",
    "sync_timers_collection": "library_routes.sync_timers_collection",
    "sync_timer_item": "library_routes.sync_timer_item",
    "streams_meta": "library_routes.streams_meta",
    "groups_collection": "library_routes.groups_collection",
    "groups_delete": "library_routes.groups_delete",
    "notes": "library_routes.notes",
    "stream_group": "library_routes.stream_group",
})


_live_blueprint = create_live_blueprint(
    stream_live_handler=stream_live,
    stream_live_invalidate_handler=stream_live_invalidate,
    legacy_stream_live_handler=legacy_stream_live,
    test_embed_handler=test_embed,
)
register_blueprint_with_legacy_aliases(app, _live_blueprint, {
    "stream_live": "live_routes.stream_live",
    "stream_live_invalidate": "live_routes.stream_live_invalidate",
    "legacy_stream_live": "live_routes.legacy_stream_live",
    "test_embed": "live_routes.test_embed",
})


_assets_blueprint = create_assets_blueprint(
    get_images_handler=get_images,
    get_random_image_handler=get_random_image,
    get_media_entries_handler=get_media_entries,
    get_random_media_handler=get_random_media,
    serve_image_handler=serve_image,
    serve_video_handler=serve_video,
    stream_thumbnail_metadata_handler=stream_thumbnail_metadata,
    stream_thumbnail_image_handler=stream_thumbnail_image,
    cached_stream_thumbnail_handler=cached_stream_thumbnail,
)
register_blueprint_with_legacy_aliases(app, _assets_blueprint, {
    "get_images": "asset_routes.get_images",
    "get_random_image": "asset_routes.get_random_image",
    "get_media_entries": "asset_routes.get_media_entries",
    "get_random_media": "asset_routes.get_random_media",
    "serve_image": "asset_routes.serve_image",
    "serve_video": "asset_routes.serve_video",
    "stream_thumbnail_metadata": "asset_routes.stream_thumbnail_metadata",
    "stream_thumbnail_image": "asset_routes.stream_thumbnail_image",
    "cached_stream_thumbnail": "asset_routes.cached_stream_thumbnail",
})


def _emit_initial_youtube_sync(stream_id: str, sid: str) -> None:
    stream_conf = settings.get(stream_id)
    if not isinstance(stream_conf, dict):
        return
    stream_url = str(stream_conf.get("stream_url") or "").strip()
    youtube_details = _parse_youtube_url_details(stream_url) if stream_url else None
    if not youtube_details:
        return
    content_type = (
        str((stream_conf.get("embed_metadata") or {}).get("content_type") or "").strip().lower()
        if isinstance(stream_conf.get("embed_metadata"), dict)
        else ""
    )
    if not content_type:
        if youtube_details.get("playlist_id"):
            content_type = "playlist"
        elif youtube_details.get("is_live"):
            content_type = "live"
        else:
            content_type = "video"
    if content_type not in {"video", "playlist"}:
        return
    sync_state = _get_youtube_sync_state(
        stream_id,
        {
            "playlist_id": youtube_details.get("playlist_id"),
            "video_id": youtube_details.get("video_id"),
            "content_type": content_type,
        },
    )
    if sync_state:
        safe_emit(YOUTUBE_SYNC_EVENT, sync_state, to=sid)


register_stream_socket_handlers(
    socketio=socketio,
    app_logger=logger,
    stable_horde_log_prefix=STABLE_HORDE_LOG_PREFIX,
    socket_noise_keywords=_SOCKET_NOISE_KEYWORDS,
    detach_listener=job_manager.detach_listener,
    attach_listener=job_manager.attach_listener,
    remove_youtube_sync_subscriber=_remove_youtube_sync_subscriber,
    assign_youtube_sync_leader=_assign_youtube_sync_leader,
    emit_initial_youtube_sync=_emit_initial_youtube_sync,
    playback_service=PLAYBACK_SERVICE,
    stream_init_event=STREAM_INIT_EVENT,
    settings=settings,
    safe_emit=safe_emit,
)


register_youtube_sync_socket_handlers(
    socketio=socketio,
    settings=settings,
    parse_youtube_url_details=_parse_youtube_url_details,
    youtube_sync_role_for_sid=_youtube_sync_role_for_sid,
    store_youtube_sync_state=_store_youtube_sync_state,
    safe_emit=safe_emit,
    youtube_sync_event=YOUTUBE_SYNC_EVENT,
)


if __name__ == "__main__":
    run_dev_server(app, socketio, port=5000)
