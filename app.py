from flask import Flask, jsonify, send_file, request, render_template, redirect, url_for
from flask_socketio import SocketIO, join_room, leave_room
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
import io
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Set
from urllib.parse import urlparse

from PIL import Image, ImageDraw, ImageFont

try:
    import cv2  # type: ignore[import]
except Exception:
    cv2 = None

from werkzeug.http import generate_etag

try:
    import requests
except Exception:
    requests = None

from stablehorde import StableHorde, StableHordeError, StableHordeCancelled
from update_helpers import backup_user_state, restore_user_state

app = Flask(__name__, static_folder="static", static_url_path="/static")
socketio = SocketIO(app)

logger = logging.getLogger(__name__)

SETTINGS_FILE = "settings.json"
CONFIG_FILE = "config.json"
IMAGE_DIR = "/mnt/viewers"  # Adjust if needed

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

IMAGE_DIR_PATH = Path(IMAGE_DIR)

THUMBNAIL_CACHE_DIR = IMAGE_DIR_PATH / THUMBNAIL_SUBDIR
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
MEDIA_MODE_CHOICES = {
    MEDIA_MODE_IMAGE,
    MEDIA_MODE_VIDEO,
    MEDIA_MODE_LIVESTREAM,
    MEDIA_MODE_AI,
}
MEDIA_MODE_VARIANTS = {
    MEDIA_MODE_IMAGE: {"random", "specific"},
    MEDIA_MODE_VIDEO: {"random", "specific"},
    MEDIA_MODE_LIVESTREAM: {"livestream"},
    MEDIA_MODE_AI: {AI_MODE},
}

NSFW_KEYWORD = "nsfw"

# Cache image paths per folder so we can serve repeated requests without rescanning the disk.
IMAGE_CACHE: Dict[str, Dict[str, Any]] = {}
IMAGE_CACHE_LOCK = threading.Lock()
STREAM_RUNTIME_STATE: Dict[str, Dict[str, Any]] = {}
STREAM_RUNTIME_LOCK = threading.Lock()

STREAM_PLAYBACK_HISTORY_LIMIT = 50
STREAM_UPDATE_EVENT = "stream_update"
STREAM_INIT_EVENT = "stream_init"
SYNC_TIME_EVENT = "sync_time"
STREAM_SYNC_INTERVAL_SECONDS = 3.0

playback_manager: Optional["StreamPlaybackManager"] = None


def _normalize_folder_key(folder: Optional[str]) -> str:
    """Normalize request folder values into the cache key used internally."""
    if not folder or folder in ("all", "."):
        return "all"
    return folder.replace("\\", "/")


def _resolve_folder_path(folder_key: str) -> str:
    """Return the absolute filesystem path for a cache key."""
    return IMAGE_DIR if folder_key == "all" else os.path.join(IMAGE_DIR, folder_key)


def _scan_folder_for_cache(folder_key: str) -> Tuple[List[Dict[str, str]], Dict[str, float]]:
    """Scan a folder on disk and build the cached payload for it."""
    target_dir = _resolve_folder_path(folder_key)
    dir_markers: Dict[str, float] = {}
    try:
        dir_markers[target_dir] = os.stat(target_dir).st_mtime
    except FileNotFoundError:
        dir_markers[target_dir] = 0.0
        return [], dir_markers
    except OSError:
        dir_markers[target_dir] = 0.0
        return [], dir_markers

    media: List[Dict[str, str]] = []
    images: List[str] = []
    for root, _, files in os.walk(target_dir):
        if root != target_dir:
            try:
                dir_markers[root] = os.stat(root).st_mtime
            except OSError:
                # If a directory becomes unavailable, surface that as a change on the next poll.
                dir_markers[root] = time.time()
        for file_name in files:
            ext = os.path.splitext(file_name)[1].lower()
            if ext in MEDIA_EXTENSIONS:
                rel_path = os.path.relpath(os.path.join(root, file_name), IMAGE_DIR)
                normalized = rel_path.replace("\\", "/")
                kind = "video" if ext in VIDEO_EXTENSIONS else "image"
                media.append({
                    "path": normalized,
                    "kind": kind,
                    "extension": ext,
                })
                if kind == "image":
                    images.append(normalized)
    images.sort(key=str.lower)
    media.sort(key=lambda item: item["path"].lower())
    return media, dir_markers


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
    all_images = refresh_image_cache("all", force=True)
    if not os.path.isdir(IMAGE_DIR):
        return

    with IMAGE_CACHE_LOCK:
        all_entry = IMAGE_CACHE.get("all")
        if not all_entry:
            return
        base_markers = dict(all_entry.get("dir_markers", {}))
        base_updated = all_entry.get("last_updated", time.time())
        all_media = [dict(item) for item in all_entry.get("media", [])]

    for root, _, _ in os.walk(IMAGE_DIR):
        rel = os.path.relpath(root, IMAGE_DIR)
        if rel == ".":
            continue
        folder_key = rel.replace("\\", "/")
        folder_root = _resolve_folder_path(folder_key)
        folder_images = [img for img in all_images if img.startswith(f"{folder_key}/")]
        folder_media = [item for item in all_media if item.get("path", "").startswith(f"{folder_key}/")]
        folder_markers = {
            path: mtime
            for path, mtime in base_markers.items()
            if path == folder_root or path.startswith(folder_root + os.sep)
        }
        if folder_root not in folder_markers:
            try:
                folder_markers[folder_root] = os.stat(folder_root).st_mtime
            except OSError:
                folder_markers[folder_root] = 0.0
        entry = {
            "images": folder_images,
            "media": folder_media,
            "dir_markers": folder_markers,
            "last_updated": base_updated,
        }
        with IMAGE_CACHE_LOCK:
            IMAGE_CACHE[folder_key] = entry


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
    if mode not in {'random', 'specific', 'livestream', AI_MODE}:
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
                seen: Set[str] = set()
                for entry in streams_raw:
                    if not isinstance(entry, str):
                        continue
                    candidate = entry.strip()
                    if candidate in valid_streams and candidate not in seen:
                        cleaned.append(candidate)
                        seen.add(candidate)
            layout = _normalize_group_layout(payload.get('layout'))
            entry: Dict[str, Any] = {'streams': cleaned}
            if layout:
                entry['layout'] = layout
            sanitized[trimmed] = entry
        elif isinstance(payload, (list, tuple)):
            cleaned = []
            seen: Set[str] = set()
            for entry in payload:
                if not isinstance(entry, str):
                    continue
                candidate = entry.strip()
                if candidate in valid_streams and candidate not in seen:
                    cleaned.append(candidate)
                    seen.add(candidate)
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

    try:
        socketio.emit(
            "streams_changed",
            {
                "action": "import",
                "added": added,
                "removed": removed,
                "updated": updated,
            },
        )
        for stream_id in new_streams:
            socketio.emit(
                "refresh",
                {"stream_id": stream_id, "config": settings[stream_id], "tags": tags_snapshot},
            )
    except Exception as exc:  # pragma: no cover
        logger.debug("Socket emit failed during settings import: %s", exc)

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



def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    return {}

config_data = load_config()


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


AI_DEFAULT_MODEL = config_data.get("AI_DEFAULT_MODEL", AI_DEFAULT_MODEL) or AI_DEFAULT_MODEL
AI_DEFAULT_SAMPLER = config_data.get("AI_DEFAULT_SAMPLER", AI_DEFAULT_SAMPLER) or AI_DEFAULT_SAMPLER
AI_DEFAULT_WIDTH = _coerce_int(config_data.get("AI_DEFAULT_WIDTH"), AI_DEFAULT_WIDTH)
AI_DEFAULT_HEIGHT = _coerce_int(config_data.get("AI_DEFAULT_HEIGHT"), AI_DEFAULT_HEIGHT)
AI_DEFAULT_STEPS = _coerce_int(config_data.get("AI_DEFAULT_STEPS"), AI_DEFAULT_STEPS)
AI_DEFAULT_CFG = _coerce_float(config_data.get("AI_DEFAULT_CFG"), AI_DEFAULT_CFG)
AI_DEFAULT_SAMPLES = _coerce_int(config_data.get("AI_DEFAULT_SAMPLES"), AI_DEFAULT_SAMPLES)
AI_OUTPUT_SUBDIR = config_data.get("AI_OUTPUT_SUBDIR", AI_OUTPUT_SUBDIR) or AI_OUTPUT_SUBDIR
AI_TEMP_SUBDIR = config_data.get("AI_TEMP_SUBDIR", AI_TEMP_SUBDIR) or AI_TEMP_SUBDIR
AI_DEFAULT_PERSIST = _coerce_bool(config_data.get("AI_DEFAULT_PERSIST"), AI_DEFAULT_PERSIST)
AI_POLL_INTERVAL = _coerce_float(config_data.get("AI_POLL_INTERVAL"), AI_POLL_INTERVAL)
AI_TIMEOUT = _coerce_float(config_data.get("AI_TIMEOUT"), AI_TIMEOUT)


AI_OUTPUT_ROOT = _ensure_dir(Path(IMAGE_DIR) / AI_OUTPUT_SUBDIR)
AI_TEMP_ROOT = _ensure_dir(Path(IMAGE_DIR) / AI_TEMP_SUBDIR)

try:
    stable_horde_client = StableHorde(
        save_dir=AI_OUTPUT_ROOT,
        persist_images=AI_DEFAULT_PERSIST,
        default_poll_interval=AI_POLL_INTERVAL,
        default_timeout=AI_TIMEOUT,
    )
except Exception as exc:  # pragma: no cover - defensive during optional setup
    logger.warning("Stable Horde client unavailable: %s", exc)
    stable_horde_client = None

ai_jobs_lock = threading.Lock()
ai_jobs: Dict[str, Dict[str, Any]] = {}
ai_job_controls: Dict[str, Dict[str, Any]] = {}
ai_model_cache: Dict[str, Any] = {"timestamp": 0.0, "data": []}
auto_scheduler: Optional['AutoGenerateScheduler'] = None


def _relative_image_path(path: Union[Path, str]) -> str:
    p = Path(path)
    try:
        rel = p.relative_to(IMAGE_DIR_PATH)
    except ValueError:
        try:
            rel = p.resolve().relative_to(IMAGE_DIR_PATH.resolve())
        except Exception:  # pragma: no cover - fallback path
            return p.as_posix()
    return rel.as_posix()


def _cleanup_temp_outputs(stream_id: str) -> None:
    temp_dir = AI_TEMP_ROOT / stream_id
    if temp_dir.exists():
        try:
            shutil.rmtree(temp_dir)
        except Exception as exc:
            logger.warning('Failed to remove temp outputs for %s: %s', stream_id, exc)


def _emit_ai_update(stream_id: str, state: Dict[str, Any], job: Optional[Dict[str, Any]] = None) -> None:
    try:
        payload: Dict[str, Any] = {"stream_id": stream_id, "state": state}
        if job is not None:
            payload['job'] = job
        socketio.emit('ai_job_update', payload)
    except Exception as exc:  # pragma: no cover - socket errors should not crash the server
        logger.debug('Socket emit failed for ai_job_update: %s', exc)


def _update_ai_state(stream_id: str, updates: Dict[str, Any], *, persist: bool = False) -> Dict[str, Any]:
    conf = settings.get(stream_id)
    if not conf:
        return {}
    ensure_ai_defaults(conf)
    state = conf[AI_STATE_KEY]
    state.update(updates)
    if persist:
        save_settings(settings)
    _emit_ai_update(stream_id, state)
    return state


def _record_job_progress(stream_id: str, stage: str, payload: Dict[str, Any]) -> None:
    with ai_jobs_lock:
        job = ai_jobs.get(stream_id)
        if not job:
            return
        job = dict(job)
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
    current_state = settings.get(stream_id, {}).get(AI_STATE_KEY, {})
    _emit_ai_update(stream_id, current_state, job=job)


def _run_ai_generation(stream_id: str, options: Dict[str, Any], cancel_event: Optional[threading.Event] = None) -> None:
    prompt = str(options.get('prompt') or '').strip()
    if not prompt:
        _update_ai_state(stream_id, {
            'status': 'error',
            'message': 'Prompt is required',
            'error': 'Prompt is required',
        }, persist=True)
        with ai_jobs_lock:
            ai_jobs.pop(stream_id, None)
            ai_job_controls.pop(stream_id, None)
        return

    persist = bool(options.get('save_output', AI_DEFAULT_PERSIST))
    if cancel_event and cancel_event.is_set():
        message = 'Cancelled by user'
        job_snapshot = None
        with ai_jobs_lock:
            current_job = ai_jobs.get(stream_id)
            if current_job:
                job_snapshot = dict(current_job)
                job_snapshot['status'] = 'cancelled'
                job_snapshot['message'] = message
                ai_jobs[stream_id] = job_snapshot
            ai_job_controls.pop(stream_id, None)
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
        )
    except StableHordeCancelled as exc:
        logger.info('Stable Horde job for %s cancelled: %s', stream_id, exc)
        message = 'Cancelled by user'
        job_snapshot = None
        with ai_jobs_lock:
            current_job = ai_jobs.get(stream_id)
            if current_job:
                job_snapshot = dict(current_job)
                job_snapshot['status'] = 'cancelled'
                job_snapshot['message'] = message
                ai_jobs[stream_id] = job_snapshot
            ai_job_controls.pop(stream_id, None)
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
        with ai_jobs_lock:
            ai_jobs.pop(stream_id, None)
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
        try:
            socketio.emit('refresh', {'stream_id': stream_id, 'config': conf})
        except Exception as exc:  # pragma: no cover
            logger.debug('Socket refresh emit failed: %s', exc)

    _record_job_progress(stream_id, 'completed', {'job_id': result.job_id})
    with ai_jobs_lock:
        ai_jobs.pop(stream_id, None)
        ai_job_controls.pop(stream_id, None)


settings = load_settings()
if ensure_settings_integrity(settings):
    save_settings(settings)
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


def _update_stream_runtime_state(
    stream_id: str,
    *,
    path: Optional[str] = None,
    kind: Optional[str] = None,
    media_mode: Optional[str] = None,
    stream_url: Optional[str] = None,
    source: str = "unknown",
) -> None:
    if not stream_id or stream_id.startswith("_"):
        return
    normalized_mode = media_mode.strip().lower() if isinstance(media_mode, str) else None
    normalized_kind = kind.strip().lower() if isinstance(kind, str) else None
    resolved_path = path if path not in ("", None) else None
    if resolved_path and not normalized_kind:
        normalized_kind = _detect_media_kind(resolved_path)
    with STREAM_RUNTIME_LOCK:
        entry = STREAM_RUNTIME_STATE.setdefault(stream_id, {})
        if media_mode is not None:
            entry["media_mode"] = normalized_mode
        if path is not None:
            entry["path"] = resolved_path
            if resolved_path is None:
                entry.pop("kind", None)
            elif normalized_kind:
                entry["kind"] = normalized_kind
        elif normalized_kind:
            entry["kind"] = normalized_kind
        if stream_url is not None:
            entry["stream_url"] = stream_url or None
        entry["timestamp"] = time.time()
        entry["source"] = source

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
    base_root = IMAGE_DIR_PATH.resolve()
    try:
        target = (base_root / rel_path).resolve()
        target.relative_to(base_root)
    except (ValueError, RuntimeError):
        return None
    except FileNotFoundError:
        return None
    if not target.exists():
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
        if new_media_mode not in (MEDIA_MODE_IMAGE, MEDIA_MODE_VIDEO):
            inferred = _infer_media_mode(conf)
            if inferred in (MEDIA_MODE_IMAGE, MEDIA_MODE_VIDEO):
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
            state.set_media(media, duration=duration, source="history_prev", playback_mode=playback_mode, history_index=target_index, add_to_history=False)
            payload = state.to_payload()
        self._emit_state(payload, room=stream_id)
        return payload

    def skip_next(self, stream_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            state = self._states.get(stream_id)
            if not state or not state.should_run():
                state_to_emit = state.to_payload() if state else None
            else:
                target_index = state.history_index + 1
                entry = state.get_history_entry(target_index)
                if entry:
                    media = entry.get("media")
                    duration = entry.get("duration")
                    playback_mode = entry.get("playback_mode")
                    if media:
                        state.set_media(media, duration=duration, source="history_next", playback_mode=playback_mode, history_index=target_index, add_to_history=False)
                        state_to_emit = state.to_payload()
                    else:
                        state_to_emit = None
                else:
                    state_to_emit = None
        if state_to_emit:
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
        try:
            if target_room:
                socketio.emit(event, payload, to=target_room)
            else:
                socketio.emit(event, payload)
        except Exception as exc:  # pragma: no cover - socket broadcast best effort
            logger.debug("Stream emit failed for %s (%s): %s", target_room or stream_id, event, exc)

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
        with self._lock:
            state = self._states.get(stream_id)
            if not state or not state.should_run():
                return None
            media = self._next_media(state)
            if media is None:
                state.set_error("no_media")
                return state.to_payload()
            duration = self._compute_duration(state, media)
            playback_mode = state.video_playback_mode if media.get("kind") == "video" else None
            state.set_media(media, duration=duration, source=reason, playback_mode=playback_mode)
            payload = state.to_payload()
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
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
    except AttributeError:
        text_width, text_height = draw.textsize(text, font=font)
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
    else:
        kind = kind or 'image'
    placeholder = False
    if not path and media_mode != MEDIA_MODE_LIVESTREAM:
        placeholder = True
    badge_map = {
        MEDIA_MODE_LIVESTREAM: 'Live',
        MEDIA_MODE_VIDEO: 'Video',
        MEDIA_MODE_AI: 'AI',
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
            desired_mode = current_mode if current_mode in allowed else "random"
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
    subfolders = ["all"]
    if os.path.isdir(IMAGE_DIR):
        for root, dirs, _ in os.walk(IMAGE_DIR):
            for d in dirs:
                rel_path = os.path.relpath(os.path.join(root, d), IMAGE_DIR)
                normalized = rel_path.replace("\\", "/")
                if hide_nsfw and _path_contains_nsfw(normalized):
                    continue
                subfolders.append(rel_path)
            break
    return subfolders


def get_folder_inventory(hide_nsfw: bool = False) -> List[Dict[str, Any]]:
    inventory: List[Dict[str, Any]] = []
    for name in get_subfolders(hide_nsfw=hide_nsfw):
        media_entries = list_media(name, hide_nsfw=hide_nsfw)
        has_images = any(entry.get("kind") == "image" for entry in media_entries)
        has_videos = any(entry.get("kind") == "video" for entry in media_entries)
        inventory.append({
            "name": name,
            "has_images": has_images,
            "has_videos": has_videos,
        })
    return inventory


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


def try_get_hls(original_url):
    if not original_url:
        return None
    try:
        result = subprocess.run(
            ["yt-dlp", "-g", original_url],
            capture_output=True,
            text=True,
            check=True
        )
        raw_url = result.stdout.strip()
        if any(ext in raw_url for ext in [".m3u8", ".mpd"]):
            return raw_url
        return None
    except subprocess.CalledProcessError:
        return None

@app.route("/")
def dashboard():
    folder_inventory = get_folder_inventory()
    subfolders = [item["name"] for item in folder_inventory]
    streams = {k: v for k, v in settings.items() if not k.startswith("_")}
    for conf in streams.values():
        if isinstance(conf, dict):
            quality = conf.get("image_quality")
            if not isinstance(quality, str) or quality.strip().lower() not in IMAGE_QUALITY_CHOICES:
                conf["image_quality"] = "auto"
            else:
                conf["image_quality"] = quality.strip().lower()
            ensure_background_defaults(conf)
            ensure_tag_defaults(conf)
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


@app.route("/stream")
def mosaic_streams():
    # Dynamic global view: include all streams ("online" assumed as configured)
    streams = {k: v for k, v in settings.items() if not k.startswith("_")}
    for conf in streams.values():
        if isinstance(conf, dict):
            ensure_background_defaults(conf)
            ensure_tag_defaults(conf)
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
            socketio.emit("streams_changed", {"action": "added", "stream_id": new_id})
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
        socketio.emit("streams_changed", {"action": "deleted", "stream_id": stream_id})
        return jsonify({"status": "deleted"})
    return jsonify({"error": "not found"}), 404

@app.route("/get-settings/<stream_id>", methods=["GET"])
def get_stream_settings(stream_id):
    if stream_id not in settings:
        return jsonify({"error": f"No stream '{stream_id}' found"}), 404
    conf = settings[stream_id]
    ensure_ai_defaults(conf)
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
    previous_mode = conf.get("mode")

    # We'll add new keys for YouTube: "yt_cc", "yt_mute", "yt_quality"
    for key in ["mode", "folder", "selected_image", "duration", "shuffle", "stream_url",
                "image_quality", "yt_cc", "yt_mute", "yt_quality", "label", "hide_nsfw",
                "background_blur_enabled", "background_blur_amount", "video_playback_mode",
                "video_volume", "selected_media_kind", "media_mode", TAG_KEY]:
        if key in data:
            val = data[key]
            if key == "stream_url":
                val = val.strip()
                conf[key] = val if val and val.lower() != "none" else None
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
    socketio.emit("refresh", {"stream_id": stream_id, "config": conf, "tags": get_global_tags()})
    return jsonify({"status": "success", "new_config": conf, "tags": get_global_tags()})


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
        }
        ai_job_controls[stream_id] = {'cancel_event': cancel_event}
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
        queued_state['last_auto_trigger'] = datetime.utcnow().isoformat() + 'Z'
    queued_state['last_auto_error'] = None

    save_settings(settings)
    _emit_ai_update(stream_id, queued_state, job=ai_jobs[stream_id])
    try:
        socketio.emit('refresh', {'stream_id': stream_id, 'config': conf, 'tags': get_global_tags()})
    except Exception as exc:  # pragma: no cover
        logger.debug('Socket refresh emit failed: %s', exc)

    job_options = dict(sanitized)
    job_options['prompt'] = prompt
    worker = threading.Thread(target=_run_ai_generation, args=(stream_id, job_options, cancel_event), daemon=True)
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
        self._update_state(stream_id, next_auto_trigger=next_dt.isoformat())

    def _compute_next_datetime(self, conf: Dict[str, Any], mode_value: str, *, base_time: Optional[float]) -> Optional[datetime]:
        ai_settings = conf[AI_SETTINGS_KEY]
        reference_ts = base_time if base_time is not None else time.time()
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
            target_ts = reference_ts + interval_seconds
            return datetime.fromtimestamp(target_ts)
        if mode_value == 'clock':
            clock_value = _normalize_clock_time(ai_settings.get('auto_generate_clock_time'))
            if not clock_value:
                return None
            hour, minute = map(int, clock_value.split(':'))
            base_dt = datetime.fromtimestamp(reference_ts)
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
        state = conf[AI_STATE_KEY]
        changed = False
        for key, value in updates.items():
            if state.get(key) != value:
                state[key] = value
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
    try:
        socketio.emit("mosaic_refresh", {"group": name})
    except Exception:
        pass
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
    stream_id = request.args.get("stream_id", "").strip()
    if stream_id not in settings:
        return jsonify({"error": f"No stream '{stream_id}' found"}), 404

    stream_url = settings[stream_id].get("stream_url", "")
    if not stream_url:
        return jsonify({"error": "No live stream URL configured"}), 404

    if "youtube.com" in stream_url or "youtu.be" in stream_url:
        embed_id = None
        if "watch?v=" in stream_url:
            parts = stream_url.split("watch?v=")[1].split("&")[0].split("#")[0]
            embed_id = parts
        elif "youtu.be/" in stream_url:
            embed_id = stream_url.split("youtu.be/")[1].split("?")[0].split("&")[0]
        return jsonify({
            "embed_type": "youtube",
            "embed_id": embed_id,
            "hls_url": None,
            "original_url": stream_url
        })

    if "twitch.tv" in stream_url:
        embed_id = stream_url.split("twitch.tv/")[1].split("/")[0]
        return jsonify({
            "embed_type": "twitch",
            "embed_id": embed_id,
            "hls_url": None,
            "original_url": stream_url
        })

    hls_link = try_get_hls(stream_url)
    if hls_link:
        return jsonify({
            "embed_type": "hls",
            "embed_id": None,
            "hls_url": hls_link,
            "original_url": stream_url
        })

    return jsonify({
        "embed_type": "iframe",
        "embed_id": None,
        "hls_url": None,
        "original_url": stream_url
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
            return "youtube", f"https://www.youtube.com/embed/{vid}"
        return "youtube", "https://www.youtube.com/embed/"
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
    return response


@app.route("/stream/image/<path:image_path>")
def serve_image(image_path):
    full_path = Path(IMAGE_DIR_PATH) / image_path
    if not full_path.exists():
        return "Not found", 404
    return _send_image_response(full_path)


@app.route("/stream/video/<path:video_path>")
def serve_video(video_path):
    base_root = IMAGE_DIR_PATH.resolve()
    target_path = (base_root / video_path).resolve()
    try:
        target_path.relative_to(base_root)
    except ValueError:
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
    timestamp = info.get('timestamp')
    cache_key = None
    if isinstance(timestamp, (int, float)):
        try:
            cache_key = str(int(timestamp))
        except (TypeError, ValueError):
            cache_key = None
    image_url = url_for('stream_thumbnail_image', stream_id=stream_id)
    if cache_key:
        image_url = f"{image_url}?v={cache_key}"
    payload = {
        'stream_id': stream_id,
        'media_mode': info.get('media_mode'),
        'kind': info.get('kind'),
        'path': info.get('path'),
        'image_url': image_url,
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
    kind = info.get('kind')
    path = info.get('path')
    image_obj = None
    if kind == "image" and path:
        media_path = _resolve_media_path(path)
        if media_path is not None:
            image_obj = _create_thumbnail_image(media_path)
    elif kind == "video" and path:
        media_path = _resolve_media_path(path)
        if media_path is not None:
            image_obj = _create_video_thumbnail(media_path)
    elif kind == "livestream":
        image_obj = _create_livestream_thumbnail(info.get("stream_url"))
    if image_obj is None:
        badge = info.get('badge') or 'No Preview'
        if kind == 'video':
            badge = 'Video'
        elif kind == 'livestream':
            badge = 'Live'
        image_obj = _generate_placeholder_thumbnail(badge)
    buffer = _thumbnail_image_to_bytes(image_obj)
    response = send_file(buffer, mimetype='image/jpeg')
    response.headers['Cache-Control'] = 'no-store, max-age=0'
    return response




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
    streams = {k: settings[k] for k in members if k in settings}
    return render_template("streams.html", stream_settings=streams, mosaic_settings=layout_conf)

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
    socketio.emit('video_control', message)


if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)







