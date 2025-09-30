from flask import Flask, jsonify, send_file, request, render_template, redirect, url_for
from flask_socketio import SocketIO
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
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

from PIL import Image
from werkzeug.http import generate_etag

try:
    import requests
except Exception:
    requests = None

from stablehorde import StableHorde, StableHordeError, StableHordeCancelled

app = Flask(__name__, static_folder="static", static_url_path="/static")
socketio = SocketIO(app)

logger = logging.getLogger(__name__)

SETTINGS_FILE = "settings.json"
CONFIG_FILE = "config.json"
IMAGE_DIR = "/mnt/viewers"  # Adjust if needed

# Bounding boxes for optional thumbnail sizes requested via ?size=.
THUMBNAIL_SUBDIR = "_thumbnails"
THUMBNAIL_SIZE_PRESETS = {
    "thumb": (320, 320),
    "medium": (1024, 1024),
    "full": None,  # Alias for the original size
}
IMAGE_CACHE_TIMEOUT = 60 * 60 * 24 * 7  # One week default for conditional responses
IMAGE_CACHE_CONTROL_MAX_AGE = 31536000  # One year for browser Cache-Control headers

IMAGE_QUALITY_CHOICES = {"auto", "thumb", "medium", "full"}

AI_MODE = "ai"
AI_SETTINGS_KEY = "ai_settings"
AI_STATE_KEY = "ai_state"

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

NSFW_KEYWORD = "nsfw"

# Cache image paths per folder so we can serve repeated requests without rescanning the disk.
IMAGE_CACHE: Dict[str, Dict[str, Any]] = {}
IMAGE_CACHE_LOCK = threading.Lock()


def _normalize_folder_key(folder: Optional[str]) -> str:
    """Normalize request folder values into the cache key used internally."""
    if not folder or folder in ("all", "."):
        return "all"
    return folder.replace("\\", "/")


def _resolve_folder_path(folder_key: str) -> str:
    """Return the absolute filesystem path for a cache key."""
    return IMAGE_DIR if folder_key == "all" else os.path.join(IMAGE_DIR, folder_key)


def _scan_folder_for_cache(folder_key: str) -> Tuple[List[str], Dict[str, float]]:
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

    images: List[str] = []
    for root, _, files in os.walk(target_dir):
        if root != target_dir:
            try:
                dir_markers[root] = os.stat(root).st_mtime
            except OSError:
                # If a directory becomes unavailable, surface that as a change on the next poll.
                dir_markers[root] = time.time()
        for file_name in files:
            if file_name.lower().endswith(IMAGE_EXTENSIONS):
                rel_path = os.path.relpath(os.path.join(root, file_name), IMAGE_DIR)
                images.append(rel_path.replace("\\", "/"))
    images.sort(key=str.lower)
    return images, dir_markers


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

    images, dir_markers = _scan_folder_for_cache(folder_key)
    entry = {
        "images": images,
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

    for root, _, _ in os.walk(IMAGE_DIR):
        rel = os.path.relpath(root, IMAGE_DIR)
        if rel == ".":
            continue
        folder_key = rel.replace("\\", "/")
        folder_root = _resolve_folder_path(folder_key)
        folder_images = [img for img in all_images if img.startswith(f"{folder_key}/")]
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


def _ai_settings_match_defaults(candidate: Dict[str, Any], defaults: Optional[Dict[str, Any]] = None) -> bool:
    if not isinstance(candidate, dict):
        return False
    baseline = defaults or default_ai_settings()
    for key in AI_FALLBACK_DEFAULTS.keys():
        if candidate.get(key) != baseline.get(key):
            return False
    return True


def default_mosaic_config():
    """Return the default configuration for the mosaic /stream page."""
    # ``layout`` controls how streams are arranged. ``grid`` uses the
    # classic column based approach while other values enable custom
    # layouts (e.g. horizontal or vertical stacking).
    return {
        "cols": 2,
        "rows": None,
        "layout": "grid",
        "pip_main": None,
        "pip_pip": None,
        "pip_corner": "bottom-right",
        "pip_size": 25,
    }


def default_stream_config():
    """Return the default configuration for a new stream."""
    return {
        "mode": "random",
        "folder": "all",
        "selected_image": None,
        "duration": 5,
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


def load_settings():
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, "r") as f:
            return json.load(f)
    # Start with no streams; dashboard can add them dynamically.
    return {}

def save_settings(data):
    with open(SETTINGS_FILE, "w") as f:
        json.dump(data, f, indent=4)


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
if "_mosaic" not in settings:
    settings["_mosaic"] = default_mosaic_config()
else:
    # Backwards compatibility for older settings files
    settings["_mosaic"].setdefault("layout", "grid")
    settings["_mosaic"].setdefault("cols", 2)
    settings["_mosaic"].setdefault("pip_main", None)
    settings["_mosaic"].setdefault("pip_pip", None)
    settings["_mosaic"].setdefault("pip_corner", "bottom-right")
    settings["_mosaic"].setdefault("pip_size", 25)

# Ensure global AI defaults exist and are sanitized
raw_ai_defaults = settings.get("_ai_defaults") if isinstance(settings.get("_ai_defaults"), dict) else None
settings["_ai_defaults"] = _sanitize_ai_settings(
    raw_ai_defaults or {},
    deepcopy(AI_FALLBACK_DEFAULTS),
    defaults=AI_FALLBACK_DEFAULTS,
)

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


@app.route("/folders", methods=["GET"])
def folders_collection():
    hide_nsfw = _parse_truthy(request.args.get("hide_nsfw"))
    return jsonify(get_subfolders(hide_nsfw=hide_nsfw))


def list_images(folder="all", hide_nsfw: bool = False):
    """Return cached image paths for the folder, refreshing when necessary."""
    images = refresh_image_cache(folder)
    return _filter_nsfw_images(images, hide_nsfw)


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
    subfolders = get_subfolders()
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
    mosaic = settings.get("_mosaic", default_mosaic_config())
    groups = sorted(list(settings.get("_groups", {}).keys()))
    return render_template(
        "index.html",
        subfolders=subfolders,
        stream_settings=streams,
        mosaic_settings=mosaic,
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
    mosaic = settings.get("_mosaic", default_mosaic_config())
    return render_template("streams.html", stream_settings=streams, mosaic_settings=mosaic)

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

@app.route("/settings/<stream_id>", methods=["POST"])
def update_stream_settings(stream_id):
    if stream_id not in settings:
        return jsonify({"error": f"No stream '{stream_id}' found"}), 404

    data = request.json
    conf = settings[stream_id]
    ensure_ai_defaults(conf)
    previous_mode = conf.get("mode")

    # We'll add new keys for YouTube: "yt_cc", "yt_mute", "yt_quality"
    for key in ["mode", "folder", "selected_image", "duration", "shuffle", "stream_url",
                "image_quality", "yt_cc", "yt_mute", "yt_quality", "label", "hide_nsfw",
                "background_blur_enabled", "background_blur_amount", TAG_KEY]:
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
            elif key == "image_quality":
                normalized = (val or "").strip().lower() if isinstance(val, str) else ""
                if normalized not in IMAGE_QUALITY_CHOICES:
                    normalized = "auto"
                conf[key] = normalized
            else:
                conf[key] = val

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

    save_settings(settings)
    return jsonify({"status": "success", "defaults": new_defaults})


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
        subprocess.check_call(["git", "fetch"], cwd=repo_path)
        subprocess.check_call(["git", "checkout", branch], cwd=repo_path)
        subprocess.check_call(["git", "reset", "--hard", f"origin/{branch}"], cwd=repo_path)
    except FileNotFoundError:
        return render_template(
            "update_status.html",
            message="Git executable not found. Please install Git to update the application.",
        )
    except subprocess.CalledProcessError as e:
        return render_template("update_status.html", message=f"Git update failed: {e}")
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
    repo_path = cfg.get("INSTALL_DIR") or os.getcwd()
    service_name = cfg.get("SERVICE_NAME", "echomosaic.service")
    # decide target: previous commit from history
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
    except subprocess.CalledProcessError as e:
        return render_template("update_status.html", message=f"Rollback failed: {e}")
    try:
        subprocess.Popen(["sudo", "systemctl", "restart", service_name])
    except OSError:
        pass
    return render_template("update_status.html", message=f"Rolled back to {target[:7]}. Restarting service...")


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

@app.route("/stream/image/<path:image_path>")
def serve_image(image_path):
    full_path = Path(IMAGE_DIR_PATH) / image_path
    if not full_path.exists():
        return "Not found", 404
    return _send_image_response(full_path)


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
    if isinstance(group_def, dict):
        members = group_def.get("streams", [])
        g_layout = group_def.get("layout") or {}
    else:
        members = list(group_def)
        g_layout = {}
    streams = {k: settings[k] for k in members if k in settings}
    # Build mosaic from group layout if provided, else default
    mosaic = default_mosaic_config()
    if g_layout:
        # safe merge
        for k in ["layout", "cols", "rows", "pip_main", "pip_pip", "pip_corner", "pip_size", "focus_mode", "focus_pos"]:
            if k in g_layout:
                mosaic[k] = g_layout[k]
    return render_template("streams.html", stream_settings=streams, mosaic_settings=mosaic)

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
