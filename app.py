from flask import Flask, jsonify, send_file, request, render_template, redirect, url_for
from flask_socketio import SocketIO
import json
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
from datetime import datetime
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


def _ensure_dir(path: Path) -> Path:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning("Unable to ensure directory %s: %s", path, exc)
    return path


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
    mosaic = settings.get("_mosaic", default_mosaic_config())
    groups = sorted(list(settings.get("_groups", {}).keys()))
    return render_template(
        "index.html",
        subfolders=subfolders,
        stream_settings=streams,
        mosaic_settings=mosaic,
        groups=groups,
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
                "background_blur_enabled", "background_blur_amount"]:
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
    save_settings(settings)
    socketio.emit("refresh", {"stream_id": stream_id, "config": conf})
    return jsonify({"status": "success", "new_config": conf})


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
    if stable_horde_client is None:
        return jsonify({'error': 'Stable Horde client is not configured'}), 503
    conf = settings.get(stream_id)
    if not conf:
        return jsonify({'error': f"No stream '{stream_id}' found"}), 404
    ensure_ai_defaults(conf)
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        payload = {}
    ai_settings = _sanitize_ai_settings(payload, conf[AI_SETTINGS_KEY])
    prompt = str(ai_settings.get('prompt') or '').strip()
    if not prompt:
        return jsonify({'error': 'Prompt is required'}), 400
    conf[AI_SETTINGS_KEY] = ai_settings
    conf["_ai_customized"] = not _ai_settings_match_defaults(conf[AI_SETTINGS_KEY])
    persist = bool(ai_settings.get('save_output', AI_DEFAULT_PERSIST))
    previous_state = conf.get(AI_STATE_KEY) or {}
    previous_images = list(previous_state.get('images') or [])
    previous_selected = conf.get('selected_image')
    cancel_event = threading.Event()
    with ai_jobs_lock:
        if stream_id in ai_jobs:
            return jsonify({'error': 'Generation already in progress'}), 409
        ai_jobs[stream_id] = {
            'status': 'queued',
            'job_id': None,
            'started': time.time(),
            'persisted': persist,
            'cancel_requested': False,
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
    })
    save_settings(settings)
    _emit_ai_update(stream_id, queued_state, job=ai_jobs[stream_id])
    try:
        socketio.emit('refresh', {'stream_id': stream_id, 'config': conf})
    except Exception as exc:  # pragma: no cover
        logger.debug('Socket refresh emit failed: %s', exc)
    job_options = dict(ai_settings)
    job_options['prompt'] = prompt
    worker = threading.Thread(target=_run_ai_generation, args=(stream_id, job_options, cancel_event), daemon=True)
    with ai_jobs_lock:
        controls = ai_job_controls.get(stream_id)
        if controls is not None:
            controls['thread'] = worker
            controls['cancel_event'] = cancel_event
    worker.start()
    return jsonify({'status': 'queued', 'state': conf[AI_STATE_KEY], 'job': ai_jobs[stream_id]})

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








