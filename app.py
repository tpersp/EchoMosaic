from flask import Flask, jsonify, send_from_directory, request, render_template, redirect, url_for
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
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse

try:
    import requests
except Exception:
    requests = None

from stablehorde import StableHorde, StableHordeError

app = Flask(__name__, static_folder="static", static_url_path="/static")
socketio = SocketIO(app)

logger = logging.getLogger(__name__)

SETTINGS_FILE = "settings.json"
CONFIG_FILE = "config.json"
IMAGE_DIR = "/mnt/viewers"  # Adjust if needed

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
        "stream_url": None,
        "yt_cc": False,
        "yt_mute": True,
        "yt_quality": "auto",
        "label": "",
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
        elif stage == 'completed':
            job['status'] = 'completed'
        ai_jobs[stream_id] = job
    current_state = settings.get(stream_id, {}).get(AI_STATE_KEY, {})
    _emit_ai_update(stream_id, current_state, job=job)


def _run_ai_generation(stream_id: str, options: Dict[str, Any]) -> None:
    prompt = str(options.get('prompt') or '').strip()
    if not prompt:
        _update_ai_state(stream_id, {
            'status': 'error',
            'message': 'Prompt is required',
            'error': 'Prompt is required',
        }, persist=True)
        with ai_jobs_lock:
            ai_jobs.pop(stream_id, None)
        return

    persist = bool(options.get('save_output', AI_DEFAULT_PERSIST))
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
        ensure_ai_defaults(v)

# Ensure notes key exists
settings.setdefault("_notes", "")

# Ensure groups key exists
settings.setdefault("_groups", {})


def get_subfolders():
    subfolders = ["all"]
    if os.path.isdir(IMAGE_DIR):
        for root, dirs, _ in os.walk(IMAGE_DIR):
            for d in dirs:
                subfolders.append(os.path.relpath(os.path.join(root, d), IMAGE_DIR))
            break
    return subfolders

def list_images(folder="all"):
    """
    List all images (including .webp) in the chosen folder (or 'all'),
    then sort them alphabetically.
    """
    exts = (".png", ".jpg", ".jpeg", ".gif", ".webp")
    target_dir = IMAGE_DIR if folder == "all" else os.path.join(IMAGE_DIR, folder)
    if not os.path.exists(target_dir):
        return []
    images = []
    for root, _, files in os.walk(target_dir):
        for file in files:
            if file.lower().endswith(exts):
                relative_path = os.path.relpath(os.path.join(root, file), IMAGE_DIR)
                relative_path = relative_path.replace("\\", "/")
                images.append(relative_path)
    # Sort them alphabetically
    images.sort(key=str.lower)
    return images

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
    images = list_images(conf.get("folder", "all"))
    return render_template("single_stream.html", stream_id=key, config=conf, images=images)


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
    return jsonify(settings[stream_id])

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
                "yt_cc", "yt_mute", "yt_quality", "label"]:
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
    with ai_jobs_lock:
        if stream_id in ai_jobs:
            return jsonify({'error': 'Generation already in progress'}), 409
        ai_jobs[stream_id] = {
            'status': 'queued',
            'job_id': None,
            'started': time.time(),
            'persisted': persist,
        }
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
    worker = threading.Thread(target=_run_ai_generation, args=(stream_id, job_options), daemon=True)
    worker.start()
    return jsonify({'status': 'queued', 'state': conf[AI_STATE_KEY], 'job': ai_jobs[stream_id]})


@app.route("/mosaic-settings", methods=["POST"])
def update_mosaic_settings():
    data = request.json or {}
    layout = data.get("layout", "grid")
    cols = int(data.get("cols", settings.get("_mosaic", {}).get("cols", 2)))
    rows_val = data.get("rows", settings.get("_mosaic", {}).get("rows"))
    try:
        rows = int(rows_val) if rows_val is not None else None
    except (TypeError, ValueError):
        rows = None
    mosaic = {"layout": layout, "cols": cols, "rows": rows}
    if layout == "pip":
        mosaic.update({
            "pip_main": data.get("pip_main"),
            "pip_pip": data.get("pip_pip"),
            "pip_corner": data.get("pip_corner", "bottom-right"),
            "pip_size": int(data.get("pip_size", 25)),
        })
    settings["_mosaic"] = mosaic
    save_settings(settings)
    socketio.emit("mosaic_refresh", {"mosaic": settings["_mosaic"]})
    return jsonify({"status": "success", "mosaic": settings["_mosaic"]})

@app.route("/stream/image/<path:filename>")
def serve_stream_image(filename):
    full_path = os.path.join(IMAGE_DIR, filename)
    if not os.path.exists(full_path):
        return jsonify({"error": f"File {filename} not found"}), 404
    return send_from_directory(IMAGE_DIR, filename)


@app.route("/notes", methods=["GET", "POST"])
def notes_api():
    """Simple API to store and retrieve dashboard notes server-side."""
    if request.method == "GET":
        return jsonify({"text": settings.get("_notes", "")})
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")
    settings["_notes"] = text
    save_settings(settings)
    return jsonify({"status": "ok"})


# --- Settings export/import ---
@app.route("/settings/export", methods=["GET"])
def export_settings():
    from flask import Response
    payload = json.dumps(settings, indent=2)
    return Response(
        payload,
        mimetype="application/json",
        headers={
            "Content-Disposition": "attachment; filename=echomosaic-settings.json"
        },
    )


@app.route("/settings/import", methods=["POST"])
def import_settings():
    global settings
    data = None
    # Accept JSON body or uploaded file
    if request.files and "file" in request.files:
        try:
            data = json.load(request.files["file"])  # type: ignore[arg-type]
        except Exception:
            return jsonify({"error": "Invalid JSON file"}), 400
    else:
        data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Invalid settings payload"}), 400

    # Basic normalization similar to startup
    data.setdefault("_mosaic", default_mosaic_config())
    data.setdefault("_notes", "")
    data.setdefault("_groups", {})
    for k, v in list(data.items()):
        if not isinstance(k, str) or k.startswith("_"):
            continue
        if isinstance(v, dict):
            v.setdefault("label", k.capitalize())
            v.setdefault("shuffle", True)

    # Replace current settings and persist
    settings = data
    save_settings(settings)
    try:
        socketio.emit("streams_changed", {"action": "import"})
        socketio.emit("mosaic_refresh", {"mosaic": settings.get("_mosaic", {})})
    except Exception:
        pass
    return jsonify({"status": "ok"})

@app.route("/stream/live")
def stream_live():
    stream_id = request.args.get("stream_id", "").strip()
    if stream_id not in settings:
        return jsonify({"error": f"No stream '{stream_id}' found"}), 404

    stream_url = settings[stream_id].get("stream_url", "")
    if not stream_url:
        return jsonify({"error": "No live stream URL configured"}), 404

    # 1) Check YouTube
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

    # 2) Check Twitch
    if "twitch.tv" in stream_url:
        embed_id = stream_url.split("twitch.tv/")[1].split("/")[0]
        return jsonify({
            "embed_type": "twitch",
            "embed_id": embed_id,
            "hls_url": None,
            "original_url": stream_url
        })

    # 3) Attempt HLS
    hls_link = try_get_hls(stream_url)
    if hls_link:
        return jsonify({
            "embed_type": "hls",
            "embed_id": None,
            "hls_url": hls_link,
            "original_url": stream_url
        })

    # 4) fallback
    return jsonify({
        "embed_type": "iframe",
        "embed_id": None,
        "hls_url": None,
        "original_url": stream_url
    })


def _classify_embed_target(url: str):
    """Return (kind, test_url) for a given input URL matching how we embed.
    kind: 'youtube' | 'twitch' | 'hls' | 'website'
    test_url: the URL we should probe for embeddability.
    """
    u = (url or "").strip()
    lu = u.lower()
    # YouTube: use the embed endpoint which is iframe-friendly
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
        # Fallback: treat generic YouTube URL as embeddable via player
        return "youtube", "https://www.youtube.com/embed/"
    # Twitch: use the player endpoint
    if "twitch.tv" in lu:
        try:
            channel = u.split("twitch.tv/")[1].split("/")[0]
        except Exception:
            channel = ""
        return "twitch", f"https://player.twitch.tv/?channel={channel}"
    # HLS/DASH
    if lu.endswith(".m3u8") or lu.endswith(".mpd"):
        return "hls", u
    return "website", u


@app.route("/test_embed", methods=["POST"])
def test_embed():
    data = request.get_json(silent=True) or {}
    url = (data.get("url") or "").strip()
    # Validate URL shape
    try:
        parsed = urlparse(url)
    except Exception:
        return jsonify({"status": "not_valid", "note": "Not valid"})
    if not parsed.scheme or not parsed.netloc or parsed.scheme not in ("http", "https"):
        return jsonify({"status": "not_valid", "note": "Not valid"})

    kind, target = _classify_embed_target(url)

    # For YouTube/Twitch, assume OK because we embed via their player endpoints
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
                # conservatively mark not showable unless wildcard present
                fa = csp_l.split("frame-ancestors", 1)[1]
                # Stop at semicolon
                fa = fa.split(";", 1)[0]
                if "'none'" in fa or ("*" not in fa and "http" not in fa and "https" not in fa and "'self'" not in fa):
                    return False, "CSP frame-ancestors"
        return True, None

    try:
        # Try HEAD first, then fall back to GET for servers that don't support it
        resp = requests.head(target, allow_redirects=True, timeout=6)
        if resp.status_code >= 400 or resp.status_code in (405, 501):
            resp = requests.get(target, allow_redirects=True, timeout=8, stream=True)
        ok_headers, reason = check_headers(resp.headers or {})
        if resp.status_code >= 400:
            return jsonify({"status": "unreachable", "http_status": resp.status_code, "note": "Unreachable"})
        if not ok_headers:
            return jsonify({"status": "not_showable", "http_status": resp.status_code, "note": "Not showable"})
        return jsonify({"status": "ok", "http_status": resp.status_code, "note": "OK", "final_url": str(resp.url)})
    except requests.exceptions.RequestException:
        return jsonify({"status": "unreachable", "note": "Unreachable"})

@app.route("/images", methods=["GET"])
def get_images():
    folder = request.args.get("folder", "all")
    imgs = list_images(folder)
    return jsonify(imgs)


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
