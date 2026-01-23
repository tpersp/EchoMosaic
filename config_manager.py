import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

CONFIG_FILE = Path("config.json")
DEFAULT_CONFIG_FILE = Path("config.default.json")
ENV_FILE = Path(".env")

DEFAULT_ENV_PLACEHOLDERS: Dict[str, str] = {
    "STABLE_HORDE_API_KEY": "",
    "API_KEY": "",
}

DEFAULT_CONFIG_FALLBACK: Dict[str, Any] = {
    "INSTALL_DIR": "/opt/echomosaic",
    "SERVICE_NAME": "echomosaic.service",
    "UPDATE_BRANCH": "main",
    "API_KEY": "",
    "AI_DEFAULT_MODEL": "stable_diffusion",
    "AI_DEFAULT_SAMPLER": "k_euler",
    "AI_DEFAULT_WIDTH": 512,
    "AI_DEFAULT_HEIGHT": 512,
    "AI_DEFAULT_STEPS": 30,
    "AI_DEFAULT_CFG": 7.5,
    "AI_DEFAULT_SAMPLES": 1,
    "AI_OUTPUT_SUBDIR": "ai_generated",
    "AI_TEMP_SUBDIR": "_ai_temp",
    "AI_DEFAULT_PERSIST": True,
    "AI_POLL_INTERVAL": 5.0,
    "AI_TIMEOUT": 0.0,
    "TIMER_SNAP_ENABLED": False,
    "LIVE_HLS_ASYNC": True,
    "LIVE_HLS_TTL_SECS": 3600,
    "LIVE_HLS_MAX_WORKERS": 3,
    "LIVE_HLS_ERROR_RETRY_SECS": 30,
    "MEDIA_PATHS": ["./media"],
    "MEDIA_MANAGEMENT_ALLOW_EDIT": True,
    "MEDIA_UPLOAD_MAX_MB": 256,
    "MEDIA_ALLOWED_EXTS": [
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".gif",
        ".mp4",
        ".webm",
        ".mkv",
    ],
    "MEDIA_THUMB_WIDTH": 320,
    "MEDIA_PREVIEW_ENABLED": True,
    "MEDIA_PREVIEW_FRAMES": 8,
    "MEDIA_PREVIEW_WIDTH": 320,
    "MEDIA_PREVIEW_MAX_DURATION": 300,
    "MEDIA_PREVIEW_MAX_MB": 512,
    "logging": {
        "level": "INFO",
        "retention_days": 7,
    },
}


@dataclass
class MediaRoot:
    alias: str
    path: Path
    display_name: str


def ensure_env_file(env_path: Path = ENV_FILE, placeholders: Optional[Dict[str, str]] = None) -> None:
    """Ensure a .env file exists and includes placeholders for known keys."""

    placeholders = placeholders or DEFAULT_ENV_PLACEHOLDERS
    env_path = Path(env_path)
    existing_lines: List[str]

    if env_path.exists():
        existing_lines = env_path.read_text(encoding="utf-8").splitlines()
    else:
        existing_lines = ["# Auto-generated .env"]

    existing_keys = _extract_env_keys(existing_lines)
    missing = [key for key in placeholders if key not in existing_keys]

    if not missing and env_path.exists():
        return

    if missing:
        for key in missing:
            existing_lines.append(f"{key}={placeholders[key]}")

    env_path.write_text("\n".join(existing_lines) + "\n", encoding="utf-8")


def _extract_env_keys(lines: Sequence[str]) -> List[str]:
    keys: List[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            continue
        key = stripped.split("=", 1)[0].strip()
        if key:
            keys.append(key)
    return keys


def load_env_file(env_path: Path = ENV_FILE) -> Dict[str, str]:
    """Populate os.environ with values from a .env file without overriding existing env vars."""

    env_path = Path(env_path)
    if not env_path.is_file():
        return {}
    loaded: Dict[str, str] = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, raw_value = stripped.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = raw_value.strip().strip('"').strip("'")
        if os.environ.get(key) not in (None, ""):
            continue
        os.environ[key] = value
        loaded[key] = value
    return loaded


def ensure_config_file(
    config_path: Path = CONFIG_FILE,
    default_path: Path = DEFAULT_CONFIG_FILE,
    default_fallback: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Ensure config.json exists and contains all keys from config.default.json."""

    default_data = _load_json(default_path)
    if not default_data:
        default_data = dict(default_fallback or DEFAULT_CONFIG_FALLBACK)

    config_path = Path(config_path)
    if config_path.exists():
        config_data = _load_json(config_path)
        if not config_data:
            config_data = {}
    else:
        config_data = {}

    merged = dict(config_data)
    if _merge_defaults(merged, default_data):
        _write_json(config_path, merged)
    elif not config_path.exists():
        _write_json(config_path, merged)

    return merged


def load_config(
    config_path: Path = CONFIG_FILE,
    default_path: Path = DEFAULT_CONFIG_FILE,
    env_path: Path = ENV_FILE,
) -> Dict[str, Any]:
    """Return the merged configuration with environment overrides applied."""

    ensure_env_file(env_path)
    load_env_file(env_path)
    ensure_config_file(config_path, default_path)

    default_data = _load_json(default_path) or dict(DEFAULT_CONFIG_FALLBACK)
    config_data = _load_json(config_path) or {}

    merged = _deep_merge(default_data, config_data)
    merged["MEDIA_PATHS"] = _normalize_media_paths(merged.get("MEDIA_PATHS"))

    env_overrides = _collect_environment_overrides(merged)
    for key, value in env_overrides.items():
        merged[key] = value

    merged["MEDIA_PATHS"] = _normalize_media_paths(merged.get("MEDIA_PATHS"))
    return merged


def save_config(
    data: Dict[str, Any],
    config_path: Path = CONFIG_FILE,
) -> None:
    """Persist ``data`` to ``config_path``."""

    _write_json(Path(config_path), data)


def validate_media_paths(paths: Iterable[str]) -> List[Path]:
    """Return the list of accessible media directories, logging warnings for invalid entries."""

    valid_paths: List[Path] = []
    for raw in _normalize_media_paths(paths):
        candidate = Path(raw).expanduser()
        try:
            if not candidate.exists():
                logger.warning("Media path '%s' does not exist", candidate)
                continue
            if not candidate.is_dir():
                logger.warning("Media path '%s' is not a directory", candidate)
                continue
            if not os.access(candidate, os.R_OK):
                logger.warning("Media path '%s' is not readable", candidate)
                continue
        except OSError as exc:
            logger.warning("Media path '%s' is not accessible: %s", candidate, exc)
            continue
        valid_paths.append(candidate)
    if not valid_paths:
        logger.warning("No accessible media paths were found in configuration")
    return valid_paths


def build_media_roots(paths: Iterable[str], *, log_warnings: bool = False) -> List[MediaRoot]:
    """Create MediaRoot entries with unique aliases for the provided paths."""

    roots: List[MediaRoot] = []
    seen_aliases: set[str] = set()
    seen_paths: set[str] = set()

    for index, raw in enumerate(_normalize_media_paths(paths)):
        path = Path(raw).expanduser()
        try:
            resolved = path.resolve(strict=False)
        except OSError:
            resolved = path.absolute()
        resolved_key = resolved.as_posix()
        if resolved_key in seen_paths:
            if log_warnings:
                logger.warning("Duplicate media path '%s' skipped", path)
            continue
        seen_paths.add(resolved_key)
        alias = _derive_alias(path, seen_aliases, index)
        display_name = path.name or alias
        if log_warnings:
            if not path.exists():
                logger.warning("Media path '%s' does not exist", path)
            elif not path.is_dir():
                logger.warning("Media path '%s' is not a directory", path)
            elif not os.access(path, os.R_OK):
                logger.warning("Media path '%s' is not readable", path)
        seen_aliases.add(alias)
        roots.append(MediaRoot(alias=alias, path=path, display_name=display_name))
    return roots


def _derive_alias(path: Path, seen: set[str], index: int) -> str:
    base_name = path.name.strip() or f"path{index + 1}"
    slug = re.sub(r"[^a-z0-9]+", "-", base_name.lower()).strip("-") or f"path{index + 1}"
    alias = slug
    suffix = 2
    while alias in seen or alias == "all":
        alias = f"{slug}-{suffix}"
        suffix += 1
    return alias


def _collect_environment_overrides(base: Dict[str, Any]) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    for key in base.keys():
        env_value = os.getenv(key)
        if env_value is None or env_value == "":
            continue
        if key == "MEDIA_PATHS":
            overrides[key] = _normalize_media_paths(env_value)
        else:
            overrides[key] = env_value
    media_env = os.getenv("ECHOMOSAIC_MEDIA_PATHS")
    if media_env:
        overrides["MEDIA_PATHS"] = _normalize_media_paths(media_env)
    return overrides


def _normalize_media_paths(value: Any) -> List[str]:
    if isinstance(value, (list, tuple)):
        candidates = [str(item).strip() for item in value]
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            candidates = []
        elif text.startswith("["):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    candidates = [str(item).strip() for item in parsed]
                else:
                    candidates = [text]
            except json.JSONDecodeError:
                candidates = [line.strip() for line in text.split(os.pathsep)]
        else:
            sep = ";" if ";" in text and os.pathsep != ";" else os.pathsep
            candidates = [segment.strip() for segment in text.split(sep)]
    else:
        candidates = []
    cleaned: List[str] = []
    seen: set[str] = set()
    for item in candidates:
        if not item:
            continue
        normalized = os.path.abspath(os.path.expanduser(item))
        if normalized in seen:
            continue
        seen.add(normalized)
        cleaned.append(normalized)
    if not cleaned:
        cleaned = [os.path.abspath("./media")]
    return cleaned


def _deep_merge(defaults: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key in defaults.keys() | overrides.keys():
        default_value = defaults.get(key)
        override_value = overrides.get(key)
        if isinstance(default_value, dict) and isinstance(override_value, dict):
            result[key] = _deep_merge(default_value, override_value)
        elif override_value is not None:
            result[key] = override_value
        else:
            result[key] = default_value
    return result


def _merge_defaults(target: Dict[str, Any], defaults: Dict[str, Any]) -> bool:
    changed = False
    for key, value in defaults.items():
        if key not in target:
            target[key] = value
            changed = True
        elif isinstance(value, dict) and isinstance(target.get(key), dict):
            if _merge_defaults(target[key], value):
                changed = True
    return changed


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    path = Path(path)
    try:
        if not path.is_file():
            return None
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to load JSON config '%s': %s", path, exc)
        return None


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path = Path(path)
    try:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
            handle.write("\n")
    except OSError as exc:
        logger.warning("Failed to write JSON config '%s': %s", path, exc)
