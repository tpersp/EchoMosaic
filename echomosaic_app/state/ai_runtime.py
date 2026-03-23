"""AI runtime bootstrap and mutable runtime state."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from stablehorde import StableHorde


@dataclass
class AIRuntime:
    ai_default_model: str
    ai_default_sampler: str
    ai_default_width: int
    ai_default_height: int
    ai_default_steps: int
    ai_default_cfg: float
    ai_default_samples: int
    ai_output_subdir: str
    ai_temp_subdir: str
    ai_default_persist: bool
    ai_poll_interval: float
    ai_timeout: float
    ai_output_root: Path
    ai_temp_root: Path
    stable_horde_client: Optional[StableHorde]
    ai_jobs_lock: threading.RLock
    ai_jobs: Dict[str, Dict[str, Any]]
    ai_job_controls: Dict[str, Dict[str, Any]]
    ai_model_cache: Dict[str, Any]


def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if value is None:
        return default
    return bool(value)


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def build_ai_runtime(
    *,
    config: Dict[str, Any],
    default_model: str,
    default_sampler: str,
    default_width: int,
    default_height: int,
    default_steps: int,
    default_cfg: float,
    default_samples: int,
    output_subdir: str,
    temp_subdir: str,
    default_persist: bool,
    poll_interval: float,
    timeout: float,
    primary_ai_root: Path,
    ensure_dir,
    logger,
) -> AIRuntime:
    ai_default_model = config.get("AI_DEFAULT_MODEL", default_model) or default_model
    ai_default_sampler = config.get("AI_DEFAULT_SAMPLER", default_sampler) or default_sampler
    ai_default_width = _coerce_int(config.get("AI_DEFAULT_WIDTH"), default_width)
    ai_default_height = _coerce_int(config.get("AI_DEFAULT_HEIGHT"), default_height)
    ai_default_steps = _coerce_int(config.get("AI_DEFAULT_STEPS"), default_steps)
    ai_default_cfg = _coerce_float(config.get("AI_DEFAULT_CFG"), default_cfg)
    ai_default_samples = _coerce_int(config.get("AI_DEFAULT_SAMPLES"), default_samples)
    ai_output_subdir = config.get("AI_OUTPUT_SUBDIR", output_subdir) or output_subdir
    ai_temp_subdir = config.get("AI_TEMP_SUBDIR", temp_subdir) or temp_subdir
    ai_default_persist = _coerce_bool(config.get("AI_DEFAULT_PERSIST"), default_persist)
    ai_poll_interval = _coerce_float(config.get("AI_POLL_INTERVAL"), poll_interval)
    ai_timeout = _coerce_float(config.get("AI_TIMEOUT"), timeout)

    ai_output_root = ensure_dir(primary_ai_root)
    ai_temp_root = ensure_dir(primary_ai_root / ai_temp_subdir)

    try:
        stable_horde_client = StableHorde(
            save_dir=ai_output_root,
            persist_images=ai_default_persist,
            default_poll_interval=ai_poll_interval,
            default_timeout=ai_timeout,
            logger=logger,
        )
    except Exception as exc:  # pragma: no cover - defensive during optional setup
        logger.warning("Stable Horde client unavailable: %s", exc)
        stable_horde_client = None

    return AIRuntime(
        ai_default_model=ai_default_model,
        ai_default_sampler=ai_default_sampler,
        ai_default_width=ai_default_width,
        ai_default_height=ai_default_height,
        ai_default_steps=ai_default_steps,
        ai_default_cfg=ai_default_cfg,
        ai_default_samples=ai_default_samples,
        ai_output_subdir=ai_output_subdir,
        ai_temp_subdir=ai_temp_subdir,
        ai_default_persist=ai_default_persist,
        ai_poll_interval=ai_poll_interval,
        ai_timeout=ai_timeout,
        ai_output_root=ai_output_root,
        ai_temp_root=ai_temp_root,
        stable_horde_client=stable_horde_client,
        ai_jobs_lock=threading.RLock(),
        ai_jobs={},
        ai_job_controls={},
        ai_model_cache={"timestamp": 0.0, "data": []},
    )
