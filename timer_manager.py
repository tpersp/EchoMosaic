"""Shared helpers for scheduling and presenting auto-generation timers."""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Optional

__all__ = [
    "TimerManager",
    "compute_next_trigger",
    "default_timer_config",
    "ensure_timer_config",
    "format_display_time",
    "normalize_display_label",
    "sanitize_timer_config",
]

_MONTH_LABELS = (
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
)


def compute_next_trigger(
    interval_seconds: float,
    *,
    reference: Optional[datetime] = None,
    snap_to_increment: bool = False,
) -> datetime:
    """Return the datetime for the next trigger.

    Args:
        interval_seconds: Desired interval between triggers in seconds.
        reference: Optional base time. When omitted, uses ``datetime.now()``.
        snap_to_increment: When True, align the trigger to the next whole
            multiple of the interval (relative to the Unix epoch) that is
            strictly in the future.
    """

    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be positive")

    base = reference or datetime.now()
    if not snap_to_increment:
        return base + timedelta(seconds=interval_seconds)

    interval = max(float(interval_seconds), 1.0)
    base_ts = base.timestamp()
    # Always advance at least once so we never schedule in the past.
    snapped_index = math.floor(base_ts / interval) + 1
    snapped_ts = snapped_index * interval
    snapped = datetime.fromtimestamp(snapped_ts)
    return snapped.replace(microsecond=0)


def format_display_time(moment: datetime) -> str:
    """Format a datetime in the unified display style ``19 Oct 05:21``."""

    day = f"{moment.day:02d}"
    month = _MONTH_LABELS[moment.month - 1]
    hour = f"{moment.hour:02d}"
    minute = f"{moment.minute:02d}"
    return f"{day} {month} {hour}:{minute}"


def normalize_display_label(label: Optional[str]) -> Optional[str]:
    """Return a label coerced into the unified display format when possible."""

    if not label:
        return None
    text = str(label).strip()
    if not text:
        return None
    if _looks_like_display_format(text):
        return text
    parsed = _parse_datetime(text)
    if not parsed:
        return text
    return format_display_time(parsed)


def _looks_like_display_format(text: str) -> bool:
    if len(text) != 11:
        return False
    if text[2] != " " or text[6] != " " or text[9] != ":":
        return False
    day, month, time_part = text[:2], text[3:6], text[7:]
    if not day.isdigit() or not time_part.replace(":", "").isdigit():
        return False
    return month in _MONTH_LABELS


def _parse_datetime(value: str) -> Optional[datetime]:
    candidates = [value]
    if value.endswith("Z"):
        candidates.append(value[:-1])
    for candidate in candidates:
        try:
            dt = datetime.fromisoformat(candidate)
        except ValueError:
            continue
        if dt.tzinfo is not None:
            # Convert aware datetime to naive local time for display consistency.
            dt = dt.astimezone().replace(tzinfo=None)
        return dt
    return None


CONFIG_KEY = "timer"
DEFAULT_INTERVAL_MINUTES = 10.0
MIN_INTERVAL_MINUTES = 0.1
MAX_INTERVAL_MINUTES = 24 * 60.0
MIN_OFFSET_MINUTES = 0.0


def default_timer_config() -> Dict[str, Any]:
    """Return a fresh default timer configuration."""

    return {
        "enabled": False,
        "interval": DEFAULT_INTERVAL_MINUTES,
        "offset": 0.0,
        "next_run": None,
        "last_run": None,
    }


def sanitize_timer_config(payload: Any, *, defaults: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Normalise timer configuration payload into the canonical structure."""

    baseline = dict(defaults or default_timer_config())
    incoming = dict(payload or {}) if isinstance(payload, dict) else {}

    enabled_raw = incoming.get("enabled", baseline.get("enabled", False))
    if isinstance(enabled_raw, str):
        enabled = enabled_raw.strip().lower() in {"1", "true", "yes", "on"}
    else:
        enabled = bool(enabled_raw)

    def _coerce_float(value: Any, fallback: float) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return float(fallback)
        if math.isnan(numeric) or math.isinf(numeric):
            return float(fallback)
        return float(numeric)

    interval_default = float(baseline.get("interval", DEFAULT_INTERVAL_MINUTES))
    interval_value = _coerce_float(incoming.get("interval", interval_default), interval_default)
    interval_value = max(MIN_INTERVAL_MINUTES, min(MAX_INTERVAL_MINUTES, interval_value))
    # Round to avoid storing excessively precise floats.
    interval_value = round(interval_value, 3)

    offset_default = float(baseline.get("offset", 0.0))
    offset_value = _coerce_float(incoming.get("offset", offset_default), offset_default)
    offset_value = max(MIN_OFFSET_MINUTES, offset_value)
    max_offset = max(interval_value - MIN_INTERVAL_MINUTES, MIN_OFFSET_MINUTES)
    offset_value = min(offset_value, max_offset)
    offset_value = round(offset_value, 3)

    next_raw = incoming.get("next_run", baseline.get("next_run"))
    last_raw = incoming.get("last_run", baseline.get("last_run"))

    sanitized = {
        "enabled": enabled,
        "interval": interval_value,
        "offset": offset_value,
        "next_run": normalize_display_label(next_raw),
        "last_run": normalize_display_label(last_raw),
    }
    return sanitized


def ensure_timer_config(container: Dict[str, Any], *, key: str = CONFIG_KEY, defaults: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Ensure *container* holds a sanitised timer configuration under ``key``."""

    sanitized = sanitize_timer_config(container.get(key), defaults=defaults)
    container[key] = sanitized
    return sanitized


class TimerManager:
    """Utility wrapper that encapsulates timer configuration and scheduling."""

    def __init__(
        self,
        *,
        mode: str,
        stream_id: str,
        config_owner: Dict[str, Any],
        snap_provider: Optional[Callable[[], bool]] = None,
    ) -> None:
        self.mode = (mode or "").strip().lower()
        self.stream_id = stream_id
        self._owner = config_owner
        self._snap_provider = snap_provider or (lambda: False)
        self.config = ensure_timer_config(self._owner)

    # --------------------------------------------------------------------- #
    # Configuration helpers
    # --------------------------------------------------------------------- #

    def refresh(self) -> Dict[str, Any]:
        """Re-sanitise the backing configuration."""

        self.config = ensure_timer_config(self._owner, defaults=self.config)
        return self.config

    def is_enabled(self) -> bool:
        return bool(self.config.get("enabled"))

    def interval_minutes(self) -> float:
        return float(self.config.get("interval", DEFAULT_INTERVAL_MINUTES))

    def offset_minutes(self) -> float:
        return float(self.config.get("offset", 0.0))

    def apply_updates(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Persist a partial update to the timer configuration."""

        base = dict(self.config)
        merged = dict(base)
        if isinstance(updates, dict):
            merged.update(updates)
        self.config = sanitize_timer_config(merged, defaults=base)
        self._owner[CONFIG_KEY] = self.config
        return self.config

    # --------------------------------------------------------------------- #
    # Scheduling logic
    # --------------------------------------------------------------------- #

    def _snap_enabled(self) -> bool:
        try:
            return bool(self._snap_provider())
        except Exception:
            return False

    def compute_next(
        self,
        *,
        reference: Optional[datetime] = None,
        snap_to_increment: Optional[bool] = None,
    ) -> Optional[datetime]:
        """Return the next trigger datetime or ``None`` when disabled."""

        self.refresh()
        if not self.is_enabled():
            return None
        snap = self._snap_enabled() if snap_to_increment is None else bool(snap_to_increment)
        reference_dt = reference or datetime.now()
        interval_seconds = max(MIN_INTERVAL_MINUTES * 60.0, self.interval_minutes() * 60.0)
        offset_seconds = max(0.0, self.offset_minutes() * 60.0)
        if offset_seconds:
            adjusted_reference = reference_dt - timedelta(seconds=offset_seconds)
        else:
            adjusted_reference = reference_dt
        next_dt = compute_next_trigger(
            interval_seconds,
            reference=adjusted_reference,
            snap_to_increment=snap,
        )
        if offset_seconds:
            next_dt += timedelta(seconds=offset_seconds)
        return next_dt

    def update_next(self, moment: Optional[datetime]) -> None:
        """Persist the display label for the next scheduled trigger."""

        self.refresh()
        if moment is None:
            self.config["next_run"] = None
        else:
            self.config["next_run"] = format_display_time(moment)
        self._owner[CONFIG_KEY] = self.config

    def mark_trigger(self, *, when: Optional[datetime] = None) -> None:
        """Record the last time the timer fired."""

        self.refresh()
        moment = when or datetime.now()
        self.config["last_run"] = format_display_time(moment)
        self._owner[CONFIG_KEY] = self.config

    def to_payload(self) -> Dict[str, Any]:
        """Return a serialisable snapshot of the timer configuration."""

        return dict(self.config)
