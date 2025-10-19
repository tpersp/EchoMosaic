"""Shared helpers for scheduling and presenting auto-generation timers."""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Optional

__all__ = [
    "compute_next_trigger",
    "format_display_time",
    "normalize_display_label",
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
