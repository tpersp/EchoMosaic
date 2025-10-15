"""Picsum Photos integration helpers for EchoMosaic.

This module exposes a lightweight Flask blueprint (legacy) and helper
functions so the rest of the project can construct consistent Picsum URLs
without duplicating sanitisation logic.
"""

from __future__ import annotations

import secrets
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union
from urllib.parse import quote

from flask import Blueprint, jsonify, request

__all__ = [
    "register_picsum_routes",
    "build_picsum_url",
    "assign_new_picsum_to_stream",
    "configure_socketio",
    "STREAM_STATE",
]

DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080
MIN_DIMENSION = 16
MAX_DIMENSION = 4096
MIN_BLUR = 0
MAX_BLUR = 10
DEFAULT_BLUR = 0
CACHE_TTL_SECONDS = 60.0


_bp = Blueprint("picsum", __name__)


@dataclass
class _CacheEntry:
    payload: Dict[str, Any]
    expires_at: float


_CACHE: Dict[Tuple[int, int, int, bool, Optional[str]], _CacheEntry] = {}
STREAM_STATE: Dict[str, Dict[str, Any]] = {}
_socketio: Any = None


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return default


def configure_socketio(socketio: Any) -> None:
    """
    Allow the host application to provide a SocketIO instance that can be reused
    without creating circular imports.
    """
    global _socketio
    _socketio = socketio


def _normalize_seed(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    return cleaned[:64]


def build_picsum_url(
    width: Union[int, Dict[str, Any]],
    height: Optional[int] = None,
    blur: Optional[int] = None,
    grayscale: Optional[bool] = None,
    seed: Optional[str] = None,
) -> str:
    if isinstance(width, dict) and height is None and blur is None and grayscale is None:
        params = width
        width = _coerce_int(params.get("width"), DEFAULT_WIDTH)
        height = _coerce_int(params.get("height"), DEFAULT_HEIGHT)
        blur = _coerce_int(params.get("blur"), DEFAULT_BLUR)
        grayscale = _coerce_bool(params.get("grayscale"), False)
        seed = params.get("seed")
    else:
        assert height is not None and blur is not None and grayscale is not None
    width = _coerce_int(width, DEFAULT_WIDTH)
    height = _coerce_int(height, DEFAULT_HEIGHT)
    blur = _coerce_int(blur, DEFAULT_BLUR)
    grayscale = _coerce_bool(grayscale, False)
    if seed:
        base = f"https://picsum.photos/seed/{quote(seed, safe='')}/{width}/{height}"
    else:
        base = f"https://picsum.photos/{width}/{height}"
    query_parts = []
    if grayscale:
        query_parts.append("grayscale")
    if blur > 0:
        query_parts.append(f"blur={blur}")
    if not query_parts:
        return base
    return f"{base}?{'&'.join(query_parts)}"


def _get_cached_payload(key: Tuple[int, int, int, bool, Optional[str]]) -> Optional[Dict[str, Any]]:
    entry = _CACHE.get(key)
    if not entry:
        return None
    if entry.expires_at <= time.time():
        _CACHE.pop(key, None)
        return None
    return dict(entry.payload)


def _store_cache(key: Tuple[int, int, int, bool, Optional[str]], payload: Dict[str, Any]) -> None:
    _CACHE[key] = _CacheEntry(payload=dict(payload), expires_at=time.time() + CACHE_TTL_SECONDS)


def _get_stream_state(stream_id: str) -> Dict[str, Any]:
    state = STREAM_STATE.get(stream_id)
    if state is None:
        state = {
            "current_media": None,
            "last_seed": None,
            "seed_custom": False,
            "last_updated": None,
        }
        STREAM_STATE[stream_id] = state
    return state


def assign_new_picsum_to_stream(stream_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
    if not stream_id:
        raise ValueError("stream_id is required")
    params = dict(params or {})
    normalized_seed = _normalize_seed(params.get("seed"))
    seed_custom = normalized_seed is not None
    seed = normalized_seed if seed_custom else secrets.token_hex(6)
    params.update({
        "width": _coerce_int(params.get("width"), DEFAULT_WIDTH),
        "height": _coerce_int(params.get("height"), DEFAULT_HEIGHT),
        "blur": max(MIN_BLUR, min(MAX_BLUR, _coerce_int(params.get("blur"), DEFAULT_BLUR))),
        "grayscale": _coerce_bool(params.get("grayscale"), False),
        "seed": seed,
    })
    url = build_picsum_url(params)
    state = _get_stream_state(stream_id)
    state.update({
        "current_media": url,
        "last_seed": seed,
        "seed_custom": seed_custom,
        "last_updated": time.time(),
    })
    # Defer emitting socket events to the host application so the payload can
    # include additional fields required by clients.
    return {
        "url": url,
        "seed": seed,
        "seed_custom": seed_custom,
        "params": params,
        "state": dict(state),
    }


@_bp.route("/picsum/fetch", methods=["GET"])
def fetch_picsum_image():
    width = _coerce_int(request.args.get("width"), DEFAULT_WIDTH)
    height = _coerce_int(request.args.get("height"), DEFAULT_HEIGHT)
    blur = _coerce_int(request.args.get("blur"), DEFAULT_BLUR)
    grayscale = _coerce_bool(request.args.get("grayscale"), False)
    seed = _normalize_seed(request.args.get("seed"))

    width = max(MIN_DIMENSION, min(MAX_DIMENSION, width))
    height = max(MIN_DIMENSION, min(MAX_DIMENSION, height))
    blur = max(MIN_BLUR, min(MAX_BLUR, blur))

    cache_key = (width, height, blur, bool(grayscale), seed)
    cached = _get_cached_payload(cache_key)
    if cached:
        return jsonify(cached)

    image_url = build_picsum_url(width, height, blur, grayscale, seed)
    payload = {
        "source": "picsum",
        "image_url": image_url,
        "width": width,
        "height": height,
        "blur": blur,
        "grayscale": bool(grayscale),
        "seed": seed,
    }
    _store_cache(cache_key, payload)
    return jsonify(payload)


def register_picsum_routes(app) -> None:
    """Register the Picsum blueprint with the Flask app."""
    if "picsum" not in app.blueprints:
        app.register_blueprint(_bp)
