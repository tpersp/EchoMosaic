"""Picsum Photos integration for EchoMosaic.

This module exposes a lightweight Flask blueprint that returns parameterized
Picsum image URLs. The dashboard can request new images without embedding URL
construction logic throughout the app.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
from urllib.parse import quote

from flask import Blueprint, jsonify, request

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


def _normalize_seed(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    sanitized = re.sub(r"[^A-Za-z0-9_.-]", "", cleaned)
    if not sanitized:
        return None
    return sanitized[:64]


def _build_picsum_url(width: int, height: int, blur: int, grayscale: bool, seed: Optional[str]) -> str:
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

    image_url = _build_picsum_url(width, height, blur, grayscale, seed)
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
