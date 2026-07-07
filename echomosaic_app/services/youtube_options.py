"""Shared YouTube request options."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional


DEFAULT_YOUTUBE_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)


def normalize_youtube_user_agent(value: Optional[str]) -> str:
    user_agent = value.strip() if isinstance(value, str) else ""
    return user_agent or DEFAULT_YOUTUBE_USER_AGENT


def normalize_youtube_cookie_file(value: Optional[str]) -> Optional[str]:
    cookie_file = value.strip() if isinstance(value, str) else ""
    if not cookie_file:
        return None
    return str(Path(cookie_file).expanduser())


def normalize_youtube_js_runtime(value: Optional[str]) -> Optional[Dict[str, Dict[str, str]]]:
    runtime = value.strip() if isinstance(value, str) else ""
    if not runtime:
        return None
    name, _, path = runtime.partition(":")
    name = name.strip().lower()
    path = path.strip()
    if not name:
        return None
    config: Dict[str, str] = {}
    if path:
        config["path"] = str(Path(path).expanduser())
    return {name: config}


def add_youtube_dl_request_options(
    options: Dict[str, Any],
    *,
    user_agent: Optional[str],
    cookie_file: Optional[str],
    js_runtime: Optional[str] = None,
    remote_components: Optional[str] = None,
) -> Dict[str, Any]:
    options["user_agent"] = normalize_youtube_user_agent(user_agent)
    normalized_cookie_file = normalize_youtube_cookie_file(cookie_file)
    if normalized_cookie_file:
        options["cookiefile"] = normalized_cookie_file
    normalized_js_runtime = normalize_youtube_js_runtime(js_runtime)
    if normalized_js_runtime:
        options["js_runtimes"] = normalized_js_runtime
    components = [
        component.strip()
        for component in (remote_components or "").split(",")
        if component.strip()
    ]
    if components:
        options["remote_components"] = set(components)
    return options
