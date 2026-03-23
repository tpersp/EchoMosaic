"""Config and runtime bootstrap helpers for EchoMosaic."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

import config_manager
from media_manager import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS, MediaManager


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        stripped = value.strip().lower()
        if not stripped:
            return default
        return stripped in {"1", "true", "yes", "on"}
    if value is None:
        return default
    return bool(value)


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_extensions(value: Any) -> List[str]:
    if isinstance(value, (list, tuple, set)):
        raw_items = list(value)
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raw_items = []
        else:
            raw_items = [segment.strip() for segment in stripped.replace(",", " ").split() if segment.strip()]
    else:
        raw_items = []
    result: List[str] = []
    for item in raw_items:
        text = str(item).strip().lower()
        if not text:
            continue
        if not text.startswith("."):
            text = f".{text}"
        if text not in result:
            result.append(text)
    if not result:
        result = sorted(set(IMAGE_EXTENSIONS) | set(VIDEO_EXTENSIONS))
    return result


def _dedupe_media_root_aliases(roots: List[config_manager.MediaRoot]) -> List[config_manager.MediaRoot]:
    seen: Set[str] = set()
    deduped: List[config_manager.MediaRoot] = []
    for root in roots:
        alias = root.alias
        suffix = 2
        while alias in seen or alias == "all":
            alias = f"{root.alias}-{suffix}"
            suffix += 1
        seen.add(alias)
        if alias == root.alias:
            deduped.append(root)
        else:
            deduped.append(
                config_manager.MediaRoot(
                    alias=alias,
                    path=root.path,
                    display_name=root.display_name,
                    library=root.library,
                )
            )
    return deduped


@dataclass(frozen=True)
class MediaRuntime:
    config: Dict[str, Any]
    standard_media_roots: List[config_manager.MediaRoot]
    ai_media_roots: List[config_manager.MediaRoot]
    media_roots: List[config_manager.MediaRoot]
    available_media_roots: List[config_manager.MediaRoot]
    available_media_roots_by_library: Dict[str, List[config_manager.MediaRoot]]
    media_root_lookup: Dict[str, config_manager.MediaRoot]
    primary_media_root: config_manager.MediaRoot
    primary_ai_media_root: config_manager.MediaRoot
    thumbnail_cache_dir: Path
    media_management_allow_edit: bool
    media_upload_max_mb: int
    media_allowed_exts: List[str]
    media_thumb_width: int
    media_preview_enabled: bool
    media_preview_frames: int
    media_preview_width: int
    media_preview_max_duration: float
    media_preview_max_mb: int
    media_preview_max_bytes: Optional[int]
    media_manager: MediaManager
    media_upload_max_bytes: int


def build_media_runtime(
    *,
    config: Dict[str, Any],
    media_library_default: str,
    ai_media_library: str,
    thumbnail_subdir: str,
    internal_media_dirs: Iterable[str],
    nsfw_keyword: str = "nsfw",
) -> MediaRuntime:
    default_media_root_path = Path(os.path.abspath("./media")).resolve()
    default_ai_media_root_path = Path(os.path.abspath("./ai_media")).resolve()

    standard_media_roots = config_manager.build_media_roots(
        config.get("MEDIA_PATHS", []),
        library=media_library_default,
    )
    if not standard_media_roots:
        default_path = default_media_root_path
        default_alias = default_path.name or "media"
        standard_media_roots = [
            config_manager.MediaRoot(
                alias=default_alias,
                path=default_path,
                display_name=default_alias,
                library=media_library_default,
            )
        ]

    ai_media_roots = config_manager.build_media_roots(
        config.get("AI_MEDIA_PATHS", []),
        library=ai_media_library,
    )
    if not ai_media_roots:
        default_ai_path = default_ai_media_root_path
        default_ai_alias = default_ai_path.name or "ai-media"
        ai_media_roots = [
            config_manager.MediaRoot(
                alias=default_ai_alias,
                path=default_ai_path,
                display_name=default_ai_alias,
                library=ai_media_library,
            )
        ]

    media_roots = _dedupe_media_root_aliases(standard_media_roots + ai_media_roots)
    standard_media_roots = [root for root in media_roots if root.library == media_library_default]
    ai_media_roots = [root for root in media_roots if root.library == ai_media_library]

    for root in standard_media_roots + ai_media_roots:
        try:
            if root.path.resolve(strict=False) in {default_media_root_path, default_ai_media_root_path}:
                root.path.mkdir(parents=True, exist_ok=True)
        except OSError:
            continue

    config_manager.validate_media_paths([root.path.as_posix() for root in media_roots])

    available_media_roots: List[config_manager.MediaRoot] = []
    for candidate_root in media_roots:
        try:
            if candidate_root.path.exists() and candidate_root.path.is_dir() and os.access(candidate_root.path, os.R_OK):
                available_media_roots.append(candidate_root)
        except OSError:
            continue

    if not available_media_roots:
        fallback_root = standard_media_roots[0]
        try:
            fallback_root.path.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        available_media_roots = [fallback_root]

    available_media_roots_by_library: Dict[str, List[config_manager.MediaRoot]] = {
        media_library_default: [root for root in available_media_roots if root.library == media_library_default],
        ai_media_library: [root for root in available_media_roots if root.library == ai_media_library],
    }
    if not available_media_roots_by_library[media_library_default]:
        available_media_roots_by_library[media_library_default] = [standard_media_roots[0]]
    if not available_media_roots_by_library[ai_media_library]:
        fallback_ai_root = ai_media_roots[0]
        try:
            fallback_ai_root.path.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        available_media_roots_by_library[ai_media_library] = [fallback_ai_root]

    media_root_lookup = {root.alias: root for root in media_roots}
    primary_media_root = available_media_roots_by_library[media_library_default][0]
    primary_ai_media_root = available_media_roots_by_library[ai_media_library][0]
    thumbnail_cache_dir = primary_media_root.path / thumbnail_subdir

    media_management_allow_edit = _as_bool(config.get("MEDIA_MANAGEMENT_ALLOW_EDIT"), True)
    media_upload_max_mb = max(1, _as_int(config.get("MEDIA_UPLOAD_MAX_MB"), 256))
    media_allowed_exts = _normalize_extensions(config.get("MEDIA_ALLOWED_EXTS"))
    media_thumb_width = max(64, _as_int(config.get("MEDIA_THUMB_WIDTH"), 320))
    media_preview_enabled = _as_bool(config.get("MEDIA_PREVIEW_ENABLED"), True)
    media_preview_frames = max(1, _as_int(config.get("MEDIA_PREVIEW_FRAMES"), 8))
    media_preview_width = max(32, _as_int(config.get("MEDIA_PREVIEW_WIDTH"), media_thumb_width))
    preview_duration_raw = config.get("MEDIA_PREVIEW_MAX_DURATION", 300)
    try:
        media_preview_max_duration = float(preview_duration_raw)
    except (TypeError, ValueError):
        media_preview_max_duration = 300.0
    if media_preview_max_duration < 0:
        media_preview_max_duration = 0.0
    media_preview_max_mb = max(0, _as_int(config.get("MEDIA_PREVIEW_MAX_MB"), 512))
    media_preview_max_bytes: Optional[int]
    if media_preview_max_mb > 0:
        media_preview_max_bytes = media_preview_max_mb * 1024 * 1024
    else:
        media_preview_max_bytes = None

    media_manager = MediaManager(
        roots=media_roots,
        allowed_exts=media_allowed_exts,
        max_upload_mb=media_upload_max_mb,
        thumb_width=media_thumb_width,
        nsfw_keyword=nsfw_keyword,
        internal_dirs=internal_media_dirs,
        preview_enabled=media_preview_enabled,
        preview_frames=media_preview_frames,
        preview_width=media_preview_width,
        preview_max_duration=media_preview_max_duration,
        preview_max_bytes=media_preview_max_bytes,
    )

    return MediaRuntime(
        config=config,
        standard_media_roots=standard_media_roots,
        ai_media_roots=ai_media_roots,
        media_roots=media_roots,
        available_media_roots=available_media_roots,
        available_media_roots_by_library=available_media_roots_by_library,
        media_root_lookup=media_root_lookup,
        primary_media_root=primary_media_root,
        primary_ai_media_root=primary_ai_media_root,
        thumbnail_cache_dir=thumbnail_cache_dir,
        media_management_allow_edit=media_management_allow_edit,
        media_upload_max_mb=media_upload_max_mb,
        media_allowed_exts=media_allowed_exts,
        media_thumb_width=media_thumb_width,
        media_preview_enabled=media_preview_enabled,
        media_preview_frames=media_preview_frames,
        media_preview_width=media_preview_width,
        media_preview_max_duration=media_preview_max_duration,
        media_preview_max_mb=media_preview_max_mb,
        media_preview_max_bytes=media_preview_max_bytes,
        media_manager=media_manager,
        media_upload_max_bytes=media_manager.max_upload_bytes(),
    )
