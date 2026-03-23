"""Read-side media library service."""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional


class MediaLibraryService:
    def __init__(
        self,
        *,
        settings,
        parse_truthy,
        normalize_library_key,
        list_images,
        list_media,
        infer_media_mode,
        update_stream_runtime_state,
        media_library_default: str,
    ) -> None:
        self.settings = settings
        self.parse_truthy = parse_truthy
        self.normalize_library_key = normalize_library_key
        self.list_images = list_images
        self.list_media = list_media
        self.infer_media_mode = infer_media_mode
        self.update_stream_runtime_state = update_stream_runtime_state
        self.media_library_default = media_library_default

    def get_images_payload(
        self,
        *,
        folder: str = "all",
        hide_nsfw: Any = False,
        library: Optional[str] = None,
        offset: int = 0,
        limit: Optional[int] = None,
    ) -> List[str]:
        normalized_library = self.normalize_library_key(library, self.media_library_default)
        images = self.list_images(folder, hide_nsfw=self.parse_truthy(hide_nsfw), library=normalized_library)
        if offset or limit is not None:
            end = offset + limit if limit is not None else None
            images = images[offset:end]
        return images

    def get_random_image_payload(
        self,
        *,
        folder: str = "all",
        hide_nsfw: Any = False,
        library: Optional[str] = None,
    ) -> Dict[str, str]:
        images = self.get_images_payload(folder=folder, hide_nsfw=hide_nsfw, library=library)
        if not images:
            raise LookupError("No images found")
        return {"path": random.choice(images)}

    def get_media_entries_payload(
        self,
        *,
        folder: str = "all",
        hide_nsfw: Any = False,
        kind: Optional[str] = None,
        library: Optional[str] = None,
        offset: int = 0,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        normalized_library = self.normalize_library_key(library, self.media_library_default)
        media = self.list_media(folder, hide_nsfw=self.parse_truthy(hide_nsfw), library=normalized_library)
        kind_filter = (kind or "").strip().lower()
        if kind_filter in ("image", "video"):
            media = [item for item in media if item.get("kind") == kind_filter]
        if offset or limit is not None:
            end = offset + limit if limit is not None else None
            media = media[offset:end]
        return media

    def get_random_media_payload(
        self,
        *,
        folder: str = "all",
        hide_nsfw: Any = False,
        kind: Optional[str] = None,
        library: Optional[str] = None,
        stream_id: str = "",
    ) -> Dict[str, Any]:
        entries = self.get_media_entries_payload(folder=folder, hide_nsfw=hide_nsfw, kind=kind, library=library)
        if not entries:
            raise LookupError("No media found")
        choice = dict(random.choice(entries))
        if stream_id and stream_id in self.settings and isinstance(self.settings.get(stream_id), dict):
            conf = self.settings.get(stream_id) or {}
            media_mode_value = conf.get("media_mode")
            normalized_mode = media_mode_value.strip().lower() if isinstance(media_mode_value, str) else None
            if not normalized_mode and isinstance(conf, dict):
                normalized_mode = self.infer_media_mode(conf)
            self.update_stream_runtime_state(
                stream_id,
                path=choice.get("path"),
                kind=choice.get("kind"),
                media_mode=normalized_mode,
                source="random_media",
            )
        return choice
