"""Global link library service."""

from __future__ import annotations

import secrets
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlparse


class GlobalLinksService:
    def __init__(
        self,
        *,
        settings: Dict[str, Any],
        save_settings_debounced: Callable[[], None],
        parse_youtube_url_details: Callable[[str], Optional[Dict[str, Any]]],
    ) -> None:
        self.settings = settings
        self.save_settings_debounced = save_settings_debounced
        self.parse_youtube_url_details = parse_youtube_url_details
        self.links_key = "_links"

    @staticmethod
    def _trimmed(value: Any) -> str:
        return value.strip() if isinstance(value, str) else ""

    def _detect_link_metadata(self, url: str) -> Dict[str, Optional[str]]:
        lowered = url.lower()
        youtube = self.parse_youtube_url_details(url)
        if youtube:
            if youtube.get("playlist_id"):
                return {"provider": "youtube", "content_type": "playlist"}
            if youtube.get("is_live"):
                return {"provider": "youtube", "content_type": "live"}
            return {"provider": "youtube", "content_type": "video"}
        if lowered.endswith(".m3u8") or lowered.endswith(".mpd"):
            return {"provider": "hls", "content_type": "live"}
        parsed = urlparse(url if "://" in url else f"https://{url}")
        if parsed.scheme in {"http", "https"} and parsed.netloc:
            return {"provider": "website", "content_type": "website"}
        return {"provider": None, "content_type": None}

    def sanitize_link(self, raw: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(raw, dict):
            return None
        label = self._trimmed(raw.get("label"))
        url = self._trimmed(raw.get("url"))
        category = self._trimmed(raw.get("category"))
        if not label or not url or not category:
            return None
        link_id = self._trimmed(raw.get("id")) or secrets.token_hex(8)
        detected = self._detect_link_metadata(url)
        provider = (detected.get("provider") or self._trimmed(raw.get("provider")).lower())
        content_type = (detected.get("content_type") or self._trimmed(raw.get("content_type")).lower())
        sanitized: Dict[str, Any] = {
            "id": link_id,
            "label": label,
            "url": url,
            "category": category,
        }
        if provider:
            sanitized["provider"] = provider
        if content_type:
            sanitized["content_type"] = content_type
        return sanitized

    def sanitize_collection(self, raw: Any) -> List[Dict[str, Any]]:
        if not isinstance(raw, list):
            return []
        cleaned: List[Dict[str, Any]] = []
        seen_ids: set[str] = set()
        for item in raw:
            sanitized = self.sanitize_link(item)
            if not sanitized:
                continue
            link_id = str(sanitized.get("id") or "").strip()
            if not link_id or link_id in seen_ids:
                sanitized["id"] = secrets.token_hex(8)
                link_id = sanitized["id"]
            seen_ids.add(link_id)
            cleaned.append(sanitized)
        cleaned.sort(key=lambda item: ((item.get("category") or "").lower(), (item.get("label") or "").lower()))
        return cleaned

    def ensure_links_defaults(self) -> None:
        self.settings[self.links_key] = self.sanitize_collection(self.settings.get(self.links_key, []))

    def list_links(self) -> List[Dict[str, Any]]:
        self.ensure_links_defaults()
        return deepcopy(self.settings.get(self.links_key, []))

    def list_links_payload(self) -> Dict[str, Any]:
        links = self.list_links()
        categories = sorted({str(link.get("category") or "").strip() for link in links if str(link.get("category") or "").strip()}, key=str.lower)
        return {"links": links, "categories": categories}

    def create_link(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        sanitized = self.sanitize_link(payload)
        if not sanitized:
            raise ValueError("Label, URL, and category are required.")
        links = self.list_links()
        links.append(sanitized)
        self.settings[self.links_key] = self.sanitize_collection(links)
        self.save_settings_debounced()
        return {"status": "success", "link": deepcopy(sanitized), **self.list_links_payload()}

    def update_link(self, link_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        target_id = self._trimmed(link_id)
        if not target_id:
            raise KeyError("Link id required")
        links = self.list_links()
        for index, existing in enumerate(links):
            if self._trimmed(existing.get("id")) != target_id:
                continue
            merged = dict(existing)
            merged.update(payload or {})
            merged["id"] = target_id
            sanitized = self.sanitize_link(merged)
            if not sanitized:
                raise ValueError("Label, URL, and category are required.")
            links[index] = sanitized
            self.settings[self.links_key] = self.sanitize_collection(links)
            self.save_settings_debounced()
            return {"status": "success", "link": deepcopy(sanitized), **self.list_links_payload()}
        raise KeyError(target_id)

    def delete_link(self, link_id: str) -> Dict[str, Any]:
        target_id = self._trimmed(link_id)
        if not target_id:
            raise KeyError("Link id required")
        links = self.list_links()
        remaining = [link for link in links if self._trimmed(link.get("id")) != target_id]
        if len(remaining) == len(links):
            raise KeyError(target_id)
        self.settings[self.links_key] = self.sanitize_collection(remaining)
        self.save_settings_debounced()
        return {"status": "success", **self.list_links_payload()}
