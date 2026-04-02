"""YouTube/embed metadata and sync-state service."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse


class YouTubeEmbedService:
    def __init__(
        self,
        *,
        requests_module,
        youtube_dl_cls=None,
        eventlet_module,
        logger,
        youtube_domains,
        youtube_oembed_endpoint: str,
        youtube_oembed_cache_ttl: float,
        youtube_live_probe_cache_ttl: float,
        youtube_playlist_cache_ttl: float = 900.0,
        youtube_live_probe_max_bytes: int,
        youtube_live_html_markers,
        youtube_oembed_cache,
        youtube_oembed_cache_lock,
        youtube_live_probe_cache,
        youtube_live_probe_cache_lock,
        youtube_playlist_cache=None,
        youtube_playlist_cache_lock=None,
        youtube_sync_state_lock,
        youtube_sync_state,
        youtube_sync_subscribers,
        youtube_sync_leaders,
        youtube_in_flight,
        youtube_in_flight_lock,
        stream_runtime_lock,
        stream_runtime_state,
        safe_emit,
        youtube_sync_role_event: str,
        youtube_sync_max_age_seconds: float,
        media_mode_livestream: str,
    ) -> None:
        self.requests = requests_module
        self.youtube_dl_cls = youtube_dl_cls
        self.eventlet = eventlet_module
        self.logger = logger
        self.youtube_domains = set(youtube_domains)
        self.youtube_oembed_endpoint = youtube_oembed_endpoint
        self.youtube_oembed_cache_ttl = float(youtube_oembed_cache_ttl)
        self.youtube_live_probe_cache_ttl = float(youtube_live_probe_cache_ttl)
        self.youtube_playlist_cache_ttl = float(youtube_playlist_cache_ttl)
        self.youtube_live_probe_max_bytes = int(youtube_live_probe_max_bytes)
        self.youtube_live_html_markers = tuple(youtube_live_html_markers)
        self.youtube_oembed_cache = youtube_oembed_cache
        self.youtube_oembed_cache_lock = youtube_oembed_cache_lock
        self.youtube_live_probe_cache = youtube_live_probe_cache
        self.youtube_live_probe_cache_lock = youtube_live_probe_cache_lock
        self.youtube_playlist_cache = youtube_playlist_cache if youtube_playlist_cache is not None else {}
        self.youtube_playlist_cache_lock = youtube_playlist_cache_lock if youtube_playlist_cache_lock is not None else youtube_oembed_cache_lock
        self.youtube_sync_state_lock = youtube_sync_state_lock
        self.youtube_sync_state = youtube_sync_state
        self.youtube_sync_subscribers = youtube_sync_subscribers
        self.youtube_sync_leaders = youtube_sync_leaders
        self.youtube_in_flight = youtube_in_flight
        self.youtube_in_flight_lock = youtube_in_flight_lock
        self.stream_runtime_lock = stream_runtime_lock
        self.stream_runtime_state = stream_runtime_state
        self.safe_emit = safe_emit
        self.youtube_sync_role_event = youtube_sync_role_event
        self.youtube_sync_max_age_seconds = float(youtube_sync_max_age_seconds)
        self.media_mode_livestream = media_mode_livestream

    def _oembed_like_payload(self, raw: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(raw, dict):
            return None
        payload: Dict[str, Any] = {}
        title = raw.get("title")
        if isinstance(title, str) and title.strip():
            payload["title"] = title.strip()
        provider = raw.get("provider_name") or raw.get("provider")
        if isinstance(provider, str) and provider.strip():
            payload["provider_name"] = provider.strip()
        author = raw.get("author_name")
        if isinstance(author, str) and author.strip():
            payload["author_name"] = author.strip()
        thumbnail = raw.get("thumbnail_url")
        if isinstance(thumbnail, str) and thumbnail.strip():
            payload["thumbnail_url"] = thumbnail.strip()
        oembed_type = raw.get("oembed_type") or raw.get("type")
        if isinstance(oembed_type, str) and oembed_type.strip():
            payload["type"] = oembed_type.strip().lower()
        if raw.get("is_live") is not None:
            payload["is_live"] = raw.get("is_live")
        if raw.get("content_type") is not None:
            payload["content_type"] = raw.get("content_type")
        return payload

    def sanitize_embed_metadata(self, raw: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(raw, dict):
            return None
        title = raw.get("title")
        title = title.strip() or None if isinstance(title, str) else None
        content_type = raw.get("content_type")
        if isinstance(content_type, str):
            lowered = content_type.strip().lower()
            content_type = lowered if lowered in {"video", "playlist", "live"} else None
        else:
            content_type = None
        provider = raw.get("provider") or raw.get("provider_name")
        provider = provider.strip() or None if isinstance(provider, str) else None
        video_id = raw.get("video_id")
        video_id = video_id.strip() or None if isinstance(video_id, str) else None
        playlist_id = raw.get("playlist_id")
        playlist_id = playlist_id.strip() or None if isinstance(playlist_id, str) else None
        start_index = raw.get("start_index")
        if isinstance(start_index, int):
            index_value = start_index
        else:
            try:
                index_value = int(start_index)
            except (TypeError, ValueError):
                index_value = None
        is_live_raw = raw.get("is_live")
        is_live = bool(is_live_raw) if is_live_raw is not None else (content_type == "live")
        if is_live:
            content_type = "live"
        canonical_url = raw.get("canonical_url") or raw.get("url")
        canonical_url = canonical_url.strip() or None if isinstance(canonical_url, str) else None
        thumbnail_url = raw.get("thumbnail_url")
        thumbnail_url = thumbnail_url.strip() or None if isinstance(thumbnail_url, str) else None
        author_name = raw.get("author_name")
        author_name = author_name.strip() or None if isinstance(author_name, str) else None
        fetched_at = raw.get("fetched_at")
        fetched_at = fetched_at.strip() or None if isinstance(fetched_at, str) else None
        meta = {
            "title": title,
            "content_type": content_type,
            "provider": provider or "YouTube",
            "video_id": video_id,
            "playlist_id": playlist_id,
            "start_index": index_value,
            "is_live": bool(is_live),
        }
        if canonical_url:
            meta["canonical_url"] = canonical_url
        if thumbnail_url:
            meta["thumbnail_url"] = thumbnail_url
        if author_name:
            meta["author_name"] = author_name
        if fetched_at:
            meta["fetched_at"] = fetched_at
        return meta

    def is_youtube_host(self, host: str) -> bool:
        if not host:
            return False
        host = host.lower()
        if host in self.youtube_domains:
            return True
        return any(host.endswith(f".{domain}") for domain in self.youtube_domains)

    def is_youtube_url(self, url: Optional[str]) -> bool:
        if not url or not isinstance(url, str):
            return False
        candidate = url.strip()
        if not candidate:
            return False
        if "://" not in candidate:
            candidate = f"https://{candidate}"
        try:
            parsed = urlparse(candidate)
        except ValueError:
            return False
        return self.is_youtube_host(parsed.netloc)

    def parse_youtube_url_details(self, url: str) -> Optional[Dict[str, Any]]:
        if not self.is_youtube_url(url):
            return None
        candidate = url.strip()
        if "://" not in candidate:
            candidate = f"https://{candidate}"
        try:
            parsed = urlparse(candidate)
        except ValueError:
            return None
        query_map = parse_qs(parsed.query or "")
        segments = [segment for segment in parsed.path.split("/") if segment]
        video_id: Optional[str] = None
        playlist_id: Optional[str] = None
        start_index: Optional[int] = None
        start_seconds: Optional[int] = None
        if "v" in query_map and query_map["v"]:
            video_id = query_map["v"][0]
        elif parsed.netloc.endswith("youtu.be") and segments:
            video_id = segments[0]
        elif segments:
            if segments[0] in {"embed", "shorts", "live"} and len(segments) > 1:
                video_id = segments[1]
            elif segments[0] == "watch" and len(segments) > 1:
                video_id = segments[-1]
        playlist_candidates = query_map.get("list")
        if playlist_candidates:
            playlist_id = playlist_candidates[0] or None
        index_candidates = query_map.get("index")
        if index_candidates:
            try:
                start_index = int(index_candidates[0])
            except (TypeError, ValueError):
                start_index = None
        start_candidates = query_map.get("start")
        if start_candidates:
            try:
                start_seconds = int(start_candidates[0])
            except (TypeError, ValueError):
                start_seconds = None
        is_live = False
        if segments and segments[0] == "live":
            is_live = True
        else:
            live_candidate = query_map.get("live") or query_map.get("live_stream")
            if live_candidate:
                value = live_candidate[0]
                is_live = value.strip().lower() in {"1", "true", "yes", "live"} if isinstance(value, str) else bool(value)
        canonical_url = None
        if playlist_id and not video_id:
            canonical_url = f"https://www.youtube.com/playlist?list={playlist_id}"
        else:
            params = []
            if video_id:
                params.append(f"v={video_id}")
            if playlist_id:
                params.append(f"list={playlist_id}")
            if start_index is not None:
                params.append(f"index={start_index}")
            if start_seconds is not None:
                params.append(f"start={start_seconds}")
            canonical_url = "https://www.youtube.com/watch"
            if params:
                canonical_url += f"?{'&'.join(params)}"
        if playlist_id:
            embed_base = "https://www.youtube-nocookie.com/embed/videoseries"
        elif video_id:
            embed_base = f"https://www.youtube-nocookie.com/embed/{video_id}"
        else:
            embed_base = "https://www.youtube-nocookie.com/embed/"
        return {
            "original_url": candidate,
            "video_id": video_id,
            "playlist_id": playlist_id,
            "start_index": start_index,
            "start_seconds": start_seconds,
            "is_live": bool(is_live),
            "canonical_url": canonical_url,
            "embed_base": embed_base,
            "host": parsed.netloc.lower(),
            "path": parsed.path,
            "query": query_map,
        }

    def youtube_cache_key(self, details: Dict[str, Any]) -> Tuple[str, ...]:
        playlist_id = details.get("playlist_id") or ""
        video_id = details.get("video_id") or ""
        if playlist_id and video_id:
            return ("playlist_video", playlist_id, video_id)
        if playlist_id:
            return ("playlist", playlist_id)
        if video_id:
            return ("video", video_id)
        canonical = details.get("canonical_url") or details.get("original_url") or ""
        return ("url", canonical)

    def youtube_playlist_cache_key(self, details: Dict[str, Any]) -> Tuple[str, ...]:
        playlist_id = str(details.get("playlist_id") or "").strip()
        if playlist_id:
            return ("playlist", playlist_id)
        canonical = str(details.get("canonical_url") or details.get("original_url") or "").strip()
        return ("url", canonical)

    def youtube_oembed_html_says_live(self, oembed: Optional[Dict[str, Any]]) -> bool:
        if not isinstance(oembed, dict):
            return False
        html = oembed.get("html")
        if isinstance(html, str) and "live_stream" in html.lower():
            return True
        live_broadcast = oembed.get("is_live") or oembed.get("live_broadcast")
        if isinstance(live_broadcast, str):
            return live_broadcast.strip().lower() in {"1", "true", "yes", "live"}
        if isinstance(live_broadcast, (int, float, bool)):
            return bool(live_broadcast)
        return False

    def youtube_page_looks_live(self, details: Dict[str, Any]) -> Optional[bool]:
        if self.requests is None:
            return None
        cache_key = self.youtube_cache_key(details)
        now = time.time()
        with self.youtube_live_probe_cache_lock:
            cached = self.youtube_live_probe_cache.get(cache_key)
            if cached and now - cached.get("timestamp", 0) < self.youtube_live_probe_cache_ttl:
                return cached.get("result")
        url_candidates: List[str] = []
        for key in ("canonical_url", "original_url", "embed_base"):
            candidate = details.get(key)
            if isinstance(candidate, str) and candidate and candidate not in url_candidates:
                url_candidates.append(candidate)
        if not url_candidates:
            return None
        headers = {
            "User-Agent": "EchoMosaic/1.0 (+https://github.com/tpersp/EchoMosaic)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
        result: Optional[bool] = None
        for url in url_candidates:
            try:
                resp = self.requests.get(url, headers=headers, timeout=5, stream=True)
                resp.raise_for_status()
            except Exception as exc:
                self.logger.debug("YouTube live probe failed for %s: %s", url, exc)
                continue
            encoding = resp.encoding or "utf-8"
            text_bytes = bytearray()
            try:
                for chunk in resp.iter_content(chunk_size=4096, decode_unicode=False):
                    if not chunk:
                        continue
                    text_bytes.extend(chunk)
                    if len(text_bytes) >= self.youtube_live_probe_max_bytes:
                        break
            finally:
                resp.close()
            if not text_bytes:
                continue
            try:
                snippet_text = text_bytes.decode(encoding, "ignore")
            except Exception:
                snippet_text = text_bytes.decode("utf-8", "ignore")
            snippet = snippet_text.lower()
            if not snippet:
                continue
            if any(marker in snippet for marker in self.youtube_live_html_markers):
                result = True
                break
            if '"livebroadcastdetails"' in snippet and '"islive":true' in snippet:
                result = True
                break
            if '"livebroadcastdetails"' in snippet and '"islivenow":true' in snippet:
                result = True
                break
        if result is None:
            result = False
        with self.youtube_live_probe_cache_lock:
            self.youtube_live_probe_cache[cache_key] = {"timestamp": time.time(), "result": result}
        return result

    def derive_youtube_content_type(self, details: Dict[str, Any], oembed: Optional[Dict[str, Any]] = None) -> str:
        if details.get("playlist_id"):
            return "playlist"
        if details.get("is_live"):
            return "live"
        if isinstance(oembed, dict):
            raw_type = oembed.get("type")
            if isinstance(raw_type, str):
                lowered = raw_type.strip().lower()
                if lowered in {"video", "playlist", "live"}:
                    if lowered == "playlist" and details.get("playlist_id"):
                        return "playlist"
                    if lowered == "live":
                        return "live"
            if self.youtube_oembed_html_says_live(oembed):
                return "live"
        title = ""
        if isinstance(oembed, dict):
            title_candidate = oembed.get("title")
            if isinstance(title_candidate, str):
                title = title_candidate.lower()
        if " live " in f" {title} " or title.startswith("live "):
            return "live"
        if self.youtube_page_looks_live(details):
            return "live"
        return "video"

    def build_youtube_metadata(self, details: Dict[str, Any], oembed: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {}
        if isinstance(oembed, dict):
            title = oembed.get("title")
            if isinstance(title, str) and title.strip():
                metadata["title"] = title.strip()
            provider = oembed.get("provider_name")
            if isinstance(provider, str) and provider.strip():
                metadata["provider"] = provider.strip()
            author = oembed.get("author_name")
            if isinstance(author, str) and author.strip():
                metadata["author_name"] = author.strip()
            thumbnail = oembed.get("thumbnail_url")
            if isinstance(thumbnail, str) and thumbnail.strip():
                metadata["thumbnail_url"] = thumbnail.strip()
            oembed_type = oembed.get("type")
            if isinstance(oembed_type, str) and oembed_type.strip():
                metadata["oembed_type"] = oembed_type.strip().lower()
        if "provider" not in metadata:
            metadata["provider"] = "YouTube"
        metadata["video_id"] = details.get("video_id")
        metadata["playlist_id"] = details.get("playlist_id")
        metadata["start_index"] = details.get("start_index")
        metadata["start_seconds"] = details.get("start_seconds")
        metadata["canonical_url"] = details.get("canonical_url")
        content_type = self.derive_youtube_content_type(details, oembed)
        metadata["content_type"] = content_type
        metadata["is_live"] = content_type == "live"
        return metadata

    def build_youtube_playlist_item_url(
        self,
        *,
        playlist_id: str,
        video_id: Optional[str] = None,
        index: Optional[int] = None,
        start_seconds: Optional[int] = None,
    ) -> str:
        params = []
        if video_id:
            params.append(f"v={video_id}")
        if playlist_id:
            params.append(f"list={playlist_id}")
        if isinstance(index, int) and index > 0:
            params.append(f"index={index}")
        if isinstance(start_seconds, int) and start_seconds > 0:
            params.append(f"start={start_seconds}")
        base = "https://www.youtube.com/watch"
        if not video_id and playlist_id:
            base = "https://www.youtube.com/playlist"
            params = [f"list={playlist_id}"]
            if isinstance(index, int) and index > 0:
                params.append(f"index={index}")
        if params:
            return f"{base}?{'&'.join(params)}"
        return base

    def _normalize_playlist_entry(
        self,
        raw: Any,
        *,
        playlist_id: str,
        fallback_index: int,
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(raw, dict):
            return None
        video_id_raw = raw.get("id") or raw.get("url") or raw.get("video_id")
        video_id = video_id_raw.strip() if isinstance(video_id_raw, str) else ""
        if video_id and ("/" in video_id or "?" in video_id):
            details = self.parse_youtube_url_details(video_id)
            if details and details.get("video_id"):
                video_id = str(details.get("video_id") or "").strip()
        title_raw = raw.get("title")
        title = title_raw.strip() if isinstance(title_raw, str) else ""
        duration_raw = raw.get("duration")
        try:
            duration = int(duration_raw) if duration_raw is not None else None
        except (TypeError, ValueError):
            duration = None
        thumbnail_raw = raw.get("thumbnail")
        thumbnail = thumbnail_raw.strip() if isinstance(thumbnail_raw, str) else None
        entry_index_raw = raw.get("playlist_index") or raw.get("playlist_autonumber") or raw.get("index")
        try:
            entry_index = int(entry_index_raw) if entry_index_raw is not None else fallback_index
        except (TypeError, ValueError):
            entry_index = fallback_index
        if entry_index <= 0:
            entry_index = fallback_index
        if not title:
            title = f"Video {entry_index}"
        if not video_id:
            webpage_url = raw.get("webpage_url") or raw.get("url")
            details = self.parse_youtube_url_details(webpage_url) if isinstance(webpage_url, str) else None
            if details and details.get("video_id"):
                video_id = str(details.get("video_id") or "").strip()
        if not video_id:
            return None
        entry = {
            "index": entry_index,
            "video_id": video_id,
            "title": title,
            "url": self.build_youtube_playlist_item_url(playlist_id=playlist_id, video_id=video_id, index=entry_index),
        }
        if thumbnail:
            entry["thumbnail_url"] = thumbnail
        if duration is not None and duration >= 0:
            entry["duration"] = duration
        return entry

    def fetch_youtube_playlist(
        self,
        url: str,
        details: Optional[Dict[str, Any]] = None,
        *,
        force: bool = False,
    ) -> Optional[Dict[str, Any]]:
        if self.youtube_dl_cls is None:
            return None
        resolved = details if isinstance(details, dict) else self.parse_youtube_url_details(url)
        if not isinstance(resolved, dict):
            return None
        playlist_id = str(resolved.get("playlist_id") or "").strip()
        if not playlist_id:
            return None
        cache_key = self.youtube_playlist_cache_key(resolved)
        now = time.time()
        with self.youtube_playlist_cache_lock:
            cached = self.youtube_playlist_cache.get(cache_key)
            if cached and not force and now - cached.get("timestamp", 0) < self.youtube_playlist_cache_ttl:
                return dict(cached.get("data") or {})

        info: Optional[Dict[str, Any]] = None
        options = {
            "quiet": True,
            "skip_download": True,
            "extract_flat": "in_playlist",
            "lazy_playlist": False,
            "noplaylist": False,
        }
        try:
            with self.youtube_dl_cls(options) as ydl:
                extracted = ydl.extract_info(url, download=False)
            info = extracted if isinstance(extracted, dict) else None
        except Exception as exc:
            self.logger.debug("YouTube playlist fetch failed for %s: %s", url, exc)
            return None
        if not info:
            return None
        title_raw = info.get("title") or info.get("playlist_title")
        title = title_raw.strip() if isinstance(title_raw, str) else ""
        entries_raw = info.get("entries")
        normalized_entries: List[Dict[str, Any]] = []
        if isinstance(entries_raw, list):
            for fallback_index, item in enumerate(entries_raw, start=1):
                normalized = self._normalize_playlist_entry(item, playlist_id=playlist_id, fallback_index=fallback_index)
                if normalized:
                    normalized_entries.append(normalized)
        normalized_entries.sort(key=lambda item: (int(item.get("index") or 0), item.get("title") or ""))
        timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        payload: Dict[str, Any] = {
            "playlist_id": playlist_id,
            "title": title or "YouTube Playlist",
            "entry_count": len(normalized_entries),
            "entries": normalized_entries,
            "fetched_at": timestamp,
            "url": self.build_youtube_playlist_item_url(playlist_id=playlist_id),
        }
        with self.youtube_playlist_cache_lock:
            self.youtube_playlist_cache[cache_key] = {"data": dict(payload), "timestamp": time.time()}
        return payload

    def resolve_youtube_playlist_current_item(
        self,
        playlist: Optional[Dict[str, Any]],
        details: Optional[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(playlist, dict):
            return None
        entries = playlist.get("entries")
        if not isinstance(entries, list) or not entries:
            return None
        metadata = metadata if isinstance(metadata, dict) else {}
        desired_video_id = str((metadata.get("video_id") or (details or {}).get("video_id") or "")).strip()
        desired_index_raw = metadata.get("start_index")
        if desired_index_raw is None and isinstance(details, dict):
            desired_index_raw = details.get("start_index")
        try:
            desired_index = int(desired_index_raw) if desired_index_raw is not None else None
        except (TypeError, ValueError):
            desired_index = None
        for entry in entries:
            if desired_video_id and str(entry.get("video_id") or "").strip() == desired_video_id:
                return dict(entry)
        for entry in entries:
            if desired_index is not None and int(entry.get("index") or 0) == desired_index:
                return dict(entry)
        return dict(entries[0])

    def youtube_oembed_lookup(self, url: str, details: Dict[str, Any], *, force: bool = False) -> Optional[Dict[str, Any]]:
        cache_key = self.youtube_cache_key(details)
        now = time.time()
        with self.youtube_oembed_cache_lock:
            cached = self.youtube_oembed_cache.get(cache_key)
            if cached and not force and now - cached.get("timestamp", 0) < self.youtube_oembed_cache_ttl:
                cached_data = dict(cached.get("data", {}))
                refreshed = self.build_youtube_metadata(details, self._oembed_like_payload(cached_data))
                fetched_at = cached_data.get("fetched_at")
                if isinstance(fetched_at, str) and fetched_at.strip():
                    refreshed["fetched_at"] = fetched_at.strip()
                if refreshed != cached_data:
                    self.youtube_oembed_cache[cache_key] = {"data": dict(refreshed), "timestamp": time.time()}
                return refreshed

        with self.youtube_in_flight_lock:
            if url in self.youtube_in_flight:
                return self.build_youtube_metadata(details, None)
            self.youtube_in_flight.add(url)

        def _async_fetch_oembed() -> None:
            try:
                response = self.requests.get(
                    self.youtube_oembed_endpoint,
                    params={"url": url, "format": "json"},
                    timeout=6,
                )
                response.raise_for_status()
                payload = response.json()
                metadata = self.build_youtube_metadata(details, payload)
                timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
                metadata["fetched_at"] = timestamp
                with self.youtube_oembed_cache_lock:
                    self.youtube_oembed_cache[cache_key] = {"data": dict(metadata), "timestamp": time.time()}
            except Exception as exc:
                self.logger.debug("Async YouTube oEmbed lookup failed for %s: %s", url, exc)
            finally:
                with self.youtube_in_flight_lock:
                    self.youtube_in_flight.discard(url)

        if self.requests is None or self.eventlet is None:
            with self.youtube_in_flight_lock:
                self.youtube_in_flight.discard(url)
            return self.build_youtube_metadata(details, None)
        self.eventlet.spawn(_async_fetch_oembed)
        return self.build_youtube_metadata(details, None)

    def set_runtime_embed_metadata(self, stream_id: str, metadata: Optional[Dict[str, Any]]) -> None:
        if not stream_id or stream_id.startswith("_"):
            return
        with self.stream_runtime_lock:
            entry = self.stream_runtime_state.setdefault(stream_id, {})
            if metadata is None:
                entry.pop("embed_metadata", None)
            else:
                entry["embed_metadata"] = dict(metadata)

    def refresh_embed_metadata(self, stream_id: str, conf: Dict[str, Any], *, force: bool = False) -> Optional[Dict[str, Any]]:
        if not isinstance(conf, dict):
            return None
        media_mode_raw = conf.get("media_mode") or conf.get("mode")
        media_mode = media_mode_raw.strip().lower() if isinstance(media_mode_raw, str) else ""
        stream_url = conf.get("stream_url")
        if media_mode not in {self.media_mode_livestream, "livestream"} or not stream_url:
            conf["embed_metadata"] = None
            self.set_runtime_embed_metadata(stream_id, None)
            return None
        details = self.parse_youtube_url_details(stream_url)
        if details is None:
            conf["embed_metadata"] = None
            self.set_runtime_embed_metadata(stream_id, None)
            return None
        metadata = self.youtube_oembed_lookup(stream_url, details, force=force)
        sanitized = self.sanitize_embed_metadata(metadata or {})
        conf["embed_metadata"] = sanitized
        self.set_runtime_embed_metadata(stream_id, sanitized)
        return sanitized

    def youtube_sync_source_signature(self, details: Optional[Dict[str, Any]]) -> Tuple[str, str, str]:
        if not isinstance(details, dict):
            return ("", "", "")
        return (
            str(details.get("playlist_id") or "").strip(),
            str(details.get("video_id") or "").strip(),
            str(details.get("content_type") or "").strip().lower(),
        )

    def get_youtube_sync_state(
        self,
        stream_id: str,
        details: Optional[Dict[str, Any]],
        *,
        max_age: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        expected = self.youtube_sync_source_signature(details)
        now = time.time()
        max_age_value = self.youtube_sync_max_age_seconds if max_age is None else float(max_age)
        with self.youtube_sync_state_lock:
            entry = self.youtube_sync_state.get(stream_id)
            if not entry or entry.get("source_signature") != expected:
                return None
            server_time = entry.get("server_time")
            if not isinstance(server_time, (int, float)):
                return None
            age = max(0.0, now - float(server_time))
            if age > max_age_value:
                return None
            synced = dict(entry)
        base_seconds = synced.get("start_seconds")
        if isinstance(base_seconds, (int, float)):
            synced["start_seconds"] = max(0.0, float(base_seconds) + age)
        synced["server_time"] = now
        return synced

    def store_youtube_sync_state(
        self,
        stream_id: str,
        details: Dict[str, Any],
        *,
        position: float,
        playlist_index: Optional[int] = None,
        video_id: Optional[str] = None,
        reporter_sid: Optional[str] = None,
    ) -> Dict[str, Any]:
        now = time.time()
        content_type = str(details.get("content_type") or "").strip().lower()
        normalized_video_id = str(video_id or details.get("video_id") or "").strip() or None
        normalized_index = playlist_index if isinstance(playlist_index, int) and playlist_index >= 0 else None
        payload: Dict[str, Any] = {
            "stream_id": stream_id,
            "content_type": content_type,
            "video_id": normalized_video_id,
            "playlist_id": str(details.get("playlist_id") or "").strip() or None,
            "playlist_index": normalized_index,
            "start_seconds": max(0.0, float(position)),
            "server_time": now,
            "source_signature": self.youtube_sync_source_signature(details),
        }
        if reporter_sid:
            payload["origin_sid"] = reporter_sid
        with self.youtube_sync_state_lock:
            self.youtube_sync_state[stream_id] = dict(payload)
        return payload

    def youtube_subscriber_ids(self, stream_id: str) -> List[str]:
        with self.youtube_sync_state_lock:
            return sorted(self.youtube_sync_subscribers.get(stream_id, set()))

    def assign_youtube_sync_leader(self, stream_id: str, sid: str) -> None:
        with self.youtube_sync_state_lock:
            subscribers = self.youtube_sync_subscribers.setdefault(stream_id, set())
            subscribers.add(sid)
            leader_sid = self.youtube_sync_leaders.get(stream_id)
            if not leader_sid or leader_sid not in subscribers:
                self.youtube_sync_leaders[stream_id] = sid
                leader_sid = sid
        self.safe_emit(self.youtube_sync_role_event, {"stream_id": stream_id, "is_leader": leader_sid == sid}, to=sid)

    def promote_youtube_sync_leader(self, stream_id: str) -> None:
        new_leader: Optional[str] = None
        with self.youtube_sync_state_lock:
            subscribers = self.youtube_sync_subscribers.get(stream_id, set())
            if subscribers:
                new_leader = sorted(subscribers)[0]
                self.youtube_sync_leaders[stream_id] = new_leader
            else:
                self.youtube_sync_leaders.pop(stream_id, None)
        if new_leader:
            self.safe_emit(self.youtube_sync_role_event, {"stream_id": stream_id, "is_leader": True}, to=new_leader)

    def remove_youtube_sync_subscriber(self, sid: str, stream_id: Optional[str] = None) -> None:
        promotions: List[str] = []
        with self.youtube_sync_state_lock:
            stream_ids = [stream_id] if stream_id else list(self.youtube_sync_subscribers.keys())
            for current_stream_id in stream_ids:
                subscribers = self.youtube_sync_subscribers.get(current_stream_id)
                if not subscribers or sid not in subscribers:
                    continue
                subscribers.discard(sid)
                if not subscribers:
                    self.youtube_sync_subscribers.pop(current_stream_id, None)
                    self.youtube_sync_leaders.pop(current_stream_id, None)
                    continue
                if self.youtube_sync_leaders.get(current_stream_id) == sid:
                    self.youtube_sync_leaders.pop(current_stream_id, None)
                    promotions.append(current_stream_id)
        for current_stream_id in promotions:
            self.promote_youtube_sync_leader(current_stream_id)

    def youtube_sync_role_for_sid(self, stream_id: str, sid: str) -> bool:
        with self.youtube_sync_state_lock:
            return self.youtube_sync_leaders.get(stream_id) == sid
