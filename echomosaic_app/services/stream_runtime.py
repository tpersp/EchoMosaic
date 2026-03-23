"""Stream runtime metadata service."""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional


class StreamRuntimeService:
    def __init__(
        self,
        *,
        stream_runtime_lock,
        stream_runtime_state,
        video_extensions,
        ai_mode,
        ai_generate_mode,
        ai_random_mode,
        ai_specific_mode,
        media_mode_ai,
        media_mode_livestream,
        media_mode_picsum,
        media_mode_video,
        media_mode_image,
        refresh_stream_thumbnail,
    ) -> None:
        self.stream_runtime_lock = stream_runtime_lock
        self.stream_runtime_state = stream_runtime_state
        self.video_extensions = set(video_extensions)
        self.ai_mode = ai_mode
        self.ai_generate_mode = ai_generate_mode
        self.ai_random_mode = ai_random_mode
        self.ai_specific_mode = ai_specific_mode
        self.media_mode_ai = media_mode_ai
        self.media_mode_livestream = media_mode_livestream
        self.media_mode_picsum = media_mode_picsum
        self.media_mode_video = media_mode_video
        self.media_mode_image = media_mode_image
        self.refresh_stream_thumbnail = refresh_stream_thumbnail

    def detect_media_kind(self, value: Optional[str]) -> str:
        if not value:
            return "image"
        ext = os.path.splitext(str(value))[1].lower()
        if ext in self.video_extensions:
            return "video"
        return "image"

    def infer_media_mode(self, conf: Dict[str, Any]) -> str:
        mode_raw = conf.get("mode")
        mode = mode_raw.strip().lower() if isinstance(mode_raw, str) else ""
        if mode in {self.ai_mode, self.ai_generate_mode, self.ai_random_mode, self.ai_specific_mode}:
            return self.media_mode_ai
        if mode == "livestream":
            return self.media_mode_livestream
        if mode == self.media_mode_picsum:
            return self.media_mode_picsum

        selected_kind_raw = conf.get("selected_media_kind")
        selected_kind = selected_kind_raw.strip().lower() if isinstance(selected_kind_raw, str) else ""
        if selected_kind == "video":
            return self.media_mode_video

        playback_raw = conf.get("video_playback_mode")
        playback_mode = playback_raw.strip().lower() if isinstance(playback_raw, str) else ""
        if mode in ("random", "specific") and playback_mode in ("until_end", "loop"):
            return self.media_mode_video

        return self.media_mode_image

    def get_stream_runtime_state(self, stream_id: str) -> Dict[str, Any]:
        with self.stream_runtime_lock:
            entry = self.stream_runtime_state.get(stream_id)
            return dict(entry) if entry else {}

    def get_runtime_thumbnail_payload(self, stream_id: str) -> Optional[Dict[str, Any]]:
        with self.stream_runtime_lock:
            entry = self.stream_runtime_state.get(stream_id)
            if not entry:
                return None
            record = entry.get("thumbnail")
            if not isinstance(record, dict):
                return None
            payload = {k: v for k, v in record.items() if not k.startswith("_")}
            return payload or None

    def update_stream_runtime_state(
        self,
        stream_id: str,
        *,
        path: Optional[str] = None,
        kind: Optional[str] = None,
        media_mode: Optional[str] = None,
        stream_url: Optional[str] = None,
        source: str = "unknown",
        force_thumbnail: bool = False,
    ) -> Optional[Dict[str, Any]]:
        if not stream_id or stream_id.startswith("_"):
            return None
        normalized_mode = media_mode.strip().lower() if isinstance(media_mode, str) else None
        normalized_kind = kind.strip().lower() if isinstance(kind, str) else None
        resolved_path = path if path not in ("", None) else None
        if resolved_path and not normalized_kind:
            normalized_kind = self.detect_media_kind(resolved_path)
        timestamp = time.time()
        changed = False
        existing_thumbnail: Optional[Dict[str, Any]] = None
        with self.stream_runtime_lock:
            entry = self.stream_runtime_state.setdefault(stream_id, {})
            previous_mode = entry.get("media_mode")
            previous_path = entry.get("path")
            previous_kind = entry.get("kind")
            previous_url = entry.get("stream_url")
            existing_thumbnail = dict(entry["thumbnail"]) if isinstance(entry.get("thumbnail"), dict) else None

            if normalized_mode is not None:
                if previous_mode != normalized_mode:
                    changed = True
                entry["media_mode"] = normalized_mode
            elif "media_mode" not in entry:
                entry["media_mode"] = None

            if path is not None:
                if resolved_path is None:
                    if previous_path is not None:
                        changed = True
                    entry.pop("path", None)
                    if "kind" in entry:
                        entry.pop("kind", None)
                else:
                    if previous_path != resolved_path:
                        changed = True
                    entry["path"] = resolved_path
                    detected_kind = normalized_kind or self.detect_media_kind(resolved_path)
                    if previous_kind != detected_kind:
                        changed = True
                    entry["kind"] = detected_kind
            elif normalized_kind is not None:
                if previous_kind != normalized_kind:
                    changed = True
                entry["kind"] = normalized_kind

            if stream_url is not None:
                if isinstance(stream_url, str):
                    candidate = stream_url.strip()
                    normalized_url = candidate or None
                else:
                    normalized_url = None
                if previous_url != normalized_url:
                    changed = True
                entry["stream_url"] = normalized_url

            entry["timestamp"] = timestamp
            entry["source"] = source

        if not changed and not force_thumbnail:
            return self.get_runtime_thumbnail_payload(stream_id)
        thumbnail_info = self.refresh_stream_thumbnail(stream_id, force=force_thumbnail)
        if thumbnail_info is None and existing_thumbnail:
            return {k: v for k, v in existing_thumbnail.items() if not k.startswith("_")}
        return thumbnail_info

    def runtime_timestamp_to_iso(self, ts: Optional[float]) -> Optional[str]:
        if not ts:
            return None
        try:
            return (
                datetime.fromtimestamp(ts, tz=timezone.utc)
                .replace(microsecond=0)
                .isoformat()
                .replace("+00:00", "Z")
            )
        except (ValueError, OSError, OverflowError):
            return None
