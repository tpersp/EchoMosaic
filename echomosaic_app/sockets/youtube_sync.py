"""YouTube sync Socket.IO registration."""

from __future__ import annotations

import math
from typing import Any, Callable

from flask import request


def register_youtube_sync_socket_handlers(
    *,
    socketio,
    settings,
    parse_youtube_url_details: Callable[[str], dict[str, Any] | None],
    youtube_sync_role_for_sid: Callable[[str, str], bool],
    store_youtube_sync_state: Callable[..., dict[str, Any]],
    safe_emit: Callable[..., None],
    youtube_sync_event: str,
) -> None:
    @socketio.on("youtube_state")
    def handle_youtube_state(payload: Any) -> None:
        if not isinstance(payload, dict):
            return
        stream_id = str(payload.get("stream_id") or "").strip()
        if not stream_id:
            return
        stream_conf = settings.get(stream_id)
        if not isinstance(stream_conf, dict):
            return
        stream_url = str(stream_conf.get("stream_url") or "").strip()
        youtube_details = parse_youtube_url_details(stream_url) if stream_url else None
        if not youtube_details:
            return
        content_type = (
            str((stream_conf.get("embed_metadata") or {}).get("content_type") or "").strip().lower()
            if isinstance(stream_conf.get("embed_metadata"), dict)
            else ""
        )
        if not content_type:
            if youtube_details.get("playlist_id"):
                content_type = "playlist"
            elif youtube_details.get("is_live"):
                content_type = "live"
            else:
                content_type = "video"
        if content_type not in {"video", "playlist"}:
            return
        if not youtube_sync_role_for_sid(stream_id, request.sid):
            return
        try:
            position = float(payload.get("current_time"))
        except (TypeError, ValueError):
            return
        if not math.isfinite(position) or position < 0:
            return
        position = min(position, 12 * 60 * 60)
        playlist_index = None
        raw_index = payload.get("playlist_index")
        if isinstance(raw_index, int):
            playlist_index = raw_index if raw_index >= 0 else None
        else:
            try:
                parsed_index = int(raw_index)
                playlist_index = parsed_index if parsed_index >= 0 else None
            except (TypeError, ValueError):
                playlist_index = None
        video_id = str(payload.get("video_id") or "").strip() or None
        sync_payload = store_youtube_sync_state(
            stream_id,
            {
                "playlist_id": youtube_details.get("playlist_id"),
                "video_id": youtube_details.get("video_id"),
                "content_type": content_type,
            },
            position=position,
            playlist_index=playlist_index,
            video_id=video_id,
            reporter_sid=request.sid,
        )
        safe_emit(youtube_sync_event, sync_payload, to=stream_id)

