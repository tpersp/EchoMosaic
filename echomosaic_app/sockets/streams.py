"""Stream/playback Socket.IO registration."""

from __future__ import annotations

import socket
import time
from typing import Any, Callable, Optional

from flask import request
from flask_socketio import join_room, leave_room


def register_stream_socket_handlers(
    *,
    socketio,
    app_logger,
    stable_horde_log_prefix: str,
    socket_noise_keywords,
    detach_listener: Callable[..., list[Any]],
    attach_listener: Callable[[str, str], None],
    remove_youtube_sync_subscriber: Callable[..., None],
    assign_youtube_sync_leader: Callable[[str, str], None],
    emit_initial_youtube_sync: Callable[[str, str], None],
    playback_service,
    stream_init_event: str,
    settings,
    safe_emit: Callable[..., None],
) -> None:
    @socketio.on_error_default
    def _socketio_default_error_handler(exc: Exception) -> None:
        if isinstance(exc, (OSError, socket.error, TimeoutError)):
            text = str(exc).lower()
            if any(keyword in text for keyword in socket_noise_keywords):
                app_logger.debug("[SocketIO] Ignored harmless disconnect: %s", exc)
                return
        raise exc

    @socketio.on("disconnect")
    def handle_socket_disconnect() -> None:
        sid = request.sid
        remove_youtube_sync_subscriber(sid)
        detached_jobs = detach_listener(sid)
        if not detached_jobs:
            app_logger.info(
                "%s Client %s disconnected; no active Stable Horde jobs linked.",
                stable_horde_log_prefix,
                sid,
            )
            return
        for job in detached_jobs:
            if job.stable_id:
                app_logger.info(
                    "%s Client %s disconnected; continuing job %s (%s) in background",
                    stable_horde_log_prefix,
                    sid,
                    job.stable_id,
                    job.stream_id,
                )
            else:
                app_logger.info(
                    "%s Client %s disconnected; continuing job for %s in background",
                    stable_horde_log_prefix,
                    sid,
                    job.stream_id,
                )

    @socketio.on("ai_watch")
    def handle_ai_watch(payload: Any) -> None:
        stream_id = ""
        if isinstance(payload, dict):
            stream_id = str(payload.get("stream_id") or "").strip()
        elif isinstance(payload, str):
            stream_id = payload.strip()
        if not stream_id:
            return
        attach_listener(stream_id, request.sid)

    @socketio.on("ai_unwatch")
    def handle_ai_unwatch(payload: Any) -> None:
        stream_id = ""
        if isinstance(payload, dict):
            stream_id = str(payload.get("stream_id") or "").strip()
        elif isinstance(payload, str):
            stream_id = payload.strip()
        if not stream_id:
            return
        detach_listener(request.sid, stream_id)

    @socketio.on("stream_subscribe")
    def handle_stream_subscribe(payload: Any) -> None:
        stream_id = ""
        if isinstance(payload, dict):
            stream_id = str(payload.get("stream_id") or "").strip()
        elif isinstance(payload, str):
            stream_id = payload.strip()
        if not stream_id:
            return
        join_room(stream_id)
        attach_listener(stream_id, request.sid)
        assign_youtube_sync_leader(stream_id, request.sid)
        if playback_service is not None:
            playback_service.emit_initial_state(stream_id, request.sid, stream_init_event)
        if isinstance(settings.get(stream_id), dict):
            emit_initial_youtube_sync(stream_id, request.sid)

    @socketio.on("stream_unsubscribe")
    def handle_stream_unsubscribe(payload: Any) -> None:
        stream_id = ""
        if isinstance(payload, dict):
            stream_id = str(payload.get("stream_id") or "").strip()
        elif isinstance(payload, str):
            stream_id = payload.strip()
        if not stream_id:
            return
        leave_room(stream_id)
        remove_youtube_sync_subscriber(request.sid, stream_id)
        detach_listener(request.sid, stream_id)

    @socketio.on("video_control")
    def handle_video_control(payload: Any) -> None:
        if not isinstance(payload, dict):
            return
        stream_id = str(payload.get("stream_id") or "").strip()
        if not stream_id:
            return
        action = str(payload.get("action") or "").strip().lower()
        allowed_actions = {"play", "pause", "toggle", "skip_next", "skip_prev", "set_volume"}
        if action not in allowed_actions:
            return
        volume_value: Optional[float] = None
        if action == "set_volume":
            try:
                volume_value = float(payload.get("volume"))
            except (TypeError, ValueError):
                return
        if playback_service is not None:
            playback_service.handle_control_action(stream_id, action, volume_value=volume_value)
        else:
            message: Dict[str, Any] = {
                "stream_id": stream_id,
                "action": action,
                "timestamp": time.time(),
            }
            if volume_value is not None:
                message["volume"] = max(0.0, min(1.0, volume_value))
            safe_emit("video_control", message)
