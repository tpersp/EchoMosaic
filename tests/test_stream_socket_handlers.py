from __future__ import annotations

import time

from flask import Flask
from flask_socketio import SocketIO

from echomosaic_app.sockets.streams import register_stream_socket_handlers


def test_stream_socket_handlers_register_expected_events() -> None:
    app = Flask(__name__)
    socketio = SocketIO(app, async_mode="threading")

    register_stream_socket_handlers(
        socketio=socketio,
        app_logger=type("L", (), {"debug": lambda *args, **kwargs: None, "info": lambda *args, **kwargs: None})(),
        stable_horde_log_prefix="[SH]",
        socket_noise_keywords=("broken pipe",),
        detach_listener=lambda *args: [],
        attach_listener=lambda stream_id, sid: None,
        remove_youtube_sync_subscriber=lambda *args: None,
        assign_youtube_sync_leader=lambda stream_id, sid: None,
        emit_initial_youtube_sync=lambda stream_id, sid: None,
        playback_service=None,
        stream_init_event="stream_init",
        settings={},
        safe_emit=lambda *args, **kwargs: None,
    )

    handlers = socketio.server.handlers.get("/", {})
    assert "disconnect" in handlers
    assert "ai_watch" in handlers
    assert "ai_unwatch" in handlers
    assert "stream_subscribe" in handlers
    assert "stream_unsubscribe" in handlers
    assert "video_control" in handlers


def test_video_control_fallback_emits_message_without_playback_service() -> None:
    app = Flask(__name__)
    socketio = SocketIO(app, async_mode="threading")
    emitted: list[tuple[str, dict]] = []

    register_stream_socket_handlers(
        socketio=socketio,
        app_logger=type("L", (), {"debug": lambda *args, **kwargs: None, "info": lambda *args, **kwargs: None})(),
        stable_horde_log_prefix="[SH]",
        socket_noise_keywords=("broken pipe",),
        detach_listener=lambda *args: [],
        attach_listener=lambda stream_id, sid: None,
        remove_youtube_sync_subscriber=lambda *args: None,
        assign_youtube_sync_leader=lambda stream_id, sid: None,
        emit_initial_youtube_sync=lambda stream_id, sid: None,
        playback_service=None,
        stream_init_event="stream_init",
        settings={},
        safe_emit=lambda event, payload, *args, **kwargs: emitted.append((event, payload)),
    )

    client = socketio.test_client(app)
    before = time.time()
    client.emit("video_control", {"stream_id": "stream-1", "action": "play"})
    after = time.time()

    assert emitted
    event, payload = emitted[-1]
    assert event == "video_control"
    assert payload["stream_id"] == "stream-1"
    assert payload["action"] == "play"
    assert before <= float(payload["timestamp"]) <= after
