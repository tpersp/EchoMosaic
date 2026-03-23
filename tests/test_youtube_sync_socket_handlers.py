from __future__ import annotations

from flask import Flask
from flask_socketio import SocketIO

from echomosaic_app.sockets.youtube_sync import register_youtube_sync_socket_handlers


def test_youtube_sync_socket_handlers_register_expected_events() -> None:
    app = Flask(__name__)
    socketio = SocketIO(app, async_mode="threading")

    register_youtube_sync_socket_handlers(
        socketio=socketio,
        settings={},
        parse_youtube_url_details=lambda url: None,
        youtube_sync_role_for_sid=lambda stream_id, sid: False,
        store_youtube_sync_state=lambda *args, **kwargs: {},
        safe_emit=lambda *args, **kwargs: None,
        youtube_sync_event="youtube_sync",
    )

    handlers = socketio.server.handlers.get("/", {})
    assert "youtube_state" in handlers
