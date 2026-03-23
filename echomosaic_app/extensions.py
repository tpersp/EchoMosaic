"""Shared extension instances for EchoMosaic."""

from flask_socketio import SocketIO

socketio = SocketIO(
    async_mode="eventlet",
    ping_timeout=120,
    ping_interval=25,
    cors_allowed_origins="*",
    max_http_buffer_size=100_000_000,
)

