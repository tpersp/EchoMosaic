from __future__ import annotations

import app
from echomosaic_app import server


def test_app_wrapper_reexports_server_objects() -> None:
    assert app.app is server.app
    assert app.socketio is server.socketio
