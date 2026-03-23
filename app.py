"""Compatibility wrapper for the EchoMosaic server.

The full server implementation now lives in ``echomosaic_app.server``.
This file intentionally stays tiny so ``python app.py`` and imports from
``app`` continue to work while the de-monolith refactor keeps feature
code out of the top-level entrypoint.
"""

from echomosaic_app.server import *  # noqa: F401,F403
from echomosaic_app.server import app, socketio
from echomosaic_app.bootstrap import run_dev_server


if __name__ == "__main__":
    run_dev_server(app, socketio, port=5000)
