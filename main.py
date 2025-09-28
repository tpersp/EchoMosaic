"""Compatibility wrapper to run the EchoMosaic server.

This module simply imports the Flask application from ``app`` so that
``python main.py`` works the same as ``python app.py``.  Keeping a
lightweight wrapper avoids route duplication that previously caused the
UI to break when the wrong entry point was used.
"""
import os

from app import app, socketio

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    socketio.run(app, host="0.0.0.0", port=port, debug=True)
