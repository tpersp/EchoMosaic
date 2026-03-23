"""Live/embed route blueprint."""

from __future__ import annotations

from typing import Callable

from flask import Blueprint


def create_live_blueprint(
    *,
    stream_live_handler: Callable[[], object],
    stream_live_invalidate_handler: Callable[[], object],
    legacy_stream_live_handler: Callable[[], object],
    test_embed_handler: Callable[[], object],
) -> Blueprint:
    blueprint = Blueprint("live_routes", __name__)

    @blueprint.route("/stream/live")
    def stream_live():
        return stream_live_handler()

    @blueprint.route("/stream/live/invalidate", methods=["POST"])
    def stream_live_invalidate():
        return stream_live_invalidate_handler()

    @blueprint.route("/live")
    def legacy_stream_live():
        return legacy_stream_live_handler()

    @blueprint.route("/test_embed", methods=["POST"])
    def test_embed():
        return test_embed_handler()

    return blueprint
