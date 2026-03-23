"""Dashboard and stream route blueprint."""

from __future__ import annotations

from typing import Callable

from flask import Blueprint


def create_dashboard_blueprint(
    *,
    dashboard_handler: Callable[[], object],
    mosaic_streams_handler: Callable[[], object],
    render_stream_handler: Callable[[str], object],
    add_stream_handler: Callable[[], object],
    delete_stream_handler: Callable[[str], object],
    get_stream_settings_handler: Callable[[str], object],
    get_stream_playback_state_handler: Callable[[str], object],
    update_stream_settings_handler: Callable[[str], object],
    update_stream_timer_handler: Callable[[str], object],
    refresh_picsum_image_handler: Callable[[], object],
) -> Blueprint:
    blueprint = Blueprint("dashboard_routes", __name__)

    @blueprint.route("/")
    def dashboard():
        return dashboard_handler()

    @blueprint.route("/stream")
    def mosaic_streams():
        return mosaic_streams_handler()

    @blueprint.route("/stream/<name>")
    def render_stream(name: str):
        return render_stream_handler(name)

    @blueprint.route("/streams", methods=["POST"])
    def add_stream():
        return add_stream_handler()

    @blueprint.route("/streams/<stream_id>", methods=["DELETE"])
    def delete_stream(stream_id: str):
        return delete_stream_handler(stream_id)

    @blueprint.route("/get-settings/<stream_id>", methods=["GET"])
    def get_stream_settings(stream_id: str):
        return get_stream_settings_handler(stream_id)

    @blueprint.route("/stream/state/<stream_id>", methods=["GET"])
    def get_stream_playback_state(stream_id: str):
        return get_stream_playback_state_handler(stream_id)

    @blueprint.route("/settings/<stream_id>", methods=["POST"])
    def update_stream_settings(stream_id: str):
        return update_stream_settings_handler(stream_id)

    @blueprint.route("/api/timer/update/<stream_id>", methods=["POST"])
    def update_stream_timer(stream_id: str):
        return update_stream_timer_handler(stream_id)

    @blueprint.route("/picsum/refresh", methods=["POST"])
    def refresh_picsum_image():
        return refresh_picsum_image_handler()

    return blueprint
