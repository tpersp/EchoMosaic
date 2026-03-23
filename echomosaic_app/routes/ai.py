"""AI feature blueprint."""

from __future__ import annotations

from typing import Callable

from flask import Blueprint


def create_ai_blueprint(
    *,
    list_ai_presets_handler: Callable[[], object],
    create_ai_preset_handler: Callable[[], object],
    update_ai_preset_handler: Callable[[str], object],
    delete_ai_preset_handler: Callable[[str], object],
    ai_loras_handler: Callable[[], object],
    list_ai_models_handler: Callable[[], object],
    ai_status_handler: Callable[[str], object],
    latest_job_handler: Callable[[str], object],
    ai_generate_handler: Callable[[str], object],
    ai_cancel_handler: Callable[[str], object],
) -> Blueprint:
    blueprint = Blueprint("ai_routes", __name__)

    @blueprint.route("/ai/presets", methods=["GET"])
    def list_ai_presets():
        return list_ai_presets_handler()

    @blueprint.route("/ai/presets", methods=["POST"])
    def create_ai_preset():
        return create_ai_preset_handler()

    @blueprint.route("/ai/presets/<preset_name>", methods=["PATCH"])
    def update_ai_preset(preset_name: str):
        return update_ai_preset_handler(preset_name)

    @blueprint.route("/ai/presets/<preset_name>", methods=["DELETE"])
    def delete_ai_preset(preset_name: str):
        return delete_ai_preset_handler(preset_name)

    @blueprint.route("/ai/loras")
    def ai_loras():
        return ai_loras_handler()

    @blueprint.route("/ai/models")
    def list_ai_models():
        return list_ai_models_handler()

    @blueprint.route("/ai/status/<stream_id>")
    def ai_status(stream_id: str):
        return ai_status_handler(stream_id)

    @blueprint.route("/api/jobs/<stream_id>/latest")
    def latest_job(stream_id: str):
        return latest_job_handler(stream_id)

    @blueprint.route("/ai/generate/<stream_id>", methods=["POST"])
    def ai_generate(stream_id: str):
        return ai_generate_handler(stream_id)

    @blueprint.route("/ai/cancel/<stream_id>", methods=["POST"])
    def ai_cancel(stream_id: str):
        return ai_cancel_handler(stream_id)

    return blueprint
