from __future__ import annotations

from flask import Flask

from echomosaic_app.routes.ai import create_ai_blueprint


def test_ai_blueprint_registers_expected_routes() -> None:
    app = Flask(__name__)
    app.register_blueprint(
        create_ai_blueprint(
            list_ai_presets_handler=lambda: ("ok", 200),
            create_ai_preset_handler=lambda: ("ok", 200),
            update_ai_preset_handler=lambda preset_name: ("ok", 200),
            delete_ai_preset_handler=lambda preset_name: ("ok", 200),
            ai_loras_handler=lambda: ("ok", 200),
            list_ai_models_handler=lambda: ("ok", 200),
            ai_status_handler=lambda stream_id: ("ok", 200),
            latest_job_handler=lambda stream_id: ("ok", 200),
            ai_generate_handler=lambda stream_id: ("ok", 200),
            ai_cancel_handler=lambda stream_id: ("ok", 200),
        )
    )

    endpoints = {rule.rule for rule in app.url_map.iter_rules()}
    assert "/ai/presets" in endpoints
    assert "/ai/presets/<preset_name>" in endpoints
    assert "/ai/loras" in endpoints
    assert "/ai/models" in endpoints
    assert "/ai/status/<stream_id>" in endpoints
    assert "/api/jobs/<stream_id>/latest" in endpoints
    assert "/ai/generate/<stream_id>" in endpoints
    assert "/ai/cancel/<stream_id>" in endpoints
