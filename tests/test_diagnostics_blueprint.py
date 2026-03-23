from __future__ import annotations

from flask import Flask

from echomosaic_app.routes.diagnostics import create_diagnostics_blueprint


def test_diagnostics_blueprint_registers_expected_routes() -> None:
    app = Flask(__name__)
    app.register_blueprint(
        create_diagnostics_blueprint(
            load_config=lambda: {"SERVICE_NAME": "echomosaic.service", "logging": {}},
            get_system_stats=lambda: {"cpu_percent": 12.3},
        )
    )

    endpoints = {rule.rule for rule in app.url_map.iter_rules()}
    assert "/debug" in endpoints
    assert "/api/debug/stream" in endpoints
    assert "/api/debug/download" in endpoints
    assert "/health" in endpoints
    assert "/api/system_stats" in endpoints
