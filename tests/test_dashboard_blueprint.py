from __future__ import annotations

from flask import Flask

from echomosaic_app.routes.dashboard import create_dashboard_blueprint


def test_dashboard_blueprint_registers_expected_routes() -> None:
    app = Flask(__name__)
    app.register_blueprint(
        create_dashboard_blueprint(
            dashboard_handler=lambda: ("ok", 200),
            mosaic_streams_handler=lambda: ("ok", 200),
            render_stream_handler=lambda name: ("ok", 200),
            add_stream_handler=lambda: ("ok", 200),
            delete_stream_handler=lambda stream_id: ("ok", 200),
            get_stream_settings_handler=lambda stream_id: ("ok", 200),
            get_stream_playback_state_handler=lambda stream_id: ("ok", 200),
            update_stream_settings_handler=lambda stream_id: ("ok", 200),
            update_stream_timer_handler=lambda stream_id: ("ok", 200),
            refresh_picsum_image_handler=lambda: ("ok", 200),
        )
    )

    endpoints = {rule.rule for rule in app.url_map.iter_rules()}
    assert "/" in endpoints
    assert "/stream" in endpoints
    assert "/stream/<name>" in endpoints
    assert "/streams" in endpoints
    assert "/streams/<stream_id>" in endpoints
    assert "/get-settings/<stream_id>" in endpoints
    assert "/stream/state/<stream_id>" in endpoints
    assert "/settings/<stream_id>" in endpoints
    assert "/api/timer/update/<stream_id>" in endpoints
    assert "/picsum/refresh" in endpoints
