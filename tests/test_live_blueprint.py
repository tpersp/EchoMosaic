from __future__ import annotations

from flask import Flask

from echomosaic_app.routes.live import create_live_blueprint


def test_live_blueprint_registers_expected_routes() -> None:
    app = Flask(__name__)
    app.register_blueprint(
        create_live_blueprint(
            stream_live_handler=lambda: ("ok", 200),
            stream_live_invalidate_handler=lambda: ("ok", 200),
            legacy_stream_live_handler=lambda: ("ok", 200),
            test_embed_handler=lambda: ("ok", 200),
        )
    )

    routes = {rule.rule for rule in app.url_map.iter_rules()}
    assert "/stream/live" in routes
    assert "/stream/live/invalidate" in routes
    assert "/live" in routes
    assert "/test_embed" in routes
