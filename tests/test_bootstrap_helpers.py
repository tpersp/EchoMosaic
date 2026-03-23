from __future__ import annotations

from flask import Blueprint, Flask, url_for

from echomosaic_app.bootstrap import register_blueprint_with_legacy_aliases


def test_register_blueprint_with_legacy_aliases_preserves_old_endpoint_names() -> None:
    app = Flask(__name__)
    blueprint = Blueprint("sample_routes", __name__)

    @blueprint.route("/sample")
    def sample():
        return "ok"

    register_blueprint_with_legacy_aliases(
        app,
        blueprint,
        {"legacy_sample": "sample_routes.sample"},
    )

    assert app.view_functions["legacy_sample"] is app.view_functions["sample_routes.sample"]
    assert any(rule.rule == "/sample" for rule in app.url_map.iter_rules())
    with app.test_request_context():
        assert url_for("legacy_sample") == "/sample"
