"""Bootstrap helpers for the EchoMosaic Flask application."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

from flask import Blueprint, Flask
from flask_socketio import SocketIO

from .extensions import socketio

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def create_flask_app() -> Flask:
    """Create the Flask application with repo-root template/static paths."""

    return Flask(
        __name__,
        static_folder=str(PROJECT_ROOT / "static"),
        static_url_path="/static",
        template_folder=str(PROJECT_ROOT / "templates"),
    )


def initialize_socketio(app: Flask) -> SocketIO:
    """Bind the shared Socket.IO instance to the Flask app."""

    socketio.init_app(app)
    return socketio


def register_bootstrap_features(
    app: Flask,
    *,
    configure_socketio: Callable[[Any], None],
    register_picsum_routes: Callable[[Flask], None],
) -> None:
    """Register early runtime features that still attach directly to the app."""

    configure_socketio(socketio)
    register_picsum_routes(app)


def register_blueprint_with_legacy_aliases(
    app: Flask,
    blueprint: Blueprint,
    legacy_aliases: Mapping[str, str] | None = None,
) -> None:
    """Register a blueprint and preserve legacy endpoint aliases."""

    app.register_blueprint(blueprint)
    if not legacy_aliases:
        return
    for legacy_endpoint, blueprint_endpoint in legacy_aliases.items():
        app.view_functions[legacy_endpoint] = app.view_functions[blueprint_endpoint]
        for rule in list(app.url_map.iter_rules(blueprint_endpoint)):
            alias_rule = app.url_rule_class(
                rule.rule,
                endpoint=legacy_endpoint,
                defaults=rule.defaults,
                subdomain=rule.subdomain,
                methods=rule.methods,
                build_only=rule.build_only,
                strict_slashes=rule.strict_slashes,
                merge_slashes=rule.merge_slashes,
                redirect_to=rule.redirect_to,
                alias=True,
                host=rule.host,
                websocket=rule.websocket,
            )
            app.url_map.add(alias_rule)


def run_dev_server(app: Flask, socketio_instance: SocketIO, *, port: int = 5000) -> None:
    """Run the local development server with the shared Socket.IO instance."""

    socketio_instance.run(app, host="0.0.0.0", port=port, debug=True)
