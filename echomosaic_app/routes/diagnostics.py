"""Diagnostics and system blueprints."""

from __future__ import annotations

from typing import Any, Callable

import debug_manager
from flask import Blueprint, Response, jsonify, render_template, request, stream_with_context


def create_diagnostics_blueprint(
    *,
    load_config: Callable[[], dict[str, Any]],
    get_system_stats: Callable[[], dict[str, Any]],
) -> Blueprint:
    blueprint = Blueprint("diagnostics", __name__)

    @blueprint.route("/debug")
    def debug_page() -> str:
        cfg = load_config()
        logging_config = cfg.get("logging") or {}
        service_name = cfg.get("SERVICE_NAME", "echomosaic.service")
        return render_template(
            "diagnostics/debug.html",
            service_name=service_name,
            logging_config=logging_config,
            default_initial_lines=debug_manager.DEFAULT_INITIAL_LINES,
        )

    @blueprint.route("/api/debug/stream")
    def debug_stream() -> Response:
        include_initial = request.args.get("initial", "1").strip().lower() not in {"0", "false", "no"}
        initial_limit = debug_manager.DEFAULT_INITIAL_LINES if include_initial else 0
        stream_generator = stream_with_context(debug_manager.stream_journal_follow(initial_limit=initial_limit))
        response = Response(stream_generator, mimetype="text/event-stream")
        response.headers["Cache-Control"] = "no-cache"
        response.headers["X-Accel-Buffering"] = "no"
        return response

    @blueprint.route("/api/debug/download")
    def debug_download() -> Response:
        limit = request.args.get("limit", type=int) or debug_manager.DEFAULT_DOWNLOAD_LINES
        try:
            log_data = debug_manager.get_recent_log_lines(limit=limit)
        except debug_manager.JournalAccessError as exc:
            return Response(str(exc), status=503, mimetype="text/plain")

        cfg = load_config()
        service_name = cfg.get("SERVICE_NAME", "echomosaic.service")
        safe_name = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in service_name)
        filename = f"{safe_name or 'echomosaic'}.log"

        response = Response(log_data, mimetype="text/plain")
        response.headers["Content-Disposition"] = f'attachment; filename="{filename}"'
        response.headers["Cache-Control"] = "no-store"
        return response

    @blueprint.route("/health")
    def health():
        return jsonify({"status": "ok"})

    @blueprint.route("/api/system_stats", methods=["GET"])
    def system_stats():
        return jsonify(get_system_stats())

    return blueprint
