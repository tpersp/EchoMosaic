"""Settings and operations blueprint."""

from __future__ import annotations

import io
import json
from copy import deepcopy
from datetime import datetime
from typing import Any, Callable, Dict

from flask import Blueprint, jsonify, render_template, request, send_file

from echomosaic_app.services.operations import OperationsService, UpdateAlreadyRunningError


def create_settings_operations_blueprint(
    *,
    settings,
    load_config: Callable[[], Dict[str, Any]],
    media_settings_handler: Callable[[], Any],
    default_ai_settings: Callable[[], Dict[str, Any]],
    ai_fallback_defaults: Dict[str, Any],
    post_processors,
    max_loras: int,
    settings_export_payload: Callable[[], Dict[str, Any]],
    import_settings_handler: Callable[[], Any],
    update_ai_defaults_handler: Callable[[], Any],
    operations_service: OperationsService,
    logger,
) -> Blueprint:
    blueprint = Blueprint("settings_operations", __name__)

    def _require_api_key(cfg: Dict[str, Any]):
        api_key = cfg.get("API_KEY")
        if api_key and request.headers.get("X-API-Key") != api_key:
            return jsonify({"error": "Unauthorized"}), 401
        return None

    @blueprint.route("/settings/export", methods=["GET"])
    def export_settings_download():
        snapshot = settings_export_payload()
        buffer = io.BytesIO()
        buffer.write(json.dumps(snapshot, indent=2).encode("utf-8"))
        buffer.seek(0)
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        filename = f"echo-settings-{timestamp}.json"
        response = send_file(
            buffer,
            mimetype="application/json",
            as_attachment=True,
            download_name=filename,
            max_age=0,
        )
        response.headers["Cache-Control"] = "no-store, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Export-Filename"] = filename
        return response

    @blueprint.route("/settings/import", methods=["POST"])
    def import_settings():
        return import_settings_handler()

    @blueprint.route("/settings/ai-defaults", methods=["GET"])
    def get_ai_defaults():
        return jsonify(
            {
                "defaults": default_ai_settings(),
                "fallback": deepcopy(ai_fallback_defaults),
            }
        )

    @blueprint.route("/settings/ai-defaults", methods=["POST"])
    def update_ai_defaults():
        return update_ai_defaults_handler()

    @blueprint.route("/api/settings/media", methods=["GET", "POST"])
    def media_settings():
        return media_settings_handler()

    @blueprint.route("/settings")
    def app_settings():
        cfg = load_config()
        return render_template(
            "settings/settings.html",
            config=cfg,
            ai_defaults=default_ai_settings(),
            ai_fallback_defaults=ai_fallback_defaults,
            post_processors=post_processors,
            max_loras=max_loras,
        )

    @blueprint.route("/restore_points", methods=["GET", "POST"])
    def restore_points_collection():
        cfg = load_config()
        unauthorized = _require_api_key(cfg)
        if unauthorized:
            return unauthorized
        repo_path = operations_service.repo_path_from_config(cfg)
        if request.method == "GET":
            return jsonify({"restore_points": operations_service.list_restore_points(repo_path)})
        data = request.get_json(silent=True) or {}
        label = (data.get("label") or "").strip()
        if not label:
            return jsonify({"error": "Label required"}), 400
        try:
            point = operations_service.create_restore_point(repo_path, label)
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 500
        except Exception:
            logger.exception("Failed to create restore point")
            return jsonify({"error": "Failed to create restore point"}), 500
        return jsonify({"restore_point": point}), 201

    @blueprint.route("/restore_points/<point_id>", methods=["DELETE"])
    def restore_points_delete(point_id: str):
        cfg = load_config()
        unauthorized = _require_api_key(cfg)
        if unauthorized:
            return unauthorized
        repo_path = operations_service.repo_path_from_config(cfg)
        try:
            deleted = operations_service.delete_restore_point(repo_path, point_id)
        except Exception:
            logger.exception("Failed to delete restore point %s", point_id)
            return jsonify({"error": "Failed to delete restore point"}), 500
        if not deleted:
            return jsonify({"error": "Restore point not found"}), 404
        return jsonify({"status": "deleted"})

    @blueprint.route("/restore_points/<point_id>/restore", methods=["POST"])
    def restore_points_restore(point_id: str):
        cfg = load_config()
        unauthorized = _require_api_key(cfg)
        if unauthorized:
            return unauthorized
        repo_path = operations_service.repo_path_from_config(cfg)
        service_name = cfg.get("SERVICE_NAME", "echomosaic.service")
        try:
            point = operations_service.restore_point(repo_path, service_name, point_id)
        except FileNotFoundError:
            return jsonify({"error": "Restore point not found"}), 404
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 500
        return jsonify({"status": "ok", "restore_point": point})

    @blueprint.route("/update_app", methods=["POST"])
    def update_app():
        cfg = load_config()
        unauthorized = _require_api_key(cfg)
        if unauthorized:
            return unauthorized
        data = request.get_json(silent=True) or {}
        socket_id = str(data.get("socket_id") or "").strip()
        try:
            payload = operations_service.start_update(cfg, socket_id or None)
        except UpdateAlreadyRunningError as exc:
            return jsonify({"error": str(exc)}), 409
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(payload)

    @blueprint.route("/update_info", methods=["GET"])
    def update_info():
        cfg = load_config()
        unauthorized = _require_api_key(cfg)
        if unauthorized:
            return unauthorized
        force_refresh = str(request.args.get("refresh") or "").strip().lower() in {"1", "true", "yes", "on"}
        return jsonify(operations_service.read_update_info(cfg, force_refresh=force_refresh))

    @blueprint.route("/update_history", methods=["GET"])
    def update_history():
        cfg = load_config()
        unauthorized = _require_api_key(cfg)
        if unauthorized:
            return unauthorized
        return jsonify({"history": operations_service.read_update_history(cfg)})

    @blueprint.route("/rollback_app", methods=["POST"])
    def rollback_app():
        cfg = load_config()
        api_key = cfg.get("API_KEY")
        if api_key and request.headers.get("X-API-Key") != api_key:
            return "Unauthorized", 401
        repo_path = operations_service.repo_path_from_config(cfg)
        service_name = cfg.get("SERVICE_NAME", "echomosaic.service")
        payload = request.get_json(silent=True)
        restore_point_id = ""
        if isinstance(payload, dict):
            restore_point_id = str(payload.get("restore_point_id") or payload.get("restore_point") or "").strip()
        if not restore_point_id:
            form_value = request.form.get("restore_point_id") if hasattr(request, "form") else None
            restore_point_id = str(
                form_value
                or request.args.get("restore_point_id")
                or request.args.get("restore_point")
                or ""
        ).strip()
        if restore_point_id:
            try:
                message = operations_service.rollback_to_restore_point(repo_path, service_name, restore_point_id)
            except FileNotFoundError:
                return render_template("settings/update_status.html", message="Restore point not found."), 404
            except ValueError as exc:
                return render_template("settings/update_status.html", message=str(exc)), 400
            except RuntimeError as exc:
                return render_template("settings/update_status.html", message=str(exc)), 500
            return render_template("settings/update_status.html", message=message)
        try:
            message = operations_service.rollback_to_previous_history(repo_path, service_name)
        except ValueError as exc:
            return render_template("settings/update_status.html", message=str(exc)), 400
        except RuntimeError as exc:
            return render_template("settings/update_status.html", message=str(exc))
        return render_template("settings/update_status.html", message=message)

    @blueprint.route("/update")
    def update_view():
        cfg = load_config()
        return render_template("settings/update_progress.html", api_key=cfg.get("API_KEY", ""))

    return blueprint
