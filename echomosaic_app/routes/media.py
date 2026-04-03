"""Media management blueprint."""

from __future__ import annotations

from typing import Any, Callable

from flask import Blueprint, jsonify, render_template, request, send_file
from werkzeug.http import http_date, quote_etag


def create_media_blueprint(
    *,
    media_manager,
    media_error_type,
    media_thumb_width: int,
    media_allow_edit: bool,
    media_allowed_exts,
    media_upload_max_mb: int,
    media_upload_max_mb_getter=None,
    media_preview_enabled: bool,
    media_preview_frames: int,
    media_preview_width: int,
    media_preview_max_duration: float,
    available_media_roots,
    media_library_default: str,
    media_error_response: Callable[[Exception], Any],
    require_media_edit: Callable[[], None],
    invalidate_media_cache: Callable[[str], None],
    parse_truthy: Callable[[Any], bool],
    normalize_library_key: Callable[[Any, str], str],
    get_folder_inventory: Callable[..., list[dict[str, Any]]],
    as_int: Callable[[Any, int], int],
    virtual_leaf: Callable[[str], str],
    logger,
    app_response_class,
) -> Blueprint:
    blueprint = Blueprint("media_routes", __name__)

    @blueprint.route("/api/media/list", methods=["GET"])
    def api_media_list():
        path = request.args.get("path", "")
        page = max(1, as_int(request.args.get("page"), 1))
        page_size = as_int(request.args.get("page_size"), 100)
        page_size = max(1, min(page_size, 500))
        sort = request.args.get("sort", "name")
        order = request.args.get("order") or request.args.get("direction") or "asc"
        hide_nsfw_raw = request.args.get("hide_nsfw")
        hide_nsfw = False if hide_nsfw_raw is None else parse_truthy(hide_nsfw_raw)
        try:
            payload = media_manager.list_directory(
                path,
                hide_nsfw=hide_nsfw,
                page=page,
                page_size=page_size,
                sort=sort,
                order=order or "asc",
            )
        except Exception as exc:
            return media_error_response(exc)
        return jsonify(payload)

    @blueprint.route("/api/media/create_folder", methods=["POST"])
    def api_media_create_folder():
        payload = request.get_json(silent=True) or {}
        try:
            require_media_edit()
            parent = payload.get("path") or ""
            name = payload.get("name")
            if not name or not isinstance(name, str):
                raise media_error_type("Folder name is required", code="invalid_name")
            new_path = media_manager.create_folder(parent, name)
        except Exception as exc:
            return media_error_response(exc)
        logger.info("media.create_folder parent=%s name=%s", parent or "", name)
        invalidate_media_cache(new_path)
        return jsonify({"path": new_path, "name": virtual_leaf(new_path)})

    @blueprint.route("/api/media/rename", methods=["POST"])
    def api_media_rename():
        payload = request.get_json(silent=True) or {}
        try:
            require_media_edit()
            target_path = payload.get("path")
            new_name = payload.get("new_name") or payload.get("name")
            if not target_path or not isinstance(target_path, str):
                raise media_error_type("Path is required", code="invalid_request")
            if not new_name or not isinstance(new_name, str):
                raise media_error_type("New name is required", code="invalid_name")
            updated = media_manager.rename(target_path, new_name)
        except Exception as exc:
            return media_error_response(exc)
        logger.info("media.rename path=%s new_name=%s", target_path, new_name)
        invalidate_media_cache(target_path)
        invalidate_media_cache(updated)
        return jsonify({"path": updated, "name": virtual_leaf(updated)})

    @blueprint.route("/api/media/delete", methods=["DELETE"])
    def api_media_delete():
        payload = request.get_json(silent=True) or {}
        target = payload.get("path") if isinstance(payload, dict) else None
        if not target:
            target = request.args.get("path")
        try:
            require_media_edit()
            if not target or not isinstance(target, str):
                raise media_error_type("Path is required", code="invalid_request")
            media_manager.delete(target)
        except Exception as exc:
            return media_error_response(exc)
        logger.info("media.delete path=%s", target)
        invalidate_media_cache(target)
        return jsonify({"ok": True})

    @blueprint.route("/api/media/upload", methods=["POST"])
    def api_media_upload():
        try:
            require_media_edit()
            destination = request.form.get("path") or ""
            files = request.files.getlist("files")
            relative_paths = request.form.getlist("relative_paths")
            if not files:
                raise media_error_type("No files were provided", code="invalid_request")
            saved = media_manager.upload(destination, files, relative_paths=relative_paths)
        except Exception as exc:
            return media_error_response(exc)
        logger.info("media.upload path=%s count=%d", destination, len(saved))
        invalidate_media_cache(destination)
        return jsonify({"uploaded": saved, "count": len(saved)})

    @blueprint.route("/api/media/thumbnail", methods=["GET"])
    def api_media_thumbnail():
        path = request.args.get("path")
        if not path:
            return jsonify({"error": "Path is required", "code": "invalid_request"}), 400
        width_raw = request.args.get("w")
        height_raw = request.args.get("h")
        width = as_int(width_raw, media_thumb_width) if width_raw else media_thumb_width
        if width <= 0:
            width = media_thumb_width
        height = as_int(height_raw, 0) if height_raw else None
        if height is not None and height <= 0:
            height = None
        if request.args.get("meta"):
            try:
                metadata = media_manager.get_thumbnail_metadata(path, width=width, height=height)
            except Exception as exc:
                return media_error_response(exc)
            return jsonify(metadata)
        try:
            thumb_path, source_mtime, etag = media_manager.get_thumbnail(path, width=width, height=height)
        except Exception as exc:
            return media_error_response(exc)
        etag_value = str(etag)
        weak = False
        if etag_value.startswith("W/"):
            weak = True
            etag_value = etag_value[2:]
        etag_value = etag_value.strip('"')
        quoted = quote_etag(etag_value, weak=weak)
        incoming_etag = request.headers.get("If-None-Match")
        if incoming_etag and incoming_etag == quoted:
            response = app_response_class(status=304)
            response.headers["ETag"] = quoted
            response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
            return response
        response = send_file(thumb_path, mimetype="image/jpeg", conditional=True)
        response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        response.headers["ETag"] = quoted
        response.headers["Last-Modified"] = http_date(source_mtime)
        return response

    @blueprint.route("/api/media/preview_frame", methods=["GET"])
    def api_media_preview_frame():
        path = request.args.get("path")
        if not path:
            return jsonify({"error": "Path is required", "code": "invalid_request"}), 400
        index_raw = request.args.get("i") or request.args.get("index") or "1"
        try:
            index = int(index_raw)
        except (TypeError, ValueError):
            return jsonify({"error": "Invalid frame index", "code": "invalid_request"}), 400
        try:
            frame_path, source_mtime, etag, frame_count = media_manager.get_preview_frame(path, index)
        except Exception as exc:
            code = getattr(exc, "code", "")
            fallback_codes = {"preview_failed", "preview_skipped", "preview_disabled", "unsupported_media", "not_found"}
            if code in fallback_codes:
                try:
                    thumb_path, source_mtime, thumb_etag = media_manager.get_thumbnail(path)
                except Exception:
                    return media_error_response(exc)
                etag_value = str(thumb_etag)
                weak = False
                if etag_value.startswith("W/"):
                    weak = True
                    etag_value = etag_value[2:]
                etag_value = etag_value.strip('"')
                quoted = quote_etag(etag_value, weak=weak)
                incoming_etag = request.headers.get("If-None-Match")
                if incoming_etag and incoming_etag == quoted:
                    response = app_response_class(status=304)
                    response.headers["ETag"] = quoted
                    response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
                    response.headers["X-Preview-Frame-Count"] = "0"
                    response.headers["X-Preview-Fallback"] = "thumbnail"
                    return response
                response = send_file(thumb_path, mimetype="image/jpeg", conditional=True)
                response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
                response.headers["ETag"] = quoted
                response.headers["Last-Modified"] = http_date(source_mtime)
                response.headers["X-Preview-Frame-Count"] = "0"
                response.headers["X-Preview-Fallback"] = "thumbnail"
                return response
            return media_error_response(exc)
        etag_value = str(etag)
        weak = False
        if etag_value.startswith("W/"):
            weak = True
            etag_value = etag_value[2:]
        etag_value = etag_value.strip('"')
        quoted = quote_etag(etag_value, weak=weak)
        incoming_etag = request.headers.get("If-None-Match")
        if incoming_etag and incoming_etag == quoted:
            response = app_response_class(status=304)
            response.headers["ETag"] = quoted
            response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
            return response
        response = send_file(frame_path, mimetype="image/webp", conditional=True)
        response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        response.headers["ETag"] = quoted
        response.headers["Last-Modified"] = http_date(source_mtime)
        response.headers["X-Preview-Frame-Count"] = str(frame_count)
        return response

    @blueprint.route("/folders", methods=["GET"])
    def folders_collection():
        hide_nsfw = False if request.args.get("hide_nsfw") is None else parse_truthy(request.args.get("hide_nsfw"))
        library = normalize_library_key(request.args.get("library"), media_library_default)
        inventory = get_folder_inventory(hide_nsfw=hide_nsfw, library=library)
        return jsonify(inventory)

    @blueprint.route("/media/manage")
    def media_management_page():
        roots_payload = [
            {
                "alias": root.alias,
                "display_name": root.display_name or root.alias,
                "path": f"{root.alias}:/",
                "library": root.library,
            }
            for root in available_media_roots
        ]
        return render_template(
            "media/media_manage.html",
            media_roots=roots_payload,
            media_allow_edit=media_allow_edit,
            media_allowed_exts=media_allowed_exts,
            media_upload_max_mb=media_upload_max_mb_getter() if callable(media_upload_max_mb_getter) else media_upload_max_mb,
            media_thumb_width=media_thumb_width,
            media_preview_enabled=media_preview_enabled,
            media_preview_frames=media_preview_frames,
            media_preview_width=media_preview_width,
            media_preview_max_duration=media_preview_max_duration,
        )

    return blueprint
