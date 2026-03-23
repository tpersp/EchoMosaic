"""Library, tags, groups, timer, and notes blueprints."""

from __future__ import annotations

from typing import Callable

from flask import Blueprint


def create_library_blueprint(
    *,
    list_tags_handler: Callable[[], object],
    create_tag_handler: Callable[[], object],
    delete_tag_handler: Callable[[str], object],
    timer_settings_handler: Callable[[], object],
    sync_timers_collection_handler: Callable[[], object],
    sync_timer_item_handler: Callable[[str], object],
    streams_meta_handler: Callable[[], object],
    groups_collection_handler: Callable[[], object],
    groups_delete_handler: Callable[[str], object],
    notes_handler: Callable[[], object],
    stream_group_handler: Callable[[str], object],
) -> Blueprint:
    blueprint = Blueprint("library_routes", __name__)

    @blueprint.route("/tags", methods=["GET"])
    def list_tags():
        return list_tags_handler()

    @blueprint.route("/tags", methods=["POST"])
    def create_tag():
        return create_tag_handler()

    @blueprint.route("/tags/<path:tag_name>", methods=["DELETE"])
    def delete_tag(tag_name: str):
        return delete_tag_handler(tag_name)

    @blueprint.route("/api/settings/timers", methods=["GET", "POST"])
    def api_timer_settings():
        return timer_settings_handler()

    @blueprint.route("/api/sync_timers", methods=["GET", "POST"])
    def sync_timers_collection():
        return sync_timers_collection_handler()

    @blueprint.route("/api/sync_timers/<timer_id>", methods=["PUT", "DELETE"])
    def sync_timer_item(timer_id: str):
        return sync_timer_item_handler(timer_id)

    @blueprint.route("/streams_meta", methods=["GET"])
    def streams_meta():
        return streams_meta_handler()

    @blueprint.route("/groups", methods=["GET", "POST"])
    def groups_collection():
        return groups_collection_handler()

    @blueprint.route("/groups/<name>", methods=["DELETE"])
    def groups_delete(name: str):
        return groups_delete_handler(name)

    @blueprint.route("/notes", methods=["GET", "POST"])
    def notes():
        return notes_handler()

    @blueprint.route("/stream/group/<name>")
    def stream_group(name: str):
        return stream_group_handler(name)

    return blueprint
