from __future__ import annotations

from flask import Flask

from echomosaic_app.routes.library import create_library_blueprint


def test_library_blueprint_registers_expected_routes() -> None:
    app = Flask(__name__)
    app.register_blueprint(
        create_library_blueprint(
            list_tags_handler=lambda: ("ok", 200),
            create_tag_handler=lambda: ("ok", 200),
            delete_tag_handler=lambda tag_name: (tag_name, 200),
            timer_settings_handler=lambda: ("ok", 200),
            sync_timers_collection_handler=lambda: ("ok", 200),
            sync_timer_item_handler=lambda timer_id: (timer_id, 200),
            streams_meta_handler=lambda: ("ok", 200),
            groups_collection_handler=lambda: ("ok", 200),
            groups_delete_handler=lambda name: (name, 200),
            notes_handler=lambda: ("ok", 200),
            stream_group_handler=lambda name: (name, 200),
        )
    )

    routes = {rule.rule for rule in app.url_map.iter_rules()}
    assert "/tags" in routes
    assert "/tags/<path:tag_name>" in routes
    assert "/api/settings/timers" in routes
    assert "/api/sync_timers" in routes
    assert "/api/sync_timers/<timer_id>" in routes
    assert "/streams_meta" in routes
    assert "/groups" in routes
    assert "/groups/<name>" in routes
    assert "/notes" in routes
    assert "/stream/group/<name>" in routes
