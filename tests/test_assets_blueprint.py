from __future__ import annotations

from flask import Flask

from echomosaic_app.routes.assets import create_assets_blueprint


def test_assets_blueprint_registers_expected_routes() -> None:
    app = Flask(__name__)
    app.register_blueprint(
        create_assets_blueprint(
            get_images_handler=lambda: ("ok", 200),
            get_random_image_handler=lambda: ("ok", 200),
            get_media_entries_handler=lambda: ("ok", 200),
            get_random_media_handler=lambda: ("ok", 200),
            serve_image_handler=lambda image_path: (image_path, 200),
            serve_video_handler=lambda video_path: (video_path, 200),
            stream_thumbnail_metadata_handler=lambda stream_id: (stream_id, 200),
            stream_thumbnail_image_handler=lambda stream_id: (stream_id, 200),
            cached_stream_thumbnail_handler=lambda stream_id: (stream_id, 200),
        )
    )

    routes = {rule.rule for rule in app.url_map.iter_rules()}
    assert "/images" in routes
    assert "/images/random" in routes
    assert "/media" in routes
    assert "/media/random" in routes
    assert "/stream/image/<path:image_path>" in routes
    assert "/stream/video/<path:video_path>" in routes
    assert "/stream/thumbnail/<stream_id>" in routes
    assert "/stream/thumbnail/<stream_id>/image" in routes
    assert "/thumbnails/<stream_id>.jpg" in routes
