"""Media-library and stream-asset route blueprint."""

from __future__ import annotations

from typing import Callable

from flask import Blueprint


def create_assets_blueprint(
    *,
    get_images_handler: Callable[[], object],
    get_random_image_handler: Callable[[], object],
    get_media_entries_handler: Callable[[], object],
    get_random_media_handler: Callable[[], object],
    serve_image_handler: Callable[[str], object],
    serve_video_handler: Callable[[str], object],
    stream_thumbnail_metadata_handler: Callable[[str], object],
    stream_thumbnail_image_handler: Callable[[str], object],
    cached_stream_thumbnail_handler: Callable[[str], object],
) -> Blueprint:
    blueprint = Blueprint("asset_routes", __name__)

    @blueprint.route("/images", methods=["GET"])
    def get_images():
        return get_images_handler()

    @blueprint.route("/images/random", methods=["GET"])
    def get_random_image():
        return get_random_image_handler()

    @blueprint.route("/media", methods=["GET"])
    def get_media_entries():
        return get_media_entries_handler()

    @blueprint.route("/media/random", methods=["GET"])
    def get_random_media():
        return get_random_media_handler()

    @blueprint.route("/stream/image/<path:image_path>")
    def serve_image(image_path: str):
        return serve_image_handler(image_path)

    @blueprint.route("/stream/video/<path:video_path>")
    def serve_video(video_path: str):
        return serve_video_handler(video_path)

    @blueprint.route("/stream/thumbnail/<stream_id>", methods=["GET"])
    def stream_thumbnail_metadata(stream_id: str):
        return stream_thumbnail_metadata_handler(stream_id)

    @blueprint.route("/stream/thumbnail/<stream_id>/image", methods=["GET"])
    def stream_thumbnail_image(stream_id: str):
        return stream_thumbnail_image_handler(stream_id)

    @blueprint.route("/thumbnails/<stream_id>.jpg", methods=["GET"])
    def cached_stream_thumbnail(stream_id: str):
        return cached_stream_thumbnail_handler(stream_id)

    return blueprint
