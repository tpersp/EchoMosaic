from __future__ import annotations

from flask import Flask

from echomosaic_app.routes.media import create_media_blueprint


class _MediaManagerStub:
    class error_type(Exception):
        def __init__(self, message: str, code: str = "error") -> None:
            super().__init__(message)
            self.code = code


def test_media_blueprint_registers_expected_routes() -> None:
    app = Flask(__name__)
    app.register_blueprint(
        create_media_blueprint(
            media_manager=_MediaManagerStub(),
            media_error_type=_MediaManagerStub.error_type,
            media_thumb_width=320,
            media_allow_edit=True,
            media_allowed_exts=[".png"],
            media_upload_max_mb=256,
            media_preview_enabled=True,
            media_preview_frames=8,
            media_preview_width=320,
            media_preview_max_duration=300.0,
            available_media_roots=[],
            media_library_default="media",
            media_error_response=lambda exc: ("error", 400),
            require_media_edit=lambda: None,
            invalidate_media_cache=lambda path: None,
            parse_truthy=bool,
            normalize_library_key=lambda value, default="media": default,
            get_folder_inventory=lambda **kwargs: [],
            as_int=lambda value, default=0: default,
            virtual_leaf=lambda path: path,
            logger=type("L", (), {"info": lambda *args, **kwargs: None})(),
            app_response_class=app.response_class,
        )
    )

    endpoints = {rule.rule for rule in app.url_map.iter_rules()}
    assert "/api/media/list" in endpoints
    assert "/api/media/create_folder" in endpoints
    assert "/api/media/rename" in endpoints
    assert "/api/media/delete" in endpoints
    assert "/api/media/upload" in endpoints
    assert "/api/media/thumbnail" in endpoints
    assert "/api/media/preview_frame" in endpoints
    assert "/folders" in endpoints
    assert "/media/manage" in endpoints
