from __future__ import annotations

from app import app


def test_feature_template_pages_render() -> None:
    client = app.test_client()

    dashboard = client.get("/")
    media_page = client.get("/media/manage")
    settings_page = client.get("/settings")
    debug_page = client.get("/debug")

    assert dashboard.status_code == 200
    assert media_page.status_code == 200
    assert settings_page.status_code == 200
    assert debug_page.status_code == 200
