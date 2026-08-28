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
    assert b'id="low-memory-mode-toggle"' in settings_page.data
    assert b'Save Runtime Settings' in settings_page.data
    assert b'id="runtime-profile-badge"' in settings_page.data
    assert b'Reclaimable file cache' in settings_page.data
    assert b'id="service-restart-button"' in settings_page.data
    assert debug_page.status_code == 200


def test_runtime_settings_expose_low_memory_mode() -> None:
    response = app.test_client().get("/api/settings/media")

    assert response.status_code == 200
    payload = response.get_json()
    assert isinstance(payload.get("low_memory_mode"), bool)
    assert "media_upload_max_mb" in payload
    assert isinstance(payload.get("effective_runtime"), dict)
    assert "live_hls_max_workers" in payload["effective_runtime"]
