from __future__ import annotations

from echomosaic_app.services.links_service import GlobalLinksService


def _service(settings=None):
    state = settings if isinstance(settings, dict) else {}
    saved = {"calls": 0}

    def parse_youtube(url: str):
        if "playlist?list=" in url:
            return {"playlist_id": "pl1", "video_id": None, "is_live": False}
        if "watch?v=" in url:
            return {"playlist_id": None, "video_id": "vid1", "is_live": False}
        if "/live/" in url:
            return {"playlist_id": None, "video_id": "live1", "is_live": True}
        return None

    service = GlobalLinksService(
        settings=state,
        save_settings_debounced=lambda: saved.__setitem__("calls", saved["calls"] + 1),
        parse_youtube_url_details=parse_youtube,
        youtube_metadata_lookup=lambda url, details: {
            "content_type": "live" if "/live/" in url else ("playlist" if "playlist?list=" in url else "video"),
            "thumbnail_url": "https://i.ytimg.com/vi/live1/hqdefault.jpg" if "/live/" in url else "https://i.ytimg.com/vi/vid1/hqdefault.jpg",
            "video_id": details.get("video_id"),
        } if "youtube" in url else None,
    )
    return service, state, saved


def test_links_service_sanitizes_required_fields_and_detects_type() -> None:
    service, _, _ = _service()

    link = service.sanitize_link(
        {
            "label": " Bob Ross Playlist ",
            "url": "https://www.youtube.com/playlist?list=pl1",
            "category": " Cartoons ",
        }
    )

    assert link is not None
    assert link["label"] == "Bob Ross Playlist"
    assert link["category"] == "Cartoons"
    assert link["provider"] == "youtube"
    assert link["content_type"] == "playlist"


def test_links_service_marks_youtube_live_urls_as_live_and_keeps_thumbnail() -> None:
    service, _, _ = _service()

    link = service.sanitize_link(
        {
            "label": "Live Channel",
            "url": "https://www.youtube.com/live/live1",
            "category": "Live",
        }
    )

    assert link is not None
    assert link["provider"] == "youtube"
    assert link["content_type"] == "live"
    assert "thumbnail_url" in link


def test_links_service_create_update_delete_roundtrip() -> None:
    service, state, saved = _service()

    created = service.create_link(
        {
            "label": "Lofi Stream",
            "url": "https://example.com/live.m3u8",
            "category": "Music",
        }
    )
    link_id = created["link"]["id"]
    assert state["_links"][0]["provider"] == "hls"
    assert saved["calls"] == 1

    updated = service.update_link(
        link_id,
        {
            "label": "Lofi Stream HD",
            "url": "https://www.youtube.com/watch?v=vid1",
            "category": "Music",
        },
    )
    assert updated["link"]["label"] == "Lofi Stream HD"
    assert updated["link"]["provider"] == "youtube"
    assert saved["calls"] == 2

    deleted = service.delete_link(link_id)
    assert deleted["links"] == []
    assert saved["calls"] == 3
