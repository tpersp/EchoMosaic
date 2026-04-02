from __future__ import annotations

from echomosaic_app.services.youtube_embed import YouTubeEmbedService


class _DummyLock:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _ImmediateEventlet:
    @staticmethod
    def spawn(fn, *args, **kwargs):
        fn(*args, **kwargs)


class _Response:
    def __init__(self, payload):
        self._payload = payload
        self.encoding = "utf-8"
        self.url = "https://example.com/final"
        self.status_code = 200
        self.headers = {}

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload

    def iter_content(self, chunk_size=4096, decode_unicode=False):
        yield b""

    def close(self):
        return None


class _ChunkedResponse(_Response):
    def __init__(self, chunks):
        super().__init__({})
        self._chunks = list(chunks)

    def iter_content(self, chunk_size=4096, decode_unicode=False):
        for chunk in self._chunks:
            yield chunk


class _FakeYoutubeDL:
    payload = None

    def __init__(self, options):
        self.options = options

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def extract_info(self, url, download=False):
        return self.payload


def _build_service(**overrides):
    emitted = []
    runtime_state = {}
    requests_module = overrides.get(
        "requests_module",
        type("Requests", (), {"get": staticmethod(lambda *args, **kwargs: _Response({"title": "Live Test", "provider_name": "YouTube"}))}),
    )
    service = YouTubeEmbedService(
        requests_module=requests_module,
        youtube_dl_cls=overrides.get("youtube_dl_cls"),
        eventlet_module=overrides.get("eventlet_module", _ImmediateEventlet()),
        logger=type("Logger", (), {"debug": lambda *args, **kwargs: None})(),
        youtube_domains={"youtube.com", "www.youtube.com", "youtu.be"},
        youtube_oembed_endpoint="https://www.youtube.com/oembed",
        youtube_oembed_cache_ttl=1200,
        youtube_live_probe_cache_ttl=900,
        youtube_playlist_cache_ttl=900,
        youtube_live_probe_max_bytes=30000,
        youtube_live_html_markers=('"islive":true',),
        youtube_oembed_cache=overrides.get("youtube_oembed_cache", {}),
        youtube_oembed_cache_lock=_DummyLock(),
        youtube_live_probe_cache=overrides.get("youtube_live_probe_cache", {}),
        youtube_live_probe_cache_lock=_DummyLock(),
        youtube_playlist_cache=overrides.get("youtube_playlist_cache", {}),
        youtube_playlist_cache_lock=_DummyLock(),
        youtube_sync_state_lock=_DummyLock(),
        youtube_sync_state=overrides.get("youtube_sync_state", {}),
        youtube_sync_subscribers=overrides.get("youtube_sync_subscribers", {}),
        youtube_sync_leaders=overrides.get("youtube_sync_leaders", {}),
        youtube_in_flight=overrides.get("youtube_in_flight", set()),
        youtube_in_flight_lock=_DummyLock(),
        stream_runtime_lock=_DummyLock(),
        stream_runtime_state=runtime_state,
        safe_emit=lambda *args, **kwargs: emitted.append((args, kwargs)),
        youtube_sync_role_event="youtube_sync_role",
        youtube_sync_max_age_seconds=20.0,
        media_mode_livestream="livestream",
    )
    return service, emitted, runtime_state


def test_youtube_embed_service_parses_watch_urls() -> None:
    service, _, _ = _build_service()

    details = service.parse_youtube_url_details("https://www.youtube.com/watch?v=abc123&list=pl1&index=4&start=20")

    assert details is not None
    assert details["video_id"] == "abc123"
    assert details["playlist_id"] == "pl1"
    assert details["start_index"] == 4
    assert details["start_seconds"] == 20


def test_youtube_embed_service_refreshes_embed_metadata_and_runtime() -> None:
    service, _, runtime_state = _build_service()
    conf = {"media_mode": "livestream", "stream_url": "https://www.youtube.com/watch?v=abc123"}

    metadata = service.refresh_embed_metadata("stream1", conf, force=True)

    assert metadata is not None
    assert metadata["video_id"] == "abc123"
    assert conf["embed_metadata"]["video_id"] == "abc123"
    assert runtime_state["stream1"]["embed_metadata"]["video_id"] == "abc123"


def test_youtube_embed_service_promotes_next_leader_on_disconnect() -> None:
    service, emitted, _ = _build_service(
        youtube_sync_subscribers={"stream1": {"sid-a", "sid-b"}},
        youtube_sync_leaders={"stream1": "sid-a"},
    )

    service.remove_youtube_sync_subscriber("sid-a", "stream1")

    assert service.youtube_sync_role_for_sid("stream1", "sid-b") is True
    assert emitted[-1][0][1]["is_leader"] is True


def test_youtube_embed_service_normalizes_live_content_type() -> None:
    service, _, _ = _build_service()

    metadata = service.sanitize_embed_metadata({"content_type": "video", "is_live": True, "video_id": "abc123"})

    assert metadata is not None
    assert metadata["content_type"] == "live"
    assert metadata["is_live"] is True


def test_youtube_embed_service_live_probe_reads_late_markers() -> None:
    marker = b'"livebroadcastdetails":{"islivenow":true}'
    requests_module = type(
        "Requests",
        (),
        {"get": staticmethod(lambda *args, **kwargs: _ChunkedResponse([b"x" * 40_000, marker]))},
    )
    service, _, _ = _build_service(
        requests_module=requests_module,
        youtube_live_probe_cache={},
    )
    service.youtube_live_probe_max_bytes = 80_000

    looks_live = service.youtube_page_looks_live(
        {
            "video_id": "abc123",
            "canonical_url": "https://www.youtube.com/watch?v=abc123",
            "original_url": "https://www.youtube.com/watch?v=abc123",
            "embed_base": "https://www.youtube-nocookie.com/embed/abc123",
        }
    )

    assert looks_live is True


def test_youtube_embed_service_fetches_playlist_entries_and_current_item() -> None:
    _FakeYoutubeDL.payload = {
        "id": "pl1",
        "title": "Cartoon Queue",
        "entries": [
            {"id": "vid-a", "title": "Episode A", "playlist_index": 1},
            {"id": "vid-b", "title": "Episode B", "playlist_index": 2},
            {"id": "vid-c", "title": "Episode C", "playlist_index": 3},
        ],
    }
    service, _, _ = _build_service(youtube_dl_cls=_FakeYoutubeDL, youtube_playlist_cache={})
    details = service.parse_youtube_url_details("https://www.youtube.com/watch?v=vid-b&list=pl1&index=2")

    playlist = service.fetch_youtube_playlist("https://www.youtube.com/watch?v=vid-b&list=pl1&index=2", details, force=True)
    current = service.resolve_youtube_playlist_current_item(playlist, details, {"video_id": "vid-b", "start_index": 2})

    assert playlist is not None
    assert playlist["title"] == "Cartoon Queue"
    assert playlist["entry_count"] == 3
    assert playlist["entries"][1]["url"].endswith("v=vid-b&list=pl1&index=2")
    assert current is not None
    assert current["video_id"] == "vid-b"
    assert current["index"] == 2
