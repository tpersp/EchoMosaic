from __future__ import annotations

from echomosaic_app.services.live_hls import HLSCacheEntry, LiveHLSService


class _DummyLock:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _DummyContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _ImmediateExecutor:
    def __init__(self) -> None:
        self.calls = []

    def submit(self, fn, *args):
        self.calls.append((fn, args))
        fn(*args)

        class _DoneFuture:
            def done(self) -> bool:
                return True

            def cancel(self) -> bool:
                return False

        return _DoneFuture()


class _CapturingYoutubeDL:
    options = None

    def __init__(self, opts):
        type(self).options = opts

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def extract_info(self, url, download=False):
        return {"url": "https://cdn/test.m3u8"}


def _build_service(**overrides):
    emitted = []
    logger_messages = []
    service = LiveHLSService(
        live_hls_async=overrides.get("live_hls_async", True),
        hls_ttl_secs=overrides.get("hls_ttl_secs", 60.0),
        hls_error_retry_secs=overrides.get("hls_error_retry_secs", 30.0),
        hls_metrics=overrides.get("hls_metrics", {}),
        hls_lock=overrides.get("hls_lock", _DummyLock()),
        hls_log_prefix="live_hls",
        hls_executor=overrides.get("hls_executor"),
        hls_cache=overrides.get("hls_cache", {}),
        hls_jobs=overrides.get("hls_jobs", {}),
        youtube_dl_cls=overrides.get("youtube_dl_cls"),
        logger=type("Logger", (), {"info": lambda *args, **kwargs: logger_messages.append((args, kwargs))})(),
        app_context_factory=lambda: _DummyContext(),
        safe_emit=lambda *args, **kwargs: emitted.append((args, kwargs)),
        youtube_user_agent=overrides.get("youtube_user_agent"),
        youtube_cookie_file=overrides.get("youtube_cookie_file"),
        youtube_js_runtime=overrides.get("youtube_js_runtime"),
        youtube_remote_components=overrides.get("youtube_remote_components"),
    )
    return service, emitted, logger_messages


def test_live_hls_service_resolves_cached_url_without_reprobe() -> None:
    cache = {"live:stream1:https://example.com/live": HLSCacheEntry(url="https://cdn/test.m3u8", extracted_at=99999999999.0)}
    service, _, _ = _build_service(
        hls_cache=cache,
        hls_executor=_ImmediateExecutor(),
        hls_ttl_secs=999999999.0,
    )

    payload = service.resolve_hls_url("stream1", "https://example.com/live")

    assert payload == "https://cdn/test.m3u8"


def test_live_hls_service_invalidates_matching_cache_and_reschedules() -> None:
    executor = _ImmediateExecutor()
    cache = {"live:stream1:https://example.com/live": HLSCacheEntry(url="https://cdn/test.m3u8", extracted_at=1.0)}
    jobs = {}
    service, _, _ = _build_service(
        hls_cache=cache,
        hls_jobs=jobs,
        hls_executor=executor,
        youtube_dl_cls=type(
            "FakeYoutubeDL",
            (),
            {
                "__init__": lambda self, opts: None,
                "__enter__": lambda self: self,
                "__exit__": lambda self, exc_type, exc, tb: False,
                "extract_info": lambda self, url, download=False: {"url": "https://cdn/refreshed.m3u8"},
            },
        ),
    )

    payload = service.invalidate_stream("stream1", "https://example.com/live")

    assert payload["status"] == "ok"
    assert payload["removed"] == 1
    assert payload["rescheduled"] is True
    assert "live:stream1:https://example.com/live" in cache


def test_live_hls_service_passes_browser_identity_to_youtube_dl() -> None:
    service, _, _ = _build_service(
        youtube_dl_cls=_CapturingYoutubeDL,
        youtube_user_agent="Custom Browser UA",
        youtube_cookie_file="~/youtube-cookies.txt",
        youtube_js_runtime="node:/usr/bin/node",
        youtube_remote_components="ejs:github",
    )

    url = service.detect_hls_stream_url("https://www.youtube.com/watch?v=abc123")

    assert url == "https://cdn/test.m3u8"
    assert _CapturingYoutubeDL.options["user_agent"] == "Custom Browser UA"
    assert _CapturingYoutubeDL.options["cookiefile"].endswith("/youtube-cookies.txt")
    assert _CapturingYoutubeDL.options["js_runtimes"] == {"node": {"path": "/usr/bin/node"}}
    assert _CapturingYoutubeDL.options["remote_components"] == {"ejs:github"}
