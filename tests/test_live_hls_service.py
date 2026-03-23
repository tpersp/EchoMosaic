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
