from __future__ import annotations

from echomosaic_app.state.youtube_runtime import build_youtube_runtime


class _FakeCache:
    def __init__(self, maxsize: int) -> None:
        self.maxsize = maxsize


def test_build_youtube_runtime_creates_isolated_runtime_state() -> None:
    runtime = build_youtube_runtime(cache_factory=lambda maxsize: _FakeCache(maxsize))

    assert runtime.youtube_oembed_cache.maxsize == 256
    assert runtime.youtube_live_probe_cache.maxsize == 256
    assert runtime.youtube_sync_state == {}
    assert runtime.youtube_sync_subscribers == {}
    assert runtime.youtube_sync_leaders == {}
    assert runtime.youtube_in_flight == set()
