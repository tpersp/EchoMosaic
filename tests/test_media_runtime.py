from __future__ import annotations

from echomosaic_app.state.media_runtime import build_media_cache_runtime


def test_build_media_cache_runtime_creates_owned_caches() -> None:
    runtime = build_media_cache_runtime(cache_factory=lambda maxsize: {"maxsize": maxsize})

    assert runtime.image_cache == {"maxsize": 64}
    assert runtime.bad_media_log_cache == {"maxsize": 1024}
