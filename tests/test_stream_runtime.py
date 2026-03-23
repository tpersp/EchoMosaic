from __future__ import annotations

from echomosaic_app.state.stream_runtime import build_stream_runtime


class _FakeCache:
    def __init__(self, maxsize: int) -> None:
        self.maxsize = maxsize


def test_build_stream_runtime_creates_runtime_containers() -> None:
    runtime = build_stream_runtime(cache_factory=lambda maxsize: _FakeCache(maxsize))

    assert runtime.stream_runtime_state == {}
    assert runtime.resized_image_locks.maxsize == 512
    assert runtime.video_duration_cache.maxsize == 128
