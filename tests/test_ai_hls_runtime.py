from __future__ import annotations

from pathlib import Path

from echomosaic_app.state.ai_runtime import build_ai_runtime
from echomosaic_app.state.hls_runtime import build_hls_runtime


def test_build_ai_runtime_applies_config_and_creates_runtime_state(tmp_path: Path) -> None:
    root = tmp_path / "ai-media"
    root.mkdir()

    runtime = build_ai_runtime(
        config={
            "AI_DEFAULT_MODEL": "custom-model",
            "AI_DEFAULT_WIDTH": 640,
            "AI_TEMP_SUBDIR": "_scratch",
            "AI_DEFAULT_PERSIST": False,
        },
        default_model="stable_diffusion",
        default_sampler="k_euler",
        default_width=512,
        default_height=512,
        default_steps=30,
        default_cfg=7.5,
        default_samples=1,
        output_subdir="ai_generated",
        temp_subdir="_ai_temp",
        default_persist=True,
        poll_interval=5.0,
        timeout=0.0,
        primary_ai_root=root,
        ensure_dir=lambda path: Path(path).mkdir(parents=True, exist_ok=True) or Path(path),
        logger=type("L", (), {"warning": lambda *args, **kwargs: None})(),
    )

    assert runtime.ai_default_model == "custom-model"
    assert runtime.ai_default_width == 640
    assert runtime.ai_temp_subdir == "_scratch"
    assert runtime.ai_default_persist is False
    assert runtime.ai_output_root == root
    assert runtime.ai_temp_root == root / "_scratch"
    assert isinstance(runtime.ai_jobs, dict)


def test_build_hls_runtime_applies_configured_limits() -> None:
    runtime = build_hls_runtime(
        config={
            "LIVE_HLS_ASYNC": True,
            "LIVE_HLS_TTL_SECS": 120,
            "LIVE_HLS_MAX_WORKERS": 4,
            "LIVE_HLS_ERROR_RETRY_SECS": 999,
        },
        cache_factory=lambda maxsize: {"maxsize": maxsize},
        live_hls_async=True,
        hls_ttl_secs=3600,
        max_hls_workers=3,
        hls_error_retry_secs=30,
    )

    try:
        assert runtime.live_hls_async is True
        assert runtime.hls_ttl_secs == 120
        assert runtime.max_hls_workers == 4
        assert runtime.hls_error_retry_secs == 120
        assert runtime.hls_cache == {"maxsize": 256}
        assert runtime.hls_jobs == {"maxsize": 128}
    finally:
        if runtime.hls_executor is not None:
            runtime.hls_executor.shutdown(wait=False, cancel_futures=True)
