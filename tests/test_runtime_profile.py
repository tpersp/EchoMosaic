from echomosaic_app.runtime_profile import LOW_MEMORY_LIMITS, apply_runtime_profile


def test_default_profile_preserves_configuration() -> None:
    original = {"LOW_MEMORY_MODE": False, "LIVE_HLS_MAX_WORKERS": 8, "MEDIA_PREVIEW_ENABLED": True}
    assert apply_runtime_profile(original) == original


def test_low_memory_profile_caps_expensive_settings() -> None:
    effective = apply_runtime_profile({
        "LOW_MEMORY_MODE": "true",
        "LIVE_HLS_MAX_WORKERS": 8,
        "MEDIA_CATALOG_CACHE_SIZE": 1000,
        "MEDIA_PREVIEW_ENABLED": True,
    })

    assert effective["LIVE_HLS_MAX_WORKERS"] == LOW_MEMORY_LIMITS["LIVE_HLS_MAX_WORKERS"]
    assert effective["MEDIA_CATALOG_CACHE_SIZE"] == LOW_MEMORY_LIMITS["MEDIA_CATALOG_CACHE_SIZE"]
    assert effective["MEDIA_PREVIEW_ENABLED"] is False


def test_low_memory_profile_does_not_increase_smaller_values() -> None:
    effective = apply_runtime_profile({"LOW_MEMORY_MODE": True, "MEDIA_PREVIEW_FRAMES": 1})
    assert effective["MEDIA_PREVIEW_FRAMES"] == 1
