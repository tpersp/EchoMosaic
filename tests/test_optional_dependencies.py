import sys

from echomosaic_app.optional_dependencies import LazyYoutubeDL


def test_importing_lazy_youtube_wrapper_does_not_import_yt_dlp() -> None:
    sys.modules.pop("yt_dlp", None)
    assert LazyYoutubeDL is not None
    assert "yt_dlp" not in sys.modules
