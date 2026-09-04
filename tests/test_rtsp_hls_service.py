from pathlib import Path

from echomosaic_app.services.rtsp_hls import RtspHlsService


class FakeProcess:
    def __init__(self):
        self.returncode = None
        self.terminated = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = 0

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        self.returncode = -9


def test_rtsp_service_reuses_a_live_converter(tmp_path: Path) -> None:
    calls = []
    service = RtspHlsService(root=tmp_path, ffmpeg_path="/usr/bin/ffmpeg", popen=lambda *a, **kw: calls.append(a[0]) or FakeProcess())

    first = service.ensure("front door", "rtsp://user:secret@camera/live")
    second = service.ensure("front door", "rtsp://user:secret@camera/live")

    assert first == second
    assert first.startswith("/stream/rtsp/") and first.endswith("/index.m3u8")
    assert "front" not in first
    assert len(calls) == 1
    assert "rtsp://user:secret@camera/live" in calls[0]


def test_rtsp_service_rejects_other_protocols(tmp_path: Path) -> None:
    service = RtspHlsService(root=tmp_path, ffmpeg_path="ffmpeg")
    try:
        service.ensure("one", "https://example.com/video")
    except ValueError as exc:
        assert "RTSP" in str(exc)
    else:
        raise AssertionError("expected ValueError")
