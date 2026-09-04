from pathlib import Path

from echomosaic_app.services.rtsp_hls import RtspHlsService


class FakeProcess:
    def __init__(self):
        self.returncode = None
        self.terminated = False
        self.stderr = None

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
    assert calls[0][calls[0].index("-hls_time") + 1] == "1"
    assert calls[0][calls[0].index("-hls_list_size") + 1] == "3"

    token = first.split("/")[3]
    playlist = tmp_path / token / "index.m3u8"
    playlist.write_text("#EXTM3U\n")
    assert service.asset_path(token, "index.m3u8") == playlist
    assert service.asset_path("front door", "index.m3u8") is None
    assert service.asset_path(token, "../settings.json") is None


def test_rtsp_service_rejects_other_protocols(tmp_path: Path) -> None:
    service = RtspHlsService(root=tmp_path, ffmpeg_path="ffmpeg")
    try:
        service.ensure("one", "https://example.com/video")
    except ValueError as exc:
        assert "RTSP" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_rtsp_service_restarts_failed_active_converter(tmp_path: Path) -> None:
    now = [100.0]
    processes = []

    def spawn(*args, **kwargs):
        process = FakeProcess()
        processes.append(process)
        return process

    service = RtspHlsService(
        root=tmp_path,
        ffmpeg_path="ffmpeg",
        popen=spawn,
        clock=lambda: now[0],
        idle_timeout=60,
    )
    service.ensure("one", "rtsp://camera/live")
    processes[0].returncode = 1

    with service._lock:
        service._stop_idle_locked(now[0])

    assert len(processes) == 2
    assert service.diagnostics("one")["active"] is True
    service.stop_all()


def test_rtsp_service_redacts_credentials_from_launch_errors(tmp_path: Path) -> None:
    source = "rtsp://user:secret@camera/live"

    def fail(command, **kwargs):
        raise OSError(f"could not open {source}")

    service = RtspHlsService(root=tmp_path, ffmpeg_path="ffmpeg", popen=fail)
    service.ensure("one", source)
    diagnostics = service.diagnostics("one")

    assert diagnostics["active"] is False
    assert "user" not in diagnostics["error"]
    assert "secret" not in diagnostics["error"]
    assert "<redacted-rtsp-url>" in diagnostics["error"]
