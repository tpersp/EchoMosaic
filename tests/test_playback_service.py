from __future__ import annotations

from echomosaic_app.services.playback import PlaybackService


class _PlaybackManager:
    def __init__(self) -> None:
        self.states = {}
        self.calls = []

    def get_state(self, stream_id):
        self.calls.append(("get_state", stream_id))
        return self.states.get(stream_id)

    def update_stream_config(self, stream_id, conf):
        self.calls.append(("update_stream_config", stream_id))

    def ensure_started(self, stream_id):
        self.calls.append(("ensure_started", stream_id))
        return self.states.get(stream_id)

    def emit_state(self, payload, room=None, event=None):
        self.calls.append(("emit_state", room, event))

    def skip_next(self, stream_id):
        self.calls.append(("skip_next", stream_id))

    def skip_previous(self, stream_id):
        self.calls.append(("skip_previous", stream_id))

    def pause(self, stream_id):
        self.calls.append(("pause", stream_id))

    def resume(self, stream_id):
        self.calls.append(("resume", stream_id))

    def toggle(self, stream_id):
        self.calls.append(("toggle", stream_id))

    def set_volume(self, stream_id, volume):
        self.calls.append(("set_volume", stream_id, volume))


def test_playback_service_returns_existing_state() -> None:
    manager = _PlaybackManager()
    manager.states["stream1"] = {"status": "playing"}
    service = PlaybackService(
        settings={"stream1": {"label": "One"}},
        playback_manager=manager,
        ensure_sync_defaults=lambda conf: conf.setdefault("_sync", {}),
        safe_emit=lambda *args, **kwargs: None,
    )

    payload = service.get_stream_playback_state_payload("stream1")

    assert payload["status"] == "playing"


def test_playback_service_handles_control_action() -> None:
    manager = _PlaybackManager()
    emitted = []
    service = PlaybackService(
        settings={},
        playback_manager=manager,
        ensure_sync_defaults=lambda conf: conf,
        safe_emit=lambda event, payload: emitted.append((event, payload)),
    )

    payload = service.handle_control_action("stream1", "set_volume", volume_value=0.75)

    assert payload["action"] == "set_volume"
    assert payload["volume"] == 0.75
    assert ("set_volume", "stream1", 0.75) in manager.calls
    assert emitted[0][0] == "video_control"


def test_playback_service_emits_initial_state() -> None:
    manager = _PlaybackManager()
    manager.states["stream1"] = {"status": "playing"}
    service = PlaybackService(
        settings={},
        playback_manager=manager,
        ensure_sync_defaults=lambda conf: conf,
        safe_emit=lambda *args, **kwargs: None,
    )

    payload = service.emit_initial_state("stream1", "sid-1", "stream_init")

    assert payload["status"] == "playing"
    assert ("emit_state", "sid-1", "stream_init") in manager.calls
