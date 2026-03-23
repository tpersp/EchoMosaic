"""Playback coordination service."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional


class PlaybackService:
    def __init__(
        self,
        *,
        settings,
        playback_manager,
        ensure_sync_defaults,
        safe_emit,
    ) -> None:
        self.settings = settings
        self.playback_manager = playback_manager
        self.ensure_sync_defaults = ensure_sync_defaults
        self.safe_emit = safe_emit

    def get_stream_playback_state_payload(self, stream_id: str) -> Dict[str, Any]:
        if self.playback_manager is None:
            raise RuntimeError("Playback manager unavailable")
        conf = self.settings.get(stream_id)
        if conf is None or not isinstance(conf, dict):
            raise KeyError(stream_id)
        self.ensure_sync_defaults(conf)
        state = self.playback_manager.get_state(stream_id)
        if state is None:
            self.playback_manager.update_stream_config(stream_id, conf)
            state = self.playback_manager.ensure_started(stream_id)
        if state is None:
            raise LookupError("No playback state")
        return state

    def emit_initial_state(self, stream_id: str, sid: str, event: str) -> Optional[Dict[str, Any]]:
        if self.playback_manager is None:
            return None
        state = self.playback_manager.ensure_started(stream_id)
        if not state:
            state = self.playback_manager.get_state(stream_id)
        if state:
            self.playback_manager.emit_state(state, room=sid, event=event)
        return state

    def handle_control_action(self, stream_id: str, action: str, volume_value: Optional[float] = None) -> Dict[str, Any]:
        if self.playback_manager is not None:
            if action == "skip_next":
                self.playback_manager.skip_next(stream_id)
            elif action == "skip_prev":
                self.playback_manager.skip_previous(stream_id)
            elif action == "pause":
                self.playback_manager.pause(stream_id)
            elif action == "play":
                state = self.playback_manager.get_state(stream_id)
                if state and state.get("status") == "paused":
                    self.playback_manager.resume(stream_id)
                else:
                    self.playback_manager.ensure_started(stream_id)
            elif action == "toggle":
                self.playback_manager.toggle(stream_id)
            elif action == "set_volume" and volume_value is not None:
                self.playback_manager.set_volume(stream_id, volume_value)
        message: Dict[str, Any] = {
            "stream_id": stream_id,
            "action": action,
            "timestamp": time.time(),
        }
        if volume_value is not None:
            message["volume"] = max(0.0, min(1.0, volume_value))
        self.safe_emit("video_control", message)
        return message
