"""Playback engine extracted from the legacy app shell."""

from __future__ import annotations

import math
import random
import threading
import time
from typing import Any, Dict, List, Optional, Tuple


class StreamPlaybackState:
    """Track shared playback data for a single stream."""

    def __init__(
        self,
        stream_id: str,
        *,
        infer_media_mode,
        media_mode_choices,
        media_mode_image,
        media_mode_video,
        media_mode_ai,
        ai_random_mode,
        video_playback_modes,
        sync_config_key,
        sync_supported_media_modes,
        stream_playback_history_limit: int,
        get_runtime_thumbnail_payload,
    ) -> None:
        self.stream_id = stream_id
        self._infer_media_mode = infer_media_mode
        self._media_mode_choices = media_mode_choices
        self._media_mode_image = media_mode_image
        self._media_mode_video = media_mode_video
        self._media_mode_ai = media_mode_ai
        self._ai_random_mode = ai_random_mode
        self._video_playback_modes = video_playback_modes
        self._sync_config_key = sync_config_key
        self._sync_supported_media_modes = sync_supported_media_modes
        self._stream_playback_history_limit = stream_playback_history_limit
        self._get_runtime_thumbnail_payload = get_runtime_thumbnail_payload

        self.mode: str = "random"
        self.media_mode: str = media_mode_image
        self.folder: str = "all"
        self.shuffle: bool = True
        self.duration_setting: float = 5.0
        self.video_playback_mode: str = "duration"
        self.video_volume: float = 1.0
        self.current_media: Optional[Dict[str, Any]] = None
        self.started_at: Optional[float] = None
        self.position: float = 0.0
        self.duration: Optional[float] = None
        self.is_paused: bool = False
        self.next_auto_event: Optional[float] = None
        self.sequence_index: int = 0
        self.history: List[Dict[str, Any]] = []
        self.history_index: int = -1
        self.last_reason: str = "init"
        self.error: Optional[str] = None
        self.updated_at: float = time.time()
        self._source_signature: Optional[Tuple[Any, ...]] = None
        self._duration_signature: Optional[Tuple[Any, ...]] = None
        self.last_sync_emit: float = 0.0
        self.sync_timer_id: Optional[str] = None
        self.sync_offset: float = 0.0
        self._thumbnail_cache: Optional[Dict[str, Any]] = None

    def apply_config(self, conf: Dict[str, Any]) -> Dict[str, bool]:
        previous_should_run = self.should_run()
        previous_source_signature = self._source_signature
        previous_duration_signature = self._duration_signature
        previous_sync_signature = (self.sync_timer_id, self.sync_offset)

        mode_raw = conf.get("mode")
        new_mode = mode_raw.strip().lower() if isinstance(mode_raw, str) else "random"

        media_mode_raw = conf.get("media_mode")
        new_media_mode = media_mode_raw.strip().lower() if isinstance(media_mode_raw, str) else ""
        if new_media_mode not in self._media_mode_choices:
            inferred = self._infer_media_mode(conf)
            if inferred in self._media_mode_choices:
                new_media_mode = inferred
            else:
                new_media_mode = self._media_mode_image

        folder_raw = conf.get("folder")
        new_folder = folder_raw.strip() if isinstance(folder_raw, str) and folder_raw.strip() else "all"

        shuffle_value = conf.get("shuffle")
        new_shuffle = False if shuffle_value is False else True

        duration_raw = conf.get("duration")
        try:
            new_duration = float(duration_raw)
        except (TypeError, ValueError):
            new_duration = self.duration_setting
        if not (new_duration and new_duration > 0):
            new_duration = 5.0

        playback_raw = conf.get("video_playback_mode")
        new_playback_mode = playback_raw.strip().lower() if isinstance(playback_raw, str) else "duration"
        if new_playback_mode not in self._video_playback_modes:
            new_playback_mode = "duration"

        volume_raw = conf.get("video_volume")
        try:
            new_volume = float(volume_raw)
        except (TypeError, ValueError):
            new_volume = self.video_volume
        new_volume = max(0.0, min(1.0, new_volume))

        sync_conf = conf.get(self._sync_config_key) if isinstance(conf, dict) else None
        new_sync_timer_id: Optional[str] = None
        new_sync_offset = 0.0
        if isinstance(sync_conf, dict):
            timer_raw = sync_conf.get("timer_id")
            if isinstance(timer_raw, str):
                candidate = timer_raw.strip()
                new_sync_timer_id = candidate or None
            offset_raw = sync_conf.get("offset")
            try:
                offset_value = float(offset_raw)
            except (TypeError, ValueError):
                offset_value = 0.0
            if math.isnan(offset_value) or math.isinf(offset_value):
                offset_value = 0.0
            new_sync_offset = max(0.0, offset_value)

        self.mode = new_mode
        self.media_mode = new_media_mode
        self.folder = new_folder
        self.shuffle = new_shuffle
        self.duration_setting = new_duration
        self.video_playback_mode = new_playback_mode
        self.video_volume = new_volume
        self.sync_timer_id = new_sync_timer_id
        self.sync_offset = new_sync_offset

        new_source_signature = (self.mode, self.media_mode, self.folder, self.shuffle)
        new_duration_signature = (self.duration_setting, self.video_playback_mode)

        self._source_signature = new_source_signature
        self._duration_signature = new_duration_signature

        enabled_changed = previous_should_run != self.should_run()
        sources_changed = new_source_signature != previous_source_signature
        duration_changed = new_duration_signature != previous_duration_signature
        sync_changed = (new_sync_timer_id, new_sync_offset) != previous_sync_signature
        return {
            "enabled_changed": enabled_changed,
            "sources_changed": sources_changed,
            "duration_changed": duration_changed,
            "sync_changed": sync_changed,
        }

    def should_run(self) -> bool:
        if self.media_mode == self._media_mode_ai:
            return self.mode == self._ai_random_mode
        return self.mode == "random" and self.media_mode in (self._media_mode_image, self._media_mode_video)

    def sync_active(self) -> bool:
        return bool(self.sync_timer_id) and self.mode == "random" and self.media_mode in self._sync_supported_media_modes

    def reset_state(self) -> None:
        self.current_media = None
        self.started_at = None
        self.position = 0.0
        self.duration = None
        self.is_paused = False
        self.next_auto_event = None
        self.sequence_index = 0
        self.history = []
        self.history_index = -1
        self.error = None
        self.last_reason = "reset"
        self.updated_at = time.time()
        self.last_sync_emit = 0.0

    def set_error(self, code: str) -> None:
        self.current_media = None
        self.started_at = None
        self.position = 0.0
        self.duration = None
        self.is_paused = False
        self.next_auto_event = None
        self.error = code
        self.last_reason = code
        self.updated_at = time.time()
        self.last_sync_emit = 0.0

    def get_position(self, now: Optional[float] = None) -> float:
        if now is None:
            now = time.time()
        if self.current_media is None:
            return 0.0
        if self.is_paused or self.started_at is None:
            return max(0.0, self.position)
        return max(0.0, self.position + (now - self.started_at))

    def pause(self) -> bool:
        if self.is_paused or not self.current_media:
            return False
        now = time.time()
        self.position = self.get_position(now)
        self.is_paused = True
        self.started_at = now
        self.next_auto_event = None
        self.updated_at = now
        return True

    def resume(self) -> bool:
        if not self.current_media or not self.is_paused:
            return False
        now = time.time()
        self.is_paused = False
        self.started_at = now
        if self.duration is not None:
            remaining = max(0.0, self.duration - self.position)
            self.next_auto_event = now + remaining if remaining > 0 else now
        self.updated_at = now
        return True

    def _append_history_entry(self, media: Dict[str, Any], playback_mode: Optional[str]) -> None:
        entry = {"media": dict(media), "duration": self.duration, "playback_mode": playback_mode}
        self.history.append(entry)
        if len(self.history) > self._stream_playback_history_limit:
            overflow = len(self.history) - self._stream_playback_history_limit
            self.history = self.history[overflow:]
        self.history_index = len(self.history) - 1

    def set_media(
        self,
        media: Dict[str, Any],
        *,
        duration: Optional[float],
        source: str,
        playback_mode: Optional[str],
        history_index: Optional[int] = None,
        add_to_history: bool = True,
        now: Optional[float] = None,
    ) -> None:
        if now is None:
            now = time.time()
        actual_duration = duration if duration is None or duration > 0 else None
        self.current_media = dict(media)
        self.duration = actual_duration
        self.started_at = now
        self.position = 0.0
        self.is_paused = False
        self.error = None
        self.last_reason = source
        self.updated_at = now
        self.next_auto_event = now + actual_duration if actual_duration is not None else None

        if add_to_history:
            if self.history_index < len(self.history) - 1:
                self.history = self.history[: self.history_index + 1]
            self._append_history_entry(media, playback_mode)
            self._thumbnail_cache = None
        elif history_index is not None:
            self.history_index = history_index

    def get_history_entry(self, index: int) -> Optional[Dict[str, Any]]:
        if 0 <= index < len(self.history):
            return self.history[index]
        return None

    def status(self) -> str:
        if self.error:
            return "error"
        if not self.current_media:
            return "idle"
        if self.is_paused:
            return "paused"
        return "playing"

    def to_payload(self) -> Dict[str, Any]:
        now = time.time()
        position = self.get_position(now)
        if self.duration is not None:
            position = min(self.duration, position)
        payload = {
            "stream_id": self.stream_id,
            "mode": self.mode,
            "media_mode": self.media_mode,
            "status": self.status(),
            "media": dict(self.current_media) if self.current_media else None,
            "duration": self.duration,
            "position": position,
            "started_at": self.started_at,
            "is_paused": self.is_paused,
            "next_update_at": self.next_auto_event,
            "history_index": self.history_index,
            "history_length": len(self.history),
            "video_playback_mode": self.video_playback_mode if self.media_mode == self._media_mode_video else None,
            "video_volume": self.video_volume,
            "error": self.error,
            "source": self.last_reason,
            "server_time": now,
            "thumbnail": self._thumbnail_cache or self._get_runtime_thumbnail_payload(self.stream_id),
        }
        if self._thumbnail_cache is None:
            self._thumbnail_cache = payload.get("thumbnail")
        return payload

    def to_sync_payload(self, now: Optional[float] = None) -> Dict[str, Any]:
        snapshot = now if now is not None else time.time()
        position = self.get_position(snapshot)
        if self.duration is not None:
            position = min(self.duration, position)
        return {
            "stream_id": self.stream_id,
            "media": dict(self.current_media) if self.current_media else None,
            "duration": self.duration,
            "position": position,
            "started_at": self.started_at,
            "is_paused": self.is_paused,
            "server_time": snapshot,
        }


class StreamPlaybackManager:
    """Coordinate synchronized playback across connected viewers."""

    def __init__(
        self,
        *,
        safe_emit,
        list_media,
        library_for_media_mode,
        update_stream_runtime_state,
        get_runtime_thumbnail_payload,
        get_sync_timer_config,
        compute_next_sync_tick,
        coerce_float,
        infer_media_mode,
        resolve_media_path,
        video_duration_cache,
        cv2_module,
        media_mode_choices,
        media_mode_image,
        media_mode_video,
        media_mode_ai,
        ai_random_mode,
        video_playback_modes,
        sync_config_key,
        sync_supported_media_modes,
        stream_playback_history_limit: int,
        stream_update_event: str,
        sync_time_event: str,
        stream_sync_interval_seconds: float,
        sync_switch_lead_seconds: float,
    ) -> None:
        self.safe_emit = safe_emit
        self.list_media = list_media
        self.library_for_media_mode = library_for_media_mode
        self.update_stream_runtime_state = update_stream_runtime_state
        self.get_runtime_thumbnail_payload = get_runtime_thumbnail_payload
        self.get_sync_timer_config = get_sync_timer_config
        self.compute_next_sync_tick = compute_next_sync_tick
        self.coerce_float = coerce_float
        self.infer_media_mode = infer_media_mode
        self.resolve_media_path = resolve_media_path
        self.video_duration_cache = video_duration_cache
        self.cv2_module = cv2_module
        self.media_mode_choices = media_mode_choices
        self.media_mode_image = media_mode_image
        self.media_mode_video = media_mode_video
        self.media_mode_ai = media_mode_ai
        self.ai_random_mode = ai_random_mode
        self.video_playback_modes = video_playback_modes
        self.sync_config_key = sync_config_key
        self.sync_supported_media_modes = sync_supported_media_modes
        self.stream_playback_history_limit = stream_playback_history_limit
        self.stream_update_event = stream_update_event
        self.sync_time_event = sync_time_event
        self.stream_sync_interval_seconds = stream_sync_interval_seconds
        self.sync_switch_lead_seconds = sync_switch_lead_seconds

        self._lock = threading.Lock()
        self._states: Dict[str, StreamPlaybackState] = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="StreamPlaybackManager", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def emit_state(self, payload: Dict[str, Any], *, room: Optional[str] = None, event: Optional[str] = None) -> None:
        self._emit_state(payload, room=room, event=event or self.stream_update_event)

    def bootstrap(self, stream_settings: Dict[str, Any]) -> None:
        for stream_id, conf in stream_settings.items():
            if stream_id.startswith("_"):
                continue
            if isinstance(conf, dict):
                self.update_stream_config(stream_id, conf)

    def _new_state(self, stream_id: str) -> StreamPlaybackState:
        return StreamPlaybackState(
            stream_id,
            infer_media_mode=self.infer_media_mode,
            media_mode_choices=self.media_mode_choices,
            media_mode_image=self.media_mode_image,
            media_mode_video=self.media_mode_video,
            media_mode_ai=self.media_mode_ai,
            ai_random_mode=self.ai_random_mode,
            video_playback_modes=self.video_playback_modes,
            sync_config_key=self.sync_config_key,
            sync_supported_media_modes=self.sync_supported_media_modes,
            stream_playback_history_limit=self.stream_playback_history_limit,
            get_runtime_thumbnail_payload=self.get_runtime_thumbnail_payload,
        )

    def update_stream_config(self, stream_id: str, conf: Dict[str, Any]) -> None:
        payload_to_emit: Optional[Dict[str, Any]] = None
        needs_refresh = False
        with self._lock:
            state = self._states.get(stream_id)
            if state is None:
                state = self._new_state(stream_id)
                self._states[stream_id] = state
            apply_result = state.apply_config(conf)
            if not state.should_run():
                if state.current_media or state.error:
                    state.reset_state()
                    payload_to_emit = state.to_payload()
                return
            if state.current_media is None:
                needs_refresh = True
            elif apply_result.get("enabled_changed") or apply_result.get("sources_changed") or apply_result.get("duration_changed"):
                state.reset_state()
                needs_refresh = True
            elif apply_result.get("sync_changed"):
                if state.sync_active():
                    self._align_sync_schedule(state)
                else:
                    self._restore_default_schedule(state)
        if payload_to_emit:
            self._emit_state(payload_to_emit, room=stream_id)
        if needs_refresh:
            payload = self._advance_stream(stream_id, reason="config")
            if payload:
                self._emit_state(payload, room=stream_id)

    def realign_sync_timer(self, stream_id: str) -> None:
        with self._lock:
            state = self._states.get(stream_id)
            if not state:
                return
            if state.sync_active():
                self._align_sync_schedule(state)
            else:
                self._restore_default_schedule(state)

    def sync_timer_group(self, timer_id: str, *, force_refresh: bool = False) -> None:
        if not timer_id:
            return
        offset_map: Dict[float, List[str]] = {}
        with self._lock:
            for stream_id, state in self._states.items():
                if not state.sync_active() or state.sync_timer_id != timer_id:
                    continue
                if force_refresh:
                    offset_map.setdefault(float(state.sync_offset or 0.0), []).append(stream_id)
                else:
                    self._align_sync_schedule(state)
        if force_refresh:
            base_switch_at = time.time() + self.sync_switch_lead_seconds
            for stream_ids in offset_map.values():
                for stream_id in stream_ids:
                    payload = self._advance_stream(stream_id, reason="sync_group")
                    if payload:
                        payload["switch_at"] = base_switch_at
                        self._emit_state(payload, room=stream_id)

    def remove_stream(self, stream_id: str) -> None:
        with self._lock:
            self._states.pop(stream_id, None)

    def get_state(self, stream_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            state = self._states.get(stream_id)
            return state.to_payload() if state else None

    def ensure_started(self, stream_id: str) -> Optional[Dict[str, Any]]:
        payload = self.get_state(stream_id)
        if payload and payload.get("status") != "idle":
            return payload
        payload = self._advance_stream(stream_id, reason="initial")
        if payload:
            self._emit_state(payload, room=stream_id)
        return payload

    def skip_previous(self, stream_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            state = self._states.get(stream_id)
            if not state or not state.should_run():
                return None
            target_index = state.history_index - 1
            entry = state.get_history_entry(target_index)
            if entry is None:
                return None
            media = entry.get("media")
            duration = entry.get("duration")
            playback_mode = entry.get("playback_mode")
            if not media:
                return None
            media_copy = dict(media)
            state.set_media(
                media_copy,
                duration=duration,
                source="history_prev",
                playback_mode=playback_mode,
                history_index=target_index,
                add_to_history=False,
            )
            if state.sync_active():
                self._align_sync_schedule(state)
            payload = state.to_payload()
            media_mode = state.media_mode
        thumbnail_info = self.update_stream_runtime_state(
            stream_id,
            path=media_copy.get("path"),
            kind=media_copy.get("kind"),
            media_mode=media_mode,
            stream_url=media_copy.get("stream_url"),
            source="history_prev",
        )
        if payload and thumbnail_info is not None:
            payload["thumbnail"] = thumbnail_info
        self._emit_state(payload, room=stream_id)
        return payload

    def skip_next(self, stream_id: str) -> Optional[Dict[str, Any]]:
        media_copy: Optional[Dict[str, Any]] = None
        media_mode: Optional[str] = None
        with self._lock:
            state = self._states.get(stream_id)
            if not state or not state.should_run():
                state_to_emit = state.to_payload() if state else None
                media_mode = state.media_mode if state else None
            else:
                target_index = state.history_index + 1
                entry = state.get_history_entry(target_index)
                if entry:
                    media = entry.get("media")
                    duration = entry.get("duration")
                    playback_mode = entry.get("playback_mode")
                    if media:
                        media_copy = dict(media)
                        state.set_media(
                            media_copy,
                            duration=duration,
                            source="history_next",
                            playback_mode=playback_mode,
                            history_index=target_index,
                            add_to_history=False,
                        )
                        if state.sync_active():
                            self._align_sync_schedule(state)
                        state_to_emit = state.to_payload()
                    else:
                        state_to_emit = None
                        media_copy = None
                else:
                    state_to_emit = None
                    media_copy = None
                media_mode = state.media_mode
        if state_to_emit:
            if media_copy is not None:
                thumbnail_info = self.update_stream_runtime_state(
                    stream_id,
                    path=media_copy.get("path"),
                    kind=media_copy.get("kind"),
                    media_mode=media_mode,
                    stream_url=media_copy.get("stream_url"),
                    source="history_next",
                )
                if thumbnail_info is not None:
                    state_to_emit["thumbnail"] = thumbnail_info
            self._emit_state(state_to_emit, room=stream_id)
            return state_to_emit
        payload = self._advance_stream(stream_id, reason="manual")
        if payload:
            self._emit_state(payload, room=stream_id)
        return payload

    def pause(self, stream_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            state = self._states.get(stream_id)
            if not state or not state.pause():
                return None
            payload = state.to_payload()
        self._emit_state(payload, room=stream_id)
        return payload

    def resume(self, stream_id: str) -> Optional[Dict[str, Any]]:
        payload: Optional[Dict[str, Any]] = None
        need_new_media = False
        with self._lock:
            state = self._states.get(stream_id)
            if not state:
                return None
            if state.current_media is None:
                need_new_media = True
            else:
                if not state.resume():
                    return None
                payload = state.to_payload()
        if need_new_media:
            payload = self._advance_stream(stream_id, reason="resume")
            if payload:
                self._emit_state(payload, room=stream_id)
            return payload
        if payload:
            self._emit_state(payload, room=stream_id)
        return payload

    def set_volume(self, stream_id: str, volume: float) -> Optional[Dict[str, Any]]:
        clamped = max(0.0, min(1.0, float(volume)))
        with self._lock:
            state = self._states.get(stream_id)
            if not state:
                return None
            state.video_volume = clamped
            payload = state.to_payload()
        self._emit_state(payload, room=stream_id)
        return payload

    def toggle(self, stream_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            state = self._states.get(stream_id)
            if not state:
                return None
            paused = state.is_paused
        return self.resume(stream_id) if paused else self.pause(stream_id)

    def _mark_sync_sent(self, stream_id: Optional[str], server_time: Optional[float]) -> None:
        if not stream_id:
            return
        with self._lock:
            state = self._states.get(stream_id)
            if not state:
                return
            state.last_sync_emit = float(server_time) if isinstance(server_time, (int, float)) else time.time()

    def _emit_state(self, payload: Dict[str, Any], *, room: Optional[str] = None, event: Optional[str] = None) -> None:
        stream_id = payload.get("stream_id")
        server_time = payload.get("server_time")
        if stream_id:
            self._mark_sync_sent(stream_id, server_time if isinstance(server_time, (int, float)) else None)
        target_room = room or stream_id
        event_name = event or self.stream_update_event
        if target_room:
            self.safe_emit(event_name, payload, to=target_room)
        else:
            self.safe_emit(event_name, payload)

    def _next_media(self, state: StreamPlaybackState) -> Optional[Dict[str, Any]]:
        entries = self.list_media(
            state.folder,
            library=self.library_for_media_mode(state.media_mode),
        )
        if state.media_mode in (self.media_mode_image, self.media_mode_ai):
            entries = [item for item in entries if item.get("kind") == "image"]
        elif state.media_mode == self.media_mode_video:
            entries = [item for item in entries if item.get("kind") == "video"]
        if not entries:
            return None
        if state.shuffle:
            pool = entries
            if state.current_media and len(entries) > 1:
                current_path = state.current_media.get("path")
                filtered = [item for item in entries if item.get("path") != current_path]
                if filtered:
                    pool = filtered
            choice = random.choice(pool)
        else:
            index = state.sequence_index % len(entries)
            choice = entries[index]
            state.sequence_index = (index + 1) % len(entries)
        return dict(choice)

    def _compute_video_duration_seconds(self, rel_path: Optional[str]) -> Optional[float]:
        if not rel_path:
            return None
        absolute = self.resolve_media_path(rel_path)
        if not absolute or self.cv2_module is None:
            return None
        try:
            mtime_ns = absolute.stat().st_mtime_ns
        except OSError:
            mtime_ns = 0
        cache_key = f"{rel_path}:{mtime_ns}"
        cached = self.video_duration_cache.get(cache_key)
        if cached is not None:
            return cached if cached > 0 else None
        capture = self.cv2_module.VideoCapture(str(absolute))
        if not capture.isOpened():
            capture.release()
            self.video_duration_cache[cache_key] = -1.0
            return None
        duration = None
        try:
            fps = capture.get(self.cv2_module.CAP_PROP_FPS) or 0.0
            frame_count = capture.get(self.cv2_module.CAP_PROP_FRAME_COUNT) or 0.0
            if fps > 0 and frame_count > 0:
                duration = frame_count / fps
            else:
                milliseconds = capture.get(self.cv2_module.CAP_PROP_POS_MSEC) or 0.0
                if milliseconds > 0:
                    duration = milliseconds / 1000.0
        except Exception:
            duration = None
        finally:
            capture.release()
        if duration is None or duration <= 0:
            self.video_duration_cache[cache_key] = -1.0
            return None
        self.video_duration_cache[cache_key] = float(duration)
        return float(duration)

    def _compute_duration(self, state: StreamPlaybackState, media: Dict[str, Any]) -> Optional[float]:
        kind = media.get("kind")
        if kind == "image":
            return max(1.0, float(state.duration_setting))
        if kind == "video":
            playback_mode = state.video_playback_mode
            if playback_mode == "loop":
                return None
            if playback_mode == "duration":
                return max(1.0, float(state.duration_setting))
            if playback_mode == "until_end":
                duration = self._compute_video_duration_seconds(media.get("path"))
                if duration is None:
                    return max(1.0, float(state.duration_setting))
                return duration
        return max(1.0, float(state.duration_setting))

    def _resolve_sync_timer(self, state: StreamPlaybackState) -> Optional[Dict[str, float]]:
        if not state.sync_active():
            return None
        entry = self.get_sync_timer_config(state.sync_timer_id)
        if not isinstance(entry, dict):
            return None
        interval = self.coerce_float(entry.get("interval"), 0.0)
        try:
            interval = float(interval)
        except (TypeError, ValueError):
            interval = 0.0
        if math.isnan(interval) or math.isinf(interval) or interval <= 0:
            return None
        return {"interval": interval}

    def _align_sync_schedule(self, state: StreamPlaybackState) -> None:
        if not state.sync_active() or state.is_paused or not state.current_media:
            return
        timer_conf = self._resolve_sync_timer(state)
        if not timer_conf:
            return
        now = time.time()
        next_tick = self.compute_next_sync_tick(now, timer_conf["interval"], state.sync_offset)
        if next_tick is None:
            return
        state.next_auto_event = next_tick
        state.duration = max(0.1, next_tick - now)

    def _restore_default_schedule(self, state: StreamPlaybackState) -> None:
        if state.is_paused or not state.current_media:
            return
        now = time.time()
        duration = self._compute_duration(state, state.current_media)
        if duration is None:
            state.duration = None
            state.next_auto_event = None
            return
        position = state.get_position(now)
        remaining = max(0.1, duration - position)
        state.duration = duration
        state.next_auto_event = now + remaining

    def _advance_stream(self, stream_id: str, *, reason: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            state = self._states.get(stream_id)
            if not state or not state.should_run():
                return None
            media = self._next_media(state)
            media_mode = state.media_mode
            if media is None:
                state.set_error("no_media")
                payload = state.to_payload()
                runtime_args = {
                    "path": None,
                    "kind": None,
                    "media_mode": media_mode,
                    "stream_url": None,
                    "source": f"playback_{reason}",
                    "force_thumbnail": True,
                }
            else:
                now = time.time()
                duration = self._compute_duration(state, media)
                next_tick: Optional[float] = None
                if state.sync_active():
                    timer_conf = self._resolve_sync_timer(state)
                    if timer_conf:
                        next_tick = self.compute_next_sync_tick(now, timer_conf["interval"], state.sync_offset)
                        if next_tick is not None:
                            duration = max(0.1, next_tick - now)
                playback_mode = state.video_playback_mode if media.get("kind") == "video" else None
                state.set_media(media, duration=duration, source=reason, playback_mode=playback_mode, now=now)
                if next_tick is not None:
                    state.next_auto_event = next_tick
                    state.duration = duration
                payload = state.to_payload()
                runtime_args = {
                    "path": media.get("path"),
                    "kind": media.get("kind"),
                    "media_mode": media_mode,
                    "stream_url": media.get("stream_url"),
                    "source": f"playback_{reason}",
                }
        thumbnail_info = self.update_stream_runtime_state(stream_id, **runtime_args)
        if payload and thumbnail_info is not None:
            payload["thumbnail"] = thumbnail_info
        return payload

    def _loop(self) -> None:
        while not self._stop.is_set():
            now = time.time()
            next_deadline: Optional[float] = None
            due_streams: List[str] = []
            sync_groups: Dict[Tuple[str, float], List[str]] = {}
            sync_payloads: List[Dict[str, Any]] = []
            with self._lock:
                for stream_id, state in self._states.items():
                    if not state.should_run():
                        continue
                    is_due = False
                    if not state.is_paused:
                        deadline = state.next_auto_event
                        if state.sync_active() and deadline is None:
                            timer_conf = self._resolve_sync_timer(state)
                            if timer_conf:
                                deadline = self.compute_next_sync_tick(now, timer_conf["interval"], state.sync_offset)
                                state.next_auto_event = deadline
                        if deadline is not None:
                            if deadline <= now:
                                if state.sync_active() and state.sync_timer_id:
                                    sync_groups.setdefault((state.sync_timer_id, state.sync_offset), []).append(stream_id)
                                else:
                                    due_streams.append(stream_id)
                                is_due = True
                            elif next_deadline is None or deadline < next_deadline:
                                next_deadline = deadline
                    if state.current_media and not state.is_paused and not is_due:
                        last_sync = state.last_sync_emit if isinstance(state.last_sync_emit, (int, float)) else 0.0
                        if last_sync <= 0.0 or (now - last_sync) >= self.stream_sync_interval_seconds:
                            sync_payloads.append(state.to_sync_payload(now))
            for stream_id in due_streams:
                payload = self._advance_stream(stream_id, reason="auto")
                if payload:
                    payload["switch_at"] = time.time() + self.sync_switch_lead_seconds
                    self._emit_state(payload, room=stream_id, event=self.stream_update_event)
            if sync_groups:
                base_switch_at = time.time() + self.sync_switch_lead_seconds
                for stream_ids in sync_groups.values():
                    for stream_id in stream_ids:
                        payload = self._advance_stream(stream_id, reason="auto_sync")
                        if payload:
                            payload["switch_at"] = base_switch_at
                            self._emit_state(payload, room=stream_id, event=self.stream_update_event)
            for payload in sync_payloads:
                self._emit_state(payload, event=self.sync_time_event)
            sleep_for = self.stream_sync_interval_seconds if next_deadline is None else max(
                0.1,
                min(self.stream_sync_interval_seconds, next_deadline - time.time()),
            )
            self._stop.wait(sleep_for)
