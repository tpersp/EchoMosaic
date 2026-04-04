"""Background schedulers for AI auto-generation and Picsum refresh."""

from __future__ import annotations

import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


class BaseScheduler:
    """Generic interval-based scheduler for background tasks."""

    def __init__(self, name: str, *, settings) -> None:
        self.name = name
        self.settings = settings
        self._lock = threading.Lock()
        self._next_run: Dict[str, float] = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name=name, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def reschedule_all(self) -> None:
        for stream_id in list(self.settings.keys()):
            if not isinstance(stream_id, str) or stream_id.startswith("_"):
                continue
            self.reschedule(stream_id)

    def remove(self, stream_id: str) -> None:
        with self._lock:
            self._next_run.pop(stream_id, None)
        self._on_remove(stream_id)

    def _loop(self) -> None:
        while not self._stop.is_set():
            now = time.time()
            due: List[str] = []
            with self._lock:
                for stream_id, next_ts in list(self._next_run.items()):
                    if next_ts <= now:
                        due.append(stream_id)
            for stream_id in due:
                self._trigger_stream(stream_id)
            self._stop.wait(5.0)

    def reschedule(self, stream_id: str, *, base_time: Optional[float] = None) -> None:
        raise NotImplementedError

    def _trigger_stream(self, stream_id: str) -> None:
        raise NotImplementedError

    def _on_remove(self, stream_id: str) -> None:
        pass


class AutoGenerateScheduler(BaseScheduler):
    def __init__(
        self,
        *,
        settings,
        timer_manager_cls,
        canonical_timer_mode,
        timer_snap_enabled,
        ensure_ai_defaults,
        ensure_picsum_defaults,
        ensure_timer_defaults,
        queue_ai_generation,
        emit_ai_update,
        format_timer_label,
        normalize_timer_label,
        log_timer_schedule,
        ai_state_key: str,
        ai_settings_key: str,
        ai_modes,
        auto_generation_busy_cls,
        auto_generation_prompt_missing_cls,
        auto_generation_unavailable_cls,
        auto_generation_error_cls,
    ) -> None:
        super().__init__("AutoGenerateScheduler", settings=settings)
        self.timer_manager_cls = timer_manager_cls
        self.canonical_timer_mode = canonical_timer_mode
        self.timer_snap_enabled = timer_snap_enabled
        self.ensure_ai_defaults = ensure_ai_defaults
        self.ensure_picsum_defaults = ensure_picsum_defaults
        self.ensure_timer_defaults = ensure_timer_defaults
        self.queue_ai_generation = queue_ai_generation
        self.emit_ai_update = emit_ai_update
        self.format_timer_label = format_timer_label
        self.normalize_timer_label = normalize_timer_label
        self.log_timer_schedule = log_timer_schedule
        self.ai_state_key = ai_state_key
        self.ai_settings_key = ai_settings_key
        self.ai_modes = set(ai_modes)
        self.auto_generation_busy_cls = auto_generation_busy_cls
        self.auto_generation_prompt_missing_cls = auto_generation_prompt_missing_cls
        self.auto_generation_unavailable_cls = auto_generation_unavailable_cls
        self.auto_generation_error_cls = auto_generation_error_cls

    def _build_timer(self, stream_id: str, conf: Dict[str, Any]):
        timer_mode = self.canonical_timer_mode(conf)
        timer = self.timer_manager_cls(
            mode=timer_mode or conf.get("mode") or "",
            stream_id=stream_id,
            config_owner=conf,
            snap_provider=self.timer_snap_enabled,
        )
        return timer_mode, timer

    def _on_remove(self, stream_id: str) -> None:
        conf = self.settings.get(stream_id)
        if isinstance(conf, dict):
            self.ensure_timer_defaults(conf)
            _, timer = self._build_timer(stream_id, conf)
            timer.update_next(None)
        self._update_state(stream_id, next_auto_trigger=None, last_auto_error=None)

    def reschedule(self, stream_id: str, *, base_time: Optional[float] = None) -> None:
        conf = self.settings.get(stream_id)
        if not isinstance(conf, dict):
            self.remove(stream_id)
            return
        self.ensure_ai_defaults(conf)
        self.ensure_picsum_defaults(conf)
        self.ensure_timer_defaults(conf)

        timer_mode, timer = self._build_timer(stream_id, conf)
        reference_ts = base_time if base_time is not None else time.time()

        if timer_mode not in self.ai_modes or not timer.is_enabled():
            timer.update_next(None)
            with self._lock:
                self._next_run.pop(stream_id, None)
            self._update_state(stream_id, next_auto_trigger=None, last_auto_error=None)
            return

        next_dt = timer.compute_next(reference=datetime.fromtimestamp(reference_ts))
        if next_dt is None:
            timer.update_next(None)
            with self._lock:
                self._next_run.pop(stream_id, None)
            self._update_state(stream_id, next_auto_trigger=None, last_auto_error=None)
            return
        with self._lock:
            self._next_run[stream_id] = next_dt.timestamp()
        timer.update_next(next_dt)
        self._update_state(stream_id, next_auto_trigger=self.format_timer_label(next_dt), last_auto_error=None)
        self.log_timer_schedule(stream_id, timer_mode, next_dt, timer.offset_minutes())

    def _update_state(self, stream_id: str, **updates: Any) -> None:
        conf = self.settings.get(stream_id)
        if not isinstance(conf, dict):
            return
        self.ensure_ai_defaults(conf)
        self.ensure_picsum_defaults(conf)
        state = conf[self.ai_state_key]
        changed = False
        for key, value in updates.items():
            normalized_value = self.normalize_timer_label(value) if key in {"next_auto_trigger", "last_auto_trigger"} else value
            if state.get(key) != normalized_value:
                state[key] = normalized_value
                changed = True
        if changed:
            self.emit_ai_update(stream_id, state)

    def _trigger_stream(self, stream_id: str) -> None:
        conf = self.settings.get(stream_id)
        if not isinstance(conf, dict):
            self.remove(stream_id)
            return
        self.ensure_ai_defaults(conf)
        self.ensure_picsum_defaults(conf)
        self.ensure_timer_defaults(conf)
        timer_mode, timer = self._build_timer(stream_id, conf)
        ai_settings = conf[self.ai_settings_key]
        prompt = str(ai_settings.get("prompt") or "").strip()
        if timer_mode not in self.ai_modes or not timer.is_enabled():
            self.remove(stream_id)
            return
        if not prompt:
            self._update_state(stream_id, last_auto_error="Enter a prompt to generate an image.")
            self.reschedule(stream_id)
            return
        try:
            self.queue_ai_generation(stream_id, ai_settings, trigger_source="auto")
        except self.auto_generation_busy_cls:
            self._update_state(stream_id, last_auto_error=None)
            self.reschedule(stream_id, base_time=time.time())
        except self.auto_generation_prompt_missing_cls as exc:
            self._update_state(stream_id, last_auto_error=str(exc))
            self.reschedule(stream_id)
        except self.auto_generation_unavailable_cls as exc:
            self._update_state(stream_id, last_auto_error=str(exc))
            self.reschedule(stream_id)
        except self.auto_generation_error_cls as exc:
            self._update_state(stream_id, last_auto_error=str(exc))
            self.reschedule(stream_id)
        else:
            timer.mark_trigger()
            self._update_state(stream_id, last_auto_error=None)
            self.reschedule(stream_id, base_time=time.time())


class PicsumAutoScheduler(BaseScheduler):
    def __init__(
        self,
        *,
        settings,
        timer_manager_cls,
        canonical_timer_mode,
        timer_snap_enabled,
        ensure_picsum_defaults,
        ensure_timer_defaults,
        refresh_picsum_stream,
        broadcast_picsum_update,
        save_settings_debounced,
        format_timer_label,
        log_timer_schedule,
        picsum_settings_key: str,
        picsum_mode: str,
    ) -> None:
        super().__init__("PicsumAutoScheduler", settings=settings)
        self.timer_manager_cls = timer_manager_cls
        self.canonical_timer_mode = canonical_timer_mode
        self.timer_snap_enabled = timer_snap_enabled
        self.ensure_picsum_defaults = ensure_picsum_defaults
        self.ensure_timer_defaults = ensure_timer_defaults
        self.refresh_picsum_stream = refresh_picsum_stream
        self.broadcast_picsum_update = broadcast_picsum_update
        self.save_settings_debounced = save_settings_debounced
        self.format_timer_label = format_timer_label
        self.log_timer_schedule = log_timer_schedule
        self.picsum_settings_key = picsum_settings_key
        self.picsum_mode = picsum_mode

    def _build_timer(self, stream_id: str, conf: Dict[str, Any]):
        timer_mode = self.canonical_timer_mode(conf)
        timer = self.timer_manager_cls(
            mode=timer_mode or conf.get("mode") or "",
            stream_id=stream_id,
            config_owner=conf,
            snap_provider=self.timer_snap_enabled,
        )
        return timer_mode, timer

    def _on_remove(self, stream_id: str) -> None:
        conf = self.settings.get(stream_id)
        if isinstance(conf, dict):
            self.ensure_timer_defaults(conf)
            _, timer = self._build_timer(stream_id, conf)
            timer.update_next(None)
            picsum = conf.get(self.picsum_settings_key)
            if isinstance(picsum, dict):
                picsum["next_auto_trigger"] = None

    def reschedule(self, stream_id: str, *, base_time: Optional[float] = None) -> None:
        conf = self.settings.get(stream_id)
        if not isinstance(conf, dict):
            self.remove(stream_id)
            return
        self.ensure_picsum_defaults(conf)
        self.ensure_timer_defaults(conf)
        timer_mode, timer = self._build_timer(stream_id, conf)
        reference_ts = base_time if base_time is not None else time.time()
        picsum = conf.get(self.picsum_settings_key) or {}

        if timer_mode != self.picsum_mode or not timer.is_enabled():
            timer.update_next(None)
            with self._lock:
                self._next_run.pop(stream_id, None)
            picsum["next_auto_trigger"] = None
            return

        next_dt = timer.compute_next(reference=datetime.fromtimestamp(reference_ts))
        if next_dt is None:
            timer.update_next(None)
            with self._lock:
                self._next_run.pop(stream_id, None)
            picsum["next_auto_trigger"] = None
            return
        with self._lock:
            self._next_run[stream_id] = next_dt.timestamp()
        timer.update_next(next_dt)
        picsum["next_auto_trigger"] = self.format_timer_label(next_dt)
        self.log_timer_schedule(stream_id, timer_mode, next_dt, timer.offset_minutes())

    def _trigger_stream(self, stream_id: str) -> None:
        conf = self.settings.get(stream_id)
        if not isinstance(conf, dict):
            self.remove(stream_id)
            return
        self.ensure_picsum_defaults(conf)
        self.ensure_timer_defaults(conf)
        timer_mode, timer = self._build_timer(stream_id, conf)
        if timer_mode != self.picsum_mode or not timer.is_enabled():
            self.remove(stream_id)
            return
        result = self.refresh_picsum_stream(stream_id)
        if result is None:
            self.remove(stream_id)
            return
        picsum_conf = conf.get(self.picsum_settings_key) or {}
        moment = datetime.now()
        picsum_conf["last_auto_trigger"] = self.format_timer_label(moment)
        if timer_mode == self.picsum_mode and timer.is_enabled():
            timer.mark_trigger(when=moment)
        self.reschedule(stream_id, base_time=time.time())
        self.save_settings_debounced()
        self.broadcast_picsum_update(
            stream_id,
            conf,
            result["url"],
            result["settings"].get("seed"),
            bool(result.get("seed_custom")),
            result.get("thumbnail"),
        )


def build_auto_schedulers(
    *,
    settings,
    timer_manager_cls,
    canonical_timer_mode,
    timer_snap_enabled,
    ensure_ai_defaults,
    ensure_picsum_defaults,
    ensure_timer_defaults,
    queue_ai_generation,
    emit_ai_update,
    refresh_picsum_stream,
    broadcast_picsum_update,
    save_settings_debounced,
    format_timer_label,
    normalize_timer_label,
    log_timer_schedule,
    ai_state_key: str,
    ai_settings_key: str,
    picsum_settings_key: str,
    ai_modes,
    picsum_mode: str,
    auto_generation_busy_cls,
    auto_generation_prompt_missing_cls,
    auto_generation_unavailable_cls,
    auto_generation_error_cls,
) -> Tuple[AutoGenerateScheduler, PicsumAutoScheduler]:
    auto_scheduler = AutoGenerateScheduler(
        settings=settings,
        timer_manager_cls=timer_manager_cls,
        canonical_timer_mode=canonical_timer_mode,
        timer_snap_enabled=timer_snap_enabled,
        ensure_ai_defaults=ensure_ai_defaults,
        ensure_picsum_defaults=ensure_picsum_defaults,
        ensure_timer_defaults=ensure_timer_defaults,
        queue_ai_generation=queue_ai_generation,
        emit_ai_update=emit_ai_update,
        format_timer_label=format_timer_label,
        normalize_timer_label=normalize_timer_label,
        log_timer_schedule=log_timer_schedule,
        ai_state_key=ai_state_key,
        ai_settings_key=ai_settings_key,
        ai_modes=ai_modes,
        auto_generation_busy_cls=auto_generation_busy_cls,
        auto_generation_prompt_missing_cls=auto_generation_prompt_missing_cls,
        auto_generation_unavailable_cls=auto_generation_unavailable_cls,
        auto_generation_error_cls=auto_generation_error_cls,
    )

    picsum_scheduler = PicsumAutoScheduler(
        settings=settings,
        timer_manager_cls=timer_manager_cls,
        canonical_timer_mode=canonical_timer_mode,
        timer_snap_enabled=timer_snap_enabled,
        ensure_picsum_defaults=ensure_picsum_defaults,
        ensure_timer_defaults=ensure_timer_defaults,
        refresh_picsum_stream=refresh_picsum_stream,
        broadcast_picsum_update=broadcast_picsum_update,
        save_settings_debounced=save_settings_debounced,
        format_timer_label=format_timer_label,
        log_timer_schedule=log_timer_schedule,
        picsum_settings_key=picsum_settings_key,
        picsum_mode=picsum_mode,
    )
    return auto_scheduler, picsum_scheduler
