"""Timer and sync-timer coordination service."""

from __future__ import annotations

import re
import secrets
from typing import Any, Callable, Dict, List, Optional


class TimerSyncService:
    def __init__(
        self,
        *,
        settings,
        config,
        auto_scheduler,
        picsum_scheduler,
        playback_manager,
        ensure_ai_defaults: Callable[[Dict[str, Any]], None],
        ensure_picsum_defaults: Callable[[Dict[str, Any]], None],
        ensure_sync_defaults: Callable[..., Dict[str, Any]],
        save_settings_debounced: Callable[[], None],
        safe_emit: Callable[..., None],
        get_global_tags: Callable[[], List[str]],
        sanitize_sync_timer_entry: Callable[[str, Any], Dict[str, Any]],
        ai_state_key: str,
        picsum_settings_key: str,
        sync_timers_key: str,
        sync_config_key: str,
        sync_timer_default_interval: float,
    ) -> None:
        self.settings = settings
        self.config = config
        self.auto_scheduler = auto_scheduler
        self.picsum_scheduler = picsum_scheduler
        self.playback_manager = playback_manager
        self.ensure_ai_defaults = ensure_ai_defaults
        self.ensure_picsum_defaults = ensure_picsum_defaults
        self.ensure_sync_defaults = ensure_sync_defaults
        self.save_settings_debounced = save_settings_debounced
        self.safe_emit = safe_emit
        self.get_global_tags = get_global_tags
        self.sanitize_sync_timer_entry = sanitize_sync_timer_entry
        self.ai_state_key = ai_state_key
        self.picsum_settings_key = picsum_settings_key
        self.sync_timers_key = sync_timers_key
        self.sync_config_key = sync_config_key
        self.sync_timer_default_interval = sync_timer_default_interval

    def get_sync_timer_config(self, timer_id: Optional[str]) -> Optional[Dict[str, Any]]:
        if not timer_id:
            return None
        timers = self.settings.get(self.sync_timers_key)
        if not isinstance(timers, dict):
            return None
        entry = timers.get(timer_id)
        return entry if isinstance(entry, dict) else None

    def get_sync_timers_snapshot(self) -> List[Dict[str, Any]]:
        timers = self.settings.get(self.sync_timers_key)
        if not isinstance(timers, dict):
            return []
        payload: List[Dict[str, Any]] = []
        for timer_id, entry in timers.items():
            if not isinstance(entry, dict):
                continue
            payload.append(
                {
                    "id": timer_id,
                    "label": entry.get("label") or timer_id,
                    "interval": entry.get("interval"),
                }
            )
        payload.sort(key=lambda item: str(item.get("label") or item.get("id") or "").lower())
        return payload

    def sync_timer_payload(self, timer_id: str, entry: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": timer_id,
            "label": entry.get("label") or timer_id,
            "interval": entry.get("interval"),
        }

    def emit_sync_timer_update(self) -> None:
        self.safe_emit("sync_timers_updated", {"timers": self.get_sync_timers_snapshot()})

    def apply_timer_snap_setting(self, enabled: bool) -> Dict[str, Dict[str, Optional[str]]]:
        self.config["TIMER_SNAP_ENABLED"] = enabled
        if self.auto_scheduler is not None:
            self.auto_scheduler.reschedule_all()
        if self.picsum_scheduler is not None:
            self.picsum_scheduler.reschedule_all()

        summary: Dict[str, Dict[str, Optional[str]]] = {}
        for stream_id, conf in self.settings.items():
            if not isinstance(conf, dict):
                continue
            self.ensure_ai_defaults(conf)
            self.ensure_picsum_defaults(conf)
            ai_state = conf.get(self.ai_state_key)
            picsum_conf = conf.get(self.picsum_settings_key)
            summary[stream_id] = {
                "ai_next": ai_state.get("next_auto_trigger") if isinstance(ai_state, dict) else None,
                "picsum_next": picsum_conf.get("next_auto_trigger") if isinstance(picsum_conf, dict) else None,
            }

        self.save_settings_debounced()
        return summary

    def create_sync_timer(self, payload: Any) -> tuple[Dict[str, Any], int]:
        if not isinstance(payload, dict):
            payload = {}
        label = payload.get("label") or payload.get("name") or "Timer"
        interval = payload.get("interval", self.sync_timer_default_interval)

        def _sync_slugify(value: str) -> str:
            text = (value or "").strip().lower()
            text = re.sub(r"[^a-z0-9]+", "-", text)
            text = re.sub(r"-+", "-", text).strip("-")
            return text

        requested_id = payload.get("id") or _sync_slugify(str(label))
        timer_id = _sync_slugify(str(requested_id)) if requested_id else ""
        if not timer_id:
            timer_id = f"timer-{secrets.token_hex(2)}"

        timers = self.settings.get(self.sync_timers_key)
        if not isinstance(timers, dict):
            timers = {}
        base_id = timer_id
        suffix = 1
        while timer_id in timers:
            timer_id = f"{base_id}-{suffix}"
            suffix += 1

        entry = self.sanitize_sync_timer_entry(timer_id, {"label": label, "interval": interval})
        timers[timer_id] = entry
        self.settings[self.sync_timers_key] = timers
        self.save_settings_debounced()
        self.emit_sync_timer_update()
        return {"timer": self.sync_timer_payload(timer_id, entry), "timers": self.get_sync_timers_snapshot()}, 201

    def delete_sync_timer(self, timer_id: str) -> Dict[str, Any]:
        timers = self.settings.get(self.sync_timers_key)
        if not isinstance(timers, dict):
            timers = {}
        entry = timers.get(timer_id)
        if not isinstance(entry, dict):
            raise KeyError("Timer not found")

        timers.pop(timer_id, None)
        self.settings[self.sync_timers_key] = timers
        affected: List[str] = []
        for stream_id, conf in self.settings.items():
            if not isinstance(conf, dict) or stream_id.startswith("_"):
                continue
            raw_sync = conf.get(self.sync_config_key) if isinstance(conf.get(self.sync_config_key), dict) else {}
            was_using = raw_sync.get("timer_id") == timer_id
            sync_conf = self.ensure_sync_defaults(conf, timers=timers)
            if was_using:
                conf[self.sync_config_key] = {"timer_id": None, "offset": sync_conf.get("offset", 0.0)}
                affected.append(stream_id)
                if self.playback_manager is not None:
                    self.playback_manager.update_stream_config(stream_id, conf)
                self.safe_emit("refresh", {"stream_id": stream_id, "config": conf, "tags": self.get_global_tags()})
        self.save_settings_debounced()
        self.emit_sync_timer_update()
        return {"status": "deleted", "affected_streams": affected, "timers": self.get_sync_timers_snapshot()}

    def update_sync_timer(self, timer_id: str, payload: Any) -> Dict[str, Any]:
        timers = self.settings.get(self.sync_timers_key)
        if not isinstance(timers, dict):
            timers = {}
        entry = timers.get(timer_id)
        if not isinstance(entry, dict):
            raise KeyError("Timer not found")

        if not isinstance(payload, dict):
            payload = {}
        merged = dict(entry)
        for key in ("label", "interval"):
            if key in payload:
                merged[key] = payload[key]
        updated = self.sanitize_sync_timer_entry(timer_id, merged)
        timers[timer_id] = updated
        self.settings[self.sync_timers_key] = timers
        self.save_settings_debounced()
        if self.playback_manager is not None:
            for stream_id, conf in self.settings.items():
                if not isinstance(conf, dict) or stream_id.startswith("_"):
                    continue
                sync_conf = self.ensure_sync_defaults(conf, timers=timers)
                if sync_conf.get("timer_id") == timer_id:
                    self.playback_manager.update_stream_config(stream_id, conf)
                    self.playback_manager.realign_sync_timer(stream_id)
            self.playback_manager.sync_timer_group(timer_id, force_refresh=True)
        self.emit_sync_timer_update()
        return {"timer": self.sync_timer_payload(timer_id, updated), "timers": self.get_sync_timers_snapshot()}
