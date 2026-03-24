"""Stream configuration service."""

from __future__ import annotations

from typing import Any, Dict, Optional


class StreamConfigService:
    def __init__(
        self,
        *,
        settings,
        ai_jobs,
        ai_job_controls,
        ai_jobs_lock,
        playback_manager,
        auto_scheduler,
        picsum_scheduler,
        cleanup_temp_outputs,
        save_settings_debounced,
        safe_emit,
        get_global_tags,
        ensure_ai_defaults,
        ensure_picsum_defaults,
        ensure_timer_defaults,
        ensure_sync_defaults,
        ensure_background_defaults,
        ensure_tag_defaults,
        reconcile_stale_ai_state,
        update_stream_runtime_state,
        refresh_embed_metadata,
        sanitize_picsum_settings,
        default_picsum_settings,
        sanitize_sync_config,
        sanitize_ai_settings,
        ai_settings_match_defaults,
        default_ai_settings,
        default_ai_state,
        default_stream_config,
        detect_media_kind,
        infer_media_mode,
        coerce_bool,
        coerce_int,
        slugify,
        sanitize_stream_tags,
        register_global_tags,
        media_mode_choices,
        media_mode_variants,
        media_mode_ai,
        media_mode_livestream,
        media_mode_video,
        ai_modes,
        ai_generate_mode,
        image_quality_choices,
        video_playback_modes,
        tag_key,
        picsum_settings_key,
        sync_config_key,
        sync_timers_key,
        ai_settings_key,
        ai_state_key,
        stream_order_key,
        stream_runtime_lock,
        stream_runtime_state,
    ) -> None:
        self.settings = settings
        self.ai_jobs = ai_jobs
        self.ai_job_controls = ai_job_controls
        self.ai_jobs_lock = ai_jobs_lock
        self.playback_manager = playback_manager
        self.auto_scheduler = auto_scheduler
        self.picsum_scheduler = picsum_scheduler
        self.cleanup_temp_outputs = cleanup_temp_outputs
        self.save_settings_debounced = save_settings_debounced
        self.safe_emit = safe_emit
        self.get_global_tags = get_global_tags
        self.ensure_ai_defaults = ensure_ai_defaults
        self.ensure_picsum_defaults = ensure_picsum_defaults
        self.ensure_timer_defaults = ensure_timer_defaults
        self.ensure_sync_defaults = ensure_sync_defaults
        self.ensure_background_defaults = ensure_background_defaults
        self.ensure_tag_defaults = ensure_tag_defaults
        self.reconcile_stale_ai_state = reconcile_stale_ai_state
        self.update_stream_runtime_state = update_stream_runtime_state
        self.refresh_embed_metadata = refresh_embed_metadata
        self.sanitize_picsum_settings = sanitize_picsum_settings
        self.default_picsum_settings = default_picsum_settings
        self.sanitize_sync_config = sanitize_sync_config
        self.sanitize_ai_settings = sanitize_ai_settings
        self.ai_settings_match_defaults = ai_settings_match_defaults
        self.default_ai_settings = default_ai_settings
        self.default_ai_state = default_ai_state
        self.default_stream_config = default_stream_config
        self.detect_media_kind = detect_media_kind
        self.infer_media_mode = infer_media_mode
        self.coerce_bool = coerce_bool
        self.coerce_int = coerce_int
        self.slugify = slugify
        self.sanitize_stream_tags = sanitize_stream_tags
        self.register_global_tags = register_global_tags
        self.media_mode_choices = media_mode_choices
        self.media_mode_variants = media_mode_variants
        self.media_mode_ai = media_mode_ai
        self.media_mode_livestream = media_mode_livestream
        self.media_mode_video = media_mode_video
        self.ai_modes = ai_modes
        self.ai_generate_mode = ai_generate_mode
        self.image_quality_choices = image_quality_choices
        self.video_playback_modes = video_playback_modes
        self.tag_key = tag_key
        self.picsum_settings_key = picsum_settings_key
        self.sync_config_key = sync_config_key
        self.sync_timers_key = sync_timers_key
        self.ai_settings_key = ai_settings_key
        self.ai_state_key = ai_state_key
        self.stream_order_key = stream_order_key
        self.stream_runtime_lock = stream_runtime_lock
        self.stream_runtime_state = stream_runtime_state

    def _get_stream_order(self) -> list[str]:
        available = [key for key in self.settings.keys() if isinstance(key, str) and not key.startswith("_")]
        available_set = set(available)
        raw = self.settings.get(self.stream_order_key)
        ordered: list[str] = []
        seen: set[str] = set()
        if isinstance(raw, list):
            for entry in raw:
                if not isinstance(entry, str):
                    continue
                stream_id = entry.strip()
                if stream_id in available_set and stream_id not in seen:
                    ordered.append(stream_id)
                    seen.add(stream_id)
        for stream_id in available:
            if stream_id not in seen:
                ordered.append(stream_id)
        return ordered

    def _set_stream_order(self, ordered_ids: list[str]) -> None:
        self.settings[self.stream_order_key] = ordered_ids

    def create_stream(self) -> str:
        idx = 1
        while True:
            new_id = f"stream{idx}"
            if new_id not in self.settings:
                ordered_ids = self._get_stream_order()
                self.settings[new_id] = self.default_stream_config()
                self.settings[new_id]["label"] = new_id.capitalize()
                ordered_ids.append(new_id)
                self._set_stream_order(ordered_ids)
                self.update_stream_runtime_state(
                    new_id,
                    path=self.settings[new_id].get("selected_image"),
                    kind=self.settings[new_id].get("selected_media_kind"),
                    media_mode=self.settings[new_id].get("media_mode"),
                    stream_url=self.settings[new_id].get("stream_url"),
                    source="stream_created",
                )
                if self.playback_manager is not None:
                    self.playback_manager.update_stream_config(new_id, self.settings[new_id])
                self.save_settings_debounced()
                if self.auto_scheduler is not None:
                    self.auto_scheduler.reschedule(new_id)
                self.safe_emit("streams_changed", {"action": "added", "stream_id": new_id})
                return new_id
            idx += 1

    def delete_stream(self, stream_id: str) -> bool:
        if stream_id not in self.settings:
            return False
        with self.ai_jobs_lock:
            self.ai_jobs.pop(stream_id, None)
            self.ai_job_controls.pop(stream_id, None)
        self.cleanup_temp_outputs(stream_id)
        self.settings.pop(stream_id)
        self._set_stream_order([item for item in self._get_stream_order() if item != stream_id])
        with self.stream_runtime_lock:
            self.stream_runtime_state.pop(stream_id, None)
        if self.playback_manager is not None:
            self.playback_manager.remove_stream(stream_id)
        self.save_settings_debounced()
        if self.auto_scheduler is not None:
            self.auto_scheduler.remove(stream_id)
        if self.picsum_scheduler is not None:
            self.picsum_scheduler.remove(stream_id)
        self.safe_emit("streams_changed", {"action": "deleted", "stream_id": stream_id})
        return True

    def reorder_streams(self, ordered_ids: list[str]) -> list[str]:
        current = self._get_stream_order()
        current_set = set(current)
        requested: list[str] = []
        seen: set[str] = set()
        for entry in ordered_ids:
            if not isinstance(entry, str):
                continue
            stream_id = entry.strip()
            if stream_id in current_set and stream_id not in seen:
                requested.append(stream_id)
                seen.add(stream_id)
        requested.extend([stream_id for stream_id in current if stream_id not in seen])
        self._set_stream_order(requested)
        self.save_settings_debounced()
        self.safe_emit("streams_changed", {"action": "reordered", "stream_order": requested})
        return requested

    def get_stream_settings_payload(self, stream_id: str) -> Dict[str, Any]:
        if stream_id not in self.settings:
            raise KeyError(stream_id)
        conf = self.settings[stream_id]
        self.reconcile_stale_ai_state(stream_id, conf)
        self.ensure_ai_defaults(conf)
        self.ensure_picsum_defaults(conf)
        self.ensure_timer_defaults(conf)
        self.ensure_sync_defaults(conf)
        self.ensure_background_defaults(conf)
        self.ensure_tag_defaults(conf)
        return conf

    def update_stream(self, stream_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        if stream_id not in self.settings:
            raise KeyError(stream_id)

        conf = self.settings[stream_id]
        self.ensure_ai_defaults(conf)
        self.ensure_picsum_defaults(conf)
        self.ensure_timer_defaults(conf)
        self.ensure_sync_defaults(conf)
        previous_mode = conf.get("mode")
        previous_sync = dict(conf.get(self.sync_config_key)) if isinstance(conf.get(self.sync_config_key), dict) else {}
        requested_mode = data.get("mode")
        requested_mode = requested_mode.strip().lower() if isinstance(requested_mode, str) else ""

        stream_url_changed = False
        media_mode_changed = False

        for key in [
            "mode", "folder", "selected_image", "duration", "shuffle", "stream_url",
            "image_quality", "yt_cc", "yt_mute", "yt_quality", "label", "hide_nsfw",
            "background_blur_enabled", "background_blur_amount", "video_playback_mode",
            "video_volume", "selected_media_kind", "media_mode", self.tag_key,
        ]:
            if key not in data:
                continue
            val = data[key]
            if key == "stream_url":
                normalized_url = val.strip() if isinstance(val, str) else ""
                conf[key] = normalized_url if normalized_url and normalized_url.lower() != "none" else None
                stream_url_changed = True
            elif key == "mode":
                conf[key] = val
                if isinstance(val, str) and val.strip().lower() == "livestream":
                    media_mode_changed = True
            elif key == "label":
                new_label = (val or "").strip()
                new_slug = self.slugify(new_label)
                if new_slug:
                    for other_id, other_conf in self.settings.items():
                        if other_id.startswith("_") or other_id == stream_id:
                            continue
                        other_label = other_conf.get("label") or other_id
                        if self.slugify(other_label) == new_slug:
                            raise ValueError("Another stream already uses this name")
                conf[key] = new_label
            elif key == self.tag_key:
                tags_payload = self.sanitize_stream_tags(val)
                conf[self.tag_key] = tags_payload
                self.register_global_tags(tags_payload)
            elif key == "hide_nsfw":
                conf[key] = bool(val)
            elif key == "background_blur_enabled":
                conf[key] = self.coerce_bool(val, conf.get(key, False))
            elif key == "background_blur_amount":
                amount = self.coerce_int(val, conf.get(key, 50))
                conf[key] = max(0, min(100, amount))
            elif key == "video_playback_mode":
                normalized = (val or "").strip().lower() if isinstance(val, str) else ""
                if normalized not in self.video_playback_modes:
                    normalized = "duration"
                conf[key] = normalized
            elif key == "video_volume":
                try:
                    volume = float(val)
                except (TypeError, ValueError):
                    volume = conf.get(key, 1.0)
                conf[key] = max(0.0, min(1.0, volume))
            elif key == "selected_media_kind":
                kind = val.strip().lower() if isinstance(val, str) else ""
                if kind not in ("image", "video"):
                    kind = self.detect_media_kind(conf.get("selected_image"))
                conf[key] = kind
            elif key == "media_mode":
                media_mode = val.strip().lower() if isinstance(val, str) else ""
                if media_mode in self.media_mode_choices:
                    conf["media_mode"] = media_mode
                    if media_mode == self.media_mode_ai:
                        if requested_mode not in self.ai_modes:
                            conf["mode"] = self.ai_generate_mode
                    elif media_mode == self.media_mode_livestream:
                        conf["mode"] = "livestream"
                    media_mode_changed = True
            elif key == "selected_image":
                conf[key] = val
                if "selected_media_kind" not in data:
                    conf["selected_media_kind"] = self.detect_media_kind(val)
            elif key == "image_quality":
                normalized = (val or "").strip().lower() if isinstance(val, str) else ""
                if normalized not in self.image_quality_choices:
                    normalized = "auto"
                conf[key] = normalized
            else:
                conf[key] = val

        if self.picsum_settings_key in data:
            incoming_picsum = data.get(self.picsum_settings_key)
            conf[self.picsum_settings_key] = self.sanitize_picsum_settings(
                incoming_picsum,
                defaults=conf.get(self.picsum_settings_key),
            )
            if isinstance(incoming_picsum, dict):
                seed_flag: Optional[bool] = None
                if "seed_custom" in incoming_picsum:
                    seed_flag = bool(incoming_picsum.get("seed_custom"))
                elif "seed" in incoming_picsum:
                    seed_candidate = incoming_picsum.get("seed")
                    if isinstance(seed_candidate, str):
                        seed_flag = bool(seed_candidate.strip())
                    else:
                        seed_flag = bool(seed_candidate)
                if seed_flag is not None:
                    conf["_picsum_seed_custom"] = seed_flag
            if "_picsum_seed_custom" not in conf:
                conf["_picsum_seed_custom"] = False
        else:
            conf[self.picsum_settings_key] = self.sanitize_picsum_settings(
                conf.get(self.picsum_settings_key),
                defaults=self.default_picsum_settings(),
            )

        if self.sync_config_key in data:
            timers = self.settings.get(self.sync_timers_key, {}) if isinstance(self.settings.get(self.sync_timers_key), dict) else {}
            conf[self.sync_config_key] = self.sanitize_sync_config(
                data.get(self.sync_config_key),
                defaults=conf.get(self.sync_config_key),
                timers=timers,
            )

        media_mode = conf.get("media_mode")
        media_mode = media_mode.strip().lower() if isinstance(media_mode, str) else ""
        if media_mode not in self.media_mode_choices:
            media_mode = self.infer_media_mode(conf)
        conf["media_mode"] = media_mode

        current_mode_raw = conf.get("mode")
        current_mode = current_mode_raw.strip().lower() if isinstance(current_mode_raw, str) else ""
        if media_mode == self.media_mode_ai:
            if current_mode not in self.ai_modes:
                conf["mode"] = self.ai_generate_mode
            else:
                conf["mode"] = current_mode
        elif media_mode == self.media_mode_livestream:
            conf["mode"] = "livestream"
        else:
            allowed = self.media_mode_variants.get(media_mode, {"random", "specific"})
            if current_mode not in allowed:
                fallback_mode = "random" if "random" in allowed else next(iter(sorted(allowed)))
                conf["mode"] = fallback_mode
            else:
                conf["mode"] = current_mode
            if media_mode == self.media_mode_video:
                if conf["mode"] == "specific":
                    conf["video_playback_mode"] = "loop"
                elif conf["mode"] == "random" and conf.get("video_playback_mode") == "loop":
                    conf["video_playback_mode"] = "duration"

        needs_metadata_refresh = stream_url_changed or media_mode_changed or not conf.get("embed_metadata")
        if needs_metadata_refresh:
            self.refresh_embed_metadata(stream_id, conf, force=stream_url_changed or media_mode_changed)

        mode_requested = requested_mode
        if mode_requested in self.ai_modes and previous_mode not in self.ai_modes and not conf.get("_ai_customized", False):
            conf[self.ai_settings_key] = self.default_ai_settings()
            conf[self.ai_state_key] = self.default_ai_state()
            conf["_ai_customized"] = False

        if isinstance(data.get("ai_settings"), dict):
            conf[self.ai_settings_key] = self.sanitize_ai_settings(data["ai_settings"], conf[self.ai_settings_key])
            conf["_ai_customized"] = not self.ai_settings_match_defaults(conf[self.ai_settings_key])

        self.ensure_background_defaults(conf)
        self.ensure_tag_defaults(conf)
        self.update_stream_runtime_state(
            stream_id,
            path=conf.get("selected_image"),
            kind=conf.get("selected_media_kind"),
            media_mode=conf.get("media_mode"),
            stream_url=conf.get("stream_url"),
            source="settings_update",
        )
        if self.playback_manager is not None:
            self.playback_manager.update_stream_config(stream_id, conf)
            current_sync = conf.get(self.sync_config_key) if isinstance(conf.get(self.sync_config_key), dict) else {}
            sync_timer_id = current_sync.get("timer_id")
            sync_offset = current_sync.get("offset")
            sync_changed = previous_sync.get("timer_id") != sync_timer_id or previous_sync.get("offset") != sync_offset
            if sync_changed and sync_timer_id:
                self.playback_manager.sync_timer_group(sync_timer_id, force_refresh=True)
        self.save_settings_debounced()
        if self.auto_scheduler is not None:
            self.auto_scheduler.reschedule(stream_id)
        if self.picsum_scheduler is not None:
            self.picsum_scheduler.reschedule(stream_id)
        tags = self.get_global_tags()
        self.safe_emit("refresh", {"stream_id": stream_id, "config": conf, "tags": tags})
        return {"status": "success", "new_config": conf, "tags": tags}
