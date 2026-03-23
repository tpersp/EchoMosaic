"""AI worker execution helpers and long-running generation execution."""

from __future__ import annotations

import secrets
import shutil
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class AIExecutionService:
    def __init__(
        self,
        *,
        settings,
        ai_jobs,
        ai_job_controls,
        ai_jobs_lock,
        job_manager,
        stable_horde_client,
        logger,
        ensure_ai_defaults,
        ensure_picsum_defaults,
        sanitize_ai_settings,
        ai_settings_match_defaults,
        save_settings_debounced,
        safe_emit,
        emit_ai_update_callback,
        update_stream_runtime_state,
        invalidate_media_cache,
        relative_image_path,
        ensure_dir,
        playback_manager,
        ai_output_root,
        ai_temp_root,
        ai_default_persist: bool,
        ai_default_sampler,
        ai_default_width,
        ai_default_height,
        ai_default_steps,
        ai_default_cfg,
        ai_default_samples,
        ai_poll_interval,
        ai_timeout,
        stable_horde_post_processors,
        stable_horde_max_loras,
        ai_settings_key: str,
        ai_state_key: str,
        ai_generate_mode: str,
        ai_random_mode: str,
        ai_specific_mode: str,
        media_mode_ai: str,
        ai_media_library: str,
        stable_horde_cancelled_cls,
        stable_horde_error_cls,
    ) -> None:
        self.settings = settings
        self.ai_jobs = ai_jobs
        self.ai_job_controls = ai_job_controls
        self.ai_jobs_lock = ai_jobs_lock
        self.job_manager = job_manager
        self.stable_horde_client = stable_horde_client
        self.logger = logger
        self.ensure_ai_defaults = ensure_ai_defaults
        self.ensure_picsum_defaults = ensure_picsum_defaults
        self.sanitize_ai_settings = sanitize_ai_settings
        self.ai_settings_match_defaults = ai_settings_match_defaults
        self.save_settings_debounced = save_settings_debounced
        self.safe_emit = safe_emit
        self._emit_ai_update_callback = emit_ai_update_callback
        self.update_stream_runtime_state = update_stream_runtime_state
        self.invalidate_media_cache = invalidate_media_cache
        self.relative_image_path = relative_image_path
        self.ensure_dir = ensure_dir
        self.playback_manager = playback_manager
        self.ai_output_root = ai_output_root
        self.ai_temp_root = ai_temp_root
        self.ai_default_persist = ai_default_persist
        self.ai_default_sampler = ai_default_sampler
        self.ai_default_width = ai_default_width
        self.ai_default_height = ai_default_height
        self.ai_default_steps = ai_default_steps
        self.ai_default_cfg = ai_default_cfg
        self.ai_default_samples = ai_default_samples
        self.ai_poll_interval = ai_poll_interval
        self.ai_timeout = ai_timeout
        self.stable_horde_post_processors = stable_horde_post_processors
        self.stable_horde_max_loras = stable_horde_max_loras
        self.ai_settings_key = ai_settings_key
        self.ai_state_key = ai_state_key
        self.ai_generate_mode = ai_generate_mode
        self.ai_random_mode = ai_random_mode
        self.ai_specific_mode = ai_specific_mode
        self.media_mode_ai = media_mode_ai
        self.ai_media_library = ai_media_library
        self.stable_horde_cancelled_cls = stable_horde_cancelled_cls
        self.stable_horde_error_cls = stable_horde_error_cls

    def cleanup_temp_outputs(self, stream_id: str) -> None:
        temp_dir = self.ai_temp_root / stream_id
        if temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
            except Exception as exc:
                self.logger.warning("Failed to remove temp outputs for %s: %s", stream_id, exc)

    def emit_ai_update(self, stream_id: str, state: Dict[str, Any], job: Optional[Dict[str, Any]] = None) -> None:
        payload: Dict[str, Any] = {"stream_id": stream_id, "state": state}
        if job is not None:
            payload["job"] = job
        if self.job_manager.should_emit(stream_id):
            self.safe_emit("ai_job_update", payload)

    def update_ai_state(self, stream_id: str, updates: Dict[str, Any], *, persist: bool = False) -> Dict[str, Any]:
        conf = self.settings.get(stream_id)
        if not conf:
            return {}
        self.ensure_ai_defaults(conf)
        self.ensure_picsum_defaults(conf)
        state = conf[self.ai_state_key]
        state.update(updates)
        if persist:
            self.save_settings_debounced()
        self.emit_ai_update(stream_id, state)
        return state

    def reconcile_stale_ai_state(self, stream_id: str, conf: Dict[str, Any]) -> bool:
        self.ensure_ai_defaults(conf)
        ai_state = conf.get(self.ai_state_key)
        if not isinstance(ai_state, dict):
            return False
        status = str(ai_state.get("status") or "").strip().lower()
        if status != "cancelling":
            return False
        with self.ai_jobs_lock:
            job = self.ai_jobs.get(stream_id)
        if job:
            return False
        ai_state.update(
            {
                "status": "cancelled",
                "message": "Cancelled",
                "error": None,
                "job_id": None,
                "queue_position": None,
                "wait_time": None,
                "last_updated": datetime.now(UTC).replace(tzinfo=None).isoformat() + "Z",
            }
        )
        self.save_settings_debounced()
        return True

    def record_job_progress(self, stream_id: str, stage: str, payload: Dict[str, Any]) -> None:
        manager_id: Optional[str] = None
        with self.ai_jobs_lock:
            job = self.ai_jobs.get(stream_id)
            if not job:
                return
            job = dict(job)
            controls = self.ai_job_controls.get(stream_id, {})
            manager_id = job.get("manager_id") or controls.get("manager_id")
            if manager_id:
                job["manager_id"] = manager_id
            job["stage"] = stage
            if stage == "accepted":
                job["job_id"] = payload.get("job_id")
                job["status"] = "accepted"
                job.setdefault("started", time.time())
            elif stage == "status":
                status_payload = payload.get("status") or {}
                job["status"] = "running"
                job["queue_position"] = status_payload.get("queue_position")
                job["wait_time"] = status_payload.get("wait_time")
            elif stage == "fault":
                job["status"] = "error"
                job["message"] = str(payload.get("message") or payload)
            elif stage == "timeout":
                job["status"] = "timeout"
                job["message"] = "Timed out waiting for Stable Horde"
            elif stage == "cancelled":
                job["status"] = "cancelled"
                job["message"] = str(payload.get("message") or "Cancelled by user")
            elif stage == "completed":
                job["status"] = "completed"
            self.ai_jobs[stream_id] = job

        state_ref: Optional[Dict[str, Any]] = None
        conf_ref = self.settings.get(stream_id)
        if isinstance(conf_ref, dict):
            candidate_state = conf_ref.get(self.ai_state_key)
            if isinstance(candidate_state, dict):
                state_ref = candidate_state
        if state_ref is not None:
            if job.get("status"):
                state_ref["status"] = job["status"]
            if stage == "status":
                state_ref["queue_position"] = job.get("queue_position")
                state_ref["wait_time"] = job.get("wait_time")
            elif stage in ("fault", "timeout", "cancelled", "completed"):
                state_ref["queue_position"] = job.get("queue_position")
                state_ref["wait_time"] = job.get("wait_time")
                if job.get("message"):
                    state_ref["message"] = job["message"]

        if manager_id:
            if stage == "accepted":
                self.job_manager.set_stable_id(manager_id, payload.get("job_id"))
                self.job_manager.update_status(manager_id, status="running")
            elif stage == "status":
                self.job_manager.touch(manager_id)
            elif stage == "fault":
                self.job_manager.update_status(manager_id, status="error", error=str(payload.get("message") or payload))
            elif stage == "timeout":
                self.job_manager.update_status(manager_id, status="timeout", error="Timed out waiting for Stable Horde")
            elif stage == "cancelled":
                self.job_manager.update_status(manager_id, status="cancelled", error=str(payload.get("message") or "Cancelled by user"))
            elif stage == "completed":
                self.job_manager.update_status(manager_id, status="completed", result=payload, error=None)
        current_state = self.settings.get(stream_id, {}).get(self.ai_state_key, {})
        self.emit_ai_update(stream_id, current_state, job=job)

    def run_generation(
        self,
        stream_id: str,
        options: Dict[str, Any],
        cancel_event: Optional[threading.Event] = None,
        manager_id: Optional[str] = None,
    ) -> None:
        prompt = str(options.get("prompt") or "").strip()
        if not prompt:
            message = "Prompt is required"
            self.job_manager.update_status(manager_id, status="error", error=message)
            self.update_ai_state(stream_id, {"status": "error", "message": message, "error": message}, persist=True)
            with self.ai_jobs_lock:
                self.ai_jobs.pop(stream_id, None)
                self.ai_job_controls.pop(stream_id, None)
            return

        persist = bool(options.get("save_output", self.ai_default_persist))
        if cancel_event and cancel_event.is_set():
            message = "Cancelled by user"
            self.job_manager.update_status(manager_id, status="cancelled", error=message)
            job_snapshot = None
            with self.ai_jobs_lock:
                current_job = self.ai_jobs.get(stream_id)
                if current_job:
                    job_snapshot = dict(current_job)
                    job_snapshot["status"] = "cancelled"
                    job_snapshot["message"] = message
                    self.ai_jobs[stream_id] = job_snapshot
            state = self.update_ai_state(
                stream_id,
                {"status": "cancelled", "message": message, "error": None, "persisted": persist},
                persist=True,
            )
            self.emit_ai_update(stream_id, state, job_snapshot)
            with self.ai_jobs_lock:
                self.ai_jobs.pop(stream_id, None)
                self.ai_job_controls.pop(stream_id, None)
            return

        target_root = self.ensure_dir((self.ai_output_root if persist else self.ai_temp_root) / stream_id)

        def _status_callback(stage: str, payload: Dict[str, Any]) -> None:
            self.record_job_progress(stream_id, stage, payload)

        models = [options["model"]] if options.get("model") else None
        sampler = options.get("sampler") or self.ai_default_sampler
        negative_prompt = options.get("negative_prompt") or None
        seed_input = options.get("seed")
        if seed_input is None:
            seed_payload = str(secrets.randbelow(2**32))
        else:
            seed_str = str(seed_input).strip()
            if not seed_str or seed_str.lower() in {"random", "rand", "auto"}:
                seed_payload = str(secrets.randbelow(2**32))
            else:
                seed_payload = seed_str

        timeout_raw = options.get("timeout")
        if timeout_raw in (None, "", "none"):
            timeout_value = self.ai_timeout
        else:
            try:
                timeout_value = max(0.0, float(timeout_raw))
            except (TypeError, ValueError):
                timeout_value = self.ai_timeout

        post_processing = [
            proc
            for proc in (options.get("post_processing") or [])
            if isinstance(proc, str) and proc in self.stable_horde_post_processors
        ]
        loras_payload: List[Dict[str, Any]] = []
        for entry in options.get("loras") or []:
            if not isinstance(entry, dict):
                continue
            name = entry.get("name")
            if not name:
                continue
            cleaned_entry: Dict[str, Any] = {"name": name}
            for attr in ("model", "clip"):
                val = entry.get(attr)
                if isinstance(val, (int, float)):
                    cleaned_entry[attr] = float(val)
            trigger = entry.get("inject_trigger")
            if isinstance(trigger, str) and trigger:
                cleaned_entry["inject_trigger"] = trigger
            if entry.get("is_version"):
                cleaned_entry["is_version"] = bool(entry.get("is_version"))
            loras_payload.append(cleaned_entry)
            if len(loras_payload) >= self.stable_horde_max_loras:
                break
        advanced_params: Dict[str, Any] = {}
        if loras_payload:
            advanced_params["loras"] = loras_payload
        for flag in ("hires_fix", "karras", "tiling", "transparent"):
            if options.get(flag):
                advanced_params[flag] = True
        clip_skip_value = options.get("clip_skip")
        if isinstance(clip_skip_value, (int, float)):
            advanced_params["clip_skip"] = int(clip_skip_value)
        for float_key in ("facefixer_strength",):
            val = options.get(float_key)
            if isinstance(val, (int, float)):
                advanced_params[float_key] = float(val)
        for float_key in ("denoising_strength", "hires_fix_denoising_strength"):
            val = options.get(float_key)
            if isinstance(val, (int, float)):
                advanced_params[float_key] = float(val)
        extras_payload: Dict[str, Any] = {}
        for flag in ("trusted_workers", "validated_backends", "slow_workers", "extra_slow_workers", "disable_batching", "allow_downgrade"):
            if flag in options:
                extras_payload[flag] = bool(options.get(flag))
        style_value = str(options.get("style") or "").strip()
        if style_value:
            extras_payload["style"] = style_value

        generation_kwargs = {
            "negative_prompt": negative_prompt,
            "models": models,
            "width": int(options.get("width", self.ai_default_width)),
            "height": int(options.get("height", self.ai_default_height)),
            "steps": int(options.get("steps", self.ai_default_steps)),
            "cfg_scale": float(options.get("cfg_scale", self.ai_default_cfg)),
            "sampler_name": sampler,
            "seed": seed_payload,
            "samples": int(options.get("samples", self.ai_default_samples)),
            "nsfw": bool(options.get("nsfw")),
            "censor_nsfw": bool(options.get("censor_nsfw")),
            "post_processing": post_processing or None,
            "params": advanced_params or None,
            "extras": extras_payload or None,
            "poll_interval": float(options.get("poll_interval", self.ai_poll_interval)),
            "timeout": timeout_value,
            "persist": persist,
            "output_dir": target_root if persist else None,
            "status_callback": _status_callback,
            "cancel_callback": (lambda: bool(cancel_event and cancel_event.is_set())),
        }

        try:
            result = None
            for attempt in range(1, 3):
                try:
                    result = self.stable_horde_client.generate_images(prompt, **generation_kwargs)
                    break
                except Exception as exc:
                    lost_track = "lost track of job" in str(exc).lower()
                    if lost_track and attempt == 1 and not (cancel_event and cancel_event.is_set()):
                        self.logger.warning(
                            "Stable Horde lost track of job for %s on attempt %d; retrying once with a fresh request",
                            stream_id,
                            attempt,
                        )
                        self.update_ai_state(
                            stream_id,
                            {"status": "queued", "message": "Stable Horde lost the first job; retrying once", "error": None, "persisted": persist},
                            persist=True,
                        )
                        continue
                    raise
        except self.stable_horde_cancelled_cls as exc:
            self.logger.info("Stable Horde job for %s cancelled: %s", stream_id, exc)
            message = "Cancelled by user"
            self.record_job_progress(stream_id, "cancelled", {"message": message})
            job_snapshot = None
            with self.ai_jobs_lock:
                current_job = self.ai_jobs.get(stream_id)
                if current_job:
                    job_snapshot = dict(current_job)
                    job_snapshot["status"] = "cancelled"
                    job_snapshot["message"] = message
                    self.ai_jobs[stream_id] = job_snapshot
            state = self.update_ai_state(
                stream_id,
                {"status": "cancelled", "message": message, "error": None, "persisted": persist},
                persist=True,
            )
            self.emit_ai_update(stream_id, state, job_snapshot)
            if not persist:
                self.cleanup_temp_outputs(stream_id)
            self.job_manager.update_status(manager_id, status="cancelled", error=str(exc))
            with self.ai_jobs_lock:
                self.ai_jobs.pop(stream_id, None)
                self.ai_job_controls.pop(stream_id, None)
            return
        except self.stable_horde_error_cls as exc:
            self.logger.warning("Stable Horde job for %s failed: %s", stream_id, exc)
            self.record_job_progress(stream_id, "fault", {"message": str(exc)})
            self.update_ai_state(
                stream_id,
                {"status": "error", "message": str(exc), "error": str(exc), "persisted": persist},
                persist=True,
            )
            self.job_manager.update_status(manager_id, status="error", error=str(exc))
            with self.ai_jobs_lock:
                self.ai_jobs.pop(stream_id, None)
                self.ai_job_controls.pop(stream_id, None)
            return
        except Exception as exc:
            self.logger.exception("Unexpected Stable Horde failure for %s: %s", stream_id, exc)
            self.record_job_progress(stream_id, "fault", {"message": str(exc)})
            self.update_ai_state(
                stream_id,
                {"status": "error", "message": "Generation failed", "error": str(exc), "persisted": persist},
                persist=True,
            )
            self.job_manager.update_status(manager_id, status="error", error=str(exc))
            with self.ai_jobs_lock:
                self.ai_jobs.pop(stream_id, None)
                self.ai_job_controls.pop(stream_id, None)
            return

        images: List[Dict[str, Any]] = []
        if persist:
            stored_paths = [(generation.path, generation) for generation in result.generations]
        else:
            job_dir = self.ensure_dir(target_root / result.job_id)
            stored_paths = []
            for generation in result.generations:
                final_path = job_dir / Path(generation.path).name
                try:
                    shutil.copy2(generation.path, final_path)
                except Exception as exc:
                    self.logger.warning("Failed to copy temp generation %s: %s", generation.path, exc)
                    continue
                stored_paths.append((final_path, generation))
            result.cleanup()

        for final_path, generation in stored_paths:
            rel_path = self.relative_image_path(final_path)
            images.append(
                {
                    "path": rel_path,
                    "seed": generation.seed,
                    "model": generation.model or options.get("model"),
                    "worker": generation.worker,
                    "url": generation.url,
                    "persisted": persist,
                }
            )

        updates = {
            "status": "completed",
            "job_id": result.job_id,
            "queue_position": result.queue_position,
            "wait_time": result.wait_time,
            "images": images,
            "persisted": persist,
            "message": None,
            "error": None,
            "last_updated": datetime.now(UTC).replace(tzinfo=None).isoformat() + "Z",
        }
        conf = self.settings.get(stream_id)
        if conf:
            self.ensure_ai_defaults(conf)
            self.ensure_picsum_defaults(conf)
            previous_media_mode = str(conf.get("media_mode") or "").strip().lower()
            previous_mode = str(conf.get("mode") or "").strip().lower()
            conf[self.ai_settings_key] = self.sanitize_ai_settings(options, conf[self.ai_settings_key])
            conf[self.ai_settings_key]["save_output"] = persist
            conf["_ai_customized"] = not self.ai_settings_match_defaults(conf[self.ai_settings_key])
            conf[self.ai_state_key].update(updates)
            if images:
                conf["selected_image"] = images[0]["path"]
                conf["selected_media_kind"] = "image"
                self.invalidate_media_cache(images[0]["path"], library=self.ai_media_library)
            if previous_media_mode == self.media_mode_ai and previous_mode in {self.ai_generate_mode, self.ai_random_mode, self.ai_specific_mode}:
                conf["media_mode"] = self.media_mode_ai
                conf["mode"] = previous_mode
            else:
                conf["mode"] = self.ai_generate_mode
                conf["media_mode"] = self.media_mode_ai
            self.update_stream_runtime_state(
                stream_id,
                path=conf.get("selected_image"),
                kind="image",
                media_mode=conf.get("media_mode"),
                source="ai_generation",
            )
            if self.playback_manager is not None:
                self.playback_manager.update_stream_config(stream_id, conf)
            self.save_settings_debounced()
            self.emit_ai_update(stream_id, conf[self.ai_state_key])
            if self.job_manager.should_emit(stream_id):
                self.safe_emit("refresh", {"stream_id": stream_id, "config": conf})

        completion_payload = {
            "job_id": result.job_id,
            "images": images,
            "queue_position": result.queue_position,
            "wait_time": result.wait_time,
            "persisted": persist,
        }
        self.record_job_progress(stream_id, "completed", completion_payload)
        with self.ai_jobs_lock:
            self.ai_jobs.pop(stream_id, None)
            self.ai_job_controls.pop(stream_id, None)
