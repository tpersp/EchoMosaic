"""AI orchestration service layer."""

from __future__ import annotations

import threading
import time
from datetime import datetime
from typing import Any, Callable, Dict, Optional

from flask import has_request_context, request


class AIOrchestrationService:
    def __init__(
        self,
        *,
        settings,
        stable_horde_client,
        ai_model_cache: Dict[str, Any],
        ai_jobs,
        ai_job_controls,
        ai_jobs_lock,
        job_manager,
        ensure_ai_defaults: Callable[[Dict[str, Any]], None],
        ensure_picsum_defaults: Callable[[Dict[str, Any]], None],
        sanitize_ai_settings: Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]],
        ai_settings_match_defaults: Callable[[Dict[str, Any]], bool],
        default_ai_state: Callable[[], Dict[str, Any]],
        cleanup_temp_outputs: Callable[[str], None],
        save_settings_debounced: Callable[[], None],
        emit_ai_update: Callable[..., None],
        update_ai_state: Callable[..., Dict[str, Any]],
        reconcile_stale_ai_state: Callable[[str, Dict[str, Any]], bool],
        safe_emit: Callable[..., None],
        get_global_tags: Callable[[], list[str]],
        run_ai_generation: Callable[..., None],
        logger,
        stable_horde_error_cls,
        format_auto_trigger: Callable[[datetime], Optional[str]],
        ai_settings_key: str,
        ai_state_key: str,
        ai_default_persist: bool,
        ai_generate_mode: str,
        media_mode_ai: str,
        auto_generation_error_cls,
        auto_generation_unavailable_cls,
        auto_generation_busy_cls,
        auto_generation_prompt_missing_cls,
    ) -> None:
        self.settings = settings
        self.stable_horde_client = stable_horde_client
        self.ai_model_cache = ai_model_cache
        self.ai_jobs = ai_jobs
        self.ai_job_controls = ai_job_controls
        self.ai_jobs_lock = ai_jobs_lock
        self.job_manager = job_manager
        self.ensure_ai_defaults = ensure_ai_defaults
        self.ensure_picsum_defaults = ensure_picsum_defaults
        self.sanitize_ai_settings = sanitize_ai_settings
        self.ai_settings_match_defaults = ai_settings_match_defaults
        self.default_ai_state = default_ai_state
        self.cleanup_temp_outputs = cleanup_temp_outputs
        self.save_settings_debounced = save_settings_debounced
        self.emit_ai_update = emit_ai_update
        self.update_ai_state = update_ai_state
        self.reconcile_stale_ai_state = reconcile_stale_ai_state
        self.safe_emit = safe_emit
        self.get_global_tags = get_global_tags
        self.run_ai_generation = run_ai_generation
        self.logger = logger
        self.stable_horde_error_cls = stable_horde_error_cls
        self.format_auto_trigger = format_auto_trigger
        self.ai_settings_key = ai_settings_key
        self.ai_state_key = ai_state_key
        self.ai_default_persist = ai_default_persist
        self.ai_generate_mode = ai_generate_mode
        self.media_mode_ai = media_mode_ai
        self.auto_generation_error_cls = auto_generation_error_cls
        self.auto_generation_unavailable_cls = auto_generation_unavailable_cls
        self.auto_generation_busy_cls = auto_generation_busy_cls
        self.auto_generation_prompt_missing_cls = auto_generation_prompt_missing_cls

    def list_models_payload(self) -> Dict[str, Any]:
        if self.stable_horde_client is None:
            raise self.auto_generation_unavailable_cls("Stable Horde client is not configured")
        now = time.time()
        cache_ttl = 300
        if (now - self.ai_model_cache["timestamp"]) > cache_ttl or not self.ai_model_cache["data"]:
            try:
                models = self.stable_horde_client.list_models()
            except self.stable_horde_error_cls as exc:
                self.logger.warning("Model fetch failed: %s", exc)
                raise
            self.ai_model_cache["data"] = [
                {
                    "name": model.get("name"),
                    "performance": model.get("performance"),
                    "queued": model.get("queued"),
                    "jobs": model.get("jobs"),
                    "type": model.get("type"),
                }
                for model in models
                if isinstance(model, dict) and model.get("type") == "image"
            ]
            self.ai_model_cache["timestamp"] = now
        return {"models": self.ai_model_cache["data"]}

    def status_payload(self, stream_id: str) -> Dict[str, Any]:
        conf = self.settings.get(stream_id)
        if not conf:
            raise self.auto_generation_error_cls(f"No stream '{stream_id}' found")
        self.ensure_ai_defaults(conf)
        with self.ai_jobs_lock:
            job_snapshot = dict(self.ai_jobs.get(stream_id, {}))
        return {
            "state": conf[self.ai_state_key],
            "settings": conf[self.ai_settings_key],
            "job": job_snapshot,
        }

    def latest_job_payload(self, stream_id: str) -> Dict[str, Any]:
        conf = self.settings.get(stream_id)
        state_payload: Optional[Dict[str, Any]] = None
        if conf:
            self.ensure_ai_defaults(conf)
            self.ensure_picsum_defaults(conf)
            state_payload = conf.get(self.ai_state_key)
        managed = self.job_manager.get_latest(stream_id)
        job_payload = managed.to_dict() if managed else None
        active_job: Optional[Dict[str, Any]] = None
        with self.ai_jobs_lock:
            current = self.ai_jobs.get(stream_id)
            if current:
                active_job = dict(current)
        return {
            "job": job_payload,
            "active_job": active_job,
            "state": state_payload,
        }

    def queue_generation(self, stream_id: str, ai_settings: Dict[str, Any], *, trigger_source: str = "manual") -> Dict[str, Any]:
        if self.stable_horde_client is None:
            raise self.auto_generation_unavailable_cls("Stable Horde client is not configured")

        conf = self.settings.get(stream_id)
        if not conf:
            raise self.auto_generation_error_cls(f"No stream '{stream_id}' found")

        self.ensure_ai_defaults(conf)
        self.ensure_picsum_defaults(conf)
        sanitized = self.sanitize_ai_settings(ai_settings, conf[self.ai_settings_key])
        prompt = str(sanitized.get("prompt") or "").strip()
        if not prompt:
            raise self.auto_generation_prompt_missing_cls("Prompt is required")

        conf[self.ai_settings_key] = sanitized
        conf["_ai_customized"] = not self.ai_settings_match_defaults(conf[self.ai_settings_key])
        persist = bool(sanitized.get("save_output", self.ai_default_persist))

        previous_state = conf.get(self.ai_state_key) or {}
        previous_images = list(previous_state.get("images") or [])
        previous_selected = conf.get("selected_image")

        cancel_event = threading.Event()
        socket_sid: Optional[str] = None
        if has_request_context():
            header_sid = request.headers.get("X-Socket-ID")
            if header_sid:
                socket_sid = header_sid.strip() or None
        manager_id = self.job_manager.create_job(stream_id, trigger=trigger_source, sid=socket_sid)
        with self.ai_jobs_lock:
            if stream_id in self.ai_jobs:
                raise self.auto_generation_busy_cls("Generation already in progress")
            self.ai_jobs[stream_id] = {
                "status": "queued",
                "job_id": None,
                "started": time.time(),
                "persisted": persist,
                "cancel_requested": False,
                "trigger": trigger_source,
                "manager_id": manager_id,
                "queue_position": None,
                "wait_time": None,
            }
            self.ai_job_controls[stream_id] = {
                "cancel_event": cancel_event,
                "socket_sid": socket_sid,
                "manager_id": manager_id,
            }
        if not persist:
            self.cleanup_temp_outputs(stream_id)

        conf["mode"] = self.ai_generate_mode
        conf["media_mode"] = self.media_mode_ai
        if previous_selected:
            conf["selected_image"] = previous_selected

        conf[self.ai_state_key] = self.default_ai_state()
        queued_state = conf[self.ai_state_key]
        queued_state.update(
            {
                "status": "queued",
                "message": "Awaiting workers",
                "persisted": persist,
                "images": previous_images,
                "error": None,
                "last_trigger_source": trigger_source,
            }
        )
        if trigger_source != "manual":
            queued_state["last_auto_trigger"] = self.format_auto_trigger(datetime.now())
        queued_state["last_auto_error"] = None

        self.save_settings_debounced()
        self.emit_ai_update(stream_id, queued_state, job=self.ai_jobs[stream_id])
        self.job_manager.update_status(manager_id, status="queued")
        if self.job_manager.should_emit(stream_id):
            self.safe_emit("refresh", {"stream_id": stream_id, "config": conf, "tags": self.get_global_tags()})

        job_options = dict(sanitized)
        job_options["prompt"] = prompt
        worker = threading.Thread(
            target=self.run_ai_generation,
            args=(stream_id, job_options, cancel_event, manager_id),
            daemon=True,
        )
        with self.ai_jobs_lock:
            controls = self.ai_job_controls.get(stream_id)
            if controls is not None:
                controls["thread"] = worker
                controls["cancel_event"] = cancel_event
        worker.start()
        return {"status": "queued", "state": conf[self.ai_state_key], "job": self.ai_jobs[stream_id]}

    def cancel_generation(self, stream_id: str) -> Dict[str, Any]:
        if self.stable_horde_client is None:
            raise self.auto_generation_unavailable_cls("Stable Horde client is not configured")
        conf = self.settings.get(stream_id)
        if not conf:
            raise self.auto_generation_error_cls(f"No stream '{stream_id}' found")
        self.ensure_ai_defaults(conf)
        manager_id: Optional[str] = None
        with self.ai_jobs_lock:
            job = self.ai_jobs.get(stream_id)
            controls = self.ai_job_controls.get(stream_id)
        if not job:
            if self.reconcile_stale_ai_state(stream_id, conf):
                state = conf.get(self.ai_state_key) if isinstance(conf.get(self.ai_state_key), dict) else self.default_ai_state()
                self.emit_ai_update(stream_id, state, job=None)
                return {"status": "cancelled", "message": "Cancelled stale AI state"}
            raise self.auto_generation_error_cls("No active AI generation to cancel")
        with self.ai_jobs_lock:
            current_job = self.ai_jobs.get(stream_id)
            controls = self.ai_job_controls.get(stream_id)
            if not current_job:
                if self.reconcile_stale_ai_state(stream_id, conf):
                    state = conf.get(self.ai_state_key) if isinstance(conf.get(self.ai_state_key), dict) else self.default_ai_state()
                    self.emit_ai_update(stream_id, state, job=None)
                    return {"status": "cancelled", "message": "Cancelled stale AI state"}
                raise self.auto_generation_error_cls("No active AI generation to cancel")
            job = dict(current_job)
            status = (job.get("status") or "").lower()
            if status in {"completed", "error", "timeout", "cancelled"}:
                raise self.auto_generation_busy_cls("Job already finished")
            job["cancel_requested"] = True
            job["status"] = "cancelling"
            job["message"] = "Cancellation requested"
            self.ai_jobs[stream_id] = job
            cancel_event = controls.get("cancel_event") if controls else None
            if controls:
                manager_id = controls.get("manager_id")
        if cancel_event:
            cancel_event.set()
        state = self.update_ai_state(
            stream_id,
            {
                "status": "cancelling",
                "message": "Cancellation requested",
                "error": None,
                "persisted": bool(job.get("persisted")),
            },
            persist=True,
        )
        self.emit_ai_update(stream_id, state, job)
        if manager_id:
            self.job_manager.update_status(manager_id, status="cancelling")
        warning = None
        job_id = job.get("job_id")
        if job_id:
            try:
                self.stable_horde_client.cancel_job(job_id)
            except self.stable_horde_error_cls as exc:
                self.logger.warning("Stable Horde cancel for %s failed: %s", stream_id, exc)
                warning = str(exc)
        response = {"status": "cancelling"}
        if warning:
            response["warning"] = warning
        return response
