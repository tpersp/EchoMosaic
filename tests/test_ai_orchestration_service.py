from __future__ import annotations

import threading

from echomosaic_app.services.ai_orchestration import AIOrchestrationService


class _AutoGenerationError(Exception):
    pass


class _AutoGenerationUnavailable(_AutoGenerationError):
    pass


class _AutoGenerationBusy(_AutoGenerationError):
    pass


class _AutoGenerationPromptMissing(_AutoGenerationError):
    pass


class _StableHordeError(Exception):
    pass


class _JobManager:
    def __init__(self) -> None:
        self.created = []

    def create_job(self, stream_id: str, trigger: str, sid=None) -> str:
        self.created.append((stream_id, trigger, sid))
        return "job-1"

    def update_status(self, *args, **kwargs) -> None:
        pass

    def should_emit(self, stream_id: str) -> bool:
        return False

    def get_latest(self, stream_id: str):
        return None


def _build_service(settings):
    ai_jobs = {}
    ai_job_controls = {}
    lock = threading.Lock()
    recorded = {"saved": 0, "emits": 0}

    def sanitize_ai_settings(payload, current):
        merged = dict(current)
        merged.update(payload)
        return merged

    service = AIOrchestrationService(
        settings=settings,
        stable_horde_client=object(),
        ai_model_cache={"timestamp": 0, "data": []},
        ai_jobs=ai_jobs,
        ai_job_controls=ai_job_controls,
        ai_jobs_lock=lock,
        job_manager=_JobManager(),
        ensure_ai_defaults=lambda conf: conf.setdefault("_ai_settings", {"prompt": "", "save_output": True}),
        ensure_picsum_defaults=lambda conf: None,
        sanitize_ai_settings=sanitize_ai_settings,
        ai_settings_match_defaults=lambda candidate: False,
        default_ai_state=lambda: {"images": []},
        cleanup_temp_outputs=lambda stream_id: None,
        save_settings_debounced=lambda: recorded.__setitem__("saved", recorded["saved"] + 1),
        emit_ai_update=lambda *args, **kwargs: recorded.__setitem__("emits", recorded["emits"] + 1),
        update_ai_state=lambda stream_id, updates, persist=False: updates,
        reconcile_stale_ai_state=lambda stream_id, conf: False,
        safe_emit=lambda *args, **kwargs: None,
        get_global_tags=lambda: [],
        run_ai_generation=lambda *args, **kwargs: None,
        logger=type("L", (), {"warning": lambda *args, **kwargs: None})(),
        stable_horde_error_cls=_StableHordeError,
        format_auto_trigger=lambda moment: "formatted-time",
        ai_settings_key="_ai_settings",
        ai_state_key="_ai_state",
        ai_default_persist=True,
        ai_generate_mode="ai_generate",
        media_mode_ai="ai",
        auto_generation_error_cls=_AutoGenerationError,
        auto_generation_unavailable_cls=_AutoGenerationUnavailable,
        auto_generation_busy_cls=_AutoGenerationBusy,
        auto_generation_prompt_missing_cls=_AutoGenerationPromptMissing,
    )
    return service, ai_jobs, ai_job_controls, recorded


def test_ai_orchestration_service_queues_generation() -> None:
    settings = {
        "stream1": {
            "_ai_settings": {"prompt": "old prompt", "save_output": True},
            "_ai_state": {"images": []},
            "selected_image": None,
        }
    }
    service, ai_jobs, ai_job_controls, recorded = _build_service(settings)

    result = service.queue_generation("stream1", {"prompt": "new prompt"}, trigger_source="auto")

    assert result["status"] == "queued"
    assert "stream1" in ai_jobs
    assert "stream1" in ai_job_controls
    assert settings["stream1"]["_ai_state"]["last_auto_trigger"] == "formatted-time"
    assert recorded["saved"] == 1
    assert recorded["emits"] == 1


def test_ai_orchestration_service_rejects_missing_prompt() -> None:
    settings = {"stream1": {"_ai_settings": {"prompt": "", "save_output": True}, "_ai_state": {"images": []}}}
    service, _, _, _ = _build_service(settings)

    try:
        service.queue_generation("stream1", {"prompt": ""})
    except _AutoGenerationPromptMissing as exc:
        assert "Prompt is required" in str(exc)
    else:
        raise AssertionError("expected _AutoGenerationPromptMissing")


def test_ai_orchestration_service_returns_status_payload() -> None:
    settings = {"stream1": {"_ai_settings": {"prompt": "hi"}, "_ai_state": {"status": "queued"}}}
    service, ai_jobs, _, _ = _build_service(settings)
    ai_jobs["stream1"] = {"status": "queued"}

    payload = service.status_payload("stream1")

    assert payload["state"]["status"] == "queued"
    assert payload["job"]["status"] == "queued"
