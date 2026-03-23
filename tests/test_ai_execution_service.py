from __future__ import annotations

import threading
from pathlib import Path

from echomosaic_app.services.ai_execution import AIExecutionService


class _StableHordeCancelled(Exception):
    pass


class _StableHordeError(Exception):
    pass


class _Generation:
    def __init__(self, path: str) -> None:
        self.path = path
        self.seed = "123"
        self.model = "model-a"
        self.worker = "worker-a"
        self.url = "https://example.com/image"


class _Result:
    def __init__(self, path: str) -> None:
        self.generations = [_Generation(path)]
        self.job_id = "job-123"
        self.queue_position = 1
        self.wait_time = 2.5

    def cleanup(self) -> None:
        pass


class _StableHordeClient:
    def __init__(self, result_path: str) -> None:
        self.result_path = result_path

    def generate_images(self, prompt: str, **kwargs):
        return _Result(self.result_path)


class _JobManager:
    def should_emit(self, stream_id: str) -> bool:
        return False

    def update_status(self, *args, **kwargs) -> None:
        pass

    def set_stable_id(self, *args, **kwargs) -> None:
        pass

    def touch(self, *args, **kwargs) -> None:
        pass


def test_ai_execution_service_updates_state_on_completed_generation(tmp_path: Path) -> None:
    output_image = tmp_path / "generated.png"
    output_image.write_bytes(b"png")
    settings = {
        "stream1": {
            "_ai_settings": {"prompt": "hello", "save_output": True},
            "_ai_state": {},
            "media_mode": "ai",
            "mode": "ai_generate",
        }
    }
    ai_jobs = {"stream1": {"status": "queued", "manager_id": "mgr-1"}}
    ai_job_controls = {"stream1": {}}
    updates = []

    service = AIExecutionService(
        settings=settings,
        ai_jobs=ai_jobs,
        ai_job_controls=ai_job_controls,
        ai_jobs_lock=threading.Lock(),
        job_manager=_JobManager(),
        stable_horde_client=_StableHordeClient(str(output_image)),
        logger=type("Logger", (), {"warning": lambda *args, **kwargs: None, "info": lambda *args, **kwargs: None, "exception": lambda *args, **kwargs: None})(),
        ensure_ai_defaults=lambda conf: conf.setdefault("_ai_settings", {"prompt": "", "save_output": True}) or conf.setdefault("_ai_state", {}),
        ensure_picsum_defaults=lambda conf: None,
        sanitize_ai_settings=lambda payload, current: {**current, **payload},
        ai_settings_match_defaults=lambda payload: False,
        save_settings_debounced=lambda: updates.append("saved"),
        safe_emit=lambda *args, **kwargs: None,
        emit_ai_update_callback=lambda *args, **kwargs: None,
        update_stream_runtime_state=lambda *args, **kwargs: updates.append(("runtime", args, kwargs)),
        invalidate_media_cache=lambda *args, **kwargs: updates.append(("invalidate", args, kwargs)),
        relative_image_path=lambda path: f"media:/{Path(path).name}",
        ensure_dir=lambda path: Path(path),
        playback_manager=None,
        ai_output_root=tmp_path,
        ai_temp_root=tmp_path / "_temp",
        ai_default_persist=True,
        ai_default_sampler="sampler",
        ai_default_width=512,
        ai_default_height=512,
        ai_default_steps=30,
        ai_default_cfg=7.5,
        ai_default_samples=1,
        ai_poll_interval=1.0,
        ai_timeout=0.0,
        stable_horde_post_processors=[],
        stable_horde_max_loras=4,
        ai_settings_key="_ai_settings",
        ai_state_key="_ai_state",
        ai_generate_mode="ai_generate",
        ai_random_mode="ai_random",
        ai_specific_mode="ai_specific",
        media_mode_ai="ai",
        ai_media_library="ai",
        stable_horde_cancelled_cls=_StableHordeCancelled,
        stable_horde_error_cls=_StableHordeError,
    )

    service.run_generation("stream1", {"prompt": "hello", "save_output": True}, None, "mgr-1")

    assert settings["stream1"]["_ai_state"]["status"] == "completed"
    assert settings["stream1"]["selected_image"] == "media:/generated.png"
    assert "stream1" not in ai_jobs
