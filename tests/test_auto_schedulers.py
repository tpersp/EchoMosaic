from __future__ import annotations

from echomosaic_app.services.auto_schedulers import build_auto_schedulers


class _TimerStub:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def update_next(self, value) -> None:
        self.next_value = value

    def is_enabled(self) -> bool:
        return False

    def compute_next(self, reference=None):
        return None

    def offset_minutes(self) -> float:
        return 0.0

    def mark_trigger(self, when=None) -> None:
        self.marked = when


def test_build_auto_schedulers_smoke() -> None:
    settings = {"_notes": "ignore me"}
    auto_scheduler, picsum_scheduler = build_auto_schedulers(
        settings=settings,
        timer_manager_cls=_TimerStub,
        canonical_timer_mode=lambda conf: None,
        timer_snap_enabled=lambda: False,
        ensure_ai_defaults=lambda conf: conf.setdefault("ai_settings", {"prompt": ""}) or None,
        ensure_picsum_defaults=lambda conf: conf.setdefault("picsum", {}) or None,
        ensure_timer_defaults=lambda conf: conf.setdefault("timer", {}) or None,
        queue_ai_generation=lambda *args, **kwargs: {},
        emit_ai_update=lambda *args, **kwargs: None,
        refresh_picsum_stream=lambda stream_id: None,
        broadcast_picsum_update=lambda *args, **kwargs: None,
        save_settings_debounced=lambda: None,
        format_timer_label=lambda moment: None,
        normalize_timer_label=lambda value: value,
        log_timer_schedule=lambda *args, **kwargs: None,
        ai_state_key="ai_state",
        ai_settings_key="ai_settings",
        picsum_settings_key="picsum",
        ai_modes={"ai_generate"},
        picsum_mode="picsum",
        auto_generation_busy_cls=RuntimeError,
        auto_generation_prompt_missing_cls=ValueError,
        auto_generation_unavailable_cls=LookupError,
        auto_generation_error_cls=Exception,
    )

    auto_scheduler.reschedule_all()
    picsum_scheduler.reschedule_all()
    auto_scheduler.stop()
    picsum_scheduler.stop()
