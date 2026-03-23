from __future__ import annotations

import json
from pathlib import Path

from echomosaic_app.state.settings_store import SettingsStore


class _ScheduledCall:
    def __init__(self, delay: float, callback):
        self.delay = delay
        self.callback = callback
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True


def _scheduler(calls: list[_ScheduledCall]):
    def _schedule(delay: float, callback):
        call = _ScheduledCall(delay, callback)
        calls.append(call)
        return call

    return _schedule


def test_settings_store_loads_empty_dict_for_missing_file(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path / "settings.json")
    assert store.load() == {}


def test_settings_store_saves_atomically(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    store = SettingsStore(path)

    store.save({"stream1": {"label": "Demo"}})

    assert json.loads(path.read_text(encoding="utf-8")) == {"stream1": {"label": "Demo"}}
    assert list(tmp_path.glob("settings.json.*.tmp")) == []


def test_settings_store_debounced_save_replaces_pending_job_and_flushes_latest_data(tmp_path: Path) -> None:
    calls: list[_ScheduledCall] = []
    state = {"value": 1}
    store = SettingsStore(
        tmp_path / "settings.json",
        debounce_seconds=2.0,
        scheduler=_scheduler(calls),
    )

    store.save_debounced(lambda: dict(state))
    state["value"] = 2
    store.save_debounced(lambda: dict(state))

    assert len(calls) == 2
    assert calls[0].cancelled is True
    assert calls[1].cancelled is False

    store.flush_pending(lambda: dict(state))

    saved = json.loads((tmp_path / "settings.json").read_text(encoding="utf-8"))
    assert saved == {"value": 2}
    assert calls[1].cancelled is True
