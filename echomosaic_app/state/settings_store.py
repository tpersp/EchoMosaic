"""Canonical settings persistence for EchoMosaic."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Callable, Optional

import eventlet


class SettingsStore:
    """Atomic settings persistence with optional debounced saves."""

    def __init__(
        self,
        path: Path | str,
        *,
        debounce_seconds: float = 2.0,
        scheduler: Optional[Callable[[float, Callable[[], None]], Any]] = None,
    ) -> None:
        self.path = Path(path)
        self.debounce_seconds = float(debounce_seconds)
        self._scheduler = scheduler or eventlet.spawn_after
        self._write_lock = threading.Lock()
        self._pending_save: Any = None

    def load(self) -> dict[str, Any]:
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        return {}

    def save(self, data: dict[str, Any]) -> None:
        with self._write_lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=str(self.path.parent),
                prefix=f"{self.path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                json.dump(data, handle, indent=4)
                handle.flush()
                os.fsync(handle.fileno())
                temp_path = Path(handle.name)
            os.replace(temp_path, self.path)

    def save_debounced(self, data_provider: Callable[[], dict[str, Any]]) -> None:
        self._cancel_pending()
        self._pending_save = self._scheduler(self.debounce_seconds, lambda: self._run_pending(data_provider))

    def flush_pending(self, data_provider: Callable[[], dict[str, Any]]) -> None:
        if self._pending_save is None:
            return
        self._cancel_pending()
        self.save(data_provider())

    def _run_pending(self, data_provider: Callable[[], dict[str, Any]]) -> None:
        self._pending_save = None
        self.save(data_provider())

    def _cancel_pending(self) -> None:
        pending = self._pending_save
        self._pending_save = None
        if pending is not None and hasattr(pending, "cancel"):
            pending.cancel()
