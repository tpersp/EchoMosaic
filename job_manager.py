"""Background job manager for Stable Horde generations.

This helper keeps track of generation jobs independently of any connected
clients so the backend can continue polling Stable Horde even when the UI
disconnects.  The data maintained here is intentionally lightweight: jobs
are indexed by a generated identifier, optionally store the Stable Horde job
id, record the last known status/result, and remember which Socket.IO
clients are actively listening for progress updates.
"""

from __future__ import annotations

import atexit
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set


def _to_iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


@dataclass
class ManagedJob:
    manager_id: str
    stream_id: str
    trigger: str
    created_at: float
    status: str = "queued"
    stable_id: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    listeners: Set[str] = field(default_factory=set)
    last_update: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.manager_id,
            "stream_id": self.stream_id,
            "trigger": self.trigger,
            "status": self.status,
            "stable_id": self.stable_id,
            "result": self.result,
            "error": self.error,
            "listeners": list(self.listeners),
            "created_at": _to_iso(self.created_at),
            "last_update": _to_iso(self.last_update),
        }


class JobManager:
    """Track Stable Horde jobs independently of UI connections."""

    def __init__(self, *, cleanup_interval: int = 900, expire_after: int = 3600) -> None:
        self._jobs: Dict[str, ManagedJob] = {}
        self._stream_latest: Dict[str, str] = {}
        self._sid_index: Dict[str, Set[str]] = {}
        self._cleanup_interval = max(60, int(cleanup_interval))
        self._expire_after = max(300, int(expire_after))
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._cleanup_loop, name="StableHordeJobCleanup", daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------ Lifecycle
    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    # ------------------------------------------------------------------ Job creation / updates
    def create_job(self, stream_id: str, *, trigger: str, sid: Optional[str] = None) -> str:
        manager_id = str(uuid.uuid4())
        job = ManagedJob(manager_id=manager_id, stream_id=stream_id, trigger=trigger, created_at=time.time())
        if sid:
            job.listeners.add(sid)
            self._sid_index.setdefault(sid, set()).add(manager_id)
        with self._lock:
            self._jobs[manager_id] = job
            self._stream_latest[stream_id] = manager_id
        return manager_id

    def set_stable_id(self, manager_id: Optional[str], stable_id: Optional[str]) -> None:
        if not manager_id:
            return
        with self._lock:
            job = self._jobs.get(manager_id)
            if not job:
                return
            job.stable_id = stable_id
            job.last_update = time.time()

    def update_status(
        self,
        manager_id: Optional[str],
        *,
        status: Optional[str] = None,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        if not manager_id:
            return
        with self._lock:
            job = self._jobs.get(manager_id)
            if not job:
                return
            if status:
                job.status = status
            if result is not None:
                job.result = result
            if error is not None:
                job.error = error
            job.last_update = time.time()

    def touch(self, manager_id: Optional[str]) -> None:
        if not manager_id:
            return
        with self._lock:
            job = self._jobs.get(manager_id)
            if job:
                job.last_update = time.time()

    # ------------------------------------------------------------------ Listener tracking
    def attach_listener(self, stream_id: str, sid: str) -> Optional[str]:
        with self._lock:
            manager_id = self._stream_latest.get(stream_id)
            if not manager_id:
                return None
            job = self._jobs.get(manager_id)
            if not job:
                return None
            job.listeners.add(sid)
            self._sid_index.setdefault(sid, set()).add(manager_id)
            job.last_update = time.time()
            return manager_id

    def detach_listener(self, sid: str, stream_id: Optional[str] = None) -> List[ManagedJob]:
        detached: List[ManagedJob] = []
        with self._lock:
            if stream_id:
                manager_id = self._stream_latest.get(stream_id)
                if not manager_id:
                    return detached
                job = self._jobs.get(manager_id)
                if not job:
                    return detached
                job.listeners.discard(sid)
                sid_jobs = self._sid_index.get(sid)
                if sid_jobs:
                    sid_jobs.discard(manager_id)
                    if not sid_jobs:
                        self._sid_index.pop(sid, None)
                detached.append(job)
                return detached

            job_ids = self._sid_index.pop(sid, set())
            for manager_id in job_ids:
                job = self._jobs.get(manager_id)
                if job:
                    job.listeners.discard(sid)
                    detached.append(job)
        return detached

    def should_emit(self, stream_id: str) -> bool:
        with self._lock:
            manager_id = self._stream_latest.get(stream_id)
            if not manager_id:
                return True
            job = self._jobs.get(manager_id)
            if not job:
                return True
            # Only pause live progress for in-flight jobs when nobody is listening.
            if job.status in {"queued", "running"}:
                return bool(job.listeners)
            return True

    # ------------------------------------------------------------------ Queries
    def get_latest(self, stream_id: str) -> Optional[ManagedJob]:
        with self._lock:
            manager_id = self._stream_latest.get(stream_id)
            if not manager_id:
                return None
            return self._jobs.get(manager_id)

    # ------------------------------------------------------------------ Cleanup loop
    def _cleanup_loop(self) -> None:
        while not self._stop.wait(self._cleanup_interval):
            self._cleanup_once()

    def _cleanup_once(self) -> None:
        now = time.time()
        expire_before = now - self._expire_after
        with self._lock:
            for manager_id, job in list(self._jobs.items()):
                if job.last_update >= expire_before:
                    continue
                # Only expire jobs that have finished (successfully or otherwise)
                if job.status in {"queued", "running"}:
                    continue
                self._jobs.pop(manager_id, None)
                for sid in list(job.listeners):
                    sid_jobs = self._sid_index.get(sid)
                    if sid_jobs:
                        sid_jobs.discard(manager_id)
                        if not sid_jobs:
                            self._sid_index.pop(sid, None)
                latest_id = self._stream_latest.get(job.stream_id)
                if latest_id == manager_id:
                    self._stream_latest.pop(job.stream_id, None)


job_manager = JobManager()


def _shutdown_job_manager() -> None:
    job_manager.stop()


atexit.register(_shutdown_job_manager)
