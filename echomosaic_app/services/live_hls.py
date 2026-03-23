"""Live HLS detection and cache orchestration."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional


@dataclass
class HLSCacheEntry:
    url: Optional[str]
    extracted_at: float
    error: Optional[str] = None


class LiveHLSService:
    """Own live-stream HLS detection, caching, and invalidation."""

    def __init__(
        self,
        *,
        live_hls_async: bool,
        hls_ttl_secs: float,
        hls_error_retry_secs: float,
        hls_metrics,
        hls_lock,
        hls_log_prefix: str,
        hls_executor,
        hls_cache,
        hls_jobs,
        youtube_dl_cls,
        logger,
        app_context_factory: Callable[[], Any],
        safe_emit: Callable[..., None],
        live_hls_ready_event: str = "live_hls_ready",
    ) -> None:
        self.live_hls_async = bool(live_hls_async)
        self.hls_ttl_secs = float(hls_ttl_secs)
        self.hls_error_retry_secs = float(hls_error_retry_secs)
        self.hls_metrics = hls_metrics
        self.hls_lock = hls_lock
        self.hls_log_prefix = hls_log_prefix
        self.hls_executor = hls_executor
        self.hls_cache = hls_cache
        self.hls_jobs = hls_jobs
        self.youtube_dl_cls = youtube_dl_cls
        self.logger = logger
        self.app_context_factory = app_context_factory
        self.safe_emit = safe_emit
        self.live_hls_ready_event = live_hls_ready_event

    def hls_url_fingerprint(self, original_url: str) -> str:
        if not original_url:
            return "none"
        try:
            return hashlib.sha1(original_url.encode("utf-8")).hexdigest()[:10]
        except Exception:
            return "error"

    def log_hls_event(self, event: str, stream_id: str, original_url: str, **extra: Any) -> None:
        if not self.live_hls_async:
            return
        details = " ".join(f"{k}={v}" for k, v in sorted(extra.items())) if extra else ""
        self.logger.info(
            "%s.%s stream=%s url=%s%s",
            self.hls_log_prefix,
            event,
            stream_id or "-",
            self.hls_url_fingerprint(original_url),
            f" {details}" if details else "",
        )

    def record_hls_metric(self, name: str, delta: int = 1) -> None:
        if not self.live_hls_async:
            return
        with self.hls_lock:
            self.hls_metrics[name] = self.hls_metrics.get(name, 0) + delta

    def live_hls_cache_key(self, stream_id: Optional[str], original_url: str) -> str:
        sid = (stream_id or "").strip() or "unknown"
        return f"live:{sid}:{original_url}"

    def is_manifest_url(self, candidate: Optional[str]) -> bool:
        if not isinstance(candidate, str):
            return False
        lower = candidate.lower()
        return any(marker in lower for marker in (".m3u8", ".mpd", "manifest.mpd", "format=m3u8"))

    def extract_hls_candidate(self, info: Dict[str, Any]) -> Optional[str]:
        if not isinstance(info, dict):
            return None

        for key in ("url", "manifest_url", "hls_manifest_url"):
            candidate = info.get(key)
            if isinstance(candidate, str) and self.is_manifest_url(candidate):
                return candidate

        for formats_key in ("formats", "requested_formats"):
            formats = info.get(formats_key)
            if not isinstance(formats, list):
                continue
            for fmt in formats:
                if not isinstance(fmt, dict):
                    continue
                candidate = fmt.get("url") or fmt.get("manifest_url")
                if not isinstance(candidate, str):
                    continue
                protocol = str(fmt.get("protocol") or "").lower()
                ext = str(fmt.get("ext") or "").lower()
                if self.is_manifest_url(candidate) or "m3u8" in protocol or "dash" in protocol or ext in {"m3u8", "mpd"}:
                    return candidate
            for fmt in formats:
                if not isinstance(fmt, dict):
                    continue
                manifest_url = fmt.get("manifest_url")
                if isinstance(manifest_url, str) and self.is_manifest_url(manifest_url):
                    return manifest_url

        entries = info.get("entries")
        if isinstance(entries, list):
            for entry in entries:
                candidate = self.extract_hls_candidate(entry)
                if candidate:
                    return candidate

        return None

    def detect_hls_stream_url(self, original_url: str) -> Optional[str]:
        if not original_url:
            return None
        if self.youtube_dl_cls is None:
            raise RuntimeError("yt_dlp module is not available")
        ydl_opts = {
            "quiet": True,
            "nocheckcertificate": True,
            "skip_download": True,
            "noplaylist": True,
            "extract_flat": False,
        }
        with self.youtube_dl_cls(ydl_opts) as ydl:
            info = ydl.extract_info(original_url, download=False)
        if not info:
            return None
        return self.extract_hls_candidate(info)

    def get_hls_cache_entry(self, key: str) -> Optional[HLSCacheEntry]:
        with self.hls_lock:
            return self.hls_cache.get(key)

    def cancel_hls_job(self, key: str) -> bool:
        with self.hls_lock:
            future = self.hls_jobs.get(key)
            if not future:
                return False
            cancelled = future.cancel()
            if cancelled:
                self.hls_jobs.pop(key, None)
            return cancelled

    def schedule_hls_detection(self, stream_id: str, original_url: str) -> None:
        if not self.live_hls_async or not original_url or self.hls_executor is None:
            return
        key = self.live_hls_cache_key(stream_id, original_url)
        with self.hls_lock:
            future = self.hls_jobs.get(key)
            if future and not future.done():
                return
            future = self.hls_executor.submit(self._run_hls_detection_job, key, stream_id, original_url)
            self.hls_jobs[key] = future
            in_flight = len(self.hls_jobs)
        self.record_hls_metric("jobs_started")
        self.log_hls_event("job_start", stream_id, original_url, inflight=in_flight)

    def _run_hls_detection_job(self, key: str, stream_id: str, original_url: str) -> None:
        started_at = time.time()
        self.log_hls_event("job_run", stream_id, original_url)
        try:
            url = self.detect_hls_stream_url(original_url)
            entry = HLSCacheEntry(url=url, extracted_at=time.time(), error=None)
            success = bool(url)
            error_text = None
        except Exception as exc:
            entry = HLSCacheEntry(url=None, extracted_at=time.time(), error=str(exc))
            success = False
            error_text = entry.error
            self.record_hls_metric("errors")
        finally:
            self.record_hls_metric("jobs_completed")
        with self.hls_lock:
            self.hls_cache[key] = entry
            self.hls_jobs.pop(key, None)
            in_flight = len(self.hls_jobs)
        duration_ms = int((time.time() - started_at) * 1000)
        self.log_hls_event(
            "job_done",
            stream_id,
            original_url,
            success=success,
            inflight=in_flight,
            duration_ms=duration_ms,
            error=error_text or "none",
        )
        if entry.url:
            with self.app_context_factory():
                self.safe_emit(
                    self.live_hls_ready_event,
                    {"stream_id": stream_id, "cache_key": key, "hls_url": entry.url},
                )

    def resolve_hls_url(self, stream_id: str, stream_url: str) -> Optional[str]:
        hls_url = None
        if self.live_hls_async and self.hls_executor is not None:
            now = time.time()
            cache_key = self.live_hls_cache_key(stream_id, stream_url)
            entry = self.get_hls_cache_entry(cache_key)
            schedule_needed = False

            if entry is None:
                self.record_hls_metric("misses")
                self.log_hls_event("cache_miss", stream_id, stream_url)
                schedule_needed = True
            else:
                age = now - entry.extracted_at
                age_ms = int(age * 1000)
                if entry.url and age < self.hls_ttl_secs:
                    hls_url = entry.url
                    self.record_hls_metric("hits")
                    self.log_hls_event("cache_hit", stream_id, stream_url, age_ms=age_ms)
                else:
                    self.record_hls_metric("stale")
                    self.log_hls_event(
                        "cache_stale",
                        stream_id,
                        stream_url,
                        age_ms=age_ms,
                        had_url=bool(entry.url),
                        error=entry.error or "none",
                    )
                    if entry.url:
                        schedule_needed = True
                    else:
                        schedule_needed = age >= self.hls_error_retry_secs
                        if not schedule_needed:
                            wait_ms = max(0, int((self.hls_error_retry_secs - age) * 1000))
                            self.log_hls_event(
                                "retry_pending",
                                stream_id,
                                stream_url,
                                retry_in_ms=wait_ms,
                                error=entry.error or "none",
                            )
            if schedule_needed:
                self.schedule_hls_detection(stream_id, stream_url)
            return hls_url

        return self.detect_hls_stream_url(stream_url)

    def invalidate_stream(self, stream_id: str, target_url: str) -> Dict[str, Any]:
        prefix = f"live:{stream_id}:"
        with self.hls_lock:
            cache_keys = [key for key in list(self.hls_cache.keys()) if key.startswith(prefix)]
            job_keys = [key for key in list(self.hls_jobs.keys()) if key.startswith(prefix)]
            removed = 0
            for key in cache_keys:
                if self.hls_cache.pop(key, None) is not None:
                    removed += 1
        cancelled = 0
        for key in job_keys:
            if self.cancel_hls_job(key):
                cancelled += 1

        self.log_hls_event("invalidate", stream_id, target_url, removed=removed, cancelled=cancelled)

        rescheduled = False
        if self.live_hls_async and self.hls_executor is not None:
            self.schedule_hls_detection(stream_id, target_url)
            rescheduled = True

        return {
            "status": "ok",
            "removed": removed,
            "jobs_cancelled": cancelled,
            "rescheduled": rescheduled,
        }

    def try_get_hls(self, original_url: str) -> Optional[str]:
        try:
            return self.detect_hls_stream_url(original_url)
        except Exception:
            return None
