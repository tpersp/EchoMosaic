"""Stable Horde image generation client for EchoMosaic.

This module wraps the Stable Horde public API so the rest of the project can
request AI generated images without embedding HTTP logic inside the Flask app.
"""

from __future__ import annotations

import base64
import logging
import os
import socket
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union
from urllib.parse import urlparse

try:
    import requests
except ImportError as exc:  # pragma: no cover - requests is a hard dependency
    raise RuntimeError("Stable Horde integration requires the 'requests' package") from exc

__all__ = [
    "StableHorde",
    "StableHordeError",
    "StableHordeCancelled",
    "StableHordeOrphaned",
    "StableHordeResult",
    "StableHordeGeneration",
]

DEFAULT_BASE_URL = "https://stablehorde.net/api/v2"
DEFAULT_CLIENT_AGENT = "EchoMosaic StableHorde Client/1.0"
DEFAULT_SAMPLER = "k_euler"
DEFAULT_WIDTH = 512
DEFAULT_HEIGHT = 512
DEFAULT_TIMEOUT_SECONDS = 900.0
PAYLOAD_DONE_GRACE_SECONDS = 45.0
MIN_DIMENSION = 64
MAX_DIMENSION = 2048
DIMENSION_STEP = 64
LOG_PREFIX = "[StableHorde]"


class StableHordeError(RuntimeError):
    """Raised when a Stable Horde request fails or returns an error response."""


class StableHordeCancelled(StableHordeError):
    """Raised when a Stable Horde job is cancelled by the caller."""


class StableHordeOrphaned(StableHordeError):
    """Raised when polling stops because the caller requested it."""


@dataclass
class StableHordeGeneration:
    """Represents a single generated image saved to disk."""

    path: Path
    seed: Optional[str] = None
    model: Optional[str] = None
    worker: Optional[str] = None
    url: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def read_bytes(self) -> bytes:
        """Return the image contents."""

        return self.path.read_bytes()

    def __str__(self) -> str:  # pragma: no cover - convenience representation
        return f"{self.path.name} (model={self.model}, seed={self.seed})"


@dataclass
class StableHordeResult:
    """Holds the completed generation details."""

    job_id: str
    prompt: str
    generations: List[StableHordeGeneration]
    kudos: Optional[float] = None
    wait_time: Optional[float] = None
    queue_position: Optional[int] = None
    faulted: bool = False
    raw: Dict[str, Any] = field(default_factory=dict)
    persisted: bool = True
    _temp_dir: Optional[tempfile.TemporaryDirectory] = field(default=None, repr=False)

    def cleanup(self) -> None:
        """Delete any temporary directory that was created for this job."""

        if self._temp_dir is not None:
            self._temp_dir.cleanup()
            self._temp_dir = None


def _load_env_file(path: Union[str, os.PathLike[str]]) -> None:
    """Populate `os.environ` with key/value pairs from a `.env` file."""

    env_path = Path(path)
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        if not key:
            continue
        existing = os.environ.get(key)
        if existing not in (None, ""):
            continue
        value = value.strip().strip('"').strip("'")
        os.environ[key] = value


def _as_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _detect_extension(data: bytes) -> str:
    if data.startswith(b"\x89PNG"):
        return ".png"
    if data.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return ".webp"
    if data.startswith(b"GIF8"):
        return ".gif"
    return ".bin"


def _fire_callback(callback: Optional[Callable[[str, Dict[str, Any]], None]], event: str, payload: Dict[str, Any]) -> None:
    # Helper to invoke status callbacks safely.
    if callback is None:
        return
    try:
        callback(event, payload)
    except Exception:
        pass


class StableHorde:
    """Client wrapper around the Stable Horde API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        base_url: str = DEFAULT_BASE_URL,
        client_agent: str = DEFAULT_CLIENT_AGENT,
        logger: Optional[logging.Logger] = None,
        save_dir: Optional[Union[str, Path]] = None,
        persist_images: Optional[bool] = None,
        request_timeout: int = 30,
        default_poll_interval: float = 3.0,
        default_timeout: float = 600.0,
        load_env: bool = True,
    ) -> None:
        default_logger = logging.getLogger("echomosaic.stablehorde")
        if default_logger.level == logging.NOTSET:
            default_logger.setLevel(logging.INFO)
        root_logger = logging.getLogger()
        if not default_logger.handlers and not root_logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
            default_logger.addHandler(handler)
        self._loggers: List[logging.Logger] = []
        if logger is not None:
            self._loggers.append(logger)
        self._loggers.append(default_logger)
        self.logger = logger or default_logger
        if load_env:
            env_file = os.getenv("STABLE_HORDE_ENV_FILE", ".env")
            _load_env_file(env_file)

        if api_key is None:
            api_key = os.getenv("STABLE_HORDE_API_KEY")
        self.api_key = api_key.strip() if api_key else None
        self.base_url = base_url.rstrip("/")
        self.client_agent = client_agent
        self.request_timeout = int(request_timeout)
        self.default_poll_interval = (
            float(default_poll_interval)
            if isinstance(default_poll_interval, (int, float)) and float(default_poll_interval) > 0
            else 3.0
        )
        self.default_timeout = (
            float(default_timeout)
            if isinstance(default_timeout, (int, float)) and float(default_timeout) > 0
            else DEFAULT_TIMEOUT_SECONDS
        )
        self._parsed_base = urlparse(self.base_url)

        if save_dir is None:
            save_dir_env = os.getenv("STABLE_HORDE_SAVE_DIR")
            if save_dir_env:
                save_dir = save_dir_env
        self._persistent_dir = Path(save_dir).expanduser() if save_dir else None
        if self._persistent_dir:
            self._persistent_dir.mkdir(parents=True, exist_ok=True)

        if persist_images is None:
            persist_images = _as_bool(
                os.getenv("STABLE_HORDE_PERSIST_IMAGES"),
                default=self._persistent_dir is not None,
            )
        self.persist_images = bool(persist_images)

        self._session = requests.Session()
        self._session.headers.update(
            {
                "Client-Agent": self.client_agent,
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
        )
        if self.api_key:
            self._session.headers["apikey"] = self.api_key
        else:
            # Anonymous access still requires the placeholder API key.
            self._session.headers["apikey"] = "0000000000"

        fallback_dir = Path(os.getenv("STABLE_HORDE_FALLBACK_DIR", Path.cwd() / "stablehorde_outputs"))
        self._default_output_dir = fallback_dir.expanduser()
        if self.persist_images and self._persistent_dir is None:
            self._persistent_dir = self._default_output_dir
            self._persistent_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------
    def _log(self, level: int, message: str, *args: Any) -> None:
        """Emit a log message with a Stable Horde prefix."""

        seen: Set[int] = set()
        for logger in self._loggers:
            if logger is None:
                continue
            ident = id(logger)
            if ident in seen:
                continue
            seen.add(ident)
            try:
                if logger.isEnabledFor(level):
                    logger.log(level, "%s " + message, LOG_PREFIX, *args)
            except Exception:
                # Logging must never raise upstream.
                continue

    def _ensure_horde_ready(self) -> None:
        """Verify Stable Horde is reachable before attempting a generation."""

        if self._ping_horde():
            return
        raise StableHordeError("Stable Horde unreachable after health check attempts")

    def _ping_horde(self, *, attempts: int = 5, base_delay: int = 10, timeout: int = 3) -> bool:
        """Attempt to resolve and open a socket to the Stable Horde host."""

        host = self._parsed_base.hostname or "stablehorde.net"
        scheme = self._parsed_base.scheme or "https"
        port = self._parsed_base.port or (443 if scheme == "https" else 80)

        for attempt in range(1, attempts + 1):
            try:
                socket.getaddrinfo(host, port)
                with socket.create_connection((host, port), timeout=timeout):
                    pass
            except OSError as exc:
                self._log(
                    logging.WARNING,
                    "Ping attempt %d/%d to %s:%s failed (%s)",
                    attempt,
                    attempts,
                    host,
                    port,
                    exc,
                )
                if attempt < attempts:
                    delay = base_delay * attempt
                    self._log(logging.INFO, "Waiting %ds before retrying ping", delay)
                    time.sleep(delay)
                continue

            self._log(
                logging.INFO,
                "Ping to %s:%s succeeded on attempt %d",
                host,
                port,
                attempt,
            )
            return True

        self._log(
            logging.ERROR,
            "Ping to %s:%s failed after %d attempts",
            host,
            port,
            attempts,
        )
        return False

    def _parse_retry_after(self, header_value: Optional[str]) -> int:
        """Parse a Retry-After header into seconds, defaulting gracefully."""

        default_wait = 60
        if not header_value:
            return default_wait

        value = header_value.strip()
        if not value:
            return default_wait

        try:
            wait_seconds = float(value)
            if wait_seconds >= 0:
                return max(1, int(wait_seconds))
        except ValueError:
            pass

        try:
            target_dt = parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return default_wait

        if target_dt is None:
            return default_wait

        if target_dt.tzinfo is None:
            target_dt = target_dt.replace(tzinfo=timezone.utc)

        now = datetime.now(timezone.utc)
        delta = (target_dt - now).total_seconds()
        if delta <= 0:
            return default_wait
        return max(1, int(delta))

    def _extract_response_detail(self, response: "requests.Response") -> Any:
        """Safely extract JSON or text details from a response for logging."""

        try:
            return response.json()
        except ValueError:
            return response.text

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def list_models(self) -> List[Dict[str, Any]]:
        """Return the currently available models."""

        payload = self._request("GET", "/status/models")
        if not isinstance(payload, list):
            raise StableHordeError(f"Unexpected response from model listing: {payload}")
        return payload

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Fetch the raw status payload for an existing job."""

        return self._request("GET", f"/generate/status/{job_id}")

    def cancel_job(self, job_id: str) -> None:
        """Request job cancellation."""

        self._request("DELETE", f"/generate/status/{job_id}")

    def generate_images(
        self,
        prompt: str,
        *,
        negative_prompt: Optional[str] = None,
        models: Optional[Iterable[str]] = None,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        steps: int = 30,
        cfg_scale: float = 7.5,
        sampler_name: str = DEFAULT_SAMPLER,
        seed: Optional[Union[int, str]] = None,
        samples: int = 1,
        nsfw: bool = False,
        censor_nsfw: bool = False,
        post_processing: Optional[Iterable[str]] = None,
        params: Optional[Dict[str, Any]] = None,
        extras: Optional[Dict[str, Any]] = None,
        poll_interval: Optional[float] = None,
        timeout: Optional[float] = None,
        persist: Optional[bool] = None,
        output_dir: Optional[Union[str, Path]] = None,
        status_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        cancel_callback: Optional[Callable[[], bool]] = None,
        stop_callback: Optional[Callable[[], bool]] = None,
    ) -> StableHordeResult:
        """Generate one or more images from the provided prompt."""

        if not prompt or not prompt.strip():
            raise StableHordeError("Prompt is required for Stable Horde generation")

        self._ensure_horde_ready()

        width = int(width)
        height = int(height)
        steps = int(steps)
        samples = int(samples)
        if samples < 1 or samples > 10:
            raise StableHordeError("samples must be between 1 and 10 for Stable Horde")
        self._validate_dimensions(width, height)

        params_payload: Dict[str, Any] = {
            "sampler_name": sampler_name,
            "cfg_scale": cfg_scale,
            "steps": steps,
            "width": width,
            "height": height,
            "n": samples,
            "seed": str(seed) if seed is not None else "random",
        }
        if post_processing:
            params_payload["post_processing"] = list(post_processing)
        if params:
            params_payload.update(params)

        payload: Dict[str, Any] = {
            "prompt": prompt,
            "params": params_payload,
            "nsfw": bool(nsfw),
            "censor_nsfw": bool(censor_nsfw),
        }
        if negative_prompt:
            payload["negative_prompt"] = negative_prompt
        if models:
            payload["models"] = list(models)
        if extras:
            payload.update(extras)

        response = self._request("POST", "/generate/async", json=payload)
        job_id = response.get("id") if isinstance(response, dict) else None
        if not job_id:
            raise StableHordeError(f"Stable Horde did not return a job id: {response}")

        _fire_callback(status_callback, "accepted", {"job_id": job_id})

        if cancel_callback and cancel_callback():
            try:
                self.cancel_job(job_id)
            except StableHordeError:
                pass
            _fire_callback(status_callback, "cancelled", {"job_id": job_id, "message": "Cancelled before start"})
            raise StableHordeCancelled(f"Stable Horde job {job_id} cancelled before polling")

        self._log(logging.INFO, "Job %s started", job_id)

        persist_flag = self.persist_images if persist is None else bool(persist)
        destination, temp_dir = self._resolve_output_dir(persist_flag, output_dir)

        poll_value = float(poll_interval) if isinstance(poll_interval, (int, float)) and float(poll_interval) > 0 else self.default_poll_interval
        timeout_value = self.default_timeout if timeout is None or (isinstance(timeout, (int, float)) and float(timeout) <= 0) else float(timeout)
        status = self._poll_job(
            job_id,
            poll_value,
            timeout_value,
            status_callback=status_callback,
            cancel_callback=cancel_callback,
            stop_callback=stop_callback,
        )

        generations: List[StableHordeGeneration] = []
        for index, generation_payload in enumerate(status.get("generations", [])):
            img_value = generation_payload.get("img")
            if not img_value:
                continue
            saved_path, source_url = self._save_image(job_id, index, img_value, destination)
            metadata = {k: v for k, v in generation_payload.items() if k not in {"img"}}
            seed_value = metadata.pop("seed", None)
            model_value = metadata.pop("model", None)
            worker_value = metadata.pop("worker_name", None)
            generations.append(
                StableHordeGeneration(
                    path=saved_path,
                    seed=seed_value,
                    model=model_value,
                    worker=worker_value,
                    url=source_url,
                    metadata=metadata,
                )
            )

        if not generations:
            if temp_dir is not None:
                temp_dir.cleanup()
            raise StableHordeError(f"Stable Horde job {job_id} completed without returning images")

        result = StableHordeResult(
            job_id=job_id,
            prompt=prompt,
            generations=generations,
            kudos=status.get("kudos"),
            wait_time=status.get("wait_time"),
            queue_position=status.get("queue_position"),
            faulted=bool(status.get("faulted")),
            raw=status,
            persisted=persist_flag,
        )
        if temp_dir is not None:
            result._temp_dir = temp_dir

        _fire_callback(
            status_callback,
            "completed",
            {
                "job_id": job_id,
                "status": status,
                "images": [
                    {
                        "path": str(gen.path),
                        "seed": gen.seed,
                        "model": gen.model,
                        "worker": gen.worker,
                        "url": gen.url,
                    }
                    for gen in generations
                ],
            },
        )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _url(self, path: str) -> str:
        if path.startswith("http://") or path.startswith("https://"):
            return path
        if not path.startswith("/"):
            path = "/" + path
        return f"{self.base_url}{path}"

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        url = self._url(path)
        attempts = 3
        backoff_schedule = [1, 5, 15]

        for attempt in range(1, attempts + 1):
            try:
                response = self._session.request(method, url, timeout=self.request_timeout, **kwargs)
            except requests.RequestException as exc:
                if attempt >= attempts:
                    self._log(
                        logging.ERROR,
                        "%s %s failed after %d attempts (%s)",
                        method,
                        url,
                        attempt,
                        exc,
                    )
                    raise StableHordeError(f"Request to Stable Horde failed after retries: {exc}") from exc

                delay = backoff_schedule[min(attempt - 1, len(backoff_schedule) - 1)]
                self._log(
                    logging.WARNING,
                    "%s %s attempt %d/%d failed (%s). Retrying in %ds",
                    method,
                    url,
                    attempt,
                    attempts,
                    exc,
                    delay,
                )
                time.sleep(delay)
                continue

            status = response.status_code
            if status == 429:
                wait_seconds = self._parse_retry_after(response.headers.get("Retry-After"))
                if attempt >= attempts:
                    self._log(
                        logging.ERROR,
                        "%s %s hit rate limit after %d attempts (Retry-After %ds)",
                        method,
                        url,
                        attempt,
                        wait_seconds,
                    )
                    detail = self._extract_response_detail(response)
                    raise StableHordeError(f"Stable Horde error 429: {detail}")

                self._log(
                    logging.WARNING,
                    "%s %s received 429 rate limit (attempt %d/%d). Waiting %ds before retry",
                    method,
                    url,
                    attempt,
                    attempts,
                    wait_seconds,
                )
                time.sleep(wait_seconds)
                continue

            if 500 <= status:
                detail = self._extract_response_detail(response)
                if attempt >= attempts:
                    self._log(
                        logging.ERROR,
                        "%s %s received server error %d after %d attempts: %s",
                        method,
                        url,
                        status,
                        attempt,
                        detail,
                    )
                    raise StableHordeError(f"Stable Horde error {status}: {detail}")

                delay = backoff_schedule[min(attempt - 1, len(backoff_schedule) - 1)]
                self._log(
                    logging.WARNING,
                    "%s %s received server error %d (attempt %d/%d): %s. Retrying in %ds",
                    method,
                    url,
                    status,
                    attempt,
                    attempts,
                    detail,
                    delay,
                )
                time.sleep(delay)
                continue

            if status >= 400:
                detail = self._extract_response_detail(response)
                self._log(
                    logging.ERROR,
                    "%s %s failed with status %d: %s",
                    method,
                    url,
                    status,
                    detail,
                )
                raise StableHordeError(f"Stable Horde error {status}: {detail}")

            content_type = response.headers.get("Content-Type", "")
            self._log(
                logging.INFO,
                "%s %s succeeded with status %d",
                method,
                url,
                status,
            )
            if "application/json" in content_type:
                try:
                    return response.json()
                except ValueError:
                    return response.text
            return response.content

        raise StableHordeError("Stable Horde request retries exhausted")

    def _validate_dimensions(self, width: int, height: int) -> Tuple[int, int]:
        for value, label in ((width, "width"), (height, "height")):
            if value < MIN_DIMENSION or value > MAX_DIMENSION:
                raise StableHordeError(
                    f"{label} must be between {MIN_DIMENSION} and {MAX_DIMENSION} pixels"
                )
            if value % DIMENSION_STEP != 0:
                raise StableHordeError(f"{label} must be a multiple of {DIMENSION_STEP} pixels")
        return width, height

    def _resolve_output_dir(
        self,
        persist: bool,
        override: Optional[Union[str, Path]],
    ) -> Tuple[Path, Optional[tempfile.TemporaryDirectory]]:
        if persist:
            destination = (
                Path(override).expanduser()
                if override is not None
                else (self._persistent_dir or self._default_output_dir)
            )
            destination.mkdir(parents=True, exist_ok=True)
            return destination, None
        temp_dir = tempfile.TemporaryDirectory(prefix="echomosaic-horde-")
        return Path(temp_dir.name), temp_dir

    def _poll_job(
        self,
        job_id: str,
        poll_interval: float,
        timeout: Optional[float],
        status_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        cancel_callback: Optional[Callable[[], bool]] = None,
        stop_callback: Optional[Callable[[], bool]] = None,
    ) -> Dict[str, Any]:
        poll_value = float(poll_interval) if isinstance(poll_interval, (int, float)) and float(poll_interval) > 0 else self.default_poll_interval
        effective_timeout = (
            self.default_timeout
            if timeout is None or (isinstance(timeout, (int, float)) and float(timeout) <= 0)
            else float(timeout)
        )
        start_time = time.time()
        deadline = start_time + effective_timeout if effective_timeout and effective_timeout > 0 else None
        timeout_label = f"{effective_timeout:.0f}s" if effective_timeout and effective_timeout > 0 else "disabled"
        self._log(
            logging.INFO,
            "Polling job %s every %.1fs (timeout %s)",
            job_id,
            poll_value,
            timeout_label,
        )

        cancel_notified = False
        done_seen_at: Optional[float] = None

        def _check_cancelled() -> None:
            nonlocal cancel_notified
            if cancel_callback and cancel_callback():
                if not cancel_notified:
                    try:
                        self.cancel_job(job_id)
                    except StableHordeError:
                        pass
                    _fire_callback(status_callback, "cancelled", {"job_id": job_id, "message": "Cancelled by user"})
                    cancel_notified = True
                raise StableHordeCancelled(f"Stable Horde job {job_id} cancelled by caller")

        def _check_stopped() -> None:
            if stop_callback and stop_callback():
                self._log(
                    logging.INFO,
                    "Polling for job %s halted by stop callback",
                    job_id,
                )
                raise StableHordeOrphaned(f"Polling halted for Stable Horde job {job_id}")

        while True:
            _check_stopped()
            _check_cancelled()
            status = self._request("GET", f"/generate/status/{job_id}")
            _fire_callback(status_callback, "status", {"job_id": job_id, "status": status})
            done_flag = bool(status.get("done"))
            faulted_flag = bool(status.get("faulted"))
            generations = status.get("generations") or []
            wait_time = status.get("wait_time")
            queue_position = status.get("queue_position")
            now = time.time()
            self._log(
                logging.INFO,
                "Job %s status: done=%s faulted=%s queue=%s wait=%s generations=%d",
                job_id,
                done_flag,
                faulted_flag,
                queue_position,
                wait_time,
                len(generations),
            )
            if faulted_flag:
                message = status.get("message") or status
                self._log(logging.ERROR, "Job %s faulted — %s", job_id, message)
                _fire_callback(status_callback, "fault", {"job_id": job_id, "status": status, "message": message})
                raise StableHordeError(f"Stable Horde job {job_id} faulted: {message}")
            if generations:
                self._log(logging.INFO, "Job %s complete — %d generation(s) ready", job_id, len(generations))
                return status
            if done_flag and not generations:
                if done_seen_at is None:
                    done_seen_at = now
                    self._log(logging.INFO, "Job %s marked done; awaiting generation payload", job_id)
                elif now - done_seen_at >= PAYLOAD_DONE_GRACE_SECONDS:
                    message = "Stable Horde reported completion without returning images"
                    self._log(
                        logging.ERROR,
                        "Job %s done but no images after %.1fs",
                        job_id,
                        now - done_seen_at,
                    )
                    _fire_callback(status_callback, "fault", {"job_id": job_id, "status": status, "message": message})
                    raise StableHordeError(f"{message}: job {job_id}")
                sleep_for = min(poll_value, 2.0)
            else:
                done_seen_at = None
                if isinstance(wait_time, (int, float)) and wait_time > 0:
                    sleep_for = max(1.0, min(float(wait_time), 15.0))
                else:
                    sleep_for = poll_value
            if deadline is not None and now >= deadline:
                self._log(logging.WARNING, "Job %s timed out after %.1fs", job_id, now - start_time)
                _fire_callback(status_callback, "timeout", {"job_id": job_id, "status": status})
                raise StableHordeError(f"Timed out waiting for Stable Horde job {job_id}")
            if deadline is not None and sleep_for > 0 and now + sleep_for >= deadline:
                sleep_for = max(0.0, deadline - now)
                if sleep_for <= 0:
                    self._log(logging.WARNING, "Job %s timed out after %.1fs", job_id, now - start_time)
                    _fire_callback(status_callback, "timeout", {"job_id": job_id, "status": status})
                    raise StableHordeError(f"Timed out waiting for Stable Horde job {job_id}")
            if sleep_for > 0:
                time.sleep(sleep_for)

    def _save_image(
        self,
        job_id: str,
        index: int,
        image_payload: str,
        destination: Path,
    ) -> Tuple[Path, Optional[str]]:
        url = image_payload if image_payload.startswith(("http://", "https://")) else None
        data = self._decode_image_payload(image_payload)
        extension = _detect_extension(data)
        filename = f"{job_id}-{index + 1}{extension}"
        path = destination / filename
        path.write_bytes(data)
        return path, url

    def _decode_image_payload(self, payload: str) -> bytes:
        if payload.startswith(("http://", "https://")):
            return self._download_image(payload)
        if payload.startswith("data:"):
            _, _, raw = payload.partition(",")
            payload = raw
        try:
            return base64.b64decode(payload, validate=False)
        except Exception as exc:  # pragma: no cover - defensive path
            raise StableHordeError("Failed to decode base64 image data") from exc

    def _download_image(self, url: str) -> bytes:
        try:
            response = self._session.get(url, timeout=self.request_timeout, headers={"Accept": "*/*"})
        except requests.RequestException as exc:
            raise StableHordeError(f"Failed to download generated image: {exc}") from exc
        if response.status_code >= 400:
            raise StableHordeError(f"Stable Horde returned {response.status_code} when downloading {url}")
        return response.content
