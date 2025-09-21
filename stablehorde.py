"""Stable Horde image generation client for EchoMosaic.

This module wraps the Stable Horde public API so the rest of the project can
request AI generated images without embedding HTTP logic inside the Flask app.
"""

from __future__ import annotations

import base64
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

try:
    import requests
except ImportError as exc:  # pragma: no cover - requests is a hard dependency
    raise RuntimeError("Stable Horde integration requires the 'requests' package") from exc

__all__ = [
    "StableHorde",
    "StableHordeError",
    "StableHordeResult",
    "StableHordeGeneration",
]

DEFAULT_BASE_URL = "https://stablehorde.net/api/v2"
DEFAULT_CLIENT_AGENT = "EchoMosaic StableHorde Client/1.0"
DEFAULT_SAMPLER = "k_euler"
DEFAULT_WIDTH = 512
DEFAULT_HEIGHT = 512
MIN_DIMENSION = 64
MAX_DIMENSION = 2048
DIMENSION_STEP = 64


class StableHordeError(RuntimeError):
    """Raised when a Stable Horde request fails or returns an error response."""


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
        save_dir: Optional[Union[str, Path]] = None,
        persist_images: Optional[bool] = None,
        request_timeout: int = 30,
        default_poll_interval: float = 3.0,
        default_timeout: float = 600.0,
        load_env: bool = True,
    ) -> None:
        if load_env:
            env_file = os.getenv("STABLE_HORDE_ENV_FILE", ".env")
            _load_env_file(env_file)

        if api_key is None:
            api_key = os.getenv("STABLE_HORDE_API_KEY")
        self.api_key = api_key.strip() if api_key else None
        self.base_url = base_url.rstrip("/")
        self.client_agent = client_agent
        self.request_timeout = request_timeout
        self.default_poll_interval = default_poll_interval
        self.default_timeout = default_timeout

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
    ) -> StableHordeResult:
        """Generate one or more images from the provided prompt."""

        if not prompt or not prompt.strip():
            raise StableHordeError("Prompt is required for Stable Horde generation")

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

        persist_flag = self.persist_images if persist is None else bool(persist)
        destination, temp_dir = self._resolve_output_dir(persist_flag, output_dir)

        status = self._poll_job(
            job_id,
            poll_interval or self.default_poll_interval,
            timeout or self.default_timeout,
            status_callback=status_callback,
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
        try:
            response = self._session.request(method, url, timeout=self.request_timeout, **kwargs)
        except requests.RequestException as exc:
            raise StableHordeError(f"Request to Stable Horde failed: {exc}") from exc
        if response.status_code >= 400:
            try:
                detail: Any = response.json()
            except ValueError:
                detail = response.text
            raise StableHordeError(f"Stable Horde error {response.status_code}: {detail}")
        content_type = response.headers.get("Content-Type", "")
        if "application/json" in content_type:
            return response.json()
        return response.content

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
        timeout: float,
        status_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        deadline = time.time() + timeout
        while True:
            status = self._request("GET", f"/generate/status/{job_id}")
            _fire_callback(status_callback, "status", {"job_id": job_id, "status": status})
            if status.get("faulted"):
                message = status.get("message") or status
                _fire_callback(status_callback, "fault", {"job_id": job_id, "status": status, "message": message})
                raise StableHordeError(f"Stable Horde job {job_id} faulted: {message}")
            generations = status.get("generations") or []
            if generations:
                return status
            if status.get("done") and not generations:
                time.sleep(1.0)
                continue
            if time.time() >= deadline:
                _fire_callback(status_callback, "timeout", {"job_id": job_id, "status": status})
                raise StableHordeError(f"Timed out waiting for Stable Horde job {job_id}")
            wait_time = status.get("wait_time")
            if isinstance(wait_time, (int, float)) and wait_time > 0:
                sleep_for = max(1.0, min(float(wait_time), 15.0))
            else:
                sleep_for = poll_interval
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
