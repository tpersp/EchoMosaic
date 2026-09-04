"""On-demand RTSP to HLS conversion for browser playback."""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional


@dataclass
class RtspSession:
    source_url: str
    output_dir: Path
    process: Optional[subprocess.Popen]
    last_access: float
    restart_attempts: int = 0
    next_restart: float = 0.0
    last_error: str = ""


class RtspHlsService:
    """Own one FFmpeg HLS process per configured RTSP stream."""

    ALLOWED_ASSETS = {"index.m3u8"}

    def __init__(
        self,
        *,
        root: Path,
        ffmpeg_path: Optional[str] = None,
        popen: Callable[..., subprocess.Popen] = subprocess.Popen,
        clock: Callable[[], float] = time.monotonic,
        idle_timeout: float = 60.0,
        segment_seconds: float = 1.0,
        playlist_size: int = 3,
        restart_max_delay: float = 30.0,
    ) -> None:
        self.root = Path(root)
        self.ffmpeg_path = ffmpeg_path or shutil.which("ffmpeg")
        self.popen = popen
        self.clock = clock
        self.idle_timeout = max(10.0, float(idle_timeout))
        self.segment_seconds = max(0.5, float(segment_seconds))
        self.playlist_size = max(3, int(playlist_size))
        self.restart_max_delay = max(2.0, float(restart_max_delay))
        self._lock = threading.RLock()
        self._sessions: Dict[str, RtspSession] = {}
        self._stop_event = threading.Event()
        self._reaper: Optional[threading.Thread] = None

    @staticmethod
    def is_rtsp_url(url: str) -> bool:
        return str(url or "").strip().lower().startswith(("rtsp://", "rtsps://"))

    @staticmethod
    def _safe_id(stream_id: str) -> str:
        # Use an opaque stable identifier: stream labels may contain sensitive site names,
        # and punctuation-only labels must not collide in the shared runtime directory.
        return hashlib.sha256(stream_id.encode("utf-8")).hexdigest()[:20]

    def _command(self, source_url: str, output_dir: Path) -> list[str]:
        if not self.ffmpeg_path:
            raise RuntimeError("FFmpeg is required for RTSP playback")
        return [
            self.ffmpeg_path,
            "-hide_banner", "-loglevel", "warning", "-nostdin",
            "-rtsp_transport", "tcp", "-i", source_url,
            "-map", "0:v:0", "-map", "0:a:0?",
            "-c:v", "copy", "-c:a", "aac", "-ar", "48000", "-ac", "2",
            "-f", "hls", "-hls_time", f"{self.segment_seconds:g}",
            "-hls_list_size", str(self.playlist_size),
            "-hls_flags", "delete_segments+append_list+omit_endlist+independent_segments",
            "-hls_segment_filename", str(output_dir / "%06d.ts"),
            str(output_dir / "index.m3u8"),
        ]

    def ensure(self, stream_id: str, source_url: str) -> str:
        """Start or reuse a converter and return its browser-facing playlist path."""
        if not self.is_rtsp_url(source_url):
            raise ValueError("Only RTSP URLs are accepted")
        now = self.clock()
        safe_id = self._safe_id(stream_id)
        with self._lock:
            self._stop_idle_locked(now)
            current = self._sessions.get(stream_id)
            if current and current.source_url == source_url:
                current.last_access = now
                if current.process is None or current.process.poll() is not None:
                    current.next_restart = 0.0
                    self._restart_locked(stream_id, current, now)
                return f"/stream/rtsp/{safe_id}/index.m3u8"
            self._stop_locked(stream_id)
            output_dir = self.root / safe_id
            shutil.rmtree(output_dir, ignore_errors=True)
            output_dir.mkdir(parents=True, exist_ok=True)
            session = RtspSession(source_url, output_dir, None, now)
            self._sessions[stream_id] = session
            self._launch_locked(stream_id, session)
            return f"/stream/rtsp/{safe_id}/index.m3u8"

    def _launch_locked(self, stream_id: str, session: RtspSession) -> None:
        session.output_dir.mkdir(parents=True, exist_ok=True)
        for path in session.output_dir.glob("*.ts"):
            path.unlink(missing_ok=True)
        (session.output_dir / "index.m3u8").unlink(missing_ok=True)
        try:
            process = self.popen(
                self._command(session.source_url, session.output_dir),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                errors="replace",
                start_new_session=True,
            )
        except Exception as exc:
            session.process = None
            session.last_error = self._redact(str(exc), session.source_url)
            session.restart_attempts += 1
            session.next_restart = self.clock() + self._restart_delay(session.restart_attempts)
            return
        session.process = process
        stderr = getattr(process, "stderr", None)
        if stderr is not None:
            threading.Thread(
                target=self._drain_stderr,
                args=(stream_id, process, stderr, session.source_url),
                name=f"rtsp-log-{self._safe_id(stream_id)}",
                daemon=True,
            ).start()

    @staticmethod
    def _redact(message: str, source_url: str) -> str:
        clean = str(message or "").replace(source_url, "<redacted-rtsp-url>")
        return re.sub(r"(rtsps?://)[^\s]+", r"\1<redacted>", clean, flags=re.IGNORECASE)[-2000:]

    def _drain_stderr(self, stream_id: str, process, stderr, source_url: str) -> None:
        recent: list[str] = []
        try:
            for line in stderr:
                line = self._redact(line.strip(), source_url)
                if line:
                    recent.append(line)
                    recent = recent[-8:]
        finally:
            try:
                stderr.close()
            except Exception:
                pass
            if recent:
                with self._lock:
                    session = self._sessions.get(stream_id)
                    if session and session.process is process:
                        session.last_error = "\n".join(recent)

    def _restart_delay(self, attempts: int) -> float:
        return min(self.restart_max_delay, float(2 ** min(max(0, attempts - 1), 5)))

    def _restart_locked(self, stream_id: str, session: RtspSession, now: float) -> None:
        if now < session.next_restart:
            return
        session.restart_attempts += 1
        session.next_restart = now + self._restart_delay(session.restart_attempts)
        self._launch_locked(stream_id, session)

    def asset_path(self, stream_token: str, filename: str) -> Optional[Path]:
        if filename != "index.m3u8" and not (filename.endswith(".ts") and filename[:-3].isdigit()):
            return None
        # The route contains the opaque token returned by ``ensure``. Hashing it
        # again would point at a different directory and make every asset 404.
        if len(stream_token) != 20 or any(ch not in "0123456789abcdef" for ch in stream_token):
            return None
        now = self.clock()
        with self._lock:
            for session in self._sessions.values():
                if session.output_dir.name == stream_token:
                    session.last_access = now
                    path = session.output_dir / filename
                    return path if path.is_file() else None
        return None

    def _stop_idle_locked(self, now: float) -> None:
        for stream_id, session in list(self._sessions.items()):
            if now - session.last_access >= self.idle_timeout:
                self._stop_locked(stream_id)
            elif session.process is None or session.process.poll() is not None:
                self._restart_locked(stream_id, session, now)

    def _stop_locked(self, stream_id: str) -> None:
        session = self._sessions.pop(stream_id, None)
        if not session:
            return
        if session.process is not None and session.process.poll() is None:
            session.process.terminate()
            try:
                session.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                session.process.kill()
        shutil.rmtree(session.output_dir, ignore_errors=True)

    def stop(self, stream_id: str) -> None:
        with self._lock:
            self._stop_locked(stream_id)

    def diagnostics(self, stream_id: str) -> Dict[str, object]:
        with self._lock:
            session = self._sessions.get(stream_id)
            if not session:
                return {"active": False, "error": ""}
            running = session.process is not None and session.process.poll() is None
            return {
                "active": running,
                "restart_attempts": session.restart_attempts,
                "error": session.last_error,
            }

    def stop_all(self) -> None:
        self._stop_event.set()
        with self._lock:
            for stream_id in list(self._sessions):
                self._stop_locked(stream_id)

    def start_reaper(self) -> None:
        if self._reaper and self._reaper.is_alive():
            return
        self._reaper = threading.Thread(target=self._reap_loop, name="rtsp-hls-reaper", daemon=True)
        self._reaper.start()

    def _reap_loop(self) -> None:
        interval = min(15.0, self.idle_timeout / 2)
        while not self._stop_event.wait(interval):
            with self._lock:
                self._stop_idle_locked(self.clock())
