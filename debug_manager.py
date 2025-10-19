"""Helpers for serving EchoMosaic debug log streams."""

from __future__ import annotations

import html
import logging
import re
import select
import subprocess
import time
from typing import Generator, List, Optional

import config_manager

logger = logging.getLogger(__name__)

DEFAULT_INITIAL_LINES = 200
DEFAULT_DOWNLOAD_LINES = 5000
KEEPALIVE_SECONDS = 15

TIMESTAMP_PATTERN = re.compile(
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}|[A-Z][a-z]{2} \d{1,2} \d{2}:\d{2}:\d{2})"
)
URL_PATTERN = re.compile(r"(https?://[^\s]+)")


def colorize_log_line(line: str) -> str:
    """Return HTML markup with spans that highlight log metadata."""

    plain_line = line.rstrip("\n")
    escaped = html.escape(plain_line)

    escaped = TIMESTAMP_PATTERN.sub(r'<span class="timestamp">\1</span>', escaped)
    escaped = URL_PATTERN.sub(
        r'<a class="log-link" href="\1" target="_blank" rel="noopener noreferrer">\1</a>',
        escaped,
    )

    upper_line = plain_line.upper()
    if "ERROR" in upper_line:
        css_class = "log-error"
    elif "WARNING" in upper_line or "WARN" in upper_line:
        css_class = "log-warning"
    elif "INFO" in upper_line:
        css_class = "log-info"
    elif "DEBUG" in upper_line:
        css_class = "log-debug"
    else:
        css_class = "log-generic"

    return f'<span class="{css_class}">{escaped}</span>'


class JournalAccessError(RuntimeError):
    """Raised when journalctl logs cannot be retrieved."""


def _resolve_log_unit() -> str:
    """Return the systemd unit name to inspect."""

    cfg = config_manager.load_config()
    unit = cfg.get("SERVICE_NAME") or "echomosaic.service"
    return str(unit)


def _build_recent_command(unit: str, limit: int) -> List[str]:
    """Return the journalctl command used to fetch recent lines."""

    lines = max(int(limit), 0)
    return [
        "journalctl",
        "-u",
        unit,
        f"-n{lines}",
        "--no-pager",
        "--output=short",
    ]


def _build_follow_command(unit: str) -> List[str]:
    """Return the journalctl command used to follow the journal."""

    return [
        "journalctl",
        "-u",
        unit,
        "--no-pager",
        "--output=short",
        "-n0",
        "-f",
    ]


def _format_sse(data: str, event: Optional[str] = None) -> str:
    """Format a Server-Sent Event payload."""

    lines = []
    if event:
        lines.append(f"event: {event}")
    payload = data.splitlines() or [""]
    for part in payload:
        lines.append(f"data: {part}")
    return "\n".join(lines) + "\n\n"


def get_recent_log_lines(limit: int = DEFAULT_INITIAL_LINES) -> str:
    """Return the most recent log lines for the configured systemd unit."""

    unit = _resolve_log_unit()
    try:
        output = subprocess.check_output(
            _build_recent_command(unit, limit),
            text=True,
            stderr=subprocess.STDOUT,
        )
    except FileNotFoundError as exc:
        raise JournalAccessError("journalctl command not found on this system.") from exc
    except subprocess.CalledProcessError as exc:
        message = exc.output.strip() or f"journalctl returned exit code {exc.returncode}"
        raise JournalAccessError(message) from exc
    return output


def stream_journal_follow(initial_limit: int = DEFAULT_INITIAL_LINES) -> Generator[str, None, None]:
    """Yield Server-Sent Event messages containing live journal output."""

    unit = _resolve_log_unit()

    if initial_limit:
        try:
            recent_lines = get_recent_log_lines(initial_limit)
        except JournalAccessError as exc:
            yield _format_sse(str(exc), event="error")
            return
        for line in recent_lines.splitlines():
            yield _format_sse(colorize_log_line(line))

    try:
        process = subprocess.Popen(
            _build_follow_command(unit),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except FileNotFoundError as exc:
        yield _format_sse("journalctl command not found; live streaming unavailable.", event="error")
        return
    except Exception as exc:
        logger.warning("Unable to start journalctl stream for %s: %s", unit, exc)
        yield _format_sse(f"Unable to start journal stream: {exc}", event="error")
        return

    assert process.stdout is not None  # For type-checkers; stdout is PIPE

    last_emit = time.monotonic()
    try:
        while True:
            if process.poll() is not None:
                # Drain any remaining buffered output before exiting.
                for line in process.stdout:
                    yield _format_sse(colorize_log_line(line))
                break

            ready, _, _ = select.select([process.stdout], [], [], 1.0)
            if ready:
                line = process.stdout.readline()
                if not line:
                    break
                yield _format_sse(colorize_log_line(line))
                last_emit = time.monotonic()
            elif time.monotonic() - last_emit >= KEEPALIVE_SECONDS:
                yield ": keep-alive\n\n"
                last_emit = time.monotonic()
    except GeneratorExit:
        raise
    except Exception as exc:
        logger.warning("Error while streaming journalctl output: %s", exc)
        yield _format_sse(f"Stream halted: {exc}", event="error")
    finally:
        _terminate_process(process)


def _terminate_process(process: subprocess.Popen) -> None:
    """Terminate the journalctl process gracefully."""

    if process.poll() is not None:
        return
    try:
        process.terminate()
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        process.kill()
    except Exception as exc:
        logger.debug("Failed to terminate journalctl process cleanly: %s", exc)
