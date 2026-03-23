"""Thumbnail rendering, snapshot building, and async refresh helpers."""

from __future__ import annotations

import io
import os
import random
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple


class ThumbnailService:
    def __init__(
        self,
        *,
        Image,
        ImageDraw,
        ImageFont,
        ImageOps,
        cv2_module,
        requests_module,
        eventlet_module,
        logger,
        dashboard_thumbnail_size: Tuple[int, int],
        image_thumbnail_filter,
        thumbnail_jpeg_quality: int,
        settings,
        get_stream_runtime_state,
        detect_media_kind,
        infer_media_mode,
        resolve_media_path,
        ensure_thumbnail_dir,
        thumbnail_disk_path,
        thumbnail_public_url,
        public_thumbnail_payload,
        runtime_timestamp_to_iso,
        stream_runtime_lock,
        stream_runtime_state,
        safe_emit,
        playback_manager_getter,
        media_mode_choices,
        media_mode_livestream: str,
        media_mode_video: str,
        media_mode_ai: str,
        media_mode_picsum: str,
    ) -> None:
        self.Image = Image
        self.ImageDraw = ImageDraw
        self.ImageFont = ImageFont
        self.ImageOps = ImageOps
        self.cv2 = cv2_module
        self.requests = requests_module
        self.eventlet = eventlet_module
        self.logger = logger
        self.dashboard_thumbnail_size = dashboard_thumbnail_size
        self.image_thumbnail_filter = image_thumbnail_filter
        self.thumbnail_jpeg_quality = thumbnail_jpeg_quality
        self.settings = settings
        self.get_stream_runtime_state = get_stream_runtime_state
        self.detect_media_kind = detect_media_kind
        self.infer_media_mode = infer_media_mode
        self.resolve_media_path = resolve_media_path
        self.ensure_thumbnail_dir = ensure_thumbnail_dir
        self.thumbnail_disk_path = thumbnail_disk_path
        self.thumbnail_public_url = thumbnail_public_url
        self.public_thumbnail_payload = public_thumbnail_payload
        self.runtime_timestamp_to_iso = runtime_timestamp_to_iso
        self.stream_runtime_lock = stream_runtime_lock
        self.stream_runtime_state = stream_runtime_state
        self.safe_emit = safe_emit
        self.playback_manager_getter = playback_manager_getter
        self.media_mode_choices = media_mode_choices
        self.media_mode_livestream = media_mode_livestream
        self.media_mode_video = media_mode_video
        self.media_mode_ai = media_mode_ai
        self.media_mode_picsum = media_mode_picsum
        self._thumbnail_in_flight: Set[str] = set()
        self._thumbnail_in_flight_lock = threading.Lock()

    def generate_placeholder_thumbnail(self, label: str):
        background = self.Image.new("RGB", self.dashboard_thumbnail_size, (32, 34, 46))
        draw = self.ImageDraw.Draw(background)
        font = self.ImageFont.load_default()
        text = (label or "").strip() or "No Preview"
        text = text.upper()
        bbox = None
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
        except AttributeError:
            if hasattr(font, "getbbox"):
                bbox = font.getbbox(text)
        if bbox:
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        else:
            text_width, text_height = font.getsize(text)  # type: ignore[attr-defined]
        x = max((self.dashboard_thumbnail_size[0] - text_width) // 2, 4)
        y = max((self.dashboard_thumbnail_size[1] - text_height) // 2, 4)
        draw.text((x, y), text, fill=(210, 210, 210), font=font)
        return background

    def compose_thumbnail(self, frame):
        background = self.Image.new("RGB", self.dashboard_thumbnail_size, (20, 20, 24))
        prepared = frame.convert("RGB")
        prepared.thumbnail(self.dashboard_thumbnail_size, self.image_thumbnail_filter)
        offset_x = max((self.dashboard_thumbnail_size[0] - prepared.width) // 2, 0)
        offset_y = max((self.dashboard_thumbnail_size[1] - prepared.height) // 2, 0)
        background.paste(prepared, (offset_x, offset_y))
        return background

    def create_thumbnail_image(self, media_path: Path):
        try:
            with self.Image.open(media_path) as src:
                if getattr(src, "is_animated", False):
                    try:
                        src.seek(0)
                    except EOFError:
                        pass
                return self.compose_thumbnail(src)
        except Exception as exc:
            self.logger.debug("Failed to render image thumbnail for %s: %s", media_path, exc)
        return self.generate_placeholder_thumbnail("Image")

    def create_video_thumbnail(self, media_path: Path):
        if self.cv2 is None:
            return None
        capture = self.cv2.VideoCapture(str(media_path))
        if not capture.isOpened():
            return None
        try:
            frame_count = capture.get(self.cv2.CAP_PROP_FRAME_COUNT) or 0
            target_frame: Optional[int] = None
            if frame_count and frame_count > 0:
                start = int(max(0, frame_count * 0.15))
                end = int(max(start + 1, frame_count * 0.85))
                if end <= start:
                    end = start + 1
                try:
                    target_frame = random.randint(start, max(start + 1, end - 1))
                    capture.set(self.cv2.CAP_PROP_POS_FRAMES, float(target_frame))
                except Exception:
                    target_frame = None
            if target_frame is None:
                try:
                    duration_ms = capture.get(self.cv2.CAP_PROP_POS_MSEC) or 0
                    if duration_ms > 0:
                        capture.set(self.cv2.CAP_PROP_POS_MSEC, duration_ms * 0.5)
                except Exception:
                    pass
            success, frame = capture.read()
            if (not success or frame is None) and frame_count and frame_count > 0:
                try:
                    capture.set(self.cv2.CAP_PROP_POS_FRAMES, max(0.0, float(frame_count) * 0.5))
                    success, frame = capture.read()
                except Exception:
                    pass
        finally:
            capture.release()
        if not success or frame is None:
            return None
        try:
            rgb_frame = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
        except Exception:
            rgb_frame = frame[:, :, ::-1]
        image = self.Image.fromarray(rgb_frame)
        return self.compose_thumbnail(image)

    def load_remote_image(self, url: str):
        if not url or self.requests is None:
            return None
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        }
        try:
            resp = self.requests.get(url, headers=headers, timeout=6)
            if resp.status_code != 200:
                return None
            content_type = resp.headers.get("Content-Type", "").lower()
            if content_type and "image" not in content_type:
                return None
            buffer = io.BytesIO(resp.content)
            image = self.Image.open(buffer)
            image.load()
            return image
        except Exception as exc:
            self.logger.debug("Remote thumbnail fetch failed for %s: %s", url, exc)
            return None

    def create_livestream_thumbnail(self, stream_url: Optional[str]):
        if not stream_url:
            return None
        url = stream_url.strip()
        if not url:
            return None
        lower = url.lower()
        remote_image = None
        if "youtube.com" in lower or "youtu.be/" in lower:
            video_id = None
            if "watch?v=" in url:
                video_id = url.split("watch?v=")[1].split("&")[0].split("#")[0]
            elif "youtu.be/" in url:
                video_id = url.split("youtu.be/")[1].split("?")[0].split("&")[0]
            if video_id:
                for variant in ("maxresdefault", "sddefault", "hqdefault", "mqdefault", "default"):
                    remote_image = self.load_remote_image(f"https://img.youtube.com/vi/{video_id}/{variant}.jpg")
                    if remote_image:
                        break
        elif "twitch.tv" in lower:
            try:
                channel = url.split("twitch.tv/")[1].split("/")[0]
            except Exception:
                channel = ""
            if channel:
                remote_image = self.load_remote_image(
                    f"https://static-cdn.jtvnw.net/previews-ttv/live_user_{channel}-192x108.jpg"
                )
        if remote_image is None:
            remote_image = self.load_remote_image(url)
        if remote_image is None:
            return None
        try:
            return self.compose_thumbnail(remote_image)
        except Exception as exc:
            self.logger.debug("Livestream thumbnail compose failed for %s: %s", stream_url, exc)
            return None

    def render_thumbnail_image(self, snapshot: Dict[str, Any]):
        kind = snapshot.get("kind")
        path = snapshot.get("path")
        badge = snapshot.get("badge") or None
        image_obj = None
        placeholder = False
        if kind == "image" and path:
            if isinstance(path, str) and path.startswith(("http://", "https://")):
                remote_image = self.load_remote_image(path)
                if remote_image is not None:
                    try:
                        image_obj = self.compose_thumbnail(remote_image)
                    except Exception as exc:
                        self.logger.debug("Remote thumbnail compose failed for %s: %s", path, exc)
                        image_obj = None
            else:
                media_path = self.resolve_media_path(path)
                if media_path is not None:
                    image_obj = self.create_thumbnail_image(media_path)
        elif kind == "video" and path:
            media_path = self.resolve_media_path(path)
            if media_path is not None:
                image_obj = self.create_video_thumbnail(media_path)
        elif kind == "livestream":
            image_obj = self.create_livestream_thumbnail(snapshot.get("stream_url"))
        if image_obj is None:
            placeholder = True
            if kind == "video":
                badge_text = badge or "Video"
            elif kind == "livestream":
                badge_text = badge or "Live"
            else:
                badge_text = badge or "Image"
            image_obj = self.generate_placeholder_thumbnail(badge_text)
        return image_obj, placeholder

    def thumbnail_image_to_bytes(self, image) -> io.BytesIO:
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=self.thumbnail_jpeg_quality, optimize=True)
        buffer.seek(0)
        return buffer

    def compute_thumbnail_snapshot(self, stream_id: str) -> Optional[Dict[str, Any]]:
        conf = self.settings.get(stream_id)
        if not isinstance(conf, dict):
            return None
        runtime = self.get_stream_runtime_state(stream_id)
        media_mode = runtime.get("media_mode")
        if media_mode not in self.media_mode_choices:
            media_mode_value = conf.get("media_mode")
            if isinstance(media_mode_value, str):
                candidate = media_mode_value.strip().lower()
                media_mode = candidate if candidate in self.media_mode_choices else None
            if media_mode not in self.media_mode_choices:
                media_mode = self.infer_media_mode(conf)
        path = runtime.get("path")
        if path is None:
            path = conf.get("selected_image")
        stream_url = runtime.get("stream_url")
        if stream_url is None:
            stream_url = conf.get("stream_url")
        timestamp = runtime.get("timestamp") or time.time()
        kind = runtime.get("kind")
        if not kind and path:
            kind = self.detect_media_kind(path)
        if media_mode == self.media_mode_livestream:
            kind = "livestream"
        elif media_mode == self.media_mode_video:
            kind = "video"
        elif media_mode in {self.media_mode_ai, self.media_mode_picsum}:
            kind = "image"
        else:
            kind = kind or "image"
        placeholder = False
        if not path and media_mode != self.media_mode_livestream:
            placeholder = True
        badge_map = {
            self.media_mode_livestream: "Live",
            self.media_mode_video: "Video",
            self.media_mode_ai: "AI",
            self.media_mode_picsum: "Picsum",
        }
        badge = badge_map.get(media_mode, "Image")
        return {
            "stream_id": stream_id,
            "media_mode": media_mode,
            "path": path,
            "kind": kind,
            "stream_url": stream_url,
            "timestamp": timestamp,
            "badge": badge,
            "placeholder": placeholder,
            "source": runtime.get("source"),
        }

    def thumbnail_signature(self, snapshot: Dict[str, Any]) -> Tuple[Any, ...]:
        return (
            snapshot.get("media_mode"),
            snapshot.get("kind"),
            snapshot.get("path"),
            snapshot.get("stream_url"),
        )

    def refresh_stream_thumbnail(self, stream_id: str, snapshot: Optional[Dict[str, Any]] = None, *, force: bool = False):
        info = snapshot if snapshot is not None else self.compute_thumbnail_snapshot(stream_id)
        if info is None:
            return None
        signature = self.thumbnail_signature(info)
        with self.stream_runtime_lock:
            entry = self.stream_runtime_state.setdefault(stream_id, {})
            existing = entry.get("thumbnail")
            if not force and isinstance(existing, dict) and existing.get("_signature") == signature:
                return self.public_thumbnail_payload(existing)

        placeholder_payload = {
            "url": self.thumbnail_public_url(stream_id) if existing else None,
            "placeholder": info.get("placeholder", "Loading..."),
            "badge": info.get("badge"),
            "updated_at": existing.get("updated_at") if isinstance(existing, dict) else None,
        }

        with self._thumbnail_in_flight_lock:
            if stream_id in self._thumbnail_in_flight:
                return placeholder_payload
            self._thumbnail_in_flight.add(stream_id)

        def _async_generate_thumbnail():
            try:
                image_obj, placeholder = self.render_thumbnail_image(info)
                if image_obj is None:
                    return
                buffer = self.thumbnail_image_to_bytes(image_obj)
                binary = buffer.getvalue()
                updated_ts = time.time()
                record: Dict[str, Any] = {
                    "url": None,
                    "placeholder": placeholder,
                    "badge": info.get("badge"),
                    "updated_at": self.runtime_timestamp_to_iso(updated_ts),
                    "_signature": signature,
                    "_updated_ts": updated_ts,
                }
                cache_dir = self.ensure_thumbnail_dir()
                if cache_dir is not None:
                    target_path = self.thumbnail_disk_path(stream_id)
                    temp_path = target_path.with_suffix(".tmp")
                    try:
                        with open(temp_path, "wb") as fh:
                            fh.write(binary)
                        os.replace(temp_path, target_path)
                        record["url"] = self.thumbnail_public_url(stream_id)
                        with self.stream_runtime_lock:
                            self.stream_runtime_state.setdefault(stream_id, {})["thumbnail"] = record
                            playback_manager = self.playback_manager_getter()
                            if playback_manager is not None:
                                state = playback_manager._states.get(stream_id)
                                if state is not None:
                                    state._thumbnail_cache = self.public_thumbnail_payload(record)
                        self.safe_emit(
                            "thumbnail_update",
                            {
                                "stream": stream_id,
                                "url": record["url"],
                                "placeholder": placeholder,
                                "badge": info.get("badge"),
                                "updated_at": record["updated_at"],
                            },
                        )
                    except OSError as exc:
                        self.logger.debug("Failed to persist thumbnail for %s: %s", stream_id, exc)
            finally:
                with self._thumbnail_in_flight_lock:
                    self._thumbnail_in_flight.discard(stream_id)

        self.eventlet.spawn(_async_generate_thumbnail)
        return placeholder_payload
