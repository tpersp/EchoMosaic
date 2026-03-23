"""Asset delivery and thumbnail route helpers."""

from __future__ import annotations

import base64
import errno
import hashlib
import io
import os
import time
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union


class AssetDeliveryService:
    def __init__(
        self,
        *,
        send_file,
        jsonify,
        url_for,
        request_args_get,
        parse_truthy,
        as_int,
        generate_etag,
        logger,
        bad_media_log_cache,
        bad_media_log_ttl: float,
        image_extensions,
        video_extensions,
        max_image_dimension: int,
        thumbnail_size_presets,
        thumbnail_subdir: str,
        thumbnail_jpeg_quality: int,
        image_thumbnail_filter,
        image_cache_timeout: int,
        image_cache_control_max_age: int,
        media_root_lookup,
        split_virtual_media_path,
        resolve_virtual_media_path,
        ensure_thumbnail_dir,
        thumbnail_disk_path,
        thumbnail_public_url,
        public_thumbnail_payload,
        compute_thumbnail_snapshot,
        refresh_stream_thumbnail,
        get_runtime_thumbnail_payload,
        runtime_timestamp_to_iso,
        render_thumbnail_image,
        thumbnail_image_to_bytes,
        resized_image_locks,
        resized_image_locks_guard,
        stream_runtime_lock,
        stream_runtime_state,
        Image,
        ImageOps,
    ) -> None:
        self.send_file = send_file
        self.jsonify = jsonify
        self.url_for = url_for
        self.request_args_get = request_args_get
        self.parse_truthy = parse_truthy
        self.as_int = as_int
        self.generate_etag = generate_etag
        self.logger = logger
        self.bad_media_log_cache = bad_media_log_cache
        self.bad_media_log_ttl = bad_media_log_ttl
        self.image_extensions = image_extensions
        self.video_extensions = video_extensions
        self.max_image_dimension = max_image_dimension
        self.thumbnail_size_presets = thumbnail_size_presets
        self.thumbnail_subdir = thumbnail_subdir
        self.thumbnail_jpeg_quality = thumbnail_jpeg_quality
        self.image_thumbnail_filter = image_thumbnail_filter
        self.image_cache_timeout = image_cache_timeout
        self.image_cache_control_max_age = image_cache_control_max_age
        self.media_root_lookup = media_root_lookup
        self.split_virtual_media_path = split_virtual_media_path
        self.resolve_virtual_media_path = resolve_virtual_media_path
        self.ensure_thumbnail_dir = ensure_thumbnail_dir
        self.thumbnail_disk_path = thumbnail_disk_path
        self.thumbnail_public_url = thumbnail_public_url
        self.public_thumbnail_payload = public_thumbnail_payload
        self.compute_thumbnail_snapshot = compute_thumbnail_snapshot
        self.refresh_stream_thumbnail = refresh_stream_thumbnail
        self.get_runtime_thumbnail_payload = get_runtime_thumbnail_payload
        self.runtime_timestamp_to_iso = runtime_timestamp_to_iso
        self.render_thumbnail_image = render_thumbnail_image
        self.thumbnail_image_to_bytes = thumbnail_image_to_bytes
        self.resized_image_locks = resized_image_locks
        self.resized_image_locks_guard = resized_image_locks_guard
        self.stream_runtime_lock = stream_runtime_lock
        self.stream_runtime_state = stream_runtime_state
        self.Image = Image
        self.ImageOps = ImageOps

    def acquire_resized_image_lock(self, path: Path) -> threading.Lock:
        with self.resized_image_locks_guard:
            lock = self.resized_image_locks.get(path)
            if lock is None:
                lock = threading.Lock()
                self.resized_image_locks[path] = lock
            return lock

    def parse_image_resize_request(self) -> Optional[Tuple[int, int]]:
        size_key = (self.request_args_get("size") or "").strip().lower()
        if size_key:
            if size_key == "full":
                return None
            preset = self.thumbnail_size_presets.get(size_key)
            if preset:
                return preset
        width = self.as_int(self.request_args_get("width"), 0)
        height = self.as_int(self.request_args_get("height"), 0)
        if width <= 0 and height <= 0:
            return None
        if width <= 0:
            width = height
        if height <= 0:
            height = width
        width = min(max(1, width), self.max_image_dimension)
        height = min(max(1, height), self.max_image_dimension)
        return (width, height)

    def get_resized_image_path(
        self,
        virtual_path: str,
        source_path: Path,
        bounds: Tuple[int, int],
    ) -> Optional[Path]:
        suffix = source_path.suffix.lower()
        if suffix not in self.image_extensions or suffix == ".gif":
            return None
        alias, relative = self.split_virtual_media_path(virtual_path)
        root = self.media_root_lookup.get(alias)
        if root is None:
            return None
        try:
            stat_info = source_path.stat()
        except OSError:
            return None
        key_source = f"{alias}:{relative}:{bounds[0]}x{bounds[1]}:{stat_info.st_mtime_ns}"
        cache_key = hashlib.sha1(key_source.encode("utf-8")).hexdigest()
        cache_dir = root.path / self.thumbnail_subdir
        cache_path = cache_dir / f"{cache_key}.jpg"
        if cache_path.exists():
            return cache_path
        lock = self.acquire_resized_image_lock(cache_path)
        with lock:
            if cache_path.exists():
                return cache_path
            try:
                with self.Image.open(source_path) as img:
                    img = self.ImageOps.exif_transpose(img)
                    target_width = min(bounds[0], img.width)
                    target_height = min(bounds[1], img.height)
                    if target_width >= img.width and target_height >= img.height:
                        return source_path
                    img.thumbnail((target_width, target_height), self.image_thumbnail_filter)
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    img.save(cache_path, "JPEG", quality=self.thumbnail_jpeg_quality)
            except Exception as exc:
                self.logger.debug("Failed to resize image %s: %s", source_path, exc)
                return None
        return cache_path

    def log_bad_media_once(self, path: str, exc: BaseException) -> None:
        now = time.time()
        last_log = self.bad_media_log_cache.get(path, 0.0)
        if now - last_log >= self.bad_media_log_ttl:
            self.logger.warning("Media unavailable path=%s exc=%r", path, exc)
            self.bad_media_log_cache[path] = now

    def media_unavailable_response(self, path: str):
        return self.jsonify(
            {
                "error": "media_unavailable",
                "path": path,
                "detail": "backend_read_error",
            }
        ), 404

    def send_image_response(self, path: Union[str, Path]):
        abs_path = os.fspath(path)
        try:
            stat = os.stat(abs_path)
            etag_source = f"{stat.st_mtime_ns}-{stat.st_size}".encode("utf-8")
            etag_value = self.generate_etag(etag_source)
            response = self.send_file(
                abs_path,
                conditional=True,
                max_age=self.image_cache_timeout,
                etag=etag_value,
            )
        except (FileNotFoundError, PermissionError) as exc:
            self.log_bad_media_once(abs_path, exc)
            return self.media_unavailable_response(abs_path)
        except OSError as exc:
            if getattr(exc, "errno", None) == errno.EINVAL:
                self.log_bad_media_once(abs_path, exc)
                return self.media_unavailable_response(abs_path)
            raise
        response.headers["Cache-Control"] = f"public, max-age={self.image_cache_control_max_age}"
        return response

    def send_video_response(self, path: Union[str, Path]):
        abs_path = os.fspath(path)
        try:
            response = self.send_file(abs_path, conditional=True)
        except (FileNotFoundError, PermissionError) as exc:
            self.log_bad_media_once(abs_path, exc)
            return self.media_unavailable_response(abs_path)
        except OSError as exc:
            if getattr(exc, "errno", None) == errno.EINVAL:
                self.log_bad_media_once(abs_path, exc)
                return self.media_unavailable_response(abs_path)
            raise
        response.headers.setdefault("Cache-Control", f"public, max-age={self.image_cache_timeout}")
        response.headers.setdefault("Accept-Ranges", "bytes")
        return response

    def serve_image(self, image_path: str):
        full_path = self.resolve_virtual_media_path(image_path)
        if full_path is None or not full_path.exists() or not full_path.is_file():
            return "Not found", 404
        bounds = self.parse_image_resize_request()
        if bounds:
            resized = self.get_resized_image_path(image_path, full_path, bounds)
            if resized is not None and resized != full_path:
                return self.send_image_response(resized)
        return self.send_image_response(full_path)

    def serve_video(self, video_path: str):
        target_path = self.resolve_virtual_media_path(video_path)
        if target_path is None:
            return "Invalid path", 400
        if not target_path.exists() or not target_path.is_file():
            return "Not found", 404
        if target_path.suffix.lower() not in self.video_extensions:
            return "Unsupported media type", 415
        return self.send_video_response(target_path)

    def stream_thumbnail_metadata(self, stream_id: str):
        info = self.compute_thumbnail_snapshot(stream_id)
        if info is None:
            return self.jsonify({"error": f"No stream '{stream_id}' found"}), 404
        force_refresh = self.parse_truthy(self.request_args_get("force"))
        if force_refresh:
            thumbnail_info = self.refresh_stream_thumbnail(stream_id, info, force=True)
        else:
            thumbnail_info = self.get_runtime_thumbnail_payload(stream_id)
            if thumbnail_info is None:
                thumbnail_info = self.refresh_stream_thumbnail(stream_id, info)
        timestamp = info.get("timestamp")
        cache_key = None
        if isinstance(timestamp, (int, float)):
            try:
                cache_key = str(int(timestamp))
            except (TypeError, ValueError):
                cache_key = None
        raw_url = thumbnail_info.get("url") if isinstance(thumbnail_info, dict) else None
        if raw_url and raw_url.startswith("data:"):
            image_url = raw_url
        else:
            image_url = raw_url or self.url_for("stream_thumbnail_image", stream_id=stream_id)
            if cache_key:
                image_url = f"{image_url}?v={cache_key}"
        payload = {
            "stream_id": stream_id,
            "media_mode": info.get("media_mode"),
            "kind": info.get("kind"),
            "path": info.get("path"),
            "image_url": image_url,
            "thumbnail": thumbnail_info,
            "placeholder": bool(info.get("placeholder")),
            "badge": info.get("badge"),
            "updated_at": self.runtime_timestamp_to_iso(info.get("timestamp")),
            "source": info.get("source"),
        }
        stream_url = info.get("stream_url")
        if stream_url:
            payload["stream_url"] = stream_url
        return self.jsonify(payload)

    def stream_thumbnail_image(self, stream_id: str):
        info = self.compute_thumbnail_snapshot(stream_id)
        if info is None:
            return "Not found", 404
        self.refresh_stream_thumbnail(stream_id, info)
        target_path = self.thumbnail_disk_path(stream_id)
        raw_record: Optional[Dict[str, Any]] = None
        with self.stream_runtime_lock:
            entry = self.stream_runtime_state.get(stream_id)
            if entry and isinstance(entry.get("thumbnail"), dict):
                raw_record = dict(entry["thumbnail"])
        if target_path.exists():
            response = self.send_file(str(target_path), mimetype="image/jpeg")
        else:
            data_url = raw_record.get("_data_url") if isinstance(raw_record, dict) else None
            if isinstance(data_url, str) and "," in data_url:
                encoded = data_url.split(",", 1)[1]
                try:
                    binary = base64.b64decode(encoded)
                except Exception:
                    binary = b""
                response = self.send_file(io.BytesIO(binary), mimetype="image/jpeg")
            else:
                image_obj, _ = self.render_thumbnail_image(info)
                response = self.send_file(self.thumbnail_image_to_bytes(image_obj), mimetype="image/jpeg")
        response.headers["Cache-Control"] = "no-store, max-age=0"
        return response

    def cached_stream_thumbnail(self, stream_id: str):
        return self.stream_thumbnail_image(stream_id)
