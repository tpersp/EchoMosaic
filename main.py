"""
Dynamic EchoMosaic server.

This module implements a Flask/Socket.IO server that allows users to
create and remove arbitrary numbers of streams at runtime. Each stream
can operate in random image rotation, specific image selection, or
livestream modes. Stream configuration is persisted to disk so that
settings survive restarts. The server also respects an optional
``ECHO_IMAGE_DIR`` environment variable to locate images.

To run the server locally for development:

.. code-block:: bash

    PORT=5000 ECHO_IMAGE_DIR=/path/to/images python main.py

For production the accompanying ``install_network.sh`` script sets up a
systemd unit that launches this application via gunicorn.
"""

from flask import Flask, jsonify, send_from_directory, request, render_template
from flask_socketio import SocketIO, emit
import os
import json
import subprocess

app = Flask(__name__)
socketio = SocketIO(app)

# JSON file used to persist stream configuration
SETTINGS_FILE = "settings.json"
# Directory containing images. Override via the ECHO_IMAGE_DIR environment variable.
IMAGE_DIR = os.environ.get("ECHO_IMAGE_DIR", "/mnt/piviewers")


def default_stream_config() -> dict:
    """Return a fresh default configuration for a new stream."""
    return {
        "mode": "random",
        "folder": "all",
        "selected_image": None,
        "duration": 5,
        "stream_url": None,
        "yt_cc": False,
        "yt_mute": True,
        "yt_quality": "auto",
    }


def load_settings() -> dict:
    """Load settings from a JSON file or initialise with a default stream."""
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, "r") as f:
            data = json.load(f)
        # Ensure each stream config has all required keys
        for conf in data.values():
            for key, val in default_stream_config().items():
                conf.setdefault(key, val)
        return data
    # If no file exists, create a single default stream
    return {"stream1": default_stream_config()}


def save_settings(data: dict) -> None:
    """Write the provided settings dictionary to disk as JSON."""
    with open(SETTINGS_FILE, "w") as f:
        json.dump(data, f, indent=4)


# Global settings loaded at startup
settings = load_settings()


def get_subfolders() -> list:
    """Return a list of subfolders under IMAGE_DIR (plus 'all')."""
    subfolders = ["all"]
    if os.path.isdir(IMAGE_DIR):
        for root, dirs, _ in os.walk(IMAGE_DIR):
            for d in dirs:
                subfolders.append(os.path.relpath(os.path.join(root, d), IMAGE_DIR))
            # Only top‑level directories
            break
    return subfolders


def list_images(folder: str = "all") -> list:
    """Return a sorted list of images from the given folder or all."""
    exts = (".png", ".jpg", ".jpeg", ".gif", ".webp")
    target_dir = IMAGE_DIR if folder == "all" else os.path.join(IMAGE_DIR, folder)
    if not os.path.exists(target_dir):
        return []
    images: list[str] = []
    for root, _, files in os.walk(target_dir):
        for file in files:
            if file.lower().endswith(exts):
                rel = os.path.relpath(os.path.join(root, file), IMAGE_DIR)
                images.append(rel.replace("\\", "/"))
    images.sort(key=str.lower)
    return images


def try_get_hls(original_url: str) -> str | None:
    """
    Attempt to extract an HLS or DASH URL from a livestream using yt-dlp.
    Returns the URL if successful, otherwise ``None``.
    """
    if not original_url:
        return None
    try:
        result = subprocess.run([
            "yt-dlp", "-g", original_url
        ], capture_output=True, text=True, check=True)
        raw = result.stdout.strip()
        if any(ext in raw for ext in [".m3u8", ".mpd"]):
            return raw
    except subprocess.CalledProcessError:
        pass
    return None


@app.route("/")
def dashboard():
    """Render the main dashboard with a card for each stream."""
    subfolders = get_subfolders()
    return render_template("index.html", subfolders=subfolders, stream_settings=settings)


@app.route("/stream/<stream_id>")
def stream_page(stream_id: str):
    """Render a single stream page for the given stream."""
    if stream_id not in settings:
        return "Stream not found", 404
    conf = settings[stream_id]
    images = list_images(conf.get("folder", "all"))
    return render_template("single_stream.html", stream_id=stream_id, config=conf, images=images)


@app.route("/streams", methods=["GET", "POST"])
def manage_streams():
    """List all streams or create a new one."""
    global settings
    if request.method == "GET":
        return jsonify(list(settings.keys()))
    # POST: create a new stream
    idx = 1
    while f"stream{idx}" in settings:
        idx += 1
    new_id = f"stream{idx}"
    settings[new_id] = default_stream_config()
    save_settings(settings)
    socketio.emit("streams_changed", {"action": "added", "stream_id": new_id})
    return jsonify({"stream_id": new_id})


@app.route("/streams/<stream_id>", methods=["DELETE"])
def delete_stream(stream_id: str):
    """Delete the specified stream."""
    global settings
    if stream_id not in settings:
        return jsonify({"error": "not found"}), 404
    settings.pop(stream_id)
    save_settings(settings)
    socketio.emit("streams_changed", {"action": "removed", "stream_id": stream_id})
    return jsonify({"status": "deleted"})


@app.route("/get-settings/<stream_id>", methods=["GET"])
def get_stream_settings(stream_id: str):
    """Return the configuration for the requested stream."""
    if stream_id not in settings:
        return jsonify({"error": f"No stream '{stream_id}' found"}), 404
    return jsonify(settings[stream_id])


@app.route("/settings/<stream_id>", methods=["POST"])
def update_stream_settings(stream_id: str):
    """Update a stream's configuration and notify connected clients."""
    if stream_id not in settings:
        return jsonify({"error": f"No stream '{stream_id}' found"}), 404
    data = request.json or {}
    conf = settings[stream_id]
    for key in ["mode", "folder", "selected_image", "duration", "stream_url",
                "yt_cc", "yt_mute", "yt_quality"]:
        if key in data:
            val = data[key]
            if key == "stream_url":
                val = val.strip()
                conf[key] = val if val and val.lower() != "none" else None
            else:
                conf[key] = val
    save_settings(settings)
    socketio.emit("refresh", {"stream_id": stream_id, "config": conf})
    return jsonify({"status": "success", "new_config": conf})


@app.route("/stream/image/<path:filename>")
def serve_stream_image(filename: str):
    """Serve an image from the configured image directory."""
    full_path = os.path.join(IMAGE_DIR, filename)
    if not os.path.exists(full_path):
        return jsonify({"error": f"File {filename} not found"}), 404
    return send_from_directory(IMAGE_DIR, filename)


@app.route("/stream/live")
def stream_live():
    """Return embed data for a livestream URL for YouTube, Twitch or HLS."""
    stream_id = request.args.get("stream_id", "").strip()
    if stream_id not in settings:
        return jsonify({"error": f"No stream '{stream_id}' found"}), 404
    stream_url = settings[stream_id].get("stream_url", "") or ""
    if not stream_url:
        return jsonify({"error": "No live stream URL configured"}), 404
    # YouTube embed
    if "youtube.com" in stream_url or "youtu.be" in stream_url:
        embed_id = None
        if "watch?v=" in stream_url:
            parts = stream_url.split("watch?v=")[1].split("&")[0].split("#")[0]
            embed_id = parts
        elif "youtu.be/" in stream_url:
            embed_id = stream_url.split("youtu.be/")[1].split("?")[0].split("&")[0]
        return jsonify({
            "embed_type": "youtube",
            "embed_id": embed_id,
            "hls_url": None,
            "original_url": stream_url,
        })
    # Twitch embed
    if "twitch.tv" in stream_url:
        embed_id = stream_url.split("twitch.tv/")[1].split("/")[0]
        return jsonify({
            "embed_type": "twitch",
            "embed_id": embed_id,
            "hls_url": None,
            "original_url": stream_url,
        })
    # HLS detection
    hls_url = try_get_hls(stream_url)
    if hls_url:
        return jsonify({
            "embed_type": "hls",
            "embed_id": None,
            "hls_url": hls_url,
            "original_url": stream_url,
        })
    # Fallback to iframe
    return jsonify({
        "embed_type": "iframe",
        "embed_id": None,
        "hls_url": None,
        "original_url": stream_url,
    })


@app.route("/images", methods=["GET"])
def get_images():
    """Return a list of images for the requested folder."""
    folder = request.args.get("folder", "all")
    return jsonify(list_images(folder))


if __name__ == "__main__":
    # When running directly we allow the port to be overridden via PORT env var
    port = int(os.environ.get("PORT", "5000"))
    socketio.run(app, host="0.0.0.0", port=port, debug=True)
