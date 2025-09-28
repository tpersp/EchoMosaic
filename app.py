from flask import Flask, jsonify, send_from_directory, request, render_template
from flask_socketio import SocketIO
import os
import json
import subprocess
import time

app = Flask(__name__)
socketio = SocketIO(app)

SETTINGS_FILE = "settings.json"
IMAGE_DIR = "/mnt/viewers"  # Adjust if needed


def default_stream_config():
    """Return the default configuration for a new stream."""
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


def load_settings():
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, "r") as f:
            return json.load(f)
    # Start with no streams; dashboard can add them dynamically.
    return {}

def save_settings(data):
    with open(SETTINGS_FILE, "w") as f:
        json.dump(data, f, indent=4)

settings = load_settings()

def get_subfolders():
    subfolders = ["all"]
    if os.path.isdir(IMAGE_DIR):
        for root, dirs, _ in os.walk(IMAGE_DIR):
            for d in dirs:
                subfolders.append(os.path.relpath(os.path.join(root, d), IMAGE_DIR))
            break
    return subfolders

def list_images(folder="all"):
    """
    List all images (including .webp) in the chosen folder (or 'all'),
    then sort them alphabetically.
    """
    exts = (".png", ".jpg", ".jpeg", ".gif", ".webp")
    target_dir = IMAGE_DIR if folder == "all" else os.path.join(IMAGE_DIR, folder)
    if not os.path.exists(target_dir):
        return []
    images = []
    for root, _, files in os.walk(target_dir):
        for file in files:
            if file.lower().endswith(exts):
                relative_path = os.path.relpath(os.path.join(root, file), IMAGE_DIR)
                relative_path = relative_path.replace("\\", "/")
                images.append(relative_path)
    # Sort them alphabetically
    images.sort(key=str.lower)
    return images

def try_get_hls(original_url):
    if not original_url:
        return None
    try:
        result = subprocess.run(
            ["yt-dlp", "-g", original_url],
            capture_output=True,
            text=True,
            check=True
        )
        raw_url = result.stdout.strip()
        if any(ext in raw_url for ext in [".m3u8", ".mpd"]):
            return raw_url
        return None
    except subprocess.CalledProcessError:
        return None

@app.route("/")
def dashboard():
    subfolders = get_subfolders()
    return render_template("index.html", subfolders=subfolders, stream_settings=settings)

@app.route("/stream/<stream_id>")
def render_stream(stream_id):
    if stream_id not in settings:
        return f"No stream '{stream_id}'", 404
    conf = settings[stream_id]
    images = list_images(conf.get("folder", "all"))
    return render_template("single_stream.html", stream_id=stream_id, config=conf, images=images)


@app.route("/streams", methods=["POST"])
def add_stream():
    """Create a new stream configuration and return its ID."""
    idx = 1
    while True:
        new_id = f"stream{idx}"
        if new_id not in settings:
            settings[new_id] = default_stream_config()
            save_settings(settings)
            socketio.emit("streams_changed", {"action": "added", "stream_id": new_id})
            return jsonify({"stream_id": new_id})
        idx += 1


@app.route("/streams/<stream_id>", methods=["DELETE"])
def delete_stream(stream_id):
    if stream_id in settings:
        settings.pop(stream_id)
        save_settings(settings)
        socketio.emit("streams_changed", {"action": "deleted", "stream_id": stream_id})
        return jsonify({"status": "deleted"})
    return jsonify({"error": "not found"}), 404

@app.route("/get-settings/<stream_id>", methods=["GET"])
def get_stream_settings(stream_id):
    if stream_id not in settings:
        return jsonify({"error": f"No stream '{stream_id}' found"}), 404
    return jsonify(settings[stream_id])

@app.route("/settings/<stream_id>", methods=["POST"])
def update_stream_settings(stream_id):
    if stream_id not in settings:
        return jsonify({"error": f"No stream '{stream_id}' found"}), 404

    data = request.json
    conf = settings[stream_id]

    # We'll add new keys for YouTube: "yt_cc", "yt_mute", "yt_quality"
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
def serve_stream_image(filename):
    full_path = os.path.join(IMAGE_DIR, filename)
    if not os.path.exists(full_path):
        return jsonify({"error": f"File {filename} not found"}), 404
    return send_from_directory(IMAGE_DIR, filename)

@app.route("/stream/live")
def stream_live():
    stream_id = request.args.get("stream_id", "").strip()
    if stream_id not in settings:
        return jsonify({"error": f"No stream '{stream_id}' found"}), 404

    stream_url = settings[stream_id].get("stream_url", "")
    if not stream_url:
        return jsonify({"error": "No live stream URL configured"}), 404

    # 1) Check YouTube
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
            "original_url": stream_url
        })

    # 2) Check Twitch
    if "twitch.tv" in stream_url:
        embed_id = stream_url.split("twitch.tv/")[1].split("/")[0]
        return jsonify({
            "embed_type": "twitch",
            "embed_id": embed_id,
            "hls_url": None,
            "original_url": stream_url
        })

    # 3) Attempt HLS
    hls_link = try_get_hls(stream_url)
    if hls_link:
        return jsonify({
            "embed_type": "hls",
            "embed_id": None,
            "hls_url": hls_link,
            "original_url": stream_url
        })

    # 4) fallback
    return jsonify({
        "embed_type": "iframe",
        "embed_id": None,
        "hls_url": None,
        "original_url": stream_url
    })

@app.route("/images", methods=["GET"])
def get_images():
    folder = request.args.get("folder", "all")
    imgs = list_images(folder)
    return jsonify(imgs)

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
