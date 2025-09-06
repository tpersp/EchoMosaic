from flask import Flask, jsonify, send_from_directory, request, render_template
from flask_socketio import SocketIO, emit
import os
import json
import subprocess
import time

app = Flask(__name__)
socketio = SocketIO(app)

SETTINGS_FILE = "settings.json"
IMAGE_DIR = "/mnt/viewers"  # Adjust if needed

def load_settings():
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, "r") as f:
            return json.load(f)
    # Default structure: we also add placeholders for yt_cc, yt_mute, yt_quality
    return {
        "stream1": {
            "mode": "random",
            "folder": "all",
            "selected_image": None,
            "duration": 5,
            "stream_url": None,
            # Additional defaults for YouTube
            "yt_cc": False,
            "yt_mute": True,
            "yt_quality": "auto"
        },
        "stream2": {
            "mode": "random",
            "folder": "all",
            "selected_image": None,
            "duration": 5,
            "stream_url": None,
            "yt_cc": False,
            "yt_mute": True,
            "yt_quality": "auto"
        },
        "stream3": {
            "mode": "random",
            "folder": "all",
            "selected_image": None,
            "duration": 5,
            "stream_url": None,
            "yt_cc": False,
            "yt_mute": True,
            "yt_quality": "auto"
        },
        "stream4": {
            "mode": "random",
            "folder": "all",
            "selected_image": None,
            "duration": 5,
            "stream_url": None,
            "yt_cc": False,
            "yt_mute": True,
            "yt_quality": "auto"
        }
    }

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

@app.route("/stream")
def main_stream():
    return render_template("main_stream.html")

@app.route("/stream1")
def stream1():
    conf = settings["stream1"]
    images = list_images(conf.get("folder", "all"))
    return render_template("single_stream.html", stream_id="stream1", config=conf, images=images)

@app.route("/stream2")
def stream2():
    conf = settings["stream2"]
    images = list_images(conf.get("folder", "all"))
    return render_template("single_stream.html", stream_id="stream2", config=conf, images=images)

@app.route("/stream3")
def stream3():
    conf = settings["stream3"]
    images = list_images(conf.get("folder", "all"))
    return render_template("single_stream.html", stream_id="stream3", config=conf, images=images)

@app.route("/stream4")
def stream4():
    conf = settings["stream4"]
    images = list_images(conf.get("folder", "all"))
    return render_template("single_stream.html", stream_id="stream4", config=conf, images=images)

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
