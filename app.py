from flask import Flask, jsonify, send_from_directory, request, render_template, redirect, url_for
from flask_socketio import SocketIO
import os
import json
import subprocess
import time
import shutil
from datetime import datetime
import re

app = Flask(__name__, static_folder="static", static_url_path="/static")
socketio = SocketIO(app)

SETTINGS_FILE = "settings.json"
CONFIG_FILE = "config.json"
IMAGE_DIR = "/mnt/viewers"  # Adjust if needed


def default_mosaic_config():
    """Return the default configuration for the mosaic /stream page."""
    # ``layout`` controls how streams are arranged. ``grid`` uses the
    # classic column based approach while other values enable custom
    # layouts (e.g. horizontal or vertical stacking).
    return {
        "cols": 2,
        "layout": "grid",
        "pip_main": None,
        "pip_pip": None,
        "pip_corner": "bottom-right",
        "pip_size": 25,
    }


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
        "label": "",
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


def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    return {}

settings = load_settings()
if "_mosaic" not in settings:
    settings["_mosaic"] = default_mosaic_config()
else:
    # Backwards compatibility for older settings files
    settings["_mosaic"].setdefault("layout", "grid")
    settings["_mosaic"].setdefault("cols", 2)
    settings["_mosaic"].setdefault("pip_main", None)
    settings["_mosaic"].setdefault("pip_pip", None)
    settings["_mosaic"].setdefault("pip_corner", "bottom-right")
    settings["_mosaic"].setdefault("pip_size", 25)

# Backfill defaults for existing stream entries
for k, v in list(settings.items()):
    if not k.startswith("_") and isinstance(v, dict):
        v.setdefault("label", k.capitalize())

# Ensure notes key exists
settings.setdefault("_notes", "")

# Ensure groups key exists
settings.setdefault("_groups", {})


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
    streams = {k: v for k, v in settings.items() if not k.startswith("_")}
    mosaic = settings.get("_mosaic", default_mosaic_config())
    groups = sorted(list(settings.get("_groups", {}).keys()))
    return render_template(
        "index.html",
        subfolders=subfolders,
        stream_settings=streams,
        mosaic_settings=mosaic,
        groups=groups,
    )


@app.route("/stream")
def mosaic_streams():
    # Dynamic global view: include all streams ("online" assumed as configured)
    streams = {k: v for k, v in settings.items() if not k.startswith("_")}
    mosaic = settings.get("_mosaic", default_mosaic_config())
    return render_template("streams.html", stream_settings=streams, mosaic_settings=mosaic)

def _slugify(name: str) -> str:
    name = (name or "").strip().lower()
    name = re.sub(r"[^a-z0-9]+", "-", name)
    name = re.sub(r"-+", "-", name).strip("-")
    return name or ""

@app.template_filter('slugify')
def jinja_slugify(s):
    return _slugify(s)

@app.route("/stream/<name>")
def render_stream(name):
    # Accept either stream id or slugified label
    key = None
    if name in settings:
        key = name
    else:
        wanted = _slugify(name)
        for sid, conf in settings.items():
            if sid.startswith("_"):
                continue
            label = conf.get("label") or sid
            if _slugify(label) == wanted:
                key = sid
                break
    if not key or key not in settings:
        return f"No stream '{name}'", 404
    conf = settings[key]
    images = list_images(conf.get("folder", "all"))
    return render_template("single_stream.html", stream_id=key, config=conf, images=images)


@app.route("/streams", methods=["POST"])
def add_stream():
    """Create a new stream configuration and return its ID."""
    idx = 1
    while True:
        new_id = f"stream{idx}"
        if new_id not in settings:
            settings[new_id] = default_stream_config()
            settings[new_id]["label"] = new_id.capitalize()
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
                "yt_cc", "yt_mute", "yt_quality", "label"]:
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


@app.route("/mosaic-settings", methods=["POST"])
def update_mosaic_settings():
    data = request.json or {}
    layout = data.get("layout", "grid")
    cols = int(data.get("cols", settings.get("_mosaic", {}).get("cols", 2)))
    mosaic = {"layout": layout, "cols": cols}
    if layout == "pip":
        mosaic.update({
            "pip_main": data.get("pip_main"),
            "pip_pip": data.get("pip_pip"),
            "pip_corner": data.get("pip_corner", "bottom-right"),
            "pip_size": int(data.get("pip_size", 25)),
        })
    settings["_mosaic"] = mosaic
    save_settings(settings)
    socketio.emit("mosaic_refresh", {"mosaic": settings["_mosaic"]})
    return jsonify({"status": "success", "mosaic": settings["_mosaic"]})

@app.route("/stream/image/<path:filename>")
def serve_stream_image(filename):
    full_path = os.path.join(IMAGE_DIR, filename)
    if not os.path.exists(full_path):
        return jsonify({"error": f"File {filename} not found"}), 404
    return send_from_directory(IMAGE_DIR, filename)


@app.route("/notes", methods=["GET", "POST"])
def notes_api():
    """Simple API to store and retrieve dashboard notes server-side."""
    if request.method == "GET":
        return jsonify({"text": settings.get("_notes", "")})
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")
    settings["_notes"] = text
    save_settings(settings)
    return jsonify({"status": "ok"})

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


@app.route("/settings")
def app_settings():
    cfg = load_config()
    return render_template("settings.html", config=cfg)


@app.route("/update_app", methods=["POST"])
def update_app():
    cfg = load_config()
    api_key = cfg.get("API_KEY")
    if api_key and request.headers.get("X-API-Key") != api_key:
        return "Unauthorized", 401
    repo_path = cfg.get("INSTALL_DIR") or os.getcwd()
    branch = cfg.get("UPDATE_BRANCH", "main")
    service_name = cfg.get("SERVICE_NAME", "echomosaic.service")
    if not os.path.isdir(repo_path):
        return render_template(
            "update_status.html",
            message=f"Repository path '{repo_path}' not found. Check INSTALL_DIR in config.json",
        )
    # capture current commit before update
    def git_cmd(args, cwd=repo_path):
        return subprocess.check_output(["git", *args], cwd=cwd, stderr=subprocess.STDOUT).decode().strip()
    try:
        current_commit = git_cmd(["rev-parse", "HEAD"]) 
    except Exception:
        current_commit = None
    try:
        subprocess.check_call(["git", "fetch"], cwd=repo_path)
        subprocess.check_call(["git", "checkout", branch], cwd=repo_path)
        subprocess.check_call(["git", "reset", "--hard", f"origin/{branch}"], cwd=repo_path)
    except FileNotFoundError:
        return render_template(
            "update_status.html",
            message="Git executable not found. Please install Git to update the application.",
        )
    except subprocess.CalledProcessError as e:
        return render_template("update_status.html", message=f"Git update failed: {e}")
    # record update history
    try:
        new_commit = git_cmd(["rev-parse", "HEAD"]) if 'git_cmd' in locals() else None
        history_path = os.path.join(repo_path, "update_history.json")
        history = []
        if os.path.exists(history_path):
            with open(history_path, "r") as hf:
                history = json.load(hf)
        history.append({
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "from": current_commit,
            "to": new_commit,
            "branch": branch,
        })
        with open(history_path, "w") as hf:
            json.dump(history[-50:], hf, indent=2)
    except Exception:
        pass
    try:
        subprocess.check_call([
            os.path.join(repo_path, "venv", "bin", "pip"),
            "install",
            "--upgrade",
            "-r",
            "requirements.txt",
        ], cwd=repo_path)
    except (subprocess.CalledProcessError, OSError):
        pass
    try:
        subprocess.Popen(["sudo", "systemctl", "restart", service_name])
    except OSError:
        pass
    return render_template(
        "update_status.html", message="Soft update complete. Restarting service..."
    )


def read_update_info():
    cfg = load_config()
    repo_path = cfg.get("INSTALL_DIR") or os.getcwd()
    branch = cfg.get("UPDATE_BRANCH", "main")
    info = {"branch": branch}
    def safe(cmd):
        try:
            return subprocess.check_output(cmd, cwd=repo_path, stderr=subprocess.STDOUT).decode().strip()
        except Exception:
            return None
    current = safe(["git", "rev-parse", "HEAD"]) or ""
    current_short = safe(["git", "rev-parse", "--short", "HEAD"]) or ""
    current_desc = safe(["git", "log", "-1", "--pretty=%h %s (%cr)"]) or current_short
    # fetch remote to learn about latest without changing local state
    _ = safe(["git", "fetch", "--quiet"])  # ignore errors silently
    remote = safe(["git", "rev-parse", f"origin/{branch}"]) or ""
    remote_short = remote[:7] if remote else ""
    remote_desc = safe(["git", "log", "-1", f"origin/{branch}", "--pretty=%h %s (%cr)"]) or remote_short
    info.update({
        "current_commit": current,
        "current_desc": current_desc,
        "remote_commit": remote,
        "remote_desc": remote_desc,
        "update_available": (current and remote and current != remote)
    })
    # previous commit from history
    history_path = os.path.join(repo_path, "update_history.json")
    if os.path.exists(history_path):
        try:
            with open(history_path, "r") as hf:
                history = json.load(hf)
            if history:
                last = history[-1]
                info["previous_commit"] = last.get("from")
                # resolve desc for previous
                prev = last.get("from")
                if prev:
                    prev_desc = safe(["git", "log", "-1", prev, "--pretty=%h %s (%cr)"]) or (prev[:7])
                else:
                    prev_desc = None
                info["previous_desc"] = prev_desc
        except Exception:
            pass
    return info


@app.route("/update_info", methods=["GET"])
def update_info():
    cfg = load_config()
    api_key = cfg.get("API_KEY")
    if api_key and request.headers.get("X-API-Key") != api_key:
        return jsonify({"error": "Unauthorized"}), 401
    return jsonify(read_update_info())


@app.route("/rollback_app", methods=["POST"])
def rollback_app():
    cfg = load_config()
    api_key = cfg.get("API_KEY")
    if api_key and request.headers.get("X-API-Key") != api_key:
        return "Unauthorized", 401
    repo_path = cfg.get("INSTALL_DIR") or os.getcwd()
    service_name = cfg.get("SERVICE_NAME", "echomosaic.service")
    # decide target: previous commit from history
    history_path = os.path.join(repo_path, "update_history.json")
    try:
        with open(history_path, "r") as hf:
            history = json.load(hf)
    except Exception:
        return render_template("update_status.html", message="No previous version to roll back to."), 400
    if not history:
        return render_template("update_status.html", message="No previous version to roll back to."), 400
    target = history[-1].get("from")
    if not target:
        return render_template("update_status.html", message="History does not include a valid commit."), 400
    try:
        subprocess.check_call(["git", "reset", "--hard", target], cwd=repo_path)
    except subprocess.CalledProcessError as e:
        return render_template("update_status.html", message=f"Rollback failed: {e}")
    try:
        subprocess.Popen(["sudo", "systemctl", "restart", service_name])
    except OSError:
        pass
    return render_template("update_status.html", message=f"Rolled back to {target[:7]}. Restarting service...")


@app.route("/update")
def update_view():
    # A simple progress UI that will kick off the update via fetch
    cfg = load_config()
    return render_template("update_progress.html", api_key=cfg.get("API_KEY", ""))


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


# --- Stream groups and metadata ---
@app.route("/streams_meta", methods=["GET"])
def streams_meta():
    meta = {}
    for k, v in settings.items():
        if k.startswith("_"):
            continue
        meta[k] = {
            "label": v.get("label", k),
            "include_in_global": v.get("include_in_global", True),
        }
    return jsonify(meta)


@app.route("/groups", methods=["GET", "POST"])
def groups_collection():
    if request.method == "GET":
        return jsonify(settings.get("_groups", {}))
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    streams = data.get("streams") or []
    if not name:
        return jsonify({"error": "Name required"}), 400
    settings.setdefault("_groups", {})
    settings["_groups"][name] = [s for s in streams if s in settings]
    save_settings(settings)
    return jsonify({"status": "ok", "group": {name: settings["_groups"][name]}})


@app.route("/groups/<name>", methods=["DELETE"])
def groups_delete(name):
    if "_groups" in settings and name in settings["_groups"]:
        del settings["_groups"][name]
        save_settings(settings)
        return jsonify({"status": "deleted"})
    return jsonify({"error": "not found"}), 404




@app.route("/stream/group/<name>")
def stream_group(name):
    groups = settings.get("_groups", {})
    group = groups.get(name)
    if not group and name.lower() == "default":
        # Dynamic default group = all configured streams
        group = [k for k in settings.keys() if not k.startswith("_")]
    if not group:
        return f"No group '{name}'", 404
    streams = {k: settings[k] for k in group if k in settings}
    mosaic = settings.get("_mosaic", default_mosaic_config())
    return render_template("streams.html", stream_settings=streams, mosaic_settings=mosaic)

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
