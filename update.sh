#!/usr/bin/env bash

# Update the EchoMosaic installation by pulling the latest
# changes from the Git repository, reinstalling dependencies,
# and restarting the systemd service.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/config.json"

if [ ! -f "$CONFIG_FILE" ]; then
  echo "Configuration file $CONFIG_FILE not found."
  exit 1
fi

PYTHON_READ=""
if command -v python3 >/dev/null 2>&1; then
  PYTHON_READ="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_READ="python"
fi

if [ -n "$PYTHON_READ" ]; then
  mapfile -t CONFIG_VALUES < <("$PYTHON_READ" - "$CONFIG_FILE" <<'PY'
import json
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
try:
    data = json.loads(config_path.read_text(encoding="utf-8"))
except Exception:
    data = {}

def get_value(key: str, default: str) -> str:
    if isinstance(data, dict):
        value = data.get(key, default)
    else:
        value = default
    if value in (None, ""):
        value = default
    return str(value)

branch = get_value("UPDATE_BRANCH", "main").strip() or "main"
channel = get_value("UPDATE_CHANNEL", "").strip().lower()
service_name = get_value("SERVICE_NAME", "echomosaic.service").strip().lower()
install_dir = get_value("INSTALL_DIR", "/opt/echomosaic").strip().lower()
install_basename = Path(install_dir).name.lower() if install_dir else ""
is_dev_install = (
    branch == "dev"
    or "echomosaic-dev" in service_name
    or "echomosaic-dev" in install_basename
)
if channel not in {"branch", "release"}:
    channel = "branch" if is_dev_install else "release"
if is_dev_install:
    channel = "branch"

print(get_value("INSTALL_DIR", "/opt/echomosaic"))
print(get_value("SERVICE_NAME", "echomosaic.service"))
print(channel)
print(branch)
print(get_value("REPO_SLUG", "tpersp/EchoMosaic"))
PY
)
  INSTALL_DIR="${CONFIG_VALUES[0]:-/opt/echomosaic}"
  SERVICE_NAME="${CONFIG_VALUES[1]:-echomosaic.service}"
  UPDATE_CHANNEL="${CONFIG_VALUES[2]:-branch}"
  BRANCH="${CONFIG_VALUES[3]:-main}"
  REPO_SLUG="${CONFIG_VALUES[4]:-tpersp/EchoMosaic}"
else
  if ! command -v jq >/dev/null 2>&1; then
    echo "Error: python or jq is required to read $CONFIG_FILE."
    exit 1
  fi
  INSTALL_DIR=$(jq -r '.INSTALL_DIR' "$CONFIG_FILE")
  SERVICE_NAME=$(jq -r '.SERVICE_NAME' "$CONFIG_FILE")
  UPDATE_CHANNEL=$(jq -r '.UPDATE_CHANNEL // "branch"' "$CONFIG_FILE")
  BRANCH=$(jq -r '.UPDATE_BRANCH' "$CONFIG_FILE")
  REPO_SLUG=$(jq -r '.REPO_SLUG // "tpersp/EchoMosaic"' "$CONFIG_FILE")
fi

INSTALL_BASENAME="$(basename "$INSTALL_DIR" | tr '[:upper:]' '[:lower:]')"
SERVICE_NAME_LOWER="$(printf '%s' "$SERVICE_NAME" | tr '[:upper:]' '[:lower:]')"
if [ "$BRANCH" = "dev" ] || [[ "$SERVICE_NAME_LOWER" == *"echomosaic-dev"* ]] || [[ "$INSTALL_BASENAME" == *"echomosaic-dev"* ]]; then
  UPDATE_CHANNEL="branch"
fi

if ! git -C "$INSTALL_DIR" rev-parse --is-inside-work-tree &>/dev/null; then
  echo "Error: $INSTALL_DIR does not appear to be a git repository."
  exit 1
fi

cd "$INSTALL_DIR"

PRE_UPDATE_COMMIT="$(git rev-parse HEAD 2>/dev/null || true)"

PYTHON_BIN="$INSTALL_DIR/venv/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  else
    echo "Python interpreter not found."
    exit 1
  fi
fi

BACKUP_DIR="$("$PYTHON_BIN" - "$INSTALL_DIR" <<'PY'
import sys
from update_helpers import backup_user_state

print(backup_user_state(sys.argv[1]), end="")
PY
)"

mapfile -t CLEAN_EXCLUDES < <("$PYTHON_BIN" - "$INSTALL_DIR" <<'PY'
import json
import sys
from pathlib import Path

repo = Path(sys.argv[1]).resolve()

def load_json(path: Path):
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}

config_data = load_json(repo / "config.json")
default_data = load_json(repo / "config.default.json")

def normalize_paths(value, default):
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, list):
        items = value
    else:
        items = [default]
    cleaned = []
    seen_local = set()
    for raw in items:
        text = str(raw).strip() if raw is not None else ""
        if not text or text in seen_local:
            continue
        seen_local.add(text)
        cleaned.append(text)
    return cleaned or [default]

raw_paths = []
raw_paths.extend(normalize_paths(config_data.get("MEDIA_PATHS") or default_data.get("MEDIA_PATHS"), "./media"))
raw_paths.extend(normalize_paths(config_data.get("AI_MEDIA_PATHS") or default_data.get("AI_MEDIA_PATHS"), "./ai_media"))

for persistent_dir in ("backups", "restorepoints"):
    print(f"{persistent_dir}/")

seen = set()
for raw in raw_paths:
    text = str(raw).strip() if raw is not None else ""
    if not text:
        continue
    candidate = Path(text).expanduser()
    if not candidate.is_absolute():
        candidate = (repo / candidate).resolve(strict=False)
    else:
        candidate = candidate.resolve(strict=False)
    try:
        relative = candidate.relative_to(repo)
    except ValueError:
        continue
    rel_text = relative.as_posix().strip("/")
    if not rel_text:
        continue
    pattern = f"{rel_text}/"
    if pattern in seen:
        continue
    seen.add(pattern)
    print(pattern)
PY
)

restore_and_cleanup() {
  if [ -n "${BACKUP_DIR:-}" ]; then
    "$PYTHON_BIN" - "$INSTALL_DIR" "$BACKUP_DIR" <<'PY'
import sys
from update_helpers import restore_user_state

restore_user_state(sys.argv[1], sys.argv[2], cleanup=True)
PY
    BACKUP_DIR=""
  fi
}

trap restore_and_cleanup EXIT

if [ "$UPDATE_CHANNEL" = "release" ]; then
  echo -e "\nResolving latest release from GitHub..."
  TARGET_REF="$("$PYTHON_BIN" - "$REPO_SLUG" <<'PY'
import json
import sys
import urllib.request

repo_slug = sys.argv[1]
request = urllib.request.Request(
    f"https://api.github.com/repos/{repo_slug}/releases/latest",
    headers={
        "Accept": "application/vnd.github+json",
        "User-Agent": "EchoMosaic-Updater/1.0",
    },
)
with urllib.request.urlopen(request, timeout=8) as response:
    payload = json.loads(response.read().decode("utf-8"))
print(str(payload.get("tag_name") or "").strip(), end="")
PY
)"
  if [ -z "$TARGET_REF" ]; then
    echo "Error: unable to determine latest release tag for $REPO_SLUG."
    exit 1
  fi
else
  TARGET_REF="origin/${BRANCH}"
fi

echo -e "\nFetching latest changes from origin..."
# Reset any local changes and sync with the remote branch
git fetch origin --tags
if [ "$UPDATE_CHANNEL" = "release" ]; then
  git checkout "$TARGET_REF"
  git reset --hard "$TARGET_REF"
else
  git checkout "$BRANCH"
  git reset --hard "$TARGET_REF"
fi
GIT_CLEAN_CMD=(git clean -fd)
for exclude in "${CLEAN_EXCLUDES[@]:-}"; do
  [ -n "$exclude" ] || continue
  GIT_CLEAN_CMD+=(-e "$exclude")
done
"${GIT_CLEAN_CMD[@]}"

restore_and_cleanup
trap - EXIT

echo -e "\nUpdating Python dependencies..."
"$INSTALL_DIR/venv/bin/pip" install --upgrade -r requirements.txt

echo -e "\nRecording installed version metadata..."
"$PYTHON_BIN" - "$INSTALL_DIR" "$UPDATE_CHANNEL" "$BRANCH" "$REPO_SLUG" "$TARGET_REF" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

repo = Path(sys.argv[1]).resolve()
update_channel = sys.argv[2]
branch = sys.argv[3]
repo_slug = sys.argv[4]
target_ref = sys.argv[5]
config_path = repo / "config.json"
default_path = repo / "config.default.json"

data = {}
for candidate in (default_path, config_path):
    if candidate.is_file():
        try:
            loaded = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            loaded = {}
        if isinstance(loaded, dict):
            data.update(loaded)

def git_safe(cmd):
    try:
        return subprocess.check_output(cmd, cwd=repo, stderr=subprocess.STDOUT).decode().strip()
    except Exception:
        return ""

installed_commit = git_safe(["git", "rev-parse", "HEAD"])
installed_version = ""
if update_channel == "release":
    installed_version = git_safe(["git", "describe", "--tags", "--exact-match", "HEAD"]) or target_ref
if not installed_version:
    installed_version = branch

data["UPDATE_CHANNEL"] = update_channel
data["UPDATE_BRANCH"] = branch
data["REPO_SLUG"] = repo_slug
data["INSTALLED_VERSION"] = installed_version
data["INSTALLED_COMMIT"] = installed_commit
config_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
PY

POST_UPDATE_COMMIT="$(git rev-parse HEAD 2>/dev/null || true)"
if [ -n "$PRE_UPDATE_COMMIT" ] && [ -n "$POST_UPDATE_COMMIT" ] && [ "$PRE_UPDATE_COMMIT" != "$POST_UPDATE_COMMIT" ]; then
  echo -e "\nRecording update history..."
  "$PYTHON_BIN" - "$INSTALL_DIR" "$PRE_UPDATE_COMMIT" "$POST_UPDATE_COMMIT" "$BRANCH" <<'PY'
import sys
from update_helpers import append_update_history

append_update_history(
    sys.argv[1],
    sys.argv[2],
    sys.argv[3],
    sys.argv[4] if len(sys.argv) > 4 else None,
)
PY
fi

echo -e "\nRestarting $SERVICE_NAME..."
systemctl --user restart "$SERVICE_NAME"

echo -e "\nUpdate complete."
