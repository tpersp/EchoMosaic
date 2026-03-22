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

print(get_value("INSTALL_DIR", "/opt/echomosaic"))
print(get_value("SERVICE_NAME", "echomosaic.service"))
print(get_value("UPDATE_BRANCH", "main"))
PY
)
  INSTALL_DIR="${CONFIG_VALUES[0]:-/opt/echomosaic}"
  SERVICE_NAME="${CONFIG_VALUES[1]:-echomosaic.service}"
  BRANCH="${CONFIG_VALUES[2]:-main}"
else
  if ! command -v jq >/dev/null 2>&1; then
    echo "Error: python or jq is required to read $CONFIG_FILE."
    exit 1
  fi
  INSTALL_DIR=$(jq -r '.INSTALL_DIR' "$CONFIG_FILE")
  SERVICE_NAME=$(jq -r '.SERVICE_NAME' "$CONFIG_FILE")
  BRANCH=$(jq -r '.UPDATE_BRANCH' "$CONFIG_FILE")
fi

if ! git -C "$INSTALL_DIR" rev-parse --is-inside-work-tree &>/dev/null; then
  echo "Error: $INSTALL_DIR does not appear to be a git repository."
  exit 1
fi

cd "$INSTALL_DIR"

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

echo -e "\nFetching latest changes from origin..."
# Reset any local changes and sync with the remote branch
git fetch origin
git checkout "$BRANCH"
git reset --hard "origin/${BRANCH}"
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

echo -e "\nRestarting $SERVICE_NAME..."
systemctl --user restart "$SERVICE_NAME"

echo -e "\nUpdate complete."
