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

if [ ! -d "$INSTALL_DIR/.git" ]; then
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
git clean -fd

restore_and_cleanup
trap - EXIT

echo -e "\nUpdating Python dependencies..."
"$INSTALL_DIR/venv/bin/pip" install --upgrade -r requirements.txt

echo -e "\nRestarting $SERVICE_NAME..."
sudo systemctl restart "$SERVICE_NAME"

echo -e "\nUpdate complete."
