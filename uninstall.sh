#!/usr/bin/env bash
# Remove EchoMosaic service wiring for the current cloned repository.
# By default this leaves the repo, config, media, and settings intact.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMOVE_VENV=false

usage() {
  cat <<'EOF'
Usage: ./uninstall.sh [--remove-venv] [--help]

Options:
  --remove-venv  Also remove the local virtual environment in this repo
  --help         Show this help message

This script removes the systemd --user service created by install.sh.
It does not delete the repository, config, media folders, or settings data by default.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --remove-venv)
      REMOVE_VENV=true
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

SERVICE_NAME="$(
python3 - "$SCRIPT_DIR" <<'PY'
import json
import sys
from pathlib import Path

repo = Path(sys.argv[1]).resolve()
config_path = repo / "config.json"
default_path = repo / "config.default.json"
data = {}
for candidate in (default_path, config_path):
    if not candidate.is_file():
        continue
    try:
        loaded = json.loads(candidate.read_text(encoding="utf-8"))
    except Exception:
        loaded = {}
    if isinstance(loaded, dict):
        data.update(loaded)
print(str(data.get("SERVICE_NAME") or "echomosaic.service").strip(), end="")
PY
)"

SERVICE_NAME="${SERVICE_NAME:-echomosaic.service}"
SERVICE_PATH="$HOME/.config/systemd/user/${SERVICE_NAME}"

echo "Repository path: $SCRIPT_DIR"
echo "Service name: ${SERVICE_NAME}"
echo
echo "This will remove the systemd --user service wiring for this EchoMosaic clone."
echo "Repo files, config, settings, and media will remain."

read -r -p "Continue? [y/N]: " CONFIRM_UNINSTALL
if [[ ! "$CONFIRM_UNINSTALL" =~ ^[Yy]$ ]]; then
  echo "Uninstall cancelled."
  exit 1
fi

if systemctl --user list-unit-files | grep -q "^${SERVICE_NAME}"; then
  systemctl --user stop "$SERVICE_NAME" || true
  systemctl --user disable "$SERVICE_NAME" || true
fi

rm -f "$SERVICE_PATH"
systemctl --user daemon-reload
systemctl --user reset-failed

if [ "$REMOVE_VENV" = true ]; then
  rm -rf "$SCRIPT_DIR/venv"
  echo "Removed local virtual environment: $SCRIPT_DIR/venv"
else
  echo "Left local virtual environment intact: $SCRIPT_DIR/venv"
fi

echo
echo "EchoMosaic service wiring removed."
echo "Remaining files stay in place under:"
echo "  $SCRIPT_DIR"
