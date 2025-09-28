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

INSTALL_DIR=$(jq -r '.INSTALL_DIR' "$CONFIG_FILE")
SERVICE_NAME=$(jq -r '.SERVICE_NAME' "$CONFIG_FILE")
BRANCH=$(jq -r '.UPDATE_BRANCH' "$CONFIG_FILE")

if [ ! -d "$INSTALL_DIR/.git" ]; then
  echo "Error: $INSTALL_DIR does not appear to be a git repository."
  exit 1
fi

cd "$INSTALL_DIR"

echo -e "\nFetching latest changes from origin..."
# Reset any local changes and sync with the remote branch
git fetch origin
git checkout "$BRANCH"
git reset --hard "origin/${BRANCH}"
git clean -fd

echo -e "\nUpdating Python dependencies..."
"$INSTALL_DIR/venv/bin/pip" install --upgrade -r requirements.txt

echo -e "\nRestarting $SERVICE_NAME..."
sudo systemctl restart "$SERVICE_NAME"

echo -e "\nUpdate complete."
