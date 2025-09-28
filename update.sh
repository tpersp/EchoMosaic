#!/usr/bin/env bash

# Update the EchoView Dashboard/EchoMosaic installation by pulling the latest
# changes from the Git repository, reinstalling dependencies, and restarting
# the systemd service.

set -euo pipefail

# Prompt for installation directory
default_install_dir="/opt/echoview-dashboard"
read -r -p "Enter installation directory [${default_install_dir}]: " INSTALL_DIR
INSTALL_DIR="${INSTALL_DIR:-$default_install_dir}"

# Prompt for systemd service name
default_service="echoview-dashboard.service"
read -r -p "Enter systemd service name [${default_service}]: " SERVICE_NAME
SERVICE_NAME="${SERVICE_NAME:-$default_service}"

if [ ! -d "$INSTALL_DIR/.git" ]; then
  echo "Error: $INSTALL_DIR does not appear to be a git repository."
  exit 1
fi

cd "$INSTALL_DIR"

echo "\nFetching latest changes from origin..."
# Reset any local changes and sync with the remote branch
git fetch origin
current_branch=$(git rev-parse --abbrev-ref HEAD)
git reset --hard "origin/${current_branch}"
git clean -fd

echo "\nUpdating Python dependencies..."
"$INSTALL_DIR/venv/bin/pip" install --upgrade -r requirements.txt

echo "\nRestarting $SERVICE_NAME..."
sudo systemctl restart "$SERVICE_NAME"

echo "\nUpdate complete."
