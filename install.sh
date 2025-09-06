#!/usr/bin/env bash

# This installation script automates the setup of the EchoView Dashboard
# application. It performs the following steps:
#   1. Installs system packages required for Python and virtual environments.
#   2. Copies the application files to a dedicated installation directory.
#   3. Creates a Python virtual environment and installs dependencies.
#   4. Configures and enables a systemd service so the app starts on boot.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Prompt for the service user (default: current user)
default_user="$(whoami)"
read -r -p "Enter the user account that should run the service [${default_user}]: " SERVICE_USER
SERVICE_USER="${SERVICE_USER:-$default_user}"

# Prompt for installation directory (default: /opt/echoview-dashboard)
default_install_dir="/opt/echoview-dashboard"
read -r -p "Enter installation directory [${default_install_dir}]: " INSTALL_DIR
INSTALL_DIR="${INSTALL_DIR:-$default_install_dir}"

# Prompt for HTTP port (default: 5000)
default_port="5000"
read -r -p "Enter the port the server should listen on [${default_port}]: " PORT
PORT="${PORT:-$default_port}"

echo "\nInstalling system packages…"
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip

echo "\nCopying files to ${INSTALL_DIR}…"
sudo mkdir -p "$INSTALL_DIR"
# Copy everything in this repository into the installation directory.  We avoid
# copying any pre‑existing virtual environment directory.
sudo cp -r "$SCRIPT_DIR/." "$INSTALL_DIR/"
sudo chown -R "$SERVICE_USER":"$SERVICE_USER" "$INSTALL_DIR"

echo "\nCreating virtual environment…"
sudo -u "$SERVICE_USER" python3 -m venv "$INSTALL_DIR/venv"
sudo -u "$SERVICE_USER" "$INSTALL_DIR/venv/bin/pip" install --upgrade pip
sudo -u "$SERVICE_USER" "$INSTALL_DIR/venv/bin/pip" install -r "$INSTALL_DIR/requirements.txt"

echo "\nCreating systemd service…"
SERVICE_NAME="echoview-dashboard.service"
SERVICE_PATH="/etc/systemd/system/${SERVICE_NAME}"
sudo tee "$SERVICE_PATH" > /dev/null <<EOF
[Unit]
Description=EchoView Dashboard Web Application
After=network.target

[Service]
Type=simple
User=$SERVICE_USER
WorkingDirectory=$INSTALL_DIR
Environment="PATH=$INSTALL_DIR/venv/bin"
ExecStart=$INSTALL_DIR/venv/bin/gunicorn --worker-class eventlet -w 1 -b 0.0.0.0:$PORT app:app
Restart=always

[Install]
WantedBy=multi-user.target
EOF

echo "\nEnabling and starting systemd service…"
sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE_NAME"
sudo systemctl restart "$SERVICE_NAME"

echo "\nInstallation complete!"
echo "The EchoView Dashboard should now be accessible at http://<your-host>:$PORT/"