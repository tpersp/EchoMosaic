#!/usr/bin/env bash
# This installation script automates the setup of the EchoMosaic
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
# Prompt for installation directory (default: /opt/echomosaic)
default_install_dir="/opt/echomosaic"
read -r -p "Enter installation directory [${default_install_dir}]: " INSTALL_DIR
INSTALL_DIR="${INSTALL_DIR:-$default_install_dir}"
# Prompt for HTTP port (default: 5000)
default_port="5000"
read -r -p "Enter the port the server should listen on [${default_port}]: " PORT
PORT="${PORT:-$default_port}"
SERVICE_NAME="echomosaic.service"
echo "\nInstalling system packages…"
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip ffmpeg
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
# == Step: (Optional) Configure MEDIA_PATHS (CIFS or local) ==
# MEDIA_PATHS lives in config.json (or config.default.json as a fallback).
# We either mount a CIFS share (default //192.168.1.0/images -> /mnt/viewers)
# and point MEDIA_PATHS there, or set a local folder and point MEDIA_PATHS to that.
update_config() {
  local target_path="$1"
  sudo -u "$SERVICE_USER" python3 - "$INSTALL_DIR" "$target_path" "$SERVICE_NAME" <<'PY'
import json
import sys
from pathlib import Path

install_dir = Path(sys.argv[1]).expanduser().resolve()
target_path = sys.argv[2]
service_name = sys.argv[3]
config_path = install_dir / "config.json"
default_path = install_dir / "config.default.json"

data = {}
for candidate in (config_path, default_path):
    if candidate.is_file():
        try:
            data = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            data = {}
        if isinstance(data, dict) and data:
            break

if not isinstance(data, dict):
    data = {}

data["MEDIA_PATHS"] = [target_path]
data["INSTALL_DIR"] = str(install_dir)
data["SERVICE_NAME"] = service_name
data.setdefault("UPDATE_BRANCH", "main")
config_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
PY
}
echo
echo "== Optional: Configure a network share for MEDIA_PATHS =="
read -r -p "Mount a CIFS network share for images? (y/n): " mount_answer
if [[ "$mount_answer" =~ ^[Yy]$ ]]; then
  # Ensure cifs-utils is available
  if ! dpkg -s cifs-utils >/dev/null 2>&1; then
    echo "Installing cifs-utils..."
    sudo apt-get update && sudo apt-get install -y cifs-utils
  fi
  default_share="//192.168.1.0/images"
  default_mount="/mnt/viewers"
  read -r -p "Enter server share path [${default_share}]: " SERVER_SHARE
  SERVER_SHARE="${SERVER_SHARE:-$default_share}"
  read -r -p "Enter local mount point [${default_mount}]: " MOUNT_POINT
  MOUNT_POINT="${MOUNT_POINT:-$default_mount}"
  echo
  echo "Auth options:"
  echo "  1) guest (no username/password)  [default]"
  echo "  2) username/password (stored in /etc/samba/creds-echomosaic)"
  read -r -p "Choose authentication [1/2]: " AUTH_CHOICE
  AUTH_CHOICE="${AUTH_CHOICE:-1}"
  USER_ID="$(id -u "$SERVICE_USER")"
  GROUP_ID="$(id -g "$SERVICE_USER")"
  BASE_OPTS="uid=$USER_ID,gid=$GROUP_ID,iocharset=utf8,file_mode=0644,dir_mode=0755,noperm,vers=3.0,x-systemd.automount,_netdev,noauto"
  if [[ "$AUTH_CHOICE" == "2" ]]; then
    CRED_FILE="/etc/samba/creds-echomosaic"
    read -r -p "Username: " CIFS_USER
    read -r -s -p "Password: " CIFS_PASS; echo
    read -r -p "Domain (optional, ENTER to skip): " CIFS_DOMAIN
    sudo mkdir -p "$(dirname "$CRED_FILE")"
    if [[ -n "$CIFS_DOMAIN" ]]; then
      printf "username=%s\npassword=%s\ndomain=%s\n" "$CIFS_USER" "$CIFS_PASS" "$CIFS_DOMAIN" | sudo tee "$CRED_FILE" >/dev/null
    else
      printf "username=%s\npassword=%s\n" "$CIFS_USER" "$CIFS_PASS" | sudo tee "$CRED_FILE" >/dev/null
    fi
    sudo chmod 600 "$CRED_FILE"
    MOUNT_OPTS="credentials=$CRED_FILE,$BASE_OPTS"
  else
    MOUNT_OPTS="guest,username=guest,password=,$BASE_OPTS"
  fi
  echo "Creating mount dir: $MOUNT_POINT"
  sudo mkdir -p "$MOUNT_POINT"
  sudo chown "$SERVICE_USER:$SERVICE_USER" "$MOUNT_POINT"
  # Append to /etc/fstab if not already present
  FSTAB_LINE="$SERVER_SHARE  $MOUNT_POINT  cifs  $MOUNT_OPTS  0  0"
  if grep -qsF "$SERVER_SHARE  $MOUNT_POINT  cifs" /etc/fstab; then
    echo "Entry already in /etc/fstab; skipping append."
  else
    echo "Adding to /etc/fstab:"
    echo "  $FSTAB_LINE"
    echo "$FSTAB_LINE" | sudo tee -a /etc/fstab >/dev/null
  fi
  echo "Mounting $MOUNT_POINT..."
  if ! sudo mount "$MOUNT_POINT"; then
    echo "WARNING: mount failed. Check network/credentials/options."
  else
    echo "Mounted $SERVER_SHARE at $MOUNT_POINT."
  fi
  # Update MEDIA_PATHS in config.json to the chosen mount point
  echo "Setting MEDIA_PATHS in config.json -> $MOUNT_POINT"
  update_config "$MOUNT_POINT"
else
  echo
  echo "No CIFS share selected. Using a local folder for images."
  LOCAL_DIR="$INSTALL_DIR/media"
  sudo mkdir -p "$LOCAL_DIR"
  sudo chown "$SERVICE_USER:$SERVICE_USER" "$LOCAL_DIR"
  echo "Setting MEDIA_PATHS in config.json -> $LOCAL_DIR"
  update_config "$LOCAL_DIR"
fi
echo "\nCreating systemd service…"
SERVICE_PATH="/etc/systemd/system/${SERVICE_NAME}"
sudo tee "$SERVICE_PATH" > /dev/null <<EOF
[Unit]
Description=EchoMosaic Web Application
After=network.target
[Service]
Type=simple
User=$SERVICE_USER
WorkingDirectory=$INSTALL_DIR
Environment="PATH=$INSTALL_DIR/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
ExecStart=$INSTALL_DIR/venv/bin/gunicorn -w 1 -k eventlet --bind 0.0.0.0:$PORT --timeout 120 --graceful-timeout 30 --keep-alive 5 --no-sendfile app:app
Restart=always
[Install]
WantedBy=multi-user.target
EOF
echo "\nEnabling and starting systemd service…"
sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE_NAME"
sudo systemctl restart "$SERVICE_NAME"
echo "\nInstallation complete!"
echo "The EchoMosaic Dashboard should now be accessible at http://<your-host>:$PORT/"
