#!/usr/bin/env bash

# Installation script for EchoMosaic with optional network share support.
#
# This script automates installation of required packages, copies the
# application files to a target directory, creates a Python virtual
# environment and installs dependencies, optionally configures a CIFS/SMB
# network share for the image directory, and sets up a systemd service
# to run the application via gunicorn.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Prompt for the service user
default_user="$(whoami)"
read -r -p "Enter the user account that should run the service [${default_user}]: " SERVICE_USER
SERVICE_USER="${SERVICE_USER:-$default_user}"

# Prompt for installation directory
default_install_dir="/opt/echomosaic"
read -r -p "Enter installation directory [${default_install_dir}]: " INSTALL_DIR
INSTALL_DIR="${INSTALL_DIR:-$default_install_dir}"

# Prompt for HTTP port
default_port="5000"
read -r -p "Enter the port the server should listen on [${default_port}]: " PORT
PORT="${PORT:-$default_port}"

# Determine default image directory
default_image_dir="/mnt/piviewers"

echo
echo "== Network Share Configuration =="
read -r -p "Mount a network share for the image directory? (y/n): " mount_answer
mount_answer=${mount_answer:-n}

IMAGE_DIR=""
if [[ "${mount_answer}" =~ ^[Yy]$ ]]; then
  read -r -p "Enter server share path (e.g. //192.168.1.100/MyShare): " SERVER_SHARE
  if [[ -z "${SERVER_SHARE}" ]]; then
    echo "No share path entered. Skipping network mount."
  else
    USER_ID="$(id -u "$SERVICE_USER")"
    read -r -p "Mount options (e.g. guest,uid=$USER_ID,gid=$USER_ID,vers=3.0) [ENTER for default]: " MOUNT_OPTS
    if [[ -z "${MOUNT_OPTS}" ]]; then
      MOUNT_OPTS="guest,uid=$USER_ID,gid=$USER_ID,vers=3.0"
    fi
    IMAGE_DIR="$default_image_dir"
    echo "Creating mount directory: $IMAGE_DIR"
    sudo mkdir -p "$IMAGE_DIR"
    FSTAB_LINE="$SERVER_SHARE  $IMAGE_DIR  cifs  $MOUNT_OPTS,x-systemd.automount  0  0"
    if grep -qs "$SERVER_SHARE" /etc/fstab; then
      echo "Share already present in /etc/fstab; skipping append."
    else
      echo "Appending to /etc/fstab: $FSTAB_LINE"
      echo "$FSTAB_LINE" | sudo tee -a /etc/fstab
    fi
    echo "Mounting all filesystems..."
    if ! sudo mount -a; then
      echo "WARNING: mount -a failed. Check credentials/options."
    else
      echo "Share mounted at $IMAGE_DIR."
    fi
  fi
fi

if [[ -z "$IMAGE_DIR" ]]; then
  # Local uploads folder
  IMAGE_DIR="$INSTALL_DIR/uploads"
  echo "No network share chosen. Creating local uploads folder at $IMAGE_DIR."
  sudo mkdir -p "$IMAGE_DIR"
  sudo chown "$SERVICE_USER":"$SERVICE_USER" "$IMAGE_DIR"
fi

echo
echo "== Installing system packages =="
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip

echo
echo "== Copying application files =="
sudo mkdir -p "$INSTALL_DIR"
sudo cp -r "$SCRIPT_DIR/." "$INSTALL_DIR/"
sudo chown -R "$SERVICE_USER":"$SERVICE_USER" "$INSTALL_DIR"

echo
echo "== Creating virtual environment and installing Python dependencies =="
sudo -u "$SERVICE_USER" python3 -m venv "$INSTALL_DIR/venv"
sudo -u "$SERVICE_USER" "$INSTALL_DIR/venv/bin/pip" install --upgrade pip
sudo -u "$SERVICE_USER" "$INSTALL_DIR/venv/bin/pip" install -r "$INSTALL_DIR/requirements.txt"

echo
echo "== Creating systemd service =="
SERVICE_NAME="echomosaic.service"
SERVICE_PATH="/etc/systemd/system/${SERVICE_NAME}"
sudo tee "$SERVICE_PATH" > /dev/null <<EOF
[Unit]
Description=EchoMosaic Dynamic Stream Application
After=network.target

[Service]
Type=simple
User=$SERVICE_USER
WorkingDirectory=$INSTALL_DIR
Environment="PATH=$INSTALL_DIR/venv/bin"
Environment="ECHO_IMAGE_DIR=$IMAGE_DIR"
ExecStart=$INSTALL_DIR/venv/bin/gunicorn --worker-class eventlet -w 1 -b 0.0.0.0:$PORT main:app
Restart=always

[Install]
WantedBy=multi-user.target
EOF

echo
echo "== Enabling and starting service =="
sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE_NAME"
sudo systemctl restart "$SERVICE_NAME"

echo
echo "Installation complete! The EchoMosaic dashboard should now be available at http://<your-host>:$PORT/"
