#!/usr/bin/env bash
# This installation script automates the setup of the EchoMosaic
# application. It performs the following steps:
#   1. Installs system packages required for Python and virtual environments.
#   2. Copies the application files to a dedicated installation directory.
#   3. Creates a Python virtual environment and installs dependencies.
#   4. Configures and enables a systemd service so the app starts on boot.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse arguments
IS_DEV=false
if [[ "${1:-}" == "--dev" ]]; then
  IS_DEV=true
  echo "Development mode enabled via --dev flag."
fi

# Set defaults based on mode
if [ "$IS_DEV" = true ]; then
  default_install_dir="$HOME/.local/share/echomosaic-dev"
  default_port="5001"
  default_service="echomosaic-dev.service"
  BRANCH="dev"
else
  default_install_dir="$HOME/.local/share/echomosaic"
  default_port="5000"
  default_service="echomosaic.service"
  BRANCH="main"
fi

# Prompt for the service user (default: current user)
default_user="$(whoami)"
read -r -p "Enter the user account that should run the service [${default_user}]: " SERVICE_USER
SERVICE_USER="${SERVICE_USER:-$default_user}"
# Prompt for installation directory
read -r -p "Enter installation directory [${default_install_dir}]: " INSTALL_DIR
INSTALL_DIR="${INSTALL_DIR:-$default_install_dir}"
# Expand tilde if present
INSTALL_DIR="${INSTALL_DIR/#\~/$HOME}"
# Prompt for HTTP port
read -r -p "Enter the port the server should listen on [${default_port}]: " PORT
PORT="${PORT:-$default_port}"

# Prompt for systemd service name
read -r -p "Enter the systemd service name [${default_service}]: " SERVICE_NAME
SERVICE_NAME="${SERVICE_NAME:-$default_service}"

echo -e "\nInstalling system packages…"
sudo apt-get update
sudo apt-get install < /dev/null -y python3 python3-venv python3-pip ffmpeg
echo -e "\nCopying files to ${INSTALL_DIR}…"
mkdir -p "$INSTALL_DIR"
# Copy everything in this repository into the installation directory.  We avoid
# copying any pre‑existing virtual environment directory.
cp -r "$SCRIPT_DIR/." "$INSTALL_DIR/"
echo -e "\nCreating virtual environment…"
python3 -m venv "$INSTALL_DIR/venv"
"$INSTALL_DIR/venv/bin/pip" install --upgrade pip
"$INSTALL_DIR/venv/bin/pip" install -r "$INSTALL_DIR/requirements.txt"
# == Step: Configure MEDIA_PATHS and AI_MEDIA_PATHS independently ==
# Normal media and AI media are configured separately on purpose.
# AI media defaults to a local path unless the user explicitly chooses otherwise.
update_config_path() {
  local config_key="$1"
  local target_path="$2"
  python3 - "$INSTALL_DIR" "$config_key" "$target_path" "$SERVICE_NAME" "$BRANCH" <<'PY'
import json
import sys
from pathlib import Path

install_dir = Path(sys.argv[1]).expanduser().resolve()
config_key = sys.argv[2]
target_path = sys.argv[3]
service_name = sys.argv[4]
branch = sys.argv[5]
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

data[config_key] = [target_path]
data["INSTALL_DIR"] = str(install_dir)
data["SERVICE_NAME"] = service_name
data["UPDATE_BRANCH"] = branch
config_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
PY
}

ensure_cifs_utils() {
  if ! dpkg -s cifs-utils >/dev/null 2>&1; then
    echo "Installing cifs-utils..."
    sudo apt-get update && sudo apt-get install < /dev/null -y cifs-utils
  fi
}

configure_library_path() {
  local config_key="$1"
  local label="$2"
  local local_default="$3"
  local share_default="$4"
  local mount_default="$5"

  echo
  echo "== Configure ${label} =="
  read -r -p "Mount a CIFS network share for ${label}? (y/n): " mount_answer
  if [[ "$mount_answer" =~ ^[Yy]$ ]]; then
    ensure_cifs_utils
    read -r -p "Enter server share path [${share_default}]: " SERVER_SHARE
    SERVER_SHARE="${SERVER_SHARE:-$share_default}"
    read -r -p "Enter local mount point [${mount_default}]: " MOUNT_POINT
    MOUNT_POINT="${MOUNT_POINT:-$mount_default}"
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
    echo "Setting ${config_key} in config.json -> $MOUNT_POINT"
    update_config_path "$config_key" "$MOUNT_POINT"
  else
    echo
    echo "No CIFS share selected. Using a local folder for ${label}."
    mkdir -p "$local_default"
    echo "Setting ${config_key} in config.json -> $local_default"
    update_config_path "$config_key" "$local_default"
  fi
}

configure_library_path "MEDIA_PATHS" "main media" "$INSTALL_DIR/media" "//192.168.1.0/images" "/mnt/viewers"
configure_library_path "AI_MEDIA_PATHS" "AI media" "$INSTALL_DIR/ai_media" "//192.168.1.0/ai-images" "/mnt/echomosaic-ai"

echo -e "\nCreating systemd user service…"
SERVICE_DIR="$HOME/.config/systemd/user"
mkdir -p "$SERVICE_DIR"
SERVICE_PATH="$SERVICE_DIR/${SERVICE_NAME}"
cat > "$SERVICE_PATH" <<EOF
[Unit]
Description=EchoMosaic Web Application
After=network.target
[Service]
Type=simple
WorkingDirectory=$INSTALL_DIR
Environment="PATH=$INSTALL_DIR/venv/bin:$PATH"
$( [ "$IS_DEV" = true ] && echo 'Environment="FLASK_ENV=development"' )
ExecStart=$INSTALL_DIR/venv/bin/gunicorn -w 1 -k eventlet --bind 0.0.0.0:$PORT --timeout 120 --graceful-timeout 30 --keep-alive 5 --no-sendfile app:app
Restart=always
[Install]
WantedBy=default.target
EOF
echo -e "\nEnabling and starting systemd user service…"
systemctl --user daemon-reload
systemctl --user enable "$SERVICE_NAME"
systemctl --user restart "$SERVICE_NAME"
if ! loginctl show-user "$USER" | grep -q "Linger=yes"; then
  echo "Enabling linger for user $USER so the service runs on boot…"
  sudo loginctl enable-linger "$USER" || true
fi
echo "\nInstallation complete!"
echo "The EchoMosaic Dashboard should now be accessible at http://<your-host>:$PORT/"
