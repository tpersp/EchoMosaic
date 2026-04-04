#!/usr/bin/env bash
# Install EchoMosaic from the current cloned repository.
# The installer configures the current repo as the live app location,
# creates a local virtual environment, and manages a systemd --user service.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'EOF'
Usage: ./install.sh [--dev] [--help]

Options:
  --dev   Install with development defaults (port 5001, echomosaic-dev.service, dev branch tracking)
  --help  Show this help message

This installer now runs EchoMosaic directly from the current cloned repository.
Do not run it from a copied tree or extracted archive.
EOF
}

IS_DEV=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dev)
      IS_DEV=true
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

if [ "$IS_DEV" = true ]; then
  default_port="5001"
  default_service="echomosaic-dev.service"
  BRANCH="dev"
  UPDATE_CHANNEL="branch"
else
  default_port="5000"
  default_service="echomosaic.service"
  BRANCH="main"
  UPDATE_CHANNEL="release"
fi

REPO_SLUG="tpersp/EchoMosaic"
INSTALL_DIR="$SCRIPT_DIR"
SERVICE_USER="$(whoami)"

if ! git -C "$SCRIPT_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Error: $SCRIPT_DIR is not a git repository."
  echo "Clone the repo first, then run install.sh from inside that clone."
  exit 1
fi

if ! git -C "$SCRIPT_DIR" remote get-url origin >/dev/null 2>&1; then
  echo "Error: git remote 'origin' is not configured for $SCRIPT_DIR."
  echo "EchoMosaic installs are expected to run from a normal git clone."
  exit 1
fi

if [ "$UPDATE_CHANNEL" = "release" ] && [ -n "$(git -C "$SCRIPT_DIR" status --porcelain --untracked-files=no)" ]; then
  echo "Error: release install requires a clean working tree."
  echo "Commit, stash, or discard local changes before running install.sh."
  exit 1
fi

EXISTING_INSTALL_DIR=""
if command -v python3 >/dev/null 2>&1; then
EXISTING_INSTALL_DIR="$(
python3 - "$INSTALL_DIR" <<'PY'
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
        continue
    if isinstance(loaded, dict):
        data.update(loaded)
value = str(data.get("INSTALL_DIR") or "").strip()
print(value, end="")
PY
)"
fi

if [ -n "$EXISTING_INSTALL_DIR" ] && [ "$EXISTING_INSTALL_DIR" != "$INSTALL_DIR" ]; then
  echo "Legacy split-install metadata detected."
  echo "Current repo path: $INSTALL_DIR"
  echo "Configured install path: $EXISTING_INSTALL_DIR"
  echo "This installer will retarget the app to run from the current cloned repo."
  read -r -p "Continue? [y/N]: " CONTINUE_RETARGET
  if [[ ! "$CONTINUE_RETARGET" =~ ^[Yy]$ ]]; then
    echo "Install cancelled."
    exit 1
  fi
fi

echo "Installing as user: ${SERVICE_USER}"
echo "Install location: ${INSTALL_DIR}"

read -r -p "Enter the port the server should listen on [${default_port}]: " PORT
PORT="${PORT:-$default_port}"

read -r -p "Enter the systemd service name [${default_service}]: " SERVICE_NAME
SERVICE_NAME="${SERVICE_NAME:-$default_service}"

echo -e "\nInstalling system packages..."
sudo apt-get update
sudo apt-get install < /dev/null -y python3 python3-venv python3-pip ffmpeg git

update_config_path() {
  local config_key="$1"
  local target_path="$2"
  python3 - "$INSTALL_DIR" "$config_key" "$target_path" "$SERVICE_NAME" "$BRANCH" "$UPDATE_CHANNEL" "$REPO_SLUG" <<'PY'
import json
import sys
from pathlib import Path

install_dir = Path(sys.argv[1]).expanduser().resolve()
config_key = sys.argv[2]
target_path = sys.argv[3]
service_name = sys.argv[4]
branch = sys.argv[5]
update_channel = sys.argv[6]
repo_slug = sys.argv[7]
config_path = install_dir / "config.json"
default_path = install_dir / "config.default.json"

data = {}
for candidate in (default_path, config_path):
    if candidate.is_file():
        try:
            loaded = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            loaded = {}
        if isinstance(loaded, dict):
            data.update(loaded)

data[config_key] = [target_path]
data["INSTALL_DIR"] = str(install_dir)
data["SERVICE_NAME"] = service_name
data["UPDATE_CHANNEL"] = update_channel
data["UPDATE_BRANCH"] = branch
data["REPO_SLUG"] = repo_slug
config_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
PY
}

write_install_metadata() {
  python3 - "$INSTALL_DIR" "$BRANCH" "$UPDATE_CHANNEL" "$REPO_SLUG" "$SERVICE_NAME" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

install_dir = Path(sys.argv[1]).expanduser().resolve()
branch = sys.argv[2]
update_channel = sys.argv[3]
repo_slug = sys.argv[4]
service_name = sys.argv[5]
config_path = install_dir / "config.json"
default_path = install_dir / "config.default.json"

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
        return subprocess.check_output(cmd, cwd=install_dir, stderr=subprocess.STDOUT).decode().strip()
    except Exception:
        return ""

installed_commit = git_safe(["git", "rev-parse", "HEAD"])
installed_version = ""
if update_channel == "release":
    installed_version = (
        git_safe(["git", "describe", "--tags", "--exact-match", "HEAD"])
        or git_safe(["git", "describe", "--tags", "--abbrev=0"])
    )
if not installed_version:
    installed_version = branch

data["INSTALL_DIR"] = str(install_dir)
data["SERVICE_NAME"] = service_name
data["UPDATE_BRANCH"] = branch
data["UPDATE_CHANNEL"] = update_channel
data["REPO_SLUG"] = repo_slug
data["INSTALLED_VERSION"] = installed_version
data["INSTALLED_COMMIT"] = installed_commit
config_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
PY
}

checkout_latest_release() {
  if [ "$UPDATE_CHANNEL" != "release" ]; then
    return
  fi
  echo -e "\nResolving latest stable release..."
  RELEASE_TAG="$(
  python3 - "$REPO_SLUG" <<'PY'
import json
import sys
import urllib.request

repo_slug = sys.argv[1]
request = urllib.request.Request(
    f"https://api.github.com/repos/{repo_slug}/releases/latest",
    headers={
        "Accept": "application/vnd.github+json",
        "User-Agent": "EchoMosaic-Installer/1.0",
    },
)
with urllib.request.urlopen(request, timeout=8) as response:
    payload = json.loads(response.read().decode("utf-8"))
print(str(payload.get("tag_name") or "").strip(), end="")
PY
)"
  if [ -z "$RELEASE_TAG" ]; then
    echo "Unable to determine latest release tag."
    exit 1
  fi
  git -C "$INSTALL_DIR" fetch origin --tags
  git -C "$INSTALL_DIR" checkout "$RELEASE_TAG"
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

checkout_latest_release

echo -e "\nCreating virtual environment..."
python3 -m venv "$INSTALL_DIR/venv"
"$INSTALL_DIR/venv/bin/pip" install --upgrade pip
"$INSTALL_DIR/venv/bin/pip" install -r "$INSTALL_DIR/requirements.txt"

configure_library_path "MEDIA_PATHS" "main media" "$INSTALL_DIR/media" "//192.168.1.0/images" "/mnt/viewers"
configure_library_path "AI_MEDIA_PATHS" "AI media" "$INSTALL_DIR/ai_media" "//192.168.1.0/ai-images" "/mnt/echomosaic-ai"
write_install_metadata

echo -e "\nCreating systemd user service..."
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

echo -e "\nEnabling and starting systemd user service..."
systemctl --user daemon-reload
systemctl --user enable "$SERVICE_NAME"
systemctl --user restart "$SERVICE_NAME"
if ! loginctl show-user "$USER" | grep -q "Linger=yes"; then
  echo "Enabling linger for user $USER so the service runs on boot..."
  sudo loginctl enable-linger "$USER" || true
fi

echo
echo "Installation complete."
echo "EchoMosaic now runs directly from this repository:"
echo "  $INSTALL_DIR"
echo "The dashboard should be available at http://<your-host>:$PORT/"
