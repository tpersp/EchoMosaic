# EchoMosaic

EchoMosaic is a self-hosted Flask + Flask-SocketIO dashboard for building image, video, embed, and AI-generated media streams across one or more displays. Changes made in the dashboard are reflected live on connected viewers, so it works well for wall displays, passive media boards, art screens, and always-on ambient setups.

## Highlights

- Multi-stream dashboard with live viewer pages and per-stream controls.
- Shared sidebar UI across Dashboard, Media, Settings, and updater pages.
- Local media playback for images, GIFs, and video, plus URL/embed, Picsum, and Stable Horde AI modes.
- Media manager with folder uploads, queue controls, thumbnails, and configurable upload limits.
- Tags, groups, sync timers, layout editor, and global `Links` management.
- In-app update, rollback, restore-point, and update-history support.

## Requirements
- Linux host with `systemd` available.
- Python 3.9+.
- `python3-venv` and `pip`.
- `ffmpeg` for media helper features.
- Optional: `yt-dlp`-compatible livestream resolution support via the Python dependency stack.
- Optional: Stable Horde access if you want AI generation.

Python dependencies are listed in `requirements.txt`. Core runtime packages include Flask, Flask-SocketIO, eventlet, Pillow, requests, gunicorn, and yt-dlp.

## Installation

EchoMosaic now runs directly from the cloned repo. The installer creates a local `venv` and a `systemd --user` service that points at that clone.

### Stable install

```bash
git clone https://github.com/tpersp/EchoMosaic.git
cd EchoMosaic
bash install.sh
```

Defaults:
- port: `5000`
- service: `echomosaic.service`
- updates follow GitHub releases

### Development install

```bash
git clone --branch dev https://github.com/tpersp/EchoMosaic.git EchoMosaic-dev
cd EchoMosaic-dev
bash install.sh --dev
```

Defaults:
- port: `5001`
- service: `echomosaic-dev.service`
- updates follow the `dev` branch

### What the installer does

- verifies the current folder is a real git clone
- installs required system packages
- creates a local virtual environment
- installs Python dependencies
- optionally configures media paths
- creates and starts a `systemd --user` service

### Uninstall

```bash
bash uninstall.sh
```

To also remove the local virtual environment:

```bash
bash uninstall.sh --remove-venv
```

`uninstall.sh` removes the user service but leaves the repo, config, settings, and media paths in place by default.

## Migration From `v2026.04.02` And Earlier

Releases up to `v2026.04.02` used the older copied-install layout, where EchoMosaic was cloned in one place and then copied into a separate install directory. The new installer no longer uses that model.

If you are migrating from `v2026.04.02` or earlier, the safest path is:

1. Back up anything you need from the old install:
   - `settings.json`
   - `config.json`
   - media folders if they lived inside the old install dir
2. Stop and remove the old user service.
3. Remove the old copied install directory.
4. Clone EchoMosaic fresh into the folder you want to use long-term.
5. Run the new installer from inside that clone.
6. Re-import settings and restore media if needed.

Example cleanup for an older production-style install:

```bash
systemctl --user stop echomosaic.service
systemctl --user disable echomosaic.service
rm -f ~/.config/systemd/user/echomosaic.service
systemctl --user daemon-reload
rm -rf ~/.local/share/echomosaic
```

Example cleanup for an older development install:

```bash
systemctl --user stop echomosaic-dev.service
systemctl --user disable echomosaic-dev.service
rm -f ~/.config/systemd/user/echomosaic-dev.service
systemctl --user daemon-reload
rm -rf ~/.local/share/echomosaic-dev
```

Then install again from a real clone using the stable or dev flow above. The new installer keeps update history and git-based update behavior consistent because the app now runs from the actual clone.

### Manual setup

If you prefer to run EchoMosaic directly from a working checkout:

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

By default the app listens on port `5000`. You can also use:

```bash
python main.py
```

for the development server path.

## Configuration

Key configuration lives in:
- `config.json`: runtime config, install metadata, update branch, media roots, AI defaults
- `settings.json`: streams, tags, groups, presets, sync timers, custom stream order

Important config keys:
- `MEDIA_PATHS`: main media roots
- `AI_MEDIA_PATHS`: AI media roots
- `INSTALL_DIR`: active repo path used by `update.sh`
- `SERVICE_NAME`: `systemd --user` service name used by `update.sh`
- `UPDATE_CHANNEL`: `release` for stable installs, `branch` for branch-tracking installs
- `UPDATE_BRANCH`: branch pulled by the update flow
- `REPO_SLUG`: GitHub repository used for release checks
- `INSTALLED_VERSION` / `INSTALLED_COMMIT`: tracked install metadata used by update status checks

Notes:
- Gunicorn should stay on a single worker unless you redesign the shared runtime state and add a proper Socket.IO message queue.
- The Settings page is the preferred way to export/import stream definitions and related state.
- Custom stream layout order is persisted separately from default stream ordering.

## Running, Updating, and Service Checks

### Run from a checkout

```bash
python app.py
```

or:

```bash
python main.py
```

### Update an installed instance

```bash
bash update.sh
```

`update.sh` reads install metadata from `config.json`, follows the configured release or branch target, backs up user state, updates the repo, reinstalls Python dependencies, and restarts the `systemd --user` service. The Settings page also exposes browser-driven update, rollback, restore point, and history actions.

### Common service commands

```bash
systemctl --user status echomosaic.service
systemctl --user restart echomosaic.service
journalctl --user -u echomosaic.service -n 200 --no-pager
```

For dev installs, replace `echomosaic.service` with `echomosaic-dev.service`.

If you are unsure which service name your install uses, check `config.json` for `SERVICE_NAME`.

### Common things to check

- Confirm the install is still a valid git repo:

```bash
git -C /path/to/EchoMosaic rev-parse --is-inside-work-tree
```

- If the dashboard loads but media is missing, verify `MEDIA_PATHS` and `AI_MEDIA_PATHS` in `config.json`.
- If viewers do not sync correctly, confirm you are still running a single gunicorn worker.
- If the update page closes too quickly, inspect the service logs with `journalctl --user -u <service-name> -n 200 --no-pager`.
- If uploads or generated media behave strangely after moving installs between hosts, confirm the service is still pointing at the expected repo path and media roots.
- Health/debug endpoints: `/health`, `/api/system_stats`, `/debug`

## Development Notes

- The server entrypoints are `app.py` and `main.py`.
- The main application wiring lives in `echomosaic_app/server.py`.
- Feature routes live under `echomosaic_app/routes`.
- Socket handlers live under `echomosaic_app/sockets`.
- Durable architecture guidance lives in `docs/architecture-guide.md`.
- Running `install.sh --dev` configures the current dev clone as a development install and sets updates to follow the `dev` branch.

## Limitations

- No built-in authentication or multi-user permission model.
- Eventlet is still in use and emits a deprecation warning; it works today, but it is not a future-proof long-term async stack.
- The update flow assumes the active repo remains a valid git checkout with a working `origin` remote.
- Media roots are still config-driven rather than managed through a richer storage UI.

## License

EchoMosaic is available under the MIT License. See `LICENSE`.
