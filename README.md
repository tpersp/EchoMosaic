# EchoMosaic

EchoMosaic is a self-hosted Flask + Flask-SocketIO dashboard for building image, video, embed, and AI-generated media streams across one or more displays. Changes made in the dashboard are reflected live on connected viewers, so it works well for wall displays, passive media boards, art screens, and always-on ambient setups.

## Features

### Dashboard and stream control
- Multi-stream dashboard with per-stream cards for configuration, playback, and monitoring.
- Dashboard workspace shell with a shared sidebar across Dashboard, Media, Settings, and update pages.
- Collapsible desktop sidebar that remembers its state while you move between pages.
- Stream viewer pages for individual displays and a mosaic `/stream` view for multiple streams at once.
- Per-stream labels, folders, shuffle, durations, pinned/specific file selection, and background blur for letterboxed media.
- Dashboard organization with:
  - `Default` sort
  - `Custom` sort with a dedicated layout editor
  - `Group by Tag`
- Custom layout editor with persistent slot ordering, move left/right controls, and direct slot-number editing.
- Minimize controls for stream cards plus a dashboard-wide minimize toggle.

### Media modes
- Image / GIF playback from one or more configured media roots.
- Video playback with playback controls and configurable playback behavior.
- URL / embed mode for livestreams and external embeds.
- Picsum Photos mode with configurable dimensions, blur, grayscale, and seed reuse.
- AI image mode backed by Stable Horde.

### AI generation
- Stable Horde prompt generation per stream.
- AI preset management with create, update, delete, and per-stream reuse.
- LoRA selection and model listing endpoints.
- Auto-generation timers and queue/status tracking.
- Generated images stored alongside reusable media.

### Organization and layouts
- Shared tag system for filtering and grouping.
- Group management with named groups and saved layouts.
- Group display routes at `/stream/group/<name>`.
- Layout styles including grid, horizontal, vertical, focus, and picture-in-picture for grouped displays.

### Media library and thumbnails
- Media manager page for browsing folders and uploads.
- Create folder, rename, delete, upload, thumbnail, and preview-frame operations.
- Folder uploads that preserve the selected folder structure under the destination library path.
- Concurrent uploads with queue progress details and cancel support.
- Configurable media upload size limit from the Settings page.
- Cached thumbnails for dashboard cards and media browsing.
- Read-only media endpoints for random or explicit media fetches.
- Optional NSFW-folder hiding in stream folder selectors.

### Sync, timers, and playback
- Per-stream timers.
- Shared sync timers for coordinated stream advancement.
- Stream subscribe/unsubscribe live events via Socket.IO.
- Video control events over Socket.IO for viewer synchronization.
- YouTube sync handling for embed/live state updates.
- Improved YouTube playlist embeds, including playlist item resolution and synchronized active-item handling.

### Operations and diagnostics
- Settings import/export from the browser.
- Restore point creation, deletion, and restore flows.
- In-app update, update info, rollback, and update history views.
- Dashboard update badge when a newer release or branch update is available.
- Debug page with live log streaming and downloadable diagnostics.
- Health and system stats endpoints.
- Dashboard system monitor for CPU, RAM, GPU (when available), and storage.

## Requirements
- Linux host with `systemd` available.
- Python 3.9+.
- `python3-venv` and `pip`.
- `ffmpeg` for media helper features.
- Optional: `yt-dlp`-compatible livestream resolution support via the Python dependency stack.
- Optional: Stable Horde access if you want AI generation.

Python dependencies are listed in `requirements.txt`. Core runtime packages include Flask, Flask-SocketIO, eventlet, Pillow, requests, gunicorn, and yt-dlp.

## Installation

### Quick install script

The installer creates a local user-space install and a `systemd --user` service.

```bash
chmod +x install.sh
./install.sh
```

Defaults:
- install dir: `~/.local/share/echomosaic`
- port: `5000`
- service: `echomosaic.service`
- update channel: GitHub releases

For a development install:

```bash
chmod +x install.sh
./install.sh --dev
```

Development defaults:
- install dir: `~/.local/share/echomosaic-dev`
- port: `5001`
- service: `echomosaic-dev.service`
- update channel: `dev` branch

The installer will:
- install required system packages
- copy the repo into the chosen install directory
- create a virtual environment
- install Python dependencies
- optionally configure separate main-media and AI-media paths
- create and start a `systemd --user` service

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
- `INSTALL_DIR`: install target used by `update.sh`
- `SERVICE_NAME`: `systemd --user` service name used by `update.sh`
- `UPDATE_CHANNEL`: `release` for stable installs, `branch` for branch-tracking installs
- `UPDATE_BRANCH`: branch pulled by the update flow
- `REPO_SLUG`: GitHub repository used for release checks
- `INSTALLED_VERSION` / `INSTALLED_COMMIT`: tracked install metadata used by update status checks

Notes:
- Gunicorn should stay on a single worker unless you redesign the shared runtime state and add a proper Socket.IO message queue.
- The Settings page is the preferred way to export/import stream definitions and related state.
- Custom stream layout order is persisted separately from default stream ordering.

## Running and Updating

### Start from a working checkout

```bash
python app.py
```

or:

```bash
python main.py
```

### Update an installed instance

```bash
chmod +x update.sh
./update.sh
```

`update.sh`:
- reads `INSTALL_DIR`, `SERVICE_NAME`, and `UPDATE_BRANCH` from `config.json`
- uses `UPDATE_CHANNEL` to decide whether to follow GitHub releases or a branch head
- backs up user state
- checks out the latest configured release or branch target
- reinstalls Python dependencies
- restarts the configured `systemd --user` service

The Settings page also exposes browser-driven update, rollback, restore point, and history actions.

## Service Management

The installer creates a `systemd --user` service, so the most useful commands are usually:

```bash
systemctl --user status echomosaic.service
systemctl --user restart echomosaic.service
systemctl --user stop echomosaic.service
```

For a development install:

```bash
systemctl --user status echomosaic-dev.service
systemctl --user restart echomosaic-dev.service
```

If you are unsure which service name your install uses, check `config.json` for `SERVICE_NAME`.

## Troubleshooting

### Check service status

Production-style install:

```bash
systemctl --user status echomosaic.service
```

Development install:

```bash
systemctl --user status echomosaic-dev.service
```

### View recent logs

Production-style install:

```bash
journalctl --user -u echomosaic.service -n 200 --no-pager
```

Development install:

```bash
journalctl --user -u echomosaic-dev.service -n 200 --no-pager
```

### Follow logs live

```bash
journalctl --user -u echomosaic.service -f
```

or:

```bash
journalctl --user -u echomosaic-dev.service -f
```

### Common things to check

- If the app will not update, confirm the installed copy is a valid git repository:

```bash
git -C ~/.local/share/echomosaic rev-parse --is-inside-work-tree
```

- If the dashboard loads but media is missing, verify `MEDIA_PATHS` and `AI_MEDIA_PATHS` in `config.json`.
- If the service starts but viewers do not sync correctly, confirm you are still running a single gunicorn worker.
- If the browser update page closes too quickly, use `journalctl --user -u <service-name> -n 200 --no-pager` to inspect the actual failure.
- If uploads or generated media behave strangely after moving installs between hosts, confirm the installed service is pointing at the expected install dir and media roots.

### Health endpoints

You can also check:
- `/health`
- `/api/system_stats`
- `/debug`

## Development Notes

- The server entrypoints are `app.py` and `main.py`.
- The main application wiring lives in `echomosaic_app/server.py`.
- Feature routes live under `echomosaic_app/routes`.
- Socket handlers live under `echomosaic_app/sockets`.
- Durable architecture guidance lives in `docs/architecture-guide.md`.
- Running `install.sh --dev` creates a development install with dev defaults and configures updates to follow the `dev` branch.

## Limitations

- No built-in authentication or multi-user permission model.
- Eventlet is still in use and emits a deprecation warning; it works today, but it is not a future-proof long-term async stack.
- The update flow assumes the installed copy remains a valid git checkout.
- Media roots are still config-driven rather than managed through a richer storage UI.

## License

EchoMosaic is available under the MIT License. See `LICENSE`.
