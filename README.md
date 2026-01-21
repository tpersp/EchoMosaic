# EchoMosaic

EchoMosaic is a self-hosted web dashboard for curating image, video, and AI-generated streams across one or more displays. It runs on Flask with Flask-SocketIO so that edits you make in the browser show up immediately on connected viewers.

## Current Capabilities
- Multi-stream dashboard where each card can rotate through a folder, pin a single file, run Stable Horde prompts, or embed an external livestream. Cards expose timers, shuffle options, playback controls, and per-stream settings without reloading the page.
- Stable Horde integration that supports prompt editing, job queues, LoRA stacks, post-processing toggles, auto-generation timers, and preset management. Results are stored alongside other media so they can be reused by any stream.
- Stream organization features including tags, groups, and a mosaic viewer with configurable layouts (grid, horizontal, vertical, focus, and picture-in-picture). Groups can be exported to dedicated `/stream/<group>` URLs for display screens.
- Media handling helpers such as cached thumbnails, adjustable image quality, optional NSFW folder hiding, background blur for filler space, and basic video playback controls (loop, play to end, duration).
- Operations tooling on the Settings page: installer/update helpers, Stable Horde default overrides, import/export of stream definitions, and a rollback button that reverts to the previous recorded update.

## Limitations and Open Work
EchoMosaic is still evolving. The most notable gaps are:
- No authentication or user accounts; anyone who can reach the server can control it.
- `MEDIA_PATHS` lives in `config.json`, so changing media roots requires editing that file or re-running the installer prompt.
- Livestream HLS lookups use a blocking `yt-dlp` call and are not cached yet.
- Low-bandwidth and slideshow-sync modes are not implemented.
- Real-time update logs, server-side rendered playback, and system health dashboards are tracked in `IDEAS.md`.

See `IDEAS.md` for the fuller backlog and status.

## Requirements
- Linux host tested with Debian/Ubuntu (the helper scripts assume systemd).
- Python 3.9+ with `python3-venv` and `pip`.
- Optional: `yt-dlp` binary available in `PATH` for resolving livestream URLs.
- Optional: access to Stable Horde API if you plan to use AI generation.

Python dependencies live in `requirements.txt` and include Flask, Flask-SocketIO, eventlet, Pillow, requests, and yt-dlp.

## Installation

### Quick install script (Debian/Ubuntu)
The `install.sh` script installs Python prerequisites, copies the project to a target directory, builds a virtual environment, and registers a systemd service.

```
chmod +x install.sh
sudo ./install.sh
```

During the prompts you can choose the service account, install path (default `/opt/echomosaic`), listening port, and whether to mount a CIFS share for media. The script also updates `MEDIA_PATHS` inside `config.json` to point at the chosen location.

### Manual setup
If you prefer to manage everything yourself:

1. Install Python and virtualenv support.
   ```
   sudo apt update
   sudo apt install -y python3 python3-venv python3-pip
   ```
2. Clone the repository and create a virtual environment.
   ```
   python3 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
3. Adjust `MEDIA_PATHS` inside `config.json` so it matches where your media lives.
4. Start the development server.
   ```
   python app.py
   ```
   The app listens on port 5000 by default.

For systemd or reverse proxy setups, you can use `install.sh` and `update.sh` as references.

## Configuration Notes
- `MEDIA_PATHS` controls the root media location. Update it manually or re-run the installer script if you move your library.
- Gunicorn should run with a single worker unless you add shared state and a Socket.IO message queue; multiple workers will cause stream data to appear/disappear.
- `config.json` stores settings for the update helper (install path, service name, branch) and Stable Horde defaults (model, output folders, etc.).
- Stream definitions, groups, tags, and AI presets live in `settings.json`. Use the Settings page to export/import backups.

## Updating and Rollbacks
Run `update.sh` to pull the latest code, reinstall Python dependencies, and restart the service.

```
chmod +x update.sh
./update.sh
```

`update.sh` records the previous and current commit in `update_history.json`. The Settings page offers:
- **Update now**: wraps `update.sh` so you can kick off updates from the browser.
- **Rollback**: resets to the last recorded commit and restarts the service.
- **History**: lists recent updates with commit information.

## Development Tips
- Use `python main.py` if you prefer Flask-SocketIO's development server with reloads; it wraps the same app as `python app.py`.
- The front-end assets live under `static/` and `templates/`. `temp_script.js` currently holds the dashboard logic.
- Enable debug logging by setting `FLASK_ENV=development` or adjusting the logger configuration inside `app.py`.

## Contributing
Bug reports, fixes, and modest feature additions are welcome. Please open an issue or pull request with a clear description and any relevant reproduction steps. Tests are limited right now, so manual verification or screenshots help a lot.

## License
EchoMosaic is available under the MIT License (see `LICENSE`).
