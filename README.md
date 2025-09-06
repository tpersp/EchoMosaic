# EchoView Dashboard

EchoView Dashboard is a web‑based control panel and viewer for one or more
EchoView receivers. It provides a simple way to configure up to four
independent streams, each of which can display images from a folder or
play a live stream from a service such as YouTube or Twitch. The
application is built with Flask and uses Socket.IO to update the
interface in real time.

## Features

* **Multi‑stream viewer** – manage up to four concurrent streams from a
  single dashboard. Each stream can operate in one of three modes:
  random image rotation, selection of a specific image, or live
  streaming via an embedded player.
* **Folder and image selection** – choose a folder on the server where
  images are stored and let the application randomly cycle through
  them, or pick a specific image to display.
* **Livestream support** – specify a YouTube or Twitch URL to
  embed a live stream; other streaming services are supported by
  attempting to extract an HLS link via `yt‑dlp`.
* **Real‑time updates** – settings changes are propagated instantly to
  all connected clients using Socket.IO, eliminating the need for
  manual refreshes.

## Getting started

The easiest way to set up the application on a Linux system is to run
the provided `install.sh` script. This script installs the required
packages, creates a Python virtual environment, installs Python
dependencies, sets up a systemd service and enables it to start on
boot. You should run the script with sufficient privileges (e.g. using
`sudo`) on a Debian/Ubuntu system.

```
chmod +x install.sh
sudo ./install.sh
```

The script will prompt you for the user account under which the service
should run, the installation directory and the TCP port to bind to.
After installation completes you can access the dashboard from a
browser at `http://your‑server:PORT/`.

> **Note**: If you need to store images in a different location
> than the default `/mnt/piviewers` you can edit the `IMAGE_DIR`
> constant near the top of `app.py` before installing.

## Manual installation

If you prefer a manual setup, follow these steps:

1. Install system dependencies:
   ```bash
   sudo apt update
   sudo apt install -y python3 python3-venv python3-pip
   ```
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the application for development:
   ```bash
   python app.py
   ```
   By default it listens on all interfaces on port 5000.

For production deployments it is recommended to run the application via
`systemd` or a process manager such as Gunicorn behind a reverse
proxy like Nginx. See `install.sh` for a working example of a systemd
service.

## Contributing

Contributions are welcome! Please open an issue or a pull request on
GitHub to discuss features or report bugs. When submitting a pull
request, make sure to format your code with a consistent style and
include docstrings where appropriate.

## License

This project is licensed under the MIT License. See the `LICENSE` file
for details.