# List of Ideas

A running list of ideas and future improvements. Add new items anywhere below.

## Active Ideas
- [ ] Stream real-time update logs via Socket.IO instead of the current client-side animation.
- [ ] "Slideshow sync": sync transitions across all streams (for example, they all change every 10 seconds together) to prevent random offsets when multiple screens are visible at once.
- [ ] Low-bandwidth mode: automatically reduce image resolution and caching behaviour for remote access.
- [ ] Add server-side rendering as an option so everyone who opens the site sees the same frame at the same time. This should be controlled at the group level (a single-stream group covers the one-off case).
- [ ] Add system monitoring to surface CPU usage, memory usage (used/max), GPU usage (if available), storage available for media, and other useful info.
- [ ] Implement a better rollback feature that lets admins mark a restore point so the server can always roll back to a known good state on demand.

## Suggestions from a friend
- [ ] [2025-09-08] Add optional authentication or simple login with a session cache so casual users are not blocked but the dashboard and update endpoints are protected from anyone on the network.
- [ ] [2025-09-08] Make paths and constants configurable via `config.json` or environment variables. Today users still edit `app.py` to change where images are stored.
- [ ] [2025-09-08] Offload HLS lookup to a background task and cache results. `try_get_hls()` currently invokes `yt-dlp` synchronously on each request, which can block the server under load.
- [ ] [2025-09-08] Enhance group management and stream ordering. Persist layouts server-side, allow drag-and-drop reordering, and offer options to clone or share groups.
- [ ] [2025-09-08] Improve error handling and user feedback. When an image or stream URL is missing, surface clear messages in the UI instead of generic JSON errors.
- [ ] [2025-09-08] Add configurable logging and monitoring. Operations like `yt-dlp` calls and update scripts either fail silently or log to the console; configurable log levels and rotating file logs would help.
- [ ] Add a page and function similar to `stablehorde.py` that can pull images from https://picsum.photos/ with the available options exposed in the UI.
- [ ] Add a page and function similar to `stablehorde.py` that can pull images from the https://www.pexels.com/api/ with the available options exposed in the UI.

## Implemented / Completed Ideas
- [x] Implement Stable Horde image generation. (Streams can switch to AI mode, queue jobs, and manage presets.)
- [x] [2025-09-08] Cache directory listings so media browsing does not rescan the filesystem on every request. (Implemented via the in-process `IMAGE_CACHE`.)
- [x] Add folder filter and toggle so the user can hide folders containing `nsfw`/`NSFW` by default.
- [x] Add ability to cancel Stable Horde queue jobs.
- [x] Add ability to show movie files (mp4, mkv, webm, mov, avi, m4v, mpg, mpeg).
- [x] Add ability to save AI image generation presets so users can store different settings per workflow.
- [x] Add a feature to categorise streams on the dashboard and provide sorting and filtering for categories to improve overview when many streams exist.
- [x] Add a simple "Update history" view in Settings showing prior updates and commit messages.
- [x] Add a friendlier update flow so users are not dropped onto a browser error page during restarts.
- [x] Add stream quality options instead of free-text fields (1080p, 720p, 480p, 360p, 240p, 144p, Auto) with automatic Auto selection for non-YouTube streams.
- [x] Grid strict-mode: when a group has more streams than Rows x Cols, optionally show only the first N instead of auto-expanding the grid. Provide pagination/scroll or a toggle to auto-fit vs. strict.
- [x] Provide bulk toggle actions (for example, Add all / Remove all) in the group editor.
- [x] Add a "Show only selected" indicator in the dashboard to quickly see which streams are included.
- [x] Add a settings backup/export feature so custom streams, folders, and tags can be restored or moved to another device.
- [x] Add an option to shuffle the display order of media in folders (enabled by default with a toggle to turn it off).
- [x] [2025-09-08] Add a light/dark theme toggle.
- [x] [2025-09-21] Expand Stable Horde controls with LoRA stacks, post-processing chains, and worker preference toggles directly in the dashboard.
- [x] [2025-09-21] Move AI generator controls into a modal window and surface per-stream summaries on the dashboard.
- [x] Show a live thumbnail preview for each stream's current content directly on the dashboard so you can see all streams at a glance before opening them.

Notes
- Use checkboxes to track status (unchecked = planned, checked = done).
- Optionally prefix entries with a date, e.g. `[2025-09-07] Idea text...`.
- Group related items under short headings if this grows large.

