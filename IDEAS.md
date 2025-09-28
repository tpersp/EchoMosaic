# List of Ideas

A running list of ideas and future improvements. Add new items anywhere below.


- [ ] Stream real-time update logs via Socket.IO instead of a client-side animation.

- [x] Add a simple “Update history” view in Settings showing prior updates and commit messages.

- [ ] Add a “Show only selected” count or filter in the dashboard to quickly see which streams are included.

- [ ] Provide bulk toggle actions (e.g., include/exclude all).

- [ ] “Slideshow sync”: sync transitions across all streams (e.g., all change every 10s together), to prevent them from changing at "random interval" when having multiple streams on one screen or near each other.

- [ ] Low-bandwidth mode: auto-reduces image resolution for remote access.

- [ ] Show a live thumbnail preview for each stream’s current content directly on the dashboard (so you can see all streams at a glance before opening them).

- [ ] Add a settings backup/export feature so custom streams/folders/tags can be quickly restored or moved to another device.

- [x] add prettier update feature so user isn't hit with "This site can’t be reached <site/ip> refused to connect.

- [ ] add server sided rendering as an option if possible. so it feels more like a proper stream as everyone who opens the site sees the same thing at the same time. - maybe not all streams, but the option to enable server side rendering for a specific group would be nice, if single stream is needed then the user would just make group with only one stream, hence SSR should be controlled on group level.

- [ ] Add system monitoring, to watch cpu usage, memory usage(using/max), gpu usage(if any), storage available for media, other useful info.

- [ ] Add stream quality options instead of free text format, options should be 1080p, 720p, 480, 360p, 240p, 144p, Auto. if a non-youtube stream is used, it should automatically select "auto" for compatibility.

- [ ] Add option to shuffle display order of media in folders - should by default shuffle, with the option to turn off shuffle.

## Suggestions from a friend:
- [ ]  [2025-09-08] Add optional authentication or API‑key protection for the dashboard and update endpoints. The README notes that after installation you can access the dashboard directly at http://your‑server:PORT/, so anyone on the network can modify streams and run updates. Implementing a simple login or token-based auth would improve security. - Lets drop the api key but use a simple login with session cache for some time to not annoy the user.

- [ ]  [2025-09-08] Make paths and constants configurable via config.json or environment variables. Users currently have to edit app.py to change where images are stored. Exposing settings like IMAGE_DIR (and other constants) through configuration files or a settings UI would improve maintainability and avoid manual code changes.

- [ ]  [2025-09-08] Cache directory listings or use file-system watchers. Functions like get_subfolders() and list_images() walk the entire image directory and sort the results on every request. Caching these results and refreshing them when the file system changes (e.g., using inotify on Linux) would boost performance, especially with large image libraries.

- [ ]  [2025-09-08] Offload HLS lookup to a background task and cache results. The try_get_hls() function synchronously calls yt‑dlp on each request, which can block the server and slow down the UI. Running the lookup asynchronously, caching the resolved HLS URLs, and providing placeholder content or error messages during processing would improve responsiveness and stability.

- [ ]  [2025-09-08] Add a light/dark theme toggle. The existing CSS defines a global dark theme with variables for colors. Allow users to switch between dark and light modes to improve accessibility and user comfort.

- [ ]  [2025-09-08] Enhance group management and stream ordering. The dashboard stores group tile states in local storage, but doesn’t persist layouts server‑side or allow drag‑and‑drop reordering. Adding drag‑and‑drop ordering, server‑side persistence of group layouts, and options to clone/share groups would make managing many streams easier.

- [ ]  [2025-09-08] Improve error handling and user feedback. When an image or stream URL is missing, the server returns a generic JSON error or 404. Surfacing these errors in the UI with clear messages (e.g., “Image not found” or “Unsupported stream URL”) would help users troubleshoot issues without checking logs.

- [ ]  [2025-09-08] Add configurable logging and monitoring. Operations like yt‑dlp calls and update scripts either fail silently or log to the console. Implementing configurable logging levels and persistent logs (e.g., rotating file logs) would aid debugging and provide insight into system health.


Notes
- Use checkboxes to track status (unchecked → planned, checked → done).
- Optionally prefix entries with a date, e.g., `[2025-09-07] Idea text…`.
- Group related items under short headings if this grows large.
