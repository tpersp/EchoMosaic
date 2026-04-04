# TODO

A cleaned-up list of what is still open, what is done, and what was tried and later replaced or removed.

## Current Priorities
- [ ] Add a `System Events` panel for recent high-signal app activity.
  Note: this should be a curated in-app event feed, not a raw server log viewer. Good candidates are updates, media/upload failures, broken stream sources, and AI job warnings/errors.
- [ ] Reduce RAM usage without limiting usability or performance.

## Open Suggestions / Future Ideas
- [ ] [2025-09-08] Add optional authentication or simple login with a session cache so the dashboard and update endpoints are protected from anyone on the network.
  Note: still undecided whether this is actually needed for a personal/local service.
- [ ] [2025-09-08] Offload HLS lookup to a background task and cache results. `try_get_hls()` / live HLS resolution should not block request handling under load.
- [ ] [2025-09-08] Improve error handling and user feedback. Missing images, broken media paths, and bad stream URLs should surface clearer UI messages instead of generic failures.

## Completed

### Bugs Fixed
- [x] Folders are shown twice.
- [x] Images uploaded with capital `.JPG` extensions do not appear in Media or stream rotation.
- [x] Folders created in the UI are not always shown on the dashboard.
- [x] Folders and images created/uploaded through the UI are not always available immediately in the dashboard.
- [x] [2026-03-23] Importing settings from exported JSON did not preserve dashboard stream order.

### Features Implemented
- [x] Split `AI Images` into `Generate`, `View Random`, and `View Specific`, with generated media stored in a separate `/ai_media` library isolated from normal `/media`.
- [x] Implement Stable Horde image generation with stream controls, queue handling, and presets.
- [x] [2025-09-08] Cache directory listings so media browsing does not rescan the filesystem on every request.
- [x] Add the ability to cancel Stable Horde queue jobs.
- [x] Add support for movie/video files (`mp4`, `mkv`, `webm`, `mov`, `avi`, `m4v`, `mpg`, `mpeg`).
- [x] Add the ability to save AI image generation presets.
- [x] Add stream tags/categories with sorting and filtering for better dashboard overview.
- [x] Add an update history view in Settings showing prior updates and commit messages.
- [x] Add a friendlier update flow so users are not dropped onto a browser error page during restarts.
- [x] Add stream quality options instead of free-text fields (`1080p`, `720p`, `480p`, `360p`, `240p`, `144p`, `Auto`) with automatic `Auto` for non-YouTube streams.
- [x] Grid strict-mode for groups: when a group has more streams than `Rows x Cols`, show only the first `N` instead of auto-expanding.
- [x] Provide bulk toggle actions in the group editor (`Add all`, `Remove all`).
- [x] Add a `Show only selected` indicator in the dashboard for group membership.
- [x] Add settings backup/export so streams, folders, tags, and related config can be restored or moved.
- [x] Add an option to shuffle media display order in folders.
- [x] [2025-09-08] Add a light/dark theme toggle.
- [x] [2025-09-21] Expand Stable Horde controls with LoRA stacks, post-processing chains, and worker preference toggles directly in the dashboard.
- [x] [2025-09-21] Move AI generator controls into a modal window and surface per-stream summaries on the dashboard.
- [x] Show a live thumbnail preview for each stream's current content directly on the dashboard.
- [x] Add low-bandwidth mode to reduce image resolution and caching behavior for remote access.
- [x] Add a Picsum Photos mode with exposed options in the UI.
- [x] Add system monitoring for CPU, memory, GPU (if available), storage, and related stats.
- [x] Implement rollback/restore points so the app can return to a known-good version on demand.
- [x] [2025-09-08] Make paths and constants configurable via `config.json` or environment variables, including support for multiple media roots.
- [x] Enable minimizing streams on the dashboard for better overview.
- [x] Add a custom stream layout mode with persistent manual ordering and an `Edit Layout` UI.
- [x] [2025-09-08] Add configurable logging and monitoring.

### Major UI / Workflow Work Completed
- [x] Rework the main app UI into a cohesive sidebar workspace across Dashboard, Media, Settings, and the updater flow.
- [x] Add global `Links` management with categories, source/type detection, dashboard picker flow, and a dedicated dashboard manager.
- [x] Improve YouTube playlist handling with playlist metadata, item selection, and better stream-card integration.
- [x] Fix YouTube livestream detection and playback handling so live streams are recognized and behave as live streams instead of normal videos.
- [x] Add folder upload support in Media, including preserved folder structure.
- [x] Add upload queue improvements including parallel uploads and cancellation support.
- [x] Add configurable media upload size in Settings and surface the current limit in Media.

## Replaced / Removed
- [x] Add drag-and-drop stream reordering on the dashboard and persist the chosen order.
  Replaced by the current `Custom` layout mode and `Edit Layout` controls, because drag-and-drop in the grid was too unstable and unpleasant.
- [x] [2026-01-27] Sync timers for random image/GIF streams so multiple streams change at the exact same time.
  Implemented through the current Sync Timers manager and per-stream timer assignment/offset flow.
- [x] Add folder filtering so the user can hide folders containing `nsfw` / `NSFW` by default.
  This feature was implemented earlier, but later removed from the app because it depended on an unclear naming convention and added clutter without enough value.

## Notes
- For new ideas, add a short `Note:` line whenever the title could be interpreted in more than one way.
- Keep idea titles short, but use the note to clarify scope, intent, or what should explicitly be avoided.
- Keep `TODO.md` limited to actionable items that can be completed, removed, or explicitly discarded.
- If an item is large enough to need design notes, careful handling, or acceptance criteria, create a matching file in [`plans/`](/home/doden/workspace/EchoMosaic-Project/EchoMosaic-dev/plans) and keep only the short trackable entry here.
- Use unchecked boxes for real open work only.
- Move completed items into `Completed` instead of leaving them mixed into idea sections.
- If a feature is later removed, keep it in `Replaced / Removed` so the history stays understandable.
