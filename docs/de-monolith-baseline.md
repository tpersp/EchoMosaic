# De-Monolith Baseline

This note captures the pre-blueprint baseline for EchoMosaic so the
refactor can proceed in small, verifiable steps without losing track of
public behavior or hidden couplings.

## Current Entrypoints

- `app.py`: main Flask + Socket.IO module, route definitions, runtime state, and background orchestration
- `main.py`: compatibility runner that imports `app` and `socketio`
- `config_manager.py`: config/default/env loading
- `media_manager.py`: media filesystem operations and thumbnails
- `stablehorde.py`: Stable Horde API client
- `debug_manager.py`: journal streaming for the debug page
- `timer_manager.py`: timer math and timer config normalization
- `job_manager.py`: background AI job listener tracking
- `picsum.py`: Picsum helper routes and stream assignment

## Route Inventory

Current HTTP handlers registered from the monolith:

- `GET /`
- `GET /debug`
- `GET /api/debug/stream`
- `GET /api/debug/download`
- `GET /stream`
- `GET /stream/<name>`
- `POST /streams`
- `DELETE /streams/<stream_id>`
- `GET /get-settings/<stream_id>`
- `GET /stream/state/<stream_id>`
- `POST /settings/<stream_id>`
- `POST /api/timer/update/<stream_id>`
- `POST /picsum/refresh`
- `GET /settings/ai-defaults`
- `POST /settings/ai-defaults`
- `GET /ai/presets`
- `POST /ai/presets`
- `PATCH /ai/presets/<preset_name>`
- `DELETE /ai/presets/<preset_name>`
- `GET /ai/loras`
- `GET /tags`
- `POST /tags`
- `DELETE /tags/<path:tag_name>`
- `GET /ai/models`
- `GET /ai/status/<stream_id>`
- `GET /api/jobs/<stream_id>/latest`
- `POST /ai/generate/<stream_id>`
- `POST /ai/cancel/<stream_id>`
- `GET /settings`
- `GET|POST /api/settings/timers`
- `GET|POST /api/sync_timers`
- `PUT|DELETE /api/sync_timers/<timer_id>`
- `GET|POST /restore_points`
- `DELETE /restore_points/<point_id>`
- `POST /restore_points/<point_id>/restore`
- `POST /update_app`
- `GET /update_info`
- `GET /update_history`
- `POST /rollback_app`
- `GET /update`
- `GET /health`
- `GET /api/system_stats`
- `GET /streams_meta`
- `GET|POST /groups`
- `DELETE /groups/<name>`
- `GET /stream/live`
- `POST /stream/live/invalidate`
- `GET /live`
- `POST /test_embed`
- `GET /images`
- `GET /images/random`
- `GET /media`
- `GET /media/random`
- `GET|POST /notes`
- `GET /stream/image/<path:image_path>`
- `GET /stream/video/<path:video_path>`
- `GET /stream/thumbnail/<stream_id>`
- `GET /stream/thumbnail/<stream_id>/image`
- `GET /thumbnails/<stream_id>.jpg`
- `GET /stream/group/<name>`
- `GET /media/manage`
- `GET /folders`
- `GET /api/media/list`
- `POST /api/media/create_folder`
- `POST /api/media/rename`
- `DELETE /api/media/delete`
- `POST /api/media/upload`
- `GET /api/media/thumbnail`
- `GET /api/media/preview_frame`
- `GET /settings/export`
- `POST /settings/import`

## Socket Inventory

Current Socket.IO handlers registered from the monolith:

- `disconnect`
- `ai_watch`
- `ai_unwatch`
- `stream_subscribe`
- `stream_unsubscribe`
- `video_control`
- `youtube_state`
- default error handler

## Current State Owners

- Config loading: `config_manager.load_config()` inside `app.py`
- Settings loading/saving: currently centralized through `app.py`; now extracted to `echomosaic_app.state.settings_store`
- AI job runtime state: `ai_jobs_lock`, `ai_jobs`, `ai_job_controls`, `ai_model_cache` in `app.py`
- Playback runtime state: playback manager globals and stream runtime dicts in `app.py`
- Timer runtime state: per-stream timer data inside `settings.json` plus scheduler globals in `app.py`
- Media manager bootstrap: config-derived media roots and `MediaManager(...)` initialization in `app.py`

## Persistence Touchpoints

### `settings.json`

Canonical write path:

- `echomosaic_app/state/settings_store.py`
  `SettingsStore.save()` performs the atomic write.
- `app.py`
  `save_settings()` and `save_settings_debounced()` are the compatibility wrappers that call the settings store.

Callers that mutate in-memory settings and then persist through the canonical debounced path include:

- dashboard/stream update flows
- tag creation/deletion
- notes updates
- AI preset/default updates
- Picsum refresh and timer updates
- extracted services:
  - `echomosaic_app/services/stream_config.py`
  - `echomosaic_app/services/timer_sync.py`
  - `echomosaic_app/services/groups.py`
  - `echomosaic_app/services/ai_orchestration.py`

There should not be any direct ad hoc JSON writes to `settings.json` outside the settings store.

### `config.json`

Runtime/config write paths:

- `config_manager.py`
  `save_config()` is the canonical config persistence helper.
- `config_manager.py`
  `ensure_config_file()` may create or backfill `config.json` when keys are missing.
- `app.py`
  the timer-snap settings endpoint persists through `config_manager.save_config(...)`.

Install/bootstrap write paths:

- `install.sh`
  writes the initial `config.json` during installation/bootstrap.

### Related persisted metadata

These are not the main app settings/config files, but they are also written by the app:

- `config_manager.ensure_env_file()`
  ensures `.env` placeholder keys exist.
- `echomosaic_app/services/operations.py`
  writes restore-point metadata under `restorepoints/.../metadata.json`.
- `update_helpers.py`
  writes backup and restore metadata during update/rollback flows.

## Known Couplings To Respect During Extraction

- Route handlers and socket handlers share the same mutable globals.
- Settings save/debounce behavior is relied on widely across stream updates.
- Stream rendering, playback, AI status, and youtube sync all cross-call shared helpers in `app.py`.
- Flask app and Socket.IO are created early because decorators register at import time.
- Template/static paths must remain repo-root based even if code moves into packages.

## First Extraction Seams

- bootstrap and shared extension initialization
- canonical settings persistence
- config/runtime bootstrap
- route groups with low external coupling: debug, system, media admin
