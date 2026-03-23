# EchoMosaic Architecture Guide

This is the working guide for the post-monolith server layout in `EchoMosaic-dev`.
It describes where new code should go and how to extend the app without growing `app.py` again.

## Core Rule

`app.py` is the compatibility entrypoint only.
The real server implementation now lives in `echomosaic_app/server.py`, and new feature work should not start in `app.py` unless the change is strictly bootstrap-related.

## Server Module Map

### `echomosaic_app/routes/`

Use route blueprints for HTTP endpoints.

- Put new page routes and API routes in the blueprint that owns that feature area.
- Keep route functions thin: parse request input, call a service/helper, return a response.
- If a feature does not fit an existing blueprint, create a new feature blueprint instead of appending unrelated routes to another file.
- When practical, feature-owned templates should live under matching subfolders in `templates/`, and page-specific CSS/JS should live under matching subfolders in `static/`.

Current feature blueprints:

- `dashboard.py`
  Dashboard, stream views, stream CRUD/settings, Picsum refresh.
- `media.py`
  Media-management UI and admin API.
- `ai.py`
  AI presets, model lookup, AI generation/status API.
- `settings_operations.py`
  Settings export/import, restore points, update, rollback.
- `diagnostics.py`
  Health, system stats, debug streaming routes.
- `library.py`
  Tags, notes, timer settings, sync timers, groups, stream group view.
- `live.py`
  Livestream/embed resolution and embed testing.
- `assets.py`
  Media-library reads, stream image/video delivery, thumbnails.

### `echomosaic_app/sockets/`

Use socket modules for Socket.IO event registration.

- Register feature handlers from one module per live feature area.
- Keep event handlers thin and delegate state changes to services/helpers where possible.
- Preserve event names and payload shapes unless a dedicated compatibility change is planned.

Current socket modules:

- `streams.py`
  Stream subscribe/unsubscribe, AI watch/unwatch, playback control, disconnect cleanup.
- `youtube_sync.py`
  YouTube sync state propagation.

### `echomosaic_app/services/`

Use services for business logic and orchestration.

- Put behavior here when it is used by more than one route, by both HTTP and sockets, or when a route is getting too large.
- Prefer explicit constructor dependencies over hidden globals.
- Services should be importable and testable without booting the full app.

Current services:

- `operations.py`
- `ai_orchestration.py`
- `ai_execution.py`
- `timer_sync.py`
- `groups.py`
- `stream_config.py`
- `stream_runtime.py`
- `playback.py`
- `playback_engine.py`
- `media_catalog.py`
- `media_library.py`
- `live_hls.py`
- `youtube_embed.py`
- `auto_schedulers.py`
- `asset_delivery.py`
- `thumbnailing.py`

### `echomosaic_app/state/`

Use state modules for runtime state owners and persistence seams.

- Settings/config file access belongs here.
- Mutable runtime caches and shared locks belong here when they are cross-feature.
- New direct writes to `settings.json` should go through the settings store path rather than ad hoc file writes.
- Current runtime owners now cover AI, HLS, media caches, playback, stream state, and YouTube sync.

### `echomosaic_app/bootstrap.py`

Use bootstrap helpers for integration-level registration.

- Flask and Socket.IO creation
- bootstrap-time feature registration
- shared helpers for blueprint registration and compatibility aliasing
- dev server runner helpers

Do not put feature-specific request logic here.

### `echomosaic_app/server.py`

This is the current integration shell for the server.

- It wires together the extracted state owners, services, blueprints, and socket modules.
- Compatibility wrappers can still live here temporarily while refactor cleanup continues.
- When more legacy helpers are extracted, they should move out of this file rather than back into `app.py`.
- Template/static ownership now follows feature folders where practical:
  `templates/dashboard/`, `templates/streams/`, `templates/media/`, `templates/settings/`, `templates/diagnostics/`
  and matching `static/shared/`, `static/streams/`, `static/media/`.

## Feature Extension Workflow

When adding a new backend feature, follow this order:

1. Decide the feature owner.
   If it is HTTP-facing, start by picking or creating a blueprint module.
2. Add or extend a service if the behavior is more than request parsing.
3. Add/update state ownership only if the feature needs shared mutable runtime state or persistence.
4. Register the blueprint/socket module through bootstrap helpers rather than wiring aliases manually everywhere.
5. Add a focused test for the new seam:
   blueprint registration, service logic, or socket registration.

## Guardrails

Use these as review rules:

- Do not add new `@app.route(...)` handlers to `app.py` unless the route is strictly bootstrap/compatibility-related.
- Do not add new `@socketio.on(...)` handlers directly in `app.py`.
- Do not add new business logic to blueprint functions when it can live in a service.
- Do not create new ad hoc settings file write paths.
- Do not create a generic `utils.py` dumping ground. Shared helpers should only be promoted when they have clear multi-feature use.

## Practical Examples

If you add:

- a new page for group analytics:
  add a route in a feature blueprint, then add service logic if the page needs computed data.
- a new live playback event:
  put the Socket.IO registration in `echomosaic_app/sockets/`, and use a service if it mutates shared playback behavior.
- a new AI preset transform:
  add that logic in the AI service layer, not in the route handler.
- a new persisted app setting:
  update the settings/state seam and make the route call that seam.

## What Still Lives In `app.py`

The refactor is not fully complete yet. `echomosaic_app/server.py` still owns some larger legacy areas:

- stream runtime update helpers and related compatibility glue
- assorted low-level sanitizers and normalization helpers

New work in those areas should still try to preserve the same rule:
move the new logic toward feature modules instead of expanding the legacy surface.
