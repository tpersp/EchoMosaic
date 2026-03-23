# Tests

This repo uses a small pytest suite for backend smoke coverage and helper logic.
It is intentionally lightweight so AI agents can run it quickly before suggesting merges.

## What These Tests Cover

- `test_config_manager.py`
  Verifies config defaults, nested default merging, and environment overrides.
- `test_update_helpers.py`
  Verifies updater backup/restore behavior for user files, restore points, and repo-local media folders.
- `test_settings_store.py`
  Verifies the canonical settings persistence seam used by the de-monolith refactor.
- `test_config_runtime.py`
  Verifies the config/media runtime bootstrap seam used to move media root setup out of the monolith.
- `test_media_runtime.py`
  Verifies the extracted media cache runtime owner now provides the shared image and bad-media caches.
- `test_ai_hls_runtime.py`
  Verifies the extracted AI runtime and live-HLS runtime bootstrap/state seams.
- `test_youtube_runtime.py`
  Verifies the extracted YouTube cache/sync runtime seam.
- `test_stream_runtime.py`
  Verifies the extracted stream runtime cache/state seam.
- `test_stream_runtime_service.py`
  Verifies the extracted stream runtime service for media-kind inference, runtime updates, and UTC timestamp formatting.
- `test_playback_runtime.py`
  Verifies the extracted playback runtime defaults and manager ownership seam.
- `test_bootstrap_helpers.py`
  Verifies the shared blueprint registration helper preserves legacy endpoint aliases during the de-monolith refactor.
- `test_app_wrapper.py`
  Verifies the top-level `app.py` compatibility wrapper re-exports the server app and socket objects.
- `test_template_paths.py`
  Verifies the moved feature templates still render through the main page routes after the Phase 6 folder reorganization.
- `test_auto_schedulers.py`
  Verifies the extracted background scheduler builder can be instantiated safely outside the monolith.
- `test_operations_service.py`
  Verifies the extracted update/rollback/restore business logic service.
- `test_ai_orchestration_service.py`
  Verifies the extracted AI orchestration service for queueing and status behavior.
- `test_ai_execution_service.py`
  Verifies the extracted AI execution service for long-running generation completion behavior.
- `test_timer_sync_service.py`
  Verifies the extracted timer/sync service for sync timer creation, update, and deletion.
- `test_group_service.py`
  Verifies the extracted group/layout service for group CRUD and mosaic layout normalization.
- `test_stream_config_service.py`
  Verifies the extracted stream-configuration service for stream creation, settings normalization, and label validation.
- `test_playback_service.py`
  Verifies the extracted playback service for playback state retrieval and control actions.
- `test_playback_engine.py`
  Verifies the extracted playback engine can bootstrap and emit initial playback state outside the monolith.
- `test_media_library_service.py`
  Verifies the extracted read-side media library service for image/media queries and random selection.
- `test_media_catalog_service.py`
  Verifies the extracted cache-backed media catalog service for refresh and invalidation behavior.
- `test_asset_delivery_service.py`
  Verifies the extracted asset-delivery service for image serving and thumbnail metadata responses.
- `test_thumbnail_service.py`
  Verifies the extracted thumbnail generation service for snapshot computation and refresh behavior.
- `test_live_hls_service.py`
  Verifies the extracted live-HLS service for cache reuse and invalidation/reschedule behavior.
- `test_youtube_embed_service.py`
  Verifies the extracted YouTube/embed service for URL parsing, embed metadata refresh, and sync leader promotion.
- `test_diagnostics_blueprint.py`
  Verifies the first extracted blueprint registers the expected diagnostics/system routes.
- `test_dashboard_blueprint.py`
  Verifies the extracted dashboard/streams blueprint registers the expected routes.
- `test_media_blueprint.py`
  Verifies the extracted media-management blueprint registers the expected routes.
- `test_ai_blueprint.py`
  Verifies the extracted AI blueprint registers the expected routes.
- `test_settings_operations_blueprint.py`
  Verifies the extracted settings/update/restore blueprint registers the expected routes.
- `test_library_blueprint.py`
  Verifies the extracted tags/groups/timer/notes blueprint registers the expected routes.
- `test_live_blueprint.py`
  Verifies the extracted livestream/embed blueprint registers the expected routes.
- `test_assets_blueprint.py`
  Verifies the extracted media-library and stream-asset blueprint registers the expected routes.
- `test_stream_socket_handlers.py`
  Verifies the extracted stream/playback Socket.IO module registers the expected live events.
- `test_youtube_sync_socket_handlers.py`
  Verifies the extracted YouTube sync Socket.IO module registers the expected live event.

These tests do **not** exercise the live installed dev service directly.
They run against isolated temporary directories so they are safe to execute on the same server where the dev app is installed and running.

## Run The Tests

From the repo root:

```bash
cd /home/doden/workspace/EchoMosaic-Project/EchoMosaic-dev
./venv/bin/python -m pytest -q
```

If the local virtualenv does not exist yet:

```bash
cd /home/doden/workspace/EchoMosaic-Project/EchoMosaic-dev
python3 -m venv venv
./venv/bin/python -m pip install --upgrade pip setuptools wheel pytest pytest-cov
./venv/bin/python -m pip install -r requirements.txt
```

## Useful Variants

Run one file:

```bash
./venv/bin/python -m pytest -q tests/test_config_manager.py
```

Run one test:

```bash
./venv/bin/python -m pytest -q tests/test_update_helpers.py -k restore
```

Show extra output while debugging:

```bash
./venv/bin/python -m pytest -q -vv
```

## Notes For AI Agents

- Run tests from the repo root so imports resolve correctly.
- Prefer the repo-local `venv` instead of system Python.
- These tests are the first-pass safety net, not full application coverage.
- The installed dev app can stay running while these tests execute.
