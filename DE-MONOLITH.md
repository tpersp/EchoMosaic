# De-Monolith Plan For EchoMosaic

## Summary

Refactor EchoMosaic from a single large `app.py` into a feature-oriented package built around Flask blueprints plus service modules, while preserving current routes, Socket.IO events, settings formats, and user-facing behavior.

The goal is not only to shrink `app.py`, but to create an architecture that makes new features easier to add without touching unrelated systems. The end state should make it obvious where to place:
- new HTTP routes
- new Socket.IO handlers
- new background services
- new persistence/state helpers
- new front-end feature assets

The refactor should be phased, testable after each phase, and should avoid a big-bang rewrite.

## Target Architecture

### App structure

Adopt an app package layout with these roles:

- `app/` package as the new source root for server code
- `app/__init__.py` for Flask app and Socket.IO creation plus module registration
- `app/extensions.py` for shared instances like `socketio`
- `app/config_runtime.py` for loaded config, media roots, and runtime bootstrap objects
- `app/state/` for persisted settings/config access and in-memory runtime state
- `app/services/` for business logic and background orchestration
- `app/routes/` for HTTP blueprints grouped by feature
- `app/sockets/` for Socket.IO event registration grouped by feature
- `app/utils/` for isolated low-level helpers that are genuinely cross-cutting

Keep `main.py` and the top-level `app.py` as thin compatibility entrypoints during the transition.

### Feature boundaries

Split by feature, not by technical layer alone. The first-class feature areas should be:

- dashboard and stream configuration
- stream rendering and playback
- media browsing/management
- AI generation and presets
- timers and sync timers
- groups and mosaic views
- settings, update, rollback, restore points
- debug/log streaming
- system stats/health

### Rules for future modules

Use these rules as part of the refactor so the architecture stays healthy:

- New HTTP endpoints must live in a feature blueprint, not in the app bootstrap file.
- New Socket.IO events must be registered from a feature socket module.
- New background behaviors must be implemented as services with explicit dependencies.
- New settings mutations must go through shared state helpers, not direct ad hoc file writes.
- Templates and static assets should be owned by the same feature area they serve when practical.
- Shared helpers should only be created after at least two concrete feature consumers exist.

## Public Interfaces And Compatibility

Preserve these during the refactor unless a later separate cleanup project is approved:

- all existing route URLs
- all existing HTTP request/response shapes
- all existing Socket.IO event names and payload shapes
- `settings.json` layout
- `config.json` and `.env` semantics
- install/update script behavior
- current template URLs and page entrypoints
- current `python app.py` and `python main.py` workflow

Add compatibility shims as needed so internal movement does not require front-end rewrites all at once.

## Progress Checklist

### Overall status

- [x] Phase 0 complete
- [x] Phase 1 complete
- [x] Phase 2 complete
- [x] Phase 3 complete
- [x] Phase 4 complete
- [x] Phase 5 complete
- [x] Phase 6 complete
- [x] Phase 7 complete

### Current progress snapshot

- [x] Baseline architecture note added in `docs/de-monolith-baseline.md`
- [x] Route and Socket.IO inventory documented
- [x] Package/bootstrap seam created under `echomosaic_app/`
- [x] Full server implementation relocated under `echomosaic_app/server.py`
- [x] Shared `socketio` bootstrap extracted
- [x] Canonical settings persistence extracted
- [x] Direct `settings.json` and `config.json` write paths documented
- [x] Config/media runtime bootstrap extracted
- [x] AI runtime bootstrap/state extracted
- [x] Live-HLS runtime bootstrap/state extracted
- [x] Live-HLS orchestration service extracted
- [x] Auto-generation and Picsum scheduler module extracted
- [x] Asset delivery and thumbnail route service extracted
- [x] AI execution worker service extracted
- [x] Thumbnail generation/render service extracted
- [x] Media catalog/cache service extracted
- [x] Playback engine extracted
- [x] YouTube/embed metadata and sync service extracted
- [x] Stream runtime metadata service extracted
- [x] YouTube cache/sync runtime state extracted
- [x] Stream runtime state extracted
- [x] Playback runtime state extracted
- [x] Media/shared cache runtime owners extracted
- [x] Route groups extracted into blueprints
- [x] Socket.IO handlers extracted into feature modules
- [x] Remaining misc/live/asset routes extracted into dedicated blueprints
- [x] Front-end assets reorganized by feature
- [x] Top-level `app.py` reduced to compatibility wrapper only
- [x] Module placement and feature-extension guide added
- [x] Clean up trailing newline-only diff in `settings.json` as final housekeeping

## Where We're At

### Current state of the refactor

- The app now has a real package structure under `echomosaic_app/` with dedicated `routes/`, `sockets/`, `services/`, and `state/` folders.
- The repo now has an explicit architecture guide in `docs/architecture-guide.md` to document module placement and extension workflow.
- The main HTTP route groups have been extracted into blueprints:
  - dashboard/streams
  - media management
  - AI
  - settings/operations
  - diagnostics/system
  - tags/groups/timers/notes
  - live/embed handling
  - media-library and stream-asset delivery
- The main Socket.IO event groups have been extracted into feature modules:
  - stream/playback live events
  - YouTube sync live event
- Several shared runtime/state seams have already been pulled out of `app.py`:
  - settings store
  - config/media runtime
  - AI runtime
  - HLS runtime
  - YouTube runtime
  - stream runtime
  - playback runtime
- Service extraction has started for business logic:
  - update/rollback/restore operations service extracted
  - AI orchestration service extracted
  - timer/sync service extracted
  - group/layout service extracted
  - stream-configuration service extracted
- playback service extracted
  - playback engine extracted
  - media-library service extracted
  - stream runtime metadata service extracted
  - live-HLS detection/cache service extracted
  - background auto-scheduler module extracted
  - asset delivery/thumbnail route service extracted
  - AI execution worker service extracted
  - thumbnail generation/render service extracted
  - YouTube/embed metadata and sync service extracted

### What still lives heavily in the integration shell

- mostly compatibility glue and a smaller set of miscellaneous helper/util functions in `echomosaic_app/server.py`

### What is already validated

- `./venv/bin/python -m pytest -q`
- `./venv/bin/python -m compileall app.py echomosaic_app tests`
- `bash -n install.sh`
- `bash -n update.sh`
- import/registration smoke checks have been run after each major extraction pass

### Current test baseline

- Current automated suite status: `68 passed`
- Existing warning still present: Eventlet deprecation warning from the settings-store import path
- No functional drift has been seen from the automated checks during the latest passes
- Page-render smoke checks now pass for `/`, `/media/manage`, `/settings`, `/debug`, and `/update`

### Current repo note

- The deferred `settings.json` newline-only diff has been cleaned up
- The remaining recommended validation is now live/manual testing in the browser and against the running dev instance

### Recommended next step when we resume

- The de-monolith plan is complete for the scope defined in this file
- The next step is live/manual verification of the running app
- If manual testing finds a regression, fix it against the extracted module/service/blueprint seams rather than reintroducing logic into compatibility wrappers

### Restart guidance

- Treat this file as the handoff source of truth before resuming work
- Re-read:
  - `DE-MONOLITH.md`
  - `docs/de-monolith-baseline.md`
  - `docs/architecture-guide.md`
  - current `app.py` imports and service/blueprint registrations
- Resume from live/manual verification or any regression follow-up rather than redoing completed extraction work

### Phase-by-phase checklist

#### Phase 0 checklist

- [x] Inventory route handlers
- [x] Inventory Socket.IO handlers
- [x] Add baseline architecture note
- [x] Add first seam tests for refactor-safe extraction
- [x] Document every direct `settings.json` and `config.json` write path
- [x] Run baseline pytest and compile checks

#### Phase 1 checklist

- [x] Create application package
- [x] Extract shared Socket.IO instance
- [x] Extract Flask/bootstrap helper
- [x] Keep `main.py` as compatibility runner
- [x] Reduce top-level `app.py` to bootstrap-only compatibility shell
- [x] Verify app import/startup path still works

#### Phase 2 checklist

- [x] Extract canonical settings store
- [x] Extract config/media runtime bootstrap
- [x] Extract AI runtime bootstrap/state
- [x] Extract HLS runtime bootstrap/state
- [x] Extract YouTube cache/sync runtime state
- [x] Extract stream runtime state
- [x] Extract playback runtime state
- [x] Replace remaining ad hoc global state owners

#### Phase 3 checklist

- [x] Extract media service module
- [x] Extract stream configuration service module
- [x] Extract playback service module
- [x] Extract AI orchestration service module
- [x] Extract timer/sync service modules
- [x] Extract group/layout service module
- [x] Extract update/rollback service module
- [x] Extract live-HLS service module
- [x] Extract auto-scheduler module
- [x] Extract asset delivery service module
- [x] Extract AI execution worker service module
- [x] Extract thumbnail generation/render service module

#### Phase 4 checklist

- [x] Create dashboard/streams blueprint
- [x] Create media blueprint
- [x] Create AI blueprint
- [x] Create settings/operations blueprint
- [x] Create debug/system blueprint
- [x] Create tags/groups/timers/notes blueprint
- [x] Create live/embed blueprint
- [x] Create media-library/asset blueprint
- [x] Preserve route URLs while moving handlers

#### Phase 5 checklist

- [x] Extract stream subscribe/unsubscribe handlers
- [x] Extract AI watch/unwatch handlers
- [x] Extract video control handlers
- [x] Extract YouTube sync handlers
- [x] Register sockets from feature modules

#### Phase 6 checklist

- [x] Reorganize templates by feature ownership
- [x] Reorganize static assets by feature ownership
- [x] Keep current page behavior/URLs intact

#### Phase 7 checklist

- [x] Shrink `app.py` to compatibility-only wrapper
- [x] Document module placement rules
- [x] Document feature-extension workflow
- [x] Add guardrails against re-growing the monolith

## Phased Implementation Plan

### Phase 0: Baseline and safety rails

Create the baseline needed to refactor safely before moving code.

Changes:
- Inventory all route handlers, Socket.IO handlers, background schedulers, global state objects, and persistence touchpoints.
- Document current ownership of:
  - config loading
  - settings loading/saving
  - AI job state
  - playback runtime state
  - timer runtime state
  - media manager bootstrap
- Add a short architecture note describing current domains and hidden couplings.
- Expand smoke-test coverage around the most fragile stable behaviors before extraction begins.

Checks:
- Route inventory is complete and matches the current app.
- Socket.IO event inventory is complete.
- Every direct write to `settings.json` and `config.json` is identified.
- Existing tests still pass.
- `python -m compileall .` succeeds.

Acceptance:
- We have a trustworthy map of what exists today and where the dangerous couplings are.
- No repo behavior changes yet.

### Phase 1: Create the application package and bootstrap seam

Introduce the new package structure without changing behavior.

Changes:
- Create `app/` package and move app creation/bootstrap code behind a single initializer.
- Move Flask app creation and Socket.IO initialization into package bootstrap code.
- Keep top-level `app.py` and `main.py` as thin wrappers that import from the new package.
- Move shared extension objects and app registration code out of the monolith.
- Add a registration pattern for blueprints and socket modules, even if only one or two are initially wired in.

Checks:
- `python app.py` still starts the server.
- `python main.py` still starts the server.
- Existing imports used by helper modules still resolve.
- The current dashboard, settings page, media page, and stream page still load.
- Existing tests still pass.

Acceptance:
- The app has a real bootstrap seam.
- New features can be registered without editing a giant entrypoint.

### Phase 2: Centralize persistence and runtime state

Extract the hidden shared state first so later route moves are safer.

Changes:
- Create a settings state module for:
  - loading settings
  - atomic saving
  - debounced save behavior
  - settings integrity normalization
  - import/export helpers
- Create a config/runtime module for:
  - merged config
  - media roots
  - AI roots
  - media manager bootstrap
  - stable horde client bootstrap
- Create runtime state modules for:
  - stream runtime state
  - AI job runtime state
  - playback runtime state
  - sync/youtube runtime state
- Replace direct global access with explicit imports from state/service modules.

Checks:
- There is exactly one canonical settings save path.
- Route handlers no longer define their own ad hoc persistence behavior.
- Settings import/export still preserves current format.
- AI jobs still survive UI disconnect behavior as they do now.
- Existing tests pass, plus new state-module tests where practical.

Acceptance:
- The hardest-to-reason-about globals are isolated.
- Later route extraction no longer depends on copying hidden shared globals around.

### Phase 3: Extract feature services before feature routes

Move business logic out of handlers before moving the handlers themselves.

Changes:
- Create service modules for:
  - media catalog and path resolution
  - stream configuration normalization
  - playback coordination
  - AI generation orchestration
  - timer scheduling
  - sync timer coordination
  - groups/layout preparation
  - update/rollback/restore-point orchestration
  - embed/livestream resolution
- Route handlers should become thin adapters that validate input, call a service, and format a response.
- Socket handlers should become thin adapters that call services and emit events.

Checks:
- Feature services can be imported and unit-tested without starting the full web app.
- Route handlers shrink materially and mostly delegate.
- No duplicate business logic remains between routes and socket handlers.
- Existing route behavior remains unchanged.
- Existing tests pass.

Acceptance:
- The core complexity now lives in reusable modules instead of transport handlers.
- New features can be added by composing services instead of editing one monolith.

### Phase 4: Extract HTTP routes into blueprints

Once services exist, move endpoints into blueprint modules by feature.

Changes:
- Create blueprints for:
  - dashboard and streams
  - media management
  - AI
  - settings and operations
  - debug
  - health/system
- Register blueprints from app bootstrap.
- Preserve current URLs exactly.
- Keep feature-local helper functions near their blueprint unless they are used cross-feature.

Recommended grouping:
- dashboard and stream views together
- media API and media page together
- AI routes and preset routes together
- settings/update/restore routes together
- debug/system routes together

Checks:
- Route registration matches the pre-refactor inventory.
- No routes disappear or change URL.
- Template rendering still works from moved modules.
- Main pages and representative API endpoints respond correctly.
- Existing tests pass.

Acceptance:
- Adding a new HTTP feature no longer requires touching a god file.
- Feature ownership is visible from the directory structure.

### Phase 5: Extract Socket.IO handlers into feature modules

Move websocket behavior out of the monolith in the same feature-oriented way.

Changes:
- Create socket registration modules for:
  - stream subscribe/unsubscribe
  - AI watch/unwatch
  - video control
  - youtube sync
  - any future feature-scoped live events
- Use one registration function per feature module that accepts `socketio` and required services/state.
- Keep event names and payloads unchanged.

Checks:
- Socket event inventory still matches pre-refactor behavior.
- Stream subscribe/unsubscribe still joins and leaves rooms correctly.
- AI progress updates still emit correctly.
- Video control actions still propagate correctly.
- Disconnect behavior still cleans up listeners and leadership state.
- Manual websocket smoke verification passes.

Acceptance:
- New live behaviors can be added in isolated modules.
- Socket logic no longer competes with route logic in the same file.

### Phase 6: Split templates/static ownership by feature

Make front-end extension easier by improving ownership boundaries.

Changes:
- Reorganize templates and front-end assets so feature ownership is clearer.
- Move large page-specific JavaScript into feature-specific files if not already separated.
- Introduce small shared front-end utilities only where duplication is real.
- Keep user-visible behavior and URLs the same.

Checks:
- Dashboard still renders and edits streams.
- Media manager still navigates folders, thumbnails, uploads, and previews.
- Stream pages still render image/video/embed behavior.
- Settings page still supports update/rollback/restore operations.
- No broken asset references after reorganization.

Acceptance:
- New front-end features have an obvious home.
- Shared UI concerns are separated from page-specific logic.

### Phase 7: Shrink the compatibility shell and enforce boundaries

Finish the refactor by making the architecture hard to regress.

Changes:
- Reduce top-level `app.py` to a tiny compatibility wrapper only.
- Add lightweight architecture rules to docs:
  - where routes go
  - where sockets go
  - where services go
  - how persistence must be handled
- Add basic checks or review rules to discourage new business logic in bootstrap files.
- Update README/developer docs with the new module map and feature-extension workflow.

Checks:
- Top-level `app.py` is bootstrap-only.
- No feature logic remains in the compatibility wrapper.
- Docs explain how to add a new route, socket event, service, and page.
- Existing tests pass.
- `./scripts/check.sh` passes.

Acceptance:
- The monolith is gone in practice, not only renamed.
- Future work has a stable pattern to follow.

## Suggested Module Layout

A concrete end-state target:

```text
app/
  __init__.py
  extensions.py
  config_runtime.py
  state/
    settings_store.py
    runtime_state.py
    ai_state.py
    playback_state.py
    sync_state.py
  services/
    streams.py
    playback.py
    media_library.py
    media_admin.py
    ai_generation.py
    timers.py
    groups.py
    embeds.py
    operations.py
  routes/
    dashboard.py
    streams.py
    media.py
    ai.py
    settings.py
    debug.py
    system.py
  sockets/
    streams.py
    ai.py
    playback.py
    youtube_sync.py
  utils/
    responses.py
    validation.py
```

This is the target shape, not a requirement to move every helper one-to-one.

## Phase-by-Phase Validation Matrix

Run these after each meaningful extraction phase:

- `./venv/bin/python -m pytest -q`
- `./venv/bin/python -m compileall .`
- `bash -n install.sh`
- `bash -n update.sh`

Manual smoke checks after phases 1, 4, 5, and 6:

- dashboard loads
- add stream works
- stream settings save
- `/stream/<name>` renders correctly
- `/stream` mosaic view renders correctly
- media manager loads and browses folders
- AI preset listing and AI status endpoints respond
- settings page loads update/restore data
- debug page log stream opens
- websocket-backed playback actions still function
- AI progress updates still arrive during generation
- group view renders existing group layouts

## Test Plan

### New automated tests to add during refactor

Add focused tests around extracted seams, not broad end-to-end complexity.

Priority test areas:
- settings store load/save/debounce/import-export behavior
- settings integrity normalization
- route registration smoke test
- blueprint registration smoke test
- key service-unit tests for:
  - AI orchestration
  - timer calculations
  - media path resolution
  - group layout normalization
  - update/restore orchestration where practical
- socket registration smoke tests if the current test stack can support them

### Regression scenarios to explicitly verify

- settings import does not drop stream metadata
- debounced settings save still flushes on shutdown
- AI job state remains available after client disconnect
- stream subscribe/unsubscribe still manages playback state correctly
- youtube sync leader reassignment still works on disconnect
- restore points still list/create/delete/restore correctly
- update flow still emits progress correctly
- media virtual path handling still rejects root escapes
- hidden/internal media directories remain excluded

## Refactor Sequencing Rules

To keep risk low, follow these rules throughout implementation:

- Extract state before routes.
- Extract services before blueprints.
- Move one feature area at a time.
- Validate after every feature move.
- Prefer temporary compatibility wrappers over multi-feature rewrites.
- Do not rename public endpoints during this project.
- Do not change settings schema unless required for correctness.
- Do not mix behavior changes with structural refactor unless fixing a blocker.

## Definition Of Done

This project is complete when:

- top-level `app.py` is only a compatibility bootstrap
- feature routes live in blueprints
- Socket.IO events live in feature socket modules
- settings/config/runtime state have explicit owners
- core business logic is service-based rather than route-based
- route URLs, socket event names, and persisted data formats remain compatible
- docs explain how to add a new feature/module cleanly
- validation suite and manual smoke checks pass

## Assumptions And Defaults

- Target architecture is blueprint-based rather than a minimal module split.
- Compatibility is strict: preserve routes, sockets, and persisted formats.
- This is a structural refactor project, not a feature redesign project.
- Existing templates can remain in place initially and be reorganized later in the plan.
- Large helper functions should only be promoted to shared utilities when they have multiple consumers.
- Temporary duplication is acceptable during migration if it reduces risk, but it should be removed before the final phase.
