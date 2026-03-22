# Ready For Main

This file tracks the remaining work before `EchoMosaic-dev` should be merged into `main`.

## Status

- [x] Create a repo-local Python test environment (`venv`)
- [x] Install baseline test tooling (`pytest`, `pytest-cov`)
- [x] Add a small committed backend test suite
- [x] Add a short `tests/` usage guide for future AI agents
- [x] Run the starter test suite successfully
- [x] Fix installer/service-user mismatch
- [x] Fix update progress page disconnect handling
- [x] Escape the API key safely in the update progress page
- [x] Bring README/update docs back in sync with the current branch
- [x] Re-review `dev` after the above fixes and decide whether it is ready for `main`

## Why These Remaining Items Matter

### 1. Installer/Service User Mismatch

Status:
- Done. `install.sh` now makes the current-user install model explicit and no longer prompts for an arbitrary service user that the `systemd --user` flow cannot honor.

### 2. Update Progress Disconnect Handling

Status:
- Done. The update page now waits for an explicit restart/wait-for-restart signal before entering restart polling, so an ordinary mid-update disconnect is no longer treated as success.

### 3. API Key Escaping In `update_progress.html`

Status:
- Done. The update page now injects the API key with JSON-safe escaping via `|tojson`.

### 4. README / Docs Drift

Status:
- Done. `README.md` now reflects the current installer model, the split media-library configuration, the current update/restore-point flow, and the current front-end structure.

## Validation To Run Before Merging

Run from the repo root:

```bash
cd /home/doden/workspace/EchoMosaic-Project/EchoMosaic-dev
./venv/bin/python -m pytest -q
./venv/bin/python -m compileall .
bash -n install.sh
bash -n update.sh
```

Suggested manual checks after code fixes:
- Run the update page and confirm a normal update shows real progress.
- Confirm a temporary socket disconnect does not falsely mark the update as complete.
- Test `install.sh` with the intended install model and verify the resulting service actually runs the way the prompts describe.
- Confirm the README instructions still match the actual install/update flow.

## Merge Rule

Current recommendation:
- The tracked blockers in this file are resolved.
- `dev` is ready for `main` based on the code review and automated checks completed here.
- A quick live manual smoke test of the installed dev app is still recommended before the actual merge, but it is no longer a blocker in this checklist.
