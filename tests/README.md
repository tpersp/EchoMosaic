# Tests

This repo uses a small pytest suite for backend smoke coverage and helper logic.
It is intentionally lightweight so AI agents can run it quickly before suggesting merges.

## What These Tests Cover

- `test_config_manager.py`
  Verifies config defaults, nested default merging, and environment overrides.
- `test_update_helpers.py`
  Verifies updater backup/restore behavior for user files, restore points, and repo-local media folders.

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
