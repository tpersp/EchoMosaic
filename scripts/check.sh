#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_PYTHON="${REPO_DIR}/venv/bin/python"

cd "${REPO_DIR}"

if [[ ! -x "${VENV_PYTHON}" ]]; then
  echo "Missing repo-local virtualenv at ${VENV_PYTHON}"
  echo "Create it first with:"
  echo "  python3 -m venv venv"
  echo "  ./venv/bin/python -m pip install --upgrade pip setuptools wheel pytest pytest-cov"
  echo "  ./venv/bin/python -m pip install -r requirements.txt"
  exit 1
fi

echo "[check] Running pytest"
"${VENV_PYTHON}" -m pytest -q

echo "[check] Running compileall"
"${VENV_PYTHON}" -m compileall .

echo "[check] Syntax-checking install.sh"
bash -n install.sh

echo "[check] Syntax-checking update.sh"
bash -n update.sh

echo "[check] All validations passed"
