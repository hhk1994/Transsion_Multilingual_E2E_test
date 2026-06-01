#!/usr/bin/env bash
# Install ASR/WER packages into e2e_test/.venv
set -euo pipefail

E2E_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="${E2E_ROOT}/.venv"
REQ="${E2E_ROOT}/requirements-asr.txt"

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "Missing ${VENV_DIR}. Run ./run_synthesize.sh or ./scripts/setup_venv.sh first." >&2
  exit 1
fi

# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"
pip install -U pip
pip install -r "${REQ}"
echo "[asr-deps] OK: faster-whisper + jiwer in ${VENV_DIR}"
