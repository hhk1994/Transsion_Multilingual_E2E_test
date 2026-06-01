#!/usr/bin/env bash
# Create e2e_test/.venv and install Multilingual_LITs inference dependencies.
set -euo pipefail

E2E_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="${E2E_ROOT}/.venv"
REQ_FILE="${REQ_FILE:-${E2E_ROOT}/requirements-synthesize-min.txt}"
if [[ "${USE_FULL_LITS_REQUIREMENTS:-0}" == "1" ]]; then
  REQ_FILE="${E2E_ROOT}/requirements-synthesize.txt"
fi
LITS_ROOT="$(cd "${E2E_ROOT}/.." && pwd)"

if [[ ! -f "${LITS_ROOT}/lits_requirements.txt" ]]; then
  echo "Multilingual_LITs repo root not found at ${LITS_ROOT}" >&2
  exit 1
fi

PYTHON="${PYTHON:-python3}"
if ! command -v "${PYTHON}" >/dev/null 2>&1; then
  echo "python3 not found" >&2
  exit 1
fi

if ! "${PYTHON}" -m venv --help >/dev/null 2>&1; then
  echo "[venv] python venv module missing; install python3-venv (see install_synth_deps.sh)" >&2
  exit 1
fi

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "[venv] Creating ${VENV_DIR} ..."
  "${PYTHON}" -m venv "${VENV_DIR}"
fi

# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"
python -m pip install -U pip wheel setuptools

echo "[venv] Installing from ${REQ_FILE} (may take several minutes)..."
pip install -r "${REQ_FILE}"

echo "[venv] Installing ttsfrd (optional frontend helper from ModelScope)..."
if ! pip install "ttsfrd==0.2.1" -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html; then
  echo "[venv] WARN: ttsfrd install failed; inference may still work for bn-en." >&2
fi

echo "[venv] OK: $(which python) ($(python -V))"
python -c "import torch; print('[venv] torch', torch.__version__, 'cuda', torch.cuda.is_available())"
