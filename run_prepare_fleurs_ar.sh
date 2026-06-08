#!/usr/bin/env bash
# Download & sample 1000 FLEURS ar_eg clips (10–30s) for Whisper ASR eval.
#
# Usage:
#   ./run_prepare_fleurs_ar.sh
#   COUNT=500 MIN_DURATION=8 ./run_prepare_fleurs_ar.sh
set -euo pipefail

E2E_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "${E2E_ROOT}"

export COUNT="${COUNT:-1000}"
export MIN_DURATION="${MIN_DURATION:-10}"
export MAX_DURATION="${MAX_DURATION:-30}"
export SEED="${SEED:-42}"

VENV_DIR="${E2E_ROOT}/.venv"
if [[ ! -d "${VENV_DIR}" ]]; then
  echo "Missing ${VENV_DIR}. Run: bash scripts/setup_venv.sh" >&2
  exit 1
fi

# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"
pip install -q -r "${E2E_ROOT}/requirements-fleurs-prep.txt"

echo "=== Prepare FLEURS ar_eg (${COUNT} clips, ${MIN_DURATION}-${MAX_DURATION}s) ==="
python "${E2E_ROOT}/scripts/prepare_fleurs_ar.py" \
  --count "${COUNT}" \
  --min-duration "${MIN_DURATION}" \
  --max-duration "${MAX_DURATION}" \
  --seed "${SEED}"

echo
echo "Done. Next (Whisper ASR + WER):"
echo "  ./run_asr_wer_ar_fleurs.sh"
