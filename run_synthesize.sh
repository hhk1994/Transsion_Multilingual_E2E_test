#!/usr/bin/env bash
# One-click TTS synthesis: venv + deps + LITs inference -> e2e_test/output/wav/<run_id>/
#
# Usage:
#   ./run_synthesize.sh
#   LIMIT=5 ./run_synthesize.sh
#   CKPT_PATH=/path/to/bn-en.ckpt ./run_synthesize.sh
#   SKIP_VENV=1 ./run_synthesize.sh    # reuse existing .venv
set -euo pipefail

E2E_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "${E2E_ROOT}"

export LOCALE="${LOCALE:-bn}"
export MODEL_LANG="${MODEL_LANG:-bn-en}"
export MANIFEST_TXT="${MANIFEST_TXT:-${E2E_ROOT}/output/tts_manifest.txt}"
export LIMIT="${LIMIT:-0}"
export INFER_ID="${INFER_ID:-$(date +%Y%m%d_%H%M%S)}"
export OUTPUT_RUN_DIR="${OUTPUT_RUN_DIR:-${E2E_ROOT}/output/wav/${INFER_ID}}"

echo "=== e2e_test TTS synthesis (model=${MODEL_LANG}) ==="
echo "  manifest: ${MANIFEST_TXT}"
echo "  output:   ${OUTPUT_RUN_DIR}"
echo

bash "${E2E_ROOT}/scripts/install_synth_deps.sh"

if [[ "${SKIP_VENV:-0}" != "1" ]] || [[ ! -d "${E2E_ROOT}/.venv" ]]; then
  bash "${E2E_ROOT}/scripts/setup_venv.sh"
fi

# Best-effort LFS pull when credentials are configured
LITS_ROOT="$(cd "${E2E_ROOT}/.." && pwd)"
if command -v git-lfs >/dev/null 2>&1; then
  if [[ -d "${LITS_ROOT}/.git" ]]; then
    (cd "${LITS_ROOT}" && git lfs pull --include="model_checkpoints/${MODEL_LANG}.ckpt" 2>/dev/null) || true
  fi
fi

bash "${E2E_ROOT}/scripts/synthesize.sh"

echo
echo "Done. Audio: ${OUTPUT_RUN_DIR}"
echo "  meta: ${OUTPUT_RUN_DIR}/meta.txt"
echo "  log:  ${OUTPUT_RUN_DIR}/synthesis.log"
