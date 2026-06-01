#!/usr/bin/env bash
# Ensure bn-en (or other) checkpoint is a real file, not a Git LFS pointer.
set -euo pipefail

E2E_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LITS_ROOT="$(cd "${E2E_ROOT}/.." && pwd)"
MODEL_LANG="${MODEL_LANG:-bn-en}"
CKPT_PATH="${CKPT_PATH:-}"
if [[ -z "${CKPT_PATH}" ]]; then
  for cand in \
    "${E2E_ROOT}/models/${MODEL_LANG}.ckpt" \
    "${LITS_ROOT}/model_checkpoints/${MODEL_LANG}.ckpt"; do
    if [[ -f "${cand}" ]]; then
      CKPT_PATH="${cand}"
      break
    fi
  done
fi
CKPT_PATH="${CKPT_PATH:-${LITS_ROOT}/model_checkpoints/${MODEL_LANG}.ckpt}"
export CKPT_PATH
MIN_BYTES="${MIN_CKPT_BYTES:-1000000}"

if [[ -f "${CKPT_PATH}" ]]; then
  SIZE="$(stat -c%s "${CKPT_PATH}" 2>/dev/null || stat -f%z "${CKPT_PATH}")"
  if [[ "${SIZE}" -ge "${MIN_BYTES}" ]]; then
    echo "[ckpt] OK: ${CKPT_PATH} (${SIZE} bytes)"
    exit 0
  fi
fi

echo "[ckpt] Checkpoint missing or too small (LFS pointer?): ${CKPT_PATH}" >&2
echo "[ckpt] Try one of:" >&2
echo "  cd ${LITS_ROOT} && git lfs pull --include=model_checkpoints/${MODEL_LANG}.ckpt" >&2
echo "  CKPT_PATH=/path/to/${MODEL_LANG}.ckpt ./run_synthesize.sh" >&2
echo "  Or copy to: ${E2E_ROOT}/models/${MODEL_LANG}.ckpt" >&2
exit 1
