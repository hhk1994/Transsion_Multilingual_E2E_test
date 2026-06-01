#!/usr/bin/env bash
# Run Multilingual_LITs streaming inference; write wav + meta under e2e_test/output/wav/<run_id>/
set -euo pipefail

E2E_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LITS_ROOT="$(cd "${E2E_ROOT}/.." && pwd)"
VENV_DIR="${E2E_ROOT}/.venv"

LOCALE="${LOCALE:-bn}"
MODEL_LANG="${MODEL_LANG:-bn-en}"
MANIFEST_TXT="${MANIFEST_TXT:-${E2E_ROOT}/output/tts_manifest.txt}"
CKPT_PATH="${CKPT_PATH:-${LITS_ROOT}/model_checkpoints/${MODEL_LANG}.ckpt}"
LIMIT="${LIMIT:-0}"
INFER_ID="${INFER_ID:-$(date +%Y%m%d_%H%M%S)}"

OUTPUT_RUN_DIR="${OUTPUT_RUN_DIR:-${E2E_ROOT}/output/wav/${INFER_ID}}"
OUTPUT_WAV_DIR="${OUTPUT_WAV_DIR:-${OUTPUT_RUN_DIR}}"
META_TXT="${META_TXT:-${OUTPUT_RUN_DIR}/meta.txt}"
SYNTH_LOG="${SYNTH_LOG:-${OUTPUT_RUN_DIR}/synthesis.log}"

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "Virtualenv not found. Run: ${E2E_ROOT}/scripts/setup_venv.sh" >&2
  exit 1
fi

if [[ ! -f "${MANIFEST_TXT}" ]]; then
  echo "Manifest not found: ${MANIFEST_TXT}" >&2
  echo "Run: ${E2E_ROOT}/run_build_manifest.sh" >&2
  exit 1
fi

export MODEL_LANG CKPT_PATH
bash "${E2E_ROOT}/scripts/ensure_checkpoint.sh"

# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

RUN_MANIFEST="${MANIFEST_TXT}"
TMP_MANIFEST=""
if [[ "${LIMIT}" =~ ^[0-9]+$ ]] && [[ "${LIMIT}" -gt 0 ]]; then
  TMP_MANIFEST="$(mktemp)"
  head -n "${LIMIT}" "${MANIFEST_TXT}" > "${TMP_MANIFEST}"
  RUN_MANIFEST="${TMP_MANIFEST}"
  echo "[synth] LIMIT=${LIMIT}"
fi

cleanup() {
  if [[ -n "${TMP_MANIFEST}" && -f "${TMP_MANIFEST}" ]]; then
    rm -f "${TMP_MANIFEST}"
  fi
}
trap cleanup EXIT

mkdir -p "${OUTPUT_WAV_DIR}"

if [[ "${MODEL_LANG}" == "bn-en" ]]; then
  AFNAS_FRONTEND_SR=22050
else
  AFNAS_FRONTEND_SR=24000
fi

VOCODER_DIR="${VOCODER_DIR:-${LITS_ROOT}/afnas_pupuvocoder_mix22050_24k_100band}"
if [[ ! -f "${VOCODER_DIR}/afnas_generator.pt" ]]; then
  echo "Vocoder weights not found: ${VOCODER_DIR}/afnas_generator.pt" >&2
  exit 1
fi

echo "[synth] model=${MODEL_LANG} manifest=${RUN_MANIFEST}"
echo "[synth] output_dir=${OUTPUT_WAV_DIR}"
echo "[synth] log=${SYNTH_LOG}"

cd "${LITS_ROOT}"

set +e
python inference_stream.py \
  --model_lang "${MODEL_LANG}" \
  --checkpoint "${CKPT_PATH}" \
  --input_txt "${RUN_MANIFEST}" \
  --output_dir "${OUTPUT_WAV_DIR}" \
  --output_txt "${META_TXT}" \
  --vocoder_dir "${VOCODER_DIR}" \
  --afnas_frontend_sr "${AFNAS_FRONTEND_SR}" \
  --output_sample_rate 24000 \
  --num_decoding_left_chunks -1 \
  --text_normalization off \
  2>&1 | tee "${SYNTH_LOG}"
PIPE_STATUS="${PIPESTATUS[0]}"
set -e

if [[ "${PIPE_STATUS}" -ne 0 ]]; then
  echo "[synth] FAILED (exit ${PIPE_STATUS}). See ${SYNTH_LOG}" >&2
  exit "${PIPE_STATUS}"
fi

N_WAV="$(find "${OUTPUT_WAV_DIR}" -maxdepth 1 -name '*.wav' 2>/dev/null | wc -l | tr -d ' ')"
echo "[synth] wrote ${N_WAV} wav file(s) under ${OUTPUT_WAV_DIR}"

ln -snf "${OUTPUT_RUN_DIR}" "${E2E_ROOT}/output/wav/latest"
echo "[synth] latest -> output/wav/latest"
echo "[synth] OK"
