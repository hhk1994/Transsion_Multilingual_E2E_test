#!/usr/bin/env bash
# Run Multilingual_LITs streaming inference; write wav + meta under e2e_test/output/wav/<run_id>/
set -euo pipefail

E2E_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LITS_ROOT="$(cd "${E2E_ROOT}/.." && pwd)"
VENV_DIR="${E2E_ROOT}/.venv"

_abs_path() {
  local p="$1"
  if [[ "${p}" != /* ]]; then
    p="${E2E_ROOT}/${p#./}"
  fi
  local dir base
  base="$(basename "${p}")"
  dir="$(dirname "${p}")"
  mkdir -p "${dir}"
  echo "$(cd "${dir}" && pwd)/${base}"
}

LOCALE="${LOCALE:-bn}"
MODEL_LANG="${MODEL_LANG:-bn-en}"
MANIFEST_TXT="${MANIFEST_TXT:-${E2E_ROOT}/output/tts_manifest.txt}"
MANIFEST_TXT="$(_abs_path "${MANIFEST_TXT}")"
CKPT_PATH="${CKPT_PATH:-${LITS_ROOT}/model_checkpoints/${MODEL_LANG}.ckpt}"
if [[ "${CKPT_PATH}" != /* ]]; then
  CKPT_PATH="$(cd "${LITS_ROOT}" && cd "$(dirname "${CKPT_PATH}")" && pwd)/$(basename "${CKPT_PATH}")"
fi
LIMIT="${LIMIT:-0}"
INFER_ID="${INFER_ID:-$(date +%Y%m%d_%H%M%S)}"

OUTPUT_RUN_DIR="${OUTPUT_RUN_DIR:-${E2E_ROOT}/output/wav/${INFER_ID}}"
OUTPUT_RUN_DIR="$(_abs_path "${OUTPUT_RUN_DIR}")"
OUTPUT_WAV_DIR="${OUTPUT_WAV_DIR:-${OUTPUT_RUN_DIR}}"
if [[ "${OUTPUT_WAV_DIR}" != /* ]]; then
  OUTPUT_WAV_DIR="$(_abs_path "${OUTPUT_WAV_DIR}")"
fi
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

VOCODER_TYPE="${VOCODER_TYPE:-vocos}"
VOCOS_CHECKPOINT="${VOCOS_CHECKPOINT:-${LITS_ROOT}/vocos/generator.ckpt}"
VOCOS_ROOT="${VOCOS_ROOT:-${LITS_ROOT}}"
HIFIGAN_CHECKPOINT="${HIFIGAN_CHECKPOINT:-}"
HIFIGAN_CONFIG="${HIFIGAN_CONFIG:-auto}"
LEGACY_SYMBOLS_FILE="${LEGACY_SYMBOLS_FILE:-}"
OUTPUT_SAMPLE_RATE="${OUTPUT_SAMPLE_RATE:-}"

if [[ "${VOCODER_TYPE}" == "vocos" ]]; then
  OUTPUT_SAMPLE_RATE="${OUTPUT_SAMPLE_RATE:-24000}"
  if [[ ! -f "${VOCOS_CHECKPOINT}" ]]; then
    echo "Vocos checkpoint not found: ${VOCOS_CHECKPOINT}" >&2
    exit 1
  fi
elif [[ "${MODEL_LANG}" == "bn-en" ]]; then
  AFNAS_FRONTEND_SR=22050
else
  AFNAS_FRONTEND_SR=24000
fi

VOCODER_DIR="${VOCODER_DIR:-${LITS_ROOT}/afnas_pupuvocoder_mix22050_24k_100band}"
if [[ "${VOCODER_TYPE}" == "afnas" ]]; then
  OUTPUT_SAMPLE_RATE="${OUTPUT_SAMPLE_RATE:-24000}"
  if [[ ! -f "${VOCODER_DIR}/afnas_generator.pt" ]]; then
    echo "Vocoder weights not found: ${VOCODER_DIR}/afnas_generator.pt" >&2
    exit 1
  fi
elif [[ "${VOCODER_TYPE}" == "hifigan" ]]; then
  if [[ -z "${HIFIGAN_CHECKPOINT}" ]]; then
    echo "HIFIGAN_CHECKPOINT is required when VOCODER_TYPE=hifigan" >&2
    exit 1
  fi
  if [[ ! -f "${HIFIGAN_CHECKPOINT}" ]]; then
    echo "HiFi-GAN checkpoint not found: ${HIFIGAN_CHECKPOINT}" >&2
    exit 1
  fi
  if [[ "${HIFIGAN_CHECKPOINT}" != /* ]]; then
    HIFIGAN_CHECKPOINT="$(cd "$(dirname "${HIFIGAN_CHECKPOINT}")" && pwd)/$(basename "${HIFIGAN_CHECKPOINT}")"
  fi
  if [[ "${HIFIGAN_CONFIG}" == "v1_16k" ]] || [[ "${HIFIGAN_CONFIG}" == "auto" ]]; then
    if [[ -n "${OUTPUT_SAMPLE_RATE}" && "${OUTPUT_SAMPLE_RATE}" != "16000" ]]; then
      echo "[synth] NOTE: legacy HiFi-GAN uses 16 kHz; overriding OUTPUT_SAMPLE_RATE=${OUTPUT_SAMPLE_RATE} -> 16000" >&2
    fi
    OUTPUT_SAMPLE_RATE=16000
  else
    OUTPUT_SAMPLE_RATE="${OUTPUT_SAMPLE_RATE:-22050}"
  fi
elif [[ "${VOCODER_TYPE}" != "vocos" ]]; then
  echo "Unsupported VOCODER_TYPE: ${VOCODER_TYPE} (expected vocos, afnas, or hifigan)" >&2
  exit 1
fi

echo "[synth] model=${MODEL_LANG} manifest=${RUN_MANIFEST}"
echo "[synth] output_dir=${OUTPUT_WAV_DIR}"
echo "[synth] log=${SYNTH_LOG}"

cd "${LITS_ROOT}"

set +e
SYNTH_ARGS=(
  --model_lang "${MODEL_LANG}"
  --checkpoint "${CKPT_PATH}"
  --input_txt "${RUN_MANIFEST}"
  --output_dir "${OUTPUT_WAV_DIR}"
  --output_txt "${META_TXT}"
  --vocoder_type "${VOCODER_TYPE}"
  --output_sample_rate "${OUTPUT_SAMPLE_RATE}"
  --num_decoding_left_chunks -1
  --text_normalization off
)

if [[ "${VOCODER_TYPE}" == "vocos" ]]; then
  SYNTH_ARGS+=(
    --vocos_checkpoint "${VOCOS_CHECKPOINT}"
    --vocos_root "${VOCOS_ROOT}"
  )
elif [[ "${VOCODER_TYPE}" == "afnas" ]]; then
  SYNTH_ARGS+=(
    --vocoder_dir "${VOCODER_DIR}"
    --afnas_frontend_sr "${AFNAS_FRONTEND_SR}"
  )
else
  SYNTH_ARGS+=(
    --hifigan_checkpoint "${HIFIGAN_CHECKPOINT}"
    --hifigan_config "${HIFIGAN_CONFIG}"
  )
  if [[ -n "${LEGACY_SYMBOLS_FILE}" ]]; then
    if [[ "${LEGACY_SYMBOLS_FILE}" != /* ]]; then
      LEGACY_SYMBOLS_FILE="${E2E_ROOT}/${LEGACY_SYMBOLS_FILE#./}"
    fi
    if [[ ! -f "${LEGACY_SYMBOLS_FILE}" ]]; then
      echo "Legacy symbols file not found: ${LEGACY_SYMBOLS_FILE}" >&2
      exit 1
    fi
    SYNTH_ARGS+=(--legacy_symbols_file "${LEGACY_SYMBOLS_FILE}")
  fi
fi

python inference_stream.py "${SYNTH_ARGS[@]}" \
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
