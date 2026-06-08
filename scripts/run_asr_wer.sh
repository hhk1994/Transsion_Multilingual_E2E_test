#!/usr/bin/env bash
set -euo pipefail

E2E_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="${E2E_ROOT}/.venv"

WAV_DIR="${WAV_DIR:-${E2E_ROOT}/output/wav/bn_smoke_5}"
NORMALIZED_TXT="${NORMALIZED_TXT:-${E2E_ROOT}/output/normalized.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-${E2E_ROOT}/output/asr/$(basename "${WAV_DIR}")}"
LIMIT="${LIMIT:-0}"
ASR_BACKEND="${ASR_BACKEND:-auto}"
WHISPER_MODEL="${WHISPER_MODEL:-large-v3}"
WHISPER_DEVICE="${WHISPER_DEVICE:-cuda}"
WHISPER_LANG="${WHISPER_LANG:-bn}"
DECODING="${DECODING:-ctc}"
TN_NORMALIZE_HYP="${TN_NORMALIZE_HYP:-1}"
TN_BIN="${TN_BIN:-${E2E_ROOT}/bin/bn_tts}"
UTT_PREFIX="${UTT_PREFIX:-}"

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "Missing ${VENV_DIR}. Create venv via ./run_synthesize.sh first." >&2
  exit 1
fi

if [[ ! -d "${WAV_DIR}" ]]; then
  echo "WAV dir not found: ${WAV_DIR}" >&2
  exit 1
fi

if [[ "${TN_NORMALIZE_HYP}" != "0" ]] && [[ ! -x "${TN_BIN}" ]]; then
  echo "TN binary not found/executable: ${TN_BIN}" >&2
  echo "Run ./run_normalize.sh once (or set TN_BIN) before ASR WER." >&2
  exit 1
fi

# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

ARGS=(
  --wav-dir "${WAV_DIR}"
  --normalized "${NORMALIZED_TXT}"
  --output-dir "${OUTPUT_DIR}"
  --backend "${ASR_BACKEND}"
  --model "${WHISPER_MODEL}"
  --language "${WHISPER_LANG}"
  --device "${WHISPER_DEVICE}"
  --decoding "${DECODING}"
  --tn-bin "${TN_BIN}"
)

if [[ "${TN_NORMALIZE_HYP}" == "0" ]]; then
  ARGS+=(--no-tn-normalize-hyp)
else
  ARGS+=(--tn-normalize-hyp)
fi

if [[ "${LIMIT}" =~ ^[0-9]+$ ]] && [[ "${LIMIT}" -gt 0 ]]; then
  ARGS+=(--limit "${LIMIT}")
fi

if [[ -n "${UTT_PREFIX}" ]]; then
  ARGS+=(--utt-prefix "${UTT_PREFIX}")
fi

python "${E2E_ROOT}/scripts/transcribe_and_wer.py" "${ARGS[@]}"

echo "[asr] OK: ${OUTPUT_DIR}"
