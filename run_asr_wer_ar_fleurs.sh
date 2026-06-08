#!/usr/bin/env bash
# Whisper ASR + WER on FLEURS ar_eg 1000-clip benchmark.
#
# Default: tn_normalize_hyp=off (pure Whisper vs FLEURS reference text).
# To normalize ASR hypotheses with Arabic TN before WER:
#   bash scripts/build_tn_ar.sh
#   TN_NORMALIZE_HYP=1 TN_BIN=bin/ar_tts ./run_asr_wer_ar_fleurs.sh
#
# Prerequisite: ./run_prepare_fleurs_ar.sh
set -euo pipefail

E2E_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "${E2E_ROOT}"

export WAV_DIR="${WAV_DIR:-${E2E_ROOT}/output/wav/ar_1000_fleurs}"
export NORMALIZED_TXT="${NORMALIZED_TXT:-${E2E_ROOT}/input/ar_1000_sample_sent.txt}"
export OUTPUT_DIR="${OUTPUT_DIR:-${E2E_ROOT}/output/asr/ar_1000_fleurs_whisper}"
export WHISPER_MODEL="${WHISPER_MODEL:-large-v3}"
export WHISPER_DEVICE="${WHISPER_DEVICE:-cuda}"
export WHISPER_LANG="${WHISPER_LANG:-ar}"
export UTT_PREFIX="${UTT_PREFIX:-ar}"
export TN_NORMALIZE_HYP="${TN_NORMALIZE_HYP:-0}"
export TN_BIN="${TN_BIN:-${E2E_ROOT}/bin/ar_tts}"
export SKIP_ASR_SETUP="${SKIP_ASR_SETUP:-0}"

if [[ "${TN_NORMALIZE_HYP}" == "1" ]] && [[ ! -x "${TN_BIN}" ]]; then
  echo "TN_NORMALIZE_HYP=1 but ${TN_BIN} missing. Run: bash scripts/build_tn_ar.sh" >&2
  exit 1
fi

if [[ ! -d "${WAV_DIR}" ]] || [[ -z "$(find "${WAV_DIR}" -maxdepth 1 -name 'ar_*.wav' 2>/dev/null | head -1)" ]]; then
  echo "Missing wav under ${WAV_DIR}. Run ./run_prepare_fleurs_ar.sh first." >&2
  exit 1
fi
if [[ ! -f "${NORMALIZED_TXT}" ]]; then
  echo "Missing reference: ${NORMALIZED_TXT}" >&2
  exit 1
fi

echo "=== FLEURS ar_eg Whisper ASR + WER ==="
bash "${E2E_ROOT}/run_asr_wer.sh"
