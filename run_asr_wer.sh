#!/usr/bin/env bash
# One-click: install ASR deps -> Whisper transcribe -> WER vs normalized.txt
#
# Usage:
#   ./run_asr_wer.sh
#   WAV_DIR=output/wav/bn_smoke_5 ./run_asr_wer.sh
#   WAV_DIR=output/wav/latest SKIP_ASR_SETUP=1 ./run_asr_wer.sh
set -euo pipefail

E2E_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "${E2E_ROOT}"

export WAV_DIR="${WAV_DIR:-${E2E_ROOT}/output/wav/bn_smoke_5}"
export NORMALIZED_TXT="${NORMALIZED_TXT:-${E2E_ROOT}/output/normalized.txt}"
export OUTPUT_DIR="${OUTPUT_DIR:-${E2E_ROOT}/output/asr/$(basename "${WAV_DIR}")}"
export WHISPER_MODEL="${WHISPER_MODEL:-large-v3}"
export WHISPER_DEVICE="${WHISPER_DEVICE:-cuda}"
export WHISPER_LANG="${WHISPER_LANG:-bn}"

echo "=== e2e_test ASR + WER (Whisper) ==="
echo "  wav_dir:    ${WAV_DIR}"
echo "  reference:  ${NORMALIZED_TXT}"
echo "  output:     ${OUTPUT_DIR}"
echo "  model:      ${WHISPER_MODEL} (${WHISPER_DEVICE})"
echo

if [[ "${SKIP_ASR_SETUP:-0}" != "1" ]]; then
  bash "${E2E_ROOT}/scripts/setup_asr_deps.sh"
fi

bash "${E2E_ROOT}/scripts/run_asr_wer.sh"

echo
echo "Done. See ${OUTPUT_DIR}/wer_report.json"
