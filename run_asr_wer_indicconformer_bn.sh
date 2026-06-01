#!/usr/bin/env bash
# Run Bengali ASR+WER with IndicConformer-600M multilingual.
#
# Usage:
#   ./run_asr_wer_indicconformer_bn.sh
#   WAV_DIR=output/wav/bn_smoke_5 ./run_asr_wer_indicconformer_bn.sh
#   WAV_DIR=output/wav/latest DECODING=rnnt ./run_asr_wer_indicconformer_bn.sh
set -euo pipefail

E2E_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "${E2E_ROOT}"

WAV_BASENAME="$(basename "${WAV_DIR:-${E2E_ROOT}/output/wav/bn_smoke_5}")"

export WAV_DIR="${WAV_DIR:-${E2E_ROOT}/output/wav/bn_smoke_5}"
export NORMALIZED_TXT="${NORMALIZED_TXT:-${E2E_ROOT}/output/normalized.txt}"
export OUTPUT_DIR="${OUTPUT_DIR:-${E2E_ROOT}/output/asr/${WAV_BASENAME}_indicconformer_bn}"
export ASR_BACKEND="${ASR_BACKEND:-indic-conformer}"
export WHISPER_MODEL="${WHISPER_MODEL:-ai4bharat/indic-conformer-600m-multilingual}"
export WHISPER_DEVICE="${WHISPER_DEVICE:-cuda}"
export WHISPER_LANG="${WHISPER_LANG:-bn}"
export DECODING="${DECODING:-ctc}"

echo "=== e2e_test ASR + WER (IndicConformer-600M, Bengali) ==="
echo "  wav_dir:   ${WAV_DIR}"
echo "  reference: ${NORMALIZED_TXT}"
echo "  output:    ${OUTPUT_DIR}"
echo "  model:     ${WHISPER_MODEL}"
echo "  decoding:  ${DECODING}"
echo

if [[ "${SKIP_ASR_SETUP:-0}" != "1" ]]; then
  bash "${E2E_ROOT}/scripts/setup_asr_deps.sh"
fi

bash "${E2E_ROOT}/scripts/run_asr_wer.sh"

echo
echo "Done. See ${OUTPUT_DIR}/wer_report.json"
