#!/usr/bin/env bash
# Run bn_tts: input text file -> normalized.txt
set -euo pipefail

E2E_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOCALE="${LOCALE:-bn}"

INPUT_TXT="${INPUT_TXT:-${E2E_ROOT}/trassion_test/${LOCALE}_1000_sample_sent.txt}"
OUTPUT_TXT="${OUTPUT_TXT:-${E2E_ROOT}/output/normalized.txt}"
BN_TTS="${BN_TTS:-${E2E_ROOT}/bin/bn_tts}"
LIMIT="${LIMIT:-0}"

if [[ ! -f "${INPUT_TXT}" ]]; then
  echo "Input not found: ${INPUT_TXT}" >&2
  exit 1
fi

if [[ ! -x "${BN_TTS}" ]]; then
  echo "Normalizer binary missing: ${BN_TTS}" >&2
  echo "Run: ${E2E_ROOT}/run_normalize.sh" >&2
  exit 1
fi

mkdir -p "$(dirname "${OUTPUT_TXT}")"

RUN_INPUT="${INPUT_TXT}"
TMP_INPUT=""
if [[ "${LIMIT}" =~ ^[0-9]+$ ]] && [[ "${LIMIT}" -gt 0 ]]; then
  TMP_INPUT="$(mktemp)"
  head -n "${LIMIT}" "${INPUT_TXT}" > "${TMP_INPUT}"
  RUN_INPUT="${TMP_INPUT}"
  echo "[normalize] LIMIT=${LIMIT} (first ${LIMIT} lines of ${INPUT_TXT})"
fi

cleanup() {
  if [[ -n "${TMP_INPUT}" && -f "${TMP_INPUT}" ]]; then
    rm -f "${TMP_INPUT}"
  fi
}
trap cleanup EXIT

echo "[normalize] ${RUN_INPUT} -> ${OUTPUT_TXT}"
"${BN_TTS}" < "${RUN_INPUT}" > "${OUTPUT_TXT}"

IN_LINES="$(awk 'END{print NR+0}' "${RUN_INPUT}")"
OUT_LINES="$(awk 'END{print NR+0}' "${OUTPUT_TXT}")"
EMPTY_OUT="$(awk 'NF==0{c++} END{print c+0}' "${OUTPUT_TXT}")"

echo "[normalize] lines: in=${IN_LINES} out=${OUT_LINES} empty_out=${EMPTY_OUT}"

if [[ "${IN_LINES}" != "${OUT_LINES}" ]]; then
  echo "[normalize] WARN: line count mismatch" >&2
  exit 2
fi

echo "[normalize] OK: ${OUTPUT_TXT}"
