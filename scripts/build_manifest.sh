#!/usr/bin/env bash
# Build TTS manifest (wav|text) from output/normalized.txt
set -euo pipefail

E2E_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOCALE="${LOCALE:-bn}"

NORMALIZED_TXT="${NORMALIZED_TXT:-${E2E_ROOT}/output/normalized.txt}"
MANIFEST_TXT="${MANIFEST_TXT:-${E2E_ROOT}/output/tts_manifest.txt}"
INDEX_TSV="${INDEX_TSV:-${E2E_ROOT}/output/manifest.index.tsv}"
LIMIT="${LIMIT:-0}"

if [[ ! -f "${NORMALIZED_TXT}" ]]; then
  echo "Normalized file not found: ${NORMALIZED_TXT}" >&2
  echo "Run first: ${E2E_ROOT}/run_normalize.sh" >&2
  exit 1
fi

ARGS=(
  --normalized "${NORMALIZED_TXT}"
  --output "${MANIFEST_TXT}"
  --index "${INDEX_TSV}"
  --locale "${LOCALE}"
)

if [[ "${LIMIT}" =~ ^[0-9]+$ ]] && [[ "${LIMIT}" -gt 0 ]]; then
  ARGS+=(--limit "${LIMIT}")
  echo "[manifest] LIMIT=${LIMIT}"
fi

python3 "${E2E_ROOT}/scripts/build_manifest.py" "${ARGS[@]}"

echo "[manifest] OK: ${MANIFEST_TXT}"
