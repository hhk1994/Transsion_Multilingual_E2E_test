#!/usr/bin/env bash
# One-click: normalized.txt -> tts_manifest.txt (for Multilingual_LITs bn-en)
#
# Usage:
#   ./run_build_manifest.sh
#   LIMIT=20 ./run_build_manifest.sh
set -euo pipefail

E2E_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "${E2E_ROOT}"

export LOCALE="${LOCALE:-bn}"
export NORMALIZED_TXT="${NORMALIZED_TXT:-${E2E_ROOT}/output/normalized.txt}"
export MANIFEST_TXT="${MANIFEST_TXT:-${E2E_ROOT}/output/tts_manifest.txt}"
export INDEX_TSV="${INDEX_TSV:-${E2E_ROOT}/output/manifest.index.tsv}"
export LIMIT="${LIMIT:-0}"

echo "=== e2e_test build TTS manifest (locale=${LOCALE}) ==="
echo "  normalized: ${NORMALIZED_TXT}"
echo "  manifest:   ${MANIFEST_TXT}"
echo "  index:      ${INDEX_TSV}"
echo

bash "${E2E_ROOT}/scripts/build_manifest.sh"

echo
echo "Done. Feed to LITs, e.g.:"
echo "  cd .. && ./infer_example.sh bn-en ${MANIFEST_TXT}"
