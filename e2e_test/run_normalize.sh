#!/usr/bin/env bash
# One-click: install deps (if needed) -> build bn_tts -> write output/normalized.txt
#
# Usage:
#   ./run_normalize.sh
#   LIMIT=50 ./run_normalize.sh
#   INPUT_TXT=trassion_test/bn_1000_sample_sent.txt OUTPUT_TXT=output/normalized.txt ./run_normalize.sh
set -euo pipefail

E2E_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "${E2E_ROOT}"

export LOCALE="${LOCALE:-bn}"
export INPUT_TXT="${INPUT_TXT:-${E2E_ROOT}/trassion_test/${LOCALE}_1000_sample_sent.txt}"
export OUTPUT_TXT="${OUTPUT_TXT:-${E2E_ROOT}/output/normalized.txt}"
export LIMIT="${LIMIT:-0}"

echo "=== e2e_test text normalization (locale=${LOCALE}) ==="
echo "  input:  ${INPUT_TXT}"
echo "  output: ${OUTPUT_TXT}"
echo

bash "${E2E_ROOT}/scripts/install_deps.sh"
bash "${E2E_ROOT}/scripts/build_tn.sh"
bash "${E2E_ROOT}/scripts/normalize.sh"

echo
echo "Done. Normalized text: ${OUTPUT_TXT}"
