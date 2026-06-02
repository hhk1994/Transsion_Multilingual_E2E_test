#!/usr/bin/env bash
# One-click: build ru_tts into e2e_test/bin with MorphoDiTa checks.
set -euo pipefail

E2E_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "${E2E_ROOT}"

echo "=== e2e_test Russian normalizer build (ru_tts) ==="
echo "  tn_root: ${TN_ROOT:-auto-detect}"
echo

bash "${E2E_ROOT}/scripts/install_deps.sh"
bash "${E2E_ROOT}/scripts/build_tn_ru.sh"

echo
echo "Done. Russian normalizer: ${E2E_ROOT}/bin/ru_tts"
