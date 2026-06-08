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

if [[ "${LOCALE}" == "ru" ]]; then
  bash "${E2E_ROOT}/scripts/build_tn_ru.sh"
  RU_MORPH_MODEL="${RU_MORPH_MODEL:-${E2E_ROOT}/../Transsion_Multilingual_Text_Normalization_for_TTS/original/morphodita/models/russian-syntagrus-morphodita-only.tagger}"
  if [[ -z "${BN_TTS:-}" ]]; then
    cat > "${E2E_ROOT}/bin/ru_tts_wrapper" <<EOF
#!/usr/bin/env bash
exec "${E2E_ROOT}/bin/ru_tts" --morph-model "${RU_MORPH_MODEL}" "\$@"
EOF
    chmod +x "${E2E_ROOT}/bin/ru_tts_wrapper"
    export BN_TTS="${E2E_ROOT}/bin/ru_tts_wrapper"
  fi
elif [[ "${SKIP_TN_BUILD:-0}" != "1" ]]; then
  bash "${E2E_ROOT}/scripts/build_tn.sh"
fi

bash "${E2E_ROOT}/scripts/normalize.sh"

echo
echo "Done. Normalized text: ${OUTPUT_TXT}"
