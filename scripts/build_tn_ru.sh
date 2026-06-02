#!/usr/bin/env bash
# Build ru_tts normalizer into e2e_test/bin/
set -euo pipefail

E2E_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [[ -n "${TN_ROOT:-}" ]]; then
  TN_ROOT="$(cd "${TN_ROOT}" && pwd)"
else
  if [[ -d "${E2E_ROOT}/../Transsion_Multilingual_Text_Normalization_for_TTS" ]]; then
    TN_ROOT="$(cd "${E2E_ROOT}/../Transsion_Multilingual_Text_Normalization_for_TTS" && pwd)"
  else
    TN_ROOT="$(cd "${E2E_ROOT}/../../Transsion_Multilingual_Text_Normalization_for_TTS" && pwd)"
  fi
fi

BIN_DIR="${E2E_ROOT}/bin"
OUT_BIN="${BIN_DIR}/ru_tts"
PREP_SCRIPT="${TN_ROOT}/test/scripts/prepare_morphodita.sh"
MORPH_INC="${TN_ROOT}/original/morphodita/build/_deps/morphodita-src/src_lib_only"
MORPH_CPP="${MORPH_INC}/morphodita.cpp"
MORPH_H="${MORPH_INC}/morphodita.h"
RU_MORPH_MODEL="${RU_MORPH_MODEL:-${TN_ROOT}/original/morphodita/models/russian-syntagrus-morphodita-only.tagger}"

if [[ ! -f "${TN_ROOT}/ru.cpp" ]]; then
  echo "TN repo not found or missing ru.cpp: ${TN_ROOT}" >&2
  exit 1
fi
if [[ ! -f "${PREP_SCRIPT}" ]]; then
  echo "Missing MorphoDiTa bootstrap script: ${PREP_SCRIPT}" >&2
  echo "Update TN submodule, then retry." >&2
  exit 1
fi

ICU_ROOT="${ICU_ROOT:-}"
if [[ -z "${ICU_ROOT}" ]]; then
  for cand in /usr /opt/homebrew/opt/icu4c /usr/local/opt/icu4c; do
    if [[ -f "${cand}/include/unicode/locid.h" ]]; then
      ICU_ROOT="${cand}"
      break
    fi
  done
fi
if [[ -z "${ICU_ROOT}" ]]; then
  echo "ICU not found. Run scripts/install_deps.sh or set ICU_ROOT." >&2
  exit 1
fi

echo "[build-ru] Preparing MorphoDiTa sources ..."
bash "${PREP_SCRIPT}"
if [[ ! -f "${MORPH_H}" || ! -f "${MORPH_CPP}" ]]; then
  echo "MorphoDiTa headers/sources missing: ${MORPH_INC}" >&2
  exit 1
fi
if [[ ! -f "${RU_MORPH_MODEL}" ]]; then
  echo "Russian morph model not found: ${RU_MORPH_MODEL}" >&2
  echo "Set RU_MORPH_MODEL=/path/to/russian-syntagrus-morphodita-only.tagger" >&2
  exit 1
fi

mkdir -p "${BIN_DIR}"

echo "[build-ru] Compiling ru_tts (ICU_ROOT=${ICU_ROOT})..."
g++ -std=c++17 -O2 \
  "${TN_ROOT}/ru.cpp" "${TN_ROOT}/tts_normalizer_engine.cpp" "${MORPH_CPP}" \
  -I"${TN_ROOT}" -I"${TN_ROOT}/third_party" -I"${MORPH_INC}" \
  -I"${ICU_ROOT}/include" \
  -L"${ICU_ROOT}/lib" -licui18n -licuuc -licudata \
  -lpthread \
  -o "${OUT_BIN}"

echo "[build-ru] Done: ${OUT_BIN}"
echo "[build-ru] Runtime model example:"
echo "           ${OUT_BIN} --morph-model \"${RU_MORPH_MODEL}\""
