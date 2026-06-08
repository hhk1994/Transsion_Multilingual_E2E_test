#!/usr/bin/env bash
# Build bn_tts normalizer into e2e_test/bin/
set -euo pipefail

E2E_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Allow overriding external TN repo path.
if [[ -n "${TN_ROOT:-}" ]]; then
  TN_ROOT="$(cd "${TN_ROOT}" && pwd)"
else
  # Prefer the TN git submodule under this repo.
  if [[ -d "${E2E_ROOT}/../Transsion_Multilingual_Text_Normalization_for_TTS" ]]; then
    TN_ROOT="$(cd "${E2E_ROOT}/../Transsion_Multilingual_Text_Normalization_for_TTS" && pwd)"
  else
    # Backward-compatible fallback for external sibling layout.
    TN_ROOT="$(cd "${E2E_ROOT}/../../Transsion_Multilingual_Text_Normalization_for_TTS" && pwd)"
  fi
fi
BIN_DIR="${E2E_ROOT}/bin"
OUT_BIN="${BIN_DIR}/bn_tts"

if [[ ! -f "${TN_ROOT}/bn.cpp" ]]; then
  echo "TN repo not found at ${TN_ROOT}" >&2
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

mkdir -p "${BIN_DIR}"

echo "[build] Compiling bn_tts (ICU_ROOT=${ICU_ROOT})..."
g++ -std=c++17 -O2 \
  "${TN_ROOT}/bn.cpp" "${TN_ROOT}/tts_normalizer_engine.cpp" "${TN_ROOT}/ru_year_spellout.cpp" \
  -I"${TN_ROOT}" -I"${TN_ROOT}/third_party" \
  -I"${ICU_ROOT}/include" \
  -L"${ICU_ROOT}/lib" -licui18n -licuuc -licudata \
  -o "${OUT_BIN}"

echo "[build] Done: ${OUT_BIN}"
