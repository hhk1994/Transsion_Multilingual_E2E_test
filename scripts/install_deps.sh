#!/usr/bin/env bash
# Install system packages required to build bn_tts (ICU + compiler).
set -euo pipefail

need_sudo=0
if [[ "${EUID}" -ne 0 ]]; then
  if command -v sudo >/dev/null 2>&1; then
    need_sudo=1
  else
    echo "Need root or sudo to install packages." >&2
    exit 1
  fi
fi

run_apt() {
  if [[ "${need_sudo}" -eq 1 ]]; then
    sudo "$@"
  else
    "$@"
  fi
}

if ! command -v g++ >/dev/null 2>&1; then
  echo "[deps] Installing build-essential..."
  run_apt apt-get update -qq
  run_apt apt-get install -y -qq build-essential
fi

if [[ ! -f /usr/include/unicode/locid.h ]]; then
  echo "[deps] Installing libicu-dev..."
  run_apt apt-get update -qq
  run_apt apt-get install -y -qq libicu-dev
fi

echo "[deps] OK: g++=$(command -v g++), ICU headers at /usr/include/unicode"
