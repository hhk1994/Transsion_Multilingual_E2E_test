#!/usr/bin/env bash
# System packages for TTS synthesis (espeak-ng, optional git-lfs for checkpoints).
set -euo pipefail

need_sudo=0
if [[ "${EUID}" -ne 0 ]]; then
  if command -v sudo >/dev/null 2>&1; then
    need_sudo=1
  fi
fi

run_apt() {
  if [[ "${need_sudo}" -eq 1 ]]; then
    sudo "$@"
  elif [[ "${EUID}" -eq 0 ]]; then
    "$@"
  else
    echo "[synth-deps] Skip apt (no sudo): $*" >&2
    return 0
  fi
}

_py_ver="$(${PYTHON:-python3} -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "3.10")"

if ! "${PYTHON:-python3}" -m venv /tmp/e2e_venv_probe_$$ 2>/dev/null; then
  echo "[synth-deps] Installing python${_py_ver}-venv..."
  run_apt apt-get update -qq
  run_apt apt-get install -y -qq "python${_py_ver}-venv" || run_apt apt-get install -y -qq python3-venv
  rm -rf "/tmp/e2e_venv_probe_$$"
fi

if ! command -v espeak-ng >/dev/null 2>&1; then
  echo "[synth-deps] Installing espeak-ng..."
  run_apt apt-get update -qq
  run_apt apt-get install -y -qq espeak-ng
fi

if ! command -v git-lfs >/dev/null 2>&1; then
  echo "[synth-deps] Installing git-lfs..."
  run_apt apt-get update -qq
  run_apt apt-get install -y -qq git-lfs
  git lfs install >/dev/null 2>&1 || true
fi

echo "[synth-deps] espeak-ng: $(command -v espeak-ng)"
