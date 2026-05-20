#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALLER="$ROOT/_internal/dependencies/install_runtime.py"
TORCH_MODE="${TORCH_MODE:-auto}"
PYTHON_STANDALONE_RELEASE="${PYTHON_STANDALONE_RELEASE:-20240726}"
PYTHON_STANDALONE_VERSION="${PYTHON_STANDALONE_VERSION:-3.11.9}"

if [[ ! -f "$INSTALLER" ]]; then
  echo "install_runtime.py was not found in _internal/dependencies." >&2
  exit 1
fi

OS_NAME="$(uname -s)"
ARCH_NAME="$(uname -m)"

if [[ "$OS_NAME" == "Darwin" ]]; then
  DATA_HOME="$HOME/Library/Application Support/Scenespy"
  case "$ARCH_NAME" in
    arm64|aarch64) PYTHON_TARGET="aarch64-apple-darwin" ;;
    x86_64|amd64) PYTHON_TARGET="x86_64-apple-darwin" ;;
    *) echo "Unsupported macOS architecture: $ARCH_NAME" >&2; exit 1 ;;
  esac
else
  DATA_HOME="${XDG_CONFIG_HOME:-$HOME/.config}/scenespy"
  case "$ARCH_NAME" in
    x86_64|amd64) PYTHON_TARGET="x86_64-unknown-linux-gnu" ;;
    aarch64|arm64) PYTHON_TARGET="aarch64-unknown-linux-gnu" ;;
    *) echo "Unsupported Linux architecture: $ARCH_NAME" >&2; exit 1 ;;
  esac
fi

PRIVATE_PY_DIR="$DATA_HOME/python311"
PRIVATE_PY="$PRIVATE_PY_DIR/bin/python3.11"
DOWNLOADS_DIR="$DATA_HOME/downloads"

download_file() {
  local url="$1"
  local output="$2"
  if command -v curl >/dev/null 2>&1; then
    curl -fL "$url" -o "$output"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$output" "$url"
  else
    echo "curl or wget is required to download private Python." >&2
    exit 1
  fi
}

ensure_private_python() {
  if [[ -x "$PRIVATE_PY" ]]; then
    return
  fi

  mkdir -p "$DOWNLOADS_DIR"
  local asset="cpython-${PYTHON_STANDALONE_VERSION}+${PYTHON_STANDALONE_RELEASE}-${PYTHON_TARGET}-install_only.tar.gz"
  local url="https://github.com/astral-sh/python-build-standalone/releases/download/${PYTHON_STANDALONE_RELEASE}/${asset}"
  local archive="$DOWNLOADS_DIR/$asset"
  local tmp
  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' RETURN

  if [[ ! -f "$archive" ]]; then
    echo "Downloading private Python ${PYTHON_STANDALONE_VERSION} for Scenespy..."
    download_file "$url" "$archive"
  fi

  echo "Installing private Python ${PYTHON_STANDALONE_VERSION} for Scenespy..."
  tar -xzf "$archive" -C "$tmp"
  local extracted_python
  extracted_python="$(find "$tmp" -type f -path '*/bin/python3.11' | head -n 1)"
  if [[ -z "$extracted_python" ]]; then
    echo "Downloaded Python archive did not contain bin/python3.11." >&2
    exit 1
  fi
  local extracted_root
  extracted_root="$(cd "$(dirname "$extracted_python")/.." && pwd)"
  rm -rf "$PRIVATE_PY_DIR"
  mkdir -p "$PRIVATE_PY_DIR"
  cp -a "$extracted_root"/. "$PRIVATE_PY_DIR"/
  chmod +x "$PRIVATE_PY"
}

ensure_private_python
PYTHON="$PRIVATE_PY"

echo "Scenespy runtime installer"
echo "This installs FFmpeg/FFprobe and Detect faces AI packages."
echo "Private Python: $PRIVATE_PY"
echo
"$PYTHON" "$INSTALLER" --torch-mode "$TORCH_MODE"
