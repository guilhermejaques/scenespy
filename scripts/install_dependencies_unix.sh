#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALLER="$ROOT/_internal/dependencies/install_dependencies.py"

echo "Scenespy dependency installer"
echo
echo "This will install the video and AI components required by Scenespy."
echo "It may use your system package manager to install Python or FFmpeg."
echo "It may download Python packages from official package indexes."
echo "AI packages are installed in a user-local Scenespy folder."
echo

if [[ ! -f "$INSTALLER" ]]; then
  echo "install_dependencies.py was not found in _internal/dependencies." >&2
  exit 1
fi

PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
  if command -v python3.11 >/dev/null 2>&1; then
    PYTHON="python3.11"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON="python3"
  elif command -v python >/dev/null 2>&1; then
    PYTHON="python"
  else
    echo "Python was not found. Trying to install Python with the system package manager..."
    if [[ "$(uname -s)" == "Darwin" ]] && command -v brew >/dev/null 2>&1; then
      brew install python@3.11
      PYTHON="python3.11"
    elif command -v apt >/dev/null 2>&1; then
      sudo apt update
      sudo apt install -y python3 python3-venv python3-pip
      PYTHON="python3"
    elif command -v dnf >/dev/null 2>&1; then
      sudo dnf install -y python3 python3-pip
      PYTHON="python3"
    elif command -v pacman >/dev/null 2>&1; then
      sudo pacman -S --needed python python-pip
      PYTHON="python"
    else
      echo "Could not install Python automatically. Install Python 3.11 and run this script again." >&2
      exit 1
    fi
  fi
fi

if ! "$PYTHON" -c "import venv, ensurepip" >/dev/null 2>&1; then
  echo "Python venv/ensurepip support is missing. Trying to install it with the system package manager..."
  if [[ "$(uname -s)" == "Darwin" ]] && command -v brew >/dev/null 2>&1; then
    brew install python@3.11
    PYTHON="python3.11"
  elif command -v apt >/dev/null 2>&1; then
    sudo apt update
    sudo apt install -y python3-venv python3-pip
  elif command -v dnf >/dev/null 2>&1; then
    sudo dnf install -y python3-pip
  elif command -v pacman >/dev/null 2>&1; then
    sudo pacman -S --needed python python-pip
  else
    echo "Could not install Python venv support automatically." >&2
    exit 1
  fi
fi

echo "Starting dependency installation. Large AI packages may take a while to download..."
"$PYTHON" "$INSTALLER"
