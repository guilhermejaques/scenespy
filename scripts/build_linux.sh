#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="python"
if [[ -x .venv/bin/python ]]; then
  PYTHON=".venv/bin/python"
fi

if [[ ! -f models/yolov8n-face.pt ]]; then
  echo "Missing models/yolov8n-face.pt" >&2
  exit 1
fi

"$PYTHON" -m pip install --upgrade pip
"$PYTHON" -m pip install -r requirements-base.txt pyinstaller
rm -rf build dist release/Scenespy-linux-x64
"$PYTHON" -m PyInstaller --clean --noconfirm --distpath release Scenespy.spec

if [[ ! -x release/Scenespy-linux-x64/Scenespy ]]; then
  echo "Build finished, but release/Scenespy-linux-x64/Scenespy was not found or is not executable" >&2
  exit 1
fi

mkdir -p release/Scenespy-linux-x64/_internal/dependencies
cp scripts/install_dependencies.py release/Scenespy-linux-x64/_internal/dependencies/install_dependencies.py
cp scripts/install_dependencies_unix.sh release/Scenespy-linux-x64/install_dependencies.sh
chmod +x release/Scenespy-linux-x64/install_dependencies.sh
rm -rf build dist

echo "Linux distribution folder ready: $ROOT/release/Scenespy-linux-x64"
