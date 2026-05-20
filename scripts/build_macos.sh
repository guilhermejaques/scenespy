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
rm -rf build dist release/Scenespy-macos
"$PYTHON" -m PyInstaller --clean --noconfirm --distpath release Scenespy.spec

if [[ ! -d release/Scenespy.app ]]; then
  echo "Build finished, but release/Scenespy.app was not found" >&2
  exit 1
fi

mkdir -p release/Scenespy-macos
mv release/Scenespy.app release/Scenespy-macos/Scenespy.app
mkdir -p release/Scenespy-macos/_internal/dependencies
cp scripts/install_dependencies.py release/Scenespy-macos/_internal/dependencies/install_dependencies.py
cp scripts/install_dependencies_unix.sh release/Scenespy-macos/install_dependencies.sh
chmod +x release/Scenespy-macos/install_dependencies.sh

rm -rf build dist

echo "macOS distribution app ready: $ROOT/release/Scenespy-macos/Scenespy.app"
