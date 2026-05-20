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

"$PYTHON" scripts/install_build_dependencies.py
if ! "$PYTHON" -c "import tkinter; print(tkinter.TkVersion)" >/dev/null 2>&1; then
  echo "Python tkinter support is missing. Install python3-tk/python3.11-tk before building." >&2
  exit 1
fi
rm -rf build dist release/Scenespy-linux-x64
"$PYTHON" -m PyInstaller --clean --noconfirm --distpath release Scenespy.spec

if [[ ! -x release/Scenespy-linux-x64/Scenespy ]]; then
  echo "Build finished, but release/Scenespy-linux-x64/Scenespy was not found or is not executable" >&2
  exit 1
fi

mkdir -p release/Scenespy-linux-x64/_internal/dependencies
cp scripts/install_runtime.py release/Scenespy-linux-x64/_internal/dependencies/install_runtime.py
cp scripts/install_runtime_unix.sh release/Scenespy-linux-x64/install_runtime.sh
chmod +x release/Scenespy-linux-x64/install_runtime.sh

tar -C release -czf release/Scenespy-linux-x64.tar.gz Scenespy-linux-x64
rm -rf build dist

echo "Linux distribution folder ready: $ROOT/release/Scenespy-linux-x64"
echo "Linux archive ready: $ROOT/release/Scenespy-linux-x64.tar.gz"
