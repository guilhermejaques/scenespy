#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="python"
if [[ -x .venv/bin/python ]]; then
  PYTHON=".venv/bin/python"
fi
TORCH_MODE="${TORCH_MODE:-cpu}"

if [[ ! -f models/yolov8n-face.pt ]]; then
  echo "Missing models/yolov8n-face.pt" >&2
  exit 1
fi

"$PYTHON" scripts/install_build_dependencies.py --torch-mode "$TORCH_MODE"
rm -rf build dist release/Scenespy-macos
"$PYTHON" -m PyInstaller --clean --noconfirm --distpath release Scenespy.spec

if [[ ! -d release/Scenespy.app ]]; then
  echo "Build finished, but release/Scenespy.app was not found" >&2
  exit 1
fi

mkdir -p release/Scenespy-macos
mv release/Scenespy.app release/Scenespy-macos/Scenespy.app

if [[ -d release-assets/macos/ffmpeg ]]; then
  mkdir -p release/Scenespy-macos/Scenespy.app/Contents/Frameworks/bin/macos
  cp release-assets/macos/ffmpeg/* release/Scenespy-macos/Scenespy.app/Contents/Frameworks/bin/macos/
  chmod +x release/Scenespy-macos/Scenespy.app/Contents/Frameworks/bin/macos/ffmpeg release/Scenespy-macos/Scenespy.app/Contents/Frameworks/bin/macos/ffprobe 2>/dev/null || true
fi

rm -rf build dist

echo "macOS distribution app ready: $ROOT/release/Scenespy-macos/Scenespy.app"
