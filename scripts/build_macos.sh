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
rm -rf build dist release/Scenespy-macos
"$PYTHON" -m PyInstaller --clean --noconfirm --distpath release Scenespy.spec

if [[ ! -d release/Scenespy.app ]]; then
  echo "Build finished, but release/Scenespy.app was not found" >&2
  exit 1
fi

rm -rf release/Scenespy-macos
mkdir -p release/Scenespy-macos
mv release/Scenespy.app release/Scenespy-macos/Scenespy.app
mkdir -p release/Scenespy-macos/_internal/dependencies
cp scripts/install_runtime.py release/Scenespy-macos/_internal/dependencies/install_runtime.py
cp scripts/install_runtime_unix.sh release/Scenespy-macos/install_runtime.sh
chmod +x release/Scenespy-macos/install_runtime.sh

APP="release/Scenespy-macos/Scenespy.app"
RUNTIME_DIR="$APP/Contents/Resources/scenespy-runtime"
mkdir -p "$RUNTIME_DIR/_internal/dependencies"
cp scripts/install_runtime.py "$RUNTIME_DIR/_internal/dependencies/install_runtime.py"
cp scripts/install_runtime_unix.sh "$RUNTIME_DIR/install_runtime.sh"
chmod +x "$RUNTIME_DIR/install_runtime.sh"

# Adding runtime resources changes the bundle after PyInstaller's signing
# phase. Seal the final bundle again so macOS never receives a modified or
# invalid ad-hoc signature. Public Developer ID signing/notarization is a
# separate release step because it requires owner-provided Apple credentials.
codesign --force --deep --sign - "$APP"
codesign --verify --deep --strict "$APP"

QT_QPA_PLATFORM=offscreen SCENESPY_APP_SELFTEST=1 \
  "$APP/Contents/MacOS/Scenespy"

bash ./scripts/build_macos_installer.sh

rm -rf build dist

echo "macOS distribution app ready: $ROOT/release/Scenespy-macos/Scenespy.app"
