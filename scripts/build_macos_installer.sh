#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="${1:-$(sed -n 's/^__version__ = "\([^"]*\)"/\1/p' "$ROOT/scenespy/__init__.py")}"
case "$(uname -m)" in
  arm64|aarch64) ARCH="arm64" ;;
  x86_64|amd64) ARCH="x64" ;;
  *) echo "Unsupported macOS architecture: $(uname -m)" >&2; exit 1 ;;
esac

SOURCE="$ROOT/release/Scenespy-macos"
WORK="$(mktemp -d)"
PAYLOAD="$WORK/payload"
CORE_SCRIPTS="$WORK/core-scripts"
AI_SCRIPTS="$WORK/ai-scripts"
OUTPUT="$ROOT/release/Scenespy-${VERSION}-macos-${ARCH}.pkg"
trap 'rm -rf "$WORK"' EXIT

if [[ ! -d "$SOURCE/Scenespy.app" ]]; then
  echo "Build Scenespy.app before creating the macOS installer." >&2
  exit 1
fi

mkdir -p "$PAYLOAD/Applications" "$CORE_SCRIPTS" "$AI_SCRIPTS"
cp -R "$SOURCE/Scenespy.app" "$PAYLOAD/Applications/Scenespy.app"

RUNTIME_DIR="$PAYLOAD/Applications/Scenespy.app/Contents/Resources/scenespy-runtime"
if [[ ! -x "$RUNTIME_DIR/install_runtime.sh" ||
      ! -f "$RUNTIME_DIR/_internal/dependencies/install_runtime.py" ]]; then
  echo "The final app bundle does not contain its runtime installer." >&2
  exit 1
fi

cat > "$CORE_SCRIPTS/postinstall" <<'POSTINSTALL'
#!/usr/bin/env bash
set -euo pipefail

CONSOLE_USER="$(stat -f '%Su' /dev/console)"
if [[ -z "$CONSOLE_USER" || "$CONSOLE_USER" == "root" || "$CONSOLE_USER" == "loginwindow" ]]; then
  echo "A signed-in macOS user is required to install the Scenespy runtime." >&2
  exit 1
fi

USER_HOME="$(dscl . -read "/Users/$CONSOLE_USER" NFSHomeDirectory | awk '{print $2}')"
RUNTIME_SCRIPT="/Applications/Scenespy.app/Contents/Resources/scenespy-runtime/install_runtime.sh"
/usr/bin/sudo -u "$CONSOLE_USER" /usr/bin/env HOME="$USER_HOME" SKIP_AI=1 /bin/bash "$RUNTIME_SCRIPT"
POSTINSTALL
chmod +x "$CORE_SCRIPTS/postinstall"

cat > "$AI_SCRIPTS/postinstall" <<'POSTINSTALL'
#!/usr/bin/env bash
set -euo pipefail

CONSOLE_USER="$(stat -f '%Su' /dev/console)"
if [[ -z "$CONSOLE_USER" || "$CONSOLE_USER" == "root" || "$CONSOLE_USER" == "loginwindow" ]]; then
  echo "A signed-in macOS user is required to install Scenespy AI features." >&2
  exit 1
fi

USER_HOME="$(dscl . -read "/Users/$CONSOLE_USER" NFSHomeDirectory | awk '{print $2}')"
RUNTIME_SCRIPT="/Applications/Scenespy.app/Contents/Resources/scenespy-runtime/install_runtime.sh"
if ! /usr/bin/sudo -u "$CONSOLE_USER" /usr/bin/env HOME="$USER_HOME" TORCH_MODE=auto SKIP_FFMPEG=1 /bin/bash "$RUNTIME_SCRIPT"; then
  # Optional AI downloads must never roll back an otherwise usable app.  The
  # same installer can be retried later from Detect faces inside Scenespy.
  echo "Optional Scenespy AI setup did not complete; it can be retried from the app." >&2
fi
exit 0
POSTINSTALL
chmod +x "$AI_SCRIPTS/postinstall"

CORE_PKG="$WORK/ScenespyCore.pkg"
AI_PKG="$WORK/ScenespyAI.pkg"
pkgbuild \
  --root "$PAYLOAD" \
  --scripts "$CORE_SCRIPTS" \
  --identifier "com.scenespy.app.core" \
  --version "$VERSION" \
  --install-location / \
  "$CORE_PKG"
pkgbuild \
  --nopayload \
  --scripts "$AI_SCRIPTS" \
  --identifier "com.scenespy.app.ai" \
  --version "$VERSION" \
  "$AI_PKG"

cat > "$WORK/distribution.xml" <<DISTRIBUTION
<?xml version="1.0" encoding="utf-8"?>
<installer-gui-script minSpecVersion="2">
  <title>Scenespy ${VERSION}</title>
  <options customize="always" require-scripts="false" hostArchitectures="${ARCH/x64/x86_64}"/>
  <allowed-os-versions>
    <os-version min="12.0"/>
  </allowed-os-versions>
  <choices-outline>
    <line choice="core"/>
    <line choice="ai"/>
  </choices-outline>
  <choice id="core" title="Scenespy" description="Required application and video runtime" visible="false" start_selected="true">
    <pkg-ref id="com.scenespy.app.core"/>
  </choice>
  <choice id="ai" title="AI face detection features" description="Optional and selected by default; downloads up to 6 GB. It can also be installed later from inside Scenespy." start_selected="true">
    <pkg-ref id="com.scenespy.app.ai"/>
  </choice>
  <pkg-ref id="com.scenespy.app.core" version="${VERSION}" auth="Root">ScenespyCore.pkg</pkg-ref>
  <pkg-ref id="com.scenespy.app.ai" version="${VERSION}" auth="Root">ScenespyAI.pkg</pkg-ref>
</installer-gui-script>
DISTRIBUTION

rm -f "$OUTPUT"
productbuild --distribution "$WORK/distribution.xml" --package-path "$WORK" "$OUTPUT"

echo "macOS installer ready: $OUTPUT"
