#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="${1:-$(sed -n 's/^__version__ = "\([^"]*\)"/\1/p' "$ROOT/scenespy/__init__.py")}"
PAYLOAD="$ROOT/release/Scenespy-linux-x64"
OUTPUT="$ROOT/release/Scenespy-${VERSION}-linux-x64.run"
HEADER="$(mktemp)"
trap 'rm -f "$HEADER"' EXIT

if [[ ! -x "$PAYLOAD/Scenespy" ]]; then
  echo "Build the Linux distribution before creating the installer." >&2
  exit 1
fi

cat > "$HEADER" <<'INSTALLER'
#!/usr/bin/env bash
set -euo pipefail

APP_NAME="Scenespy"
TARGET="${SCENESPY_INSTALL_DIR:-$HOME/.local/opt/scenespy}"
BIN_DIR="${XDG_BIN_HOME:-$HOME/.local/bin}"
APPLICATIONS_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
MARKER="__SCENESPY_PAYLOAD_BELOW__"
PAYLOAD_LINE="$(awk -v marker="$MARKER" '$0 == marker { print NR + 1; exit }' "$SELF")"

if [[ -z "$PAYLOAD_LINE" ]]; then
  echo "Invalid Scenespy installer: payload not found." >&2
  exit 1
fi

mkdir -p "$TARGET" "$BIN_DIR" "$APPLICATIONS_DIR"
tail -n "+$PAYLOAD_LINE" "$SELF" | tar -xz -C "$TARGET" --strip-components=1
ln -sfn "$TARGET/Scenespy" "$BIN_DIR/scenespy"

cat > "$APPLICATIONS_DIR/scenespy.desktop" <<DESKTOP
[Desktop Entry]
Type=Application
Name=Scenespy
Comment=Video scene detection and editing
Exec=$TARGET/Scenespy
Icon=$TARGET/_internal/scenespy/assets/images/x-256.png
Terminal=false
Categories=AudioVideo;Video;
DESKTOP

cat > "$TARGET/uninstall.sh" <<UNINSTALL
#!/usr/bin/env bash
set -euo pipefail
rm -f "$BIN_DIR/scenespy" "$APPLICATIONS_DIR/scenespy.desktop"
rm -rf "$TARGET"
echo "Scenespy removed. User runtime data was preserved."
UNINSTALL
chmod +x "$TARGET/uninstall.sh"

echo "Scenespy installed in $TARGET"
echo "Launch it from the application menu or run: scenespy"
echo "Downloading and installing Scenespy runtime components..."
INSTALL_AI="yes"
if [[ -t 0 ]]; then
  read -r -p "Install optional AI face detection now (up to 6 GB; selected by default)? [Y/n] " ANSWER
  case "${ANSWER:-y}" in
    n|N|no|NO|No) INSTALL_AI="no" ;;
  esac
fi
if [[ "$INSTALL_AI" == "yes" ]]; then
  TORCH_MODE=auto "$TARGET/install_runtime.sh"
else
  TORCH_MODE=cpu SKIP_AI=1 "$TARGET/install_runtime.sh"
fi
echo "Scenespy installation completed."
exit 0
INSTALLER

printf '\n%s\n' "__SCENESPY_PAYLOAD_BELOW__" >> "$HEADER"
cat "$HEADER" > "$OUTPUT"
tar -C "$ROOT/release" -czf - Scenespy-linux-x64 >> "$OUTPUT"
chmod +x "$OUTPUT"
echo "Linux installer ready: $OUTPUT"
