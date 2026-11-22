#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="${1:-$(sed -n 's/^__version__ = "\([^"]*\)"/\1/p' "$ROOT/scenespy/__init__.py")}"
SOURCE="$ROOT/release/Scenespy-linux-x64"
OUTPUT="$ROOT/release/Scenespy-${VERSION}-linux-amd64.deb"
WORK="$(mktemp -d)"
PACKAGE="$WORK/package"
trap 'rm -rf "$WORK"' EXIT

if [[ ! -x "$SOURCE/Scenespy" ]]; then
  echo "Build the Linux distribution before creating the Debian package." >&2
  exit 1
fi
if ! command -v dpkg-deb >/dev/null 2>&1; then
  echo "dpkg-deb is required to build the Debian package." >&2
  exit 1
fi

mkdir -p \
  "$PACKAGE/DEBIAN" \
  "$PACKAGE/opt/scenespy" \
  "$PACKAGE/usr/bin" \
  "$PACKAGE/usr/share/applications" \
  "$PACKAGE/usr/share/icons/hicolor/256x256/apps"
cp -a "$SOURCE"/. "$PACKAGE/opt/scenespy/"
ln -s /opt/scenespy/Scenespy "$PACKAGE/usr/bin/scenespy"
cp "$SOURCE/_internal/scenespy/assets/images/x-256.png" \
  "$PACKAGE/usr/share/icons/hicolor/256x256/apps/scenespy.png"

cat > "$PACKAGE/DEBIAN/control" <<CONTROL
Package: scenespy
Version: ${VERSION}
Section: video
Priority: optional
Architecture: amd64
Depends: ffmpeg, ca-certificates, curl
Maintainer: Scenespy
Description: Desktop video scene detection and editing
 Scenespy detects scene changes, cuts videos by intervals, and offers optional
 face detection components that can be installed from inside the application.
CONTROL

cat > "$PACKAGE/usr/share/applications/scenespy.desktop" <<'DESKTOP'
[Desktop Entry]
Type=Application
Name=Scenespy
Comment=Video scene detection and editing
Exec=/opt/scenespy/Scenespy
Icon=scenespy
Terminal=false
Categories=AudioVideo;Video;
StartupNotify=true
DESKTOP

find "$PACKAGE" -type d -exec chmod 755 {} +
chmod 755 "$PACKAGE/opt/scenespy/Scenespy" "$PACKAGE/opt/scenespy/install_runtime.sh"
rm -f "$OUTPUT"
dpkg-deb --root-owner-group --build "$PACKAGE" "$OUTPUT"
echo "Debian package ready: $OUTPUT"
