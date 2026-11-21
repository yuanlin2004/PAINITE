#!/bin/bash
# Build PAINITE.app with Swift Package Manager (Xcode or its command line tools).
# Usage: ./build.sh            -> build/PAINITE.app
#        ./build.sh --install  -> also copy it to ~/Applications and register it
#        ./build.sh --debug    -> debug build
set -euo pipefail
cd "$(dirname "$0")"

CONFIG=release
INSTALL=0
for arg in "$@"; do
  case "$arg" in
    --install) INSTALL=1 ;;
    --debug) CONFIG=debug ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

swift build -c "$CONFIG" --arch arm64 --product PAINITE
BIN="$(swift build -c "$CONFIG" --arch arm64 --show-bin-path)/PAINITE"

APP="build/PAINITE.app"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp Resources/Info.plist "$APP/Contents/Info.plist"
cp "$BIN" "$APP/Contents/MacOS/PAINITE"

# App icon from docs/logo/painite-icon.svg. Needs rsvg-convert (brew install librsvg); skipped without it.
ICON_SVG=../docs/logo/painite-icon.svg
if command -v rsvg-convert >/dev/null && command -v iconutil >/dev/null; then
  ICONSET="$(mktemp -d)/AppIcon.iconset"
  mkdir -p "$ICONSET"
  for size in 16 32 128 256 512; do
    rsvg-convert -w $size -h $size "$ICON_SVG" -o "$ICONSET/icon_${size}x${size}.png"
    rsvg-convert -w $((size * 2)) -h $((size * 2)) "$ICON_SVG" -o "$ICONSET/icon_${size}x${size}@2x.png"
  done
  iconutil -c icns "$ICONSET" -o "$APP/Contents/Resources/AppIcon.icns"
  rm -rf "$(dirname "$ICONSET")"
else
  echo "note: rsvg-convert not found; building without an app icon" >&2
fi

# Ad-hoc signature: enough to run locally, not for distribution.
codesign --force --sign - "$APP"
echo "built: $APP"

if [[ $INSTALL == 1 ]]; then
  mkdir -p "$HOME/Applications"
  DEST="$HOME/Applications/PAINITE.app"
  rm -rf "$DEST"
  cp -R "$APP" "$DEST"
  /System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister -f "$DEST"
  echo "installed and registered: $DEST"
fi
