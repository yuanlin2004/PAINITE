#!/bin/bash
# Build the Lightroom handoff spike app with the Xcode command line tools.
# Usage: ./build.sh            -> build/PAINITE Handoff Spike.app
#        ./build.sh --install  -> also copy it to ~/Applications and register it
set -euo pipefail
cd "$(dirname "$0")"

APP="build/PAINITE Handoff Spike.app"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS"
cp Info.plist "$APP/Contents/Info.plist"

xcrun swiftc -O -target arm64-apple-macos14 \
  -o "$APP/Contents/MacOS/PainiteHandoffSpike" Sources/main.swift

# Ad-hoc signature: enough to run locally, not for distribution.
codesign --force --sign - "$APP"
echo "built: $APP"

if [[ "${1:-}" == "--install" ]]; then
  mkdir -p "$HOME/Applications"
  DEST="$HOME/Applications/PAINITE Handoff Spike.app"
  rm -rf "$DEST"
  cp -R "$APP" "$DEST"
  /System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister -f "$DEST"
  echo "installed and registered: $DEST"
fi
