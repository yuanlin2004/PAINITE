#!/bin/bash
# Install painite-server natively on a Mac with Apple Silicon. Models run on the
# Mac's GPU through PyTorch MPS (Docker on a Mac has no GPU, so no container).
#
# Run as yourself (not sudo) from the unpacked release (painite-server-<version>/)
# or from server/ in a clone of the repository:
#
#   ./install-mac.sh                      # asks before downloading model weights
#   ./install-mac.sh --accept-licenses    # non-interactive
#   ./install-mac.sh --lan                # also serve other computers on the network
#   ./install-mac.sh --uninstall          # stop and remove the service and venv; keeps data
#
# What it does:
#   1. makes a Python venv with PyTorch, libvips (pyvips wheel) and the server
#   2. creates ~/Library/Application Support/painite-server with a private TLS CA
#      and a server certificate
#   3. downloads the models after you accept their licenses
#   4. installs ~/.local/bin/painite-server (admin commands) and a launchd agent
#      that runs the server at login, then starts it
# Re-running it upgrades the server and keeps the CA, tokens and models.
set -euo pipefail
cd "$(dirname "$0")"
SRC=$(pwd)

APP_DIR="$HOME/Library/Application Support/painite-server"
DATA_DIR="$APP_DIR"
VENV="$APP_DIR/venv"
BIN_DIR="$HOME/.local/bin"
LABEL=com.painite.server
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
LOG="$HOME/Library/Logs/painite-server.log"
PORT=8443
LAN=0
ACCEPT=0
SKIP_MODELS=0
UNINSTALL=0
HOSTNAMES=()
IPS=()

usage() {
  cat <<'EOF'
Usage: ./install-mac.sh [options]
  --accept-licenses   accept every model's license without asking
  --skip-models       don't download models now (placeholders answer until you do)
  --lan               listen on all interfaces so other computers can use this Mac
                      (default: this Mac only, on localhost)
  --hostname NAME     an extra name clients use for this server (repeatable)
  --ip ADDR           an extra IP address clients use (repeatable; --lan adds this Mac's)
  --port N            HTTPS port (default 8443)
  --uninstall         stop the server and remove the service, venv and command;
                      keeps models, tokens and the CA in the data directory
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --accept-licenses) ACCEPT=1 ;;
    --skip-models) SKIP_MODELS=1 ;;
    --lan) LAN=1 ;;
    --hostname) HOSTNAMES+=("$2"); shift ;;
    --ip) IPS+=("$2"); shift ;;
    --port) PORT="$2"; shift ;;
    --uninstall) UNINSTALL=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

say() { printf '\n==> %s\n' "$*"; }
die() { echo "error: $*" >&2; exit 1; }

[[ $(uname -s) == Darwin ]] || die "this script is for macOS; on Linux use install.sh"
[[ $EUID -ne 0 ]] || die "run as yourself, not with sudo"

stop_agent() {
  launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
}

if [[ $UNINSTALL == 1 ]]; then
  say "Removing the painite-server service"
  stop_agent
  rm -f "$PLIST" "$BIN_DIR/painite-server"
  rm -rf "$VENV"
  echo "Removed. Models, tokens and the CA are still in $DATA_DIR; delete that folder to remove them too."
  exit 0
fi

[[ $(uname -m) == arm64 ]] || echo "warning: this is not an Apple Silicon Mac; models will run on the CPU (very slow)" >&2

VERSION=$(sed -n 's/^version = "\(.*\)"/\1/p' pyproject.toml)

# Python: uv if present (it fetches its own Python 3.12), else python3 3.11+.
if command -v uv >/dev/null; then
  PY_SETUP=uv
elif command -v python3 >/dev/null && python3 -c 'import sys; sys.exit(sys.version_info < (3, 11))' 2>/dev/null; then
  PY_SETUP=python3
else
  die "needs uv (https://docs.astral.sh/uv/, e.g. \`brew install uv\`) or Python 3.11 or newer"
fi

say "Installing painite-server $VERSION into $VENV (PyTorch is about 1 GB)"
mkdir -p "$APP_DIR"
stop_agent  # don't replace files under a running server
if [[ $PY_SETUP == uv ]]; then
  [[ -x "$VENV/bin/python" ]] || uv venv --quiet --python 3.12 "$VENV"
  uv pip install --quiet --python "$VENV/bin/python" --upgrade "$SRC[models,mac]"
else
  [[ -x "$VENV/bin/python" ]] || python3 -m venv "$VENV"
  "$VENV/bin/python" -m pip install --quiet --upgrade pip
  "$VENV/bin/python" -m pip install --quiet --upgrade "$SRC[models,mac]"
fi
"$VENV/bin/python" - <<'EOF'
import pyvips, torch
mps = torch.backends.mps.is_available()
print(f"torch {torch.__version__}, MPS {'available' if mps else 'NOT available (CPU only)'}, libvips {pyvips.version(0)}.{pyvips.version(1)}")
EOF

say "Installing $BIN_DIR/painite-server"
mkdir -p "$BIN_DIR"
# python -m rather than the venv's script: its path has a space, which breaks shebangs.
cat > "$BIN_DIR/painite-server" <<EOF
#!/bin/sh
# Admin commands for the PAINITE server on this Mac (installed by install-mac.sh).
#   painite-server token create --name my-mac     painite-server models list
export PAINITE_DATA_DIR="$DATA_DIR"
exec "$VENV/bin/python" -m painite_server "\$@"
EOF
chmod 755 "$BIN_DIR/painite-server"
ps_admin() { "$BIN_DIR/painite-server" "$@"; }

SHORT=$(scutil --get LocalHostName 2>/dev/null || hostname -s)
names=("$SHORT" "$SHORT.local" localhost ${HOSTNAMES[@]+"${HOSTNAMES[@]}"})
ips=(127.0.0.1 ${IPS[@]+"${IPS[@]}"})
if [[ $LAN == 1 ]]; then
  for ifc in en0 en1; do
    ip=$(ipconfig getifaddr "$ifc" 2>/dev/null || true)
    [[ -n $ip ]] && ips+=("$ip")
  done
fi
args=()
for h in "${names[@]}"; do args+=(--hostname "$h"); done
for ip in "${ips[@]}"; do args+=(--ip "$ip"); done

say "TLS certificate for: ${names[*]} ${ips[*]}"
ps_admin admin init "${args[@]}"

if [[ $SKIP_MODELS == 0 ]]; then
  say "Models"
  ps_admin models list
  if [[ $ACCEPT == 0 ]]; then
    echo
    echo "PAINITE ships no weights. Each model above is downloaded from its authors under its own"
    echo "license (some forbid commercial use). Review them before you continue."
    read -r -p "Download all models and accept their licenses? [y/N] " answer </dev/tty
    [[ $answer == [yY]* ]] || { echo "Skipped. Later: painite-server models pull --all --accept-license"; SKIP_MODELS=1; }
  fi
  if [[ $SKIP_MODELS == 0 ]]; then
    ps_admin models pull --all --accept-license
  fi
fi

HOST=127.0.0.1
[[ $LAN == 1 ]] && HOST=0.0.0.0
say "Installing the launchd agent $LABEL (listening on $HOST:$PORT)"
mkdir -p "$(dirname "$PLIST")" "$(dirname "$LOG")"
xml() { sed -e 's/&/\&amp;/g' -e 's/</\&lt;/g' -e 's/>/\&gt;/g' <<<"$1"; }
cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>$(xml "$VENV/bin/python")</string>
    <string>-m</string><string>painite_server</string>
    <string>serve</string>
    <string>--host</string><string>$HOST</string>
    <string>--port</string><string>$PORT</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PAINITE_DATA_DIR</key><string>$(xml "$DATA_DIR")</string>
    <key>PYTORCH_ENABLE_MPS_FALLBACK</key><string>1</string>
  </dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><dict><key>SuccessfulExit</key><false/></dict>
  <key>ProcessType</key><string>Interactive</string>
  <key>StandardOutPath</key><string>$(xml "$LOG")</string>
  <key>StandardErrorPath</key><string>$(xml "$LOG")</string>
</dict>
</plist>
EOF
plutil -lint -s "$PLIST" || die "bad launchd plist $PLIST"
launchctl bootstrap "gui/$(id -u)" "$PLIST"

for _ in $(seq 1 60); do
  curl -fsS --cacert "$DATA_DIR/tls/ca.crt" "https://localhost:$PORT/v1/health" >/dev/null 2>&1 && break
  sleep 2
done
health=$(curl -fsS --cacert "$DATA_DIR/tls/ca.crt" "https://localhost:$PORT/v1/health") \
  || die "the server did not answer; see $LOG"

FP=$(ps_admin admin fingerprint | tail -n1)
say "PAINITE server $VERSION is running on https://localhost:$PORT"
echo "$health"
[[ $LAN == 1 ]] && echo "Other computers can use https://$SHORT.local:$PORT (macOS may ask to allow incoming connections for Python)."
cat <<EOF

Next, for each computer that will use it (this Mac included):
  1. Create a token:      painite-server token create --name <computer-name>
  2. On the computer:     painite config set endpoint https://localhost:$PORT
                          painite trust --fingerprint $FP
                          painite config set token <the pnt_… token from step 1>
     (from another computer, use https://$SHORT.local:$PORT; needs --lan)
  3. Check:               painite status
     In the Mac app, enter the same address, fingerprint and token in Settings.

Logs: $LOG
Stop:  launchctl bootout gui/\$(id -u)/$LABEL      Start: launchctl bootstrap gui/\$(id -u) "$PLIST"
EOF
[[ ":$PATH:" == *":$BIN_DIR:"* ]] || echo "Note: add $BIN_DIR to your PATH to run painite-server."
