#!/bin/bash
# Install painite-server on an NVIDIA DGX Spark (or another Linux box with an
# NVIDIA GPU, Docker and the NVIDIA Container Toolkit).
#
# Run from the unpacked release (painite-server-<version>/) or from server/ in a
# clone of the repository:
#
#   sudo ./install.sh                      # asks before downloading model weights
#   sudo ./install.sh --accept-licenses    # non-interactive
#
# What it does:
#   1. builds the painite-server Docker image here (from NVIDIA's PyTorch base image)
#   2. creates /var/lib/painite with a private TLS CA and a server certificate
#   3. downloads the models after you accept their licenses
#   4. installs /usr/local/bin/painite-server (runs the image's admin commands)
#      and the painite-server systemd service, then starts it
# Re-running it upgrades the image and keeps the CA, tokens and models.
set -euo pipefail
cd "$(dirname "$0")"

DATA_DIR=/var/lib/painite
PORT=8443
ACCEPT=0
SKIP_MODELS=0
HOSTNAMES=()
IPS=()

usage() {
  cat <<'EOF'
Usage: sudo ./install.sh [options]
  --accept-licenses   accept every model's license without asking
  --skip-models       don't download models now (placeholders answer until you do)
  --hostname NAME     a name clients use for this server (repeatable; default: this host's name and NAME.local)
  --ip ADDR           an IP address clients use (repeatable; default: this host's addresses)
  --port N            HTTPS port (default 8443)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --accept-licenses) ACCEPT=1 ;;
    --skip-models) SKIP_MODELS=1 ;;
    --hostname) HOSTNAMES+=("$2"); shift ;;
    --ip) IPS+=("$2"); shift ;;
    --port) PORT="$2"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

say() { printf '\n==> %s\n' "$*"; }
die() { echo "error: $*" >&2; exit 1; }

[[ $EUID -eq 0 ]] || die "run with sudo"
command -v docker >/dev/null || die "Docker is not installed"
docker info >/dev/null 2>&1 || die "Docker is not running"
docker info 2>/dev/null | grep -qi nvidia || echo "warning: Docker reports no NVIDIA runtime; models will run on the CPU (very slow)" >&2
command -v systemctl >/dev/null || die "systemd is required"

VERSION=$(sed -n 's/^version = "\(.*\)"/\1/p' pyproject.toml)
IMAGE=painite-server:$VERSION

say "Building $IMAGE (the first build downloads NVIDIA's PyTorch image, about 20 GB)"
docker build -t "$IMAGE" -t painite-server:latest .

say "Installing /usr/local/bin/painite-server"
cat > /usr/local/bin/painite-server <<EOF
#!/bin/sh
# Admin commands for the PAINITE server (installed by install.sh).
#   painite-server token create --name my-mac     painite-server models list
exec docker run --rm -i -v $DATA_DIR:/var/lib/painite painite-server:latest "\$@"
EOF
chmod 755 /usr/local/bin/painite-server
mkdir -p "$DATA_DIR"

if [[ ${#HOSTNAMES[@]} -eq 0 ]]; then
  HOSTNAMES=("$(hostname -s)" "$(hostname -s).local")
fi
if [[ ${#IPS[@]} -eq 0 ]]; then
  read -r -a IPS <<<"$(hostname -I 2>/dev/null | tr ' ' '\n' | grep -v ':' | grep -v '^172\.17\.' | tr '\n' ' ')"
fi
args=()
for h in "${HOSTNAMES[@]}"; do args+=(--hostname "$h"); done
for ip in "${IPS[@]}"; do args+=(--ip "$ip"); done

say "TLS certificate for: ${HOSTNAMES[*]} ${IPS[*]}"
painite-server admin init "${args[@]}"

if [[ $SKIP_MODELS == 0 ]]; then
  say "Models"
  painite-server models list
  if [[ $ACCEPT == 0 ]]; then
    echo
    echo "PAINITE ships no weights. Each model above is downloaded from its authors under its own"
    echo "license (some forbid commercial use). Review them before you continue."
    read -r -p "Download all models and accept their licenses? [y/N] " answer </dev/tty
    [[ $answer == [yY]* ]] || { echo "Skipped. Later: sudo painite-server models pull --all --accept-license"; SKIP_MODELS=1; }
  fi
  if [[ $SKIP_MODELS == 0 ]]; then
    painite-server models pull --all --accept-license
  fi
fi

say "Installing the painite-server service"
sed -e "s|^Environment=PAINITE_PORT=.*|Environment=PAINITE_PORT=$PORT|" \
    -e "s|-v /var/lib/painite:|-v $DATA_DIR:|" deploy/painite-server.service \
    > /etc/systemd/system/painite-server.service
systemctl daemon-reload
systemctl enable painite-server >/dev/null
systemctl restart painite-server

for _ in $(seq 1 60); do
  curl -fsS --cacert "$DATA_DIR/tls/ca.crt" "https://localhost:$PORT/v1/health" >/dev/null 2>&1 && break
  sleep 2
done
curl -fsS --cacert "$DATA_DIR/tls/ca.crt" "https://localhost:$PORT/v1/health" >/dev/null \
  || die "the server did not answer; see: journalctl -u painite-server"

FP=$(painite-server admin fingerprint | tail -n1)
say "PAINITE server $VERSION is running on https://${HOSTNAMES[0]}:$PORT"
cat <<EOF

Next, for each Mac or computer that will use it:
  1. Create a token:      sudo painite-server token create --name <computer-name>
  2. On the computer:     painite config set endpoint https://${HOSTNAMES[0]}:$PORT
                          painite trust --fingerprint $FP
                          painite config set token pnt_...
     In the Mac app, enter the same address, fingerprint and token in Settings.

User guide: https://yuanlin2004.github.io/PAINITE/
EOF
