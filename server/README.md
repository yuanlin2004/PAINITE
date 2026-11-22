# painite-server

The PAINITE inference server (FastAPI). It implements the contract in
[`api/openapi.yaml`](../api/openapi.yaml); the design is in
[`docs/DESIGN.md`](../docs/DESIGN.md).

**Models:** the catalog has the M0 bake-off picks ([bench/BAKEOFF.md](../bench/BAKEOFF.md)):
`denoise/scunet-real-gan`, `sharpen/realesrgan-x2-down` (Real-ESRGAN 2x, then
area downscale) and `upscale2x/nomosuni-esrgan-multijpg`, each followed by a
low-frequency color anchor. Install them with `models pull --all --accept-license`.
Until they are installed, placeholder models (`denoise/identity`,
`sharpen/identity`, `upscale2x/bilinear`) answer instead.

## Run locally

Needs Python 3.11+ and libvips (`apt install libvips42`, `brew install vips`).
Real models also need PyTorch with torchvision and `pip install -e '.[models]'`;
without CUDA they run on the CPU, slowly.

```sh
cd server
python3 -m venv .venv && . .venv/bin/activate
pip install -e '.[test]'
export PAINITE_DATA_DIR=$PWD/.data
painite-server token create --name dev      # prints the pnt_… secret once (stdout)
painite-server serve --port 8443
```

## Run on DGX Spark (Docker)

```sh
docker build -t painite-server:latest server/
sudo mkdir -p /var/lib/painite
alias ps-admin='docker run --rm -v /var/lib/painite:/var/lib/painite painite-server:latest'

# TLS: a private CA (created once) and a server certificate for the names clients use.
ps-admin admin init --hostname spark1 --hostname spark1.local --ip 192.168.29.160
#   prints the CA fingerprint; on each client: painite trust --fingerprint SHA256:...

ps-admin token create --name studio-mac      # prints the pnt_… secret once (stdout)

sudo cp server/deploy/painite-server.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now painite-server
curl --cacert /var/lib/painite/tls/ca.crt https://spark1:8443/v1/health
```

`serve` uses `<data-dir>/tls/server.{crt,key}` automatically when they exist;
`--insecure-http` forces plain HTTP. Re-running `admin init` keeps the CA (clients
stay trusted) and reissues the server certificate; `--new-ca` replaces both.

## Models

PAINITE does not ship weights. Each model has a manifest (download URLs,
SHA-256, license) in `painite_server/models/catalog/`, and the operator
downloads the weights after accepting the license:

```sh
ps-admin models list
ps-admin models pull --all --accept-license      # or one ID, e.g. denoise/scunet-real-gan@1.0.0
ps-admin models verify
sudo systemctl restart painite-server
```

Installed models live in `/var/lib/painite/models/<operation>/<name>@<version>/`,
so several versions can be installed side by side and clients can pin one. A
manifest marked `"default": true` becomes its operation's default; otherwise
the first installed model does. `models pull --manifest file.json` installs a
model that is not in the catalog.

## Settings

Environment variables (defaults in `painite_server/config.py`):

| Variable | Default |
| --- | --- |
| `PAINITE_DATA_DIR` | `/var/lib/painite` |
| `PAINITE_MAX_UPLOAD_BYTES` | 512 MiB |
| `PAINITE_MAX_INPUT_PIXELS` | 48,000,000 |
| `PAINITE_MAX_OUTPUT_PIXELS` | 192,000,000 |
| `PAINITE_QUEUE_CAPACITY` | 32 |
| `PAINITE_RESULT_RETENTION_S` | 86400 |
| `PAINITE_SOURCE_IDLE_S` | 900 (sources with no active job are deleted after this idle time) |
| `PAINITE_DISK_CAP_BYTES` | 200 GiB |
| `PAINITE_TILE_SIZE` / `PAINITE_TILE_OVERLAP` | 512 / 64 (a model may ask for more context) |

## Tests

```sh
pytest
```
