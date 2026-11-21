# painite-server

The PAINITE inference server (FastAPI). It implements the contract in
[`api/openapi.yaml`](../api/openapi.yaml); the design is in
[`docs/DESIGN.md`](../docs/DESIGN.md).

**Status: skeleton.** Only placeholder models are installed: `denoise/identity`,
`sharpen/identity` (both return the input), and `upscale2x/bilinear`. Real
models replace them after the model bake-off. TLS certificate setup
(`admin init`) is not built yet; pass `--tls-cert/--tls-key` or run on a trusted
network.

## Run locally

Needs Python 3.11+ and libvips (`apt install libvips42`, `brew install vips`).

```sh
cd server
python3 -m venv .venv && . .venv/bin/activate
pip install -e '.[test]'
export PAINITE_DATA_DIR=$PWD/.data
painite-server token create --name dev      # prints the token once
painite-server serve --port 8443
```

## Run on DGX Spark (Docker)

```sh
docker build -t painite-server server/
sudo mkdir -p /var/lib/painite
docker run --rm -v /var/lib/painite:/var/lib/painite painite-server token create --name studio-mac
docker run -d --name painite --gpus all -p 8443:8443 -v /var/lib/painite:/var/lib/painite painite-server
curl http://spark1.local:8443/v1/health
```

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
| `PAINITE_TILE_SIZE` / `PAINITE_TILE_OVERLAP` | 512 / 32 |

## Tests

```sh
pytest
```
