# painite-server

The PAINITE inference server (FastAPI). It implements the contract in
[`api/openapi.yaml`](../api/openapi.yaml); the design is in
[`docs/DESIGN.md`](../docs/DESIGN.md).

**Models:** the catalog has the M0 bake-off picks ([bench/BAKEOFF.md](../bench/BAKEOFF.md)):
`denoise/scunet-real-gan`, `sharpen/realesrgan-x2-down` (Real-ESRGAN 2x, then
area downscale) and `upscale2x/nomosuni-esrgan-multijpg`, each followed by a
low-frequency color anchor, plus `face/gpen-bfr-1024` (RetinaFace detection and GPEN-BFR-1024
restoration of faces 128 to 1024 px, [bench/FACE_BAKEOFF.md](../bench/FACE_BAKEOFF.md); since 1.1.0 a
BiSeNet face parser limits the paste to the face itself, so hands, arms, hair and background near a
face keep their original pixels, [bench/results/face/parse_mask](../bench/results/face/parse_mask/)). Unblur
([bench/UNBLUR.md](../bench/UNBLUR.md)) has two operations: `unblur_motion/stripformer-realblurj`
for camera shake and `unblur_focus/lakdnet-defocus-lfdpdd-s` for missed focus. They run through the
`torch_arch` adapter with network code vendored in `painite_server/models/arches/` (Stripformer:
non-commercial license; LaKDNet: AGPL-3.0). Install them with `models pull --all --accept-license`; the vendored network code is not shipped and is downloaded with the weights ([third_party/arches](third_party/arches/README.md)).
Until they are installed, placeholder models (`denoise/identity`, `sharpen/identity`,
`unblur_motion/identity`, `unblur_focus/identity`, `upscale2x/bilinear`, `face/identity`) answer instead.

## Run locally

Needs Python 3.11+ and libvips (`apt install libvips42`, `brew install vips`).
Real models also need PyTorch with torchvision and `pip install -e '.[models]'`.
Models run on an NVIDIA GPU (CUDA) if there is one, else on an Apple Silicon GPU
(MPS), else on the CPU, slowly. `PAINITE_DEVICE` (or `serve --device`) forces
`cuda`, `mps` or `cpu`.

On CUDA, most models switch to a TensorRT engine or `torch.compile` once it has
been built and checked against PyTorch. Builds run in the background after
start-up (about 5 minutes the first time, then loaded from
`$PAINITE_DATA_DIR/engines` in seconds), and PyTorch serves until then.
`GET /v1/health` shows each model's backend under `accel`. Settings:
`PAINITE_ACCEL` (`auto`, `off`, `tensorrt` or `compile`),
`PAINITE_ACCEL_BUILD_TIMEOUT_S` (900), `PAINITE_ACCEL_MIN_PSNR` (50). A model can
list several backends to try in order (unblur_motion: TensorRT, then
torch.compile), and can split its TensorRT build into several engines at named
submodules (`split_after`; Stripformer needs this on a GB10). Failed builds are
recorded and skipped on later starts, except out-of-memory and time-limit
failures, which get one more try. To retry everything, start once with
`PAINITE_ACCEL_RETRY=1`, or delete the engines directory.
See `painite_server/models/accel.py`.

```sh
cd server
python3 -m venv .venv && . .venv/bin/activate
pip install -e '.[test]'
export PAINITE_DATA_DIR=$PWD/.data
painite-server token create --name dev      # prints the pnt_… secret once (stdout)
painite-server serve --port 8443
```

## Install on DGX Spark (release)

From a release, unpack `painite-server_<version>.tar.gz` and run `sudo ./install.sh`
(`--help` for options). It builds the image, runs `admin init`, pulls the models
after asking about their licenses, installs `/usr/local/bin/painite-server` (the
admin commands below, run in the image) and the systemd service. The user guide
covers it: https://yuanlin2004.github.io/PAINITE/install.html

## Install on a Mac (Apple Silicon)

Docker on a Mac has no GPU, so the server runs natively, with PyTorch on the
Mac's GPU (MPS). From a release or from `server/` in a clone, run
`./install-mac.sh` as yourself (not sudo; `--help` for options). It needs `uv`
(`brew install uv`) or Python 3.11+. It makes a venv with PyTorch and a libvips
wheel (no Homebrew vips), keeps everything in `~/Library/Application Support/painite-server`,
pulls the models after asking about their licenses, installs
`~/.local/bin/painite-server` and a launchd agent (`com.painite.server`, log in
`~/Library/Logs/painite-server.log`) that serves `https://localhost:8443`.
`--lan` also serves other computers; `--uninstall` removes the service and venv
and keeps the data.

On a Mac the models run through Core ML rather than PyTorch where that is
faster (`painite_server/models/coreml.py`): sharpen and upscale2x on the Neural
Engine, denoise, both unblurs and the face restorer on the GPU. The installer
converts each model once from its own weights (about 4 minutes for all of them;
`painite-server models coreml` does it again by hand) and keeps the result next to
the weights; a server that finds no conversion makes it at startup. Images smaller
than one tile window (640 px) run on PyTorch. `PAINITE_COREML=off` in the launchd
agent's environment keeps everything on PyTorch (MPS).

On an M2 Pro (16 GB) a 24 MP job takes 1.2 to 2.1 minutes depending on the
operation (spark1: about 0.5 minutes), previews take 20 to 33 s, and the server
uses up to about 9 GB of memory.

## Run on DGX Spark (Docker, by hand)

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

`models pull` downloads each file from the project's mirror (the `models-v1`
release, filled by `.github/workflows/mirror-models.yml`) and falls back to the
upstream URL; both are checked against the manifest's SHA-256. Set
`PAINITE_MODEL_MIRROR` to another base URL, or to an empty string for upstream only.

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
