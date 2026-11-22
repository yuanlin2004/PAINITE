# painite CLI

Enhance photos from the terminal on macOS and Linux. Talks to `painite-server`
over the contract in [`api/openapi.yaml`](../api/openapi.yaml).

## Build

Needs Go 1.24 or later. Release packages are built with GoReleaser
(`cli/.goreleaser.yaml`, `goreleaser release --snapshot --clean`). No cgo, so cross-compiling is one command:

```sh
cd cli
go build -o painite ./cmd/painite
GOOS=linux GOARCH=arm64 CGO_ENABLED=0 go build -o painite-linux-arm64 ./cmd/painite
```

## Configure

```sh
painite config set endpoint https://spark1:8443
painite trust --fingerprint SHA256:...     # from `painite-server admin fingerprint`
painite config set token pnt_...          # from `painite-server token create`
painite status
```

`painite trust` downloads the server's private CA, checks it against the
fingerprint the operator gave you, and saves it next to the config file. Run it
without `--fingerprint` to see what the server presents.

## Install on Ubuntu

Release builds are `.deb` packages for amd64 and arm64 with a `SHA256SUMS` file:

```sh
sha256sum -c --ignore-missing SHA256SUMS
sudo apt install ./painite_<version>_amd64.deb
```

The config file lives in the user config directory (`~/.config/painite/config.json`
on Linux, `~/Library/Application Support/painite/config.json` on macOS), or at
`$PAINITE_CONFIG`. `PAINITE_ENDPOINT`, `PAINITE_TOKEN`, and `PAINITE_CA_FILE`
override it, and `--endpoint`, `--token`, `--ca-file` override both.

## Use

```sh
painite enhance --op denoise --strength 0.6 -o out/ IMG_001.jpg IMG_002.tif
painite enhance --op face -o portraits/ *.jpg
painite enhance --op upscale2x --json --overwrite -o big.tif small.tif
painite enhance --op denoise:0.6,sharpen:0.3,upscale2x -o out/ IMG_001.tif   # pipeline
painite enhance --op unblur_motion -o fixed.jpg shaky.jpg                    # camera shake, motion blur
painite enhance --op unblur_focus -o fixed.jpg soft.jpg                      # missed focus
```

- A comma-separated `--op` runs a pipeline: the steps run in order in one
  server job, so there is one upload and one download. `op:strength` sets one
  step's strength; `--strength` sets it for the strength steps that don't.
  Up to four steps, with at most one upscale. `--model` works only with a
  single operation.

- With several inputs, or when `-o` is a directory or ends in `/`, each result
  keeps its input's name inside that directory. `--format jpeg|tiff` changes
  the output format (and extension).
- Everything is validated before anything is uploaded: missing or unsupported
  inputs, two inputs that resolve to the same output (rejected even with
  `--overwrite`), an output that is one of the inputs, and existing outputs
  without `--overwrite`. Any of these exits `2` and submits nothing.
- Up to four files are in flight at once (`--jobs`). Uploads are skipped when
  the server already has the file, failed requests are retried with backoff,
  and downloads resume after a dropped connection.
- Results are written to a temporary file in the destination directory,
  checked against the server's SHA-256, and renamed into place, so a failure
  never leaves a partial file or damages an existing one.
- `--json` prints `{"results": [...]}` on stdout with one entry per input, in
  argument order: `input`, `output`, `status` (`succeeded`, `failed`,
  `invalid`, `not_run`), `error.code`, plus server and client timings.
  Progress goes to stderr.
- Exit codes: `0` all succeeded, `1` any file failed (successful outputs are
  kept), `2` invalid invocation.
