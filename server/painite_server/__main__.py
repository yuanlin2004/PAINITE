"""painite-server command line."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from .config import Settings
from .store import Store


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="painite-server")
    parser.add_argument("--data-dir", help="data directory (default $PAINITE_DATA_DIR, else /var/lib/painite on "
                                           "Linux or ~/Library/Application Support/painite-server on a Mac)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    serve = sub.add_parser("serve", help="run the HTTP server")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8443)
    serve.add_argument("--tls-cert", help="PEM certificate (default: <data-dir>/tls/server.crt if present)")
    serve.add_argument("--tls-key", help="PEM private key (default: <data-dir>/tls/server.key if present)")
    serve.add_argument("--insecure-http", action="store_true", help="serve plain HTTP even if certificates exist")
    serve.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"),
                       help="where models run (default $PAINITE_DEVICE or auto: CUDA, then Apple MPS, then CPU)")
    serve.add_argument("--log-level", default="info")

    admin = sub.add_parser("admin", help="server setup")
    asub = admin.add_subparsers(dest="acmd", required=True)
    ai = asub.add_parser("init", help="create the private CA (once) and a server certificate")
    ai.add_argument("--hostname", action="append", default=[],
                    help="DNS name clients use (repeatable; default: this host's name, name.local, localhost)")
    ai.add_argument("--ip", action="append", default=[], help="IP address clients use (repeatable)")
    ai.add_argument("--new-ca", action="store_true", help="replace the CA too (every client must trust it again)")
    asub.add_parser("fingerprint", help="print the CA fingerprint clients pin")

    tok = sub.add_parser("token", help="manage client tokens")
    tsub = tok.add_subparsers(dest="tcmd", required=True)
    tc = tsub.add_parser("create")
    tc.add_argument("--name", required=True)
    tsub.add_parser("list")
    tr = tsub.add_parser("revoke")
    tr.add_argument("--name", required=True)

    models = sub.add_parser("models", help="install and inspect models")
    msub = models.add_subparsers(dest="mcmd", required=True)
    msub.add_parser("list", help="catalog and installed models")
    mp = msub.add_parser("pull", help="download a model's weights after accepting its license")
    mp.add_argument("model", nargs="?", help="model ID from the catalog")
    mp.add_argument("--all", action="store_true", help="pull every catalog model")
    mp.add_argument("--manifest", help="install from this manifest file instead of the catalog")
    mp.add_argument("--accept-license", action="store_true", help="accept the model's license non-interactively")
    mr = msub.add_parser("remove")
    mr.add_argument("model")
    msub.add_parser("verify", help="check installed files against their SHA-256")
    return parser


def main(argv: list[str] | None = None) -> int:
    # On Apple Silicon, an op MPS lacks runs on the CPU instead of failing.
    # Must be set before torch is imported.
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    args = _parser().parse_args(argv)
    if getattr(args, "device", None):
        os.environ["PAINITE_DEVICE"] = args.device
    settings = Settings()
    if args.data_dir:
        settings.data_dir = Path(args.data_dir)
    settings.ensure_dirs()
    handler = {"serve": _serve, "admin": _admin, "token": _token, "models": _models}[args.cmd]
    return handler(args, settings)


def _serve(args, settings: Settings) -> int:
    import uvicorn

    from . import tls
    from .app import create_app

    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s %(message)s")
    logging.getLogger("pyvips").setLevel(logging.WARNING)
    log = logging.getLogger("painite")
    cert, key = args.tls_cert, args.tls_key
    p = tls.paths(settings.data_dir)
    if not cert and not args.insecure_http and p["server_cert"].exists() and p["server_key"].exists():
        cert, key = str(p["server_cert"]), str(p["server_key"])
    if args.insecure_http:
        cert = key = None
    if cert:
        log.info("TLS on (%s)", cert)
        if p["ca_cert"].exists():
            log.info("CA fingerprint %s", tls.fingerprint(p["ca_cert"].read_bytes()))
    else:
        log.warning("TLS is off: serving plain HTTP. Run `painite-server admin init` to enable it.")
    uvicorn.run(create_app(settings), host=args.host, port=args.port, ssl_certfile=cert, ssl_keyfile=key,
                log_level=args.log_level, access_log=False)
    return 0


def _admin(args, settings: Settings) -> int:
    from . import tls

    p = tls.paths(settings.data_dir)
    if args.acmd == "fingerprint":
        if not p["ca_cert"].exists():
            print("error: no CA yet; run `painite-server admin init`", file=sys.stderr)
            return 1
        print(tls.fingerprint(p["ca_cert"].read_bytes()))
        return 0
    names = args.hostname or tls.default_names()
    fp = tls.init(settings.data_dir, names, args.ip, force=args.new_ca)
    print(fp)
    print(f"Server certificate for {', '.join(names + args.ip)} written to {p['server_cert'].parent}.\n"
          "Restart the server to use it. On each client, run:\n"
          f"  painite trust --fingerprint {fp}", file=sys.stderr)
    return 0


def _token(args, settings: Settings) -> int:
    store = Store(settings.db_path)
    try:
        if args.tcmd == "create":
            try:
                _, secret = store.create_token(args.name)
            except ValueError as e:
                print(f"error: {e}", file=sys.stderr)
                return 1
            print(secret)
            print(f"Created token {args.name!r}. The line above (starting with pnt_) is the secret: give it to\n"
                  f"the client, e.g. `painite config set token {secret[:8]}…`. It is shown only once and\n"
                  "cannot be recovered; create a new token if it is lost.", file=sys.stderr)
        elif args.tcmd == "list":
            for t in store.list_tokens():
                created = datetime.fromtimestamp(t.created_at, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
                print(f"{t.name}\t{created}Z\t{'revoked' if t.revoked_at else 'active'}\tid {t.id} (not a secret)")
        elif args.tcmd == "revoke":
            if not store.revoke_token(args.name):
                print(f"error: no active token named {args.name!r}", file=sys.stderr)
                return 1
            print(f"Revoked {args.name!r}.", file=sys.stderr)
    finally:
        store.close()
    return 0


def _models(args, settings: Settings) -> int:
    from .models import ADAPTERS
    from .models import manifest as mf

    if args.mcmd == "list":
        inst = {m.id: d for m, d in mf.installed(settings.data_dir)}
        cat = mf.catalog()
        print("Built-in placeholders: denoise/identity@0.1.0, sharpen/identity@0.1.0, unblur_motion/identity@0.1.0,")
        print("  unblur_focus/identity@0.1.0, upscale2x/bilinear@0.1.0, face/identity@0.1.0")
        for mid in sorted(set(cat) | set(inst)):
            m = cat.get(mid) or mf.Manifest.load(inst[mid] / mf.MANIFEST)
            state = "installed" if mid in inst else "available"
            if m.adapter not in ADAPTERS:
                state += ", no adapter in this server"
            print(f"{mid}\t{state}\t{m.license.get('name')}")
        if not cat and not inst:
            print("No models in the catalog yet.")
        return 0

    if args.mcmd == "pull":
        try:
            if args.manifest:
                todo = [mf.Manifest.load(Path(args.manifest))]
            elif args.all:
                todo = list(mf.catalog().values())
            elif args.model:
                cat = mf.catalog()
                if args.model not in cat:
                    print(f"error: {args.model} is not in the catalog; see `painite-server models list`",
                          file=sys.stderr)
                    return 1
                todo = [cat[args.model]]
            else:
                print("error: give a model ID, --all, or --manifest", file=sys.stderr)
                return 2
        except mf.ManifestError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        for m in todo:
            if m.adapter not in ADAPTERS:
                print(f"error: {m.id} needs adapter {m.adapter!r}, which this server version does not have",
                      file=sys.stderr)
                return 1
        for m in todo:
            lic = m.license
            print(f"{m.id} is licensed under {lic.get('name')}" + (f": {lic['url']}" if lic.get("url") else ""),
                  file=sys.stderr)
        if not args.accept_license:
            print("Read the license(s), then rerun with --accept-license to download the weights.",
                  file=sys.stderr)
            return 1
        for m in todo:
            print(f"Downloading {m.id}…", file=sys.stderr)
            last = [-1]

            def progress(name: str, done: int, size: int | None) -> None:
                if size and done * 10 // size != last[0]:
                    last[0] = done * 10 // size
                    print(f"  {name}: {done * 100 // size}%", file=sys.stderr)

            try:
                d = mf.pull(m, settings.data_dir, progress)
            except (mf.ManifestError, OSError) as e:
                print(f"error: {m.id}: {e}", file=sys.stderr)
                return 1
            print(f"Installed {m.id} in {d}.", file=sys.stderr)
        print("Restart the server to load new models.", file=sys.stderr)
        return 0

    if args.mcmd == "remove":
        try:
            ok = mf.remove(args.model, settings.data_dir)
        except mf.ManifestError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        if not ok:
            print(f"error: {args.model} is not installed", file=sys.stderr)
            return 1
        print(f"Removed {args.model}. Restart the server to unload it.", file=sys.stderr)
        return 0

    if args.mcmd == "verify":
        bad = 0
        for m, d in mf.installed(settings.data_dir):
            problems = mf.verify(m, d)
            print(f"{m.id}\t{'ok' if not problems else '; '.join(problems)}")
            bad += bool(problems)
        return 1 if bad else 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
