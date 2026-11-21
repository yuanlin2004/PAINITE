"""painite-server command line: serve, token create/list/revoke."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone

from .config import Settings
from .store import Store


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="painite-server")
    parser.add_argument("--data-dir", help="data directory (default $PAINITE_DATA_DIR or /var/lib/painite)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    serve = sub.add_parser("serve", help="run the HTTP server")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8443)
    serve.add_argument("--tls-cert", help="PEM certificate; without it the server speaks plain HTTP")
    serve.add_argument("--tls-key", help="PEM private key")
    serve.add_argument("--log-level", default="info")

    tok = sub.add_parser("token", help="manage client tokens")
    tsub = tok.add_subparsers(dest="tcmd", required=True)
    tc = tsub.add_parser("create")
    tc.add_argument("--name", required=True)
    tsub.add_parser("list")
    tr = tsub.add_parser("revoke")
    tr.add_argument("--name", required=True)

    args = parser.parse_args(argv)
    settings = Settings()
    if args.data_dir:
        from pathlib import Path

        settings.data_dir = Path(args.data_dir)
    settings.ensure_dirs()

    if args.cmd == "serve":
        import uvicorn

        from .app import create_app

        logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s %(message)s")
        logging.getLogger("pyvips").setLevel(logging.WARNING)
        if not args.tls_cert:
            logging.getLogger("painite").warning("TLS is off: serving plain HTTP. Use only on a trusted network.")
        uvicorn.run(create_app(settings), host=args.host, port=args.port, ssl_certfile=args.tls_cert,
                    ssl_keyfile=args.tls_key, log_level=args.log_level, access_log=False)
        return 0

    store = Store(settings.db_path)
    try:
        if args.tcmd == "create":
            try:
                tok_, secret = store.create_token(args.name)
            except ValueError as e:
                print(f"error: {e}", file=sys.stderr)
                return 1
            print(secret)
            print(f"Created token {args.name!r}. The secret is the pnt_ line above; copy that into the client. "
                  f"It is shown only once. (Internal id {tok_.id} is not the token.)", file=sys.stderr)
        elif args.tcmd == "list":
            for t in store.list_tokens():
                created = datetime.fromtimestamp(t.created_at, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
                status = "revoked" if t.revoked_at else "active"
                print(f"{t.name}\t{t.id}\t{created}Z\t{status}")
        elif args.tcmd == "revoke":
            if not store.revoke_token(args.name):
                print(f"error: no active token named {args.name!r}", file=sys.stderr)
                return 1
            print(f"Revoked {args.name!r}.", file=sys.stderr)
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
