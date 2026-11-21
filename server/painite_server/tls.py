"""Private CA and server certificate for LAN deployments (DESIGN.md section 5,
Security). `painite-server admin init` writes them to <data_dir>/tls/; clients
pin the CA by its SHA-256 fingerprint (`painite trust`)."""

from __future__ import annotations

import datetime as dt
import ipaddress
import os
import socket
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

CA_DAYS = 3650
SERVER_DAYS = 825  # stays under client limits on certificate lifetime


def tls_dir(data_dir: Path) -> Path:
    return data_dir / "tls"


def paths(data_dir: Path) -> dict[str, Path]:
    d = tls_dir(data_dir)
    return {
        "ca_key": d / "ca.key",
        "ca_cert": d / "ca.crt",
        "server_key": d / "server.key",
        "server_cert": d / "server.crt",
    }


def fingerprint(cert_pem: bytes) -> str:
    cert = x509.load_pem_x509_certificate(cert_pem)
    return "SHA256:" + cert.fingerprint(hashes.SHA256()).hex()


def _write(path: Path, data: bytes, private: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600 if private else 0o644)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def _key_pem(key: ec.EllipticCurvePrivateKey) -> bytes:
    return key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                             serialization.NoEncryption())


def default_names() -> list[str]:
    host = socket.gethostname()
    names = [host]
    if "." not in host:
        names.append(host + ".local")
    names.append("localhost")
    return names


def init(data_dir: Path, names: list[str], ips: list[str], force: bool = False) -> str:
    """Create the CA (unless it exists) and a fresh server certificate.
    Returns the CA fingerprint."""
    p = paths(data_dir)
    now = dt.datetime.now(dt.timezone.utc)

    if p["ca_cert"].exists() and p["ca_key"].exists() and not force:
        ca_key = serialization.load_pem_private_key(p["ca_key"].read_bytes(), password=None)
        ca_cert = x509.load_pem_x509_certificate(p["ca_cert"].read_bytes())
    else:
        ca_key = ec.generate_private_key(ec.SECP256R1())
        ca_name = x509.Name([
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "PAINITE"),
            x509.NameAttribute(NameOID.COMMON_NAME, f"PAINITE private CA ({socket.gethostname()})"),
        ])
        ca_cert = (
            x509.CertificateBuilder()
            .subject_name(ca_name).issuer_name(ca_name)
            .public_key(ca_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - dt.timedelta(minutes=5))
            .not_valid_after(now + dt.timedelta(days=CA_DAYS))
            .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
            .add_extension(x509.KeyUsage(digital_signature=True, key_cert_sign=True, crl_sign=True,
                                         content_commitment=False, key_encipherment=False, data_encipherment=False,
                                         key_agreement=False, encipher_only=False, decipher_only=False),
                           critical=True)
            .add_extension(x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key()), critical=False)
            .sign(ca_key, hashes.SHA256())
        )
        _write(p["ca_key"], _key_pem(ca_key), private=True)
        _write(p["ca_cert"], ca_cert.public_bytes(serialization.Encoding.PEM), private=False)

    sans: list[x509.GeneralName] = [x509.DNSName(n) for n in dict.fromkeys(names)]
    sans += [x509.IPAddress(ipaddress.ip_address(ip)) for ip in dict.fromkeys(ips)]
    server_key = ec.generate_private_key(ec.SECP256R1())
    server_cert = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, names[0] if names else ips[0])]))
        .issuer_name(ca_cert.subject)
        .public_key(server_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - dt.timedelta(minutes=5))
        .not_valid_after(now + dt.timedelta(days=SERVER_DAYS))
        .add_extension(x509.SubjectAlternativeName(sans), critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
        .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()), critical=False)
        .sign(ca_key, hashes.SHA256())
    )
    _write(p["server_key"], _key_pem(server_key), private=True)
    _write(p["server_cert"], server_cert.public_bytes(serialization.Encoding.PEM), private=False)
    return fingerprint(p["ca_cert"].read_bytes())
