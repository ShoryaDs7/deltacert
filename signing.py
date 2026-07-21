"""
Certificate signing — turns a cert JSON from an editable file into a
tamper-evident artifact.

Mechanism: Ed25519 (asymmetric). Anyone holding the public key can verify a
signature without ever holding the private key — this is deliberate: the
whole point is that a third party (an auditor, a procurement team, a
regulator) can verify a certificate without trusting or coordinating with
whoever issued it.

What signing proves and does not prove:
    - PROVES: this exact JSON payload (byte-for-byte) was signed by whoever
      holds the private key, and has not been modified since.
    - DOES NOT PROVE: that the measurement inside the cert is correct, or
      that the collector that produced it has been flagship-validated. See
      the `validation_status` field (collectors.py) for that — it is signed
      as part of the payload precisely so a signature can never be used to
      make an unvalidated measurement look more trustworthy than it is.

Canonical form: signatures are computed over a deterministic serialization
(sorted keys, no whitespace) of the certificate with its own `signature`
field removed. This makes signing reproducible and independent of key
ordering or formatting choices made when the cert was originally written.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Optional

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

SIGNATURE_FIELD = "signature"
ALGORITHM = "Ed25519"


class SigningError(ValueError):
    """Raised for malformed keys, malformed certs, or malformed signature blocks."""


@dataclass(frozen=True)
class VerificationResult:
    ok: bool
    reason: str

    def __bool__(self) -> bool:
        return self.ok


# ──────────────────────────────────────────────────────────────────────────
# Key management
# ──────────────────────────────────────────────────────────────────────────

def generate_keypair() -> tuple[bytes, bytes]:
    """Generate a new Ed25519 keypair. Returns (private_key_pem, public_key_pem)."""
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()

    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return private_pem, public_pem


def load_private_key(path: str) -> Ed25519PrivateKey:
    with open(path, "rb") as f:
        data = f.read()
    try:
        key = serialization.load_pem_private_key(data, password=None)
    except Exception as e:
        raise SigningError(f"Could not load private key from {path}: {e}") from e
    if not isinstance(key, Ed25519PrivateKey):
        raise SigningError(
            f"{path} is not an Ed25519 private key "
            f"(got {type(key).__name__}). DeltaCert only supports Ed25519."
        )
    return key


def load_public_key(path: str) -> Ed25519PublicKey:
    with open(path, "rb") as f:
        data = f.read()
    try:
        key = serialization.load_pem_public_key(data)
    except Exception as e:
        raise SigningError(f"Could not load public key from {path}: {e}") from e
    if not isinstance(key, Ed25519PublicKey):
        raise SigningError(
            f"{path} is not an Ed25519 public key "
            f"(got {type(key).__name__}). DeltaCert only supports Ed25519."
        )
    return key


# ──────────────────────────────────────────────────────────────────────────
# Canonicalization
# ──────────────────────────────────────────────────────────────────────────

def canonicalize(cert: dict) -> bytes:
    """
    Deterministic byte serialization of a cert, excluding its own `signature`
    field. Same cert content always produces the same bytes regardless of
    key insertion order or prior formatting (indentation, whitespace).
    """
    if not isinstance(cert, dict):
        raise SigningError(f"Cannot canonicalize non-dict certificate: {type(cert).__name__}")
    payload = {k: v for k, v in cert.items() if k != SIGNATURE_FIELD}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


# ──────────────────────────────────────────────────────────────────────────
# Sign / verify
# ──────────────────────────────────────────────────────────────────────────

def sign_certificate(cert: dict, private_key: Ed25519PrivateKey, key_id: str = "") -> dict:
    """
    Returns a NEW dict — the input cert is never mutated — with a
    `signature` field added. Signing an already-signed cert re-signs over
    its current payload (the old signature is stripped by canonicalize()
    before the new one is computed, so double-signing does not nest).
    """
    payload = canonicalize(cert)
    signature = private_key.sign(payload)
    signed = dict(cert)
    signed[SIGNATURE_FIELD] = {
        "alg": ALGORITHM,
        "key_id": key_id,
        "value": base64.b64encode(signature).decode("ascii"),
    }
    return signed


def verify_certificate(cert: dict, public_key: Ed25519PublicKey) -> VerificationResult:
    """
    Recomputes the canonical payload and checks the signature. Never raises
    on a bad/tampered cert — returns a VerificationResult with a specific
    reason instead, so callers (CLI, CI gates) can report *why* it failed.
    """
    if not isinstance(cert, dict):
        return VerificationResult(False, f"certificate is not a JSON object (got {type(cert).__name__})")

    sig_block = cert.get(SIGNATURE_FIELD)
    if sig_block is None:
        return VerificationResult(False, "certificate has no 'signature' field — never signed")
    if not isinstance(sig_block, dict):
        return VerificationResult(False, "'signature' field is malformed (not an object)")

    alg = sig_block.get("alg")
    if alg != ALGORITHM:
        return VerificationResult(False, f"unsupported signature algorithm '{alg}' (expected {ALGORITHM})")

    value = sig_block.get("value")
    if not value or not isinstance(value, str):
        return VerificationResult(False, "'signature.value' is missing or not a string")

    try:
        signature_bytes = base64.b64decode(value, validate=True)
    except Exception:
        return VerificationResult(False, "'signature.value' is not valid base64")

    payload = canonicalize(cert)
    try:
        public_key.verify(signature_bytes, payload)
    except InvalidSignature:
        return VerificationResult(
            False,
            "signature does not match certificate contents — either the wrong "
            "key was used, or the certificate was modified after signing",
        )
    except Exception as e:
        return VerificationResult(False, f"signature verification raised an error: {e}")

    return VerificationResult(True, "signature valid")
