"""
One-off migration: bring existing validation_results/*/cert_*.json files up
to the current schema (adds `threshold_d` / `validation_status` where
missing) and signs each one with a local Ed25519 keypair.

Does NOT touch any measured value (d_comm, cos_sim, certified, per_domain,
n_samples, ...) — only adds the two new bookkeeping fields plus a signature
block. Run with --check to verify that invariant before trusting the output.

Usage:
    python scripts/backfill_signatures.py --keygen                 # first run: generates keys
    python scripts/backfill_signatures.py --check                  # dry run, reports diffs only
    python scripts/backfill_signatures.py                          # patches + signs in place
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from deltacert import signing as dsign
from deltacert.collectors import validation_status_for_layers, CERT_THRESHOLD_D

VALIDATION_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "validation_results")
DEFAULT_PRIVATE_KEY = os.path.join(os.path.dirname(__file__), "..", "deltacert-private.pem")
DEFAULT_PUBLIC_KEY = os.path.join(os.path.dirname(__file__), "..", "deltacert-public.pem")

# Fields this script is allowed to add/modify. Anything else changing between
# before/after is treated as a bug and aborts the run.
BOOKKEEPING_FIELDS = {"threshold_d", "validation_status", "signature"}


def _measured_content(cert: dict) -> dict:
    d = copy.deepcopy(cert)
    for f in BOOKKEEPING_FIELDS:
        d.pop(f, None)
    return d


def patch(cert: dict) -> dict:
    cert = copy.deepcopy(cert)
    cert.setdefault("threshold_d", CERT_THRESHOLD_D)
    cert["validation_status"] = validation_status_for_layers(cert.get("layers", {}).keys())
    return cert


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keygen", action="store_true", help="Generate a new signing keypair first")
    ap.add_argument("--check", action="store_true", help="Dry run: report what would change, write nothing")
    ap.add_argument("--private-key", default=DEFAULT_PRIVATE_KEY)
    ap.add_argument("--public-key", default=DEFAULT_PUBLIC_KEY)
    ap.add_argument("--key-id", default="threvo-labs-v1")
    args = ap.parse_args()

    if args.keygen:
        if os.path.exists(args.private_key):
            print(f"[backfill] {args.private_key} already exists, refusing to overwrite. "
                  f"Delete it first if you really want a new key.")
            sys.exit(1)
        priv_pem, pub_pem = dsign.generate_keypair()
        with open(args.private_key, "wb") as f:
            f.write(priv_pem)
        with open(args.public_key, "wb") as f:
            f.write(pub_pem)
        print(f"[backfill] Generated keypair: {args.private_key} / {args.public_key}")

    if not os.path.exists(args.private_key):
        print(f"[backfill] No private key at {args.private_key}. Run with --keygen first.")
        sys.exit(1)

    private_key = dsign.load_private_key(args.private_key)

    files = sorted(glob.glob(os.path.join(VALIDATION_RESULTS_DIR, "**", "cert_*.json"), recursive=True))
    print(f"[backfill] Found {len(files)} cert files.")

    n_changed = 0
    for f in files:
        with open(f) as fh:
            try:
                original = json.load(fh)
            except json.JSONDecodeError as e:
                print(f"[backfill] ERROR: malformed JSON in {f}: {e}")
                sys.exit(1)

        patched = patch(original)

        if _measured_content(original) != _measured_content(patched):
            print(f"[backfill] ABORT: {f} would change a measured field beyond "
                  f"{sorted(BOOKKEEPING_FIELDS)}. Refusing to write anything.")
            sys.exit(1)

        signed = dsign.sign_certificate(patched, private_key, key_id=args.key_id)

        rel = os.path.relpath(f, VALIDATION_RESULTS_DIR)
        if patched.get("validation_status") != original.get("validation_status"):
            print(f"[backfill] {rel}: validation_status -> {patched['validation_status']}")

        if not args.check:
            with open(f, "w") as fh:
                json.dump(signed, fh, indent=2)
        n_changed += 1

    verb = "Would sign" if args.check else "Signed"
    print(f"[backfill] {verb} {n_changed}/{len(files)} cert files.")
    if args.check:
        print("[backfill] --check mode: nothing was written.")


if __name__ == "__main__":
    main()
