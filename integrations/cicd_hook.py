"""
integrations/cicd_hook.py — DeltaCert CI/CD gate

Blocks deployment automatically when certificate fails.
Exits with code 1 (blocks pipeline) if not certified.
Exits with code 0 (passes pipeline) if certified.

Works with: GitHub Actions, GitLab CI, Jenkins, CircleCI, any CI that checks exit codes.

Usage in GitHub Actions:
    - name: DeltaCert gate
      run: python -m deltacert.integrations.cicd_hook --cert ./cert.json

Usage in Python:
    from deltacert.integrations.cicd_hook import gate
    gate("./cert.json")   # raises SystemExit(1) if not certified

Usage in Dockerfile:
    RUN python -m deltacert.integrations.cicd_hook --cert /app/cert.json
"""

import argparse
import json
import os
import sys
import deltacert as dc


def gate(
    cert_path: str,
    required_layers: list = None,
    strict: bool = True,
) -> bool:
    """
    CI/CD deployment gate. Call this in your pipeline before deploying.

    Args:
        cert_path:       path to certificate JSON
        required_layers: layer names that must pass. None = all layers.
        strict:          if True, exit(1) on failure. if False, just return False.

    Returns:
        True if certified, False if not (only if strict=False)

    Raises:
        SystemExit(1) if not certified and strict=True
    """
    if not os.path.exists(cert_path):
        msg = f"DeltaCert CI gate FAILED: certificate not found at {cert_path}"
        print(msg)
        _write_github_output("deltacert_certified", "false")
        _write_github_output("deltacert_reason", "certificate_not_found")
        if strict:
            sys.exit(1)
        return False

    cert = dc.load_certificate(cert_path)
    ok, failures = dc.check_certified(cert, required_layers)

    print(dc.summary(cert))
    print()

    meta = cert.get("metadata", {})
    if meta:
        print(f"  certified_at:  {meta.get('certified_at', 'unknown')}")
        print(f"  model_id:      {meta.get('model_id', 'unknown')}")
        print(f"  gpu:           {meta.get('gpu_name', 'unknown')}")
        print()

    if ok:
        print("DeltaCert CI gate: PASSED. Deployment allowed.")
        _write_github_output("deltacert_certified", "true")
        _write_github_output("deltacert_reason", "all_layers_certified")
        return True
    else:
        print(f"DeltaCert CI gate: FAILED. Deployment BLOCKED.")
        print(f"  Failed layers: {failures}")
        print(f"  Fix: re-run certify() with higher-precision compression or more calibration data.")
        _write_github_output("deltacert_certified", "false")
        _write_github_output("deltacert_reason", f"failed_layers:{','.join(failures)}")
        if strict:
            sys.exit(1)
        return False


def _write_github_output(key: str, value: str) -> None:
    """Write to GitHub Actions output file if running in GitHub Actions."""
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"{key}={value}\n")


def main():
    parser = argparse.ArgumentParser(
        prog="deltacert-gate",
        description="DeltaCert CI/CD gate - blocks deployment if not certified",
    )
    parser.add_argument("--cert", required=True, help="Path to certificate JSON")
    parser.add_argument(
        "--layers", nargs="*", help="Layer names that must pass (default: all)"
    )
    parser.add_argument(
        "--no-strict", action="store_true",
        help="Return exit code 0 even if not certified (for dry-run)"
    )
    args = parser.parse_args()

    gate(
        cert_path=args.cert,
        required_layers=args.layers or None,
        strict=not args.no_strict,
    )


if __name__ == "__main__":
    main()
