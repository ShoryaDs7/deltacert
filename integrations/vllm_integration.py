"""
integrations/vllm_integration.py — DeltaCert official vLLM plugin

Uses vLLM's official plugin system (vllm.general_plugins entry point).
vLLM calls register() automatically at startup in every process.

Install once:
    pip install deltacert

Register in your pyproject.toml:
    [project.entry-points."vllm.general_plugins"]
    deltacert = "deltacert.integrations.vllm_integration:register"

Opt-in enforcement — two env vars required to activate:
    export DELTACERT_ENFORCE=1
    export DELTACERT_PATH=./llama3_70b_int4_tp4_certified.json
    vllm serve meta-llama/Llama-3-70B --quantization awq --tensor-parallel-size 4

    # certified=true  → vLLM starts normally
    # certified=false → vLLM refuses to start, exit code 1

Without DELTACERT_ENFORCE=1, the plugin is a complete no-op regardless of
what other env vars are set. pip install deltacert in a shared image is safe.
"""

import os
import sys
import deltacert as dc


def register():
    """
    Official vLLM plugin entry point.

    vLLM discovers and calls this function automatically at startup
    via the vllm.general_plugins entry point — before any engine
    is initialized, before any model weights are loaded.

    Environment variables:
        DELTACERT_ENFORCE — must be "1" to activate enforcement.
                            Without this, the plugin is a complete no-op.
        DELTACERT_PATH    — path to certificate JSON from dc.certify().
        DELTACERT_LAYERS  — optional. Comma-separated layer names that must pass.

    Enforcement is something an ops team turns ON explicitly. pip install alone
    never activates enforcement — someone's prod serving cannot break from a
    data scientist installing deltacert in a shared image.
    """
    if os.environ.get("DELTACERT_ENFORCE") != "1":
        return

    cert_path = os.environ.get("DELTACERT_PATH")
    if not cert_path:
        print(
            "[DeltaCert] DELTACERT_ENFORCE=1 is set but DELTACERT_PATH is missing.\n"
            "[DeltaCert] Set DELTACERT_PATH=./cert.json or unset DELTACERT_ENFORCE."
        )
        sys.exit(1)

    required_layers = None
    layers_env = os.environ.get("DELTACERT_LAYERS")
    if layers_env:
        required_layers = [l.strip() for l in layers_env.split(",") if l.strip()]

    # Certificate must exist — missing cert = deployment blocked
    if not os.path.exists(cert_path):
        print(
            f"[DeltaCert] FATAL: certificate not found at '{cert_path}'.\n"
            f"[DeltaCert] Run certify first:\n"
            f"[DeltaCert]   deltacert certify --config stack.yaml "
            f"--calibration calib.json --output {cert_path}"
        )
        sys.exit(1)

    try:
        cert = dc.load_certificate(cert_path)
    except Exception as e:
        print(f"[DeltaCert] FATAL: failed to read certificate at '{cert_path}': {e}")
        sys.exit(1)

    ok, failures = dc.check_certified(cert, required_layers)

    # Always print metadata so ops teams have a paper trail in server logs
    meta = cert.get("metadata", {})
    print("[DeltaCert] ------------------------------------------")
    print(f"[DeltaCert] model:        {meta.get('model_id', cert.get('model', '?'))}")
    print(f"[DeltaCert] certified_at: {meta.get('certified_at', 'unknown')}")
    print(f"[DeltaCert] host:         {meta.get('host', 'unknown')}")
    print(f"[DeltaCert] gpu:          {meta.get('gpu_name', 'unknown')}")
    print(f"[DeltaCert] torch:        {meta.get('torch_version', 'unknown')}")
    print("[DeltaCert] ------------------------------------------")
    print(dc.summary(cert))
    print("[DeltaCert] ------------------------------------------")

    if not ok:
        print(f"[DeltaCert] REFUSED TO START.")
        print(f"[DeltaCert] Failed layers: {failures}")
        print(f"[DeltaCert] Fix: re-run dc.certify() with higher-precision compression")
        print(f"[DeltaCert]      or increase calibration data size.")
        sys.exit(1)

    print("[DeltaCert] Certificate verified. vLLM engine starting.")
    print("[DeltaCert] ------------------------------------------")
