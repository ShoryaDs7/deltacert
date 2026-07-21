"""
DeltaCert smoke tests — plumbing verification for collectors 14, 19, 1.

These three tests gate the full 20-collector suite:
  - prefix_cache (14): proves math path + KV cache mechanics
  - prompt_swap (19):  proves HF forward pass + real divergence detection
  - engine_swap (1):   proves save_logits/load_logits round-trip + cert schema

Passing all 3 means d_comm math, serialization, and cert schema work.
It does NOT mean the other 17 collectors are correct — each has its own
failure surface (hook mechanics, vLLM API, trajectory capture, etc.).

Run:
    pip install -e .
    pip install transformers accelerate bitsandbytes
    huggingface-cli download meta-llama/Llama-3.2-3B-Instruct --local-dir D:/models/llama3-3b
    pytest deltacert/tests/test_smoke.py -v

Or with Qwen:
    huggingface-cli download Qwen/Qwen2.5-3B-Instruct --local-dir D:/models/qwen2.5-3b
    pytest deltacert/tests/test_smoke.py -v --model D:/models/qwen2.5-3b
"""

import os
import math
import pytest

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from deltacert.collectors import (
    capture_logits_hf,
    collect_prefix_cache,
    collect_prompt_swap,
    collect_engine_swap,
    save_logits,
    CollectionError,
)
from deltacert.deltacert import certify_layer
from deltacert.collectors import CERT_THRESHOLD_D

# ---------------------------------------------------------------------------
# Model path
# ---------------------------------------------------------------------------

def _model_path():
    p = os.environ.get("DELTACERT_SMOKE_MODEL") or pytest.smoke_model
    if not os.path.isdir(p):
        pytest.skip(f"Model not found at {p} — set --model or DELTACERT_SMOKE_MODEL env var")
    return p


PROMPTS = [
    "What is the capital of France?",
    "Explain Newton's second law in one sentence.",
    "Write a Python function that returns the Fibonacci sequence.",
    "What causes lightning?",
    "Summarize the plot of Romeo and Juliet.",
]

SHARED_PREFIX = "You are a helpful assistant.\n\n"
SYSTEM_A = "You are a helpful assistant."
SYSTEM_B = "You are a pirate. Respond only in pirate dialect."


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def model_and_tokenizer():
    path = _model_path()
    print(f"\n[smoke] Loading model from {path}")
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    yield model, tokenizer
    del model
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Test 1: prefix_cache
# Expect: d >= CERT_THRESHOLD_D (same-env cache reuse → cos_sim ≈ 1.0)
# ---------------------------------------------------------------------------

def test_prefix_cache(model_and_tokenizer, tmp_path):
    model, tokenizer = model_and_tokenizer

    cos_sims = collect_prefix_cache(
        model, tokenizer, PROMPTS,
        shared_prefix=SHARED_PREFIX,
        device="cuda",
    )

    assert len(cos_sims) == len(PROMPTS), "One cos_sim per prompt expected"
    for i, c in enumerate(cos_sims):
        assert abs(c - 1.0) < 1e-4, (
            f"prefix_cache: prompt {i} cos_sim={c:.6f}, expected ≈ 1.0 — "
            "cache reuse changed logits, likely a hook or DynamicCache bug"
        )

    result = certify_layer(cos_sims)
    assert result["certified"], f"prefix_cache not certified: d={result['d_comm']}"
    assert result["d_comm"] == "inf" or result["d_comm"] >= CERT_THRESHOLD_D

    print(f"\n[smoke] prefix_cache PASSED — d={result['d_comm']}, bound={result['divergence_bound']}")


# ---------------------------------------------------------------------------
# Test 2: prompt_swap
# Expect: certified with finite d (system prompts differ → real divergence)
# ---------------------------------------------------------------------------

def test_prompt_swap(model_and_tokenizer, tmp_path):
    model, tokenizer = model_and_tokenizer

    cos_sims = collect_prompt_swap(
        model, tokenizer, PROMPTS,
        system_prompt_a=SYSTEM_A,
        system_prompt_b=SYSTEM_B,
        device="cuda",
    )

    assert len(cos_sims) == len(PROMPTS), "One cos_sim per prompt expected"

    for i, c in enumerate(cos_sims):
        assert math.isfinite(c), f"prompt_swap: prompt {i} cos_sim is not finite: {c}"
        assert -1.0 <= c <= 1.0 + 1e-6, f"prompt_swap: prompt {i} cos_sim={c} out of range"

    # Pirate vs helpful SHOULD diverge — if cos_sim ≈ 1.0 the chat template isn't applying
    mean_sim = sum(cos_sims) / len(cos_sims)
    assert mean_sim < 0.9999, (
        f"prompt_swap: mean cos_sim={mean_sim:.6f} suspiciously close to 1.0 — "
        "system prompts may not be applying correctly (check chat template)"
    )

    result = certify_layer(cos_sims)
    status = "CERTIFIED" if result["certified"] else "NOT CERTIFIED (expected for dramatic prompt swap)"
    print(
        f"\n[smoke] prompt_swap {status} — "
        f"d={result['d_comm']}, bound={result['divergence_bound']}, mean_cos_sim={mean_sim:.4f}"
    )

    # Smoke test: verify collector runs and produces valid schema — NOT that it certifies.
    # A pirate vs helpful system prompt on a 3B model produces real divergence (cos_sim ~0.4),
    # which correctly gives d < 3.0. That IS the collector working. Certification depends on
    # prompt pair and model — smoke tests prove plumbing, not a specific d value.
    for key in ("certified", "d_comm", "divergence_bound", "budget", "n_samples"):
        assert key in result, f"prompt_swap certify_layer result missing key: {key}"
    assert isinstance(result["certified"], bool)
    assert result["divergence_bound"] >= 0.0


# ---------------------------------------------------------------------------
# Test 3: engine_swap same-env
# Captures logits twice from same model — proves save_logits/load_logits round-trip
# Expect: d >= CERT_THRESHOLD_D (cuBLAS nondeterminism may give cos_sim slightly < 1.0
#         but still certified; d == "inf" only if bitwise identical)
# ---------------------------------------------------------------------------

def test_engine_swap_same_env(model_and_tokenizer, tmp_path):
    model, tokenizer = model_and_tokenizer

    capture_a = str(tmp_path / "capture_a.npz")
    capture_b = str(tmp_path / "capture_b.npz")

    logits_a = capture_logits_hf("", PROMPTS, model=model, tokenizer=tokenizer, device="cuda")
    save_logits(capture_a, logits_a, PROMPTS, engine_label="run_a", model_id="smoke-3b")

    logits_b = capture_logits_hf("", PROMPTS, model=model, tokenizer=tokenizer, device="cuda")
    save_logits(capture_b, logits_b, PROMPTS, engine_label="run_b", model_id="smoke-3b")

    cos_sims = collect_engine_swap(capture_a, capture_b)

    assert len(cos_sims) == len(PROMPTS), "One cos_sim per prompt expected"
    for i, c in enumerate(cos_sims):
        assert abs(c - 1.0) < 1e-4, (
            f"engine_swap same-env: prompt {i} cos_sim={c:.6f}, expected ≈ 1.0 — "
            "serialization or load corrupted logit vectors"
        )

    result = certify_layer(cos_sims)
    assert result["certified"], f"engine_swap same-env not certified: d={result['d_comm']}"
    assert result["d_comm"] == "inf" or result["d_comm"] >= CERT_THRESHOLD_D

    # Verify actual certify_layer schema keys
    for key in ("certified", "d_comm", "divergence_bound", "budget", "n_samples"):
        assert key in result, f"certify_layer result missing key: {key}"

    print(f"\n[smoke] engine_swap (same-env) PASSED — d={result['d_comm']}, schema keys verified")
