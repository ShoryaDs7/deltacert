"""
deltacert.collectors
====================

All 16 collectors for DeltaCert certification.

    ── Model-level compression (1-10) ──────────────────────────────────────
    1   collect_allreduce_tp        : compressed tensor-parallel AllReduce
    2   collect_alltoall_ep         : compressed expert-parallel All-to-All
    3   collect_pipeline_parallel   : compressed pipeline stage activations
    4   collect_kv_transfer         : compressed prefill→decode KV transfer
    5   collect_weight_quant        : int4/int8/fp8 weight quantization
    6   collect_kv_cache_quant      : KV cache quantization
    7   collect_activation_quant    : activation quantization
    8   collect_gradient_compress   : gradient compression (training)
    9   collect_lora                : LoRA adapter vs full model
    10  collect_prefix_cache        : prefix/prompt cache reuse

    ── System changes (11-16) ──────────────────────────────────────────────
    11  collect_engine_swap         : engine A vs engine B (vLLM 0.8 vs 0.9,
                                      HF vs vLLM, A100 vs H100)
    12  collect_batch_divergence    : batch=1 vs batch=N serving engine
    13  collect_speculative_decode  : speculative decoding ON vs OFF
    14  collect_sparse_attention    : full attention vs sparse-masked
    15  collect_moe_token_dropping  : MoE capacity factor A vs B
    16  collect_neuron_skipping     : full model vs runtime-pruned (SRED/SNAP)

    ── Change certification (17-19) ────────────────────────────────────────
    17  collect_model_swap          : checkpoint A vs checkpoint B (model
                                      update, fine-tune revision, re-quant)
    18  collect_provider_drift      : hosted API today vs saved baseline —
                                      top-k logprobs on canary prompts
    19  collect_prompt_swap         : system prompt v1 vs v2, same questions

    ── Long-context / trajectory (20) ──────────────────────────────────────
    20  collect_trajectory          : hook-style optimization over full
                                      continuations (d_min over positions)
        collect_trajectory_two_models : two-model trajectory variant

    ── Free-running / decode-dynamics (21) ─────────────────────────────────
    21  collect_free_running        : deployed-decode-policy comparison for
                                      feedback-driven failures invisible to
                                      every teacher-forced mode above (§5.5
                                      false-safe class) — measures the
                                      decoded output process itself, not
                                      logit conditionals

All 16 collectors return List[float] of cosine similarities feeding directly
into deltacert.d_comm() / deltacert.certify():

    Delta = 4 c sqrt(1 - c^2)     (commutator magnitude, Thm 4.2)
    d     = -log(Delta / 2)       (algebraic distance, Def 4.3)
    |divergence| <= 2 exp(-d)     (Corollary 6.2)

Same formula for all 16. The collector is just the mechanism for obtaining
the cos_sims. Once you have the list, d_comm() is identical everywhere.

Design rules (enforced, not advisory):
  * NO simulation, NO synthetic tensors. Every cos_sim from real forward
    passes of real models / real serving engines.
  * Hard failure on empty measurements (CollectionError — never an empty list).
  * Hook-based collectors MUST restore model state; restoration is verified,
    leaks hard-fail.
  * Deterministic capture: eval mode, inference_mode, greedy decode, seed=0.
  * Collector 11 runs as two separate processes (two engine versions cannot
    coexist in one venv): capture → .npz on disk → compare.

Dependencies: numpy always. torch + transformers for 1-10, 14-16.
vllm for 11-13. peft for 9. All imports are lazy — file is importable
on a CPU-only cert-verification box.

Copyright (c) 2026 Threvo Labs Private Limited.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import gc
import hashlib
import json
import logging
import math
import os
import platform
import socket
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass
from typing import Callable, ContextManager, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger("deltacert.collectors")

__all__ = [
    # errors
    "CollectionError",
    # math
    "d_comm",
    "divergence_bound",
    "cos_sim",
    "cos_sims_from_logit_matrices",
    # capture / persistence
    "capture_logits_hf",
    "capture_logits_vllm",
    "save_logits",
    "load_logits",
    # certificate builder
    "certify_from_layers",
    # collectors 1-10
    "collect_allreduce_tp",
    "collect_alltoall_ep",
    "collect_pipeline_parallel",
    "collect_pipeline_parallel_tensors",
    "collect_kv_transfer",
    "collect_weight_quant",
    "collect_kv_cache_quant",
    "collect_kv_cache_quant_vllm",
    "collect_activation_quant",
    "collect_gradient_compress",
    "collect_lora",
    "collect_prefix_cache",
    # collectors 11-16
    "collect_engine_swap",
    "collect_batch_divergence",
    "collect_speculative_decode",
    "collect_sparse_attention",
    "collect_moe_token_dropping",
    "collect_neuron_skipping",
    # collectors 17-19
    "collect_model_swap",
    "capture_logits_openai_api",
    "collect_provider_drift",
    "collect_prompt_swap",
    # collector 20
    "d_profile",
    "trajectory_layer_result",
    "collect_trajectory",
    "collect_trajectory_two_models",
    "collect_trajectory_vllm_two_engines",
    # collector 21
    "FreeRunPromptResult",
    "max_window_token_freq",
    "distinct_ngram_ratio",
    "is_degenerate",
    "fork_position",
    "surprisal_stats",
    "mcnemar_exact_p",
    "collect_free_running_vllm_two_engines",
    "collect_free_running_vllm",
    "certify_free_running",
]


# ──────────────────────────────────────────────────────────────────────────────
# Errors
# ──────────────────────────────────────────────────────────────────────────────

class CollectionError(RuntimeError):
    """Raised whenever a collector cannot produce trustworthy measurements.

    Callers must treat this as CERTIFICATION FAILURE, never as skippable.
    """


# ──────────────────────────────────────────────────────────────────────────────
# Core math — CANONICAL d_COMM (version 2)
#
# Single authoritative implementation for the entire package.
# deltacert.py imports from here.
#
# Fail-closed clamp: Delta(c) = 4c*sqrt(1-c^2) is non-monotone.
# It vanishes at c=1 (identical outputs, good) AND at c=0 (orthogonal,
# broken). Without the clamp, a catastrophically broken pipeline producing
# near-orthogonal logits yields Delta~0 → d=inf → CERTIFIED (wrong).
# Monotone regime: c >= 1/sqrt(2). Below it, clamp Delta=2 → d=0 → NOT
# CERTIFIED. Real production measurements live in [0.95, 1.0] where old
# and new are numerically identical.
# ──────────────────────────────────────────────────────────────────────────────

_CANONICAL_MATH_VERSION = 2
_C_MIN_VALID = 1.0 / math.sqrt(2.0)
CERT_THRESHOLD_D = 3.0

# Collectors with a real flagship validation run behind them (real hardware,
# real downstream benchmark, published in the paper). All other collectors
# in VALID_COLLECTORS (validation/harness.py) are implemented and tested but
# have not been run through that end-to-end validation process yet.
#
# This is the single source of truth for validation_status stamping — do not
# duplicate this list elsewhere. Update it only when a new flagship result is
# actually published, not preemptively.
FLAGSHIP_VALIDATED_COLLECTORS = frozenset({
    "weight_quant", "kv_cache_quant", "engine_swap", "batch_divergence",
    "spec_decoding", "provider_drift", "trajectory",
})


def validation_status_for_layers(layer_names) -> str:
    """
    'flagship_validated' only if every layer in this cert comes from a
    collector with a real flagship validation run. A single unvalidated
    layer downgrades the whole cert to 'implemented_pending_validation' —
    fail toward the less-trusted label, never the more-trusted one.
    """
    names = list(layer_names)
    if not names:
        return "implemented_pending_validation"
    if all(n in FLAGSHIP_VALIDATED_COLLECTORS for n in names):
        return "flagship_validated"
    return "implemented_pending_validation"


def _commutator_magnitude(cos_similarity: float) -> float:
    """Delta = 4c*sqrt(1-c^2) on c >= 1/sqrt(2); fail-closed below.

    Delta(c) = ||[U,V]|| for the logit-vector reflections U=2uu*-I,
    V=2vv*-I about the normalized logit vectors u, v with <u,v>=c; see
    paper §3.1 / Lemma 4.5 of the math paper. Made executable in
    tests/test_math_identity.py::test_delta_formula_is_commutator_distance.
    """
    c = max(-1.0, min(1.0, float(cos_similarity)))
    if c < _C_MIN_VALID:
        return 2.0
    return 4.0 * c * math.sqrt(max(0.0, 1.0 - c * c))


def _d_from_delta(delta: float) -> float:
    """d = -log(Delta/2); Delta==0 maps to +inf (perfect commutation)."""
    if delta <= 0.0:
        return float("inf")
    return -math.log(delta / 2.0)


def d_comm(cos_sims: Sequence[float]) -> float:
    """Certificate distance from a list of per-prompt cosine similarities.

    Raises CollectionError on empty input — zero measurements must never certify.
    """
    if not cos_sims:
        raise CollectionError(
            "d_comm called with an empty cos_sims list. Refusing to certify "
            "on zero measurements."
        )
    lam = sum(_commutator_magnitude(c) for c in cos_sims) / len(cos_sims)
    return _d_from_delta(lam)


def divergence_bound(d: float) -> float:
    """2*exp(-d)  [Corollary 6.2]"""
    return 0.0 if d == float("inf") else 2.0 * math.exp(-d)


# ──────────────────────────────────────────────────────────────────────────────
# Cosine similarity
# ──────────────────────────────────────────────────────────────────────────────

def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D float64 vectors, with hard checks."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise CollectionError(
            f"Logit vector shape mismatch: {a.shape} vs {b.shape}. "
            "Configurations are not comparable (different vocab or capture bug)."
        )
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise CollectionError(
            "Non-finite values in captured logits (NaN/Inf). "
            "Unstable kernel or dtype overflow. Refusing to certify."
        )
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        raise CollectionError("Zero-norm logit vector captured. Refusing to certify.")
    return float(np.dot(a, b) / (na * nb))


def cos_sims_from_logit_matrices(
    logits_a: np.ndarray, logits_b: np.ndarray
) -> List[float]:
    """Row-wise cosine similarity between two [n_prompts, vocab] matrices."""
    logits_a = np.atleast_2d(np.asarray(logits_a))
    logits_b = np.atleast_2d(np.asarray(logits_b))
    if logits_a.shape[0] != logits_b.shape[0]:
        raise CollectionError(
            f"Prompt count mismatch: {logits_a.shape[0]} vs {logits_b.shape[0]}. "
            "Both captures must use the identical prompt list in the same order."
        )
    if logits_a.shape[0] == 0:
        raise CollectionError("Zero prompts captured. Refusing to certify.")
    return [cos_sim(logits_a[i], logits_b[i]) for i in range(logits_a.shape[0])]


# ──────────────────────────────────────────────────────────────────────────────
# Certificate builder — one canonical schema for the whole package
# ──────────────────────────────────────────────────────────────────────────────

def certify_from_layers(
    model: str,
    layers: dict,
    threshold: float = CERT_THRESHOLD_D,
    description: str = "",
) -> dict:
    """Build a certificate dict in the canonical deltacert schema.

    `layers` is {layer_name: layer_result_dict} where each dict already has
    d_comm, divergence_bound, certified. Used by the CLI compare command and
    importable by deltacert.py.

    Schema is a superset shared with deltacert.py::certify() — both paths
    emit {model, description, certified, threshold_d, formula, theorem,
    validation_status, layers, metadata} so any consumer (signing, compliance
    parsing) can rely on one shape regardless of which internal path produced
    the cert.
    """
    all_certified = all(r.get("certified", False) for r in layers.values())
    return {
        "model": model,
        "description": description,
        "certified": all_certified,
        "threshold_d": threshold,
        "formula": "d_COMM = -log(E[4c*sqrt(1-c^2)] / 2), certified if d >= threshold",
        "theorem": "Proposition 5.1, Shorya 2026",
        "validation_status": validation_status_for_layers(layers.keys()),
        "layers": layers,
        "metadata": _environment_metadata(),
    }


def _make_layer_result(
    cos_sims: Sequence[float],
    threshold: float = CERT_THRESHOLD_D,
    extra: Optional[dict] = None,
) -> dict:
    d = d_comm(cos_sims)
    result = {
        "n_prompts": len(cos_sims),
        "cos_sim_min": float(min(cos_sims)),
        "cos_sim_mean": float(sum(cos_sims) / len(cos_sims)),
        "d_comm": d,
        "divergence_bound": divergence_bound(d),
        "certified": bool(d >= threshold),
    }
    if extra:
        result.update(extra)
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Metadata + .npz persistence (two-process path for collector 11)
# ──────────────────────────────────────────────────────────────────────────────

def _prompts_hash(prompts: Sequence[str]) -> str:
    h = hashlib.sha256()
    for p in prompts:
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def _environment_metadata() -> dict:
    meta = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    try:
        import torch
        meta["torch"] = torch.__version__
        if torch.cuda.is_available():
            meta["gpu"] = torch.cuda.get_device_name(0)
            meta["gpu_memory_bytes"] = int(torch.cuda.get_device_properties(0).total_memory)
            meta["cuda"] = torch.version.cuda or "unknown"
    except Exception:
        pass
    try:
        import vllm
        meta["vllm"] = vllm.__version__
    except Exception:
        pass
    try:
        import transformers
        meta["transformers"] = transformers.__version__
    except Exception:
        pass
    return meta


def save_logits(
    path: str,
    logits: np.ndarray,
    prompts: Sequence[str],
    engine_label: str,
    model_id: str,
    extra_metadata: Optional[dict] = None,
) -> str:
    """Persist a capture to .npz so a different process/venv can compare it."""
    logits = np.atleast_2d(np.asarray(logits, dtype=np.float32))
    if logits.shape[0] != len(prompts):
        raise CollectionError(
            f"save_logits: {logits.shape[0]} logit rows for {len(prompts)} prompts."
        )
    meta = _environment_metadata()
    meta.update({
        "engine_label": engine_label,
        "model_id": model_id,
        "n_prompts": len(prompts),
        "vocab_size": int(logits.shape[1]),
        "prompts_sha256": _prompts_hash(prompts),
    })
    if extra_metadata:
        meta.update(extra_metadata)
    np.savez_compressed(path, logits=logits, metadata=json.dumps(meta))
    logger.info("Saved capture: %s (%d prompts, vocab=%d, engine=%s)",
                path, len(prompts), logits.shape[1], engine_label)
    return path


def load_logits(path: str) -> Tuple[np.ndarray, dict]:
    """Load a capture written by save_logits. Returns (logits, metadata_dict)."""
    if not os.path.exists(path):
        raise CollectionError(f"Capture file not found: {path}")
    data = np.load(path, allow_pickle=False)
    if "logits" not in data or "metadata" not in data:
        raise CollectionError(
            f"{path} is not a deltacert capture (missing 'logits'/'metadata')."
        )
    return data["logits"], json.loads(str(data["metadata"]))


# ──────────────────────────────────────────────────────────────────────────────
# Capture backends
# ──────────────────────────────────────────────────────────────────────────────

def capture_logits_hf(
    model_name_or_path: str,
    prompts: Sequence[str],
    device: str = "cuda",
    dtype: str = "float16",
    max_length: int = 4096,
    trust_remote_code: bool = False,
    model=None,
    tokenizer=None,
) -> np.ndarray:
    """Last-token logits per prompt via HuggingFace transformers.

    Pass a preloaded (model, tokenizer) to avoid reloading between calls.
    Returns float32 [n_prompts, vocab].
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not prompts:
        raise CollectionError("capture_logits_hf: empty prompt list.")

    owns_model = model is None
    if owns_model:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=trust_remote_code
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=getattr(torch, dtype),
            device_map=device,
            trust_remote_code=trust_remote_code,
        )
    if tokenizer is None:
        raise CollectionError("capture_logits_hf: model given without tokenizer.")

    model.eval()
    rows: List[np.ndarray] = []
    try:
        with torch.inference_mode():
            for prompt in prompts:
                inputs = tokenizer(
                    prompt, return_tensors="pt",
                    truncation=True, max_length=max_length,
                ).to(model.device)
                logits = model(**inputs).logits[:, -1, :].float().squeeze(0)
                rows.append(logits.cpu().numpy().astype(np.float32))
    finally:
        if owns_model:
            del model
            gc.collect()
            try:
                import torch as _t
                if _t.cuda.is_available():
                    _t.cuda.empty_cache()
            except Exception:
                pass

    if len(rows) != len(prompts):
        raise CollectionError(
            f"capture_logits_hf: captured {len(rows)} rows for {len(prompts)} prompts."
        )
    return np.stack(rows, axis=0)


def _vllm_greedy_params(vllm_module, num_logprobs: int):
    return vllm_module.SamplingParams(
        temperature=0.0, max_tokens=1, logprobs=num_logprobs, seed=0,
    )


def _logprob_dict_to_dense(logprob_entry: dict, vocab_size: int) -> np.ndarray:
    """vLLM top-k {token_id: Logprob} → dense vocab-sized vector (fill=-50)."""
    vec = np.full(vocab_size, -50.0, dtype=np.float32)
    for token_id, lp in logprob_entry.items():
        vec[int(token_id)] = float(lp.logprob if hasattr(lp, "logprob") else lp)
    return vec


def capture_logits_vllm(
    model_name_or_path: str,
    prompts: Sequence[str],
    tensor_parallel_size: int = 1,
    num_logprobs: int = 20,
    gpu_memory_utilization: float = 0.90,
    max_model_len: Optional[int] = None,
    engine_kwargs: Optional[dict] = None,
    llm=None,
) -> np.ndarray:
    """Next-token log-distribution per prompt via an in-process vLLM engine.

    vLLM exposes top-k logprobs, not raw full-vocab logits. We capture
    top-`num_logprobs` (128 carries essentially all probability mass) embedded
    in a dense vocab-sized vector. Both sides of every comparison use the same
    num_logprobs so the certified quantity is well-defined.

    Pass a preloaded `llm` to reuse the engine (collector 12 does this).
    """
    import vllm

    if not prompts:
        raise CollectionError("capture_logits_vllm: empty prompt list.")

    owns_engine = llm is None
    if owns_engine:
        kwargs = dict(
            model=model_name_or_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=False,
        )
        if max_model_len is not None:
            kwargs["max_model_len"] = max_model_len
        if engine_kwargs:
            kwargs.update(engine_kwargs)
        llm = vllm.LLM(**kwargs)

    try:
        vocab_size = llm.llm_engine.model_config.get_vocab_size()
    except Exception:
        vocab_size = llm.get_tokenizer().vocab_size

    params = _vllm_greedy_params(vllm, num_logprobs)
    rows: List[np.ndarray] = []
    try:
        for out in llm.generate(list(prompts), params):
            if not out.outputs or out.outputs[0].logprobs is None:
                raise CollectionError(
                    f"vLLM returned no logprobs (request_id={out.request_id})."
                )
            rows.append(_logprob_dict_to_dense(out.outputs[0].logprobs[0], vocab_size))
    finally:
        if owns_engine:
            del llm
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    if len(rows) != len(prompts):
        raise CollectionError(
            f"capture_logits_vllm: {len(rows)} rows for {len(prompts)} prompts."
        )
    return np.stack(rows, axis=0)


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers shared by collectors 1-10
# ──────────────────────────────────────────────────────────────────────────────

def _require_model_tokenizer(model, tokenizer) -> None:
    if model is None:
        raise CollectionError("model is required but was not provided.")
    if tokenizer is None:
        raise CollectionError("tokenizer is required but was not provided.")


def _require_prompts(prompts: Sequence[str]) -> None:
    if not prompts:
        raise CollectionError("Empty prompt list. Refusing to certify on zero measurements.")


def _last_token_logits_np(
    model, tokenizer, prompts: Sequence[str], device: str, max_length: int = 4096
) -> np.ndarray:
    """Full forward pass → last-token logits [n_prompts, vocab] float32."""
    import torch

    _require_prompts(prompts)
    model.eval()
    rows: List[np.ndarray] = []
    with torch.inference_mode():
        for p in prompts:
            inputs = tokenizer(
                p, return_tensors="pt", truncation=True, max_length=max_length
            ).to(device)
            row = model(**inputs).logits[:, -1, :].float().squeeze(0).cpu().numpy().astype(np.float32)
            if not np.all(np.isfinite(row)):
                raise CollectionError(
                    f"Non-finite logits for prompt '{p[:60]}'. "
                    "Unstable model or dtype overflow. Refusing to certify."
                )
            rows.append(row)
    if len(rows) != len(prompts):
        raise CollectionError(f"Captured {len(rows)} rows for {len(prompts)} prompts.")
    return np.stack(rows, axis=0)


def _compare_logit_matrices(mat_a: np.ndarray, mat_b: np.ndarray) -> List[float]:
    if mat_a.shape != mat_b.shape:
        raise CollectionError(
            f"Logit matrix shape mismatch: {mat_a.shape} vs {mat_b.shape}."
        )
    return cos_sims_from_logit_matrices(mat_a, mat_b)


def _wait_for_gpu_memory(min_free_gb: float = 70.0, timeout_s: float = 90.0,
                        poll_interval_s: float = 2.0) -> None:
    """Block until at least min_free_gb of GPU memory is actually free, or
    timeout — self-contained (no imports outside this module), used only by
    collectors that create two SEPARATE sequential vLLM engines in one
    process (e.g. collect_speculative_decode). A Python-side `del llm` does
    not guarantee vLLM's background EngineCore process has finished exiting
    and released its CUDA memory before the next engine tries to start."""
    import subprocess
    import time as _time
    deadline = _time.time() + timeout_s
    while _time.time() < deadline:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=True, timeout=10,
            )
            free_mb = float(out.stdout.strip().splitlines()[0])
            if free_mb / 1024.0 >= min_free_gb:
                return
        except Exception:
            pass
        _time.sleep(poll_interval_s)
    logger.warning(
        "_wait_for_gpu_memory: %sGB not free after %ss — proceeding anyway, "
        "next engine start may fail.", min_free_gb, timeout_s,
    )


def _verify_hook_restoration(
    model, tokenizer, reference_row: np.ndarray, prompt: str, device: str
) -> None:
    """Re-run one prompt clean after hook removal; demand bitwise-identical logits."""
    import torch

    model.eval()
    with torch.inference_mode():
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=4096
        ).to(device)
        recheck = model(**inputs).logits[:, -1, :].float().squeeze(0).cpu().numpy().astype(np.float32)
    if not np.allclose(reference_row, recheck, rtol=0.0, atol=0.0):
        raise CollectionError(
            "Model state was NOT restored after hook-based modified pass. "
            "Hook is still active or weight was modified. Refusing to certify."
        )


def _find_transformer_layers(model):
    """Return the model's transformer layer list or raise CollectionError."""
    for attr in ("layers", "model.layers", "transformer.h", "model.decoder.layers"):
        obj = model
        for part in attr.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None and hasattr(obj, "__len__"):
            return obj
    raise CollectionError(
        "Cannot find transformer layer list. "
        "Tried: model.layers, model.model.layers, transformer.h, model.decoder.layers."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Collector 1 — ALLREDUCE TP
# ──────────────────────────────────────────────────────────────────────────────

def collect_allreduce_tp(
    model,
    tokenizer,
    prompts: Sequence[str],
    compress_fn: Callable,
    decompress_fn: Callable,
    device: str = "cuda",
) -> List[float]:
    """Certify compressed tensor-parallel AllReduce.

    Hooks o_proj and down_proj — the actual AllReduce boundary tensors in
    tensor-parallel LLMs. Two passes per prompt: clean baseline, then
    compress→decompress on boundary outputs. Measures end-to-end logit
    cosine similarity. Hook restoration is verified; leaks hard-fail.

    Example:
        def compress(t): return t.to(torch.float8_e4m3fn)
        def decompress(t): return t.to(torch.float16)
        cos_sims = collect_allreduce_tp(model, tok, prompts, compress, decompress)
    """
    _require_model_tokenizer(model, tokenizer)
    _require_prompts(prompts)

    boundary_modules = {
        name: mod for name, mod in model.named_modules()
        if any(name.endswith(b) for b in ("o_proj", "down_proj"))
    }
    if not boundary_modules:
        raise CollectionError(
            "No o_proj or down_proj modules found. "
            "Cannot identify AllReduce boundary tensors."
        )

    baseline = _last_token_logits_np(model, tokenizer, prompts, device)
    handles = []
    for _, mod in boundary_modules.items():
        def _hook(m, inp, out, _c=compress_fn, _d=decompress_fn):
            try:
                return _d(_c(out))
            except Exception as e:
                raise CollectionError(f"AllReduce compress/decompress failed: {e}") from e
        handles.append(mod.register_forward_hook(_hook))
    try:
        modified = _last_token_logits_np(model, tokenizer, prompts, device)
    finally:
        for h in handles:
            h.remove()
    _verify_hook_restoration(model, tokenizer, baseline[0], prompts[0], device)
    return _compare_logit_matrices(baseline, modified)


# ──────────────────────────────────────────────────────────────────────────────
# Collector 2 — ALLTOALL EP
# ──────────────────────────────────────────────────────────────────────────────

def collect_alltoall_ep(
    model,
    tokenizer,
    prompts: Sequence[str],
    compress_fn: Callable,
    decompress_fn: Callable,
    device: str = "cuda",
) -> List[float]:
    """Certify compressed expert-parallel All-to-All.

    Hooks MoE router/dispatch modules. Two passes per prompt: clean baseline,
    then compress→decompress on dispatch tensors.
    Requires a real MoE model (Mixtral, DeepSeek-MoE, Qwen-MoE, etc.).
    Hard-fails on dense models — gate_proj in Llama/Mistral MLP is NOT a
    dispatch boundary and must never be hooked here.
    """
    import torch

    _require_model_tokenizer(model, tokenizer)
    _require_prompts(prompts)

    config = getattr(model, "config", None)
    n_experts = (
        getattr(config, "num_local_experts", None)
        or getattr(config, "n_routed_experts", None)
        or getattr(config, "num_experts", None)
    )
    if not n_experts:
        raise CollectionError(
            "Model config has no MoE expert count (num_local_experts / "
            "n_routed_experts / num_experts). collect_alltoall_ep requires a "
            "real MoE model. Dense models (Llama, Mistral, Qwen-dense) will "
            "match gate_proj and produce a meaningless cert — hard-failing."
        )

    # Exclude dense MLP projections — their names contain "gate"/"expert" too.
    _DENSE_PROJ_SUFFIXES = (
        "gate_proj", "up_proj", "down_proj",
        "q_proj", "k_proj", "v_proj", "o_proj",
    )
    dispatch_modules = {
        name: mod for name, mod in model.named_modules()
        if (
            any(s in name.lower() for s in ("router", "gate", "dispatch"))
            and not any(name.endswith(s) for s in _DENSE_PROJ_SUFFIXES)
            and hasattr(mod, "weight")
        )
    }
    if not dispatch_modules:
        raise CollectionError(
            "No MoE router/gate/dispatch modules found after excluding dense "
            "MLP projections. Check model architecture."
        )

    baseline = _last_token_logits_np(model, tokenizer, prompts, device)
    handles = []
    for _, mod in dispatch_modules.items():
        def _hook(m, inp, out, _c=compress_fn, _d=decompress_fn):
            if isinstance(out, torch.Tensor):
                try:
                    return _d(_c(out))
                except Exception as e:
                    raise CollectionError(f"All-to-All compress/decompress failed: {e}") from e
            return out
        handles.append(mod.register_forward_hook(_hook))
    try:
        modified = _last_token_logits_np(model, tokenizer, prompts, device)
    finally:
        for h in handles:
            h.remove()
    _verify_hook_restoration(model, tokenizer, baseline[0], prompts[0], device)
    return _compare_logit_matrices(baseline, modified)


# ──────────────────────────────────────────────────────────────────────────────
# Collector 3 — PIPELINE PARALLEL
# ──────────────────────────────────────────────────────────────────────────────

def collect_pipeline_parallel(
    model,
    tokenizer,
    prompts: Sequence[str],
    compress_fn: Callable,
    decompress_fn: Callable,
    stage_boundary_layer_idx: int,
    device: str = "cuda",
) -> List[float]:
    """Certify compressed pipeline stage boundary activations.

    Hooks the transformer layer at `stage_boundary_layer_idx` (0-indexed) —
    the exact layer whose output crosses the wire between pipeline stages.
    For a 32-layer model on 4 stages, boundaries are at layers 8, 16, 24.
    """
    _require_model_tokenizer(model, tokenizer)
    _require_prompts(prompts)

    layers = _find_transformer_layers(model)
    if stage_boundary_layer_idx >= len(layers):
        raise CollectionError(
            f"stage_boundary_layer_idx={stage_boundary_layer_idx} out of range "
            f"(model has {len(layers)} layers)."
        )

    boundary = layers[stage_boundary_layer_idx]
    baseline = _last_token_logits_np(model, tokenizer, prompts, device)

    def _hook(m, inp, out):
        act = out[0] if isinstance(out, tuple) else out
        try:
            restored = decompress_fn(compress_fn(act))
        except Exception as e:
            raise CollectionError(f"Pipeline compress/decompress failed: {e}") from e
        return (restored,) + out[1:] if isinstance(out, tuple) else restored

    handle = boundary.register_forward_hook(_hook)
    try:
        modified = _last_token_logits_np(model, tokenizer, prompts, device)
    finally:
        handle.remove()
    _verify_hook_restoration(model, tokenizer, baseline[0], prompts[0], device)
    return _compare_logit_matrices(baseline, modified)


def collect_pipeline_parallel_tensors(
    tensors_clean: np.ndarray,
    tensors_compressed: np.ndarray,
) -> List[float]:
    """Collector 3 variant for pre-captured Megatron-LM stage boundary tensors.

    Both arrays must be [n_prompts, hidden_dim] float32.
    """
    return cos_sims_from_logit_matrices(
        np.atleast_2d(tensors_clean),
        np.atleast_2d(tensors_compressed),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Collector 4 — KV TRANSFER
# ──────────────────────────────────────────────────────────────────────────────

def collect_kv_transfer(
    model,
    tokenizer,
    prompts: Sequence[str],
    compress_fn: Callable,
    decompress_fn: Callable,
    device: str = "cuda",
) -> List[float]:
    """Certify compressed prefill→decode KV cache transfer.

    Per prompt: prefill → capture real past_key_values → compress/decompress
    each K and V → decode with clean KVs vs decode with compressed KVs.
    Measures next-token logit cosine similarity.

    compress_fn MUST NOT mutate its input tensor in place (no t.mul_(), no
    t.copy_()). The same K/V tensors from `legacy` are reused for the clean
    decode pass — an in-place compress_fn corrupts them before that pass runs.
    """
    import torch

    _require_model_tokenizer(model, tokenizer)
    _require_prompts(prompts)

    sims: List[float] = []
    model.eval()
    with torch.inference_mode():
        for prompt in prompts:
            inputs = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=2048
            ).to(device)
            out_prefill = model(**inputs, use_cache=True)
            raw_kvs = out_prefill.past_key_values
            if raw_kvs is None:
                raise CollectionError(
                    f"Model returned no past_key_values for '{prompt[:60]}'. "
                    "Model must support use_cache=True."
                )

            # DynamicCache (transformers >=4.36) — deepcopy directly, no legacy conversion.
            # to_legacy_cache/from_legacy_cache were removed in transformers 5.x.
            kvs = raw_kvs

            next_tok = out_prefill.logits[:, -1, :].argmax(dim=-1, keepdim=True)

            # clean decode — deepcopy so this forward pass cannot mutate the original cache
            kvs_clean = copy.deepcopy(kvs)
            logits_clean = (
                model(input_ids=next_tok, past_key_values=kvs_clean, use_cache=False)
                .logits[:, -1, :].float().squeeze(0).cpu().numpy()
            )

            # compressed decode — compress each (K, V) layer
            try:
                from transformers import DynamicCache as _DC
                _is_dynamic = isinstance(kvs, _DC)
            except ImportError:
                _is_dynamic = False

            if _is_dynamic:
                kvs_comp = copy.deepcopy(kvs)
                for layer_idx in range(len(kvs_comp.key_cache)):
                    try:
                        kvs_comp.key_cache[layer_idx] = decompress_fn(compress_fn(kvs_comp.key_cache[layer_idx]))
                        kvs_comp.value_cache[layer_idx] = decompress_fn(compress_fn(kvs_comp.value_cache[layer_idx]))
                    except Exception as e:
                        raise CollectionError(f"KV transfer compress/decompress failed: {e}") from e
            else:
                comp_layers = []
                for k, v in kvs:
                    try:
                        comp_layers.append((decompress_fn(compress_fn(k)), decompress_fn(compress_fn(v))))
                    except Exception as e:
                        raise CollectionError(f"KV transfer compress/decompress failed: {e}") from e
                kvs_comp = tuple(comp_layers)
            logits_comp = (
                model(input_ids=next_tok, past_key_values=kvs_comp, use_cache=False)
                .logits[:, -1, :].float().squeeze(0).cpu().numpy()
            )

            for arr in (logits_clean, logits_comp):
                if not np.all(np.isfinite(arr)):
                    raise CollectionError("Non-finite logits in KV transfer collector.")
            sims.append(cos_sim(logits_clean, logits_comp))

    if not sims:
        raise CollectionError("collect_kv_transfer produced no measurements.")
    return sims


# ──────────────────────────────────────────────────────────────────────────────
# Collector 5 — WEIGHT QUANTIZATION
# ──────────────────────────────────────────────────────────────────────────────

def collect_weight_quant(
    model_fp16,
    model_quantized,
    tokenizer,
    prompts: Sequence[str],
    device: str = "cuda",
) -> List[float]:
    """Certify int4/int8/fp8 weight quantization.

    Two real model forward passes (fp16 vs quantized). No hooks, no simulation.

        m_fp16 = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16)
        m_int8 = AutoModelForCausalLM.from_pretrained(name, load_in_8bit=True)
        cos_sims = collect_weight_quant(m_fp16, m_int8, tok, prompts)
    """
    _require_model_tokenizer(model_fp16, tokenizer)
    _require_model_tokenizer(model_quantized, tokenizer)
    _require_prompts(prompts)
    return _compare_logit_matrices(
        _last_token_logits_np(model_fp16, tokenizer, prompts, device),
        _last_token_logits_np(model_quantized, tokenizer, prompts, device),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Collector 6 — KV CACHE QUANTIZATION
# ──────────────────────────────────────────────────────────────────────────────

def collect_kv_cache_quant(
    model,
    tokenizer,
    prompts: Sequence[str],
    compress_fn: Callable,
    decompress_fn: Callable,
    device: str = "cuda",
) -> List[float]:
    """Certify KV cache quantization.

    Hooks k_proj and v_proj outputs — the actual K/V tensors before attention.
    Two passes: clean baseline, then compress→decompress on K/V projections.

    Example (int8):
        def compress(t):
            scale = t.abs().max() / 127.0 + 1e-8
            return (t / scale).round().clamp(-127, 127).to(torch.int8), scale
        def decompress(packed):
            q, scale = packed
            return q.to(torch.float16) * scale
    """
    _require_model_tokenizer(model, tokenizer)
    _require_prompts(prompts)

    kv_modules = {
        name: mod for name, mod in model.named_modules()
        if any(name.endswith(k) for k in ("k_proj", "v_proj"))
    }
    if not kv_modules:
        raise CollectionError("No k_proj or v_proj modules found.")

    baseline = _last_token_logits_np(model, tokenizer, prompts, device)
    handles = []
    for _, mod in kv_modules.items():
        def _hook(m, inp, out, _c=compress_fn, _d=decompress_fn):
            try:
                return _d(_c(out))
            except Exception as e:
                raise CollectionError(f"KV cache quant compress/decompress failed: {e}") from e
        handles.append(mod.register_forward_hook(_hook))
    try:
        modified = _last_token_logits_np(model, tokenizer, prompts, device)
    finally:
        for h in handles:
            h.remove()
    _verify_hook_restoration(model, tokenizer, baseline[0], prompts[0], device)
    return _compare_logit_matrices(baseline, modified)


def collect_kv_cache_quant_vllm(
    model_name_or_path: str,
    prompts: Sequence[str],
    kv_cache_dtype: str = "fp8",
    tensor_parallel_size: int = 1,
    num_logprobs: int = 20,
    engine_kwargs: Optional[dict] = None,
) -> List[float]:
    """Certify KV cache quantization via vLLM's REAL native
    --kv-cache-dtype engine flag — the actual production serving mechanism,
    not a hand-rolled compress/decompress hook. Two sequential in-process
    vLLM engines (default KV dtype vs the quantized dtype), same canaries.

    This is a companion to collect_kv_cache_quant (the HF-hook version,
    still valid for non-vLLM deployments) — pick whichever matches how the
    company actually serves.
    """
    _require_prompts(prompts)
    base_kwargs = dict(engine_kwargs or {})

    logits_default = capture_logits_vllm(
        model_name_or_path, prompts, tensor_parallel_size=tensor_parallel_size,
        num_logprobs=num_logprobs,
        engine_kwargs={**base_kwargs, "kv_cache_dtype": "auto"},
    )
    logits_quant = capture_logits_vllm(
        model_name_or_path, prompts, tensor_parallel_size=tensor_parallel_size,
        num_logprobs=num_logprobs,
        engine_kwargs={**base_kwargs, "kv_cache_dtype": kv_cache_dtype},
    )
    return cos_sims_from_logit_matrices(logits_default, logits_quant)


# ──────────────────────────────────────────────────────────────────────────────
# Collector 7 — ACTIVATION QUANTIZATION
# ──────────────────────────────────────────────────────────────────────────────

def collect_activation_quant(
    model,
    tokenizer,
    prompts: Sequence[str],
    quant_fn: Callable,
    dequant_fn: Callable,
    device: str = "cuda",
) -> List[float]:
    """Certify activation quantization.

    Hooks transformer layer outputs (residual stream after each block). Two
    passes: clean baseline, then quant→dequant on every layer output.
    """
    _require_model_tokenizer(model, tokenizer)
    _require_prompts(prompts)

    layers = _find_transformer_layers(model)
    baseline = _last_token_logits_np(model, tokenizer, prompts, device)
    handles = []
    for layer in layers:
        def _hook(m, inp, out, _q=quant_fn, _d=dequant_fn):
            act = out[0] if isinstance(out, tuple) else out
            try:
                restored = _d(_q(act))
            except Exception as e:
                raise CollectionError(f"Activation quant failed: {e}") from e
            return (restored,) + out[1:] if isinstance(out, tuple) else restored
        handles.append(layer.register_forward_hook(_hook))
    try:
        modified = _last_token_logits_np(model, tokenizer, prompts, device)
    finally:
        for h in handles:
            h.remove()
    _verify_hook_restoration(model, tokenizer, baseline[0], prompts[0], device)
    return _compare_logit_matrices(baseline, modified)


# ──────────────────────────────────────────────────────────────────────────────
# Collector 8 — GRADIENT COMPRESSION
# ──────────────────────────────────────────────────────────────────────────────

def collect_gradient_compress(
    model,
    tokenizer,
    prompts: Sequence[str],
    compress_fn: Callable,
    decompress_fn: Callable,
    device: str = "cuda",
) -> List[float]:
    """Certify gradient compression fidelity (training use case).

    Real backward passes, captures actual .grad tensors, applies
    compress→decompress, measures per-parameter cosine similarity.
    Returns one cos_sim per prompt (mean over parameters).
    """
    import torch
    import torch.nn as nn

    _require_model_tokenizer(model, tokenizer)
    _require_prompts(prompts)

    loss_fn = nn.CrossEntropyLoss()
    sims: List[float] = []
    # eval() disables dropout so gradients are deterministic — train() mode
    # injects random noise that contaminates gradient compression measurements.
    model.eval()
    try:
        for prompt in prompts:
            inputs = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=512
            ).to(device)
            ids = inputs["input_ids"]
            if ids.shape[1] < 2:
                continue
            model.zero_grad()
            with torch.enable_grad():
                out = model(**inputs)
                loss = loss_fn(
                    out.logits[:, :-1, :].contiguous().view(-1, out.logits.size(-1)),
                    ids[:, 1:].contiguous().view(-1),
                )
                loss.backward()

            prompt_sims: List[float] = []
            for name, param in model.named_parameters():
                if param.grad is None:
                    continue
                grad = param.grad.detach().float()
                try:
                    grad_comp = decompress_fn(compress_fn(grad))
                except Exception as e:
                    raise CollectionError(
                        f"Gradient compress/decompress failed on '{name}': {e}"
                    ) from e
                prompt_sims.append(cos_sim(
                    grad.cpu().numpy().ravel().astype(np.float64),
                    grad_comp.cpu().numpy().ravel().astype(np.float64),
                ))
            if not prompt_sims:
                raise CollectionError(
                    f"No gradients for '{prompt[:60]}'. Ensure parameters require grad."
                )
            sims.append(float(np.mean(prompt_sims)))
    finally:
        model.eval()
        model.zero_grad()

    if not sims:
        raise CollectionError("collect_gradient_compress produced no measurements.")
    return sims


# ──────────────────────────────────────────────────────────────────────────────
# Collector 9 — LoRA
# ──────────────────────────────────────────────────────────────────────────────

def collect_lora(
    model_full,
    tokenizer,
    prompts: Sequence[str],
    lora_adapter_path: str,
    device: str = "cuda",
) -> List[float]:
    """Certify LoRA adapter divergence from the base model.

    Two real forward passes: base model vs LoRA-adapted model.
    `lora_adapter_path` is the directory containing adapter_config.json.
    Requires: pip install peft
    """
    _require_model_tokenizer(model_full, tokenizer)
    _require_prompts(prompts)
    try:
        from peft import PeftModel
    except ImportError:
        raise CollectionError("peft required: pip install peft")

    # Reference row captured BEFORE adapter injection — used to verify restoration.
    ref_row = _last_token_logits_np(model_full, tokenizer, [prompts[0]], device)[0]
    mat_full = _last_token_logits_np(model_full, tokenizer, prompts, device)

    try:
        model_lora = PeftModel.from_pretrained(model_full, lora_adapter_path)
    except Exception as e:
        raise CollectionError(f"Failed to load LoRA adapter from '{lora_adapter_path}': {e}") from e

    model_lora.eval()
    try:
        mat_lora = _last_token_logits_np(model_lora, tokenizer, prompts, device)
    finally:
        try:
            model_lora.unload()
        except Exception as e:
            raise CollectionError(
                f"LoRA unload() failed - base model_full is now contaminated "
                f"with adapter weights. Refusing to certify: {e}"
            ) from e

    # Verify base model is fully restored — same standard as every hook collector.
    post_row = _last_token_logits_np(model_full, tokenizer, [prompts[0]], device)[0]
    if not np.allclose(ref_row, post_row, rtol=0.0, atol=0.0):
        raise CollectionError(
            "Base model NOT restored after LoRA unload(). Adapter weights still "
            "injected into model_full. Refusing to certify."
        )

    return _compare_logit_matrices(mat_full, mat_lora)


# ──────────────────────────────────────────────────────────────────────────────
# Collector 10 — PREFIX CACHE
# ──────────────────────────────────────────────────────────────────────────────

def collect_prefix_cache(
    model,
    tokenizer,
    prompts: Sequence[str],
    shared_prefix: str,
    device: str = "cuda",
) -> List[float]:
    """Certify prefix/prompt cache reuse.

    Per prompt: prefill prefix → cache KVs → full recompute (prefix+prompt
    from scratch) vs cached run (prompt with cached prefix KVs). Measures
    next-token logit cosine similarity of cached vs recomputed.
    """
    import torch

    _require_model_tokenizer(model, tokenizer)
    _require_prompts(prompts)
    if not shared_prefix or not shared_prefix.strip():
        raise CollectionError("shared_prefix is empty.")

    model.eval()
    sims: List[float] = []
    with torch.inference_mode():
        prefix_inputs = tokenizer(shared_prefix, return_tensors="pt").to(device)
        prefix_len = int(prefix_inputs["input_ids"].shape[1])
        raw_prefix_kvs = model(**prefix_inputs, use_cache=True).past_key_values
        if raw_prefix_kvs is None:
            raise CollectionError("Model does not support KV caching (use_cache=True).")

        # DynamicCache (transformers >=4.36) — deepcopy directly.
        # to_legacy_cache/from_legacy_cache removed in transformers 5.x.
        prefix_kvs = raw_prefix_kvs

        for prompt in prompts:
            # Pass A: full recompute with no cache — ground truth
            logits_full = (
                model(
                    **tokenizer(
                        shared_prefix + prompt, return_tensors="pt",
                        truncation=True, max_length=4096,
                    ).to(device),
                    use_cache=False,
                ).logits[:, -1, :].float().squeeze(0).cpu().numpy()
            )

            # Pass B: cached run.
            # (a) Deepcopy per prompt — forward mutates DynamicCache in place,
            #     so prompt 2+ would see contaminated KVs without the copy.
            # (b) attention_mask spans prefix + prompt so the model knows the full
            #     sequence length.
            # (c) position_ids start at prefix_len so RoPE embeddings are correct;
            #     without this, every token re-encodes at position 0..prompt_len
            #     and the logits differ from the full recompute even when caching
            #     is mathematically exact — producing false FAIL certs.
            # add_special_tokens=False: Llama/Mistral/Qwen prepend BOS by default.
            # Without this, cached pass feeds [BOS, prefix..., BOS, prompt] while
            # full recompute feeds [BOS, prefix, prompt] — different sequences,
            # real logit divergence, false FAIL cert even when caching is exact.
            prompt_enc = tokenizer(
                prompt, return_tensors="pt", truncation=True,
                max_length=3072, add_special_tokens=False,
            ).to(device)
            prompt_len = int(prompt_enc["input_ids"].shape[1])

            if prefix_len + prompt_len > 4096:
                raise CollectionError(
                    f"prefix ({prefix_len} tokens) + prompt ({prompt_len} tokens) "
                    f"= {prefix_len + prompt_len} tokens exceeds max_length=4096. "
                    "Silent truncation would make the two passes compare different "
                    "token sequences. Use a shorter prefix or prompt."
                )

            attention_mask = torch.ones(
                1, prefix_len + prompt_len, dtype=torch.long, device=device
            )
            position_ids = torch.arange(
                prefix_len, prefix_len + prompt_len, dtype=torch.long, device=device
            ).unsqueeze(0)

            per_prompt_cache = copy.deepcopy(prefix_kvs)

            logits_cached = (
                model(
                    input_ids=prompt_enc["input_ids"],
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=per_prompt_cache,
                    use_cache=False,
                ).logits[:, -1, :].float().squeeze(0).cpu().numpy()
            )

            for arr, label in ((logits_full, "full"), (logits_cached, "cached")):
                if not np.all(np.isfinite(arr)):
                    raise CollectionError(f"Non-finite logits in prefix_cache ({label} pass).")
            sims.append(cos_sim(logits_full, logits_cached))

    if not sims:
        raise CollectionError("collect_prefix_cache produced no measurements.")
    return sims


# ──────────────────────────────────────────────────────────────────────────────
# Shared driver for collectors 14/15/16
# ──────────────────────────────────────────────────────────────────────────────

def _compare_full_vs_modified(
    model,
    tokenizer,
    prompts: Sequence[str],
    modified_context: Callable[[], ContextManager],
    max_length: int = 4096,
) -> List[float]:
    """Two passes through the same model: clean, then inside modified_context.

    Context manager MUST restore model on exit. Verified by re-running
    prompt 0 clean afterwards and demanding bitwise-identical logits.
    """
    if not prompts:
        raise CollectionError("_compare_full_vs_modified: empty prompt list.")

    baseline = capture_logits_hf("", prompts, max_length=max_length, model=model, tokenizer=tokenizer)
    with modified_context():
        modified = capture_logits_hf("", prompts, max_length=max_length, model=model, tokenizer=tokenizer)

    recheck = capture_logits_hf("", [prompts[0]], max_length=max_length, model=model, tokenizer=tokenizer)
    if not np.allclose(baseline[0], recheck[0], rtol=0.0, atol=0.0):
        raise CollectionError(
            "Model state NOT restored after modified-context pass. "
            "Context manager leaks hooks or weight edits. Refusing to certify."
        )
    return cos_sims_from_logit_matrices(baseline, modified)


# ──────────────────────────────────────────────────────────────────────────────
# Collector 11 — ENGINE SWAP
# ──────────────────────────────────────────────────────────────────────────────

def collect_engine_swap(
    capture_a_path: str,
    capture_b_path: str,
) -> List[float]:
    """Compare two on-disk captures from two different environments.

    Workflow:
        # env A  (vLLM 0.8, A100, etc.)
        deltacert capture --backend vllm --model X --prompts p.txt --out A.npz
        # env B  (vLLM 0.9, H100, etc.)
        deltacert capture --backend vllm --model X --prompts p.txt --out B.npz
        # anywhere — CPU box is fine
        deltacert compare --a A.npz --b B.npz --out cert.json
    """
    logits_a, meta_a = load_logits(capture_a_path)
    logits_b, meta_b = load_logits(capture_b_path)

    if meta_a.get("prompts_sha256") != meta_b.get("prompts_sha256"):
        raise CollectionError("Prompt-set hash mismatch between captures.")
    if meta_a.get("model_id") != meta_b.get("model_id"):
        raise CollectionError(
            f"Model mismatch: '{meta_a.get('model_id')}' vs '{meta_b.get('model_id')}'."
        )
    if meta_a.get("vocab_size") != meta_b.get("vocab_size"):
        raise CollectionError("Vocab size mismatch between captures.")

    logger.info("engine_swap: %s (%s) vs %s (%s)",
                meta_a.get("engine_label"), meta_a.get("gpu", "cpu"),
                meta_b.get("engine_label"), meta_b.get("gpu", "cpu"))
    return cos_sims_from_logit_matrices(logits_a, logits_b)


# ──────────────────────────────────────────────────────────────────────────────
# Collector 12 — BATCH DIVERGENCE
# ──────────────────────────────────────────────────────────────────────────────

def collect_batch_divergence(
    model_name_or_path: str,
    prompts: Sequence[str],
    batched_size: Optional[int] = None,
    tensor_parallel_size: int = 1,
    num_logprobs: int = 20,
    engine_kwargs: Optional[dict] = None,
) -> List[float]:
    """batch=1 vs batch=N through the SAME in-process vLLM engine.

    Pass A: one prompt per generate() call.
    Pass B: all prompts in one generate() call (continuous-batching scheduler).
    Measures exact batch-invariance failure from kernel reduction strategy changes.
    """
    import vllm

    if len(prompts) < 2:
        raise CollectionError("collect_batch_divergence needs >= 2 prompts.")

    kwargs = dict(model=model_name_or_path, tensor_parallel_size=tensor_parallel_size, enforce_eager=False)
    if engine_kwargs:
        kwargs.update(engine_kwargs)
    llm = vllm.LLM(**kwargs)

    try:
        rows_single = [
            capture_logits_vllm(model_name_or_path, [p], num_logprobs=num_logprobs, llm=llm)[0]
            for p in prompts
        ]
        logits_single = np.stack(rows_single, axis=0)

        if batched_size is None or batched_size >= len(prompts):
            logits_batched = capture_logits_vllm(model_name_or_path, prompts, num_logprobs=num_logprobs, llm=llm)
        else:
            chunks = [
                capture_logits_vllm(model_name_or_path, prompts[i:i + batched_size], num_logprobs=num_logprobs, llm=llm)
                for i in range(0, len(prompts), batched_size)
            ]
            logits_batched = np.concatenate(chunks, axis=0)
    finally:
        del llm
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    return cos_sims_from_logit_matrices(logits_single, logits_batched)


# ──────────────────────────────────────────────────────────────────────────────
# Collector 13 — SPECULATIVE DECODING
# ──────────────────────────────────────────────────────────────────────────────

def collect_speculative_decode(
    model_name_or_path: str,
    prompts: Sequence[str],
    speculative_config: dict,
    tensor_parallel_size: int = 1,
    num_logprobs: int = 20,
    engine_kwargs: Optional[dict] = None,
) -> List[float]:
    """Speculative decoding OFF vs ON. Two sequential vLLM engine instantiations.

    `speculative_config` passed straight to vLLM:
        {"model": "meta-llama/Llama-3.2-1B", "num_speculative_tokens": 5}
        {"method": "ngram", "num_speculative_tokens": 4, "prompt_lookup_max": 4}
    """
    if not speculative_config:
        raise CollectionError("collect_speculative_decode: speculative_config is empty.")

    base_kwargs = dict(engine_kwargs or {})
    logits_off = capture_logits_vllm(
        model_name_or_path, prompts,
        tensor_parallel_size=tensor_parallel_size,
        num_logprobs=num_logprobs, engine_kwargs=base_kwargs,
    )
    # vLLM's engine-core process teardown (triggered by capture_logits_vllm's
    # own `del llm`) is asynchronous — starting the second engine immediately
    # after can fail with "Free memory ... is less than desired GPU memory
    # utilization" because the OS hasn't reclaimed the first engine's VRAM
    # yet. Poll real GPU memory rather than trusting Python-side del timing.
    _wait_for_gpu_memory(min_free_gb=30.0, timeout_s=60.0)
    spec_kwargs = dict(base_kwargs)
    spec_kwargs["speculative_config"] = speculative_config
    logits_on = capture_logits_vllm(
        model_name_or_path, prompts,
        tensor_parallel_size=tensor_parallel_size,
        num_logprobs=num_logprobs, engine_kwargs=spec_kwargs,
    )
    return cos_sims_from_logit_matrices(logits_off, logits_on)


# ──────────────────────────────────────────────────────────────────────────────
# Collector 14 — SPARSE ATTENTION
# ──────────────────────────────────────────────────────────────────────────────

def collect_sparse_attention(
    model,
    tokenizer,
    prompts: Sequence[str],
    sparse_context: Callable[[], ContextManager],
    max_length: int = 4096,
) -> List[float]:
    """Full attention vs sparse-masked attention on the same loaded model.

    `sparse_context` installs the sparse path on __enter__ and removes on
    __exit__ (FlexAttention mask, HART-PF mask, H2O, etc.). Restoration
    is verified; a leaky context hard-fails.

    Example:
        @contextlib.contextmanager
        def my_mask():
            handles = install_mask(model, mask)
            try:
                yield
            finally:
                for h in handles: h.remove()

        cos_sims = collect_sparse_attention(model, tok, prompts, my_mask)
    """
    return _compare_full_vs_modified(model, tokenizer, prompts, sparse_context, max_length)


# ──────────────────────────────────────────────────────────────────────────────
# Collector 15 — MoE TOKEN DROPPING
# ──────────────────────────────────────────────────────────────────────────────

def collect_moe_token_dropping(
    model,
    tokenizer,
    prompts: Sequence[str],
    set_capacity: Callable[[float], None],
    capacity_baseline: float,
    capacity_deployed: float,
    max_length: int = 4096,
) -> List[float]:
    """Capacity factor A (no drops) vs capacity factor B (deployed, token-dropping).

    `set_capacity(cf)` mutates the live model's routing config in place.
    Restoration is verified; a partial setter hard-fails.
    """
    if capacity_baseline == capacity_deployed:
        raise CollectionError("capacity_baseline == capacity_deployed; nothing to certify.")

    set_capacity(capacity_baseline)

    @contextlib.contextmanager
    def _deployed():
        set_capacity(capacity_deployed)
        try:
            yield
        finally:
            set_capacity(capacity_baseline)

    return _compare_full_vs_modified(model, tokenizer, prompts, _deployed, max_length)


# ──────────────────────────────────────────────────────────────────────────────
# Collector 16 — NEURON SKIPPING (SRED / SNAP)
# ──────────────────────────────────────────────────────────────────────────────

def collect_neuron_skipping(
    model,
    tokenizer,
    prompts: Sequence[str],
    prune_context: Callable[[], ContextManager],
    max_length: int = 4096,
) -> List[float]:
    """Full model vs runtime-pruned model (SRED dead-neuron skipping, SNAP, etc.).

    `prune_context` installs pruning on __enter__, removes on __exit__.
    Same contract as collect_sparse_attention — same restoration check.
    """
    return _compare_full_vs_modified(model, tokenizer, prompts, prune_context, max_length)


# ──────────────────────────────────────────────────────────────────────────────
# Collector 17 — MODEL SWAP (checkpoint A vs checkpoint B)
# ──────────────────────────────────────────────────────────────────────────────

def collect_model_swap(
    capture_a_path: str,
    capture_b_path: str,
) -> List[float]:
    """Certify a MODEL UPDATE: old checkpoint vs new checkpoint.

    "Did the new fine-tune / re-quantized release / Llama 4.1 change my
    outputs vs 4.0?" — the negative-flip problem: aggregate benchmarks can
    improve while specific workloads silently regress.

    Identical two-process workflow to engine_swap, but the model IDs are
    EXPECTED to differ — that is the change being certified. Prompt-set and
    vocab checks still apply hard: a model update that changes the tokenizer
    vocabulary is not logit-comparable and must fail loudly, not certify.

        # before updating
        deltacert capture --model my-org/model-v1 --output v1.npz
        # after updating
        deltacert capture --model my-org/model-v2 --output v2.npz
        deltacert certify --model my-org/model-v2 \\
            --checks model_swap --baseline v1.npz --candidate v2.npz
    """
    logits_a, meta_a = load_logits(capture_a_path)
    logits_b, meta_b = load_logits(capture_b_path)

    if meta_a.get("prompts_sha256") != meta_b.get("prompts_sha256"):
        raise CollectionError(
            "Prompt-set hash mismatch between captures. model_swap requires "
            "the identical canary prompt list on both sides."
        )
    if meta_a.get("vocab_size") != meta_b.get("vocab_size"):
        raise CollectionError(
            f"Vocab size mismatch ({meta_a.get('vocab_size')} vs "
            f"{meta_b.get('vocab_size')}). The update changed the tokenizer; "
            "logit distributions are not comparable. This IS a breaking "
            "change - treat as NOT CERTIFIED and re-baseline."
        )
    if meta_a.get("model_id") == meta_b.get("model_id"):
        logger.warning(
            "model_swap: both captures report model_id='%s'. If you meant to "
            "compare engines or hardware, use engine_swap instead.",
            meta_a.get("model_id"),
        )

    logger.info(
        "model_swap: %s -> %s",
        meta_a.get("model_id"), meta_b.get("model_id"),
    )
    return cos_sims_from_logit_matrices(logits_a, logits_b)


# ──────────────────────────────────────────────────────────────────────────────
# Collector 18 — PROVIDER DRIFT (hosted API vs saved baseline)
# ──────────────────────────────────────────────────────────────────────────────

def capture_logits_openai_api(
    api_base: str,
    model: str,
    prompts: Sequence[str],
    api_key: Optional[str] = None,
    num_logprobs: int = 20,
    timeout: float = 120.0,
    max_retries: int = 3,
    vocab_size: int = 262144,
) -> np.ndarray:
    """Next-token top-k log-distribution per prompt from an OpenAI-compatible
    completions API (OpenAI, vLLM serve, SGLang, TGI, most gateways).

    Uses chat.completions with max_tokens=1, temperature=0, logprobs=True,
    top_logprobs=num_logprobs. Providers cap top_logprobs (OpenAI: 20), so
    the certified quantity is the top-k log-distribution — the same
    dense-embedding approach as capture_logits_vllm. Token STRINGS are hashed
    into a fixed-size vector because hosted APIs return strings, not ids;
    both sides of any comparison use the same embedding, so cosine geometry
    is consistent.

    NOTE: providers that do not return logprobs cannot be certified this way —
    the call hard-fails rather than degrading to text comparison.
    """
    import urllib.request
    import urllib.error

    if not prompts:
        raise CollectionError("capture_logits_openai_api: empty prompt list.")
    key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("DELTACERT_API_KEY")
    url = api_base.rstrip("/") + "/chat/completions"

    def _token_slot(token_str: str) -> int:
        h = hashlib.sha256(token_str.encode("utf-8")).digest()
        return int.from_bytes(h[:8], "big") % vocab_size

    rows: List[np.ndarray] = []
    for prompt in prompts:
        body = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1,
            "temperature": 0,
            "logprobs": True,
            "top_logprobs": int(num_logprobs),
        }).encode("utf-8")

        resp_data = None
        last_err: Optional[Exception] = None
        for attempt in range(max_retries):
            req = urllib.request.Request(
                url, data=body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {key}" if key else "",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    resp_data = json.loads(resp.read().decode("utf-8"))
                break
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
                last_err = e
                time.sleep(2.0 * (attempt + 1))
        if resp_data is None:
            raise CollectionError(
                f"API request failed after {max_retries} retries: {last_err}"
            )

        try:
            content_lp = resp_data["choices"][0]["logprobs"]["content"][0]
            top = content_lp["top_logprobs"]
        except (KeyError, IndexError, TypeError):
            raise CollectionError(
                f"API at {api_base} returned no logprobs for model '{model}'. "
                "This provider/endpoint does not expose logprobs - "
                "provider_drift cannot certify it. Refusing to fall back to "
                "text comparison."
            )
        if not top:
            raise CollectionError("API returned an empty top_logprobs list.")

        vec = np.full(vocab_size, -50.0, dtype=np.float32)
        for entry in top:
            vec[_token_slot(entry["token"])] = float(entry["logprob"])
        rows.append(vec)

    if len(rows) != len(prompts):
        raise CollectionError(
            f"capture_logits_openai_api: {len(rows)} rows for {len(prompts)} prompts."
        )
    return np.stack(rows, axis=0)


def collect_provider_drift(
    api_base: str,
    model: str,
    prompts: Sequence[str],
    baseline_path: str,
    api_key: Optional[str] = None,
    num_logprobs: int = 20,
    save_new_baseline_path: Optional[str] = None,
) -> List[float]:
    """Certify that a HOSTED model still behaves like its saved baseline.

    The pain: providers update model aliases silently; research measured
    accuracy drops in a majority of silent updates while the endpoint name
    stayed identical. This collector captures today's top-k logprobs on a
    fixed canary prompt set and compares against a baseline captured earlier.

    Behavior:
      * baseline_path missing  → captures today's logits, saves them to
        baseline_path, raises CollectionError (cannot certify against itself
        on day one — first run establishes the baseline only).
      * baseline_path present  → captures today's logits, validates prompt
        hash + model name + num_logprobs k, returns cos_sims.
      * save_new_baseline_path → also saves today's capture for explicit
        baseline rotation after an approved update.
    """
    current = capture_logits_openai_api(
        api_base, model, prompts, api_key=api_key, num_logprobs=num_logprobs
    )

    if not os.path.exists(baseline_path):
        save_logits(
            baseline_path, current, prompts,
            engine_label=f"api:{api_base}", model_id=model,
            extra_metadata={"num_logprobs": num_logprobs, "capture_kind": "provider_api"},
        )
        raise CollectionError(
            f"No baseline existed at '{baseline_path}'. Today's capture was "
            "saved as the new baseline. Re-run after the next provider cycle "
            "to measure drift - refusing to certify a model against itself."
        )

    baseline, meta = load_logits(baseline_path)
    if meta.get("prompts_sha256") != _prompts_hash(prompts):
        raise CollectionError(
            "Canary prompt set differs from the one used for the baseline. "
            "Drift measurement requires identical canaries. Re-baseline or "
            "restore the original prompt file."
        )
    if meta.get("model_id") != model:
        raise CollectionError(
            f"Baseline was captured for '{meta.get('model_id')}', not "
            f"'{model}'. Refusing cross-model drift comparison."
        )
    if int(meta.get("num_logprobs", -1)) != int(num_logprobs):
        raise CollectionError(
            f"Baseline used top_logprobs={meta.get('num_logprobs')}, current "
            f"run uses {num_logprobs}. k must match for the top-k "
            "distribution comparison to be well-defined."
        )

    if save_new_baseline_path:
        save_logits(
            save_new_baseline_path, current, prompts,
            engine_label=f"api:{api_base}", model_id=model,
            extra_metadata={"num_logprobs": num_logprobs, "capture_kind": "provider_api"},
        )

    return cos_sims_from_logit_matrices(baseline, current)


# ──────────────────────────────────────────────────────────────────────────────
# Collector 19 — PROMPT SWAP (system prompt v1 vs v2)
# ──────────────────────────────────────────────────────────────────────────────

def collect_prompt_swap(
    model,
    tokenizer,
    questions: Sequence[str],
    system_prompt_a: str,
    system_prompt_b: str,
    device: str = "cuda",
    max_length: int = 4096,
) -> List[float]:
    """Certify a PROMPT UPDATE: system prompt v1 vs v2, same canary
    questions, same model.

    The pain: small prompt edits ("three words for conversational flow")
    have caused structured-output error spikes and production halts. This
    measures the output-distribution shift the edit causes, per question,
    before it ships.

    Uses the tokenizer's chat template when available (that is what
    production serving applies); falls back to plain concatenation with a
    warning otherwise.
    """
    _require_model_tokenizer(model, tokenizer)
    _require_prompts(questions)
    if system_prompt_a == system_prompt_b:
        raise CollectionError("system_prompt_a == system_prompt_b; nothing to certify.")
    if not system_prompt_a.strip() or not system_prompt_b.strip():
        raise CollectionError("Empty system prompt supplied.")

    import torch

    has_template = getattr(tokenizer, "chat_template", None) is not None
    if not has_template:
        logger.warning(
            "prompt_swap: tokenizer has no chat_template; falling back to "
            "plain concatenation. If production serving uses a chat template, "
            "results will not reflect production formatting."
        )

    def _encode(system_prompt: str, question: str) -> dict:
        if has_template:
            ids = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": question},
                ],
                add_generation_prompt=True,
                return_tensors="pt",
            )
            # transformers 5.x returns BatchEncoding; older returned a raw tensor
            input_ids = ids["input_ids"] if hasattr(ids, "__getitem__") and not isinstance(ids, torch.Tensor) else ids
            if input_ids.shape[1] > max_length:
                raise CollectionError(
                    f"system prompt + question = {input_ids.shape[1]} tokens "
                    f"exceeds max_length={max_length}. Silent truncation "
                    "would compare different sequences - shorten inputs."
                )
            return {"input_ids": input_ids.to(device)}
        text = system_prompt + "\n\n" + question
        enc = tokenizer(text, return_tensors="pt", truncation=False)
        if enc["input_ids"].shape[1] > max_length:
            raise CollectionError(
                f"system prompt + question = {enc['input_ids'].shape[1]} "
                f"tokens exceeds max_length={max_length}."
            )
        return {k: v.to(device) for k, v in enc.items()}

    model.eval()
    sims: List[float] = []
    with torch.inference_mode():
        for q in questions:
            l_a = (
                model(**_encode(system_prompt_a, q))
                .logits[:, -1, :].float().squeeze(0).cpu().numpy()
            )
            l_b = (
                model(**_encode(system_prompt_b, q))
                .logits[:, -1, :].float().squeeze(0).cpu().numpy()
            )
            sims.append(cos_sim(l_a, l_b))

    if not sims:
        raise CollectionError("collect_prompt_swap produced no measurements.")
    return sims


# ──────────────────────────────────────────────────────────────────────────────
# Collector 20 — TRAJECTORY DIVERGENCE (positional d over long continuations)
# ──────────────────────────────────────────────────────────────────────────────

def d_profile(positional_cos_sims: Sequence[float]) -> dict:
    """Per-position divergence profile from a list of per-position cos_sims
    along ONE continuation.

    Returns:
        {
          "d_per_position": [...],   # d at each continuation token
          "d_min": float,            # worst position — the certified quantity
          "d_min_position": int,     # where the trajectory is weakest
          "d_final": float,          # d at the last position
          "n_positions": int,
        }
    """
    if not positional_cos_sims:
        raise CollectionError(
            "d_profile called with zero positions. Refusing to certify an "
            "empty trajectory."
        )
    ds = [_d_from_delta(_commutator_magnitude(c)) for c in positional_cos_sims]
    d_min = min(ds)
    return {
        "d_per_position": ds,
        "d_min": d_min,
        "d_min_position": int(ds.index(d_min)),
        "d_final": ds[-1],
        "n_positions": len(ds),
    }


def trajectory_layer_result(
    profiles: Sequence[dict],
    threshold: float = CERT_THRESHOLD_D,
    extra: Optional[dict] = None,
) -> dict:
    """Certificate layer result for trajectory collectors.

    The certified quantity is the minimum d over ALL positions of ALL
    reference trajectories — the single weakest token anywhere in the
    calibration horizon. Deliberately the harshest statistic: averaging
    over positions would hide late-horizon forking.
    """
    if not profiles:
        raise CollectionError("trajectory_layer_result called with zero trajectories.")
    d_min_overall = min(p["d_min"] for p in profiles)
    if d_min_overall == 0.0:
        d_min_overall = 0.0  # normalize IEEE-754 negative zero (-0.0) to 0.0 for clean display
    worst = min(profiles, key=lambda p: p["d_min"])
    result = {
        "n_trajectories": len(profiles),
        "n_positions_total": int(sum(p["n_positions"] for p in profiles)),
        "d_comm": d_min_overall,
        "d_min_position_in_worst_trajectory": worst["d_min_position"],
        "d_final_mean": float(sum(p["d_final"] for p in profiles) / len(profiles)),
        "divergence_bound": divergence_bound(d_min_overall),
        "certified": bool(d_min_overall >= threshold),
        "statistic": "min d over all positions of all reference trajectories",
    }
    if extra:
        result.update(extra)
    return result


def _teacher_forced_logits_hf(
    model,
    tokenizer,
    prompt: str,
    continuation: str,
    device: str,
    max_positions: int,
):
    """One forward over prompt+continuation; return (fp16 CPU tensor
    [n_cont, vocab], n_cont).

    Position t is the model's predicted distribution for continuation token t
    (input index prompt_len-1+t).
    """
    import torch

    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    cont_ids = tokenizer(
        continuation, return_tensors="pt", add_special_tokens=False
    )["input_ids"]
    n_prompt = int(prompt_ids.shape[1])
    n_cont = int(cont_ids.shape[1])
    if n_cont < 2:
        raise CollectionError(
            "Reference continuation is under 2 tokens - no trajectory to "
            "measure. Provide real 200-2000 token workload outputs."
        )
    if n_cont > max_positions:
        raise CollectionError(
            f"Continuation is {n_cont} tokens; max_positions={max_positions}. "
            "Raise max_positions explicitly rather than silently truncating "
            "the horizon being certified."
        )
    input_ids = torch.cat([prompt_ids, cont_ids], dim=1).to(device)
    with torch.inference_mode():
        logits = model(input_ids=input_ids).logits
    span = logits[0, n_prompt - 1: n_prompt - 1 + n_cont, :]
    if not torch.isfinite(span).all():
        raise CollectionError("Non-finite logits in teacher-forced pass. Refusing to certify.")
    return span.to(torch.float16).cpu(), n_cont


def _positional_cos(baseline_fp16, modified_fp16) -> List[float]:
    """Per-position cosine between two [n_cont, vocab] fp16 CPU tensors.
    Upcast to float64 so fp16 storage does not affect the certified number.
    """
    a = baseline_fp16.numpy().astype(np.float64)
    b = modified_fp16.numpy().astype(np.float64)
    if a.shape != b.shape:
        raise CollectionError(f"Trajectory shape mismatch: {a.shape} vs {b.shape}.")
    return [cos_sim(a[t], b[t]) for t in range(a.shape[0])]


def collect_trajectory(
    model,
    tokenizer,
    cases: Sequence[Tuple[str, str]],
    modified_context: Callable[[], ContextManager],
    device: str = "cuda",
    max_positions: int = 4096,
) -> List[dict]:
    """Trajectory divergence for hook-style optimizations on one loaded model.

    Certifies KV-cache quant, activation quant, sparse attention, pruning,
    MoE capacity — anything expressible as a context manager — over full
    long continuations rather than a single next-token position.

    Args:
        cases: list of (prompt, reference_continuation) pairs. Continuations
               are fixed measuring sticks — greedy outputs of the clean model
               on your real workload, generated once and reused.
        modified_context: installs the optimization on __enter__, removes on
               __exit__. Same contract as collectors 14-16. Leaky context
               hard-fails.

    Returns list of d_profile dicts, one per case.
    Feed to trajectory_layer_result() for the certificate entry.

    Example:
        cases = [(task_prompt, clean_greedy_output), ...]
        profiles = collect_trajectory(model, tok, cases, kv_int8_context)
        layer = trajectory_layer_result(profiles)
    """
    _require_model_tokenizer(model, tokenizer)
    if not cases:
        raise CollectionError("collect_trajectory: empty case list.")
    for i, c in enumerate(cases):
        if not (isinstance(c, (tuple, list)) and len(c) == 2):
            raise CollectionError(
                f"cases[{i}] must be a (prompt, reference_continuation) pair."
            )

    model.eval()

    baselines = []
    for prompt, cont in cases:
        span, _ = _teacher_forced_logits_hf(model, tokenizer, prompt, cont, device, max_positions)
        baselines.append(span)

    profiles: List[dict] = []
    with modified_context():
        for (prompt, cont), base in zip(cases, baselines):
            span, _ = _teacher_forced_logits_hf(model, tokenizer, prompt, cont, device, max_positions)
            profiles.append(d_profile(_positional_cos(base, span)))

    recheck, _ = _teacher_forced_logits_hf(
        model, tokenizer, cases[0][0], cases[0][1], device, max_positions
    )
    if not np.array_equal(
        baselines[0].numpy().view(np.uint16),
        recheck.numpy().view(np.uint16),
    ):
        raise CollectionError(
            "Model state NOT restored after modified-context pass. "
            "Context manager leaks hooks or weight edits. Refusing to certify."
        )

    if not profiles:
        raise CollectionError("collect_trajectory produced no measurements.")
    return profiles


def collect_trajectory_two_models(
    model_a,
    model_b,
    tokenizer,
    cases: Sequence[Tuple[str, str]],
    device: str = "cuda",
    max_positions: int = 4096,
) -> List[dict]:
    """Trajectory divergence between two model instances.

    fp16 vs quantized, base vs LoRA-merged, old checkpoint vs new — over
    full continuations rather than a single position. Both models must share
    the tokenizer/vocab; a mismatch hard-fails via shape check in _positional_cos.
    """
    _require_model_tokenizer(model_a, tokenizer)
    _require_model_tokenizer(model_b, tokenizer)
    if not cases:
        raise CollectionError("collect_trajectory_two_models: empty case list.")

    model_a.eval()
    model_b.eval()
    profiles: List[dict] = []
    for prompt, cont in cases:
        span_a, _ = _teacher_forced_logits_hf(model_a, tokenizer, prompt, cont, device, max_positions)
        span_b, _ = _teacher_forced_logits_hf(model_b, tokenizer, prompt, cont, device, max_positions)
        profiles.append(d_profile(_positional_cos(span_a, span_b)))

    if not profiles:
        raise CollectionError("collect_trajectory_two_models produced no measurements.")
    return profiles


def _teacher_forced_logits_vllm(
    llm,
    tokenizer,
    prompt: str,
    continuation: str,
    vocab_size: int,
    num_logprobs: int = 20,
    max_positions: int = 4096,
):
    """One vLLM prompt_logprobs ("echo") pass over prompt+continuation;
    return (dense logit matrix [n_cont, vocab] float32 numpy, n_cont).

    Uses the SAME top-num_logprobs + dense-fill methodology as
    capture_logits_vllm (via _logprob_dict_to_dense, fill=-50.0), so the
    resulting d(t) profile is numerically comparable to a single-position
    vLLM KV-cache certificate measured on the identical engine — not a
    different logit surface. This is deliberate: a d(t) curve that started
    from a different measurement convention than the single-position 3.72
    value it's meant to be read alongside would make the two numbers
    incomparable, silently breaking the paper's "starts near the
    single-position value, then collapses" claim.

    vLLM's prompt_logprobs[i] is the model's predictive distribution for
    token i, conditioned on tokens [0:i) — so continuation token t (at
    input index n_prompt+t) reads directly from prompt_logprobs[n_prompt+t].
    Verified empirically: on a greedy-decoded reference, the actual chosen
    token appears in prompt_logprobs[n_prompt+t]'s top-k with a near-zero
    logprob, and is absent at n_prompt-1+t (an earlier, off-by-one attempt
    at this indexing produced a flat, non-decaying d(t) that didn't match
    the single-position measurement it should track — this index is what
    fixed it).
    """
    import vllm

    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    cont_ids = tokenizer(continuation, add_special_tokens=False)["input_ids"]
    n_prompt = len(prompt_ids)
    n_cont = len(cont_ids)
    if n_cont < 2:
        raise CollectionError(
            "Reference continuation is under 2 tokens - no trajectory to "
            "measure. Provide real 200-2000 token workload outputs."
        )
    if n_cont > max_positions:
        raise CollectionError(
            f"Continuation is {n_cont} tokens; max_positions={max_positions}. "
            "Raise max_positions explicitly rather than silently truncating "
            "the horizon being certified."
        )

    full_ids = prompt_ids + cont_ids
    params = vllm.SamplingParams(
        temperature=0.0, max_tokens=1, prompt_logprobs=num_logprobs
    )
    outs = llm.generate(
        [vllm.TokensPrompt(prompt_token_ids=full_ids)], sampling_params=params
    )
    prompt_logprobs = outs[0].prompt_logprobs
    if prompt_logprobs is None or len(prompt_logprobs) != len(full_ids):
        got = 0 if prompt_logprobs is None else len(prompt_logprobs)
        raise CollectionError(
            f"vLLM prompt_logprobs length mismatch: expected {len(full_ids)}, "
            f"got {got}."
        )

    rows = []
    for t in range(n_cont):
        # vLLM's prompt_logprobs[i] is the model's predicted distribution
        # for the token AT index i, conditioned on tokens[0:i] -- i.e. it's
        # already "the prediction that led to token i", not "the prediction
        # made after observing token i". Continuation token t sits at input
        # index n_prompt+t, so its predicted distribution is
        # prompt_logprobs[n_prompt+t] directly (NOT n_prompt-1+t; verified
        # empirically against greedy-decoded references, where the actual
        # chosen token must appear in prompt_logprobs[n_prompt+t]'s top-k
        # with a near-zero logprob, and does not appear at n_prompt-1+t).
        idx = n_prompt + t
        entry = prompt_logprobs[idx]
        if entry is None:
            raise CollectionError(
                f"vLLM returned no prompt_logprobs at position {idx} "
                f"(continuation token {t})."
            )
        rows.append(_logprob_dict_to_dense(entry, vocab_size))
    span = np.stack(rows, axis=0)
    if not np.isfinite(span).all():
        raise CollectionError(
            "Non-finite logits in teacher-forced vLLM pass. Refusing to certify."
        )
    return span, n_cont


def collect_trajectory_vllm_two_engines(
    llm_baseline,
    llm_modified,
    tokenizer,
    cases: Sequence[Tuple[str, str]],
    vocab_size: Optional[int] = None,
    num_logprobs: int = 20,
    max_positions: int = 4096,
) -> List[dict]:
    """Trajectory divergence for vLLM ENGINE-LEVEL changes (KV-cache dtype
    and anything else only reachable via engine construction flags, not an
    in-process context manager) — the vLLM analog of
    collect_trajectory_two_models.

    Why two engines, not one model + a hook: engine-level flags like
    kv_cache_dtype are set at vLLM.LLM(...) construction and aren't
    toggleable mid-session the way collect_trajectory's modified_context
    toggles bitsandbytes hooks on one HF model. Two separately-constructed
    engines is the only way to compare "default KV dtype" against "fp8 KV
    dtype" through vLLM's real serving path — same reasoning already used
    for validation/kv_cache_quant/run_flagship.py's single-position
    measurement (there via two subprocess-isolated engines; here the caller
    is expected to follow the same subprocess-isolation discipline if the
    two engines cannot safely coexist in one process's GPU memory).

    State-integrity check: unlike collect_trajectory (one model, hooks can
    leak across the modified_context boundary), there is no shared mutable
    state between two independently-constructed engines to leak — that
    hazard doesn't exist by construction. What CAN silently corrupt a
    result is per-engine non-determinism (e.g. a scheduler/batching path
    that isn't actually reproducible at temperature 0 across repeated
    calls). We guard against that directly: the baseline engine's first
    case is measured twice and must match bitwise, or the run hard-fails
    rather than certifying on a non-reproducible measurement.

    Args:
        cases: list of (prompt, reference_continuation) pairs — greedy
               outputs of llm_baseline on real workload prompts, generated
               once and frozen (same discipline as collect_trajectory).
        vocab_size: pass explicitly if llm.llm_engine.model_config lookup
               is unavailable in the caller's vLLM version; otherwise
               inferred from llm_baseline.

    Returns list of d_profile dicts, one per case. Feed to
    trajectory_layer_result() for the certificate entry.
    """
    if not cases:
        raise CollectionError("collect_trajectory_vllm_two_engines: empty case list.")
    for i, c in enumerate(cases):
        if not (isinstance(c, (tuple, list)) and len(c) == 2):
            raise CollectionError(
                f"cases[{i}] must be a (prompt, reference_continuation) pair."
            )

    if vocab_size is None:
        try:
            vocab_size = llm_baseline.llm_engine.model_config.get_vocab_size()
        except Exception:
            vocab_size = llm_baseline.get_tokenizer().vocab_size

    first_prompt, first_cont = cases[0]
    base_first, _ = _teacher_forced_logits_vllm(
        llm_baseline, tokenizer, first_prompt, first_cont, vocab_size,
        num_logprobs, max_positions,
    )
    base_first_recheck, _ = _teacher_forced_logits_vllm(
        llm_baseline, tokenizer, first_prompt, first_cont, vocab_size,
        num_logprobs, max_positions,
    )
    if not np.array_equal(base_first, base_first_recheck):
        raise CollectionError(
            "Baseline engine gave non-identical logits for the same "
            "prompt+continuation measured twice. This engine's forward "
            "pass is not reproducible at temperature 0 - refusing to "
            "certify a trajectory built on a non-deterministic measurement."
        )

    profiles: List[dict] = []
    for i, (prompt, cont) in enumerate(cases):
        base_span = base_first if i == 0 else _teacher_forced_logits_vllm(
            llm_baseline, tokenizer, prompt, cont, vocab_size,
            num_logprobs, max_positions,
        )[0]
        mod_span, _ = _teacher_forced_logits_vllm(
            llm_modified, tokenizer, prompt, cont, vocab_size,
            num_logprobs, max_positions,
        )
        cos_sims = [
            cos_sim(base_span[t], mod_span[t]) for t in range(base_span.shape[0])
        ]
        profiles.append(d_profile(cos_sims))

    if not profiles:
        raise CollectionError("collect_trajectory_vllm_two_engines produced no measurements.")
    return profiles


# ──────────────────────────────────────────────────────────────────────────────
# Collector 21 — free-running decode-dynamics certification
# ──────────────────────────────────────────────────────────────────────────────
#
# WHY this collector exists (not a duplicate of collect_trajectory*): by the
# chain rule, sequence-level KL(P||Q) over a length-T continuation decomposes
# as sum_t E_{prefix~P}[KL(P_t||Q_t)] -- exactly what teacher-forced trajectory
# certification measures, under one model's own prefixes. On the Qwen KV-fp8
# collapse this sum was measured and found uniformly small (d in [3.52, 4.14]
# over 30,643 positions) while the actual served output collapses into
# repetition. That is not a contradiction: greedy argmax is a discontinuous
# function of the logits, so two engines can be epsilon-close in every
# conditional distribution (hence close in the teacher-forced statistic)
# while their free-running greedy trajectories diverge into different
# attractors once a near-tied argmax flips. No bound on per-position
# conditional divergence can bound greedy-path divergence -- the quantity
# that actually failed downstream is a property of the DEPLOYED DECODE
# POLICY, not of the conditionals. This collector therefore does not compute
# d_COMM at all; it runs the deployed decode policy on both engines and
# measures the resulting output PROCESSES directly.
#
# Three orthogonal signals per prompt:
#   - degeneration (paired): sliding-window max-token-frequency and
#     distinct-3-gram ratio on the candidate's free-running generation,
#     paired against the baseline's own generation for the SAME prompt
#     (paired because some prompts induce repetition in healthy models too).
#   - cross-surprisal: teacher-force the CANDIDATE's own emitted tokens
#     through the BASELINE engine and record per-token surprisal
#     -log p_base(tok); a loop-entry token is improbable under the baseline
#     even when the loop interior is not, localizing the fork.
#   - fork position: first token index where the two engines' greedy outputs
#     differ -- context only, never a verdict by itself (benign changes fork
#     too, late and rarely).


@dataclass
class FreeRunPromptResult:
    prompt_sha: str
    fork_pos: int
    base_degenerate: bool
    cand_degenerate: bool
    cand_surprisal_mean: float
    cand_surprisal_max: float
    cand_surprisal_q95: float
    base_surprisal_q95: float
    n_tokens_base: int
    n_tokens_cand: int


def max_window_token_freq(token_ids: Sequence[int], window: int = 64) -> float:
    """Max over sliding windows of (most-common-token count / window).
    1.0 == a window that is a single repeated token ("be be be ...")."""
    n = len(token_ids)
    if n == 0:
        raise CollectionError("max_window_token_freq: empty generation.")
    if n <= window:
        c = Counter(token_ids)
        return c.most_common(1)[0][1] / n
    counts = Counter(token_ids[:window])
    best = counts.most_common(1)[0][1] / window
    for i in range(window, n):
        counts[token_ids[i]] += 1
        left = token_ids[i - window]
        counts[left] -= 1
        if counts[left] == 0:
            del counts[left]
        best = max(best, counts.most_common(1)[0][1] / window)
    return best


def distinct_ngram_ratio(token_ids: Sequence[int], n: int = 3) -> float:
    """Unique n-grams / total n-grams over a generation. Near 0 == looping."""
    if len(token_ids) < n:
        return 1.0
    grams = [tuple(token_ids[i:i + n]) for i in range(len(token_ids) - n + 1)]
    return len(set(grams)) / len(grams)


def is_degenerate(
    token_ids: Sequence[int],
    freq_thresh: float = 0.5,
    distinct_thresh: float = 0.15,
    window: int = 64,
) -> bool:
    """A generation is degenerate if EITHER signal trips. These thresholds
    are a shipped starting point, not a calibrated constant -- they must be
    validated against real stored generations (the manual repetition audit)
    before being cited in any result, same doctrine as every d_COMM
    threshold in this module."""
    return (
        max_window_token_freq(token_ids, window) >= freq_thresh
        or distinct_ngram_ratio(token_ids) <= distinct_thresh
    )


def fork_position(base_ids: Sequence[int], cand_ids: Sequence[int]) -> int:
    """First index where greedy outputs differ; -1 if identical over the
    compared span."""
    for i, (a, b) in enumerate(zip(base_ids, cand_ids)):
        if a != b:
            return i
    return -1


def surprisal_stats(logprobs_of_cand_under_base: Sequence[float]) -> Tuple[float, float, float]:
    """Per-token surprisal list -> (mean, max, q95). Input is
    log p_base(cand_token_t | cand_prefix_<t), from ONE teacher-forced pass
    of the candidate's own generation through the baseline engine."""
    if not logprobs_of_cand_under_base:
        raise CollectionError("surprisal_stats: empty scoring pass.")
    s = sorted(-lp for lp in logprobs_of_cand_under_base)
    n = len(s)
    return (sum(s) / n, s[-1], s[min(n - 1, math.ceil(0.95 * n) - 1)])


def _score_under_baseline_vllm(llm_baseline, tokenizer, prompt: str, token_ids: Sequence[int]) -> List[float]:
    """Teacher-force `token_ids` (a free-running generation) through
    llm_baseline via the same prompt_logprobs echo mechanism used for
    trajectory certification, and return per-token log p_base."""
    import vllm

    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    full_ids = list(prompt_ids) + list(token_ids)
    n_prompt = len(prompt_ids)
    n_cont = len(token_ids)
    params = vllm.SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1)
    outs = llm_baseline.generate([vllm.TokensPrompt(prompt_token_ids=full_ids)], sampling_params=params)
    prompt_logprobs = outs[0].prompt_logprobs
    if prompt_logprobs is None or len(prompt_logprobs) != len(full_ids):
        got = 0 if prompt_logprobs is None else len(prompt_logprobs)
        raise CollectionError(
            f"vLLM prompt_logprobs length mismatch scoring under baseline: "
            f"expected {len(full_ids)}, got {got}."
        )
    logprobs = []
    for t in range(n_cont):
        idx = n_prompt + t
        entry = prompt_logprobs[idx]
        if entry is None:
            raise CollectionError(f"vLLM returned no prompt_logprobs at position {idx} scoring candidate token {t}.")
        tok = int(token_ids[t])
        entry = {int(k): v for k, v in entry.items()}
        if tok in entry:
            lp = entry[tok]
            logprobs.append(float(lp.logprob if hasattr(lp, "logprob") else lp))
        else:
            # Candidate token fell outside baseline's top-k at this position --
            # exactly the "improbable under baseline" signal this function
            # exists to measure. Floor rather than drop, so a loop-entry
            # token that's completely absent from the baseline's top-k
            # (the strongest possible signal) isn't silently discarded.
            logprobs.append(-50.0)
    return logprobs


def collect_free_running_vllm_two_engines(
    llm_baseline,
    llm_modified,
    tokenizer,
    prompts: Sequence[str],
    max_new_tokens: int = 512,
    window: int = 64,
) -> List[FreeRunPromptResult]:
    """Free-running decode-dynamics comparison for vLLM ENGINE-LEVEL changes
    (KV-cache dtype and anything else only reachable via engine construction
    flags) -- the collector that measures what collect_trajectory_vllm_two_engines
    provably cannot (see module-level WHY comment above).

    Runs the deployed decode policy (greedy, matching every other collector's
    determinism discipline) on both engines for each prompt, then scores the
    candidate's own emitted tokens under the baseline engine. Hard-fails on
    any empty generation or scoring pass -- never a silent default.

    Args:
        prompts: real workload prompts (NOT frozen reference continuations --
                 free-running mode generates fresh from each engine, that's
                 the point).
        max_new_tokens: decode budget per engine per prompt.

    Returns a list of FreeRunPromptResult, one per prompt. Feed to
    certify_free_running() for the verdict.
    """
    if not prompts:
        raise CollectionError("collect_free_running_vllm_two_engines: empty prompt list.")

    import vllm

    greedy_params = vllm.SamplingParams(temperature=0.0, max_tokens=max_new_tokens)

    results: List[FreeRunPromptResult] = []
    for prompt in prompts:
        base_out = llm_baseline.generate([prompt], sampling_params=greedy_params)
        cand_out = llm_modified.generate([prompt], sampling_params=greedy_params)
        base_ids = list(base_out[0].outputs[0].token_ids)
        cand_ids = list(cand_out[0].outputs[0].token_ids)
        if not base_ids or not cand_ids:
            raise CollectionError(
                f"Empty free-running generation for prompt "
                f"{hashlib.sha256(prompt.encode()).hexdigest()[:12]}."
            )

        lp_cand = _score_under_baseline_vllm(llm_baseline, tokenizer, prompt, cand_ids)
        lp_base = _score_under_baseline_vllm(llm_baseline, tokenizer, prompt, base_ids)
        _, _, q95_base = surprisal_stats(lp_base)
        mean_cand, max_cand, q95_cand = surprisal_stats(lp_cand)

        results.append(FreeRunPromptResult(
            prompt_sha=hashlib.sha256(prompt.encode()).hexdigest()[:12],
            fork_pos=fork_position(base_ids, cand_ids),
            base_degenerate=is_degenerate(base_ids, window=window),
            cand_degenerate=is_degenerate(cand_ids, window=window),
            cand_surprisal_mean=mean_cand,
            cand_surprisal_max=max_cand,
            cand_surprisal_q95=q95_cand,
            base_surprisal_q95=q95_base,
            n_tokens_base=len(base_ids),
            n_tokens_cand=len(cand_ids),
        ))

    if not results:
        raise CollectionError("collect_free_running_vllm_two_engines produced no measurements.")
    return results


def mcnemar_exact_p(b: int, c: int) -> float:
    """Exact two-sided McNemar test on discordant pair counts.

    b = #(candidate degenerate, baseline clean), c = #(baseline degenerate,
    candidate clean) -- the two directions of disagreement in a paired
    binary outcome. Under the null that a fork (benign or not) merely
    RE-ROLLS whether the post-fork suffix degenerates -- rather than the
    change CAUSING degeneration -- the discordant pairs are exchangeable:
    b ~ Binomial(b+c, 1/2). Raw excess-rate (b/n) alone cannot distinguish
    "re-roll noise on a benign fork" from "caused collapse"; a config that
    forks on every prompt but degenerates no more than baseline's own
    lottery rate would still show nonzero excess by chance. This test can:
    it asks whether the b vs c split is more lopsided than a fair coin would
    produce, not just whether b is nonzero.

    Two-sided by convention (matches standard McNemar's test / scipy);
    b=0,c=0 (no discordant pairs at all) returns p=1.0 -- no evidence either
    way, not a hard failure, since a config that never forks and never
    degenerates on any prompt is a legitimate (if unremarkable) result.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = max(b, c)
    one_sided = sum(math.comb(n, i) for i in range(k, n + 1)) / (2 ** n)
    return min(1.0, 2 * one_sided)


def certify_free_running(
    results: Sequence[FreeRunPromptResult],
    tau_degen: float = 0.05,
    tau_surp: Optional[float] = None,
    mcnemar_alpha: float = 0.01,
) -> dict:
    """Fail-closed certificate over a FreeRunPromptResult list.

    unsafe if  (excess_degeneration_rate > tau_degen AND mcnemar_p < mcnemar_alpha)
           or  q95(surprisal_q95_candidate) - q95(surprisal_q95_baseline) > tau_surp

    excess_degeneration_rate = P(candidate degenerate AND baseline clean) --
    paired, so prompts where the baseline itself degenerates don't count
    against the candidate. The raw rate alone is not sufficient to fail
    closed: a change that forks the greedy path (including a BENIGN fork)
    re-rolls whatever spontaneous degeneration rate the model already has at
    baseline (measured nonzero on both models in this suite -- 3/43 Llama,
    7/50 Qwen), so some nonzero excess is expected even from harmless
    changes. mcnemar_exact_p() distinguishes re-roll noise (b vs c
    statistically symmetric) from a genuinely caused collapse (b >> c,
    non-symmetric) on the SAME discordant-pair data the excess rate is
    computed from -- requiring both conditions is strictly more conservative
    than the rate alone, never less (a change that trips the rate but not
    the significance test is NOT certified unsafe by this rule; it is
    reported, not refused, since more data may still resolve it).

    tau_*/mcnemar_alpha are per-change-type calibration inputs, same
    doctrine as every d_COMM threshold in this module: shipped defaults are
    a starting point, not a universal constant. Empty result sets are a
    hard error, never a vacuous pass.
    """
    if not results:
        raise CollectionError("certify_free_running called with zero results.")
    n = len(results)
    b = sum(1 for r in results if r.cand_degenerate and not r.base_degenerate)
    c = sum(1 for r in results if r.base_degenerate and not r.cand_degenerate)
    excess = b / n
    mcnemar_p = mcnemar_exact_p(b, c)

    surp_delta = None
    if n >= 20:
        cand_q95s = [r.cand_surprisal_q95 for r in results]
        base_q95s = [r.base_surprisal_q95 for r in results]
        surp_delta = statistics.quantiles(cand_q95s, n=20)[18] - statistics.quantiles(base_q95s, n=20)[18]

    degeneration_significant = excess > tau_degen and mcnemar_p < mcnemar_alpha
    unsafe = degeneration_significant or (
        tau_surp is not None and surp_delta is not None and surp_delta > tau_surp
    )
    return {
        "n_prompts": n,
        "excess_degeneration_rate": round(excess, 4),
        "mcnemar_b": b,
        "mcnemar_c": c,
        "mcnemar_p": mcnemar_p,
        "mcnemar_alpha": mcnemar_alpha,
        "degeneration_significant": degeneration_significant,
        "surprisal_q95_delta": round(surp_delta, 4) if surp_delta is not None else None,
        "fork_positions": [r.fork_pos for r in results],
        "certified": not unsafe,
        "verdict": "unsafe" if unsafe else "safe",
        "rule": (
            "fail-closed: (excess_degeneration_rate > tau_degen AND "
            "mcnemar_p < mcnemar_alpha) OR surprisal_q95_delta > tau_surp"
        ),
        "tau_degen": tau_degen,
        "tau_surp": tau_surp,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Self-contained CLI-drivable wrapper for collector 21
# ──────────────────────────────────────────────────────────────────────────────

def collect_free_running_vllm(
    model_name_or_path: str,
    prompts: Sequence[str],
    kv_cache_dtype: str = "fp8",
    max_new_tokens: int = 512,
    gpu_memory_utilization: float = 0.42,
    window: int = 64,
    tau_degen: float = 0.05,
    tau_surp: Optional[float] = None,
    mcnemar_alpha: float = 0.01,
) -> dict:
    """Self-contained, CLI-drivable free-running certification: boots BOTH
    vLLM engines (default KV dtype vs kv_cache_dtype) itself and returns the
    certify_free_running() verdict directly.

    Unlike collect_kv_cache_quant_vllm's sequential two-capture pattern, the
    two engines here must be alive CONCURRENTLY -- collect_free_running_vllm_two_engines
    interleaves calls to both per prompt (generate on each, then score the
    candidate's own tokens under the baseline). gpu_memory_utilization is
    capped low enough by default for both 7-8B-class engines to coexist on
    one H100-80GB without needing sequential boot/teardown.

    Returns the certify_free_running() dict directly -- NOT a cos_sims list
    -- because this collector is unlike every cosine-similarity-based
    collector in this module: its certificate is McNemar/degeneration-based,
    not d_COMM-based. There is no meaningful single-cosine reduction of
    "did the deployed decode policy collapse."
    """
    _require_prompts(prompts)
    import vllm
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    llm_baseline = vllm.LLM(
        model=model_name_or_path,
        kv_cache_dtype="auto",
        gpu_memory_utilization=gpu_memory_utilization,
    )
    llm_modified = vllm.LLM(
        model=model_name_or_path,
        kv_cache_dtype=kv_cache_dtype,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    results = collect_free_running_vllm_two_engines(
        llm_baseline, llm_modified, tokenizer, prompts,
        max_new_tokens=max_new_tokens, window=window,
    )
    return certify_free_running(
        results, tau_degen=tau_degen, tau_surp=tau_surp, mcnemar_alpha=mcnemar_alpha,
    )


# ──────────────────────────────────────────────────────────────────────────────
# CLI — two-process path for collector 11 (capture / compare)
# ──────────────────────────────────────────────────────────────────────────────

def _read_prompts_file(path: str) -> List[str]:
    if not os.path.exists(path):
        raise CollectionError(f"Prompts file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        prompts = [line.rstrip("\n") for line in f if line.strip()]
    if not prompts:
        raise CollectionError(f"Prompts file is empty: {path}")
    return prompts


def _cli_capture(args: argparse.Namespace) -> None:
    prompts = _read_prompts_file(args.prompts)
    if args.backend == "hf":
        logits = capture_logits_hf(
            args.model, prompts, device=args.device, dtype=args.dtype,
            max_length=args.max_length, trust_remote_code=args.trust_remote_code,
        )
    elif args.backend == "vllm":
        logits = capture_logits_vllm(
            args.model, prompts,
            tensor_parallel_size=args.tensor_parallel_size,
            num_logprobs=args.num_logprobs,
        )
    else:
        raise CollectionError(f"Unknown backend: {args.backend}")
    save_logits(args.out, logits, prompts, engine_label=args.label or args.backend, model_id=args.model)
    print(f"capture written: {args.out}")


def _cli_compare(args: argparse.Namespace) -> None:
    sims = collect_engine_swap(args.a, args.b)
    _, meta_a = load_logits(args.a)
    cert = certify_from_layers(
        model=str(meta_a.get("model_id", "unknown")),
        layers={"engine_swap": _make_layer_result(sims, extra={"capture_a": args.a, "capture_b": args.b})},
    )
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(cert, f, indent=2)
        print(f"certificate written: {args.out}")
    print(json.dumps(cert, indent=2))
    if args.enforce and not cert["certified"]:
        print(f"NOT CERTIFIED (d < {CERT_THRESHOLD_D}) - exiting nonzero.", file=sys.stderr)
        sys.exit(1)


def main(argv: Optional[Sequence[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        prog="deltacert",
        description="DeltaCert collectors — capture logits or compare two captures",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_cap = sub.add_parser("capture", help="Capture logits in THIS environment")
    p_cap.add_argument("--backend", choices=["hf", "vllm"], required=True)
    p_cap.add_argument("--model", required=True)
    p_cap.add_argument("--prompts", required=True, help="One prompt per line")
    p_cap.add_argument("--out", required=True, help="Output .npz path")
    p_cap.add_argument("--label", default=None)
    p_cap.add_argument("--device", default="cuda")
    p_cap.add_argument("--dtype", default="float16")
    p_cap.add_argument("--max-length", type=int, default=4096)
    p_cap.add_argument("--tensor-parallel-size", type=int, default=1)
    p_cap.add_argument("--num-logprobs", type=int, default=128)
    p_cap.add_argument("--trust-remote-code", action="store_true")
    p_cap.set_defaults(func=_cli_capture)

    p_cmp = sub.add_parser("compare", help="Compare two captures → cert.json")
    p_cmp.add_argument("--a", required=True)
    p_cmp.add_argument("--b", required=True)
    p_cmp.add_argument("--out", default=None)
    p_cmp.add_argument("--enforce", action="store_true", help="Exit 1 if d < 3.0")
    p_cmp.set_defaults(func=_cli_compare)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
