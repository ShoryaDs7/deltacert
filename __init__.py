"""
DeltaCert — Universal Bounded Divergence Certification for LLM Serving Systems

pip install deltacert

Quickstart:
    import deltacert as dc

    cert = dc.certify_system(
        model="meta-llama/Llama-3.1-8B",
        prompts=my_prompts,
        checks=["engine_swap", "weight_quant", "batch_divergence"],
        output_path="./cert.json",
    )

    # At server startup — refuses to start if not certified
    dc.enforce("./cert.json")

Full check names:
    "engine_swap"         — certify vLLM version upgrade / backend migration
    "batch_divergence"    — certify batch-size nondeterminism is bounded
    "spec_decoding"       — certify speculative decoding ON vs OFF (same engine)
    "sparse_attention"    — certify sparse attention mask vs full attention
    "moe_token_dropping"  — certify MoE capacity factor token dropping
    "neuron_skipping"     — certify runtime neuron pruning (SRED/SNAP)
    "weight_quant"        — certify int4/int8 weight quantization
    "kv_cache_quant"      — certify KV cache quantization
    "lora"                — certify LoRA adapter vs full model
    "allreduce_tp"        — certify compressed tensor-parallel AllReduce
    "alltoall_ep"         — certify compressed expert-parallel All-to-All
    "pipeline_parallel"   — certify compressed pipeline stage activations
    "kv_transfer"         — certify compressed prefill→decode KV transfer
    "prefix_cache"        — certify prefix/prompt cache reuse
    "activation_quant"    — certify activation quantization
    "gradient_compress"   — certify gradient compression fidelity (training)
    "model_swap"          — certify checkpoint update (old vs new fine-tune)
    "provider_drift"      — certify hosted API hasn't drifted from baseline
    "prompt_swap"         — certify system prompt edit before shipping
    "trajectory"          — certify optimization over full continuations (long-context)
"""

from deltacert.collectors import (
    CollectionError,
    collect_allreduce_tp,
    collect_alltoall_ep,
    collect_pipeline_parallel,
    collect_pipeline_parallel_tensors,
    collect_kv_transfer,
    collect_weight_quant,
    collect_kv_cache_quant,
    collect_activation_quant,
    collect_gradient_compress,
    collect_lora,
    collect_prefix_cache,
    collect_engine_swap,
    collect_batch_divergence,
    collect_speculative_decode,
    collect_sparse_attention,
    collect_moe_token_dropping,
    collect_neuron_skipping,
    collect_model_swap,
    capture_logits_openai_api,
    collect_provider_drift,
    collect_prompt_swap,
    d_profile,
    trajectory_layer_result,
    collect_trajectory,
    collect_trajectory_two_models,
    save_logits,
    load_logits,
    capture_logits_hf,
    capture_logits_vllm,
    certify_from_layers,
)

from deltacert.deltacert import (
    # Core formula
    d_comm,
    divergence_bound,
    certify_layer,
    calibrate_layer,

    # Config
    InferenceConfig,
    LayerSpec,

    # Layer name constants
    LAYER_ALLREDUCE_TP,
    LAYER_ALLTOALL_EP,
    LAYER_PIPELINE_PARALLEL,
    LAYER_KV_TRANSFER,
    LAYER_WEIGHT_QUANT,
    LAYER_KV_CACHE_QUANT,
    LAYER_ACTIVATION_QUANT,
    LAYER_GRADIENT_COMP,
    LAYER_LORA,
    LAYER_PREFIX_CACHE,
    LAYER_ENGINE_SWAP,
    LAYER_BATCH_DIVERGENCE,
    LAYER_SPEC_DECODING,
    LAYER_SPARSE_ATTENTION,
    LAYER_MOE_TOKEN_DROP,
    LAYER_NEURON_SKIPPING,
    LAYER_MODEL_SWAP,
    LAYER_PROVIDER_DRIFT,
    LAYER_PROMPT_SWAP,
    LAYER_TRAJECTORY,

    # Main API
    certify,
    certify_system,
    load_certificate,
    check_certified,
    enforce,
    compose_bounds,
    summary,
)

def certify_safe(
    baseline,
    candidate=None,
    tokenizer=None,
    prompts=None,
    *,
    check: str = None,
    budget: float = 3.0,
    raise_on_fail: bool = True,
    **kwargs,
) -> bool:
    """One call covers all 20 collectors. Raises DeploymentBlocked if not certified.

    Auto-detects the check for the two most common cases:
      two torch models  →  weight_quant
      two .npz paths    →  engine_swap
      http:// string    →  provider_drift

    For everything else pass check= explicitly:

      certify_safe(model, tok, prompts, check="kv_cache_quant",
                   compress_fn=c, decompress_fn=d)
      certify_safe(model, tok, prompts, check="lora",
                   lora_adapter_path="adapter/")
      certify_safe(model, tok, prompts, check="prefix_cache",
                   shared_prefix="You are a helpful assistant.")
      certify_safe("gpt-4o-mini", prompts=prompts, check="provider_drift",
                   api_base="https://api.openai.com/v1",
                   baseline_path="drift/baseline.npz")
      certify_safe("meta-llama/Llama-3.1-8B", prompts=prompts,
                   check="spec_decoding",
                   speculative_config={"model": "draft", "num_speculative_tokens": 5})
    """
    import os
    from deltacert.collectors import (
        collect_weight_quant, collect_engine_swap, collect_model_swap,
        collect_provider_drift, collect_lora, collect_prefix_cache,
        collect_prompt_swap, collect_kv_cache_quant, collect_kv_transfer,
        collect_activation_quant, collect_gradient_compress,
        collect_allreduce_tp, collect_alltoall_ep, collect_pipeline_parallel,
        collect_batch_divergence, collect_speculative_decode,
        collect_sparse_attention, collect_moe_token_dropping,
        collect_neuron_skipping, collect_trajectory_two_models,
        trajectory_layer_result, certify_from_layers, _make_layer_result,
        d_comm as _d_comm, CollectionError,
    )

    # ── auto-detect check for the common cases ────────────────────────────────
    if check is None:
        try:
            import torch
            if (isinstance(baseline, torch.nn.Module)
                    and isinstance(candidate, torch.nn.Module)):
                check = "weight_quant"
        except ImportError:
            pass
    if check is None:
        if (isinstance(baseline, str) and isinstance(candidate, str)
                and baseline.endswith(".npz") and candidate.endswith(".npz")):
            check = "engine_swap"
        elif (isinstance(baseline, str)
              and (baseline.startswith("http://") or baseline.startswith("https://"))):
            check = "provider_drift"
    if check is None:
        raise ValueError(
            "certify_safe: cannot auto-detect check from these arguments. "
            "Pass check='weight_quant' (or any of the 20 collector names). "
            "Full list: deltacert.__all__"
        )

    # ── route to the right collector ─────────────────────────────────────────
    _trajectory = check == "trajectory"

    if check == "weight_quant":
        cos_sims = collect_weight_quant(baseline, candidate, tokenizer, prompts)
        # Resolve the calibrated per-family threshold instead of the flat
        # default (calibrated for KV-cache/trajectory, not weight
        # quantization -- see hf_integration.py's identical fix). Unlike
        # that wrapper, certify_safe() accepts any quantized torch model,
        # not just bitsandbytes -- so detect the real backend from the
        # model's own HF quantization_config rather than assuming one.
        # Only bnb and gptq have real calibration data behind them
        # (_PROVISIONAL_METHOD_BUDGETS); anything else falls back to the
        # flat default with an explicit warning rather than a silent guess.
        if budget == 3.0:
            _hf_quant_method_map = {"bitsandbytes": "bnb", "gptq": "gptq"}
            _detected = None
            _qc = getattr(getattr(candidate, "config", None), "quantization_config", None)
            _raw_method = getattr(_qc, "quant_method", None)
            if _raw_method is not None:
                _detected = _hf_quant_method_map.get(getattr(_raw_method, "value", str(_raw_method)))
            if _detected is not None:
                from deltacert.deltacert import _PROVISIONAL_METHOD_BUDGETS
                budget = _PROVISIONAL_METHOD_BUDGETS.get(_detected, budget)
            else:
                print(
                    "DeltaCert: certify_safe could not detect a calibrated "
                    "quantization family for this weight_quant check (or "
                    "the detected method has no calibration data yet); "
                    f"using the flat default budget={budget}, which is "
                    "calibrated for KV-cache/trajectory checks, not weight "
                    "quantization. Pass budget= explicitly with your own "
                    "calibrated value (see `deltacert calibrate`) for a "
                    "trustworthy verdict."
                )
    elif check == "engine_swap":
        cos_sims = collect_engine_swap(baseline, candidate)
    elif check == "model_swap":
        cos_sims = collect_model_swap(baseline, candidate)
    elif check == "provider_drift":
        api_base = kwargs.pop("api_base", baseline)
        baseline_path = kwargs.pop("baseline_path", "deltacert_baseline.npz")
        api_key = kwargs.pop("api_key", None)
        cos_sims = collect_provider_drift(
            api_base, candidate or kwargs.pop("model", ""), prompts,
            baseline_path, api_key=api_key, **kwargs)
    elif check == "lora":
        cos_sims = collect_lora(baseline, tokenizer, prompts,
                                kwargs.pop("lora_adapter_path"), **kwargs)
    elif check == "prefix_cache":
        cos_sims = collect_prefix_cache(baseline, tokenizer, prompts,
                                        kwargs.pop("shared_prefix"), **kwargs)
    elif check == "prompt_swap":
        cos_sims = collect_prompt_swap(baseline, tokenizer, prompts,
                                       kwargs.pop("system_prompt_a"),
                                       kwargs.pop("system_prompt_b"), **kwargs)
    elif check == "kv_cache_quant":
        cos_sims = collect_kv_cache_quant(baseline, tokenizer, prompts,
                                          kwargs.pop("compress_fn"),
                                          kwargs.pop("decompress_fn"), **kwargs)
    elif check == "kv_transfer":
        cos_sims = collect_kv_transfer(baseline, tokenizer, prompts,
                                       kwargs.pop("compress_fn"),
                                       kwargs.pop("decompress_fn"), **kwargs)
    elif check == "activation_quant":
        cos_sims = collect_activation_quant(baseline, tokenizer, prompts,
                                            kwargs.pop("quant_fn"), **kwargs)
    elif check == "gradient_compress":
        cos_sims = collect_gradient_compress(baseline, tokenizer, prompts,
                                             kwargs.pop("compress_fn"), **kwargs)
    elif check == "allreduce_tp":
        cos_sims = collect_allreduce_tp(baseline, tokenizer, prompts,
                                        kwargs.pop("compress_fn"), **kwargs)
    elif check == "alltoall_ep":
        cos_sims = collect_alltoall_ep(baseline, tokenizer, prompts,
                                       kwargs.pop("compress_fn"), **kwargs)
    elif check == "pipeline_parallel":
        cos_sims = collect_pipeline_parallel(baseline, tokenizer, prompts,
                                             kwargs.pop("compress_fn"), **kwargs)
    elif check == "sparse_attention":
        cos_sims = collect_sparse_attention(baseline, tokenizer, prompts,
                                            kwargs.pop("sparse_context"), **kwargs)
    elif check == "moe_token_dropping":
        cos_sims = collect_moe_token_dropping(baseline, tokenizer, prompts,
                                              kwargs.pop("set_capacity"), **kwargs)
    elif check == "neuron_skipping":
        cos_sims = collect_neuron_skipping(baseline, tokenizer, prompts,
                                           kwargs.pop("prune_context"), **kwargs)
    elif check == "batch_divergence":
        cos_sims = collect_batch_divergence(baseline, prompts, **kwargs)
    elif check == "spec_decoding":
        cos_sims = collect_speculative_decode(
            baseline, prompts, kwargs.pop("speculative_config"), **kwargs)
    elif check == "trajectory":
        profiles = collect_trajectory_two_models(
            baseline, candidate, tokenizer, prompts, **kwargs)
        layer = trajectory_layer_result(profiles)
        certified = layer["certified"]
        if not certified and raise_on_fail:
            raise DeploymentBlocked(
                f"certify_safe(trajectory): NOT certified - "
                f"d_min={layer['d_comm']:.3f} < budget={budget}.")
        return certified
    else:
        raise ValueError(
            f"certify_safe: unknown check '{check}'. "
            f"Valid checks: {sorted(_VALID_CHECKS)}")

    layer = _make_layer_result(cos_sims, threshold=budget)
    d = layer["d_comm"]
    certified = d >= budget
    if not certified and raise_on_fail:
        raise DeploymentBlocked(
            f"certify_safe({check}): NOT certified - "
            f"d={d:.3f} < budget={budget}. Deploy blocked.")
    return certified


_VALID_CHECKS = {
    "weight_quant", "kv_cache_quant", "kv_transfer", "activation_quant",
    "gradient_compress", "lora", "prefix_cache", "engine_swap", "model_swap",
    "batch_divergence", "spec_decoding", "sparse_attention", "moe_token_dropping",
    "neuron_skipping", "allreduce_tp", "alltoall_ep", "pipeline_parallel",
    "provider_drift", "prompt_swap", "trajectory",
}


class DeploymentBlocked(RuntimeError):
    """Raised by certify_safe when raise_on_fail=True and d < budget."""


__version__ = "1.2.4"
__all__ = [
    # Main API
    "certify_safe",
    "DeploymentBlocked",
    "certify_system",
    "certify",
    "enforce",
    "load_certificate",
    "check_certified",
    "summary",
    "compose_bounds",
    "CollectionError",

    # Core formula
    "d_comm",
    "divergence_bound",
    "certify_layer",
    "calibrate_layer",

    # Config
    "InferenceConfig",
    "LayerSpec",

    # Layer name constants (all 20)
    "LAYER_ALLREDUCE_TP",
    "LAYER_ALLTOALL_EP",
    "LAYER_PIPELINE_PARALLEL",
    "LAYER_KV_TRANSFER",
    "LAYER_WEIGHT_QUANT",
    "LAYER_KV_CACHE_QUANT",
    "LAYER_ACTIVATION_QUANT",
    "LAYER_GRADIENT_COMP",
    "LAYER_LORA",
    "LAYER_PREFIX_CACHE",
    "LAYER_ENGINE_SWAP",
    "LAYER_BATCH_DIVERGENCE",
    "LAYER_SPEC_DECODING",
    "LAYER_SPARSE_ATTENTION",
    "LAYER_MOE_TOKEN_DROP",
    "LAYER_NEURON_SKIPPING",
    "LAYER_MODEL_SWAP",
    "LAYER_PROVIDER_DRIFT",
    "LAYER_PROMPT_SWAP",
    "LAYER_TRAJECTORY",

    # Collectors (all 20)
    "collect_allreduce_tp",
    "collect_alltoall_ep",
    "collect_pipeline_parallel",
    "collect_pipeline_parallel_tensors",
    "collect_kv_transfer",
    "collect_weight_quant",
    "collect_kv_cache_quant",
    "collect_activation_quant",
    "collect_gradient_compress",
    "collect_lora",
    "collect_prefix_cache",
    "collect_engine_swap",
    "collect_batch_divergence",
    "collect_speculative_decode",
    "collect_sparse_attention",
    "collect_moe_token_dropping",
    "collect_neuron_skipping",
    "collect_model_swap",
    "capture_logits_openai_api",
    "collect_provider_drift",
    "collect_prompt_swap",
    "collect_trajectory",
    "collect_trajectory_two_models",

    # Trajectory helpers
    "d_profile",
    "trajectory_layer_result",

    # Capture / persistence utilities
    "save_logits",
    "load_logits",
    "capture_logits_hf",
    "capture_logits_vllm",
    "certify_from_layers",
]
