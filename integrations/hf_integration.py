"""
integrations/hf_integration.py — DeltaCert auto-wiring for HuggingFace models

Auto-runs the right collectors based on what's active in the model config.
Companies pass their model + config — DeltaCert wires everything automatically.

Usage:
    from deltacert.integrations.hf_integration import auto_certify

    cert = auto_certify(
        model_name="meta-llama/Llama-3.1-8B",
        quantization="int8",          # or "int4", "fp8", None
        use_lora=False,
        lora_path=None,
        use_prefix_cache=True,
        calibration_prompts=prompts,  # list of strings
        output_path="./cert.json",
    )
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import Optional
import deltacert as dc


def auto_certify(
    model_name: str,
    calibration_prompts: list,
    quantization: Optional[str] = None,
    use_lora: bool = False,
    lora_path: Optional[str] = None,
    use_prefix_cache: bool = False,
    shared_prefix: Optional[str] = None,
    use_kv_cache_quant: bool = False,
    budget: float = 3.0,
    output_path: Optional[str] = None,
    device: str = "cuda",
) -> dict:
    """
    Auto-wires DeltaCert collectors for a HuggingFace model.
    Detects what's active and runs the right collectors automatically.

    Args:
        model_name:          HuggingFace model ID or local path
        calibration_prompts: list of strings (128-512 recommended)
        quantization:        "int8", "int4", "fp8", or None (fp16 baseline)
        use_lora:            whether LoRA adapter is being used
        lora_path:           path to LoRA adapter (required if use_lora=True)
        use_prefix_cache:    whether prefix caching is active
        shared_prefix:       prefix string for prefix cache test
        use_kv_cache_quant:  whether KV cache is quantized
        budget:              minimum d_COMM to certify (default 3.0)
        output_path:         path to save certificate JSON
        device:              cuda device string

    Returns:
        certificate dict
    """
    from deltacert.collectors import (
        collect_weight_quant, collect_lora, collect_prefix_cache,
        collect_kv_cache_quant,
    )

    # Hard-failure semantics (paper Contribution 2, SPEC.md): any condition
    # that would yield an untrustworthy measurement raises an error rather
    # than a default. quantization="fp8" used to fall through to
    # bnb_config=None -- silently comparing fp16 against an unquantized
    # reload and certifying a change that was never applied. Whitelist the
    # methods this backend can actually apply; anything else raises instead
    # of defaulting to a no-op.
    _SUPPORTED_QUANTIZATIONS = {"int8", "int4"}
    if quantization is not None and quantization not in _SUPPORTED_QUANTIZATIONS:
        raise ValueError(
            f"quantization={quantization!r} is not supported by the "
            "bitsandbytes backend this wrapper uses; auto_certify supports "
            f"{sorted(_SUPPORTED_QUANTIZATIONS)} (bnb) or None (no weight "
            "quantization). For fp8 KV-cache certification use the "
            "kv_cache_quant collector through vLLM (--kv-cache-dtype fp8); "
            "for fp8 weight formats, certify through an engine that "
            "actually implements them -- this wrapper must not silently "
            "compare a model against an unquantized reload of itself."
        )

    print(f"DeltaCert: loading baseline fp16 model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_fp16 = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map=device
    )
    model_fp16.eval()

    calibration = {}
    layers = []

    # ── Item 5: Weight quantization ───────────────────────────────────────────
    if quantization in _SUPPORTED_QUANTIZATIONS:
        print(f"DeltaCert: collecting weight_quantization cos_sims ({quantization})...")

        if quantization == "int8":
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        else:  # quantization == "int4" -- only other member of _SUPPORTED_QUANTIZATIONS
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )

        model_quant = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=device,
        )
        model_quant.eval()

        cos_sims = collect_weight_quant(
            model_fp16, model_quant, tokenizer, calibration_prompts, device
        )
        calibration[dc.LAYER_WEIGHT_QUANT] = cos_sims
        layers.append(dc.LayerSpec(
            dc.LAYER_WEIGHT_QUANT, budget=budget, quant_method="bnb",
        ))
        if budget == 3.0:
            print(
                "DeltaCert: using the shipped provisional bnb threshold "
                "(see deltacert.py's _PROVISIONAL_METHOD_BUDGETS), not a "
                "threshold calibrated on your own model/workload. This is "
                "a starting point, not a production guarantee -- run "
                "deltacert calibrate on your own sweep before relying on "
                "this in production."
            )

        del model_quant
        torch.cuda.empty_cache()

    # ── Item 6: KV cache quantization ─────────────────────────────────────────
    if use_kv_cache_quant:
        print("DeltaCert: collecting kv_cache_quantization cos_sims...")

        def compress_fn(t):
            scale = t.abs().max() / 127.0 + 1e-8
            return (t / scale).round().clamp(-127, 127).to(torch.int8), scale

        def decompress_fn(packed):
            q, scale = packed
            return q.to(torch.float16) * scale

        cos_sims = collect_kv_cache_quant(
            model_fp16, tokenizer, calibration_prompts,
            compress_fn, decompress_fn, device,
        )
        calibration[dc.LAYER_KV_CACHE_QUANT] = cos_sims
        layers.append(dc.LayerSpec(dc.LAYER_KV_CACHE_QUANT, budget=budget))

    # ── Item 9: LoRA vs full model ────────────────────────────────────────────
    if use_lora and lora_path:
        print(f"DeltaCert: collecting lora cos_sims from {lora_path}...")
        cos_sims = collect_lora(
            model_fp16, tokenizer, calibration_prompts, lora_path, device
        )
        calibration[dc.LAYER_LORA] = cos_sims
        layers.append(dc.LayerSpec(dc.LAYER_LORA, budget=budget))

    # ── Item 10: Prefix cache ─────────────────────────────────────────────────
    if use_prefix_cache and shared_prefix:
        print("DeltaCert: collecting prefix_cache cos_sims...")
        cos_sims = collect_prefix_cache(
            model_fp16, tokenizer, calibration_prompts[:64],
            shared_prefix, device,
        )
        calibration[dc.LAYER_PREFIX_CACHE] = cos_sims
        layers.append(dc.LayerSpec(dc.LAYER_PREFIX_CACHE, budget=budget))

    if not layers:
        raise ValueError(
            "No compression layers detected. Pass quantization, use_lora, "
            "use_prefix_cache, or use_kv_cache_quant."
        )

    config = dc.InferenceConfig(
        model=model_name,
        layers=layers,
        description=f"auto_certify: quantization={quantization}, lora={use_lora}",
    )

    print("DeltaCert: running certification...")
    cert = dc.certify(config, calibration, output_path=output_path)
    print(dc.summary(cert))

    # Sanity guard: a weight-quant d_comm this high has never been observed
    # on a real, actually-applied bnb quantization in this project's own
    # data (highest measured: ~1.9). A reading far above that is a stronger
    # signal that the comparison silently didn't apply a real change than
    # that this run found an unusually gentle quantization -- print a
    # warning rather than let a suspiciously "too safe" result pass quietly.
    wq_layer = cert.get("layers", {}).get(dc.LAYER_WEIGHT_QUANT)
    if wq_layer and isinstance(wq_layer.get("d_comm"), (int, float)) and wq_layer["d_comm"] > 2.5:
        print(
            f"DeltaCert: WARNING - weight_quantization d_comm={wq_layer['d_comm']} "
            "is atypically high for a real bnb quantization (highest "
            "observed on this project's own data: ~1.9). Verify the "
            "quantized model was actually loaded with the intended config "
            "before trusting this certificate."
        )

    del model_fp16
    torch.cuda.empty_cache()

    return cert
