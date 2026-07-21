"""
example_certify.py — DeltaCert: full offline certification for a deployment

Shows how to certify all 10 application areas in one pass.
Run this once before deployment. Save the JSON. Load it at server startup.

Offline phase: YOU run this with your calibration data.
Runtime phase: server loads JSON, calls dc.enforce(), zero math.
"""

import deltacert as dc

# ─── Step 1: Define what is compressed in your stack ─────────────────────────

config = dc.InferenceConfig(
    model="meta-llama/Llama-3-70B",
    description="int4 weight quant + 8bit KV cache + 4xTP AllReduce",
    layers=[
        # Infrastructure compression (wire) — items 1-4
        dc.LayerSpec(dc.LAYER_ALLREDUCE_TP,      budget=3.0),
        dc.LayerSpec(dc.LAYER_ALLTOALL_EP,        budget=3.0, enabled=False),  # not using EP
        dc.LayerSpec(dc.LAYER_PIPELINE_PARALLEL,  budget=3.0, enabled=False),  # not using PP
        dc.LayerSpec(dc.LAYER_KV_TRANSFER,        budget=3.0, enabled=False),  # not using disagg

        # Model-level compression — items 5-9 (Big deal)
        dc.LayerSpec(dc.LAYER_WEIGHT_QUANT,       budget=3.0),
        dc.LayerSpec(dc.LAYER_KV_CACHE_QUANT,     budget=3.0),
        dc.LayerSpec(dc.LAYER_ACTIVATION_QUANT,   budget=3.0, enabled=False),
        dc.LayerSpec(dc.LAYER_GRADIENT_COMP,      budget=3.0, enabled=False),  # training only
        dc.LayerSpec(dc.LAYER_LORA,               budget=3.0, enabled=False),  # full model here

        # Caching — item 10
        dc.LayerSpec(dc.LAYER_PREFIX_CACHE,       budget=3.0),
    ]
)

# ─── Step 2: Provide calibration data ────────────────────────────────────────
# Each entry is a list of cosine_similarity(original_output, compressed_output)
# measured on your calibration dataset (e.g., 512 prompts from your workload).
#
# How to get cos_sims:
#   import torch, torch.nn.functional as F
#   cos_sim = F.cosine_similarity(logits_full.flatten(), logits_compressed.flatten(), dim=0).item()

calibration = {
    # Wire compression
    dc.LAYER_ALLREDUCE_TP: [
        0.9999, 0.9998, 0.9997, 0.9999, 0.9998,
        # ... 512 prompts from your calibration set
    ],

    # int4 weight quantization — measured end-to-end logit similarity
    dc.LAYER_WEIGHT_QUANT: [
        0.9997, 0.9996, 0.9995, 0.9997, 0.9998,
    ],

    # 8-bit KV cache quantization
    dc.LAYER_KV_CACHE_QUANT: [
        0.9994, 0.9995, 0.9996, 0.9993, 0.9995,
    ],

    # Prefix cache — cosine similarity of logits for prompts with shared prefix
    dc.LAYER_PREFIX_CACHE: [
        1.0, 1.0, 1.0, 0.9999, 1.0,
    ],
}

# ─── Step 3: Certify + save ───────────────────────────────────────────────────

cert = dc.certify(
    config=config,
    calibration=calibration,
    output_path="./llama3_70b_int4_tp4_certified.json",
)

print(dc.summary(cert))
print()
if cert["certified"]:
    print("Certified. Safe to deploy.")
else:
    print("NOT certified. Reduce compression ratio or collect more calibration data.")

# ─── Server startup (in your vLLM / inference server entry point) ─────────────

# import deltacert as dc
# dc.enforce("./llama3_70b_int4_tp4_certified.json")
# # certified=true  → continues, serves normally
# # certified=false → raises RuntimeError, refuses to start
