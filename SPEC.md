# DeltaCert — Technical Specification

This document defends the claims made in `README.md`. If you're trying to break DeltaCert, start here.

## 1. The formula

Given two versions of a system (baseline and candidate), DeltaCert measures the cosine similarity `c` between their output logit/logprob distributions on a fixed set of prompts, then computes:

```
Δ(c) = 4c√(1-c²)          (commutator magnitude)
d = -log(Δ/2)             (algebraic distance)
divergence_bound = 2·exp(-d)
```

`d` is the certified quantity. A change is certified safe when `d ≥ tau` for some threshold `tau` (default 3.0, or a per-method/per-workload value from `deltacert calibrate`).

## 2. The clamp, and why it exists

`Δ(c) = 4c√(1-c²)` is **not monotone** in `c`. It vanishes both at `c=1` (outputs identical — genuinely good) and at `c=0` (outputs orthogonal — genuinely broken). Without correction, a catastrophically broken change that happens to produce near-orthogonal output distributions would compute `Δ≈0 → d=∞ → CERTIFIED`, which is exactly backwards.

The fix: `Δ(c)` is only used in its monotone regime, `c ≥ 1/√2 ≈ 0.7071`. Below that threshold, DeltaCert fail-closes: `Δ` is clamped to `2.0`, giving `d=0`, which never certifies. Real production measurements across all 7 validated tests in this repo live in `c ∈ [0.95, 1.0]` — comfortably inside the monotone regime; the clamp exists as a safety floor for genuinely broken changes, not as something that fires in normal operation.

Source: `collectors.py`, `_commutator_magnitude()` and the `_C_MIN_VALID = 1/√2` constant.

## 3. Per-method calibration — what's proven, what isn't

DeltaCert ships a default `tau=3.0`. For `weight_quant` specifically, it additionally ships per-method provisional budgets:

```python
_PROVISIONAL_METHOD_BUDGETS = {
    "bnb": 0.5,
    "gptq": 2.148,
}
```

**Where these numbers come from:** two real sweeps, one per model (Llama-3.1-8B-Instruct and Qwen2.5-7B-Instruct), GSM8K downstream accuracy, 6 configs per model (bnb int8, bnb nf4, GPTQ int8, GPTQ int4, GPTQ int3, GPTQ int2 — n=2 bnb + n=4 GPTQ per model). GPTQ is now genuinely two-sided **on each model separately** (τ=3.187 Llama, τ=2.1481 Qwen); bnb remains one-sided, n=2 per model, no damaged config ever observed on either model. Treat these as a firmer-than-before but still not universal calibration.

**Why the shipped `gptq` constant is 2.148, not either model's own calibrated value — read this before trusting the default:** `certify_layer()` indexes `_PROVISIONAL_METHOD_BUDGETS` only by quantization method, never by model — there is no per-model dispatch in the shipped code. Because Llama's and Qwen's own calibrated floors differ (3.187 vs 2.1481), **no single global value can be conservative for both models**: the truly conservative choice (3.187) would flip Qwen's own safe GPTQ int8 reading to unsafe, contradicting Qwen's own calibration. The shipped constant (2.148) is therefore the **permissive** compromise — Qwen's own floor, with a deliberate margin below its display-rounded 2.1481 (see the float-precision note below) — not a conservative one. It stays above every damaged GPTQ reading measured on either model (Llama int4=1.164, Qwen int4=0.825), but **a Llama config scoring between 2.15 and 3.19 would pass this shipped default while failing Llama's own calibration.** This is not a bug to be silently trusted around; it is the direct, load-bearing reason `deltacert calibrate` is not optional for production use of GPTQ certification.

**Float-precision note:** Qwen's calibrated GPTQ floor displays as `2.1481` at 4-decimal precision, and its stored config (`gptq_int8`) has `d_comm` exactly equal to that floor by construction (zero margin — the floor IS the lowest safe reading). Inverting the same result's higher-precision `divergence_bound` field recovers `d ≈ 2.148102`, only ~1.6×10⁻⁶ above `2.1481` — close enough to the display-rounding boundary that a bare `d >= 2.1481` risked failing Qwen's own reference config on floating-point noise, depending on exactly how the raw (unrounded, unstored) value falls. The shipped constant is `2.148` — one fewer digit, a deliberate safety margin below the true value, confirmed against every stored GPTQ reading on both models (see `_PROVISIONAL_METHOD_BUDGETS`'s code comment for the four-point check).

**Known gaps in the current calibration:**
- `bnb`'s floor (0.5) passes both of the only two bnb readings that exist per model (int8=1.153/1.332, nf4=0.578/0.615, Llama/Qwen) — but no deliberately-broken bnb config has ever been measured on either model, so there is no evidence-backed failure boundary for bnb specifically. The number guarantees today's known-safe readings pass; it is not derived from an observed bnb failure point.
- `gptq`'s floor is now cross-model-verified and two-sided per model, but the shipped global constant is still a single value with no per-model awareness — see the permissive-compromise explanation above. A real per-model-aware default (detecting which model is being certified and picking 3.187 vs 2.148 accordingly) is not implemented.

**The fix for production use:** run `deltacert calibrate` on your own model and workload (see README) rather than trusting these shipped numbers for anything you can't afford to get wrong — this is especially true for GPTQ now that the shipped default is known to be permissive for non-Qwen models. `calibrate_layer()` enforces one rule strictly: every config in a calibration sweep must come with a REAL measured downstream number — it never simulates or estimates one. A config without a real measurement is rejected outright.

Source: `deltacert.py`, `_PROVISIONAL_METHOD_BUDGETS` and its surrounding comment block; `validation_results/weight_quant/` (Llama) and `validation_results/qwen/weight_quant_gptq/` (Qwen).

## 4. Domain stratification — worst-domain, not blended average

When prompts span multiple domains (math, code, multilingual, chat, general), DeltaCert certifies on the **worst-domain** `d_comm`, never a blended average across domains. A severe regression confined to one domain (e.g. code generation specifically) must not be hidden by fine performance in the other four. This is the same statistic used by the trajectory collector (worst position across a generation, not an average), applied here across domains instead of token positions.

This matters concretely: in the trajectory result cited in the README, nf4 passed GSM8K (a math-domain task) while failing on long-form code generation — a domain-specific regression a single blended-average number would have masked.

Source: `deltacert.py`, `certify_layer()`'s `domain_labels` path; `deltacert.py`, `calibrate_layer()`'s `domain_labels` path (added specifically so self-calibration doesn't silently regress to a blended average either).

## 5. The top-k logprobs caveat

Hosted APIs and serving engines don't return full-vocabulary logits — they return the top-k log-probabilities per token (OpenAI caps this at 20; vLLM's engine-level cap is also 20 by default in this codebase). DeltaCert's certified quantity in these cases is explicitly the **top-k log-distribution**, embedded into a fixed-size dense vector via token-string hashing (since hosted APIs return token strings, not fixed vocabulary indices). Both sides of any comparison always use the same `k`, so the certified quantity is well-defined — but it is a top-k measurement, not a full-vocabulary one. A change that only affects probability mass outside the top 20 tokens would not be visible to this measurement.

Source: `collectors.py`, `capture_logits_openai_api()` and `capture_logits_vllm()` docstrings.

## 6. Trajectory: two independent signals, not one

The trajectory check produces two numbers that measure genuinely different things, and they are not expected to agree. **Neither field name below is part of the installed package's certificate schema** — `collect_trajectory_two_models()` (the shipped, reusable API) returns per-position `d_comm` profiles and a `d_min_position_in_worst_trajectory` summary; `safe_until_token` and `failure_after_token` are a specific downstream analysis performed on that profile by the paper's own experiment script (`validation/trajectory/run_flagship.py`, not shipped with `pip install deltacert` — see its `packages` list in `pyproject.toml`), reproducible by any caller but not automatically computed by the library:

- **`safe_until_token`** — the earliest token position, across all measured trajectories, where the `d_comm` math bound (vs. `tau`) first crosses into "unsafe." Derivable from the commutator-distance profile the shipped API already returns.
- **`failure_after_token`** — the median token position where the actual generated text from the baseline and candidate models first diverges. An independent, empirical measurement — real generations compared directly, not derived from the `d(t)` profile at all.

In the README's headline result, `failure_after_token (14) < safe_until_token (31)`: real text forked *before* the math bound alone would have flagged it. This is not a contradiction — it means the empirical signal caught the regression earlier than the theoretical bound would have on its own, which is exactly the kind of result that justifies running both measurements rather than either alone.

Source: `collectors.py`, `collect_trajectory_two_models()` (the shipped `d_comm` profile); `validation/trajectory/run_flagship.py` (the paper-specific `safe_until_token`/`failure_after_token` analysis on top of it — see `validation_results/trajectory/result.json` for its output, distinct from `cert_trajectory.json`).

**Robustness to reference quality.** A fraction of the frozen fp16 reference continuations exhibit greedy-decoding repetition artifacts (degenerate loops) unrelated to quantization. To test whether this drove the result, trajectory certification was re-run excluding all 7 references confirmed under manual inspection to exhibit degenerate repetition (`HumanEval/1, 2, 8, 13, 16, 17, 49`). On the remaining 43 clean references, every statistic was unchanged: `d_min` at floor (0.0), the global minimum at the same position (443), `safe_until_token` at the same value (31), and `failure_after_token` still 14 — identical to the full 50-reference result. The finding is not an artifact of reference-quality degradation.

Source: `validation_results/trajectory/cert_trajectory_clean7.json`, `result_clean7.json` (43-reference re-run); `cert_trajectory.json`, `result.json` (original 50-reference run) for comparison. Exclusion list reconciled from two independent reviews: an automated repetition-loop detector (caught exact token-for-token loops) and manual inspection (caught incrementing-variant repetition the detector missed) — see both cert files' `excluded_task_ids` for provenance.

## 7. What "certified" means, precisely

**"Certified safe" always means "the measured `d_comm` is at or above a calibrated threshold `tau`."** It never means "guaranteed harmless" or "proven equivalent." `tau` is either DeltaCert's shipped reference calibration (itself an initial n=5 calibration, see §3) or a threshold you calibrated yourself on your own model and workload via `deltacert calibrate`. A "safe" verdict is only as trustworthy as the calibration behind it — this is why the CLI prints a reminder every time it uses the shipped default rather than a self-calibrated one.

## 8. No-simulation rule

Every collector in `collectors.py` and every function in `deltacert.py`/`cli.py` that could accept a downstream metric enforces one rule: **real measurements only, hard failure on missing or empty data, never a fabricated or estimated substitute.** Concretely:
- `d_comm()` raises on an empty `cos_sims` list rather than returning a default.
- `calibrate_layer()` raises if any swept config is missing a real `downstream_drop_pts` value.
- `collect_provider_drift()` raises rather than certifying a hosted model against itself on day one, when no real baseline yet exists.

This rule is why some results in the README are incomplete rather than filled in with a plausible-looking placeholder (e.g. the calibration table's "caught what evals missed" column is empty for 6 of 7 tests — only trajectory has produced a fully populated example so far).
