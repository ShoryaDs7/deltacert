# DeltaCert

[![PyPI](https://img.shields.io/pypi/v/deltacert)](https://pypi.org/project/deltacert/) [![License](https://img.shields.io/pypi/l/deltacert)](LICENSE) [![Python](https://img.shields.io/pypi/pyversions/deltacert)](https://pypi.org/project/deltacert/)

**DeltaCert certifies changes to an LLM serving stack — quantization, engine upgrades, batch size, model updates — against a calibrated divergence bound, before deployment.** It measures whether a change altered model behavior, per workload domain and per token position, and catches two failure classes that standard benchmarks and single-pass checks miss: long-generation forking that short-form evals score as unchanged, and feedback-driven collapse that passes every logit-level check and appears only in the decoded output. Validated verdicts include fp8 KV-cache (2× batch capacity), batch-64 serving, and same-week engine upgrades certified safe on Llama-3.1-8B — and the identical fp8 KV-cache flag flagged as catastrophic on Qwen2.5-7B.

Built by [Threvo Labs](https://pypi.org/user/Shorya/).

[PyPI](https://pypi.org/project/deltacert/) · [SPEC.md](SPEC.md) · [LICENSE](LICENSE)

```bash
pip install deltacert
```

---

## Two failure classes standard checks miss

**Failure class 1: benchmark-blind forking.**

| Config | Short eval said | d_COMM | d(t) threshold crossing (token) | median greedy fork (token) | Verdict |
|---|---|---|---|---|---|
| Llama-3.1-8B fp16 → nf4 | GSM8K 5-shot exact-match +1.0% (looks safe) | 0.00 | 31 | 14 | **unsafe** |
| Qwen2.5-7B fp16 → nf4 | GSM8K exact-match +0.0% (looks safe) | 0.00 | 17 | 18 | **unsafe** |

Same pattern, two model families, two labs, two tokenizers. GSM8K missed both. DeltaCert's trajectory certification caught both — generations fork from the fp16 reference within ~15-18 tokens on long-form coding tasks, well before a short-form benchmark would ever see it.

**Failure class 2: feedback-driven collapse.** A config passes every teacher-forced check and is still destroyed. This failure class is what made us rewrite our own default mode.

| Config | Single-position | Trajectory (30,643 positions) | Downstream reality | Verdict |
|---|---|---|---|---|
| Qwen2.5-7B fp8 KV-cache | safe (d=3.72, cosines ≥0.9998) | safe (d≥3.52 at every position) | GSM8K 0.88 → 0.00, 0/100 correct | **unsafe** |
| Llama-3.1-8B fp8 KV-cache, identical flag | safe | not run† | GSM8K within noise | genuinely safe |

†Trajectory certification was only run on the destroyed Qwen config, to see whether it caught what single-position missed (it didn't). Llama's "genuinely safe" verdict rests on single-position certification plus the free-running collector below — not on an unmeasured trajectory pass. The identical engine flag is benign on Llama and catastrophic on Qwen. Both teacher-forced modes tried on Qwen — single-position and full trajectory — certify the collapse safe, because the damage doesn't live in the logits; it lives in the autoregressive feedback loop, which teacher forcing structurally can't see. Catching this needed a third instrument: a free-running collector that runs the deployed decode policy on both engines and measures the actual output process, with a McNemar-exact-test guard so a benign fork's ordinary spontaneous-repetition rate can't be mistaken for caused collapse. It fires decisively on Qwen (79% excess degeneration, p≈10⁻¹⁰) and stays quiet on Llama (2.3%, p=0.63) — sensitivity on the real failure, specificity on the real clean case, same instrument, same thresholds.

Robustness-checked: excluding all 7 references with degenerate repetition, every trajectory statistic is identical (`cert_trajectory_clean7.json`); the fp8-KV single-position measurement was independently reproduced on a different host/stack to within 10⁻³ (the rerun is the archived run of record; the initial run's certificate was superseded in place and is not retained — see the supplementary bookkeeping note).

Reproduce it yourself:

```bash
deltacert generate-cases --model meta-llama/Llama-3.1-8B-Instruct --output cases.jsonl
deltacert certify --model meta-llama/Llama-3.1-8B-Instruct --quantization int4 \
    --checks trajectory --trajectory-cases cases.jsonl
```

Every number above traces to a real certificate in `validation_results/`; every row in the table below reproduces with one script.

## Flagship results

Seven flagship tests, each a full before/after comparison on real hardware with a measured downstream benchmark:

| Change | Business gain | d_COMM | Downstream effect | Verdict |
|---|---|---|---|---|
| Llama-3.1-8B batch=1 → batch=64 | 64 concurrent requests, same GPU | 6.21 | GSM8K -1.0 pt | ✅ Safe |
| vLLM 0.8.5 → vLLM 0.9.0 | take the upgrade same-week, not months later | 16.09 | GSM8K -1.0 pt | ✅ Safe |
| KV cache default → fp8 (vLLM native) | 2x concurrent capacity | 4.83 | GSM8K -1.0 pt | ✅ Safe |
| gpt-4o-mini pinned snapshot → current alias | same-day provider-drift check | 6.65 | canary acc +0.0 pt | ✅ Safe |
| Standard decode → speculative decode (k=5) | claimed ~2x throughput | 15.38 | GSM8K +0.0 pt, **measured 0.28x** (slower) | ✅ Safe on quality, not on speed |
| Llama-3.1-8B fp16 → nf4 (W4) | +60% VRAM reduction | 0.00 | forks at token 14 on long generations | ❌ Unsafe |
| Llama-3.1-8B fp16 → GPTQ int4 | +75% VRAM reduction | 1.16 | GSM8K -8.0 pts | ❌ Unsafe |

Five safe, two unsafe. A tool that only ever says "safe" isn't measuring anything — the two unsafe rows above are DeltaCert doing its job.

## How it works

DeltaCert compares output distributions before and after a change. `d_COMM` ("commutator distance," from the operator-algebraic commutator bound it's derived from) is computed from the cosine similarity `c` between two runs:

```
Δ = 4c√(1-c²)        (commutator magnitude)
d = -log(Δ/2)        (algebraic distance)
divergence_bound = 2·exp(-d)
```

`d` is an algebraic distance; `2e⁻ᵈ` is the certified bound on output divergence — deterministic, minutes to compute, checkable at every token position, no eval harness or labeled data required. DeltaCert certifies on the **worst domain**, not a blended average — a change that severely degrades code generation but leaves math untouched won't get averaged away (code is the weakest domain in most weight-quantization configs we've measured; the weakest domain shifts by change type, which is exactly why blending it out would hide the damage).

> **"Certified" throughout this document means:** measured against a calibrated threshold with a stated bound — not a guarantee of downstream quality.

- **Full derivation, clamp behavior, per-method calibration (with sample sizes disclosed), and the top-k logprobs caveat:** see `SPEC.md`.
- **This README asserts. The spec defends. Nothing here is a proof.**

## Getting started

```bash
pip install deltacert

deltacert certify --model meta-llama/Llama-3.1-8B-Instruct --quantization int8
```

For the Python API — certifying multiple compression layers in one offline pass, then enforcing at server startup — see `examples/example_certify.py`. Run it with `python examples/example_certify.py` from the repo root (its calibration data is illustrative, not measured — replace with your own cosine similarities before trusting the output).

That uses DeltaCert's shipped reference calibration — from the weight-quantization sweeps, not the 7 flagship tests above. The shipped default is a single global value per method (not per-model — the code has no per-model dispatch): GPTQ is two-sided on **each** model separately (τ=3.19 Llama, τ=2.15 Qwen — per the paper, thresholds never transfer across models).

Because no single global value can be conservative for both (the truly conservative value, 3.19, would flip Qwen's own safe GPTQ int8 to unsafe), the shipped default (2.148, with a deliberate safety margin below Qwen's exact 2.1481) is the **permissive** compromise, not a conservative one — it stays above every damaged GPTQ reading measured on either model, but a Llama config scoring between 2.15 and 3.19 would pass this shipped default while failing Llama's own calibration. bnb remains one-sided and provisional (τ=0.5, no damaged bnb config has ever been observed on either model). This is exactly why calibrating per model isn't optional for production — run the sweep yourself:

```bash
deltacert capture --model your-model --output baseline.npz
deltacert capture --model your-model --quantization int8 --output candidate.npz
deltacert calibrate --baseline baseline.npz --candidates candidate.npz \
    --names int8 --method-families bnb --downstream-file your_evals.json
```

`your_evals.json` maps name to downstream drop in points, e.g. `{"int8": 0.0}`.

`deltacert certify` always tells you when it's using the shipped calibration instead of your own.

For KV-cache and any change whose damage can be feedback-driven, add a free-running check (separate subcommand — its certificate is McNemar/degeneration-based, not `d_COMM`-based):

```bash
deltacert free-running --model your-model --kv-cache-dtype fp8 --output cert_free_running.json
```

## Production checklist (day-1 rollout)

1. **Capture** baselines on *your* model and *your* workload prompts (`deltacert capture`) — shipped defaults are provisional floors derived from our models, never yours.
2. **Calibrate** per method family with your own downstream evals (`deltacert calibrate`) — thresholds don't transfer across models, and one-sided calibrations stay labeled provisional until your sweep has both safe and damaged readings.
3. **Know which threshold governs which check**: weight-quant uses your per-family τ; KV-cache and trajectory certification use the single-position/trajectory τ (default 3.0); the free-running check has its own McNemar/degeneration thresholds, independent of `d_COMM`. Every certificate states which threshold it was judged against and whether it came from your calibration or a shipped default — you never have to guess.
4. **Gate CI** on the certificate (`cicd_hook` — exit 1 blocks the pipeline).
5. **Enforce at serving** with the vLLM plugin (`DELTACERT_ENFORCE=1`; complete no-op otherwise).
6. **For KV-cache or any feedback-risk change**, add the free-running check — teacher-forced certification alone is insufficient for that class (see failure class 2 above).

## Integrations

- **vLLM plugin** — official `vllm.general_plugins` entry point, already wired in this package. Opt-in only: complete no-op unless `DELTACERT_ENFORCE=1` is set, so `pip install deltacert` is safe in a shared image; when enabled, a serving engine refuses to start on an uncertified change.
- **CI/CD gate** — `python -m deltacert.integrations.cicd_hook --cert ./cert.json` exits 1 (blocks the pipeline) if not certified, 0 if certified. Works with GitHub Actions, GitLab CI, Jenkins, or any CI that checks exit codes.
- **HuggingFace auto-wiring** — `from deltacert.integrations.hf_integration import auto_certify` picks the right collectors for you from what's active in your config (quantization, LoRA, prefix cache) instead of calling `certify_system()` with raw parameters yourself. Weight-quantization checks resolve their threshold through the same per-family calibration machinery the core uses (bnb vs. GPTQ resolved automatically, not one flat default for every method) — pass your own calibrated `budget` if you have one; otherwise it falls back to the shipped provisional per-family default and prints an explicit warning that it did so. Supported `quantization` values are `int8`/`int4` (the bitsandbytes backend this wrapper uses); an unsupported value raises rather than silently no-opping — earlier versions let `"fp8"` fall through to comparing a model against an unquantized reload of itself and certifying a change that was never applied.

## Verifying a certificate

Optional — certificates work unsigned; signing adds tamper-evidence for sharing certs across teams or with auditors.

```bash
deltacert keygen --private-key mykey.pem --public-key mykey.pub
deltacert sign --cert cert.json --key-file mykey.pem
deltacert verify --cert cert.json --key-file mykey.pub
```

`verify` exits 0 if the signature is valid, 1 if the certificate was modified after signing or signed by a different key. Every certificate also carries a `validation_status` field (`flagship_validated` vs `implemented_pending_validation`) — signed as part of the payload, so a signature can never make an unvalidated collector's result look more trustworthy than it is.

Keep your private key secret — never commit it, never share it. Only the public key is meant to be distributed.

The original 13 flagship reference certificates in `validation_results/` are signed with Threvo's key (`deltacert-public.pem`, committed in this repo); newer research results (the Qwen replication, GPTQ 8-bit extension, free-running collector runs) are not yet signed — hash-verified during collection instead, per this repo's commit history. Don't take our numbers on faith — check them yourself:

```bash
deltacert verify --cert validation_results/weight_quant/cert_nf4.json --key-file deltacert-public.pem
```

**Validating a certificate's shape.** `deltacert-schema.json` (committed in this repo) is the JSON Schema every certificate — signed or not — conforms to. Useful if you're parsing certificates in your own tooling and want to fail fast on a malformed one, independent of signature verification. Requires `pip install jsonschema` (not a DeltaCert dependency, since only this optional check needs it):

```python
import json, jsonschema
schema = json.load(open("deltacert-schema.json"))
cert = json.load(open("cert.json"))
jsonschema.validate(cert, schema)  # raises jsonschema.ValidationError if malformed
```

## What it certifies

All 21 collectors described in the design ship as implemented code; validation status is tracked per collector below. 8 have flagship validation results behind them so far (the tables above, including the free-running collector that catches feedback-driven failures teacher-forced checks miss); the rest run the same math but haven't been through an end-to-end validation pass yet.

| # | Check | CLI-drivable | Status |
|---|---|---|---|
| 1 | `weight_quant` | ✅ | ✅ validated |
| 2 | `kv_cache_quant` | ✅ | ✅ validated |
| 3 | `batch_divergence` | ✅ (needs vLLM) | ✅ validated |
| 4 | `spec_decoding` | ✅ (needs vLLM) | ✅ validated |
| 5 | `engine_swap` | ✅ | ✅ validated |
| 6 | `provider_drift` | ✅ | ✅ validated |
| 7 | `trajectory` | ✅ | ✅ validated |
| 8 | `free_running` | ✅ (needs vLLM) | ✅ validated |
| 9 | `activation_quant` | ✅ | 🔬 implemented — validation run pending |
| 10 | `prefix_cache` | ✅ | 🔬 implemented — validation run pending |
| 11 | `lora` | ✅ | 🔬 implemented — validation run pending |
| 12 | `model_swap` | ✅ | 🔬 implemented — validation run pending |
| 13 | `prompt_swap` | ✅ | 🔬 implemented — validation run pending |
| 14 | `sparse_attention` | Python API only | 🔬 implemented — validation run pending |
| 15 | `moe_token_dropping` | Python API only | 🔬 implemented — validation run pending |
| 16 | `neuron_skipping` | Python API only | 🔬 implemented — validation run pending |
| 17 | `allreduce_tp` | Python API only | 🔬 implemented — validation run pending |
| 18 | `alltoall_ep` | Python API only | 🔬 implemented — validation run pending |
| 19 | `pipeline_parallel` | Python API only | 🔬 implemented — validation run pending |
| 20 | `kv_transfer` | Python API only | 🔬 implemented — validation run pending |
| 21 | `gradient_compress` | Python API only | 🔬 implemented — validation run pending |

"Python API only" means the check needs code you supply (a custom `compress_fn`, attention mask, etc.) — the CLI can't conjure that for you; see `import deltacert as dc; dc.certify_system(...)`.

## Limitations (stated up front, not discovered by you later)

- Validated on two open model families (Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct) plus one hosted API (gpt-4o-mini); serving-time flagships beyond weight/KV-cache quantization are validated on Llama only.
- Per-method calibration is not a settled constant: GPTQ is two-sided on each model separately (n=4 configs per model; τ=3.19 Llama, τ=2.15 Qwen). bnb is one-sided on both models (n=2 configs per model, no damaged config observed on either). The shipped GPTQ default (2.148) is a single global value because the code has no per-model dispatch — and since no single global value can be conservative for both models (3.19 would wrongly flip Qwen's own safe GPTQ int8 to unsafe), it's the permissive compromise, not the conservative one: a Llama config scoring between 2.15 and 3.19 would pass the shipped default while failing Llama's own calibration. This is consistent with the paper's position that cross-model threshold transfer is unsupported — it's exactly why calibrating per model isn't optional for production. Run `deltacert calibrate` on your own model/workload rather than trusting either shipped default for anything production-critical.
- The provider_drift result above is a same-day proxy (pinned snapshot vs. current alias), not the real weekly-cadence drift measurement, which needs two runs across real time.
- `d_comm` is a reliable within-method damage indicator but is not directly comparable across different compression methods — see `SPEC.md` for the bnb-vs-GPTQ false-negative this caused and how it's handled.
- The feedback-driven failure class (fp8 KV-cache on Qwen) currently has one confirmed member after a pre-registered four-candidate hunt; whether other configurations populate it is still open. That hunt's classification rule isn't a rubber stamp: it fired on real data, catching an int4 KV-cache config with the largest free-running signal we've measured and correctly reclassifying it as ordinary context-independent damage (it also fails single-position certification) rather than a second instance of the class. A preliminary, unsigned local run on Qwen2.5-3B-Instruct (fp8 KV-cache, single-position certified safe, GSM8K 0.69→0.56) shows the same signature but wasn't run through the trajectory/free-running corroboration the paper requires before calling something a confirmed instance — see `validation_results/qwen3b_kv_fp8_local/NOTES.md`.

## Roadmap

- More models, more downstream tasks — firming up bnb's one-sided calibration (n=2 per model) the way GPTQ's was made two-sided (n=4 per model); a real per-model-aware shipped default (rather than today's single permissive-compromise global value) is also worth doing
- Full validation pass on the remaining 13 collectors
- Real weekly-cadence provider_drift run (beyond the same-day proxy)
- Cross-backend certification: extend `capture` to TensorRT-LLM / SGLang (comparison logic is already backend-agnostic)

## Citation

If you use DeltaCert, cite it as:

```bibtex
@software{deltacert2026,
  title  = {DeltaCert: Calibrated Divergence Certification for LLM Serving Systems},
  author = {Dev Shorya},
  year   = {2026},
  url    = {https://pypi.org/project/deltacert/}
}
```

## License

Apache-2.0. See [LICENSE](LICENSE).

## Contact

- Issues and questions: open a GitHub issue
- Full validation data: `validation_results/` in this repo
