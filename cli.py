"""
cli.py — DeltaCert CLI

Primary interface. Maps flags → certify_system() → cert.json.
Does not contain measurement code. All measurement lives in collectors.py.

Usage:
    # Zero-decision first run — sensible defaults for int8
    deltacert certify --model meta-llama/Llama-3.1-8B --quantization int8

    # Explicit checks
    deltacert certify --model meta-llama/Llama-3.1-8B --quantization int8 \
        --checks weight_quant batch_divergence prefix_cache \
        --output ./cert.json

    # Engine swap — capture baseline in env A, certify in env B
    deltacert capture --model meta-llama/Llama-3.1-8B --output baseline.npz
    deltacert certify --model meta-llama/Llama-3.1-8B \
        --checks engine_swap --baseline baseline.npz

    # Check an existing certificate
    deltacert check --cert ./cert.json

    # Print certificate summary
    deltacert summary --cert ./cert.json

    # Trajectory — freeze real continuations once, then certify against them
    deltacert generate-cases --model meta-llama/Llama-3.1-8B --output cases.jsonl
    deltacert certify --model meta-llama/Llama-3.1-8B --quantization int4 \
        --checks trajectory --trajectory-cases cases.jsonl

    # Free-running — feedback-driven failures invisible to single-position and
    # trajectory certification alike (e.g. KV-cache collapse); separate
    # subcommand because its certificate is McNemar/degeneration-based, not
    # d_COMM-based
    deltacert free-running --model Qwen/Qwen2.5-7B-Instruct --kv-cache-dtype fp8 \
        --output cert_free_running.json

Self-drivable checks (CLI loads models internally):
    weight_quant      → loads fp16 + quantized model
    kv_cache_quant    → loads fp16, applies default int8 KV quant
    activation_quant  → loads fp16, applies default int8 activation quant
    batch_divergence  → requires vLLM; uses --model as engine
    prefix_cache      → loads fp16; uses --prefix (or default system prompt)
    spec_decoding     → requires vLLM + --draft-model
    lora              → loads fp16; requires --lora-path
    engine_swap       → requires --baseline <path.npz> from deltacert capture
    model_swap        → requires --candidate <path.npz>
    provider_drift    → requires --drift-baseline <path.npz>, --api-key
    prompt_swap       → requires --prompt-a and --prompt-b
    trajectory        → requires --quantization + --trajectory-cases <file.jsonl>
                        from `deltacert generate-cases`

Not self-drivable (use Python API — CLI cannot conjure your code):
    sparse_attention    → needs your attention mask
    moe_token_dropping  → needs your capacity setter
    neuron_skipping     → needs your pruning kernel
    allreduce_tp        → needs your compress_fn
    alltoall_ep         → needs your compress_fn
    pipeline_parallel   → needs your stage boundary index
    kv_transfer         → needs your compress_fn
    gradient_compress   → needs your compress_fn
"""

import argparse
import json
import os
import sys
import tempfile

import deltacert as dc
from deltacert import signing as dsign
from deltacert.deltacert import _build_metadata
from deltacert.collectors import (
    CollectionError,
    capture_logits_hf,
    collect_free_running_vllm,
    load_logits,
    save_logits,
)


# ─────────────────────────────────────────────────────────────────────────────
# Default check sets per quantization config
# Zero-decision first run: --quantization int8 → sensible defaults
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULTS = {
    "int8": ["weight_quant", "kv_cache_quant", "prefix_cache"],
    "int4": ["weight_quant", "kv_cache_quant", "prefix_cache"],
    "fp8":  ["weight_quant", "prefix_cache"],
    "none": ["prefix_cache"],
}

_NEEDS_VLLM = {"batch_divergence", "spec_decoding"}

_SELF_DRIVABLE = {
    "weight_quant", "kv_cache_quant", "activation_quant",
    "batch_divergence", "prefix_cache", "spec_decoding",
    "engine_swap", "lora",
    "model_swap", "provider_drift", "prompt_swap", "trajectory",
}

_NEEDS_PYTHON_API = {
    "sparse_attention", "moe_token_dropping", "neuron_skipping",
    "allreduce_tp", "alltoall_ep", "pipeline_parallel",
    "kv_transfer", "gradient_compress",
}

# Checks that require specific CLI flags — validated before any model loading.
# Missing flag → hard fail + no certificate. Never silently skip.
_REQUIRES_FLAG = {
    "spec_decoding":   ("--draft-model",    "draft_model"),
    "lora":            ("--lora-path",      "lora_path"),
    "engine_swap":     ("--baseline",       "baseline"),
    "provider_drift":  ("--drift-baseline", "drift_baseline"),
    "trajectory":      ("--trajectory-cases", "trajectory_cases_file"),
}

# Checks that require more than one flag — checked separately.
_REQUIRES_FLAGS_MULTI = {
    "prompt_swap": [("--prompt-a", "prompt_a"), ("--prompt-b", "prompt_b")],
    # model_swap needs BOTH a baseline capture and a candidate capture
    # (collect_model_swap(capture_a_path, candidate_path) in deltacert.py)
    # -- --candidate alone used to pass the single-flag check while
    # capture_a_path stayed None, crashing deep in load_logits with a raw
    # TypeError instead of the clean pre-flight message every other
    # multi-input check gets. Found via live CLI testing.
    "model_swap": [("--baseline", "baseline"), ("--candidate", "candidate")],
}


# ─────────────────────────────────────────────────────────────────────────────
# Model loading helpers — CLI owns this, collectors never do
# ─────────────────────────────────────────────────────────────────────────────

_CLI_SUPPORTED_QUANTIZATIONS = {"int8", "int4"}


def _load_model_and_tokenizer(model_name: str, quantization: str = None, device: str = "cuda"):
    """Load HuggingFace model with optional BnB quantization. CLI-internal only."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    # Hard-failure semantics (paper Contribution 2, SPEC.md): any condition
    # that would yield an untrustworthy measurement raises an error rather
    # than a default. quantization="fp8" used to fall through to
    # bnb_config=None -- silently loading a plain fp16 model and certifying
    # a change that was never applied. Whitelist what this backend can
    # actually apply; anything else raises instead of defaulting to a no-op.
    if quantization is not None and quantization not in _CLI_SUPPORTED_QUANTIZATIONS:
        print(
            f"[DeltaCert] ERROR: quantization={quantization!r} is not "
            "supported by the bitsandbytes backend this CLI uses; "
            f"supported values are {sorted(_CLI_SUPPORTED_QUANTIZATIONS)}. "
            "For fp8 KV-cache certification use --checks kv_cache_quant "
            "--kv-cache-dtype fp8 (vLLM); for fp8 weight formats, certify "
            "through an engine that actually implements them -- this CLI "
            "must not silently compare a model against an unquantized "
            "reload of itself."
        )
        sys.exit(1)

    print(f"[DeltaCert] Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = None
    if quantization == "int8":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    elif quantization == "int4":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )

    dtype = torch.float16 if not bnb_config else None
    print(f"[DeltaCert] Loading model: {model_name} ({quantization or 'fp16'})")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        quantization_config=bnb_config,
        device_map=device if not bnb_config else "auto",
    )
    model.eval()
    return model, tokenizer


def _load_prompts_and_domains(prompts_file: str = None, n: int = 128):
    """Load prompts (+ optional domain labels) from file, or use the
    built-in calibration set.

    Supports plain files (one prompt per line) and domain-tagged files
    (`domain<TAB>prompt` per line, as written by
    validation.flagship_common.load_canaries_with_domains). Returns
    (prompts, domains) where domains is None if the file wasn't tagged.
    """
    if prompts_file and os.path.exists(prompts_file):
        with open(prompts_file) as f:
            raw_lines = [l.rstrip("\n") for l in f if l.strip()]
        tagged = raw_lines and all("\t" in l for l in raw_lines)
        if tagged:
            domains, prompts = [], []
            for line in raw_lines:
                dom, _, text = line.partition("\t")
                domains.append(dom)
                prompts.append(text)
            print(f"[DeltaCert] Loaded {len(prompts)} domain-tagged prompts "
                  f"from {prompts_file} ({len(set(domains))} domains)")
            return prompts[:n], domains[:n]
        prompts = [l.strip() for l in raw_lines]
        print(f"[DeltaCert] Loaded {len(prompts)} prompts from {prompts_file}")
        return prompts[:n], None

    return _load_prompts(None, n), None


def _load_prompts(prompts_file: str = None, n: int = 128) -> list:
    """Load prompts from file or use built-in calibration set."""
    if prompts_file and os.path.exists(prompts_file):
        prompts, _ = _load_prompts_and_domains(prompts_file, n)
        return prompts

    base = [
        "The capital of France is",
        "Explain quantum entanglement in simple terms.",
        "Write a Python function to sort a list.",
        "What is the meaning of life?",
        "The largest planet in our solar system is",
        "How does photosynthesis work?",
        "Translate 'hello' to Spanish.",
        "What are the main causes of climate change?",
        "Define machine learning in one sentence.",
        "The speed of light is approximately",
    ]
    # Return only unique prompts — cycling duplicates inflate n_samples without
    # adding calibration signal.
    count = min(n, len(base))
    print(f"[DeltaCert] Using {count} built-in calibration prompts (pass --prompts for more)")
    return base[:count]


# ─────────────────────────────────────────────────────────────────────────────
# certify command
# ─────────────────────────────────────────────────────────────────────────────

def cmd_certify(args):
    # ── Resolve which checks to run ──────────────────────────────────────────
    if args.checks:
        checks = args.checks
        # Hard fail — never silently drop an explicitly requested check.
        # Pre-flight vLLM check — clean error before model loading, not ImportError mid-run.
        # kv_cache_quant needs vLLM only when using the default vllm backend
        # (the real production measurement) — the hf backend doesn't.
        needs_vllm = [c for c in checks if c in _NEEDS_VLLM]
        if "kv_cache_quant" in checks and args.kv_cache_backend == "vllm":
            needs_vllm.append("kv_cache_quant")
        if needs_vllm:
            try:
                import vllm  # noqa: F401
            except ImportError:
                print(f"[DeltaCert] ERROR: {needs_vllm} require vLLM, which is not installed.")
                print(f"[DeltaCert]   pip install vllm")
                print(f"[DeltaCert]   or run kv_cache_quant without vLLM: --kv-cache-backend hf")
                print(f"[DeltaCert]   or run without those checks:")
                print(f"[DeltaCert]   deltacert certify --model {args.model} --checks weight_quant kv_cache_quant prefix_cache --kv-cache-backend hf")
                sys.exit(1)

        needs_api = [c for c in checks if c in _NEEDS_PYTHON_API]
        if needs_api:
            print(f"[DeltaCert] ERROR: {needs_api} cannot be driven from the CLI.")
            print(f"[DeltaCert] These checks require code you supply (compress_fn, etc.).")
            print(f"[DeltaCert] Use the Python API:")
            print(f"[DeltaCert]   import deltacert as dc")
            print(f"[DeltaCert]   cert = dc.certify_system(model=..., checks={needs_api}, ...)")
            print(f"[DeltaCert] No certificate written.")
            sys.exit(1)
    else:
        quant_key = (args.quantization or "none").lower()
        checks = _DEFAULTS.get(quant_key, _DEFAULTS["none"])
        print(f"[DeltaCert] No --checks specified. Defaults for "
              f"--quantization {quant_key}: {checks}")

    # ── Validate flags for checks that require them ───────────────────────────
    # Missing flag → hard fail before any model is loaded.
    for check in checks:
        if check in _REQUIRES_FLAG:
            flag_name, attr = _REQUIRES_FLAG[check]
            if not getattr(args, attr, None):
                print(f"[DeltaCert] ERROR: '{check}' requires {flag_name}.")
                if check == "engine_swap":
                    print(f"[DeltaCert] Capture baseline first:")
                    print(f"[DeltaCert]   deltacert capture --model {args.model} --output baseline.npz")
                print(f"[DeltaCert] No certificate written.")
                sys.exit(1)
        if check in _REQUIRES_FLAGS_MULTI:
            for flag_name, attr in _REQUIRES_FLAGS_MULTI[check]:
                if not getattr(args, attr, None):
                    print(f"[DeltaCert] ERROR: '{check}' requires {flag_name}.")
                    if check == "model_swap":
                        print(f"[DeltaCert] Capture both checkpoints first:")
                        print(f"[DeltaCert]   deltacert capture --model <old-checkpoint> --output baseline.npz")
                        print(f"[DeltaCert]   deltacert capture --model <new-checkpoint> --output candidate.npz")
                    print(f"[DeltaCert] No certificate written.")
                    sys.exit(1)

    # torch is only needed once we're actually about to load models -- all
    # argument validation above must produce clean error messages even on a
    # torch-less machine (e.g. a reviewer's minimal environment), not an
    # ImportError. Found via live testing: `certify` with a missing required
    # flag used to crash on `import torch` before ever reaching the "requires
    # --X" message.
    import torch

    prompts, domain_labels = _load_prompts_and_domains(args.prompts, n=args.n_prompts)

    # ── Resolve @file syntax for --prompt-a / --prompt-b ─────────────────────
    def _resolve_prompt_text(val: str) -> str:
        if val and val.startswith("@"):
            path = val[1:]
            if not os.path.exists(path):
                print(f"[DeltaCert] ERROR: prompt file not found: {path}")
                sys.exit(1)
            with open(path, encoding="utf-8") as f:
                return f.read()
        return val or ""

    if "prompt_swap" in checks:
        args.prompt_a = _resolve_prompt_text(getattr(args, "prompt_a", None))
        args.prompt_b = _resolve_prompt_text(getattr(args, "prompt_b", None))

    # ── Load fp16 base model if any HF-based check needs it ──────────────────
    # kv_cache_quant only needs it for the hf backend — the vllm backend
    # (default) measures via real vLLM engines, no HF model required.
    _NEEDS_FP16 = {
        "weight_quant", "activation_quant",
        "prefix_cache", "lora", "prompt_swap", "trajectory",
    }
    if "kv_cache_quant" in checks and args.kv_cache_backend == "hf":
        _NEEDS_FP16 = _NEEDS_FP16 | {"kv_cache_quant"}
    model_fp16, tokenizer = None, None
    if any(c in checks for c in _NEEDS_FP16):
        model_fp16, tokenizer = _load_model_and_tokenizer(
            args.model, quantization=None, device=args.device
        )

    # ── Load quantized model for weight_quant ────────────────────────────────
    model_q = None
    if "weight_quant" in checks:
        if not args.quantization or args.quantization.lower() == "none":
            print("[DeltaCert] ERROR: weight_quant requires --quantization int8|int4.")
            sys.exit(1)
        model_q, _ = _load_model_and_tokenizer(
            args.model, quantization=args.quantization, device=args.device
        )

    # ── trajectory: candidate model (reuses --quantization, same concept as
    # weight_quant's model_q) + frozen (prompt, continuation) cases from
    # `deltacert generate-cases` ──────────────────────────────────────────────
    trajectory_model_b = None
    trajectory_cases = None
    if "trajectory" in checks:
        if not args.quantization or args.quantization.lower() == "none":
            print("[DeltaCert] ERROR: trajectory requires --quantization int8|int4 "
                  "(the candidate model being compared against --model).")
            sys.exit(1)
        # reuse an already-loaded weight_quant candidate if the caller also
        # requested weight_quant in the same run — avoids loading it twice.
        trajectory_model_b = model_q if model_q is not None else _load_model_and_tokenizer(
            args.model, quantization=args.quantization, device=args.device)[0]

        cases_path = args.trajectory_cases_file
        if not os.path.exists(cases_path):
            print(f"[DeltaCert] ERROR: --trajectory-cases file not found: {cases_path}")
            print(f"[DeltaCert] Generate one first:")
            print(f"[DeltaCert]   deltacert generate-cases --model {args.model} "
                  f"--output {cases_path}")
            sys.exit(1)
        trajectory_cases = []
        with open(cases_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                trajectory_cases.append((row["prompt"], row["continuation"]))
        if not trajectory_cases:
            print(f"[DeltaCert] ERROR: --trajectory-cases file is empty: {cases_path}")
            sys.exit(1)
        print(f"[DeltaCert] Loaded {len(trajectory_cases)} frozen (prompt, continuation) "
              f"cases from {cases_path}")

    # ── Default quant fns for kv_cache_quant / activation_quant ─────────────
    def _int8_compress(t):
        scale = t.abs().max() / 127.0 + 1e-8
        return (t / scale).round().clamp(-127, 127).to(torch.int8), scale

    def _int8_decompress(packed):
        q, scale = packed
        return q.to(torch.float16) * scale

    def _int4_compress(t):
        scale = t.abs().max() / 7.0 + 1e-8
        return (t / scale).round().clamp(-8, 7).to(torch.int8), scale

    _int4_decompress = _int8_decompress  # same unpack logic, different scale

    _kv_compress = _int4_compress if args.quantization == "int4" else _int8_compress
    _kv_decompress = _int4_decompress if args.quantization == "int4" else _int8_decompress

    # ── engine_swap: use --candidate if given, else capture current env ───────
    capture_b_path = None
    if "engine_swap" in checks:
        if getattr(args, "candidate", None):
            capture_b_path = args.candidate
            print(f"[DeltaCert] engine_swap: using pre-captured candidate {capture_b_path}")
        else:
            print(f"[DeltaCert] engine_swap: capturing current-env logits (HF)...")
            cur_logits = capture_logits_hf(
                args.model, prompts, device=args.device,
                model=model_fp16, tokenizer=tokenizer,
            )
            with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tf:
                capture_b_path = tf.name
            save_logits(capture_b_path, cur_logits, prompts,
                        engine_label="current", model_id=args.model)

    # ── speculative_config from --draft-model ─────────────────────────────────
    speculative_config = None
    if "spec_decoding" in checks:
        speculative_config = {
            "model": args.draft_model,
            "num_speculative_tokens": 5,
        }

    # ── Dispatch everything through certify_system ────────────────────────────
    _PROVIDER_DRIFT_FIRST_RUN = "Today's capture was saved as the new baseline"
    try:
        cert = dc.certify_system(
            model=args.model,
            prompts=prompts,
            checks=checks,
            output_path=args.output,
            budget=args.budget,
            device=args.device,
            # HF model checks
            model_base=model_fp16,
            model_quantized=model_q,
            # This CLI's weight_quant path always loads model_q via
            # BitsAndBytesConfig (see _load_model_and_tokenizer) -- so when
            # weight_quant is actually being run, its threshold should
            # resolve through the bnb per-family provisional calibration,
            # not the flat --budget default (calibrated for KV-cache/
            # trajectory, not weight quantization).
            weight_quant_method="bnb" if "weight_quant" in checks else None,
            tokenizer=tokenizer,
            # KV / activation quant
            compress_fn=_kv_compress,
            decompress_fn=_kv_decompress,
            quant_fn=_int8_compress,
            dequant_fn=_int8_decompress,
            # LoRA
            lora_adapter_path=args.lora_path,
            # Prefix cache
            shared_prefix=args.prefix or "You are a helpful assistant.\n\n",
            # Engine swap
            capture_a_path=args.baseline,
            capture_b_path=capture_b_path,
            # vLLM-based checks
            model_name_or_path=args.model,
            batched_size=args.batch_size,
            # Spec decoding
            speculative_config=speculative_config,
            # Model swap (17)
            candidate_path=getattr(args, "candidate", None),
            # Provider drift (18)
            api_base=getattr(args, "api_base", None),
            api_model=getattr(args, "api_model", None),
            api_key=getattr(args, "api_key", None),
            drift_baseline_path=getattr(args, "drift_baseline", None),
            # Prompt swap (19)
            system_prompt_a=getattr(args, "prompt_a", None),
            system_prompt_b=getattr(args, "prompt_b", None),
            # Trajectory (20)
            trajectory_cases=trajectory_cases,
            trajectory_model_b=trajectory_model_b,
            # Domain-stratified certification (worst-domain, not blended average)
            domain_labels=domain_labels,
            # KV cache quant backend (default: vllm, real production measurement)
            kv_cache_backend=args.kv_cache_backend,
            kv_cache_dtype=args.kv_cache_dtype,
        )
    except CollectionError as e:
        if _PROVIDER_DRIFT_FIRST_RUN in str(e):
            drift_path = getattr(args, "drift_baseline", "?")
            print(f"[DeltaCert] provider_drift: baseline established at {drift_path}.")
            print(f"[DeltaCert] Re-run after the next provider cycle to measure drift.")
            sys.exit(0)
        raise
    finally:
        # Clean up temp file — only if we created it (not a user-provided --candidate)
        if (capture_b_path and os.path.exists(capture_b_path)
                and not getattr(args, "candidate", None)):
            os.unlink(capture_b_path)
        # trajectory_model_b is the SAME object as model_q when both
        # weight_quant and trajectory were requested together (reused) —
        # decide before deleting either name, so we never double-free or
        # reference an already-deleted name.
        _trajectory_b_is_separate = (
            trajectory_model_b is not None and trajectory_model_b is not model_q)
        if model_q is not None:
            del model_q
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        if _trajectory_b_is_separate:
            del trajectory_model_b
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    print()
    print(dc.summary(cert))
    print()

    using_defaults = not args.checks
    if using_defaults and not any(c in checks for c in _NEEDS_VLLM):
        print("[DeltaCert] Tip: with vLLM installed, add --checks batch_divergence spec_decoding "
              "to also certify serving-engine behavior.")

    print("[DeltaCert] Tip: this verdict uses DeltaCert's reference calibration "
          "(Llama-3.1-8B suite, public). For a threshold tuned to YOUR model "
          "and workload: deltacert calibrate --baseline ... --candidates ... "
          "--names ... --downstream-file evals.json")

    if cert["certified"]:
        print(f"[DeltaCert] CERTIFIED. Saved to {args.output}")
        sys.exit(0)
    else:
        print(f"[DeltaCert] NOT CERTIFIED. See failed layers above.")
        sys.exit(1 if args.strict else 0)


# ─────────────────────────────────────────────────────────────────────────────
# capture command — save logits from current engine for engine_swap comparison
# ─────────────────────────────────────────────────────────────────────────────

def cmd_capture(args):
    from deltacert.collectors import capture_logits_vllm
    prompts = _load_prompts(args.prompts, n=args.n_prompts)
    out = args.output if args.output.endswith(".npz") else args.output + ".npz"
    backend = getattr(args, "backend", "hf")
    quantization = getattr(args, "quantization", None)

    if quantization and quantization.lower() != "none":
        if backend != "hf":
            print("[DeltaCert] --quantization currently requires --backend hf "
                  "(vLLM-native quantized capture is not wired up yet - use "
                  "`deltacert certify --kv-cache-backend vllm` for vLLM-native "
                  "kv_cache_quant instead).")
            sys.exit(1)
        model, tokenizer = _load_model_and_tokenizer(
            args.model, quantization=quantization, device=args.device)
        logits = capture_logits_hf(args.model, prompts, device=args.device,
                                   model=model, tokenizer=tokenizer)
        engine_label = f"hf:{quantization}"
    elif backend == "vllm":
        print(f"[DeltaCert] Capturing via vLLM: {args.model}")
        logits = capture_logits_vllm(args.model, prompts)
        engine_label = "vllm"
    else:
        logits = capture_logits_hf(args.model, prompts, device=args.device)
        engine_label = "hf"
    save_logits(out, logits, prompts, engine_label=engine_label, model_id=args.model)
    print(f"[DeltaCert] Captured {len(prompts)} prompt logits -> {out}")


# ─────────────────────────────────────────────────────────────────────────────
# generate-cases command — freeze real (prompt, continuation) pairs once,
# for `deltacert certify --checks trajectory --trajectory-cases <file>`
# ─────────────────────────────────────────────────────────────────────────────

def cmd_generate_cases(args):
    """Generate real continuations ONCE with a single model, freeze them to
    a JSONL file. Trajectory certification compares a SECOND model's
    teacher-forced logits against these exact frozen continuations — the
    continuations must come from real generation, never be synthesized, or
    the divergence measurement means nothing."""
    import torch

    model, tokenizer = _load_model_and_tokenizer(
        args.model, quantization=None, device=args.device)
    prompts = _load_prompts(args.prompts, n=args.n_prompts)

    out_path = args.output
    with open(out_path, "w", encoding="utf-8") as f:
        for i, prompt in enumerate(prompts):
            inputs = tokenizer(prompt, return_tensors="pt").to(args.device)
            with torch.no_grad():
                out_ids = model.generate(
                    **inputs, max_new_tokens=args.max_new_tokens,
                    do_sample=False, pad_token_id=tokenizer.pad_token_id)
            continuation = tokenizer.decode(
                out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            f.write(json.dumps({"prompt": prompt, "continuation": continuation}) + "\n")
            print(f"[DeltaCert]   [{i + 1}/{len(prompts)}] generated {len(continuation)} chars")

    print(f"[DeltaCert] Wrote {len(prompts)} frozen (prompt, continuation) cases -> {out_path}")
    print(f"[DeltaCert] Use for trajectory certification:")
    print(f"[DeltaCert]   deltacert certify --model {args.model} --quantization int4 "
          f"--checks trajectory --trajectory-cases {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# free-running command — collector 21, separate from `certify` because its
# certificate is McNemar/degeneration-based, not d_COMM-based (see the WHY
# comment on collect_free_running_vllm in collectors.py). Boots two live
# vLLM engines itself; needs vLLM the same way batch_divergence/spec_decoding do.
# ─────────────────────────────────────────────────────────────────────────────

def cmd_free_running(args):
    try:
        import vllm  # noqa: F401
    except ImportError:
        print("[DeltaCert] ERROR: free-running requires vLLM, which is not installed.")
        print("[DeltaCert]   pip install vllm")
        sys.exit(1)

    prompts = _load_prompts(args.prompts, n=args.n_prompts)
    print(f"[DeltaCert] free-running: {args.model}, kv_cache_dtype={args.kv_cache_dtype}, "
          f"{len(prompts)} prompts, max_new_tokens={args.max_new_tokens}")

    result = collect_free_running_vllm(
        args.model, prompts,
        kv_cache_dtype=args.kv_cache_dtype,
        max_new_tokens=args.max_new_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tau_degen=args.tau_degen,
        mcnemar_alpha=args.mcnemar_alpha,
    )

    cert = {
        "schema_version": "1.0",
        "model": args.model,
        "check": "free_running",
        "change": f"KV cache default -> {args.kv_cache_dtype} (vLLM native, free-running)",
        "certified": result["certified"],
        "verdict": result["verdict"],
        "layers": {"free_running": result},
        "metadata": _build_metadata(args.model),
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(cert, f, indent=2)

    print()
    print(f"[DeltaCert] excess_degeneration_rate: {result['excess_degeneration_rate']}")
    print(f"[DeltaCert] McNemar: b={result['mcnemar_b']} c={result['mcnemar_c']} "
          f"p={result['mcnemar_p']:.4g} (significant={result['degeneration_significant']})")
    print(f"[DeltaCert] surprisal_q95_delta: {result['surprisal_q95_delta']}")
    print()

    if cert["certified"]:
        print(f"[DeltaCert] CERTIFIED. Saved to {args.output}")
        sys.exit(0)
    else:
        print(f"[DeltaCert] NOT CERTIFIED. See layers above.")
        sys.exit(1 if args.strict else 0)


# ─────────────────────────────────────────────────────────────────────────────
# check + summary commands
# ─────────────────────────────────────────────────────────────────────────────

def cmd_check(args):
    if not os.path.exists(args.cert):
        print(f"[DeltaCert] Certificate not found: {args.cert}")
        sys.exit(1)
    cert = dc.load_certificate(args.cert)
    ok, failures = dc.check_certified(cert)
    print(dc.summary(cert))
    if ok:
        print("\n[DeltaCert] CERTIFIED")
        sys.exit(0)
    else:
        print(f"\n[DeltaCert] NOT CERTIFIED. Failed: {failures}")
        sys.exit(1)


def cmd_summary(args):
    cert = dc.load_certificate(args.cert)
    print(dc.summary(cert))
    comp = dc.compose_bounds(cert.get("layers", {}))
    print(f"\n  total divergence bound (union): {comp['total_divergence_bound']}")
    print(f"  active layers:                  {comp['n_layers_active']}")
    meta = cert.get("metadata", {})
    if meta:
        print(f"  certified_at:                   {meta.get('certified_at', '?')}")
        print(f"  gpu:                            {meta.get('gpu_name', '?')}")


# ─────────────────────────────────────────────────────────────────────────────
# keygen + sign + verify commands
# ─────────────────────────────────────────────────────────────────────────────

def cmd_keygen(args):
    if os.path.exists(args.private_key) and not args.force:
        print(f"[DeltaCert] {args.private_key} already exists. Use --force to overwrite.")
        sys.exit(1)
    private_pem, public_pem = dsign.generate_keypair()
    with open(args.private_key, "wb") as f:
        f.write(private_pem)
    with open(args.public_key, "wb") as f:
        f.write(public_pem)
    try:
        os.chmod(args.private_key, 0o600)
    except OSError:
        pass  # best-effort on platforms without POSIX permissions (e.g. Windows)
    print(f"[DeltaCert] Generated Ed25519 keypair.")
    print(f"  private key: {args.private_key}  (keep secret, do not commit)")
    print(f"  public key:  {args.public_key}  (safe to publish/commit)")


def cmd_sign(args):
    if not os.path.exists(args.cert):
        print(f"[DeltaCert] Certificate not found: {args.cert}")
        sys.exit(1)
    cert = dc.load_certificate(args.cert)
    try:
        private_key = dsign.load_private_key(args.key_file)
        signed = dsign.sign_certificate(cert, private_key, key_id=args.key_id or "")
    except dsign.SigningError as e:
        print(f"[DeltaCert] Signing failed: {e}")
        sys.exit(1)

    out_path = args.cert if args.in_place else (args.output or args.cert)
    with open(out_path, "w") as f:
        json.dump(signed, f, indent=2)
    print(f"[DeltaCert] Signed. Wrote {out_path}")


def cmd_verify(args):
    if not os.path.exists(args.cert):
        print(f"[DeltaCert] Certificate not found: {args.cert}")
        sys.exit(1)
    cert = dc.load_certificate(args.cert)
    try:
        public_key = dsign.load_public_key(args.key_file)
    except dsign.SigningError as e:
        print(f"[DeltaCert] {e}")
        sys.exit(1)

    result = dsign.verify_certificate(cert, public_key)
    status = cert.get("validation_status", "?")
    if result.ok:
        print(f"[DeltaCert] VALID SIGNATURE — {args.cert}")
        print(f"  validation_status: {status}")
        sys.exit(0)
    else:
        print(f"[DeltaCert] INVALID — {args.cert}")
        print(f"  reason: {result.reason}")
        sys.exit(1)


def cmd_calibrate(args):
    """Self-calibration: find YOUR safe/unsafe tau from YOUR OWN sweep of
    configs, instead of trusting DeltaCert's shipped reference calibration.

    Inputs are .npz captures from `deltacert capture` — one baseline, one
    per config being swept — plus the REAL downstream drop you measured for
    each config on your own eval, given as a name->drop JSON file (not
    positionally-matched CLI floats, to avoid silently mismatching a name
    to the wrong number). Never simulates; a config with no measured
    downstream number is rejected (see calibrate_layer)."""
    from deltacert.collectors import cos_sims_from_logit_matrices, load_logits

    if len(args.candidates) != len(args.names):
        print("[DeltaCert] --candidates and --names must have the same length.")
        sys.exit(1)
    if len(args.method_families) != len(args.names):
        print("[DeltaCert] --method-families and --names must have the same length "
              "(one method family per candidate, same order).")
        sys.exit(1)

    with open(args.downstream_file, encoding="utf-8") as f:
        downstream_by_name = json.load(f)
    missing = [n for n in args.names if n not in downstream_by_name]
    if missing:
        print(f"[DeltaCert] --downstream-file is missing an entry for: {missing}. "
              f"Every name in --names needs a real measured drop in this file.")
        sys.exit(1)

    domain_labels = None
    if getattr(args, "domains_file", None):
        _, domain_labels = _load_prompts_and_domains(args.domains_file)
        if domain_labels is None:
            print(f"[DeltaCert] --domains-file '{args.domains_file}' is not "
                  "domain-tagged (expected 'domain<TAB>prompt' lines, same "
                  "format `deltacert capture --prompts` accepts).")
            sys.exit(1)

    baseline_logits, _ = load_logits(args.baseline)
    sweep = []
    for cand_path, name, family in zip(args.candidates, args.names, args.method_families):
        cand_logits, _ = load_logits(cand_path)
        cos_sims = cos_sims_from_logit_matrices(baseline_logits, cand_logits)
        if domain_labels is not None and len(domain_labels) != len(cos_sims):
            print(f"[DeltaCert] --domains-file has {len(domain_labels)} entries "
                  f"but '{name}' has {len(cos_sims)} prompts - they must match "
                  "(same prompt file used for every capture in this sweep).")
            sys.exit(1)
        entry = {"name": name, "method_family": family, "cos_sims": cos_sims,
                 "downstream_drop_pts": downstream_by_name[name]}
        if domain_labels is not None:
            entry["domain_labels"] = domain_labels
        sweep.append(entry)

    result = dc.calibrate_layer(
        sweep, downstream_degradation_threshold_pts=args.degradation_threshold)

    print(f"[DeltaCert] Calibrated {result['n_families']} method family(ies)")
    for family, fam_result in result["families"].items():
        prov = " [PROVISIONAL]" if fam_result["provisional"] else ""
        print(f"\n  family '{family}': tau = {fam_result['calibrated_tau']}{prov} "
              f"(from n={fam_result['n_configs']} config(s): "
              f"{fam_result['n_safe_configs']} safe, {fam_result['n_damaged_configs']} damaged)")
        for r in fam_result["rows"]:
            mark = "safe" if not r["materially_degraded"] else "DAMAGED"
            worst = f"  (worst domain: {r['worst_domain']})" if "worst_domain" in r else ""
            print(f"      {r['name']:<20} d_comm={r['d_comm']:<8} "
                  f"downstream={r['downstream_drop_pts']:+.1f}pts  [{mark}]{worst}")
        print(f"      {fam_result['caveat']}")
    print(f"\n{result['caveat']}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"\n[DeltaCert] Saved to {args.output}")


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="deltacert",
        description="DeltaCert — certify your LLM serving system before deployment",
    )
    sub = parser.add_subparsers(dest="command")

    # ── certify ───────────────────────────────────────────────────────────────
    p = sub.add_parser("certify", help="Run certification and save cert.json")
    p.add_argument("--model",        required=True, help="HuggingFace model ID or local path")
    p.add_argument("--quantization", default=None,  help="int8 | int4 | none")
    p.add_argument("--checks",       nargs="*",     help="Checks to run (default: auto from --quantization)")
    p.add_argument("--output",       default="./deltacert.json", help="Output certificate path")
    p.add_argument("--prompts",      default=None,  help=".txt file with calibration prompts, one per line")
    p.add_argument("--n-prompts",    type=int, default=128, help="Number of prompts (default: 128)")
    p.add_argument("--budget",       type=float, default=3.0, help="Min d_COMM to certify (default: 3.0)")
    p.add_argument("--device",       default="cuda", help="cuda | cpu")
    p.add_argument("--batch-size",   type=int, default=32, help="Batch size for batch_divergence")
    p.add_argument("--kv-cache-backend", default="vllm", choices=["vllm", "hf"],
                   help="kv_cache_quant measurement backend: vllm = real "
                        "production measurement via vLLM's native "
                        "--kv-cache-dtype flag (default); hf = hand-rolled "
                        "hook, for non-vLLM deployments")
    p.add_argument("--kv-cache-dtype", default="fp8",
                   help="KV cache dtype to certify when --kv-cache-backend=vllm (default: fp8)")
    p.add_argument("--prefix",       default=None,  help="System prompt prefix for prefix_cache")
    p.add_argument("--draft-model",  default=None,  help="Draft model path/ID for spec_decoding (vLLM)")
    p.add_argument("--lora-path",    default=None,  help="LoRA adapter path for lora check")
    p.add_argument("--baseline",     default=None,  help=".npz from deltacert capture (for engine_swap)")
    p.add_argument("--candidate",    default=None,  help=".npz of new checkpoint capture (for model_swap)")
    p.add_argument("--api-base",     default=None,  help="OpenAI-compatible API base URL (for provider_drift)")
    p.add_argument("--api-model",    default=None,  help="Model name at the API (for provider_drift)")
    p.add_argument("--api-key",      default=None,  help="API key (or set OPENAI_API_KEY env var)")
    p.add_argument("--drift-baseline", default=None, help=".npz baseline for provider_drift (created on first run)")
    p.add_argument("--prompt-a",     default=None,  help="System prompt v1 text or @file path (for prompt_swap)")
    p.add_argument("--prompt-b",     default=None,  help="System prompt v2 text or @file path (for prompt_swap)")
    p.add_argument("--trajectory-cases", dest="trajectory_cases_file", default=None,
                   help=".jsonl of frozen (prompt, continuation) pairs from "
                        "`deltacert generate-cases` (for trajectory)")
    p.add_argument("--strict",       action="store_true", default=True,
                   help="Exit 1 if not certified (default: true)")
    p.add_argument("--no-strict",    dest="strict", action="store_false",
                   help="Exit 0 even if not certified")

    # ── capture ───────────────────────────────────────────────────────────────
    p2 = sub.add_parser("capture", help="Capture logits from current engine (for engine_swap, weight_quant sweeps)")
    p2.add_argument("--model",     required=True)
    p2.add_argument("--output",    required=True, help="Output baseline .npz path")
    p2.add_argument("--prompts",   default=None)
    p2.add_argument("--n-prompts", type=int, default=128)
    p2.add_argument("--device",    default="cuda")
    p2.add_argument("--backend",   default="hf", choices=["hf", "vllm"],
                    help="hf = HuggingFace (default), vllm = capture via vLLM engine")
    p2.add_argument("--quantization", default=None, choices=["int8", "int4", "none"],
                    help="Load the model with BnB quantization before capturing "
                         "(requires --backend hf) — use this to capture a "
                         "quantized candidate for `deltacert calibrate` sweeps")

    # ── generate-cases ────────────────────────────────────────────────────────
    p2b = sub.add_parser("generate-cases",
                          help="Freeze real (prompt, continuation) pairs once, "
                               "for `deltacert certify --checks trajectory`")
    p2b.add_argument("--model",          required=True, help="Model to generate WITH (typically fp16 baseline)")
    p2b.add_argument("--output",         required=True, help="Output .jsonl path")
    p2b.add_argument("--prompts",        default=None)
    p2b.add_argument("--n-prompts",      type=int, default=50)
    p2b.add_argument("--max-new-tokens", type=int, default=256)
    p2b.add_argument("--device",         default="cuda")

    # ── free-running ──────────────────────────────────────────────────────────
    p2c = sub.add_parser("free-running",
                          help="Certify feedback-driven failures (e.g. KV-cache "
                               "collapse) invisible to single-position and "
                               "trajectory certification -- boots two live "
                               "vLLM engines, runs the deployed decode policy, "
                               "McNemar-gated degeneration + cross-surprisal verdict")
    p2c.add_argument("--model",          required=True, help="HuggingFace model ID or local path")
    p2c.add_argument("--kv-cache-dtype", default="fp8",  help="KV cache dtype to certify (default: fp8)")
    p2c.add_argument("--prompts",        default=None,   help=".txt file with prompts, one per line")
    p2c.add_argument("--n-prompts",      type=int, default=43, help="Number of prompts (default: 43)")
    p2c.add_argument("--max-new-tokens", type=int, default=512, help="Decode budget per engine per prompt")
    p2c.add_argument("--gpu-memory-utilization", type=float, default=0.42,
                      help="Per-engine GPU memory fraction -- both engines must "
                           "coexist concurrently, so this must be low enough for two")
    p2c.add_argument("--tau-degen",      type=float, default=0.05, help="Excess-degeneration-rate threshold")
    p2c.add_argument("--mcnemar-alpha",  type=float, default=0.01, help="McNemar significance threshold")
    p2c.add_argument("--output",         default="./deltacert_free_running.json", help="Output certificate path")
    p2c.add_argument("--strict",         action="store_true", default=True,
                      help="Exit 1 if not certified (default: true)")
    p2c.add_argument("--no-strict",      dest="strict", action="store_false",
                      help="Exit 0 even if not certified")

    # ── check ─────────────────────────────────────────────────────────────────
    p3 = sub.add_parser("check", help="Pass/fail check on existing certificate")
    p3.add_argument("--cert", required=True)

    # ── summary ───────────────────────────────────────────────────────────────
    p4 = sub.add_parser("summary", help="Print certificate summary")
    p4.add_argument("--cert", required=True)

    # ── keygen ────────────────────────────────────────────────────────────────
    p6 = sub.add_parser("keygen", help="Generate an Ed25519 keypair for signing certificates")
    p6.add_argument("--private-key", default="deltacert-private.pem", help="Output path for the private key (keep secret)")
    p6.add_argument("--public-key",  default="deltacert-public.pem",  help="Output path for the public key (safe to publish)")
    p6.add_argument("--force",       action="store_true", help="Overwrite existing key files")

    # ── sign ──────────────────────────────────────────────────────────────────
    p7 = sub.add_parser("sign", help="Sign a certificate with your private key")
    p7.add_argument("--cert",      required=True, help="Certificate JSON to sign")
    p7.add_argument("--key-file",  required=True, help="Private key PEM file")
    p7.add_argument("--key-id",    default="",    help="Optional identifier for which key/keyholder signed this")
    p7.add_argument("--output",    default=None,  help="Write the signed cert here (default: overwrite --cert)")
    p7.add_argument("--in-place",  action="store_true", help="Overwrite --cert directly (same as omitting --output)")

    # ── verify ────────────────────────────────────────────────────────────────
    p8 = sub.add_parser("verify", help="Verify a certificate's signature against a public key")
    p8.add_argument("--cert",     required=True, help="Certificate JSON to verify")
    p8.add_argument("--key-file", required=True, help="Public key PEM file")

    # ── calibrate ─────────────────────────────────────────────────────────────
    p5 = sub.add_parser("calibrate",
                         help="Find YOUR safe/unsafe tau from YOUR OWN sweep "
                              "(not DeltaCert's shipped reference calibration)")
    p5.add_argument("--baseline",   required=True, help=".npz from `deltacert capture` (uncompressed/reference config)")
    p5.add_argument("--candidates", required=True, nargs="+", help=".npz captures, one per swept config")
    p5.add_argument("--names",      required=True, nargs="+", help="Name for each candidate, e.g. int8 nf4 gptq_int4")
    p5.add_argument("--method-families", required=True, nargs="+",
                     help="Method family for each candidate, e.g. bnb bnb gptq gptq gptq "
                          "(one entry per --names value, same order). d_comm is not "
                          "comparable across quantization methods, so tau is calibrated "
                          "independently per family — configs that are all one family "
                          "just repeat the same string for every entry.")
    p5.add_argument("--downstream-file", required=True,
                     help="JSON file mapping name->downstream accuracy drop (pts), "
                          "e.g. {\"int8\": 0.0, \"nf4\": 1.0, \"gptq_int4\": -8.0} "
                          "(one entry per --names value — a file, not positional "
                          "CLI floats, so a name can never silently pair with the "
                          "wrong number)")
    p5.add_argument("--degradation-threshold", type=float, default=2.0,
                     help="Downstream drop (pts) above which a config counts as materially degraded (default: 2.0)")
    p5.add_argument("--domains-file", default=None,
                     help="Domain-tagged prompts file (same one used for every "
                          "`deltacert capture` in this sweep, 'domain<TAB>prompt' "
                          "format) — enables worst-domain calibration instead of "
                          "a blended average, matching how DeltaCert's own "
                          "flagship tests are calibrated. Strongly recommended "
                          "whenever prompts span more than one domain.")
    p5.add_argument("--output", default=None, help="Save calibration result JSON here")

    args = parser.parse_args()

    if args.command == "certify":
        cmd_certify(args)
    elif args.command == "capture":
        cmd_capture(args)
    elif args.command == "generate-cases":
        cmd_generate_cases(args)
    elif args.command == "free-running":
        cmd_free_running(args)
    elif args.command == "check":
        cmd_check(args)
    elif args.command == "summary":
        cmd_summary(args)
    elif args.command == "keygen":
        cmd_keygen(args)
    elif args.command == "sign":
        cmd_sign(args)
    elif args.command == "verify":
        cmd_verify(args)
    elif args.command == "calibrate":
        cmd_calibrate(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
