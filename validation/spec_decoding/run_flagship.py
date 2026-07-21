"""
validation/spec_decoding/run_flagship.py — FLAGSHIP CASE STUDY

Company scenario: "Speculative decoding promises ~2x decode throughput and
claims to be lossless — certify OUR config before enabling it fleet-wide."

Uses the canonical pairing from vLLM's own documentation
(Llama-3.1-8B target + Llama-3.2-1B draft) so the certified config is the
one companies copy-paste.

Each vLLM engine build runs in its own SEPARATE OS PROCESS (subprocess), not
just a Python object deleted in-process. vLLM v1's engine core runs as a
background process that a plain `del llm` + gc.collect() + wait-polling does
NOT reliably terminate — confirmed empirically: a 90s poll for 70GB free GPU
memory after teardown still saw only 8.88/79.18 GiB free (memory never
actually released), causing the next engine to fail with "Free memory ...
is less than desired GPU memory utilization". A real subprocess exit
guarantees the OS reclaims all GPU memory — same reliable pattern
kv_cache_quant and engine_swap already use for their two-engine comparisons.

Triple measurement:
  1. d_COMM      — collect_speculative_decode (collectors.py, untouched;
                   builds its own two sequential engines), run inside its
                   own fresh subprocess so it starts from a clean GPU
  2. downstream  — GSM8K exact-match ON vs OFF + measured tokens/sec both
                   modes (tok/s ratio IS the business-gain number), each
                   mode built in its own subprocess
  3. short_eval  — the GSM8K delta doubles as the standard-benchmark number
                   here (same task lm-eval uses), recorded explicitly

Run (RunPod H100 SXM):
    python validation/spec_decoding/run_flagship.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --draft meta-llama/Llama-3.2-1B-Instruct --gsm8k-n 100
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flagship_common import (  # noqa: E402
    DEFAULT_MODEL, assemble_and_save_result, build_cert, environment_stamp,
    load_canaries_with_domains, load_gsm8k, set_all_seeds,
)

HERE = os.path.dirname(os.path.abspath(__file__))


def _worker_downstream() -> None:
    """Builds ONE vLLM engine (spec on or off), measures GSM8K acc + tok/s,
    writes result to --out as JSON. Invoked as a fresh subprocess only."""
    import vllm
    from flagship_common import gsm8k_accuracy_vllm, load_gsm8k, set_all_seeds

    ap = argparse.ArgumentParser()
    ap.add_argument("--_worker_downstream", action="store_true")
    ap.add_argument("--model", required=True)
    ap.add_argument("--draft", default=None)
    ap.add_argument("--num-spec-tokens", type=int, default=5)
    ap.add_argument("--gsm8k-n", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    set_all_seeds()
    problems = load_gsm8k(args.gsm8k_n)
    kwargs = dict(model=args.model, tensor_parallel_size=1, gpu_memory_utilization=0.85)
    if args.draft:
        kwargs["speculative_config"] = {"model": args.draft,
                                        "num_speculative_tokens": args.num_spec_tokens}
    llm = vllm.LLM(**kwargs)
    acc, _, tps = gsm8k_accuracy_vllm(llm, problems)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"acc": acc, "tps": tps}, f)
    print(f"  worker done: spec={'on' if args.draft else 'off'} acc={acc:.3f} tok/s={tps:.1f}")


def _worker_dcomm() -> None:
    """Runs collect_speculative_decode (its own two sequential engines)
    inside a fresh subprocess, starting from a clean GPU. Writes cos_sims
    to --out as .npy."""
    import numpy as np
    from flagship_common import load_canaries_with_domains, set_all_seeds
    from deltacert.collectors import collect_speculative_decode

    ap = argparse.ArgumentParser()
    ap.add_argument("--_worker_dcomm", action="store_true")
    ap.add_argument("--model", required=True)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--num-spec-tokens", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    set_all_seeds()
    canaries, _domains = load_canaries_with_domains()
    spec_config = {"model": args.draft, "num_speculative_tokens": args.num_spec_tokens}
    cos_sims = collect_speculative_decode(args.model, canaries, spec_config, num_logprobs=20)
    np.save(args.out, cos_sims)
    print("  worker done: d_comm cos_sims computed")


def _run_subprocess(cmd: list) -> None:
    print(f"  spawning subprocess: {' '.join(cmd[2:4])}...")
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"worker subprocess failed (exit {proc.returncode}): {cmd}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--draft", default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--num-spec-tokens", type=int, default=5)
    ap.add_argument("--gsm8k-n", type=int, default=100)
    args = ap.parse_args()

    set_all_seeds()
    canaries, domains = load_canaries_with_domains()
    spec_config = {"model": args.draft, "num_speculative_tokens": args.num_spec_tokens}

    print("=== spec OFF: GSM8K + tok/s (separate subprocess) ===")
    off_out = os.path.join(HERE, "_downstream_off.json")
    _run_subprocess([sys.executable, __file__, "--_worker_downstream",
                     "--model", args.model, "--gsm8k-n", str(args.gsm8k_n),
                     "--out", off_out])
    with open(off_out, encoding="utf-8") as f:
        off = json.load(f)
    acc_off, tps_off = off["acc"], off["tps"]
    print(f"  acc={acc_off:.3f}  tok/s={tps_off:.1f}")

    print("\n=== spec ON: GSM8K + tok/s (separate subprocess) ===")
    on_out = os.path.join(HERE, "_downstream_on.json")
    _run_subprocess([sys.executable, __file__, "--_worker_downstream",
                     "--model", args.model, "--draft", args.draft,
                     "--num-spec-tokens", str(args.num_spec_tokens),
                     "--gsm8k-n", str(args.gsm8k_n), "--out", on_out])
    with open(on_out, encoding="utf-8") as f:
        on = json.load(f)
    acc_on, tps_on = on["acc"], on["tps"]
    speedup = tps_on / tps_off if tps_off else 0.0
    print(f"  acc={acc_on:.3f}  tok/s={tps_on:.1f}  speedup={speedup:.2f}x")

    print("\n=== d_COMM: collect_speculative_decode (separate subprocess, per domain, worst domain certifies) ===")
    import numpy as np
    dcomm_out = os.path.join(HERE, "_dcomm_cos_sims.npy")
    _run_subprocess([sys.executable, __file__, "--_worker_dcomm",
                     "--model", args.model, "--draft", args.draft,
                     "--num-spec-tokens", str(args.num_spec_tokens),
                     "--out", dcomm_out])
    cos_sims = np.load(dcomm_out)
    cert = build_cert("spec_decoding", cos_sims, model_id=args.model,
                      domain_labels=domains)
    d = cert["layers"]["spec_decoding"]["d_comm"]
    d = float("inf") if d == "inf" else d
    cert_path = os.path.join(HERE, "cert_spec.json")
    with open(cert_path, "w", encoding="utf-8") as f:
        json.dump(cert, f, indent=2)
    print(f"  d = {d}  certified = {cert['certified']}")

    drop = (acc_on - acc_off) * 100.0
    d_disp = "inf" if d == float("inf") else round(d, 4)
    worst_domain = cert["layers"]["spec_decoding"]["worst_domain"]
    assemble_and_save_result(
        collector="spec_decoding", tier="A", run_id="flagship_spec_decode",
        change={"baseline": f"{args.model} standard decode",
                "candidate": f"{args.model} + {args.draft} spec decode "
                             f"(k={args.num_spec_tokens})",
                "change_type": "spec_decoding"},
        business_goal={"reason": "speculative decoding ≈ 2x decode throughput "
                                 "at claimed-lossless quality",
                       "expected_gain": {"throughput_x": round(speedup, 2)}},
        workload={"task_family": "math_reasoning", "dataset": "gsm8k",
                  "num_prompts": args.gsm8k_n},
        metrics={"d_comm": d_disp, "tau": 3.0,
                 "downstream_delta": {"gsm8k_acc_drop_pts": round(drop, 2),
                                      "measured_speedup_x": round(speedup, 2)},
                 "per_domain": cert["layers"]["spec_decoding"]["per_domain"],
                 "short_eval": {"benchmark": "GSM8K 5-shot exact-match",
                                "delta_pct": round(drop, 1),
                                "verdict_by_benchmark":
                                    "looks_safe" if abs(drop) <= 2
                                    else "looks_unsafe"}},
        decision_statement=(
            f"spec decode (k={args.num_spec_tokens}): worst-domain d={d_disp} ({worst_domain}), measured "
            f"{speedup:.2f}x throughput, GSM8K {drop:+.1f} pts -> "
            f"{'certified: enable fleet-wide for this config' if d >= 3.0 else 'NOT certified for this config'}"),
        cert_path=cert_path,
        notes={"environment": environment_stamp(), "vllm_config": spec_config},
        out_path=os.path.join(HERE, "result.json"),
    )
    print("\nDone.")


if __name__ == "__main__":
    if "--_worker_downstream" in sys.argv:
        _worker_downstream()
    elif "--_worker_dcomm" in sys.argv:
        _worker_dcomm()
    else:
        main()
