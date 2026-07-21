"""
validation/kv_cache_quant/run_flagship.py — FLAGSHIP CASE STUDY

Company scenario: "KV cache quantization would double our batch size — do
long conversations survive?" (Production blocker documented in the
KV-compression literature: accuracy fear.)

Design decision: measured through vLLM's REAL native `kv_cache_dtype` engine
flag — the actual thing a company running vLLM would flip in production —
not a hand-rolled compress/decompress hook. Two real vLLM engines (default
KV dtype vs the quantized KV dtype), same model, same domain-tagged canaries.

Each engine runs in its own SEPARATE OS PROCESS (subprocess), not just a
Python object deleted in-process. vLLM v1's engine core runs as a background
process that a plain `del llm` does not reliably terminate promptly — a
second engine started too soon in the same process fails with
"Free memory ... is less than desired GPU memory utilization". A real
subprocess exit guarantees the OS reclaims all GPU memory, same reliable
pattern engine_swap already uses for its two engine-version comparison.

Triple measurement:
  1. d_COMM      — capture_logits_vllm on both engines (collectors.py,
                   untouched), cos_sims -> per-domain worst-case d
  2. downstream  — GSM8K exact-match, default vs quantized KV cache
  3. short_eval  — the GSM8K delta doubles as the standard-benchmark number

Run:
    python validation/kv_cache_quant/run_flagship.py \
        --model meta-llama/Llama-3.1-8B-Instruct --gsm8k-n 100
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
from deltacert.collectors import cos_sims_from_logit_matrices

HERE = os.path.dirname(os.path.abspath(__file__))


def _worker_main() -> None:
    """Runs ONE vLLM engine, captures canary logits + GSM8K accuracy, writes
    results to --out as JSON+npz. Invoked as a fresh subprocess per engine —
    never called directly, only via `python run_flagship.py --_worker ...`."""
    import numpy as np
    import vllm
    from flagship_common import gsm8k_accuracy_vllm, load_canaries_with_domains, load_gsm8k
    from deltacert.collectors import capture_logits_vllm

    ap = argparse.ArgumentParser()
    ap.add_argument("--_worker", action="store_true")
    ap.add_argument("--model", required=True)
    ap.add_argument("--kv-cache-dtype", required=True)
    ap.add_argument("--gsm8k-n", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    set_all_seeds()
    canaries, _domains = load_canaries_with_domains()
    problems = load_gsm8k(args.gsm8k_n)

    llm = vllm.LLM(model=args.model, tensor_parallel_size=1,
                   gpu_memory_utilization=0.85, kv_cache_dtype=args.kv_cache_dtype)
    logits = capture_logits_vllm(args.model, canaries, llm=llm, num_logprobs=20)
    acc, records, _ = gsm8k_accuracy_vllm(llm, problems)

    np.save(args.out + "_logits.npy", logits)
    with open(args.out + "_acc.json", "w", encoding="utf-8") as f:
        json.dump({"acc": acc, "records": records}, f)
    print(f"  worker done: kv_cache_dtype={args.kv_cache_dtype} acc={acc:.3f}")


def _run_engine_subprocess(model: str, kv_cache_dtype: str, gsm8k_n: int, out_prefix: str):
    """Spawn a genuinely separate OS process for one engine. Guarantees full
    GPU memory reclamation on exit — unlike deleting a Python object in the
    same process, which does not reliably terminate vLLM's background
    engine-core process in time for a second engine to start."""
    import numpy as np
    cmd = [sys.executable, __file__, "--_worker",
           "--model", model, "--kv-cache-dtype", kv_cache_dtype,
           "--gsm8k-n", str(gsm8k_n), "--out", out_prefix]
    print(f"  spawning subprocess: kv_cache_dtype={kv_cache_dtype}")
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise RuntimeError(
            f"worker subprocess for kv_cache_dtype={kv_cache_dtype} failed "
            f"(exit {proc.returncode})")
    logits = np.load(out_prefix + "_logits.npy")
    with open(out_prefix + "_acc.json", encoding="utf-8") as f:
        acc = json.load(f)["acc"]
    return logits, acc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--gsm8k-n", type=int, default=100)
    ap.add_argument("--kv-dtype", default="fp8",
                    help="vLLM's real native KV cache dtype to certify "
                         "(fp8 is vLLM's standard supported KV quant option)")
    args = ap.parse_args()

    set_all_seeds()
    canaries, domains = load_canaries_with_domains()

    print("=== BASELINE: default KV cache dtype (separate subprocess) ===")
    base_logits, base_acc = _run_engine_subprocess(
        args.model, "auto", args.gsm8k_n, os.path.join(HERE, "_base"))
    print(f"  baseline GSM8K acc = {base_acc:.3f}")

    print(f"\n=== ROW: KV {args.kv_dtype} (separate subprocess) ===")
    quant_logits, q_acc = _run_engine_subprocess(
        args.model, args.kv_dtype, args.gsm8k_n, os.path.join(HERE, "_quant"))

    cos_sims = cos_sims_from_logit_matrices(base_logits, quant_logits)
    cert = build_cert("kv_cache_quant", cos_sims, model_id=args.model,
                      domain_labels=domains)
    d = cert["layers"]["kv_cache_quant"]["d_comm"]
    d = float("inf") if d == "inf" else d
    worst_domain = cert["layers"]["kv_cache_quant"]["worst_domain"]
    cert_path = os.path.join(HERE, f"cert_kv_{args.kv_dtype}.json")
    with open(cert_path, "w", encoding="utf-8") as f:
        json.dump(cert, f, indent=2)
    print(f"  worst-domain d = {d} ({worst_domain})")

    drop = (q_acc - base_acc) * 100.0
    print(f"  acc {base_acc:.3f} -> {q_acc:.3f} ({drop:+.1f} pts)")

    d_disp = "inf" if d == float("inf") else round(d, 4)
    assemble_and_save_result(
        collector="kv_cache_quant", tier="A",
        run_id=f"flagship_kv_{args.kv_dtype}",
        change={"baseline": f"{args.model} KV cache default dtype",
                "candidate": f"{args.model} KV cache {args.kv_dtype} (vLLM native)",
                "change_type": "kv_cache_quant"},
        business_goal={"reason": f"{args.kv_dtype} KV cache via vLLM's native "
                       "--kv-cache-dtype flag — more concurrent requests on "
                       "the same VRAM",
                       "expected_gain": {"throughput_x": 2.0}},
        workload={"task_family": "math_reasoning_long_cot",
                  "dataset": "gsm8k", "num_prompts": args.gsm8k_n},
        metrics={"d_comm": d_disp, "tau": 3.0,
                 "downstream_delta": {"gsm8k_acc_drop_pts": round(drop, 2)},
                 "per_domain": cert["layers"]["kv_cache_quant"]["per_domain"],
                 "short_eval": {"benchmark": "GSM8K 5-shot exact-match",
                                "delta_pct": round(drop, 1),
                                "verdict_by_benchmark":
                                    "looks_safe" if abs(drop) <= 2
                                    else "looks_unsafe"}},
        decision_statement=(
            f"KV-{args.kv_dtype} (vLLM native): worst-domain d={d_disp} ({worst_domain}) "
            f"vs tau=3.0 -> "
            f"{'SAFE — enable via vLLM --kv-cache-dtype flag fleet-wide' if d >= 3.0 else 'NOT certified — do not enable for this workload'}; "
            f"GSM8K moved {drop:+.1f} pts."),
        cert_path=cert_path,
        notes={"environment": environment_stamp(), "kv_cache_dtype": args.kv_dtype},
        out_path=os.path.join(HERE, "result.json"),
    )
    print("\nDone.")


if __name__ == "__main__":
    if "--_worker" in sys.argv:
        _worker_main()
    else:
        main()
