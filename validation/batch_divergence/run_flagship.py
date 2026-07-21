"""
validation/batch_divergence/run_flagship.py — FLAGSHIP CASE STUDY

Company scenario: "We cap batch size conservatively out of fear that
continuous-batching kernels behave differently under load — what's the
actual maximum safe concurrency on this GPU?"

Unlike weight_quant/kv_cache_quant/spec_decoding, nothing is being
compressed or approximated here — same weights, same precision. The only
variable is how many requests are scheduled together. So a certified max
batch size is close to free money: more concurrent users per GPU, zero
model-quality tradeoff, purely from confidence instead of caution.

Sweep across batch sizes [8, 32, 64] (real vLLM engine, real continuous
batching): find the LARGEST batch size where d_comm stays >= tau on every
canary domain. That crossing point is the certified maximum safe batch size.

Each vLLM engine build (baseline batch=1, then each swept batch size) runs
in its own SEPARATE OS PROCESS. This script used to build 4 sequential
engines in one Python process with a GPU-memory-free polling wait between
them — the same pattern that failed for spec_decoding: a 90s wait for the
GPU to actually release memory was not sufficient, because vLLM v1's
background EngineCore process is not reliably torn down by
del+gc.collect()+empty_cache() in time. Subprocess exit guarantees real OS
reclamation, the same fix already proven for kv_cache_quant/spec_decoding.

Triple measurement per batch size:
  1. d_COMM      — collect_batch_divergence (batch=1 vs batch=N, same
                   in-process vLLM engine, run inside its own subprocess):
                   per-domain worst-case d
  2. downstream  — GSM8K exact-match at batch=1 (looped single requests)
                   vs batch=N (one concurrent call) — real generations both
                   ways, each batch size's engine in its own subprocess
  3. short_eval  — the GSM8K delta doubles as the standard-benchmark number,
                   recorded explicitly

Run (RunPod / GPU):
    python validation/batch_divergence/run_flagship.py \
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

HERE = os.path.dirname(os.path.abspath(__file__))
BATCH_SIZES = [8, 32, 64]


def _worker_gsm8k() -> None:
    """Builds ONE vLLM engine, measures GSM8K acc at the given batch size,
    writes result to --out as JSON. Invoked as a fresh subprocess only."""
    import vllm
    from flagship_common import (
        build_gsm8k_prompt, extract_final_number, load_gsm8k, load_gsm8k_fewshot,
        set_all_seeds,
    )

    ap = argparse.ArgumentParser()
    ap.add_argument("--_worker_gsm8k", action="store_true")
    ap.add_argument("--model", required=True)
    ap.add_argument("--gsm8k-n", type=int, required=True)
    ap.add_argument("--batch-size", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    set_all_seeds()
    problems = load_gsm8k(args.gsm8k_n)
    fewshot = load_gsm8k_fewshot()
    prompts = [build_gsm8k_prompt(p["question"], fewshot) for p in problems]
    params = vllm.SamplingParams(temperature=0.0, max_tokens=256, stop=["Question:"])

    llm = vllm.LLM(model=args.model, tensor_parallel_size=1, gpu_memory_utilization=0.85)
    correct = 0
    for i in range(0, len(prompts), args.batch_size):
        chunk_prompts = prompts[i:i + args.batch_size]
        chunk_problems = problems[i:i + args.batch_size]
        outs = llm.generate(chunk_prompts, params)
        for prob, out in zip(chunk_problems, outs):
            pred = extract_final_number(out.outputs[0].text)
            correct += int(pred == prob["gold"])
    acc = correct / len(problems)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"acc": acc}, f)
    print(f"  worker done: batch_size={args.batch_size} acc={acc:.3f}")


def _worker_dcomm() -> None:
    """Runs collect_batch_divergence (its own in-process engine) inside a
    fresh subprocess, starting from a clean GPU. Writes cos_sims to --out
    as .npy."""
    import numpy as np
    from flagship_common import load_canaries_with_domains, set_all_seeds
    from deltacert.collectors import collect_batch_divergence

    ap = argparse.ArgumentParser()
    ap.add_argument("--_worker_dcomm", action="store_true")
    ap.add_argument("--model", required=True)
    ap.add_argument("--batch-size", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    set_all_seeds()
    canaries, _domains = load_canaries_with_domains()
    cos_sims = collect_batch_divergence(args.model, canaries,
                                        batched_size=args.batch_size, num_logprobs=20)
    np.save(args.out, cos_sims)
    print(f"  worker done: d_comm cos_sims computed for batch_size={args.batch_size}")


def _run_subprocess(cmd: list) -> None:
    print(f"  spawning subprocess: {' '.join(cmd[2:5])}...")
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"worker subprocess failed (exit {proc.returncode}): {cmd}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--gsm8k-n", type=int, default=100)
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=BATCH_SIZES)
    args = ap.parse_args()

    set_all_seeds()
    canaries, domains = load_canaries_with_domains()

    print("=== BASELINE: batch=1 (isolated single requests, separate subprocess) ===")
    b1_out = os.path.join(HERE, "_gsm8k_b1.json")
    _run_subprocess([sys.executable, __file__, "--_worker_gsm8k",
                     "--model", args.model, "--gsm8k-n", str(args.gsm8k_n),
                     "--batch-size", "1", "--out", b1_out])
    with open(b1_out, encoding="utf-8") as f:
        acc_b1 = json.load(f)["acc"]
    print(f"  batch=1 GSM8K acc = {acc_b1:.3f}")

    max_safe_batch = None
    for batch_size in sorted(args.batch_sizes):
        print(f"\n=== ROW: batch={batch_size} ===")

        print("  [1/2] d_COMM: collect_batch_divergence (separate subprocess, per domain, worst domain certifies) …")
        import numpy as np
        dcomm_out = os.path.join(HERE, f"_dcomm_batch{batch_size}.npy")
        _run_subprocess([sys.executable, __file__, "--_worker_dcomm",
                         "--model", args.model, "--batch-size", str(batch_size),
                         "--out", dcomm_out])
        cos_sims = np.load(dcomm_out)
        cert = build_cert("batch_divergence", cos_sims, model_id=args.model,
                          domain_labels=domains)
        d = cert["layers"]["batch_divergence"]["d_comm"]
        d = float("inf") if d == "inf" else d
        worst_domain = cert["layers"]["batch_divergence"]["worst_domain"]
        cert_path = os.path.join(HERE, f"cert_batch{batch_size}.json")
        with open(cert_path, "w", encoding="utf-8") as f:
            json.dump(cert, f, indent=2)
        certified = d >= 3.0
        print(f"      d = {d}  certified = {certified}  worst_domain = {worst_domain}")
        if certified:
            max_safe_batch = batch_size

        print(f"  [2/2] downstream: GSM8K at batch={batch_size} (separate subprocess) …")
        bn_out = os.path.join(HERE, f"_gsm8k_b{batch_size}.json")
        _run_subprocess([sys.executable, __file__, "--_worker_gsm8k",
                         "--model", args.model, "--gsm8k-n", str(args.gsm8k_n),
                         "--batch-size", str(batch_size), "--out", bn_out])
        with open(bn_out, encoding="utf-8") as f:
            acc_bn = json.load(f)["acc"]
        drop = (acc_bn - acc_b1) * 100.0
        print(f"      batch=1 acc={acc_b1:.3f}  batch={batch_size} acc={acc_bn:.3f} ({drop:+.1f} pts)")

        d_disp = "inf" if d == float("inf") else round(d, 4)
        assemble_and_save_result(
            collector="batch_divergence", tier="A",
            run_id=f"flagship_batch_divergence_b{batch_size}",
            change={"baseline": f"{args.model} batch=1",
                    "candidate": f"{args.model} batch={batch_size}",
                    "change_type": "batch_divergence"},
            business_goal={"reason": "safely raise concurrent batch size — "
                                     "more requests served per GPU with the "
                                     "SAME weights and precision, zero quality "
                                     "tradeoff being made",
                           "expected_gain": {"concurrency_x": batch_size}},
            workload={"task_family": "math_reasoning", "dataset": "gsm8k",
                      "num_prompts": len(canaries)},
            metrics={"d_comm": d_disp, "tau": 3.0,
                     "downstream_delta": {"gsm8k_acc_drop_pts": round(drop, 2)},
                     "per_domain": cert["layers"]["batch_divergence"]["per_domain"],
                     "short_eval": {"benchmark": "GSM8K 5-shot exact-match",
                                    "delta_pct": round(drop, 1),
                                    "verdict_by_benchmark":
                                        "looks_safe" if abs(drop) <= 2
                                        else "looks_unsafe"}},
            decision_statement=(
                f"batch={batch_size}: worst-domain d={d_disp} ({worst_domain}) vs tau=3.0 -> "
                f"{'SAFE — serve this many concurrent requests per GPU' if certified else 'NOT certified — continuous-batching changed outputs materially at this concurrency'}; "
                f"GSM8K moved {drop:+.1f} pts vs batch=1."),
            cert_path=cert_path,
            notes={"environment": environment_stamp(), "batch_size": batch_size},
            out_path=os.path.join(HERE, "result.json" if batch_size == sorted(args.batch_sizes)[0]
                                  else f"result_batch{batch_size}.json"),
        )

    print(f"\nDone. Certified maximum safe batch size: {max_safe_batch or 'none tested safe'}")
    print("Run `python -m deltacert.validation.harness` to regenerate tables.")


if __name__ == "__main__":
    if "--_worker_gsm8k" in sys.argv:
        _worker_gsm8k()
    elif "--_worker_dcomm" in sys.argv:
        _worker_dcomm()
    else:
        main()
