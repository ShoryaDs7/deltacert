"""
validation/kv_cache_quant/run_trajectory_fp8.py — trajectory certification
for a vLLM ENGINE-LEVEL change (KV-cache dtype), closing the gap the paper's
Limitations section names: "A trajectory profile d_COMM(t) for the Qwen
KV-fp8 collapse... is the first item of follow-up work."

Two-phase, subprocess-isolated, matching the existing single-position
kv_cache_quant/run_flagship.py's GPU-memory-reclamation discipline (vLLM v1's
engine-core background process does not reliably release GPU memory on a
plain `del llm` in time for a second engine to start in the same process) —
NOT collect_trajectory_vllm_two_engines' in-process two-engine calling
convention, which would need both 7B engines resident simultaneously.

  phase 1 (subprocess, baseline engine only):
      generate fresh greedy fp16 references on HumanEval prompts, filtered
      to >=64 tokens (same generation-time length filter documented for the
      Qwen NF4 trajectory run — a different mechanism from the Llama run's
      post-hoc repetition audit); teacher-force each reference through the
      SAME baseline engine via _teacher_forced_logits_vllm and save the
      per-position dense logit spans to disk.
  phase 2 (subprocess, fp8 engine only):
      teacher-force the SAME frozen references through the fp8-KV engine;
      load phase 1's saved baseline spans; compute per-position cosine and
      d_profile in-process (cheap, no GPU needed for this part).
  phase 3 (parent process):
      trajectory_layer_result() over all profiles -> certificate.

Uses the identical num_logprobs + dense-fill methodology as the single-
position KV-fp8 cert (via _logprob_dict_to_dense, capture_logits_vllm's own
convention) so d(t) is numerically comparable to the 3.72 single-position
value it is meant to be read alongside.

Run:
    python validation/kv_cache_quant/run_trajectory_fp8.py \
        --model Qwen/Qwen2.5-7B-Instruct --n-cases 50 --max-new 1024
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flagship_common import (  # noqa: E402
    assemble_and_save_result, environment_stamp, set_all_seeds,
)
from deltacert.collectors import (  # noqa: E402
    _teacher_forced_logits_vllm, d_profile, trajectory_layer_result,
)

HERE = os.path.dirname(os.path.abspath(__file__))
REFS_PATH = os.path.join(HERE, "trajectory_fp8_references.json")


def _gen_references(llm, tokenizer, n_cases: int, max_new: int) -> list:
    """Greedy fp16 completions on real HumanEval prompts, filtered to
    continuations >=64 tokens (the minimum length for a meaningful
    trajectory measurement — same rule as the Qwen NF4 trajectory run).
    Written once; reruns reuse the frozen file, same as trajectory/
    run_flagship.py's gen_references."""
    import vllm
    from datasets import load_dataset

    if os.path.exists(REFS_PATH):
        return json.load(open(REFS_PATH))

    ds = load_dataset("openai/openai_humaneval", split="test")
    params = vllm.SamplingParams(temperature=0.0, max_tokens=max_new)
    cases = []
    n_attempted = min(n_cases, len(ds))
    prompts = [ds[i]["prompt"] for i in range(n_attempted)]
    outs = llm.generate(prompts, params)
    for i, out in enumerate(outs):
        cont = out.outputs[0].text
        n_tok = len(tokenizer(cont, add_special_tokens=False)["input_ids"])
        if n_tok >= 64:
            cases.append({
                "task_id": ds[i]["task_id"],
                "prompt": ds[i]["prompt"],
                "continuation": cont,
            })
    print(f"  {len(cases)}/{n_attempted} references usable (>=64 tokens)")
    with open(REFS_PATH, "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2)
    return cases


def _repetition_audit(cases: list) -> list:
    """Same audit as the Llama trajectory clean-subset re-run (§5.2):
    flag references whose continuation is dominated by a tight repeated
    token/line loop, unrelated to the change under test. Returns task_ids
    flagged, for disclosure -- does NOT auto-exclude (matches the paper's
    "reconciled list, not a partial automated flag" discipline; a human
    should confirm before excluding anything)."""
    flagged = []
    for c in cases:
        words = c["continuation"].split()
        if len(words) < 20:
            continue
        run = 1
        for i in range(1, len(words)):
            if words[i] == words[i - 1]:
                run += 1
                if run >= 15:
                    flagged.append(c["task_id"])
                    break
            else:
                run = 1
    return flagged


def _worker_phase1_baseline(args) -> None:
    """Subprocess: load ONLY the baseline (auto KV dtype) engine, generate
    references (or load frozen ones), teacher-force each through this
    engine, save per-case dense logit spans + references to --out."""
    import numpy as np
    import vllm
    from transformers import AutoTokenizer

    set_all_seeds()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    llm = vllm.LLM(model=args.model, tensor_parallel_size=1,
                   gpu_memory_utilization=0.85, kv_cache_dtype="auto")

    cases = _gen_references(llm, tokenizer, args.n_cases, args.max_new)
    if not cases:
        raise SystemExit("No references passed the >=64 token filter.")

    try:
        vocab_size = llm.llm_engine.model_config.get_vocab_size()
    except Exception:
        vocab_size = llm.get_tokenizer().vocab_size

    spans = []
    for i, c in enumerate(cases):
        span, n_cont = _teacher_forced_logits_vllm(
            llm, tokenizer, c["prompt"], c["continuation"], vocab_size,
            num_logprobs=20, max_positions=args.max_new + 64)
        spans.append(span)
        print(f"  [{i + 1}/{len(cases)}] baseline span: {n_cont} positions")

    np.savez(args.out, *spans)
    with open(args.out + "_meta.json", "w", encoding="utf-8") as f:
        json.dump({"cases": cases, "vocab_size": vocab_size}, f, indent=2)
    print(f"  phase1 done: {len(spans)} baseline spans -> {args.out}.npz")


def _worker_phase2_modified(args) -> None:
    """Subprocess: load ONLY the fp8-KV engine, teacher-force the SAME
    frozen references (read from --meta), compute per-position cosine
    against the phase-1 baseline spans (--baseline-npz), write d_profile
    JSON per case to --out."""
    import numpy as np
    import vllm
    from transformers import AutoTokenizer
    from deltacert.collectors import cos_sim

    set_all_seeds()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    llm = vllm.LLM(model=args.model, tensor_parallel_size=1,
                   gpu_memory_utilization=0.85, kv_cache_dtype=args.kv_dtype)

    meta = json.load(open(args.meta, encoding="utf-8"))
    cases = meta["cases"]
    vocab_size = meta["vocab_size"]
    baseline_npz = np.load(args.baseline_npz)
    baseline_spans = [baseline_npz[f"arr_{i}"] for i in range(len(cases))]

    profiles = []
    for i, (c, base_span) in enumerate(zip(cases, baseline_spans)):
        mod_span, n_cont = _teacher_forced_logits_vllm(
            llm, tokenizer, c["prompt"], c["continuation"], vocab_size,
            num_logprobs=20, max_positions=args.max_new + 64)
        if mod_span.shape != base_span.shape:
            raise SystemExit(
                f"Shape mismatch on {c['task_id']}: baseline {base_span.shape} "
                f"vs modified {mod_span.shape} -- references must be teacher-"
                f"forced identically on both engines.")
        cos_sims = [cos_sim(base_span[t], mod_span[t]) for t in range(base_span.shape[0])]
        prof = d_profile(cos_sims)
        prof["task_id"] = c["task_id"]
        profiles.append(prof)
        print(f"  [{i + 1}/{len(cases)}] {c['task_id']}: d_min={prof['d_min']:.4f} "
              f"at position {prof['d_min_position']}/{prof['n_positions']}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(profiles, f, indent=2)
    print(f"  phase2 done: {len(profiles)} profiles -> {args.out}")


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--_phase1", action="store_true")
    ap.add_argument("--_phase2", action="store_true")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--n-cases", type=int, default=50)
    ap.add_argument("--max-new", type=int, default=1024)
    ap.add_argument("--kv-dtype", default="fp8")
    ap.add_argument("--out", default=None)
    ap.add_argument("--meta", default=None)
    ap.add_argument("--baseline-npz", default=None)
    return ap


def main() -> None:
    args = _build_parser().parse_args()

    base_npz = os.path.join(HERE, "trajectory_fp8_baseline_spans")
    meta_path = base_npz + "_meta.json"
    profiles_path = os.path.join(HERE, f"trajectory_{args.kv_dtype}_profiles.json")

    print("=== PHASE 1: baseline engine (auto KV dtype), teacher-forcing references ===")
    cmd1 = [sys.executable, __file__, "--_phase1",
            "--model", args.model, "--n-cases", str(args.n_cases),
            "--max-new", str(args.max_new), "--out", base_npz]
    proc1 = subprocess.run(cmd1)
    if proc1.returncode != 0:
        raise RuntimeError(f"phase1 subprocess failed (exit {proc1.returncode})")

    print(f"\n=== PHASE 2: {args.kv_dtype}-KV engine, teacher-forcing + comparing ===")
    cmd2 = [sys.executable, __file__, "--_phase2",
            "--model", args.model, "--kv-dtype", args.kv_dtype,
            "--meta", meta_path, "--baseline-npz", base_npz + ".npz",
            "--max-new", str(args.max_new), "--out", profiles_path]
    proc2 = subprocess.run(cmd2)
    if proc2.returncode != 0:
        raise RuntimeError(f"phase2 subprocess failed (exit {proc2.returncode})")

    profiles = json.load(open(profiles_path, encoding="utf-8"))
    cases = json.load(open(meta_path, encoding="utf-8"))["cases"]
    flagged = _repetition_audit(cases)
    if flagged:
        print(f"\n  repetition audit flagged {len(flagged)} reference(s): {flagged}")
        print("  (not auto-excluded -- confirm manually before excluding, same "
              "discipline as the Llama trajectory clean-subset re-run)")

    cert_layer = trajectory_layer_result(profiles)
    cert_path = os.path.join(HERE, f"cert_trajectory_{args.kv_dtype}.json")
    cert = {
        "model": args.model,
        "certified": cert_layer["certified"],
        "threshold_d": 3.0,
        "layers": {"trajectory_kv_fp8": cert_layer},
        "repetition_audit_flagged": flagged,
    }
    with open(cert_path, "w", encoding="utf-8") as f:
        json.dump(cert, f, indent=2)

    d_early = [p["d_per_position"][0] for p in profiles if p["d_per_position"]]
    early_mean = sum(d_early) / len(d_early) if d_early else float("nan")

    print(f"\n=== d(t) SUMMARY ===")
    print(f"  n_trajectories: {cert_layer['n_trajectories']}")
    print(f"  n_positions_total: {cert_layer['n_positions_total']}")
    print(f"  d_min (certified quantity): {cert_layer['d_comm']:.4f}")
    print(f"  d_min_position (worst trajectory): {cert_layer['d_min_position_in_worst_trajectory']}")
    print(f"  d_final_mean: {cert_layer['d_final_mean']:.4f}")
    print(f"  d at position 0, averaged over trajectories: {early_mean:.4f} "
          f"(compare to single-position cert's 3.72)")
    print(f"  certified: {cert_layer['certified']}")
    print(f"  cert written: {cert_path}")

    assemble_and_save_result(
        collector="kv_cache_quant", tier="A",
        run_id=f"trajectory_kv_{args.kv_dtype}",
        change={"baseline": f"{args.model} KV cache default dtype",
                "candidate": f"{args.model} KV cache {args.kv_dtype} (vLLM native, trajectory mode)",
                "change_type": "kv_cache_quant_trajectory"},
        business_goal={"reason": "localize where fp8 KV-cache divergence crosses "
                       "threshold across a full generation, closing the paper's "
                       "own disclosed follow-up-work gap",
                       "expected_gain": {}},
        workload={"task_family": "coding_agent_long_horizon",
                  "dataset": "humaneval_prompts", "num_prompts": cert_layer["n_trajectories"]},
        metrics={"d_comm": round(cert_layer["d_comm"], 4), "tau": 3.0,
                 "downstream_delta": {"early_position_d_vs_single_position_cert": round(3.72 - early_mean, 4)},
                 "trajectory": {"measured": True,
                               "safe_until_token": 0 if not cert_layer["certified"] else None,
                               "d_min_position": cert_layer["d_min_position_in_worst_trajectory"],
                               "d_final_mean": round(cert_layer["d_final_mean"], 4)}},
        decision_statement=(
            f"Trajectory KV-{args.kv_dtype} on {args.model}: d_min={cert_layer['d_comm']:.4f} "
            f"at position {cert_layer['d_min_position_in_worst_trajectory']} vs tau=3.0 -> "
            f"{'SAFE' if cert_layer['certified'] else 'NOT certified'}. "
            f"Early-position d={early_mean:.4f} (single-position cert: 3.72)."),
        cert_path=cert_path,
        notes={"environment": environment_stamp(), "kv_cache_dtype": args.kv_dtype},
        out_path=os.path.join(HERE, f"result_trajectory_{args.kv_dtype}.json"),
    )
    print("\nDone.")


if __name__ == "__main__":
    if "--_phase1" in sys.argv:
        _worker_phase1_baseline(_build_parser().parse_args())
    elif "--_phase2" in sys.argv:
        _worker_phase2_modified(_build_parser().parse_args())
    else:
        main()
