"""
validation/weight_quant/humaneval_followup.py — FOLLOW-UP TO THE FLAGSHIP

Resolves the one open question the main weight_quant sweep can't answer on
its own: the domain-stratified d_comm sweep flagged "code" as the weakest
domain for nf4 quantization, but the sweep's only downstream check is GSM8K
(math) — structurally blind to whether code generation actually degrades.

This runs the REAL, industry-standard HumanEval pass@1 benchmark (the
official `human-eval` package: real code generation, real sandboxed
execution against the official test suites, real pass/fail — no homemade
scoring) on fp16 vs nf4, to distinguish the two honest possibilities:

  - code pass@1 DROPS under nf4 while GSM8K stayed flat
        -> a real result: d_comm caught a regression standard math evals
           miss entirely. This is evidence E — the "caught what evals
           missed" story, with execution-based proof, not just distribution
           divergence.
  - code pass@1 stays flat, same as GSM8K
        -> tau=3.0 is provably too strict for weight_quant's cosine-based
           d_comm; recalibrate the per-collector default from this data,
           don't ship a default that fails a config which works fine.

Either outcome is a legitimate, useful result — that's why this is worth
running before calling weight_quant's evidence complete.

Requires the human-eval package's sandboxed exec enabled (it ships disabled
by default as a safety guard against executing untrusted code — see the
patch step below, safe in an ephemeral GPU pod we control).

Run (after the main weight_quant sweep, reusing the same model):
    python validation/weight_quant/humaneval_followup.py \
        --model meta-llama/Llama-3.1-8B-Instruct --n-problems 80
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flagship_common import set_all_seeds  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def _enable_human_eval_execution():
    """human-eval ships with real code execution commented out by default
    (a safety guard against running untrusted model completions). We're in
    an ephemeral, isolated GPU pod we control — enabling it here, once,
    is the documented opt-in the package expects."""
    import human_eval.execution as execution_module
    src_path = execution_module.__file__
    with open(src_path, encoding="utf-8") as f:
        src = f.read()
    marker = "exec(check_program, exec_globals)"
    if marker in src and f"# {marker}" not in src:
        return  # already enabled
    patched = src.replace(f"# {marker}", marker)
    if patched == src:
        raise RuntimeError(
            "Could not find the expected commented-out exec() line in "
            f"human_eval/execution.py ({src_path}) — package version may "
            "have changed; inspect manually before proceeding."
        )
    with open(src_path, "w", encoding="utf-8") as f:
        f.write(patched)
    print(f"  patched {src_path}: enabled sandboxed code execution")


def load_fp16(model_name: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto")
    model.eval()
    return model, tok


def load_nf4(model_name: str):
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    cfg = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=cfg, device_map="auto")
    model.eval()
    return model


def free(model):
    import torch
    del model
    gc.collect()
    torch.cuda.empty_cache()


def generate_completions(model, tokenizer, problems: dict, max_new_tokens: int = 384):
    import torch
    completions = {}
    for task_id, prob in problems.items():
        inputs = tokenizer(prob["prompt"], return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                skip_special_tokens=True)
        # Truncate at the next top-level def/class to strip trailing
        # generation noise past the intended function body.
        for stop in ("\ndef ", "\nclass ", "\nif __name__"):
            if stop in text:
                text = text.split(stop)[0]
        completions[task_id] = text
    return completions


def pass_at_1(model_label: str, model, tokenizer, problems: dict, out_dir: str,
             problem_file: str) -> float:
    from human_eval.data import write_jsonl
    from human_eval.evaluation import evaluate_functional_correctness

    print(f"  generating {len(problems)} completions ({model_label}) …")
    completions = generate_completions(model, tokenizer, problems)
    samples = [{"task_id": tid, "completion": c} for tid, c in completions.items()]
    sample_path = os.path.join(out_dir, f"humaneval_samples_{model_label}.jsonl")
    write_jsonl(sample_path, samples)

    print(f"  executing against real test suites ({model_label}) …")
    # evaluate_functional_correctness defaults to the FULL 164-problem
    # HumanEval set and asserts every problem was attempted — pass our
    # actual subset explicitly so it checks only what we generated for.
    results = evaluate_functional_correctness(sample_path, k=[1], problem_file=problem_file)
    return results["pass@1"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--n-problems", type=int, default=80)
    args = ap.parse_args()

    set_all_seeds()
    _enable_human_eval_execution()

    from human_eval.data import read_problems, write_jsonl
    all_problems = read_problems()
    task_ids = list(all_problems.keys())[:args.n_problems]
    problems = {tid: all_problems[tid] for tid in task_ids}
    print(f"Loaded {len(problems)} real HumanEval problems.")

    # evaluate_functional_correctness needs a problem_file matching exactly
    # the subset we generated for — write it once, reuse for both models.
    problem_file = os.path.join(HERE, "humaneval_subset_problems.jsonl")
    write_jsonl(problem_file, list(problems.values()))

    print("\n=== fp16 ===")
    fp16_model, tokenizer = load_fp16(args.model)
    fp16_pass1 = pass_at_1("fp16", fp16_model, tokenizer, problems, HERE, problem_file)
    free(fp16_model)
    print(f"  fp16 pass@1 = {fp16_pass1:.3f}")

    print("\n=== nf4 ===")
    nf4_model = load_nf4(args.model)
    nf4_pass1 = pass_at_1("nf4", nf4_model, tokenizer, problems, HERE, problem_file)
    free(nf4_model)
    print(f"  nf4 pass@1 = {nf4_pass1:.3f}")

    drop_pts = (nf4_pass1 - fp16_pass1) * 100.0
    print(f"\nHumanEval pass@1: fp16={fp16_pass1:.3f} -> nf4={nf4_pass1:.3f} "
          f"({drop_pts:+.1f} pts)")

    if abs(drop_pts) <= 2.0:
        verdict = ("Code pass@1 stayed flat (same signal as GSM8K). This is "
                   "evidence tau=3.0 is too strict for weight_quant's "
                   "cosine-based d_comm on the 'code' domain — recalibrate "
                   "the per-collector default from this data.")
    else:
        verdict = ("Code pass@1 genuinely dropped while GSM8K stayed flat. "
                  "d_comm caught a real regression standard math evals miss "
                  "entirely — this is a genuine 'caught what evals missed' "
                  "result, backed by real code execution, not just "
                  "distribution divergence.")
    print(f"\nVerdict: {verdict}")

    report = {
        "model": args.model,
        "n_problems": len(problems),
        "fp16_pass_at_1": fp16_pass1,
        "nf4_pass_at_1": nf4_pass1,
        "drop_pts": round(drop_pts, 2),
        "verdict": verdict,
    }
    report_path = os.path.join(HERE, "humaneval_followup_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport written: {report_path}")


if __name__ == "__main__":
    main()
