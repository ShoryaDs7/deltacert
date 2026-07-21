"""validation/engine_swap/assemble_result.py
Called by run_flagship.sh after both vLLM envs have run."""
import argparse, glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flagship_common import assemble_and_save_result, environment_stamp, read_cert_d

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--ver-a", required=True)
ap.add_argument("--ver-b", required=True)
ap.add_argument("--here", required=True)
args = ap.parse_args()

d = read_cert_d(os.path.join(args.here, "cert_engine_swap.json"), "engine_swap")

_canaries_path = os.path.join(os.path.dirname(args.here), "canaries_v1.txt")
with open(_canaries_path, encoding="utf-8") as _f:
    num_prompts = sum(1 for _line in _f if _line.strip())


def gsm8k_acc(tag):
    for p in glob.glob(os.path.join(args.here, f"gsm8k_env{tag}", "**", "*.json"),
                       recursive=True):
        try:
            res = json.load(open(p)).get("results", {}).get("gsm8k", {})
            for k, v in res.items():
                if k.startswith("exact_match") and isinstance(v, float):
                    return v
        except Exception:
            pass
    return None


acc_a, acc_b = gsm8k_acc("A"), gsm8k_acc("B")
if acc_a is None or acc_b is None:
    print("[!] downstream numbers missing — run lm-eval in both envs first.")
    sys.exit(1)
drop = (acc_b - acc_a) * 100.0
d_disp = "inf" if d == float("inf") else round(d, 4)

assemble_and_save_result(
    collector="engine_swap", tier="A",
    run_id=f"flagship_engine_swap_{args.ver_a}_to_{args.ver_b}",
    change={"baseline": f"vLLM {args.ver_a}", "candidate": f"vLLM {args.ver_b}",
            "change_type": "engine_swap"},
    business_goal={"reason": "take each vLLM release's perf gains same-week "
                             "instead of staying pinned for months",
                   "expected_gain": {"upgrade_lag_weeks_saved": 8}},
    workload={"task_family": "math_reasoning", "dataset": "gsm8k",
              "num_prompts": num_prompts},
    metrics={"d_comm": d_disp, "tau": 3.0,
             "downstream_delta": {"gsm8k_acc_drop_pts": round(drop, 2)},
             "short_eval": {"benchmark": "GSM8K (lm-eval, both engines)",
                            "delta_pct": round(drop, 1),
                            "verdict_by_benchmark":
                                "looks_safe" if abs(drop) <= 2 else "looks_unsafe"}},
    decision_statement=(
        f"vLLM {args.ver_a} -> {args.ver_b}: d={d_disp} -> "
        f"{'upgrade CERTIFIED — ship it' if d >= 3.0 else 'upgrade BLOCKED — outputs changed materially'}; "
        f"GSM8K moved {drop:+.1f} pts across engines."),
    cert_path=os.path.join(args.here, "cert_engine_swap.json"),
    notes={"environment": environment_stamp(),
           "incident_class": "vLLM #36117 (silent output change on upgrade)"},
    out_path=os.path.join(args.here, "result.json"),
)
print("Done.")
