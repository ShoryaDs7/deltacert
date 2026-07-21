"""validation/provider_drift/check.py — run baseline day 0, then weekly.

Company scenario: "the hosted model changed under us and customers found out
first." (documented: accuracy drops in the majority of silent alias updates.)

Day 0:   python check.py            -> saves baseline.npz, exits
Weekly:  python check.py            -> d vs baseline + canary accuracy,
                                       cert committed with timestamp

Set env vars before running:
    OPENAI_API_KEY=sk-...
    DELTACERT_API_BASE=https://api.openai.com/v1   (default)
    DELTACERT_API_MODEL=gpt-4o-mini                (default)
"""
import argparse, json, os, sys, time, urllib.request
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flagship_common import (assemble_and_save_result, build_cert,
                             environment_stamp, extract_final_number,
                             load_canaries_with_domains, load_gsm8k, set_all_seeds)
from deltacert.collectors import collect_provider_drift, CollectionError

HERE = os.path.dirname(os.path.abspath(__file__))
BASELINE = os.path.join(HERE, "baseline.npz")
API_BASE = os.environ.get("DELTACERT_API_BASE", "https://api.openai.com/v1")
API_MODEL = os.environ.get("DELTACERT_API_MODEL", "gpt-4o-mini")


def canary_accuracy(problems):
    """Real downstream: ask the API the 50 known-answer canaries, score."""
    key = os.environ["OPENAI_API_KEY"]
    correct = 0
    for p in problems:
        body = json.dumps({"model": API_MODEL, "max_tokens": 300,
                           "temperature": 0,
                           "messages": [{"role": "user",
                                         "content": p["question"] +
                                         "\nGive the final number only."}]}).encode()
        req = urllib.request.Request(f"{API_BASE}/chat/completions", data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {key}"})
        with urllib.request.urlopen(req, timeout=60) as r:
            text = json.load(r)["choices"][0]["message"]["content"]
        pred = extract_final_number(text)
        correct += int(pred == p["gold"])
    return correct / len(problems)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--establish", action="store_true",
                    help="deprecated flag — first run auto-establishes baseline")
    args = ap.parse_args()
    set_all_seeds()
    canaries, domains = load_canaries_with_domains()
    problems = load_gsm8k(50)
    stamp = time.strftime("%Y%m%d")

    try:
        sims = collect_provider_drift(API_BASE, API_MODEL, canaries, BASELINE)
    except CollectionError as e:
        if "saved as the new baseline" in str(e):
            acc = canary_accuracy(problems)
            json.dump({"date": stamp, "baseline_established": True,
                       "canary_accuracy": acc},
                      open(os.path.join(HERE, f"log_{stamp}.json"), "w"), indent=2)
            print(f"[DeltaCert] Baseline established. Canary acc={acc:.3f}. "
                  "Re-run weekly.")
            return
        raise

    acc = canary_accuracy(problems)
    prev = sorted(f for f in os.listdir(HERE) if f.startswith("log_"))
    base_acc = json.load(open(os.path.join(HERE, prev[0])))["canary_accuracy"]
    drop = (acc - base_acc) * 100.0
    cert = build_cert("provider_drift", sims, model_id=API_MODEL,
                      domain_labels=domains)
    d = cert["layers"]["provider_drift"]["d_comm"]
    d = float("inf") if d == "inf" else d
    cert_path = os.path.join(HERE, f"cert_{stamp}.json")
    json.dump(cert, open(cert_path, "w"), indent=2)
    d_disp = "inf" if d == float("inf") else round(d, 4)
    worst_domain = cert["layers"]["provider_drift"]["worst_domain"]

    assemble_and_save_result(
        collector="provider_drift", tier="A",
        run_id=f"drift_{API_MODEL}_{stamp}",
        change={"baseline": f"{API_MODEL} @ baseline date",
                "candidate": f"{API_MODEL} @ {stamp}",
                "change_type": "provider_drift"},
        business_goal={"reason": "catch silent provider model updates before "
                                 "customers do"},
        workload={"task_family": "canary_monitoring", "dataset": "gsm8k_50",
                  "num_prompts": len(canaries)},
        metrics={"d_comm": d_disp, "tau": 3.0,
                 "downstream_delta": {"canary_acc_drop_pts": round(drop, 2)},
                 "per_domain": cert["layers"]["provider_drift"]["per_domain"]},
        decision_statement=(
            f"{API_MODEL} on {stamp}: worst-domain d={d_disp} ({worst_domain}) vs baseline -> "
            f"{'no material drift' if d >= 3.0 else 'DRIFT DETECTED — provider changed the model; review before trusting outputs'}; "
            f"canary accuracy {drop:+.1f} pts vs baseline."),
        cert_path=cert_path,
        notes={"environment": environment_stamp()},
        out_path=os.path.join(HERE, "result.json"),
    )
    print(f"d={d_disp}  canary_acc={acc:.3f} ({drop:+.1f} pts). "
          "Commit cert + result to the repo for the public timeline.")


if __name__ == "__main__":
    main()
