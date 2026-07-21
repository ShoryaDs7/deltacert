"""
validation/provider_drift/check_snapshot_vs_alias.py — SAME-DAY DRIFT PROXY

The real provider_drift test (check.py) needs two runs across real time
(baseline today, re-run weeks later) to measure actual drift — a floating
alias like "gpt-4o-mini" only drifts when OpenAI re-points it, which doesn't
happen on a predictable schedule.

This script gives an immediate, same-day proxy instead: compare a PINNED
dated snapshot (e.g. gpt-4o-mini-2024-07-18, immutable — OpenAI never
changes what a dated snapshot returns) against the CURRENT floating alias
(gpt-4o-mini) right now. If OpenAI has already re-pointed the alias to a
newer snapshot since 2024-07-18, this shows up as real measured divergence
today, with no waiting required.

Caveat (explicit, not hidden): this is a proxy, not the real weekly-drift
measurement. It only detects drift that already happened between the pinned
snapshot's release and today; it does not detect future re-pointing. It also
depends on the pinned snapshot still being live (OpenAI eventually retires
old dated snapshots) — reusing capture_logits_openai_api (collectors.py,
untouched) for both calls, exactly as the real test does.

Run:
    OPENAI_API_KEY=sk-... python validation/provider_drift/check_snapshot_vs_alias.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flagship_common import (
    assemble_and_save_result, build_cert, environment_stamp,
    extract_final_number, load_canaries_with_domains, load_gsm8k, set_all_seeds,
)
from deltacert.collectors import capture_logits_openai_api, cos_sims_from_logit_matrices

HERE = os.path.dirname(os.path.abspath(__file__))
API_BASE = os.environ.get("DELTACERT_API_BASE", "https://api.openai.com/v1")
PINNED_MODEL = os.environ.get("DELTACERT_PINNED_MODEL", "gpt-4o-mini-2024-07-18")
ALIAS_MODEL = os.environ.get("DELTACERT_ALIAS_MODEL", "gpt-4o-mini")


def canary_accuracy(model: str, problems, key: str) -> float:
    import urllib.request
    correct = 0
    for p in problems:
        body = json.dumps({"model": model, "max_tokens": 300, "temperature": 0,
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
    key = os.environ["OPENAI_API_KEY"]
    set_all_seeds()
    canaries, domains = load_canaries_with_domains()
    problems = load_gsm8k(50)

    print(f"=== provider_drift SAME-DAY PROXY: pinned {PINNED_MODEL} vs alias {ALIAS_MODEL} ===")
    print(f"[1/4] capturing logits: pinned snapshot {PINNED_MODEL} ...")
    pinned_logits = capture_logits_openai_api(API_BASE, PINNED_MODEL, canaries, api_key=key, num_logprobs=20)
    print(f"[2/4] capturing logits: current alias {ALIAS_MODEL} ...")
    alias_logits = capture_logits_openai_api(API_BASE, ALIAS_MODEL, canaries, api_key=key, num_logprobs=20)

    cos_sims = cos_sims_from_logit_matrices(pinned_logits, alias_logits)
    cert = build_cert("provider_drift", cos_sims,
                      model_id=f"{ALIAS_MODEL} vs pinned {PINNED_MODEL}",
                      domain_labels=domains)
    d = cert["layers"]["provider_drift"]["d_comm"]
    d = float("inf") if d == "inf" else d
    worst_domain = cert["layers"]["provider_drift"]["worst_domain"]
    cert_path = os.path.join(HERE, "cert_snapshot_vs_alias.json")
    with open(cert_path, "w", encoding="utf-8") as f:
        json.dump(cert, f, indent=2)
    print(f"  d = {d}  ({worst_domain})")

    print(f"[3/4] GSM8K via pinned {PINNED_MODEL} ...")
    acc_pinned = canary_accuracy(PINNED_MODEL, problems, key)
    print(f"[4/4] GSM8K via alias {ALIAS_MODEL} ...")
    acc_alias = canary_accuracy(ALIAS_MODEL, problems, key)
    drop = (acc_alias - acc_pinned) * 100.0
    print(f"  pinned acc={acc_pinned:.3f}  alias acc={acc_alias:.3f}  ({drop:+.1f} pts)")

    d_disp = "inf" if d == float("inf") else round(d, 4)
    assemble_and_save_result(
        collector="provider_drift", tier="A",
        run_id="provider_drift_snapshot_vs_alias_proxy",
        change={"baseline": f"{PINNED_MODEL} (immutable pinned snapshot)",
                "candidate": f"{ALIAS_MODEL} (current floating alias, today)",
                "change_type": "provider_drift_proxy"},
        business_goal={"reason": "same-day proxy for provider drift risk — "
                                 "detects drift that already happened between "
                                 "the pinned snapshot's release and today, "
                                 "without waiting weeks for a real re-run"},
        workload={"task_family": "math_reasoning", "dataset": "gsm8k_50",
                  "num_prompts": len(canaries)},
        metrics={"d_comm": d_disp, "tau": 3.0,
                 "downstream_delta": {"canary_acc_drop_pts": round(drop, 2)},
                 "per_domain": cert["layers"]["provider_drift"]["per_domain"]},
        decision_statement=(
            f"SAME-DAY PROXY (not the real weekly drift measurement): "
            f"{ALIAS_MODEL} vs pinned {PINNED_MODEL}: worst-domain d={d_disp} "
            f"({worst_domain}) -> "
            f"{'no material difference detected between pinned snapshot and current alias' if d >= 3.0 else 'DIVERGENCE DETECTED — the alias has already moved away from this pinned snapshot; review before trusting outputs'}; "
            f"GSM8K {drop:+.1f} pts vs pinned snapshot."),
        cert_path=cert_path,
        notes={"environment": environment_stamp(),
               "caveat": "This is a same-day proxy comparing a pinned dated "
                        "snapshot to the current floating alias, NOT the "
                        "real weekly-cadence drift measurement in check.py. "
                        "It only detects drift that already occurred between "
                        "the pinned snapshot's release and today."},
        out_path=os.path.join(HERE, "result_snapshot_vs_alias.json"),
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
