"""
Qwen self-calibration (Test 2): derive Qwen-specific weight_quant tau values
from the real sweep already measured (int8, nf4 = bnb family; gptq_int4/3/2
= gptq family), using the exact PER-FAMILY tau-selection rule now shipped in
deltacert.py::calibrate_layer (lowest d_comm among configs in the SAME
method_family that showed no material downstream damage; families with zero
undamaged configs are marked provisional).

Runs locally against the already-verified saved certs/gsm8k files — no GPU
needed, since d_comm and downstream accuracy are already measured and saved
to disk. Raw cos_sims aren't saved on disk (only the derived per-domain
d_comm), so this replicates calibrate_layer's per-family algorithm directly
on the trusted, already-computed d_comm values rather than re-deriving them
from cos_sims — mathematically identical since d_comm(cos_sims) is
deterministic and these are the same values certify_layer already produced
and saved in each cert.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
QWEN = os.path.dirname(HERE)
GPTQ = os.path.join(QWEN, "weight_quant_gptq")

DEGRADATION_THRESHOLD_PTS = 2.0

configs = [
    ("int8", "bnb", os.path.join(HERE, "qwen_cert_int8.json"), os.path.join(HERE, "qwen_gsm8k_int8.json"), "int8_acc"),
    ("nf4", "bnb", os.path.join(HERE, "qwen_cert_nf4.json"), os.path.join(HERE, "qwen_gsm8k_nf4.json"), "nf4_acc"),
    ("gptq_int4", "gptq", os.path.join(GPTQ, "cert_gptq_int4.json"), os.path.join(GPTQ, "gsm8k_gptq_int4.json"), "gptq_int4_acc"),
    ("gptq_int3", "gptq", os.path.join(GPTQ, "cert_gptq_int3.json"), os.path.join(GPTQ, "gsm8k_gptq_int3.json"), "gptq_int3_acc"),
    ("gptq_int2", "gptq", os.path.join(GPTQ, "cert_gptq_int2.json"), os.path.join(GPTQ, "gsm8k_gptq_int2.json"), "gptq_int2_acc"),
]

by_family = {}
for name, family, cert_path, gsm_path, acc_key in configs:
    cert = json.load(open(cert_path))["layers"]["weight_quant"]
    gsm = json.load(open(gsm_path))
    fp16_acc = gsm["fp16_acc"]
    q_acc = gsm[acc_key]
    drop_pts = (fp16_acc - q_acc) * 100.0
    d_raw = cert["d_comm"]
    materially_degraded = abs(drop_pts) > DEGRADATION_THRESHOLD_PTS
    row = {
        "name": name,
        "d_comm": d_raw,
        "downstream_drop_pts": round(drop_pts, 4),
        "downstream_drop_pts_source": "user-provided",
        "materially_degraded": materially_degraded,
        "worst_domain": cert["worst_domain"],
    }
    by_family.setdefault(family, []).append(row)

families_out = {}
for family, rows in by_family.items():
    safe_rows = [r for r in rows if not r["materially_degraded"]]
    damaged_rows = [r for r in rows if r["materially_degraded"]]
    one_sided_from_above = not safe_rows     # every config damaged
    one_sided_from_below = not damaged_rows  # every config clean
    provisional = one_sided_from_above or one_sided_from_below

    if safe_rows:
        finite_safe = [r["d_comm"] for r in safe_rows if r["d_comm"] != float("inf")]
        tau = min(finite_safe) if finite_safe else 0.0
    else:
        finite_damaged = [r["d_comm"] for r in damaged_rows if r["d_comm"] != float("inf")]
        tau = (max(finite_damaged) + 1.0) if finite_damaged else 3.0

    for r in rows:
        r["certified_at_calibrated_tau"] = (
            True if r["d_comm"] == "inf" else r["d_comm"] >= tau
        )
        r["real_damage_missed_at_calibrated_tau"] = (
            r["materially_degraded"] and r["certified_at_calibrated_tau"]
        )

    families_out[family] = {
        "calibrated_tau": round(tau, 4),
        "provisional": provisional,
        "n_configs": len(rows),
        "n_safe_configs": len(safe_rows),
        "n_damaged_configs": len(damaged_rows),
        "rows": rows,
        "caveat": (
            (
                f"PROVISIONAL: every '{family}' config in this sweep (n={len(rows)}) "
                "showed real downstream damage - no undamaged config exists to set a "
                "two-sided safe floor from. tau is a lower bound consistent with the "
                "data (placed just above the worst observed damage), not a measured "
                "safe/unsafe boundary."
            ) if one_sided_from_above else (
                f"PROVISIONAL: every '{family}' config in this sweep (n={len(rows)}) "
                "was undamaged - no damaged config exists to set a two-sided boundary "
                "from. tau is placed at the lowest observed (undamaged) d_comm, so the "
                "boundary-exact config certifies with zero margin ('>=', not '>') - "
                "measurement noise could flip its verdict."
            ) if one_sided_from_below else (
                f"Calibrated from n={len(rows)} '{family}' config(s) on "
                "Qwen2.5-7B-Instruct — not a universal constant."
            )
        ),
    }

result = {
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "layer": "weight_quant",
    "families": families_out,
    "n_families": len(families_out),
    "downstream_degradation_threshold_pts": DEGRADATION_THRESHOLD_PTS,
    "caveat": (
        "d_comm is not comparable across quantization methods, so tau is "
        "calibrated independently per method_family (bnb, gptq) — pooling "
        "them into one tau produces a false-safe (see the pooled result "
        "this replaces: tau=0.6146 pooled certified gptq_int4 safe despite "
        "-3.0pt real damage). Compare against the Llama-derived shipped "
        "defaults (bnb budget=0.5, GPTQ budget=1.3) to see how much the "
        "safe/unsafe boundary shifts across model families."
    ),
}

out_path = os.path.join(HERE, "qwen_calibration_weight_quant.json")
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result, indent=2))
print(f"\nSaved: {out_path}")
