"""
deltacert.py — DeltaCert: Universal Bounded Divergence Certification
Dev Shorya, 2026

Offline certification engine for any compression or approximation in an
LLM inference stack. Uses d_COMM (Proposition 5.1, Shorya 2026) as the
universal certificate formula.

Formula:
    Δ = 4·c·√(1−c²)      [commutator magnitude from cosine similarity]
    d = −log(E[Δ] / 2)   [algebraic distance]
    certified if d ≥ budget (default 3.0)
    divergence bound = 2·exp(−d)

Offline workflow:
    1. dc.certify(config, calibration_data)  →  saves JSON
    2. Load JSON at server startup: check certified=true per layer
    3. Refuse to serve if any required layer is not certified

The runtime phase never runs math — it just reads the JSON.
Zero inference overhead.

Correct Step 3:
    vllm serve meta-llama/Llama-3-70B \\
        --quantization awq \\
        --tensor-parallel-size 4 \\
        --deltacert ./llama3_70b_int4_tp4_certified.json
    # certified=true  → serve
    # certified=false → refuse to start
"""

import datetime
import hashlib
import json
import platform
import socket
from dataclasses import dataclass, field
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# d_COMM core formula — imported from system_collectors (canonical v2).
# system_collectors.py is the single source of truth for this math.
# ─────────────────────────────────────────────────────────────────────────────

from deltacert.collectors import (  # noqa: E402
    _CANONICAL_MATH_VERSION,
    _C_MIN_VALID,
    _commutator_magnitude,
    _d_from_delta,
    d_comm,
    divergence_bound,
    CollectionError,
    certify_from_layers,
    save_logits,
    load_logits,
    CERT_THRESHOLD_D,
    validation_status_for_layers,
)



# CALIBRATION — PER-METHOD budget for weight_quant. GPTQ is now n=6,
# genuinely two-sided, verified on TWO model families (Llama-3.1-8B-Instruct
# AND Qwen2.5-7B-Instruct, GSM8K downstream, GPTQ int8/int4/int3/int2 each —
# see validation_results/weight_quant/ and validation_results/qwen/weight_quant_gptq/).
# bnb remains n=5, one-sided, one model — see caveat 2 below, unchanged.
# State "n=6, two-model two-sided GPTQ calibration; n=5, one-sided,
# one-model bnb calibration" (not a settled universal constant) in any
# spec/README that cites these numbers, and flag both caveats below:
#
#   1. Cross-method: d_comm is a reliable *within-method* damage indicator
#      (lower d = more real damage, monotonic within GPTQ's own sweep) but is
#      NOT directly comparable *across* methods — GPTQ int4 (d=1.164 Llama,
#      0.825 Qwen) showed real GSM8K damage (-8pts / -3pts) while bnb nf4
#      (d=0.578, a LOWER number) showed none. A single universal tau=3.0
#      wrongly failed bnb int8 (d=1.153, 0.0pts real damage — a false negative).
#
#   2. bnb has NO observed failure point: only 2 bnb readings exist, both
#      safe (int8=1.153, nf4=0.578) — we never tested a deliberately-broken
#      bnb config, so there is no evidence-backed floor for where bnb
#      actually fails. 0.5 here only guarantees both known-safe bnb readings
#      pass; it is NOT derived from an observed bnb failure boundary the way
#      the gptq floor now is.
#
#   3. gptq's floor is now two-sided on each model separately (Llama tau=3.187,
#      Qwen tau=2.1481 -- see validation_results/), but this shipped constant
#      is a single GLOBAL value with no per-model dispatch (certify_layer only
#      indexes by quant_method, never by model) -- so it CANNOT be genuinely
#      conservative for both: the true conservative value (3.187) would flip
#      Qwen's own GPTQ int8 to "unsafe", contradicting its own calibration.
#      2.1481 (Qwen's floor, the lower of the two) is therefore the PERMISSIVE
#      compromise, not the conservative one -- it stays above every damaged
#      GPTQ reading measured on either model (Llama int4=1.164, Qwen
#      int4=0.825) but sits below Llama's own calibrated floor (3.187), so a
#      Llama config with d in (2.15, 3.19) would pass this shipped default
#      while failing Llama's own per-model calibration. This is exactly why
#      calibrating per-model (deltacert calibrate) is not optional for
#      production use, and per-model dispatch here is a real roadmap item.
#   4. Stored as 2.148, not the display-rounded 2.1481: Qwen's own GPTQ int8
#      reading IS 2.1481 at 4-decimal display precision, and inverting its
#      stored divergence_bound (6-decimal precision) recovers ~2.148102 --
#      close enough to the boundary that a bare `d >= 2.1481` risks failing
#      Qwen's own reference config on sub-decimal floating-point noise. 2.148
#      (one fewer digit) is a deliberate safety margin below the true value,
#      not a rounding accident.
_PROVISIONAL_METHOD_BUDGETS = {
    "bnb": 0.5,    # passes both known-safe bnb readings (int8=1.153, nf4=0.578);
                   # NO observed bnb failure point exists yet — see caveat 2 above
    "gptq": 2.148, # permissive global compromise = Qwen's own calibrated floor,
                   # with a deliberate safety margin below display precision;
                   # see caveats 3 and 4 above
}


def certify_layer(
    cos_sims: list,
    budget: float = 3.0,
    domain_labels: Optional[list] = None,
    quant_method: Optional[str] = None,
) -> dict:
    """
    Single-layer certificate.
    Returns dict with d, bound, certified — ready to embed in JSON.

    domain_labels: optional, one label per entry in cos_sims (same order,
    same length — e.g. ["math", "math", "code", "multilingual", ...]).
    When given, the certified d is the WORST-domain d_comm, never the
    blended average across domains — a severe regression in one domain
    (code, multilingual, etc.) must not be hidden by fine performance in
    others. Mirrors the trajectory collector's d_min-over-positions rule,
    applied to d_comm-over-domains. Requires at least 2 distinct domains;
    collectors.py is unmodified — it still just returns cos_sims per prompt,
    this only groups the caller-known domain of each prompt afterward.

    quant_method: optional, e.g. "bnb" or "gptq". When given AND the caller
    did not explicitly override `budget`, uses the provisional per-method
    threshold from _PROVISIONAL_METHOD_BUDGETS instead of the universal
    default — see that dict's docstring for why a single tau=3.0 produced a
    false negative on a real, harmless bnb int8 config. Explicit `budget`
    always wins if passed.
    """
    if quant_method is not None and budget == 3.0:
        budget = _PROVISIONAL_METHOD_BUDGETS.get(quant_method, budget)

    if domain_labels is None:
        d = d_comm(cos_sims)
        return {
            "d_comm": round(d, 4) if d != float('inf') else "inf",
            "divergence_bound": round(divergence_bound(d), 6),
            "certified": d >= budget,
            "budget": budget,
            "n_samples": len(cos_sims),
        }

    if len(cos_sims) != len(domain_labels):
        raise CollectionError(
            f"cos_sims has {len(cos_sims)} entries but domain_labels has "
            f"{len(domain_labels)} — they must be index-aligned, one label "
            "per measurement."
        )
    by_domain: dict = {}
    for c, dom in zip(cos_sims, domain_labels):
        by_domain.setdefault(dom, []).append(c)
    if len(by_domain) < 2:
        raise CollectionError(
            f"certify_layer got domain_labels with only {len(by_domain)} "
            f"distinct domain(s) ({sorted(by_domain)}) — stratification "
            "requires at least 2 domains. Omit domain_labels for a "
            "single-domain measurement."
        )

    per_domain_d = {dom: d_comm(vals) for dom, vals in by_domain.items()}
    worst_domain = min(per_domain_d, key=lambda dom: per_domain_d[dom])
    d = per_domain_d[worst_domain]
    return {
        "d_comm": round(d, 4) if d != float('inf') else "inf",
        "divergence_bound": round(divergence_bound(d), 6),
        "certified": d >= budget,
        "budget": budget,
        "n_samples": len(cos_sims),
        "n_domains": len(by_domain),
        "worst_domain": worst_domain,
        "per_domain": {
            dom: (round(dv, 4) if dv != float("inf") else "inf")
            for dom, dv in per_domain_d.items()
        },
        "statistic": "min d_comm over domains (worst-domain, not average)",
    }


def calibrate_layer(
    sweep: list,
    downstream_degradation_threshold_pts: float = 2.0,
) -> dict:
    """
    Self-calibration: find a safe/unsafe tau from a company's OWN sweep,
    instead of trusting DeltaCert's shipped reference calibration
    (_PROVISIONAL_METHOD_BUDGETS above, or the universal default budget=3.0).

    Why this exists: the shipped defaults were calibrated on ONE model
    (Llama-3.1-8B-Instruct) across a handful of configs per change-type (see
    validation/). d_comm is a reliable within-method damage indicator, but
    where exactly it crosses from "safe" to "real damage" can shift on a
    different model or workload. This function runs the exact same
    threshold-finding logic used to derive the shipped defaults, but on
    whatever configs the caller measured on their OWN model/workload.

    sweep: list of dicts, one per config actually measured, each with:
        "name":               str, e.g. "int8", "batch_32", "gptq_int4"
        "method_family":      str, e.g. "bnb", "gptq" — REQUIRED. d_comm is
                              not comparable across quantization methods (the
                              bnb and GPTQ scales differ — see SPEC.md/paper
                              §3.4); calibrating one tau across mixed families
                              produces a false-safe when an undamaged config
                              from one family sits, on the pooled numeric
                              axis, below a damaged config from another
                              family. tau is therefore calibrated PER family,
                              never pooled. Configs that are genuinely one
                              family (the common case) just repeat the same
                              string for every entry.
        "cos_sims":           list[float], from the caller's own collector
                              run (collectors.py, unmodified) — same-shape
                              input certify_layer already takes
        "domain_labels":      optional, one label per entry in cos_sims,
                              same length/order (e.g. ["math", "code", ...]).
                              When given, this config's d_comm is the
                              WORST-domain value (reuses certify_layer's own
                              domain-stratification), never a blended
                              average — the exact rule every flagship test
                              already relies on. A calibrate run that
                              silently blended domains instead of taking
                              the worst would contradict the tool's own
                              design and could miss a domain-specific
                              regression the same way an unstratified tau
                              would.
        "downstream_drop_pts": float, the caller's OWN real downstream metric
                              delta for this config (e.g. their eval's
                              accuracy drop) — REQUIRED. calibrate_layer
                              never simulates this; a config without a real
                              measured downstream number is rejected, same
                              "no simulation" rule as collectors.py.

    Returns a dict keyed by method_family, each value holding: per-config
    d_comm + downstream_drop_pts, the calibrated tau for that family, which
    configs are safe/unsafe under it, and an explicit n_configs count so the
    caller can judge confidence (n=2 is weaker evidence than n=10, same
    caveat that applies to the shipped defaults). A family whose sweep
    contains NO undamaged config (every measured point already shows real
    damage) has no safe floor to calibrate a two-sided boundary from; that
    family's tau is set just above its worst observed damage and the family
    is stamped "provisional": true rather than reported as if it were a
    normal two-sided calibration (any tau above the worst-observed-damage
    point is equally consistent with the data — the placement is a lower
    bound, not a measured boundary).
    """
    if not sweep:
        raise CollectionError(
            "calibrate_layer called with an empty sweep. Refusing to "
            "calibrate on zero configs."
        )

    by_family: dict = {}
    for entry in sweep:
        name = entry.get("name")
        family = entry.get("method_family")
        cos_sims = entry.get("cos_sims")
        domain_labels = entry.get("domain_labels")
        drop = entry.get("downstream_drop_pts")
        if name is None:
            raise CollectionError("Every sweep entry requires a 'name'.")
        if not family:
            raise CollectionError(
                f"Sweep entry '{name}' has no method_family. d_comm is not "
                "comparable across quantization methods (bnb vs GPTQ, etc.) "
                "- calibrate_layer requires every entry to declare which "
                "method family it belongs to so tau is calibrated per "
                "family, never pooled across incomparable scales."
            )
        if not cos_sims:
            raise CollectionError(
                f"Sweep entry '{name}' has no cos_sims. calibrate_layer "
                "never simulates a measurement - run the real collector "
                "for this config first."
            )
        if drop is None:
            raise CollectionError(
                f"Sweep entry '{name}' has no downstream_drop_pts. A config "
                "without a real measured downstream number cannot be used "
                "to calibrate a threshold - it would be guessing where the "
                "boundary is instead of measuring it."
            )

        row = {
            "name": name,
            # downstream_drop_pts: received from the caller, never simulated
            # — stamped explicitly so the artifact is honest about which
            # numbers are DeltaCert's own measurement vs. externally supplied.
            "downstream_drop_pts": float(drop),
            "downstream_drop_pts_source": "user-provided",
            "materially_degraded": abs(float(drop)) > downstream_degradation_threshold_pts,
        }
        if domain_labels:
            # reuse certify_layer's own worst-domain grouping rather than
            # duplicating that logic here.
            per_domain_cert = certify_layer(cos_sims, budget=0.0, domain_labels=domain_labels)
            d = per_domain_cert["d_comm"]
            d_raw = float("inf") if d == "inf" else float(d)
            row["d_comm"] = d
            row["d_comm_raw"] = d_raw
            row["worst_domain"] = per_domain_cert["worst_domain"]
            row["per_domain"] = per_domain_cert["per_domain"]
        else:
            d = d_comm(cos_sims)
            row["d_comm"] = round(d, 4) if d != float("inf") else "inf"
            row["d_comm_raw"] = d
        by_family.setdefault(family, []).append(row)

    families_out: dict = {}
    for family, rows in by_family.items():
        # Calibrated tau: the lowest d_comm among configs in THIS family that
        # did NOT show real downstream damage — i.e. the floor of "known-safe"
        # evidence. Any config at or above this line is at least as safe, by
        # d_comm, as the safest config we've actually observed degrade or not.
        # Mirrors exactly how _PROVISIONAL_METHOD_BUDGETS above was derived by
        # hand from the weight_quant sweep (gptq floor set just above its
        # first real-damage point; bnb floor set to pass its only known-safe
        # readings) — this function just automates that, per family.
        safe_rows = [r for r in rows if not r["materially_degraded"]]
        damaged_rows = [r for r in rows if r["materially_degraded"]]
        # A calibration is two-sided (a real boundary between two observed
        # clusters) only when BOTH clusters are non-empty. Either cluster
        # missing means tau is placed relative to just one cluster's edge,
        # not "between" anything — the exact rule §3.4 states in prose, which
        # this code must match, not just the "no safe configs" direction.
        one_sided_from_above = not safe_rows   # every config damaged
        one_sided_from_below = not damaged_rows  # every config clean
        provisional = one_sided_from_above or one_sided_from_below

        if safe_rows:
            finite_safe = [r["d_comm_raw"] for r in safe_rows if r["d_comm_raw"] != float("inf")]
            tau = min(finite_safe) if finite_safe else 0.0
        else:
            # every config in this family showed real damage — no safe floor
            # to calibrate a two-sided boundary from; fall back to the
            # highest observed d_comm + margin so nothing in this sweep is
            # misclassified safe. This placement is a lower bound consistent
            # with the data, not a measured boundary — hence "provisional".
            finite_damaged = [r["d_comm_raw"] for r in damaged_rows if r["d_comm_raw"] != float("inf")]
            tau = (max(finite_damaged) + 1.0) if finite_damaged else 3.0

        for r in rows:
            r["certified_at_calibrated_tau"] = (
                True if r["d_comm"] == "inf" else r["d_comm_raw"] >= tau
            )
            del r["d_comm_raw"]

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
                    "showed real downstream damage - there is no undamaged config to "
                    "set a two-sided safe floor from. tau is a lower bound consistent "
                    "with the data (placed just above the worst observed damage), not "
                    "a measured safe/unsafe boundary. Any tau above the worst-damage "
                    "point in this sweep is equally consistent with it - measure at "
                    "least one undamaged config in this family to firm this up."
                ) if one_sided_from_above else (
                    f"PROVISIONAL: every '{family}' config in this sweep (n={len(rows)}) "
                    "was undamaged - there is no damaged config to set a two-sided "
                    "boundary from. tau is placed at the lowest observed (undamaged) "
                    "d_comm, which means the boundary-exact config certifies with "
                    "zero margin ('>=', not '>') - measurement noise on a config "
                    "sitting at exactly this d_comm could flip its verdict. Measure "
                    "at least one damaged config in this family to firm this up."
                ) if one_sided_from_below else (
                    f"Calibrated from n={len(rows)} '{family}' config(s) on this "
                    "specific model/workload — not a universal constant. More "
                    "configs measured over time firm this up, the same way "
                    "DeltaCert's own shipped defaults are an initial calibration "
                    "from n=5 configs (see deltacert.py's _PROVISIONAL_METHOD_BUDGETS)."
                )
            ),
        }

    return {
        "families": families_out,
        "n_families": len(families_out),
        "downstream_degradation_threshold_pts": downstream_degradation_threshold_pts,
        "caveat": (
            "d_comm is not comparable across quantization methods, so tau is "
            "calibrated independently per method_family — see each family's "
            "own 'caveat' for its specific confidence level. Pooling families "
            "into one tau produces a false-safe when an undamaged config from "
            "one family sits below a damaged config from another on the "
            "pooled axis (see paper §3.4 / SPEC.md)."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# InferenceConfig — describes what is compressed in the inference stack
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LayerSpec:
    """
    One compression or approximation layer to certify.

    name:         unique ID, e.g. "weight_quantization", "allreduce_tp"
    enabled:      whether this layer is active in the deployment config
    budget:       minimum d_COMM to pass (default 3.0 -> divergence < 0.1)
    quant_method: optional, e.g. "bnb" or "gptq". When given AND budget is
                  left at its default (3.0, i.e. not explicitly overridden
                  by the caller), certify() uses the provisional per-method
                  threshold from _PROVISIONAL_METHOD_BUDGETS instead of the
                  universal default -- same rule certify_layer() already
                  documents. Without this, every layer silently gets the
                  flat 3.0 default regardless of method, which is calibrated
                  for KV-cache/trajectory checks, not weight quantization --
                  see _PROVISIONAL_METHOD_BUDGETS's docstring for the real
                  bnb/GPTQ floors this was derived from.
    """
    name: str
    enabled: bool = True
    budget: float = 3.0
    quant_method: Optional[str] = None


@dataclass
class InferenceConfig:
    """
    Describes every compression/approximation active in an inference stack.
    Pass this to certify() with calibration data.

    All 20 d_COMM application areas:
        Infrastructure (items 1-4): wire compression — theoretical guarantee
        Model-level (items 5-10):   compression and caching — empirical certificate
        System changes (items 11-16): engine, batch, spec, sparse, MoE, pruning
        Change certification (17-19): model update, provider drift, prompt swap
        Trajectory (20):            long-context / agent workloads

    Example:
        config = InferenceConfig(
            model="meta-llama/Llama-3-70B",
            layers=[
                LayerSpec("weight_quantization", budget=3.0),
                LayerSpec("kv_cache_quantization", budget=3.0),
                LayerSpec("allreduce_tp", budget=3.0),
                LayerSpec("alltoall_ep", budget=3.0),
            ]
        )
    """
    model: str
    layers: list = field(default_factory=list)
    description: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "InferenceConfig":
        layers = [LayerSpec(**l) for l in d.get("layers", [])]
        return cls(model=d["model"], layers=layers, description=d.get("description", ""))


# ─────────────────────────────────────────────────────────────────────────────
# The 10 standard layer names (use these in LayerSpec.name)
# ─────────────────────────────────────────────────────────────────────────────

LAYER_ALLREDUCE_TP       = "allreduce_tp"          # 1
LAYER_ALLTOALL_EP        = "alltoall_ep"            # 2
LAYER_PIPELINE_PARALLEL  = "pipeline_parallel"      # 3
LAYER_KV_TRANSFER        = "kv_transfer_pd"         # 4
LAYER_WEIGHT_QUANT       = "weight_quantization"    # 5
LAYER_KV_CACHE_QUANT     = "kv_cache_quantization"  # 6
LAYER_ACTIVATION_QUANT   = "activation_quantization" # 7
LAYER_GRADIENT_COMP      = "gradient_compression"   # 8
LAYER_LORA               = "lora_vs_full"           # 9
LAYER_PREFIX_CACHE       = "prefix_cache"           # 10
LAYER_ENGINE_SWAP        = "engine_swap"            # 11
LAYER_BATCH_DIVERGENCE   = "batch_divergence"       # 12
LAYER_SPEC_DECODING      = "spec_decoding"          # 13
LAYER_SPARSE_ATTENTION   = "sparse_attention"       # 14
LAYER_MOE_TOKEN_DROP     = "moe_token_dropping"     # 15
LAYER_NEURON_SKIPPING    = "neuron_skipping"        # 16
LAYER_MODEL_SWAP         = "model_swap"             # 17
LAYER_PROVIDER_DRIFT     = "provider_drift"         # 18
LAYER_PROMPT_SWAP        = "prompt_swap"            # 19
LAYER_TRAJECTORY         = "trajectory"             # 20


# ─────────────────────────────────────────────────────────────────────────────
# Certify — offline phase
# ─────────────────────────────────────────────────────────────────────────────

def certify(
    config: InferenceConfig,
    calibration: dict,
    output_path: Optional[str] = None,
    precomputed_layers: Optional[dict] = None,
    calibration_domains: Optional[dict] = None,
) -> dict:
    """
    Offline certification pass. Run once before deployment.

    Args:
        config:        InferenceConfig describing the stack
        calibration:   dict mapping layer_name → list of cosine similarities
                       Each cos_sim = cosine_similarity(original_output, compressed_output)
                       measured on your calibration dataset.

                       Example:
                           calibration = {
                               "weight_quantization": [0.9997, 0.9998, 0.9996, ...],
                               "kv_cache_quantization": [0.9994, 0.9995, ...],
                           }

        output_path:   if given, saves the certificate JSON to this path
        calibration_domains: optional, mirrors `calibration`'s keys — dict
                       mapping layer_name → list of domain labels, index-
                       aligned with that layer's cos_sims list. When a layer
                       has domain labels, its certified d_comm is the
                       WORST-domain value, not the blended average — see
                       certify_layer()'s domain_labels docstring.

    Returns:
        certificate dict — save as JSON and load at server startup

    Certificate format:
        {
            "model": "meta-llama/Llama-3-70B",
            "certified": true,          # ALL enabled layers passed
            "layers": {
                "weight_quantization": {
                    "d_comm": 3.72,
                    "divergence_bound": 0.0241,
                    "certified": true,
                    "budget": 3.0,
                    "n_samples": 512,
                },
                ...
            }
        }
    """
    layer_results = {}
    all_certified = True

    # Precomputed layer results (e.g. trajectory) bypass certify_layer —
    # they are already complete layer dicts built by trajectory_layer_result().
    for name, result in (precomputed_layers or {}).items():
        layer_results[name] = result
        if not result.get("certified", False):
            all_certified = False

    for spec in config.layers:
        if not spec.enabled:
            layer_results[spec.name] = {"enabled": False, "certified": True}
            continue

        cos_sims = calibration.get(spec.name, [])
        if not cos_sims:
            layer_results[spec.name] = {
                "d_comm": None,
                "divergence_bound": None,
                "certified": False,
                "budget": spec.budget,
                "n_samples": 0,
                "error": "no calibration data provided",
            }
            all_certified = False
            continue

        domain_labels = (calibration_domains or {}).get(spec.name)
        result = certify_layer(
            cos_sims, budget=spec.budget, domain_labels=domain_labels,
            quant_method=spec.quant_method,
        )
        layer_results[spec.name] = result
        if not result["certified"]:
            all_certified = False

    # threshold_d: kept for schema parity with collectors.py::certify_from_layers().
    # Per-layer "budget" (in each layer dict above) is the authoritative gating
    # value in both schemas — this top-level field is the informational default,
    # not necessarily what every layer was actually gated on.
    certificate = {
        "model": config.model,
        "description": config.description,
        "certified": all_certified,
        "threshold_d": CERT_THRESHOLD_D,
        "formula": "d_COMM = -log(E[4c*sqrt(1-c^2)] / 2), certified if d >= budget",
        "theorem": "Proposition 5.1, Shorya 2026",
        "validation_status": validation_status_for_layers(layer_results.keys()),
        "layers": layer_results,
    }

    certificate["metadata"] = _build_metadata(config.model)

    if output_path:
        with open(output_path, "w") as f:
            json.dump(certificate, f, indent=2)

    return certificate


def _build_metadata(model_id: str) -> dict:
    """Build versioning metadata — model hash, date, hardware info."""
    from deltacert import __version__  # lazy import: avoids circular import at module load time
    meta = {
        "certified_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model_id": model_id,
        "model_id_hash": hashlib.sha256(model_id.encode()).hexdigest()[:16],
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "deltacert_version": __version__,
    }
    try:
        import torch
        meta["torch_version"] = torch.__version__
        if torch.cuda.is_available():
            meta["cuda_version"] = torch.version.cuda
            meta["gpu_count"] = torch.cuda.device_count()
            meta["gpu_name"] = torch.cuda.get_device_name(0)
            meta["gpu_memory_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 1
            )
    except ImportError:
        pass
    return meta


# ─────────────────────────────────────────────────────────────────────────────
# Load + check — server startup phase (zero math, just reads JSON)
# ─────────────────────────────────────────────────────────────────────────────

def load_certificate(path: str) -> dict:
    """Load a certificate JSON from disk."""
    with open(path) as f:
        return json.load(f)


def check_certified(certificate: dict, required_layers: Optional[list] = None) -> tuple:
    """
    Check a loaded certificate at server startup.

    Args:
        certificate:     loaded JSON dict from load_certificate()
        required_layers: if given, only check these layer names.
                         if None, uses certificate["certified"] (all layers).

    Returns:
        (ok: bool, failures: list of layer names that failed)

    Usage:
        cert = dc.load_certificate("./llama3_70b_int4_tp4_certified.json")
        ok, failures = dc.check_certified(cert)
        if not ok:
            raise RuntimeError(f"Deployment not certified: {failures}")
    """
    if required_layers is None:
        ok = certificate.get("certified", False)
        failures = [
            name for name, r in certificate.get("layers", {}).items()
            if isinstance(r, dict) and not r.get("certified", True)
        ]
        return ok, failures

    failures = []
    for name in required_layers:
        r = certificate.get("layers", {}).get(name, {})
        if not r.get("certified", False):
            failures.append(name)
    return len(failures) == 0, failures


def enforce(certificate_path: str, required_layers: Optional[list] = None) -> None:
    """
    Load certificate and raise RuntimeError if not certified.
    Call this at server startup before accepting traffic.

    Example in vLLM / any inference server:
        import deltacert as dc
        dc.enforce("./llama3_70b_int4_tp4_certified.json")
        # proceeds normally if certified
        # raises RuntimeError and refuses to start if not certified
    """
    cert = load_certificate(certificate_path)
    ok, failures = check_certified(cert, required_layers)
    if not ok:
        raise RuntimeError(
            f"DeltaCert: deployment NOT certified. Failed layers: {failures}. "
            f"Re-run certify() or reduce compression ratio."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Multi-layer composition — union bound (no external theorems)
# ─────────────────────────────────────────────────────────────────────────────

def compose_bounds(layer_results: dict) -> dict:
    """
    Combine certificates across multiple active layers.

    Method: union bound.
    If each layer has divergence bound b_i = 2·exp(−d_i), then
    the total divergence across all layers is at most Σ b_i.
    This is a valid upper bound from basic probability (subadditivity).

    No composition theorems needed — just addition.

    Returns:
        {
            "total_divergence_bound": float,   # Σ b_i
            "certified": bool,                 # all individual layers certified
            "n_layers_active": int,
        }
    """
    active = {
        name: r for name, r in layer_results.items()
        if isinstance(r, dict) and r.get("enabled", True) and r.get("d_comm") is not None
    }

    total_bound = sum(r["divergence_bound"] for r in active.values())
    all_certified = all(r["certified"] for r in active.values())

    return {
        "total_divergence_bound": round(total_bound, 6),
        "certified": all_certified,
        "n_layers_active": len(active),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Quick summary for logging
# ─────────────────────────────────────────────────────────────────────────────

def certify_system(
    model: str,
    prompts: list,
    checks: list,
    output_path: Optional[str] = None,
    budget: float = 3.0,
    device: str = "cuda",
    # collectors 1-3: compression fns + model/tokenizer
    model_base=None,
    tokenizer=None,
    compress_fn=None,
    decompress_fn=None,
    stage_boundary_layer_idx: int = 0,
    # collector 4: kv_transfer uses model_base + compress_fn/decompress_fn
    # collector 5: weight_quant
    model_quantized=None,
    # weight_quant_method: "bnb" or "gptq" -- when the caller knows which
    # backend produced model_quantized, pass it so the weight_quant layer's
    # threshold resolves through the same per-family provisional calibration
    # (_PROVISIONAL_METHOD_BUDGETS) certify_layer()/certify() already use,
    # instead of the flat universal budget default. Without this, a caller
    # going through certify_system() gets budget=3.0 for weight_quant same
    # as every other check -- calibrated for KV-cache/trajectory, not weight
    # quantization -- which silently produces false-unsafe verdicts on real,
    # undamaged bnb quantizations (see the auto_certify fix this mirrors).
    weight_quant_method: Optional[str] = None,
    # collector 6: kv_cache_quant — vllm backend (default, real production
    # measurement via vLLM's native --kv-cache-dtype) uses model_name_or_path;
    # hf backend (opt-in, for non-vLLM deployments) uses model_base + compress_fn/decompress_fn
    kv_cache_backend: str = "vllm",
    kv_cache_dtype: str = "fp8",
    # collector 7: activation_quant
    quant_fn=None,
    dequant_fn=None,
    # collector 8: gradient_compress uses model_base + compress_fn/decompress_fn
    # collector 9: lora
    lora_adapter_path: Optional[str] = None,
    # collector 10: prefix_cache
    shared_prefix: Optional[str] = None,
    # collector 11: engine_swap — two on-disk .npz captures
    capture_a_path: Optional[str] = None,
    capture_b_path: Optional[str] = None,
    # collector 12: batch_divergence
    model_name_or_path: Optional[str] = None,
    batched_size: Optional[int] = None,
    tensor_parallel_size: int = 1,
    # vLLM hard-caps sampled logprobs at 20 (see capture_logits_vllm) —
    # this default must match, or batch_divergence/spec_decoding crash on
    # every real vLLM engine with "Requested sample logprobs of 128, which
    # is greater than max allowed: 20".
    num_logprobs: int = 20,
    engine_kwargs: Optional[dict] = None,
    # collector 13: spec_decoding
    speculative_config: Optional[dict] = None,
    # collectors 14/16: sparse_attention / neuron_skipping
    sparse_context=None,
    prune_context=None,
    # collector 15: moe_token_dropping
    set_capacity=None,
    capacity_baseline: float = 1.0,
    capacity_deployed: float = 0.9,
    # collector 17: model_swap (candidate capture path)
    candidate_path: Optional[str] = None,
    # collector 18: provider_drift
    api_base: Optional[str] = None,
    api_model: Optional[str] = None,
    api_key: Optional[str] = None,
    drift_baseline_path: Optional[str] = None,
    rotate_baseline_path: Optional[str] = None,
    # collector 19: prompt_swap
    system_prompt_a: Optional[str] = None,
    system_prompt_b: Optional[str] = None,
    # collector 20: trajectory
    trajectory_cases=None,
    trajectory_context=None,
    trajectory_model_b=None,
    # domain-stratified certification (optional, any check using `prompts`)
    domain_labels: Optional[list] = None,
) -> dict:
    """
    Single entry point — dispatches to the correct collector for each check.
    Zero measurement logic lives here. All measurement is in collectors.py.

    checks: list of check names. All 20 supported:
        "allreduce_tp", "alltoall_ep", "pipeline_parallel", "kv_transfer",
        "weight_quant", "kv_cache_quant", "activation_quant", "gradient_compress",
        "lora", "prefix_cache",
        "engine_swap", "batch_divergence", "spec_decoding",
        "sparse_attention", "moe_token_dropping", "neuron_skipping",
        "model_swap", "provider_drift", "prompt_swap", "trajectory"

    Example:
        cert = dc.certify_system(
            model="meta-llama/Llama-3.1-8B",
            prompts=my_prompts,
            checks=["weight_quant", "prefix_cache"],
            model_base=m_fp16,
            model_quantized=m_int8,
            tokenizer=tok,
            shared_prefix="You are a helpful assistant.",
            output_path="./cert.json",
        )
    """
    from deltacert.collectors import (
        collect_allreduce_tp, collect_alltoall_ep, collect_pipeline_parallel,
        collect_kv_transfer, collect_weight_quant, collect_kv_cache_quant,
        collect_kv_cache_quant_vllm,
        collect_activation_quant, collect_gradient_compress, collect_lora,
        collect_prefix_cache, collect_engine_swap, collect_batch_divergence,
        collect_speculative_decode, collect_sparse_attention,
        collect_moe_token_dropping, collect_neuron_skipping,
        collect_model_swap, collect_provider_drift, collect_prompt_swap,
        collect_trajectory, collect_trajectory_two_models, trajectory_layer_result,
    )

    calibration = {}
    precomputed = {}
    layer_specs = []

    _VALID = {
        "allreduce_tp", "alltoall_ep", "pipeline_parallel", "kv_transfer",
        "weight_quant", "kv_cache_quant", "activation_quant", "gradient_compress",
        "lora", "prefix_cache", "engine_swap", "batch_divergence",
        "spec_decoding", "sparse_attention", "moe_token_dropping", "neuron_skipping",
        "model_swap", "provider_drift", "prompt_swap", "trajectory",
    }

    for check in checks:
        if check not in _VALID:
            raise CollectionError(
                f"Unknown check: '{check}'. Valid: {sorted(_VALID)}"
            )

        if check == "allreduce_tp":
            cos_sims = collect_allreduce_tp(model_base, tokenizer, prompts, compress_fn, decompress_fn, device)
            calibration[LAYER_ALLREDUCE_TP] = cos_sims

        elif check == "alltoall_ep":
            cos_sims = collect_alltoall_ep(model_base, tokenizer, prompts, compress_fn, decompress_fn, device)
            calibration[LAYER_ALLTOALL_EP] = cos_sims

        elif check == "pipeline_parallel":
            cos_sims = collect_pipeline_parallel(model_base, tokenizer, prompts, compress_fn, decompress_fn, stage_boundary_layer_idx, device)
            calibration[LAYER_PIPELINE_PARALLEL] = cos_sims

        elif check == "kv_transfer":
            cos_sims = collect_kv_transfer(model_base, tokenizer, prompts, compress_fn, decompress_fn, device)
            calibration[LAYER_KV_TRANSFER] = cos_sims

        elif check == "weight_quant":
            cos_sims = collect_weight_quant(model_base, model_quantized, tokenizer, prompts, device)
            calibration[LAYER_WEIGHT_QUANT] = cos_sims

        elif check == "kv_cache_quant":
            if kv_cache_backend == "vllm":
                cos_sims = collect_kv_cache_quant_vllm(
                    model_name_or_path or model, prompts,
                    kv_cache_dtype=kv_cache_dtype,
                    tensor_parallel_size=tensor_parallel_size,
                    num_logprobs=num_logprobs, engine_kwargs=engine_kwargs,
                )
            elif kv_cache_backend == "hf":
                cos_sims = collect_kv_cache_quant(model_base, tokenizer, prompts, compress_fn, decompress_fn, device)
            else:
                raise CollectionError(
                    f"kv_cache_backend must be 'vllm' or 'hf', got '{kv_cache_backend}'."
                )
            calibration[LAYER_KV_CACHE_QUANT] = cos_sims

        elif check == "activation_quant":
            cos_sims = collect_activation_quant(model_base, tokenizer, prompts, quant_fn, dequant_fn, device)
            calibration[LAYER_ACTIVATION_QUANT] = cos_sims

        elif check == "gradient_compress":
            cos_sims = collect_gradient_compress(model_base, tokenizer, prompts, compress_fn, decompress_fn, device)
            calibration[LAYER_GRADIENT_COMP] = cos_sims

        elif check == "lora":
            cos_sims = collect_lora(model_base, tokenizer, prompts, lora_adapter_path, device)
            calibration[LAYER_LORA] = cos_sims

        elif check == "prefix_cache":
            cos_sims = collect_prefix_cache(model_base, tokenizer, prompts, shared_prefix, device)
            calibration[LAYER_PREFIX_CACHE] = cos_sims

        elif check == "engine_swap":
            cos_sims = collect_engine_swap(capture_a_path, capture_b_path)
            calibration[LAYER_ENGINE_SWAP] = cos_sims

        elif check == "batch_divergence":
            cos_sims = collect_batch_divergence(model_name_or_path, prompts, batched_size, tensor_parallel_size, num_logprobs, engine_kwargs)
            calibration[LAYER_BATCH_DIVERGENCE] = cos_sims

        elif check == "spec_decoding":
            cos_sims = collect_speculative_decode(model_name_or_path, prompts, speculative_config, tensor_parallel_size, num_logprobs, engine_kwargs)
            calibration[LAYER_SPEC_DECODING] = cos_sims

        elif check == "sparse_attention":
            cos_sims = collect_sparse_attention(model_base, tokenizer, prompts, sparse_context)
            calibration[LAYER_SPARSE_ATTENTION] = cos_sims

        elif check == "moe_token_dropping":
            cos_sims = collect_moe_token_dropping(model_base, tokenizer, prompts, set_capacity, capacity_baseline, capacity_deployed)
            calibration[LAYER_MOE_TOKEN_DROP] = cos_sims

        elif check == "neuron_skipping":
            cos_sims = collect_neuron_skipping(model_base, tokenizer, prompts, prune_context)
            calibration[LAYER_NEURON_SKIPPING] = cos_sims

        elif check == "model_swap":
            cos_sims = collect_model_swap(capture_a_path, candidate_path)
            calibration[LAYER_MODEL_SWAP] = cos_sims

        elif check == "provider_drift":
            cos_sims = collect_provider_drift(
                api_base, api_model or model, prompts, drift_baseline_path,
                api_key=api_key, save_new_baseline_path=rotate_baseline_path,
            )
            calibration[LAYER_PROVIDER_DRIFT] = cos_sims

        elif check == "prompt_swap":
            cos_sims = collect_prompt_swap(
                model_base, tokenizer, prompts,
                system_prompt_a, system_prompt_b, device,
            )
            calibration[LAYER_PROMPT_SWAP] = cos_sims

        elif check == "trajectory":
            if trajectory_context is not None:
                profiles = collect_trajectory(
                    model_base, tokenizer, trajectory_cases,
                    trajectory_context, device,
                )
            elif trajectory_model_b is not None:
                profiles = collect_trajectory_two_models(
                    model_base, trajectory_model_b, tokenizer,
                    trajectory_cases, device,
                )
            else:
                raise CollectionError(
                    "trajectory check requires trajectory_context (hook-style) "
                    "or trajectory_model_b (two-model style)."
                )
            # Trajectory produces a complete layer dict via precomputed_layers
            # -- it bypasses certify_layer entirely (see certify()'s docstring).
            # Do NOT also append a LayerSpec here: certify() processes
            # precomputed_layers first (setting the real, correct result),
            # then separately loops over config.layers checking
            # calibration.get(spec.name) -- trajectory never populates
            # `calibration` (by design, it has its own measurement path), so
            # a LayerSpec for it here would make that second loop find no
            # data and silently overwrite the correct precomputed result with
            # a fake "no calibration data provided" / d_comm=None error. This
            # is exactly the failure this session's CLI smoke test caught:
            # `deltacert certify --checks trajectory` always reported
            # d=N/A/not-certified regardless of the real measurement.
            precomputed[LAYER_TRAJECTORY] = trajectory_layer_result(profiles, threshold=budget)
            continue

        if check == "weight_quant" and weight_quant_method is not None:
            layer_specs.append(LayerSpec(
                _check_to_layer[check], budget=budget,
                quant_method=weight_quant_method,
            ))
        else:
            layer_specs.append(LayerSpec(_check_to_layer[check], budget=budget))

    # domain_labels applies to any check whose cos_sims are order-aligned
    # with `prompts` (every check above except engine_swap/model_swap, which
    # compare pre-captured .npz files, and trajectory, which bypasses
    # certify_layer entirely via precomputed_layers).
    calibration_domains = None
    if domain_labels is not None:
        _NOT_PROMPT_ALIGNED = {LAYER_ENGINE_SWAP, LAYER_MODEL_SWAP, LAYER_TRAJECTORY}
        calibration_domains = {
            layer: domain_labels for layer in calibration
            if layer not in _NOT_PROMPT_ALIGNED
        }

    config = InferenceConfig(model=model, layers=layer_specs)
    return certify(config, calibration, output_path=output_path,
                   precomputed_layers=precomputed,
                   calibration_domains=calibration_domains)


_check_to_layer = {
    "allreduce_tp":      LAYER_ALLREDUCE_TP,
    "alltoall_ep":       LAYER_ALLTOALL_EP,
    "pipeline_parallel": LAYER_PIPELINE_PARALLEL,
    "kv_transfer":       LAYER_KV_TRANSFER,
    "weight_quant":      LAYER_WEIGHT_QUANT,
    "kv_cache_quant":    LAYER_KV_CACHE_QUANT,
    "activation_quant":  LAYER_ACTIVATION_QUANT,
    "gradient_compress": LAYER_GRADIENT_COMP,
    "lora":              LAYER_LORA,
    "prefix_cache":      LAYER_PREFIX_CACHE,
    "engine_swap":       LAYER_ENGINE_SWAP,
    "batch_divergence":  LAYER_BATCH_DIVERGENCE,
    "spec_decoding":     LAYER_SPEC_DECODING,
    "sparse_attention":  LAYER_SPARSE_ATTENTION,
    "moe_token_dropping":LAYER_MOE_TOKEN_DROP,
    "neuron_skipping":   LAYER_NEURON_SKIPPING,
    "model_swap":        LAYER_MODEL_SWAP,
    "provider_drift":    LAYER_PROVIDER_DRIFT,
    "prompt_swap":       LAYER_PROMPT_SWAP,
    "trajectory":        LAYER_TRAJECTORY,
}


def summary(certificate: dict) -> str:
    """Human-readable one-liner per layer for logs."""
    lines = [f"DeltaCert - model: {certificate.get('model', '?')}"]
    lines.append(f"  overall certified: {certificate.get('certified', False)}")
    for name, r in certificate.get("layers", {}).items():
        if isinstance(r, dict) and r.get("enabled", True):
            d = r.get("d_comm")
            d_str = "inf" if d == "inf" else (f"{(d + 0.0):.3f}" if isinstance(d, float) else "N/A")
            status = "[OK]" if r.get("certified") else "[FAIL]"
            bound = r.get("divergence_bound")
            b_str = f"{bound:.4f}" if isinstance(bound, float) else "N/A"
            lines.append(f"  {status} {name:30s}  d={d_str}  bound={b_str}")
    return "\n".join(lines)
