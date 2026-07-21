"""
tests/test_harness.py — permanent tests for deltacert.validation.harness

What this file proves (and what it does NOT):

  PROVES: the validation container cannot be fooled —
    * verdicts cannot contradict the math (d vs tau)
    * rows cannot exist without a real certificate behind them
    * a result.json cannot state a d that differs from its certificate
    * rows cannot exist without measured downstream evidence
    * all 4 tables render, including every field required for the five
      breakthrough conditions (A: d<->downstream alignment, B: killer
      trajectory row, C: business gain, D: breadth, E: caught-what-evals-
      missed) — so when real GPU numbers arrive, nothing is missing.

  DOES NOT prove: that any collector's measurement is correct (that is
  tests/test_smoke.py + the GPU validation runs), or that d predicts
  quality (that is the calibration experiments themselves).

Fixtures here are labeled unit-test inputs that exercise code paths.
The harness never invents numbers; neither does this file — it verifies
that fabricated or inconsistent numbers are REJECTED, which is exactly
the property an auditor checks first.

Run:  pytest deltacert/tests/test_harness.py -v      (pure CPU, ~1 second)
"""

import json
import math
import os
import shutil

import pytest

from deltacert.validation.harness import (
    SchemaError,
    ValidationResult,
    save_result,
    load_result,
    load_all_results,
    build_killer_chart,
    build_full_matrix,
    build_trajectory_table,
    build_calibration_table,
    build_all_tables,
)

# ─────────────────────────────────────────────────────────────────────────────
# Fixture builders
# ─────────────────────────────────────────────────────────────────────────────


def _write_cert(dirpath: str, change_type: str, d_comm) -> str:
    os.makedirs(dirpath, exist_ok=True)
    cert_path = os.path.join(dirpath, "cert.json")
    d_val = math.inf if d_comm == "inf" else float(d_comm)
    layer = {
        "d_comm": d_comm,
        "divergence_bound": 0.0 if math.isinf(d_val) else 2.0 * math.exp(-d_val),
        "certified": d_val >= 3.0,
        "budget": 3.0,
        "n_samples": 50,
    }
    cert = {
        "model": "unittest-model",
        "certified": layer["certified"],
        "layers": {change_type: layer},
    }
    with open(cert_path, "w", encoding="utf-8") as f:
        json.dump(cert, f)
    return cert_path


def _result(
    validation_dir: str,
    collector: str,
    tier: str,
    d_comm,
    downstream: dict,
    trajectory: dict = None,
    short_eval: dict = None,
    gain: dict = None,
    tau: float = 3.0,
    cert_d=None,
) -> ValidationResult:
    folder = os.path.join(validation_dir, collector)
    cert_path = _write_cert(folder, collector, cert_d if cert_d is not None else d_comm)
    metrics = {"d_comm": d_comm, "tau": tau, "downstream_delta": downstream}
    if trajectory:
        metrics["trajectory"] = trajectory
    if short_eval:
        metrics["short_eval"] = short_eval
    return ValidationResult(
        collector=collector,
        tier=tier,
        run_id=f"unittest_{collector}",
        change={
            "baseline": f"{collector} baseline config",
            "candidate": f"{collector} candidate config",
            "change_type": collector,
        },
        business_goal={"reason": f"unit-test reason for {collector}",
                       "expected_gain": gain or {}},
        workload={"task_family": "coding_agent", "num_prompts": 50,
                  "dataset": "unittest_canaries"},
        metrics=metrics,
        decision_statement=(
            f"{collector}: unit-test decision statement long enough to pass "
            "the actionability length gate."
        ),
        cert_path=cert_path,
    )


@pytest.fixture()
def vdir(tmp_path):
    d = str(tmp_path / "validation")
    os.makedirs(d, exist_ok=True)
    return d


def _base_kwargs(cert_path="x"):
    return dict(
        collector="kv_cache_quant",
        tier="A",
        run_id="reject_test",
        change={"baseline": "b", "candidate": "c", "change_type": "kv_cache_quant"},
        business_goal={"reason": "r"},
        workload={"task_family": "qa", "num_prompts": 10},
        decision_statement="a decision statement long enough to pass the gate",
        cert_path=cert_path,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. Happy path: save -> load -> identical, across tiers, inf, trajectory
# ─────────────────────────────────────────────────────────────────────────────


def test_save_load_roundtrip_all_tiers(vdir):
    for collector, tier, d in [
        ("kv_cache_quant", "A", 2.4),
        ("engine_swap", "A", 5.1),
        ("prefix_cache", "B", "inf"),
        ("allreduce_tp", "C", 4.7),
    ]:
        r = _result(vdir, collector, tier, d, {"task_success_drop_pct": -1.0})
        path = os.path.join(vdir, collector, "result.json")
        save_result(r, path)
        loaded = load_result(path)
        assert loaded.collector == collector
        assert loaded.d_display() == r.d_display()
        assert loaded.verdict in ("safe", "unsafe")


def test_verdict_is_derived_when_absent(vdir):
    r = _result(vdir, "weight_quant", "B", 4.2, {"ppl_delta_pct": 0.3})
    assert "verdict" not in r.metrics or r.metrics.get("verdict") is None
    r.validate()
    assert r.metrics["verdict"] == "safe"
    r2 = _result(vdir, "spec_decoding", "A", 1.1, {"ppl_delta_pct": 9.0})
    r2.validate()
    assert r2.metrics["verdict"] == "unsafe"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Anti-fraud: the rejections an auditor checks first
# ─────────────────────────────────────────────────────────────────────────────


def test_rejects_verdict_contradicting_math():
    r = ValidationResult(
        **_base_kwargs(),
        metrics={"d_comm": 2.0, "tau": 3.0, "verdict": "safe",
                 "downstream_delta": {"x": 1.0}},
    )
    with pytest.raises(SchemaError, match="computed, not chosen"):
        r.validate()


def test_rejects_missing_downstream_evidence():
    r = ValidationResult(
        **_base_kwargs(),
        metrics={"d_comm": 4.0, "tau": 3.0, "downstream_delta": {}},
    )
    with pytest.raises(SchemaError, match="downstream_delta"):
        r.validate()


def test_rejects_nan_and_nonnumeric_downstream():
    r = ValidationResult(
        **_base_kwargs(),
        metrics={"d_comm": 4.0, "tau": 3.0,
                 "downstream_delta": {"x": float("nan")}},
    )
    with pytest.raises(SchemaError):
        r.validate()
    r2 = ValidationResult(
        **_base_kwargs(),
        metrics={"d_comm": float("nan"), "tau": 3.0,
                 "downstream_delta": {"x": 1.0}},
    )
    with pytest.raises(SchemaError, match="NaN"):
        r2.validate()


def test_rejects_unknown_collector_and_bad_tier():
    kw = _base_kwargs()
    kw["collector"] = "made_up_collector"
    r = ValidationResult(**kw, metrics={"d_comm": 4.0, "tau": 3.0,
                                        "downstream_delta": {"x": 1.0}})
    with pytest.raises(SchemaError, match="Unknown collector"):
        r.validate()
    kw2 = _base_kwargs()
    kw2["tier"] = "S"
    r2 = ValidationResult(**kw2, metrics={"d_comm": 4.0, "tau": 3.0,
                                          "downstream_delta": {"x": 1.0}})
    with pytest.raises(SchemaError, match="tier"):
        r2.validate()


def test_rejects_row_without_certificate_on_disk(vdir):
    r = _result(vdir, "lora", "C", 4.0, {"x": 1.0})
    r.cert_path = os.path.join(vdir, "lora", "does_not_exist.json")
    with pytest.raises(SchemaError, match="does not exist"):
        save_result(r, os.path.join(vdir, "lora", "result.json"))


def test_rejects_result_d_mismatching_certificate_d(vdir):
    r = _result(vdir, "model_swap", "B", 4.0, {"x": 0.1}, cert_d=5.0)
    with pytest.raises(SchemaError, match="does not match the"):
        save_result(r, os.path.join(vdir, "model_swap", "result.json"))


def test_accepts_result_d_matching_certificate_d_inf(vdir):
    r = _result(vdir, "prefix_cache", "B", "inf", {"x": 0.0})
    save_result(r, os.path.join(vdir, "prefix_cache", "result.json"))


def test_rejects_duplicate_collector_rows(vdir):
    r = _result(vdir, "kv_cache_quant", "A", 4.0, {"x": 0.1})
    save_result(r, os.path.join(vdir, "kv_cache_quant", "result.json"))
    dup_dir = os.path.join(vdir, "kv_cache_quant_rerun")
    shutil.copytree(os.path.join(vdir, "kv_cache_quant"), dup_dir)
    with pytest.raises(SchemaError, match="Duplicate collector"):
        load_all_results(vdir)


def test_rejects_unknown_schema_fields(vdir):
    r = _result(vdir, "weight_quant", "B", 4.0, {"x": 0.1})
    path = os.path.join(vdir, "weight_quant", "result.json")
    save_result(r, path)
    with open(path) as f:
        data = json.load(f)
    data["extra_field_someone_added"] = True
    with open(path, "w") as f:
        json.dump(data, f)
    with pytest.raises(SchemaError, match="unknown schema fields"):
        load_result(path)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Sub-block validation: trajectory and short_eval
# ─────────────────────────────────────────────────────────────────────────────


def test_trajectory_requires_safe_until_token():
    r = ValidationResult(
        **_base_kwargs(),
        metrics={"d_comm": 2.0, "tau": 3.0, "downstream_delta": {"x": 5.0},
                 "trajectory": {"measured": True}},
    )
    with pytest.raises(SchemaError, match="safe_until_token"):
        r.validate()


def test_short_eval_requires_all_three_fields():
    base = {"d_comm": 2.0, "tau": 3.0, "downstream_delta": {"x": 5.0}}
    for bad in (
        {"delta_pct": -0.3, "verdict_by_benchmark": "looks_safe"},
        {"benchmark": "HumanEval", "verdict_by_benchmark": "looks_safe"},
        {"benchmark": "HumanEval", "delta_pct": -0.3,
         "verdict_by_benchmark": "seems_fine"},
    ):
        r = ValidationResult(**_base_kwargs(),
                             metrics={**base, "short_eval": bad})
        with pytest.raises(SchemaError, match="short_eval"):
            r.validate()


# ─────────────────────────────────────────────────────────────────────────────
# 4. The five breakthrough conditions (A-E) render from the tables
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture()
def breakthrough_set(vdir):
    rows = [
        _result(vdir, "engine_swap", "A", 5.1,
                {"task_success_drop_pct": -0.2},
                gain={"throughput_pct": 18}),
        _result(vdir, "kv_cache_quant", "A", 2.4,
                {"coding_success_drop_pct": -14.0},
                trajectory={"measured": True, "safe_until_token": 1400,
                            "failure_after_token": 1700},
                short_eval={"benchmark": "HumanEval pass@1",
                            "delta_pct": -0.3,
                            "verdict_by_benchmark": "looks_safe"},
                gain={"throughput_x": 2.4}),
        _result(vdir, "spec_decoding", "A", 4.4,
                {"task_success_drop_pct": -0.1},
                gain={"throughput_x": 2.1}),
        _result(vdir, "prompt_swap", "B", 1.9,
                {"tool_routing_error_pct": 6.5}),
        _result(vdir, "allreduce_tp", "C", 4.7,
                {"task_success_drop_pct": -0.4}),
        _result(vdir, "prefix_cache", "C", "inf",
                {"task_success_drop_pct": 0.0}),
    ]
    for r in rows:
        save_result(r, os.path.join(vdir, r.collector, "result.json"))
    return vdir


def test_A_calibration_table_shows_d_downstream_alignment(breakthrough_set):
    tables = build_all_tables(breakthrough_set)
    cal = tables["calibration_table"]
    assert "tau agreement:" in cal
    assert "coding_success_drop_pct=-14.0" in cal
    assert "task_success_drop_pct=-0.2" in cal


def test_B_trajectory_table_carries_the_killer_row(breakthrough_set):
    traj = build_all_tables(breakthrough_set)["trajectory_table"]
    assert "HumanEval pass@1 -0.3% (looks safe)" in traj
    assert "1400" in traj and "1700" in traj
    assert "unsafe" in traj


def test_C_killer_chart_shows_business_gain(breakthrough_set):
    chart = build_all_tables(breakthrough_set)["killer_chart"]
    assert "+18% throughput" in chart
    assert "2.4x throughput" in chart
    assert "✅ Safe" in chart and "❌ Unsafe" in chart


def test_D_full_matrix_spans_change_families(breakthrough_set):
    matrix = build_all_tables(breakthrough_set)["full_matrix"]
    for name in ("kv_cache_quant", "engine_swap", "prompt_swap",
                 "allreduce_tp", "prefix_cache", "spec_decoding"):
        assert name in matrix
    assert "unit-test reason" in matrix


def test_E_caught_what_evals_missed_flagged_and_counted(breakthrough_set):
    cal = build_all_tables(breakthrough_set)["calibration_table"]
    assert "🎯 caught" in cal
    assert "passed standard benchmarks but were correctly" in cal
    lines = [l for l in cal.splitlines() if "engine_swap" in l]
    assert lines and "🎯" not in lines[0]


def test_tables_written_to_disk_with_generated_stamp(breakthrough_set, tmp_path):
    out = str(tmp_path / "tables")
    build_all_tables(breakthrough_set, out_dir=out)
    for name in ("killer_chart", "full_matrix", "trajectory_table",
                 "calibration_table"):
        p = os.path.join(out, f"{name}.md")
        assert os.path.exists(p)
        with open(p, encoding="utf-8") as f:
            assert "generated by harness" in f.read()


def test_killer_chart_requires_tier_A(vdir):
    r = _result(vdir, "lora", "C", 4.0, {"x": 0.1})
    save_result(r, os.path.join(vdir, "lora", "result.json"))
    with pytest.raises(SchemaError, match="Tier A"):
        build_all_tables(vdir)


def test_empty_validation_dir_hard_fails(tmp_path):
    empty = str(tmp_path / "nothing")
    os.makedirs(empty)
    with pytest.raises(SchemaError, match="No result.json"):
        load_all_results(empty)
