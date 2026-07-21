"""
DeltaCert tests — certify_safe() (the package's top-level __init__.py
"one call covers all 20 collectors" API) contract checks.

Regression coverage for two bugs found via live testing:

1. DeploymentBlocked contract violation: certify_safe's own docstring and
   __all__ export document DeploymentBlocked as "Raised by certify_safe
   when raise_on_fail=True and d < budget" -- but the code actually raised
   CollectionError in both failure paths. A caller following the documented
   `except dc.DeploymentBlocked:` pattern would never catch anything; the
   real exception would propagate unhandled.

2. Flat-budget false-unsafe: certify_safe's weight_quant path used to call
   _make_layer_result(cos_sims, threshold=budget) with budget defaulting to
   3.0 unconditionally -- the KV-cache/trajectory threshold, not weight
   quantization's. This is the same bug class fixed in hf_integration.py
   and cli.py's certify_system() call, appearing a third time in a third
   independent code path. Live-verified on a real Qwen2.5-1.5B int8
   quantization: d=1.708 (a genuinely safe reading, consistent with every
   other real bnb int8 measurement this session) was reported not
   certified before the fix.

These tests only exercise certify_safe's own control flow (the
raise-when-not-certified branch, and DeploymentBlocked's identity/message)
using synthetic cos_sims via monkeypatching -- no GPU or model download
required. The live GPU verification (real int8 model, d=1.708 -> certified
True after the fix) was performed manually this session; not re-run here
to keep the fast suite fast.
"""

import pytest

import deltacert as dc
import deltacert.collectors as collectors_module


def test_deployment_blocked_is_actually_raised_on_weight_quant_failure(monkeypatch):
    """certify_safe's documented contract: raise_on_fail=True + not
    certified -> dc.DeploymentBlocked, not dc.CollectionError."""

    def _fake_collect_weight_quant(baseline, candidate, tokenizer, prompts, device=None):
        return [0.5, 0.5, 0.5]  # low cos_sim -> low d_comm -> not certified

    monkeypatch.setattr(collectors_module, "collect_weight_quant", _fake_collect_weight_quant)

    with pytest.raises(dc.DeploymentBlocked):
        dc.certify_safe(
            baseline=object(), candidate=object(),
            tokenizer=object(), prompts=["p"],
            check="weight_quant", budget=999.0,
        )


def test_certify_safe_returns_false_without_raising_when_raise_on_fail_false(monkeypatch):
    def _fake_collect_weight_quant(baseline, candidate, tokenizer, prompts, device=None):
        return [0.5, 0.5, 0.5]

    monkeypatch.setattr(collectors_module, "collect_weight_quant", _fake_collect_weight_quant)

    result = dc.certify_safe(
        baseline=object(), candidate=object(),
        tokenizer=object(), prompts=["p"],
        check="weight_quant", budget=999.0, raise_on_fail=False,
    )
    assert result is False


def test_deployment_blocked_is_a_runtime_error_and_exported():
    """Public API surface check: DeploymentBlocked is importable from the
    top-level package (as documented) and is a RuntimeError subclass."""
    assert "DeploymentBlocked" in dc.__all__
    assert issubclass(dc.DeploymentBlocked, RuntimeError)
