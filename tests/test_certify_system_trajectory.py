"""
DeltaCert tests — certify_system() trajectory result integrity.

Regression coverage for a bug found via live CLI smoke-testing: the
trajectory branch of certify_system() computed a real result and stored it
in precomputed_layers (which certify() applies first, correctly), but also
appended a redundant LayerSpec for "trajectory" into layer_specs. certify()
then separately loops over layer_specs checking calibration.get(spec.name)
-- trajectory never populates `calibration` (it has its own measurement
path) -- so that second loop silently overwrote the correct precomputed
result with a fake {"d_comm": None, "error": "no calibration data
provided"}. `deltacert certify --checks trajectory` always reported
d=N/A/not-certified regardless of the real measurement.

This test monkeypatches the trajectory collector to avoid loading a real
model -- it only needs to prove certify_system()'s own result-plumbing is
correct, not re-verify the trajectory math (already covered by
test_math_identity.py and the real live trajectory certs).
"""

import deltacert.collectors as collectors_module
import deltacert as dc


def test_certify_system_trajectory_result_is_not_overwritten(monkeypatch):
    # d_profile() builds a correctly-shaped profile dict from real cos_sims
    # -- reuses the actual production function rather than guessing the
    # internal schema, so this test breaks (loudly) if that schema changes.
    fake_profile = collectors_module.d_profile([0.99, 0.98, 0.97, 0.96])

    def _fake_collect_trajectory_two_models(model_a, model_b, tokenizer, cases, device):
        return [fake_profile]

    monkeypatch.setattr(
        collectors_module, "collect_trajectory_two_models",
        _fake_collect_trajectory_two_models,
    )

    cert = dc.certify_system(
        model="fake-model",
        prompts=[],
        checks=["trajectory"],
        trajectory_model_b=object(),
        trajectory_cases=[("prompt", "continuation")],
        tokenizer=object(),
        model_base=object(),
        budget=3.0,
    )

    layer = cert["layers"]["trajectory"]
    assert layer.get("error") is None, (
        f"trajectory layer has an error despite a real profile being "
        f"computed -- the precomputed result was overwritten: {layer}"
    )
    assert layer.get("d_comm") is not None, (
        f"trajectory d_comm is None despite a real profile being computed "
        f"-- the precomputed result was overwritten: {layer}"
    )
