"""
DeltaCert tests — hf_integration.py argument validation.

Regression coverage for the fp8 silent-no-op bug: auto_certify used to let
quantization="fp8" (and any other unrecognized string) fall through to
bnb_config=None, silently comparing fp16 against an unquantized reload of
itself and certifying a change that was never applied. Hard-failure
semantics (paper Contribution 2): any condition that would yield an
untrustworthy measurement must raise, never default.

These two tests only exercise the whitelist check, which runs before any
model/tokenizer loading -- no GPU, network, or model download required.
(int8/int4/None are NOT covered here: they pass validation and proceed to
real tokenizer/model loading, which needs a real model name and either a
live GPU+network or a mocked HF stack -- see test_smoke.py for that
end-to-end coverage, and this session's live-run log for the manual
int8/int4/fp8 verification on a real model.)
"""

import pytest

from deltacert.integrations.hf_integration import auto_certify


def test_unsupported_fp8_quantization_raises():
    with pytest.raises(ValueError, match="fp8"):
        auto_certify(
            model_name="irrelevant/not-loaded",
            calibration_prompts=["hello"],
            quantization="fp8",
        )


def test_unsupported_quantization_string_raises():
    """Any string the backend can't actually apply must raise, not just
    the specific value 'fp8' -- the fix is a whitelist, not a special
    case, so an arbitrary typo/unknown method must be rejected too."""
    with pytest.raises(ValueError, match="not_a_real_quant_method"):
        auto_certify(
            model_name="irrelevant/not-loaded",
            calibration_prompts=["hello"],
            quantization="not_a_real_quant_method",
        )
