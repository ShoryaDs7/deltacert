"""Unit tests for collector 21 (free-running decode-dynamics certification).

Pure-Python signal-math tests only -- no GPU, no vLLM. The vLLM plumbing
(collect_free_running_vllm_two_engines, _score_under_baseline_vllm) is
validated separately on the pod (V1/V2/V3 in the design doc), the same way
collect_trajectory_vllm_two_engines was validated on the pod before being
trusted for the paper.
"""

import pytest

from deltacert.collectors import (
    CollectionError,
    FreeRunPromptResult,
    certify_free_running,
    distinct_ngram_ratio,
    fork_position,
    is_degenerate,
    max_window_token_freq,
    mcnemar_exact_p,
    surprisal_stats,
)


def test_degenerate_loop_detected():
    assert is_degenerate([7] * 200)


def test_healthy_varied_text_not_flagged():
    assert not is_degenerate(list(range(500)))


def test_loop_entered_midway_detected():
    """The Qwen signature: healthy prefix, then collapse into repetition."""
    assert is_degenerate(list(range(100)) + [7] * 150)


def test_distinct_ngram_floor():
    assert distinct_ngram_ratio([1, 2, 3] * 100) < 0.15


def test_fork_position_basic():
    assert fork_position([1, 2, 3, 4], [1, 2, 9, 4]) == 2
    assert fork_position([1, 2], [1, 2]) == -1


def test_fork_position_unequal_length_uses_shorter():
    assert fork_position([1, 2, 3], [1, 2]) == -1


def test_empty_generation_hard_fails():
    with pytest.raises(CollectionError):
        max_window_token_freq([])


def test_empty_surprisal_hard_fails():
    with pytest.raises(CollectionError):
        surprisal_stats([])


def test_certify_free_running_empty_results_hard_fails():
    with pytest.raises(CollectionError):
        certify_free_running([])


def test_short_generation_under_window_still_works():
    """n <= window branch of max_window_token_freq."""
    assert is_degenerate([3, 3, 3, 3, 3], window=64)
    assert not is_degenerate([1, 2, 3, 4, 5], window=64)


class _StubResult:
    def __init__(self, cand_degenerate, base_degenerate, cand_q95=1.0, base_q95=1.0, fork_pos=10):
        self.cand_degenerate = cand_degenerate
        self.base_degenerate = base_degenerate
        self.cand_surprisal_q95 = cand_q95
        self.base_surprisal_q95 = base_q95
        self.fork_pos = fork_pos


def test_certificate_fires_on_high_excess_degeneration():
    results = [_StubResult(True, False)] * 87 + [_StubResult(False, False)] * 13
    cert = certify_free_running(results)
    assert cert["verdict"] == "unsafe"
    assert cert["excess_degeneration_rate"] == pytest.approx(0.87)


def test_certificate_safe_when_no_excess_degeneration():
    results = [_StubResult(False, False)] * 100
    cert = certify_free_running(results)
    assert cert["verdict"] == "safe"
    assert cert["excess_degeneration_rate"] == 0.0


def test_paired_semantics_baseline_also_degenerate_not_counted():
    """Excess degeneration = P(candidate degenerate AND baseline clean).
    If the baseline degenerates on the same prompt too, it's not the
    candidate's fault -- must not count against it."""
    results = [_StubResult(True, True)] * 100
    cert = certify_free_running(results)
    assert cert["excess_degeneration_rate"] == 0.0
    assert cert["verdict"] == "safe"


def test_surprisal_delta_none_below_quantile_floor():
    """statistics.quantiles requires n>=20; below that, delta must be None
    rather than raising or silently computing a meaningless quantile."""
    results = [_StubResult(False, False)] * 5
    cert = certify_free_running(results)
    assert cert["surprisal_q95_delta"] is None


def test_surprisal_stats_shape():
    logprobs = [-0.1, -0.2, -5.0, -0.3, -0.15]
    mean, mx, q95 = surprisal_stats(logprobs)
    assert mx == 5.0  # max surprisal = -min(logprob)
    assert mean == pytest.approx(sum(-lp for lp in logprobs) / len(logprobs))


def test_mcnemar_matches_qwen_v1_real_data():
    """Regression pin against the actual pod result: b=34, c=0 must be
    overwhelmingly significant (two-sided exact test)."""
    p = mcnemar_exact_p(34, 0)
    assert p < 1e-9


def test_mcnemar_matches_llama_v2_real_data():
    """Regression pin against the actual pod result: b=1, c=3 must NOT be
    significant at alpha=0.01 -- this is re-roll noise, not caused collapse."""
    p = mcnemar_exact_p(1, 3)
    assert p == pytest.approx(0.625)
    assert p > 0.01


def test_mcnemar_symmetric_in_b_c():
    assert mcnemar_exact_p(1, 3) == mcnemar_exact_p(3, 1)
    assert mcnemar_exact_p(5, 12) == mcnemar_exact_p(12, 5)


def test_mcnemar_no_discordant_pairs_returns_one():
    """b=0, c=0: no evidence either way, not a hard failure."""
    assert mcnemar_exact_p(0, 0) == 1.0


def test_mcnemar_perfectly_balanced_is_not_significant():
    """b == c: textbook null case, p must be 1.0 (or very close)."""
    assert mcnemar_exact_p(10, 10) == pytest.approx(1.0)


def test_certify_requires_significance_not_just_rate():
    """THE TRAP THIS FIX CLOSES: a config that forks on every prompt and
    re-rolls baseline's own spontaneous degeneration rate can trip the raw
    excess-rate threshold without a real caused collapse. b=3, c=2 gives
    excess=0.05 (exactly at a naive tau_degen=0.05 boundary on n=100) but
    is statistically indistinguishable from a coin flip -- must NOT certify
    unsafe."""
    results = (
        [_StubResult(True, False)] * 3
        + [_StubResult(False, True)] * 2
        + [_StubResult(False, False)] * 95
    )
    cert = certify_free_running(results, tau_degen=0.02)
    assert cert["mcnemar_p"] > 0.01
    assert cert["degeneration_significant"] is False
    assert cert["verdict"] == "safe"


def test_certify_fires_when_rate_and_significance_both_hold():
    """The real Qwen shape: rate high AND lopsided (b>>c) -> unsafe."""
    results = [_StubResult(True, False)] * 34 + [_StubResult(False, False)] * 9
    cert = certify_free_running(results)
    assert cert["mcnemar_b"] == 34
    assert cert["mcnemar_c"] == 0
    assert cert["degeneration_significant"] is True
    assert cert["verdict"] == "unsafe"


def test_certify_llama_shape_stays_safe_under_new_rule():
    """The real Llama shape: b=1, c=3 out of 43 -- low rate AND not
    significant -- must stay safe under the corrected rule."""
    results = (
        [_StubResult(True, False)] * 1
        + [_StubResult(False, True)] * 3
        + [_StubResult(False, False)] * 39
    )
    cert = certify_free_running(results)
    assert cert["mcnemar_b"] == 1
    assert cert["mcnemar_c"] == 3
    assert cert["degeneration_significant"] is False
    assert cert["verdict"] == "safe"


def test_free_run_prompt_result_is_constructible():
    r = FreeRunPromptResult(
        prompt_sha="abc123",
        fork_pos=-1,
        base_degenerate=False,
        cand_degenerate=False,
        cand_surprisal_mean=0.5,
        cand_surprisal_max=1.0,
        cand_surprisal_q95=0.9,
        base_surprisal_q95=0.9,
        n_tokens_base=100,
        n_tokens_cand=100,
    )
    assert r.prompt_sha == "abc123"
