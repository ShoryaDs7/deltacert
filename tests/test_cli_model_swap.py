"""
DeltaCert tests — `deltacert certify --checks model_swap` flag validation.

Regression coverage for a bug found via live CLI testing: model_swap's
underlying measurement (collect_model_swap(capture_a_path, candidate_path)
in deltacert.py) needs BOTH a baseline capture and a candidate capture, but
the CLI's pre-flight flag validation only required --candidate. Running
`--checks model_swap --candidate x.npz` without --baseline passed the
single-flag check (capture_a_path stayed None) and crashed deep inside
load_logits with a raw, unhelpful TypeError instead of the same clean
"requires --baseline" pre-flight message every other multi-input check
(prompt_swap, engine_swap) already gets.

Invokes the real CLI via subprocess -- guaranteed to match actual usage
exactly, no risk of a hand-built argparse tree drifting from the real one.
No GPU/model download needed: the missing-flag check fires before any
model loading.
"""

import subprocess
import sys


def test_model_swap_without_baseline_fails_cleanly_not_a_crash():
    result = subprocess.run(
        [sys.executable, "-m", "deltacert.cli", "certify",
         "--model", "irrelevant/not-loaded",
         "--checks", "model_swap",
         "--candidate", "some_candidate.npz"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 1, (
        f"expected clean exit 1, got {result.returncode}\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "Traceback" not in result.stdout and "Traceback" not in result.stderr, (
        "model_swap without --baseline crashed with a raw traceback instead "
        f"of the pre-flight validation message\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "requires --baseline" in result.stdout, (
        f"expected the pre-flight 'requires --baseline' message\nstdout: {result.stdout}"
    )


def test_model_swap_without_candidate_fails_cleanly():
    result = subprocess.run(
        [sys.executable, "-m", "deltacert.cli", "certify",
         "--model", "irrelevant/not-loaded",
         "--checks", "model_swap",
         "--baseline", "some_baseline.npz"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 1
    assert "requires --candidate" in result.stdout
