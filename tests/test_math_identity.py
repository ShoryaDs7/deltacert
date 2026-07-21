"""Makes the paper's math claim executable: Delta(c) = ||[U,V]|| (Lemma 4.5,
Shorya 2026), where U=2uu*-I and V=2vv*-I are the reflections about the
normalized logit vectors u, v with <u,v>=c.

collectors.py never constructs U/V -- it computes 4c*sqrt(1-c^2) directly,
the simpler side of an equation that's true independent of the code (the
same way computing a circle's area as pi*r^2 doesn't require drawing the
circle). This test verifies the identity itself, literally, so "does the
formula match the paper's stated math claim" is `pytest -k commutator`
rather than something taken on faith.
"""
import math

import numpy as np
import pytest

from deltacert.collectors import _commutator_magnitude


def test_delta_formula_is_commutator_distance():
    """Paper §3.1: Delta(c) = ||[U,V]|| for reflections U=2uu*-I, V=2vv*-I
    about the normalized logit vectors (Lemma 4.5, Shorya 2026)."""
    rng = np.random.default_rng(42)
    for _ in range(20):
        n = int(rng.integers(3, 200))
        u = rng.normal(size=n)
        u /= np.linalg.norm(u)
        w = rng.normal(size=n)
        w -= (w @ u) * u
        w /= np.linalg.norm(w)
        c = float(rng.uniform(1 / math.sqrt(2), 1.0))  # operative (unclamped) range
        v = c * u + math.sqrt(1 - c * c) * w

        U = 2 * np.outer(u, u) - np.eye(n)
        V = 2 * np.outer(v, v) - np.eye(n)
        comm_norm = np.linalg.norm(U @ V - V @ U, 2)

        assert comm_norm == pytest.approx(4 * c * math.sqrt(1 - c * c), abs=1e-9)


def test_commutator_magnitude_matches_the_identity_directly():
    """Same identity, checked against the actual shipped function
    (_commutator_magnitude) rather than a hand-rolled formula -- catches
    drift if the code's implementation is ever refactored to diverge from
    what this test constructs independently."""
    rng = np.random.default_rng(7)
    for _ in range(20):
        n = int(rng.integers(3, 200))
        u = rng.normal(size=n)
        u /= np.linalg.norm(u)
        w = rng.normal(size=n)
        w -= (w @ u) * u
        w /= np.linalg.norm(w)
        c = float(rng.uniform(1 / math.sqrt(2), 1.0))
        v = c * u + math.sqrt(1 - c * c) * w

        U = 2 * np.outer(u, u) - np.eye(n)
        V = 2 * np.outer(v, v) - np.eye(n)
        comm_norm = np.linalg.norm(U @ V - V @ U, 2)

        assert comm_norm == pytest.approx(_commutator_magnitude(c), abs=1e-9)


def test_identity_does_not_hold_below_operative_range():
    """Below c=1/sqrt(2), _commutator_magnitude fail-closes to 2.0 rather
    than continuing the identity -- confirms the identity's executable
    check is correctly scoped to the operative range only, matching the
    paper's stated clamp (Eq. 3), not silently extended past it."""
    rng = np.random.default_rng(3)
    n = 50
    u = rng.normal(size=n)
    u /= np.linalg.norm(u)
    w = rng.normal(size=n)
    w -= (w @ u) * u
    w /= np.linalg.norm(w)
    c = 0.5  # below 1/sqrt(2) ~= 0.7071
    v = c * u + math.sqrt(1 - c * c) * w

    U = 2 * np.outer(u, u) - np.eye(n)
    V = 2 * np.outer(v, v) - np.eye(n)
    comm_norm = np.linalg.norm(U @ V - V @ U, 2)

    # the true commutator identity still holds mathematically here...
    assert comm_norm == pytest.approx(4 * c * math.sqrt(1 - c * c), abs=1e-9)
    # ...but the shipped fail-closed function deliberately does NOT track
    # it below the operative range -- it clamps to the max-divergence value.
    assert _commutator_magnitude(c) == 2.0
    assert _commutator_magnitude(c) != pytest.approx(comm_norm)
