import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from synchro import analytic_phase, plv_matrix, surrogate_corrected_plv  # noqa: E402

T = 400
t = np.arange(T) * 2.5


def oscillation(freq, phase=0.0, noise=0.0, rng=None):
    rng = np.random.default_rng(rng)
    return np.sin(2 * np.pi * freq * t + phase) + noise * rng.standard_normal(T)


def test_identical_signals_have_plv_one():
    x = oscillation(0.05, noise=0.3, rng=0)
    assert plv_matrix(np.c_[x, x])[0, 1] == pytest.approx(1.0, abs=1e-12)


def test_constant_phase_lag_gives_plv_one():
    # a fixed lag must not reduce phase locking (fails if the phase is folded into (-pi/2, pi/2))
    for lag in [0.3, 1.0, 2.0, 3.0]:
        X = np.c_[oscillation(0.05), oscillation(0.05, lag)]
        assert plv_matrix(X)[0, 1] > 0.98, lag


def test_phase_spans_full_circle():
    ph = analytic_phase(oscillation(0.05))
    assert ph.min() < -3.0 and ph.max() > 3.0


def test_independent_noise_has_low_plv():
    rng = np.random.default_rng(1)
    P = plv_matrix(rng.standard_normal((T, 6)))
    assert np.all(P[np.triu_indices(6, 1)] < 0.25)


def test_invariant_to_column_order():
    # the Hilbert transform must act along time, not across ROIs
    rng = np.random.default_rng(2)
    X = rng.standard_normal((T, 11)).cumsum(0)
    perm = rng.permutation(11)
    P = plv_matrix(X)
    Pp = plv_matrix(X[:, perm])
    assert np.allclose(Pp, P[np.ix_(perm, perm)], atol=1e-12)


def test_invariant_to_offset():
    # a large constant offset (uncentred BOLD signal) must not change the result
    rng = np.random.default_rng(3)
    X = rng.standard_normal((T, 4)).cumsum(0)
    assert np.allclose(plv_matrix(X), plv_matrix(X + 6500.0), atol=1e-9)


def test_surrogate_corrected_range_and_symmetry():
    rng = np.random.default_rng(4)
    common = rng.standard_normal(T).cumsum()
    X = np.c_[common + rng.standard_normal(T), common + rng.standard_normal(T), rng.standard_normal((T, 3))]
    C = surrogate_corrected_plv(X, n_surrogates=500, rng=5)
    assert np.allclose(C, C.T) and np.all(np.diag(C) == 0)
    assert np.all((C >= 0) & (C <= 1))
    assert C[0, 1] > 0.5                       # coupled pair
    assert np.all(C[2:, 2:][np.triu_indices(3, 1)] < 0.2)   # independent series


def test_surrogate_corrected_reproducible_with_seed():
    rng = np.random.default_rng(6)
    X = rng.standard_normal((T, 4))
    assert np.array_equal(surrogate_corrected_plv(X, 200, rng=7), surrogate_corrected_plv(X, 200, rng=7))
