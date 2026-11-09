"""Phase locking value (PLV) and its surrogate-corrected version.

For two time series x_u(t), x_v(t) with instantaneous phases phi_u, phi_v
obtained from the analytic signal (Hilbert transform along time),

    r_uv = | < exp(-j (phi_u(t) - phi_v(t))) >_t |.

The surrogate-corrected PLV (Eq. 2 of the Methods) is

    sigma_uv = (r_uv - g_uv) / (1 - g_uv)   if r_uv >= g_uv,   0 otherwise,

where g_uv is the mean PLV between x_u and n_surrogates versions of x_v whose
samples are randomly shuffled in time.
"""

import numpy as np
from scipy.signal import hilbert


def analytic_phase(X):
    """Instantaneous phase of each column of X (time x ROIs).

    The Hilbert transform is taken along time (axis 0), separately for each ROI,
    and the phase is the full angle in (-pi, pi]. Each series is centred first,
    because the phase of the analytic signal is only meaningful for a signal
    oscillating around zero.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    Xc = X - X.mean(axis=0, keepdims=True)
    return np.angle(hilbert(Xc, axis=0))


def plv_matrix(X):
    """PLV between all pairs of columns of X (time x ROIs); diagonal is 1."""
    E = np.exp(1j * analytic_phase(X))
    return np.abs(E.conj().T @ E) / E.shape[0]


def surrogate_corrected_plv(X, n_surrogates=5000, rng=None):
    """Surrogate-corrected PLV matrix of one participant (time x ROIs).

    Returns a symmetric ROI x ROI matrix with zeros on the diagonal.
    For each ROI v, n_surrogates time-shuffled copies of x_v are generated and
    used for all pairs (u, v) with u < v.
    """
    rng = np.random.default_rng(rng)
    X = np.asarray(X, dtype=float)
    T, R = X.shape
    phase = analytic_phase(X)
    E = np.exp(1j * phase)
    r = np.abs(E.conj().T @ E) / T
    C = np.zeros((R, R))
    for v in range(1, R):
        idx = np.argsort(rng.random((T, n_surrogates)), axis=0)   # independent permutations
        surr_phase = analytic_phase(X[:, v][idx])                # T x n_surrogates
        Es = np.exp(1j * surr_phase)
        for u in range(v):
            g = np.mean(np.abs(E[:, u].conj() @ Es) / T)
            if r[u, v] >= g:
                C[u, v] = C[v, u] = (r[u, v] - g) / (1.0 - g)
    return C


def group_plv(X, n_surrogates=5000, seed=1):
    """Surrogate-corrected PLV for every participant of X (participants x time x ROIs).

    Returns an array (participants x ROIs x ROIs). The group matrix is its mean
    over participants.
    """
    rng = np.random.default_rng(seed)
    return np.stack([surrogate_corrected_plv(x, n_surrogates, rng) for x in X])
