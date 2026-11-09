import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from synchro.config import REST_PARTITION                                   # noqa: E402
from synchro.kuramoto import observed_clusters, phase_locking, simulate, within_fiber_coherence  # noqa: E402
from synchro.structure import graph_from_edges, load_baseline, same_partition  # noqa: E402
from synchro.wholebrain import WHOLE_BRAIN_DATASETS, load_whole_brain, regress_out  # noqa: E402


def test_regress_out_removes_signal_and_keeps_orthogonal_part():
    rng = np.random.default_rng(0)
    s = rng.standard_normal((200, 2))
    e = rng.standard_normal((200, 3))
    Y = s @ rng.standard_normal((2, 3)) + e + 5.0
    res = regress_out(Y, s)
    D = np.column_stack([s, np.ones(200)])
    assert np.allclose(D.T @ res, 0, atol=1e-9)
    assert np.allclose(regress_out(Y, np.empty((200, 0))), Y)


def test_whole_brain_data():
    for name in WHOLE_BRAIN_DATASETS:
        X, R, meta = load_whole_brain(name)
        assert X.shape[0] == 20 and X.shape[2] == 11 and R.shape[2] == 6
        assert np.all(np.isfinite(X)) and np.all(np.isfinite(R))


def test_kuramoto_two_independent_fibers():
    # two disconnected 2-cycles with different frequencies: two clusters
    H = graph_from_edges([("a", "b"), ("b", "a"), ("c", "d"), ("d", "c")], nodes=["a", "b", "c", "d"])
    P = [["a", "b"], ["c", "d"]]
    t, tr = simulate(H, P, seed=0, K=1.0, T=60)
    nodes = list(H.nodes())
    assert same_partition(observed_clusters(phase_locking(tr), nodes), P)
    assert within_fiber_coherence(tr, nodes, P)[-1].min() > 0.999


def test_kuramoto_fig4b_network_weak_coupling():
    import json
    data = Path(__file__).resolve().parents[2] / "data" / "resting_state_network_fig4b.json"
    H = graph_from_edges([tuple(e) for e in json.load(open(data))["edges"]])
    assert set(H.edges()) <= set(load_baseline().edges())
    nodes = list(H.nodes())
    ok = sum(same_partition(observed_clusters(phase_locking(simulate(H, REST_PARTITION, seed=s, K=0.3)[1]), nodes),
                            REST_PARTITION) for s in range(10))
    assert ok >= 8


def _fig4b():
    import json
    data = Path(__file__).resolve().parents[2] / "data" / "resting_state_network_fig4b.json"
    return graph_from_edges([tuple(e) for e in json.load(open(data))["edges"]])


def test_fiber_synchrony_is_invariant_and_edge_direction_matters():
    # without perturbation the fibers stay exactly synchronous on the balanced network,
    # but not on the network with reversed edges (where the partition is not balanced)
    H = _fig4b()
    nodes = list(H.nodes())
    _, tr = simulate(H, REST_PARTITION, seed=3, K=0.6, T=50, noise=0.0)
    assert within_fiber_coherence(tr, nodes, REST_PARTITION).min() > 1 - 1e-12
    _, tr = simulate(H.reverse(copy=True), REST_PARTITION, seed=3, K=0.6, T=50, noise=0.0)
    assert within_fiber_coherence(tr, nodes, REST_PARTITION).min() < 0.9


def test_batch_and_single_integrators_agree():
    from synchro.kuramoto import simulate_batch
    H = _fig4b()
    b = simulate_batch(H, REST_PARTITION, range(4), K=0.6, T=60, sample_every=1)
    for s in range(4):
        _, tr = simulate(H, REST_PARTITION, seed=s, K=0.6, T=60)
        assert np.allclose(phase_locking(tr), b["plv"][s], atol=1e-10)


def test_euler_matches_adaptive_integrator():
    from scipy.integrate import solve_ivp
    from synchro.kuramoto import initial_conditions
    H = _fig4b()
    nodes = list(H.nodes())
    A = np.zeros((11, 11))
    for u, v in H.edges():
        A[nodes.index(v), nodes.index(u)] = 1.0
    om, th0 = initial_conditions(nodes, REST_PARTITION, 5)
    sol = solve_ivp(lambda t, th: om + 0.6 * np.imag(np.exp(-1j * th) * (A @ np.exp(1j * th))),
                    (0, 60), th0, t_eval=np.arange(1, 6001) * 0.01, rtol=1e-9, atol=1e-11)
    _, tr = simulate(H, REST_PARTITION, seed=5, K=0.6, T=60)
    assert same_partition(observed_clusters(phase_locking(tr), nodes),
                          observed_clusters(phase_locking(sol.y.T), nodes))
