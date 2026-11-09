"""Kuramoto dynamics on structural networks (dynamical validation, Fig. 6).

    d theta_i / dt = omega_i + K * sum_j A_ij sin(theta_j - theta_i),

with A_ij = 1 if j -> i is an edge. Nodes of the same fiber get the same
natural frequency; different fibers get different frequencies, equally spaced
in [-spread, spread] and randomly assigned. Initial phases are equal within a
fiber up to a Gaussian perturbation of standard deviation `noise`, so that a
return to within-fiber synchrony tests the stability of the fiber partition.
"""

import networkx as nx
import numpy as np


def simulate(H, partition, seed=0, K=0.3, T=120.0, dt=0.01, noise=0.6, spread=3.0):
    """Euler integration; returns (times, phases) with phases of shape (steps, nodes)."""
    nodes = list(H.nodes())
    n = len(nodes)
    A = np.zeros((n, n))
    for u, v in H.edges():
        A[nodes.index(v), nodes.index(u)] = 1.0
    omega, theta = initial_conditions(nodes, partition, seed, noise, spread)
    steps = int(round(T / dt))
    traj = np.empty((steps, n))
    for t in range(steps):
        theta = theta + dt * (omega + K * (A * np.sin(theta[None, :] - theta[:, None])).sum(1))
        traj[t] = theta
    return np.arange(1, steps + 1) * dt, traj


def within_fiber_coherence(traj, nodes, partition):
    """Order parameter r_f(t) = |mean_{j in f} exp(i theta_j)| for each fiber f."""
    return np.stack([np.abs(np.exp(1j * traj[:, [nodes.index(x) for x in g]]).mean(1))
                     for g in partition], axis=1)


def phase_locking(traj, discard=0.5):
    """PLV matrix of the simulated phases after discarding the initial transient."""
    seg = traj[int(len(traj) * discard):]
    E = np.exp(1j * seg)
    return np.abs(E.conj().T @ E) / len(seg)


def observed_clusters(P, nodes, threshold=0.9):
    """Groups of oscillators connected by links with PLV >= threshold
    (percolation of the PLV matrix down to the threshold)."""
    G = nx.Graph()
    G.add_nodes_from(range(len(nodes)))
    G.add_edges_from((i, j) for i in range(len(nodes)) for j in range(i) if P[i, j] >= threshold)
    return [[nodes[i] for i in sorted(c)] for c in nx.connected_components(G)]


def initial_conditions(nodes, partition, seed, noise=0.6, spread=3.0):
    """Natural frequencies and initial phases used by simulate() for a given seed."""
    rng = np.random.default_rng(seed)
    fiber = {x: c for c, g in enumerate(partition) for x in g}
    w = np.linspace(-spread, spread, len(partition))
    rng.shuffle(w)
    ph0 = rng.uniform(0, 2 * np.pi, len(partition))
    omega = np.array([w[fiber[x]] for x in nodes])
    theta = np.array([ph0[fiber[x]] for x in nodes]) + rng.normal(0.0, noise, len(nodes))
    return omega, theta


def simulate_batch(H, partition, seeds, K=0.3, T=3000.0, dt=0.01, noise=0.6, spread=3.0,
                   sample_every=10, coherence_every=1.0):
    """Simulate many draws at once (same model and initial conditions as simulate()).

    Returns a dict with
      plv        (draws x nodes x nodes) PLV over the second half of the run,
      coherence  (draws x fibers) mean within-fiber coherence over the second half,
      times, coherence_t   within-fiber coherence sampled every `coherence_every`
                           time units (samples x draws x fibers).
    """
    nodes = list(H.nodes())
    n = len(nodes)
    A = np.zeros((n, n))
    for u, v in H.edges():
        A[nodes.index(v), nodes.index(u)] = 1.0
    ic = [initial_conditions(nodes, partition, s, noise, spread) for s in seeds]
    omega = np.array([c[0] for c in ic])
    theta = np.array([c[1] for c in ic])
    idx = [[nodes.index(x) for x in g] for g in partition]
    steps = int(round(T / dt))
    every_c = int(round(coherence_every / dt))
    acc = np.zeros((len(seeds), n, n), complex)
    coh_acc = np.zeros((len(seeds), len(partition)))
    count = 0
    times, coh_t = [], []
    for k in range(1, steps + 1):
        z = np.exp(1j * theta)
        # sum_j A_ij sin(theta_j - theta_i) = Im(conj(z_i) * sum_j A_ij z_j)
        theta = theta + dt * (omega + K * np.imag(np.conj(z) * (z @ A.T)))
        if k % sample_every == 0 or k % every_c == 0:
            z = np.exp(1j * theta)
            coh = np.stack([np.abs(z[:, i].mean(1)) for i in idx], 1)
            if k % every_c == 0:
                times.append(k * dt)
                coh_t.append(coh)
            if k % sample_every == 0 and k > steps // 2:
                acc += np.conj(z)[:, :, None] * z[:, None, :]
                coh_acc += coh
                count += 1
    return {"nodes": nodes, "plv": np.abs(acc) / count, "coherence": coh_acc / count,
            "times": np.array(times), "coherence_t": np.array(coh_t), "omega": omega}
