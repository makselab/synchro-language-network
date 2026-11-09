import sys
from pathlib import Path

import networkx as nx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from synchro.config import REST_PARTITION, TASK_PARTITION                          # noqa: E402
from synchro.structure import (all_min_deletions, automorphisms, certify, fibers,  # noqa: E402
                               graph_from_edges, is_balanced, load_baseline, min_deletion,
                               orbits, same_partition)
from synchro.weighted import load_tract_weights, min_tolerance                    # noqa: E402


def test_fibers_of_known_graphs():
    # directed cycle: all nodes receive one input from the same color -> one fiber
    C = nx.DiGraph([(0, 1), (1, 2), (2, 0)])
    assert same_partition(fibers(C), [[0, 1, 2]])
    # chain 0 -> 1 -> 2: the source has no input, so it is distinct
    P = nx.DiGraph([(0, 1), (1, 2)])
    assert same_partition(fibers(P), [[0], [1], [2]])


def test_fibration_without_automorphism():
    # nodes 1 and 2 both receive one input from node 0 and are in the same fiber,
    # but 2 also sends an edge to 3, which breaks the automorphism exchanging them
    G = nx.DiGraph([(0, 1), (0, 2), (2, 3)])
    assert any(set(g) == {1, 2} for g in fibers(G))
    assert not any(set(g) == {1, 2} for g in orbits(G))


def test_baseline_symmetry():
    G = load_baseline()
    assert G.number_of_edges() == 46
    assert all(G.has_edge(v, u) for u, v in G.edges())
    assert len(automorphisms(G)) == 4
    assert len(fibers(G)) == 5 and same_partition(fibers(G), orbits(G))


@pytest.mark.parametrize("P,removed", [(REST_PARTITION, 10), (TASK_PARTITION, 14)])
def test_deletion_only_inference(P, removed):
    G = load_baseline()
    H = min_deletion(G, P)
    assert G.number_of_edges() - H.number_of_edges() == removed
    assert all(certify(G, H, P).values())
    assert len(automorphisms(H)) == 1


def test_infeasible_without_min_in_degree_would_be_wrong():
    # removing all inputs balances any partition trivially; the in-degree constraint forbids it
    G = load_baseline()
    H = min_deletion(G, REST_PARTITION)
    assert min(dict(H.in_degree()).values()) >= 1
    empty = graph_from_edges([])
    assert is_balanced(empty, REST_PARTITION) and not certify(G, empty, REST_PARTITION)["min_in_degree"]


@pytest.mark.slow
def test_number_of_optimal_solutions():
    G = load_baseline()
    assert len(all_min_deletions(G, REST_PARTITION)) == 16
    assert len(all_min_deletions(G, TASK_PARTITION)) == 32


def test_weighted_tolerances():
    G = load_baseline()
    W = load_tract_weights(G)
    got = [100 * min_tolerance(G, P, W) for P in (fibers(G), TASK_PARTITION, REST_PARTITION)]
    assert got == pytest.approx([1.74, 1.49, 2.14], abs=0.01)


def test_unit_weights_are_exactly_feasible():
    G = load_baseline()
    W = {e: 1.0 for e in G.edges()}
    assert min_tolerance(G, REST_PARTITION, W) == 0.0


def test_percolation_selection_is_unique_and_matches_data():
    import json
    from synchro.structure import select_by_percolation
    G = load_baseline()
    W = load_tract_weights(G)
    data = Path(__file__).resolve().parents[2] / "data"
    for P, fname, n_opt in [(REST_PARTITION, "resting_state_network_fig4b.json", 16),
                            (TASK_PARTITION, "task_network_fig4c.json", 32)]:
        H, n, ties = select_by_percolation(G, P, W)
        assert n == n_opt and ties == 1
        assert all(certify(G, H, P).values()) and len(automorphisms(H)) == 1
        shown = {tuple(e) for e in json.load(open(data / fname))["edges"]}
        assert set(H.edges()) == shown


def test_task_requires_callosal_input_to_wa_left():
    # WA_R is the only neighbour of WA_L inside its task fiber: every minimal task
    # configuration keeps WA_R -> WA_L and removes BA_L -> WA_L
    G = load_baseline()
    for H in all_min_deletions(G, TASK_PARTITION):
        assert H.has_edge("WA_R", "WA_L") and not H.has_edge("BA_L", "WA_L")
