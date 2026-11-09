import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from synchro import check_partition, cluster_plv, percolation_components  # noqa: E402

L = ["a", "b", "c", "d", "e"]


def toy():
    M = np.full((5, 5), 0.1)
    M[0, 1] = M[1, 0] = 0.9          # strong pair a-b
    for i, j in [(2, 3), (2, 4), (3, 4)]:
        M[i, j] = M[j, i] = 0.6      # triangle c-d-e
    M[1, 2] = M[2, 1] = 0.7          # b-c stronger than the triangle
    np.fill_diagonal(M, 0)
    return M


def test_cluster_plv_is_mean_of_internal_links():
    assert cluster_plv(toy(), ["c", "d", "e"], L) == pytest.approx(0.6)


def test_exclusion_of_more_synchronized_cliques():
    # c-d-e (0.6) has an outside link b-c = 0.7, but b belongs to the stronger clique a-b (0.9)
    rows = check_partition(toy(), [["a", "b"], ["c", "d", "e"]], L)
    assert [r["clique"] for r in rows] == [["a", "b"], ["c", "d", "e"]]
    assert all(r["valid"] for r in rows)
    assert rows[0]["max_outside"] == pytest.approx(0.7)


def test_violation_is_detected():
    M = toy()
    M[0, 1] = M[1, 0] = 0.65         # a-b now weaker than b-c
    rows = check_partition(M, [["a", "b"], ["c", "d", "e"]], L)
    assert not rows[0]["valid"]


def test_tolerance():
    M = toy()
    M[0, 1] = M[1, 0] = 0.695        # a-b 0.005 below b-c
    P = [["a", "b"], ["c", "d", "e"]]
    assert not check_partition(M, P, L)[0]["valid"]
    assert check_partition(M, P, L, tol=0.01)[0]["valid"]
    M[0, 1] = M[1, 0] = 0.68         # 0.02 below: not valid even with the tolerance
    assert not check_partition(M, P, L, tol=0.01)[0]["valid"]
    M[0, 1] = M[1, 0] = 0.69         # exactly 0.01 below: not valid (strict inequality)
    assert not check_partition(M, P, L, tol=0.01)[0]["valid"]


def test_overlapping_cliques_rejected():
    with pytest.raises(ValueError):
        check_partition(toy(), [["a", "b"], ["b", "c"]], L)


def test_percolation_sequence_order_and_final_state():
    seq = percolation_components(toy(), L)
    weights = [w for w, _, _ in seq]
    assert weights == sorted(weights, reverse=True)
    assert seq[0][1] in [("a", "b"), ("b", "a")]
    assert seq[-1][2] == [L]
