"""Checks on the distributed data and on the values reported in the paper.

The last test recomputes the group matrices from scratch (a few minutes);
skip it with:  pytest -m "not slow"
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from synchro import load_dataset, group_plv, check_partition, ROI_LABELS  # noqa: E402
from synchro.config import DATASETS, eq1_tolerance                       # noqa: E402

# Cluster PLVs reported in the paper (Methods and Supplementary Information)
PAPER = {
    "resting_state_spheres": {"SMG_L-SMG_R": 0.726, "PreMA_L-PreMA_R-SMA-WA_L-WA_R": 0.712,
                              "AG_L-AG_R": 0.692, "BA_L-BA_R": 0.690},
    "letter_fluency_spheres": {"BA_L-SMA": 0.758, "AG_L-AG_R": 0.621, "PreMA_L-PreMA_R": 0.570,
                               "BA_R-WA_L-WA_R": 0.477, "SMG_L-SMG_R": 0.444},
    "verb_generation_spheres": {"BA_L-SMA": 0.714, "PreMA_L-PreMA_R": 0.614, "AG_L-AG_R": 0.517,
                                "BA_R-WA_L-WA_R": 0.467, "SMG_L-SMG_R": 0.430},
}


@pytest.mark.parametrize("name", list(DATASETS))
def test_dataset_is_well_formed(name):
    X, meta = load_dataset(name)
    assert X.shape[0] == 20 and X.shape[2] == 11
    assert meta["TR_seconds"] == 2.5
    # series are centred (detrended data): the mean is negligible with respect to the fluctuations
    assert np.abs(X.mean(1)).max() < 1e-6 * X.std(1).mean() + 1e-9


@pytest.mark.slow
@pytest.mark.parametrize("name", list(DATASETS))
def test_reported_values_and_eq1(name):
    X, _ = load_dataset(name)
    M = group_plv(X).mean(0)
    partition = DATASETS[name][0]
    rows = check_partition(M, partition, ROI_LABELS, eq1_tolerance(name))
    if name in PAPER:
        got = {"-".join(r["clique"]): r["cluster_plv"] for r in rows}
        for clique, value in PAPER[name].items():
            assert got[clique] == pytest.approx(value, abs=0.0015), clique
    assert all(r["valid"] for r in rows)
