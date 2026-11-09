"""Compute the surrogate-corrected PLV of every participant for every dataset.

Writes, for each dataset in synchro.config.DATASETS,
  results/<dataset>_plv_participants.npy   participants x ROIs x ROIs
  results/<dataset>_plv_group.csv          group-averaged matrix (ROI order of synchro.io.ROI_LABELS)
and the resting-state Pearson correlation matrices
  results/resting_state_spheres_corr_participants.npy, ..._corr_group.csv

Usage:  python scripts/compute_matrices.py [dataset ...]
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from synchro import load_dataset, group_plv, ROI_LABELS            # noqa: E402
from synchro.config import DATASETS, N_SURROGATES, SEED             # noqa: E402

RESULTS = Path(__file__).resolve().parents[2] / "results"


def main(names):
    RESULTS.mkdir(exist_ok=True)
    header = ",".join(ROI_LABELS)
    for name in names:
        t0 = time.time()
        X, _ = load_dataset(name)
        C = group_plv(X, N_SURROGATES, SEED)
        np.save(RESULTS / f"{name}_plv_participants.npy", C)
        np.savetxt(RESULTS / f"{name}_plv_group.csv", C.mean(0), delimiter=",", fmt="%.6f", header=header, comments="")
        print(f"{name}: {X.shape[0]} participants, {X.shape[1]} time points, {time.time() - t0:.0f} s")
        if name == "resting_state_spheres":
            R = np.stack([np.corrcoef(x.T) for x in X])
            np.save(RESULTS / f"{name}_corr_participants.npy", R)
            np.savetxt(RESULTS / f"{name}_corr_group.csv", R.mean(0), delimiter=",", fmt="%.6f", header=header, comments="")


if __name__ == "__main__":
    main(sys.argv[1:] or list(DATASETS))
