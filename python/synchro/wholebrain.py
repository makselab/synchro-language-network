"""Whole-brain control for shared or external input (Fig. 5, Tables 1-3).

Candidate shared signals are removed from the 11 ROI time series by ordinary
least squares (design matrix: the selected signals and an intercept), and the
surrogate-corrected PLV and the Eq. (1) test of the reported CS partition are
recomputed on the residuals.
"""

import json
from pathlib import Path

import numpy as np

from .io import ROI_LABELS, DATA_DIR
from .plv import group_plv

REGRESSORS = ["global", "A1", "PSL", "thalamus", "MGN", "IC"]

# the nine conditions of the whole-brain battery, in the order of Tables 1-3
CONDITIONS = {
    "raw": [],
    "global signal": ["global"],
    "A1": ["A1"],
    "PSL": ["PSL"],
    "thalamus": ["thalamus"],
    "MGN": ["MGN"],
    "inferior colliculus": ["IC"],
    "global+A1+PSL": ["global", "A1", "PSL"],
    "all combined": ["global", "A1", "PSL", "thalamus", "MGN", "IC"],
}

WHOLE_BRAIN_DATASETS = ["resting_state", "letter_fluency", "verb_generation"]


def load_whole_brain(name, data_dir=DATA_DIR):
    """Return (X, R, meta): ROI series (participants x time x 11), candidate
    signals (participants x time x 6, columns as in REGRESSORS) and metadata."""
    with open(Path(data_dir) / f"whole_brain_{name}.json") as f:
        doc = json.load(f)
    X = np.asarray(doc.pop("data"), dtype=float)
    R = np.asarray(doc.pop("regressor_data"), dtype=float)
    if [r["label"] for r in doc["rois"]] != ROI_LABELS or doc["regressors"] != REGRESSORS:
        raise ValueError(f"whole_brain_{name}.json: unexpected ROI or regressor order")
    if X.shape[:2] != R.shape[:2] or X.shape[2] != 11 or R.shape[2] != len(REGRESSORS):
        raise ValueError(f"whole_brain_{name}.json: inconsistent shapes {X.shape}, {R.shape}")
    return X, R, doc


def regress_out(Y, S):
    """Residuals of the columns of Y (time x ROIs) after OLS on the columns of S
    (time x signals) plus an intercept."""
    Y = np.asarray(Y, dtype=float)
    if S is None or np.size(S) == 0:
        return Y.copy()
    D = np.column_stack([np.asarray(S, dtype=float), np.ones(len(Y))])
    beta, *_ = np.linalg.lstsq(D, Y, rcond=None)
    return Y - D @ beta


def condition_matrices(X, R, n_surrogates=5000, seed=1):
    """Group surrogate-corrected PLV matrix for each of the nine conditions."""
    out = {}
    for cond, regs in CONDITIONS.items():
        cols = [REGRESSORS.index(r) for r in regs]
        Xr = np.stack([regress_out(x, r[:, cols]) for x, r in zip(X, R)])
        out[cond] = group_plv(Xr, n_surrogates, seed).mean(0)
    return out
