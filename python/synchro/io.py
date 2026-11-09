"""Reading the ROI time series distributed in data/*.json."""

import json
from pathlib import Path

import numpy as np

# Order of the ROI dimension in every data file
ROI_LABELS = ["SMG_L", "SMG_R", "PreMA_L", "PreMA_R", "SMA",
              "WA_L", "WA_R", "AG_L", "AG_R", "BA_L", "BA_R"]

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def load_dataset(name, data_dir=DATA_DIR):
    """Return (X, meta) for data/<name>.json.

    X has shape (participants, time points, ROIs); meta is the JSON document
    without the 'data' field.
    """
    path = Path(data_dir) / f"{name}.json"
    with open(path) as f:
        doc = json.load(f)
    X = np.asarray(doc.pop("data"), dtype=float)
    labels = [r["label"] for r in doc["rois"]]
    if labels != ROI_LABELS:
        raise ValueError(f"{path}: unexpected ROI order {labels}")
    expected = (doc["n_participants"], doc["n_timepoints"], doc["n_rois"])
    if X.shape != expected:
        raise ValueError(f"{path}: data shape {X.shape} differs from metadata {expected}")
    if not np.all(np.isfinite(X)):
        raise ValueError(f"{path}: non-finite values in data")
    return X, doc
