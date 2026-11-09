"""Source Data file (results/Source_Data.xlsx) for the graphs of the paper.

Run after compute_matrices.py, whole_brain_control.py, kuramoto_validation.py,
weighted_tolerance.py and structure_analysis.py.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Font

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from synchro import ROI_LABELS                      # noqa: E402
from synchro.structure import load_baseline         # noqa: E402

RESULTS = Path(__file__).resolve().parents[2] / "results"
DATA = Path(__file__).resolve().parents[2] / "data"
OUT = RESULTS / "Source_Data.xlsx"


def group(name):
    return pd.read_csv(RESULTS / f"{name}_plv_group.csv", index_col=None).set_axis(ROI_LABELS, axis=0)


def part(name, kind="plv"):
    return np.load(RESULTS / f"{name}_{kind}_participants.npy")


def jsd(P, Q):
    m = (P + Q) / 2
    kl = lambda p, q: np.sum(p[(p > 0) & (q > 0)] * np.log(p[(p > 0) & (q > 0)] / q[(p > 0) & (q > 0)]))  # noqa: E731
    return 0.5 * (kl(P, m) + kl(Q, m))


def stacked(blocks):
    """Several labelled matrices one below the other, separated by a title row."""
    rows = []
    for title, M in blocks:
        rows.append([title] + [""] * len(ROI_LABELS))
        rows.append(["ROI"] + ROI_LABELS)
        for r, lab in enumerate(ROI_LABELS):
            rows.append([lab] + [round(float(x), 6) for x in M[r]])
        rows.append([""] * (len(ROI_LABELS) + 1))
    return pd.DataFrame(rows)


def main():
    sheets = {}
    readme = [
        ["Source Data for: Fibration symmetry-breaking supports functional transitions in a brain network engaged in language"],
        ["Values are the surrogate-corrected phase locking value (PLV) unless stated otherwise; n = 20 participants."],
        ["ROI order: " + ", ".join(ROI_LABELS)],
        ["All values are produced by the code at the repository cited in Code availability (scripts listed per sheet)."],
        [""], ["Sheet", "Content", "Script"]]
    names = {"Resting state": "resting_state_spheres", "Letter fluency": "letter_fluency_spheres",
             "Verb generation": "verb_generation_spheres"}

    def add(sheet, df, content, script, header=False):
        sheets[sheet] = (df, header)
        readme.append([sheet, content, script])

    # Fig. 1b: resting-state BOLD time series of three ROIs of one participant (the participant with the
    # median within-cluster correlation), volumes 20-89, z-scored, as drawn in the fMRI-signal panel
    from synchro import load_dataset
    from synchro.config import REST_PARTITION
    X, meta = load_dataset("resting_state_spheres")
    li = ROI_LABELS.index
    score = [np.mean([np.corrcoef(x.T)[li(a), li(b)] for c in REST_PARTITION for a in c for b in c if a < b]) for x in X]
    pid = int(np.argsort(score)[len(score) // 2])
    seg = np.arange(20, 90)
    cols = {"time (s)": seg * meta["TR_seconds"]}
    for roi in ["SMA", "SMG_L", "BA_L"]:
        y = X[pid][seg, li(roi)]
        cols[roi + " (z-scored)"] = np.round((y - y.mean()) / y.std(), 6)
    add("Fig 1b", pd.DataFrame(cols), f"BOLD time series of participant P{pid + 1:02d} shown in Fig. 1b", "source_data.py",
        True)
    add("Fig 2", stacked([(f"{k}: group PLV matrix (Fig. 2{p})", group(v).values) for (k, v), p in zip(names.items(), "bdf")]),
        "Group PLV matrices, Fig. 2b,d,f", "compute_matrices.py")
    base = load_baseline()
    E = []
    for lab, edges in [("baseline (Fig. 4a)", base.edges()),
                       ("resting state (Fig. 4b)", json.load(open(DATA / "resting_state_network_fig4b.json"))["edges"]),
                       ("task (Fig. 4c)", json.load(open(DATA / "task_network_fig4c.json"))["edges"])]:
        E += [[lab, u, v] for u, v in sorted(map(tuple, edges))]
    add("Fig 4", pd.DataFrame(E, columns=["network", "source ROI", "target ROI"]),
        "Directed edges of the networks of Fig. 4 and Supplementary Figs. 4-6", "structure_analysis.py", True)
    wb = pd.read_csv(RESULTS / "whole_brain_tables.csv")
    add("Fig 5 and Tables 1-3", wb, "Cluster PLV and Eq. (1) validity (PLV tolerance 0.01), whole-brain control", "whole_brain_control.py", True)
    add("Fig 6", pd.read_csv(RESULTS / "kuramoto_coherence_timecourse.csv"),
        "Within-fiber coherence, median and quartiles over 500 draws (K = 0.3)", "kuramoto_validation.py", True)
    add("Fig 6 per draw", pd.read_csv(RESULTS / "kuramoto_outcomes.csv"),
        "Outcome and mean within-fiber coherence of each draw", "kuramoto_validation.py", True)
    add("Fig 7", pd.read_csv(RESULTS / "weighted_tolerance.csv"), "Minimum tolerance of the weighted balanced coloring",
        "weighted_tolerance.py", True)
    C = {k: part(v) for k, v in names.items()}
    iu = np.triu_indices(11, 1)
    g = C["Resting state"].mean(0)[iu]
    subj = int(np.argmax([np.corrcoef(c[iu], g)[0, 1] for c in C["Resting state"]]))
    add("Suppl Fig 2", stacked([(f"{k}: participant P{subj + 1:02d}", v[subj]) for k, v in C.items()]),
        "Single-participant PLV matrices", "make_figures.py")
    add("Suppl Fig 7", stacked([("Resting state, Glasser parcels: group PLV", group("resting_state_glasser").values)]),
        "Group PLV, Glasser parcels", "compute_matrices.py")
    cv = {k: np.divide(v.std(0), v.mean(0), out=np.zeros((11, 11)), where=v.mean(0) > 0) for k, v in C.items()}
    add("Suppl Fig 8", stacked([(f"{k}: coefficient of variation", m) for k, m in cv.items()]),
        "Coefficient of variation across participants", "make_figures.py")
    rows = []
    for k, v in C.items():
        P = [c[iu] / c[iu].sum() for c in v]
        J = np.array([[jsd(P[a], P[b]) if a != b else 0.0 for b in range(20)] for a in range(20)])
        rows.append([f"{k}: Jensen-Shannon divergence between participants"])
        rows.append(["participant"] + [f"P{i + 1:02d}" for i in range(20)])
        rows += [[f"P{a + 1:02d}"] + [round(float(x), 6) for x in J[a]] for a in range(20)]
        rows.append([""])
    add("Suppl Fig 9", pd.DataFrame(rows), "Jensen-Shannon divergence between participants", "make_figures.py")
    corr = pd.read_csv(RESULTS / "resting_state_spheres_corr_group.csv").values
    add("Suppl Fig 10", stacked([("Resting state: group Pearson correlation", corr)]), "Group Pearson correlation",
        "compute_matrices.py")
    add("Suppl Fig 11", stacked([("Resting state, HRF-deconvolved, spheres: group PLV",
                                  group("resting_state_spheres_hrf_deconvolved").values)]),
        "Group PLV after HRF deconvolution, spheres", "compute_matrices.py")
    add("Suppl Fig 12", stacked([("Resting state, HRF-deconvolved, Glasser parcels: group PLV",
                                  group("resting_state_glasser_hrf_deconvolved").values)]),
        "Group PLV after HRF deconvolution, Glasser parcels", "compute_matrices.py")
    add("Suppl Fig 13", stacked([("Letter fluency, band-pass filtered: group PLV", group("letter_fluency_spheres_bandpass").values),
                                 ("Verb generation, band-pass filtered: group PLV", group("verb_generation_spheres_bandpass").values)]),
        "Group PLV of band-pass filtered task data", "compute_matrices.py")
    readme += [[""], ["HRF, hemodynamic response function; ROI, region of interest."]]
    with pd.ExcelWriter(OUT, engine="openpyxl") as xw:
        pd.DataFrame(readme).to_excel(xw, sheet_name="README", header=False, index=False)
        for name, (df, header) in sheets.items():
            df.to_excel(xw, sheet_name=name, header=header, index=False)
    wbk = load_workbook(OUT)
    for ws in wbk.worksheets:
        for row in ws.iter_rows():
            for c in row:
                c.font = Font(name="Arial", size=10, bold=(ws.title == "README" and c.row in (1, 6)))
        ws.column_dimensions["A"].width = 30 if ws.title != "README" else 26
    wbk.save(OUT)
    print("written", OUT, "sheets:", len(wbk.worksheets))


if __name__ == "__main__":
    main()
