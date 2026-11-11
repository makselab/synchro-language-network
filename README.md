# Cluster synchronization of the language network: code and data

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24094720.svg)](https://doi.org/10.5281/zenodo.24094720)

Code and data to reproduce the analyses and the data figures of

> T. Gili, B. Avila, L. Pasquini, A. Holodny, D. Phillips, P. Boldi, A. Gabrielli, G. Caldarelli, M. Zimmer, H. A. Makse,
> *Fibration symmetry-breaking supports functional transitions in a brain network engaged in language*, Nature Communications (2026).

The repository contains a Python implementation of all analyses (used for the figures) and an equivalent MATLAB implementation of the synchronization analysis. Both run on the data in `data/`.

## Contents

```
data/                 ROI time series, connectomes and tract weights (JSON, see data/README.md)
python/synchro/       Python package: PLV, Eq. (1) criterion, symmetry and MILP inference,
                      Kuramoto model, weighted tolerance, whole-brain control
python/scripts/       one script per group of results (see below)
python/tests/         unit tests and checks of the values reported in the paper
matlab/               MATLAB / GNU Octave implementation and tests
assets/               network drawings used in Fig. 2a, c, e
results/              output of the scripts (created when they run)
```

## Getting the code

```bash
git clone https://github.com/makselab/synchro-language-network.git
cd synchro-language-network
```

## Requirements

- Python 3.9 or later with the packages in `python/requirements.txt` (`pip install -r python/requirements.txt`).
- Optionally, MATLAB R2019b or later, or GNU Octave 7 or later. No toolbox is required.

## Reproducing the results

Python, from the `python/` folder:

```bash
python scripts/compute_matrices.py   # PLV of every participant and group matrices (about 2 minutes)
python scripts/report_values.py      # Cluster PLVs and Eq. (1) test, as reported in the paper
python scripts/make_figures.py       # Fig. 2 and Supplementary Figs. 2, 7-13 in results/
python scripts/structure_analysis.py # symmetries, MILP inference and network selection (Fig. 4)
python scripts/network_figures.py    # drawings of Fig. 4 and Supplementary Figs. 4-6
python scripts/figure_3_schematic.py # schematic of Fig. 3
python scripts/source_data.py        # Source Data file (results/Source_Data.xlsx), after the scripts above
python scripts/whole_brain_control.py   # Fig. 5 and Tables 1-3 (about 15 minutes)
python scripts/kuramoto_validation.py   # Fig. 6 (about 2 minutes)
python scripts/weighted_tolerance.py    # Fig. 7
pytest                               # all tests (about 2 minutes); "pytest -m 'not slow'" for the quick ones
```

MATLAB or Octave, from the `matlab/` folder:

```matlab
run_all       % group matrices (results/matlab_*_plv_group.csv) and Eq. (1) test
test_matlab   % tests, including agreement with the Python group matrix
```

| Output | Content | Data |
|---|---|---|
| Figure 2 b, d, f | group PLV matrices at rest and during the two tasks | `resting_state_spheres`, `letter_fluency_spheres`, `verb_generation_spheres` |
| Supplementary Fig. 2 | single-participant PLV matrices | same as Fig. 2 |
| Supplementary Fig. 7 | resting state, Glasser parcels | `resting_state_glasser` |
| Supplementary Fig. 8 | coefficient of variation across participants | same as Fig. 2 |
| Supplementary Fig. 9 | Jensen–Shannon divergence between participants | same as Fig. 2 |
| Supplementary Fig. 10 | resting state, Pearson correlation | `resting_state_spheres` |
| Supplementary Fig. 11 | resting state, HRF-deconvolved, spheres | `resting_state_spheres_hrf_deconvolved` |
| Supplementary Fig. 12 | resting state, HRF-deconvolved, Glasser parcels | `resting_state_glasser_hrf_deconvolved` |
| Supplementary Fig. 13 | band-pass filtered task data | `letter_fluency_spheres_bandpass`, `verb_generation_spheres_bandpass` |
| `results/cluster_values.csv` | Cluster PLVs and Eq. (1) test for every dataset | all |
| `results/structure_summary.txt` | orbits and fibers of the baseline; minimum deletions, optimal solutions and selected networks (Fig. 4) | `baseline_connectome`, `rosen_halgren_submatrix` |
| Figure 4, Supplementary Figs. 4-6 | network drawings, fibers, orbits, input trees | `baseline_connectome`, `resting_state_network_fig4b`, `task_network_fig4c` |
| Figure 5, Tables 1-3 | whole-brain control (`whole_brain_tables.tex/.csv`) | `whole_brain_*` |
| Figure 6 | Kuramoto validation (`kuramoto_summary.txt`) | `baseline_connectome`, `resting_state_network_fig4b` |
| Figure 7 | minimum tolerance of the weighted balanced coloring | `baseline_connectome`, `rosen_halgren_submatrix` |

HRF, hemodynamic response function; MILP, mixed integer linear programming.

## Method

**Phase.** For each participant and ROI, the instantaneous phase is the angle of the analytic signal obtained with the Hilbert transform along time (`synchro.analytic_phase`, `matlab/analytic_phase.m`).

**Phase locking value.** For ROIs *u* and *v*, r<sub>uv</sub> = |⟨exp(−j(φ<sub>u</sub> − φ<sub>v</sub>))⟩<sub>t</sub>|. The synchronization used in the paper is the surrogate-corrected PLV (Eq. 2 of the Methods):
σ<sub>uv</sub> = (r<sub>uv</sub> − γ<sub>uv</sub>)/(1 − γ<sub>uv</sub>) if r<sub>uv</sub> ≥ γ<sub>uv</sub>, and 0 otherwise, where γ<sub>uv</sub> is the mean PLV between ROI *u* and 5,000 copies of ROI *v* with samples shuffled in time (`synchro.surrogate_corrected_plv`). Group matrices are the average over participants.

**Cluster synchronization (Eq. 1).** The cluster synchronization (CS) partitions reported in the paper are listed in `python/synchro/config.py`. `check_partition` takes the cliques in decreasing order of Cluster PLV (mean PLV inside the clique) and verifies that each clique's Cluster PLV is not smaller than any link from its ROIs to ROIs not assigned to it or to a more strongly synchronized clique. `percolation_components` returns the percolation sequence (links added in decreasing order of weight) from which candidate cliques are read.

**Fibers and orbits.** The fiber partition (minimal balanced coloring) is computed by color refinement on the input sets of the nodes; orbits come from an exact enumeration of the automorphisms (VF2 algorithm, networkx) (`synchro/structure.py`).

**Deletion-only inference.** `min_deletion` finds the minimum number of directed edges to remove from the baseline connectome so that the target CS partition is a balanced coloring, keeping at least one input per node (MILP solved with HiGHS through `scipy.optimize.milp`). Every solution is checked independently (`certify`): it is a subgraph of the baseline, every node has an input, the partition is balanced and equals the fiber partition of the subgraph. `all_min_deletions` enumerates all optimal solutions (16 for the resting state, 32 for the tasks). `select_by_percolation` selects among them the one whose removed edges are weakest in the percolation order of the tract weights (removed weights compared from the strongest downwards); the selection is unique and gives the networks of Fig. 4b,c (`data/resting_state_network_fig4b.json`, `data/task_network_fig4c.json`).

**Kuramoto validation.** Directed Kuramoto oscillators on the resting-state network of Fig. 4b, with equal natural frequencies within fibers and perturbed initial phases; coupling K = 0.3 (option --K), 500 random draws, runs of 3,000 time units (`synchro/kuramoto.py`).

**Weighted tolerance.** Tract weights are 10^Fpt from Rosen and Halgren (2021), averaged over the parcels of each ROI. The minimum tolerance is the smallest ε (bisection) for which a deletion-only subgraph exists whose weighted inputs from each class differ by at most ε for every pair of nodes of the same class (`synchro/weighted.py`).

**Whole-brain control.** Candidate shared signals (global cortical signal, A1, PSL, thalamus, medial geniculate nucleus, inferior colliculus) are removed by least squares from the ROI series; the surrogate-corrected PLV and the Eq. (1) test of the reported partition are recomputed (`synchro/wholebrain.py`).

## Reproducibility

- The surrogates are random; with the default seed (`SEED = 1` in `config.py`) the results are deterministic. Changing the seed changes Cluster PLVs by at most about 0.001.
- The Python and MATLAB implementations use different random generators; their group matrices agree within 0.0005.
- `python/tests/test_plv.py` includes checks that the PLV does not depend on the order of the ROI columns, on a constant offset of the signals, or on a constant phase lag between them.

## Not included

Fig. 1 combines brain renderings and fMRI images with schematic drawings, and Supplementary Figs. 1, 3 and 4 were drawn by hand. All figures follow the style in `python/synchro/figstyle.py` (Nature Portfolio artwork guidelines). The MATLAB implementation covers the synchronization analysis only.

## Use of AI tools

The code was optimized with the assistance of the large language model Claude Opus 5.5 (Anthropic). The authors reviewed and tested all code (see `python/tests/`) and take full responsibility for it.

## License and citation

- Code (`python/`, `matlab/`, `assets/` and all files outside `data/`): MIT license, see [`LICENSE`](LICENSE).
- Data (`data/`): Creative Commons Attribution 4.0 International (CC BY 4.0), see [`data/LICENSE`](data/LICENSE). `data/rosen_halgren_submatrix.json` is a subset of the connectome of Rosen and Halgren (2021), itself published under CC BY 4.0; cite that paper when using it.

If you use this code or these data, please cite the paper and the archived version of this repository, Zenodo, https://doi.org/10.5281/zenodo.24094720 (GitHub also shows the citation under "Cite this repository"):

Gili, T., Avila, B., Pasquini, L., Holodny, A., Phillips, D., Boldi, P., Gabrielli, A., Caldarelli, G., Zimmer, M. & Makse, H. A. Fibration symmetry-breaking supports functional transitions in a brain network engaged in language. *Nature Communications* (2026).
