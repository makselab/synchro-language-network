# ROI time series

BOLD time series of the 11 regions of interest (ROIs) of the dorsal language stream for 20 participants, used in the synchronization analyses of the paper.

## Files

| File | Condition | ROIs | Time points |
|---|---|---|---|
| `resting_state_spheres.json` | resting state | 6-mm spheres | 155 |
| `resting_state_glasser.json` | resting state | Glasser parcels | 155 |
| `letter_fluency_spheres.json` | letter fluency task | 6-mm spheres | 160 |
| `verb_generation_spheres.json` | verb generation task | 6-mm spheres | 160 |
| `letter_fluency_spheres_bandpass.json` | letter fluency task, band-pass filtered 0.01–0.1 Hz | 6-mm spheres | 160 |
| `verb_generation_spheres_bandpass.json` | verb generation task, band-pass filtered 0.01–0.1 Hz | 6-mm spheres | 160 |
| `resting_state_spheres_hrf_deconvolved.json` | resting state, after deconvolution of the hemodynamic response function | 6-mm spheres | 155 |
| `resting_state_glasser_hrf_deconvolved.json` | resting state, after deconvolution of the hemodynamic response function | Glasser parcels | 155 |

Further files:

| File | Content |
|---|---|
| `whole_brain_resting_state.json`, `whole_brain_letter_fluency.json`, `whole_brain_verb_generation.json` | whole-brain pipeline: 11 ROI series averaged within Glasser parcels (`data`) and six candidate shared signals (`regressor_data`: global cortical signal, A1, PSL, thalamus, MGN, IC), participants × time points × signals |
| `baseline_connectome.json` | the 23 tracts of the baseline connectome (Fig. 1a, Fig. 4a) |
| `resting_state_network_fig4b.json`, `task_network_fig4c.json` | directed edges of the resting-state and task networks of Fig. 4b,c |
| `rosen_halgren_submatrix.json` | log10 fractional streamline probability between the parcels of the 11 ROIs, from Rosen and Halgren (2021), and the parcels assigned to each ROI |

A1, primary auditory cortex; PSL, perisylvian language area; MGN, medial geniculate nucleus; IC, inferior colliculus.

## Structure of the ROI time-series files

- `data`: array of size participants × time points × ROIs (repetition time 2.5 s). Each run had 160 volumes; in the resting state the first five volumes were discarded to allow the magnetization to reach a steady state (T1 equilibration), leaving 155 time points.
- `rois`: ROI labels in the order of the last dimension (SMG_L, SMG_R, PreMA_L, PreMA_R, SMA, WA_L, WA_R, AG_L, AG_R, BA_L, BA_R), with region name, hemisphere and, for the Glasser files, the corresponding parcel.
- `participants`: anonymous participant codes P01–P20.
- `preprocessing`, `roi_definition`, `TR_seconds`: description of the data.

Abbreviations: SMG, supramarginal gyrus; PreMA, premotor area; SMA, supplementary motor area; WA, Wernicke's area; AG, angular gyrus; BA, Broca's area; L and R, left and right hemisphere.

## Reading the data

Python:

```python
import json, numpy as np
d = json.load(open("resting_state_spheres.json"))
X = np.array(d["data"])          # 20 x 155 x 11
labels = [r["label"] for r in d["rois"]]
```

MATLAB:

```matlab
d = jsondecode(fileread('resting_state_spheres.json'));
X = d.data;                      % 20 x 155 x 11
labels = {d.rois.label};
```

## License

The data in this folder are released under the Creative Commons Attribution 4.0 International license (CC BY 4.0); see [`LICENSE`](LICENSE). `rosen_halgren_submatrix.json` is derived from Rosen, B. Q. & Halgren, E. A whole-cortex probabilistic diffusion tractography connectome. *eNeuro* 8, ENEURO.0416-20.2020 (2021), published under CC BY 4.0, and is redistributed with attribution: cite that paper when using the file.
