"""Datasets, CS partitions reported in the paper, and display orders of the figures."""

REST_PARTITION = [
    ["SMG_L", "SMG_R"],
    ["PreMA_L", "PreMA_R", "SMA", "WA_L", "WA_R"],
    ["AG_L", "AG_R"],
    ["BA_L", "BA_R"],
]
TASK_PARTITION = [
    ["BA_L", "SMA"],
    ["PreMA_L", "PreMA_R"],
    ["AG_L", "AG_R"],
    ["BA_R", "WA_L", "WA_R"],
    ["SMG_L", "SMG_R"],
]

# Display order of the matrices (clusters in decreasing Cluster PLV, as in Fig. 2)
REST_ORDER = ["SMG_L", "SMG_R", "PreMA_L", "PreMA_R", "SMA", "WA_L", "WA_R", "AG_L", "AG_R", "BA_L", "BA_R"]
LETTER_ORDER = ["BA_L", "SMA", "AG_L", "AG_R", "PreMA_L", "PreMA_R", "BA_R", "WA_L", "WA_R", "SMG_L", "SMG_R"]
VERB_ORDER = ["BA_L", "SMA", "PreMA_L", "PreMA_R", "AG_L", "AG_R", "BA_R", "WA_L", "WA_R", "SMG_L", "SMG_R"]
REST_SIZES = [2, 5, 2, 2]
TASK_SIZES = [2, 2, 2, 3, 2]

# Glasser parcel corresponding to each ROI
GLASSER = {"SMG": "PF", "PreMA": "6a", "SMA": "SCEF", "WA": "STSdp", "AG": "PFm", "BA": "44"}

DATASETS = {
    # name: (partition, display order, block sizes, ROI naming)
    "resting_state_spheres": (REST_PARTITION, REST_ORDER, REST_SIZES, "spheres"),
    "letter_fluency_spheres": (TASK_PARTITION, LETTER_ORDER, TASK_SIZES, "spheres"),
    "verb_generation_spheres": (TASK_PARTITION, VERB_ORDER, TASK_SIZES, "spheres"),
    "resting_state_glasser": (REST_PARTITION, REST_ORDER, REST_SIZES, "glasser"),
    "resting_state_spheres_hrf_deconvolved": (REST_PARTITION, REST_ORDER, REST_SIZES, "spheres"),
    "resting_state_glasser_hrf_deconvolved": (REST_PARTITION, REST_ORDER, REST_SIZES, "glasser"),
    "letter_fluency_spheres_bandpass": (TASK_PARTITION, LETTER_ORDER, TASK_SIZES, "spheres"),
    "verb_generation_spheres_bandpass": (TASK_PARTITION, VERB_ORDER, TASK_SIZES, "spheres"),
}

N_SURROGATES = 5000

# PLV tolerance of Eq. (1) in the control analyses (different parcellation,
# preprocessing, filtering or regression of shared signals): differences
# strictly smaller than this, well below the standard error of the group-average PLV
# across participants (about 0.05), are not considered meaningful. The main
# analysis uses no tolerance.
EQ1_TOLERANCE = 0.01
MAIN_DATASETS = ("resting_state_spheres", "letter_fluency_spheres", "verb_generation_spheres")


def eq1_tolerance(name):
    """PLV tolerance of Eq. (1) for a dataset: 0 for the main analysis."""
    return 0.0 if name in MAIN_DATASETS else EQ1_TOLERANCE

SEED = 1


def display_label(label, naming="spheres"):
    """'SMG_L' -> 'SMG L' (spheres) or 'PF L' (Glasser parcels)."""
    base, _, side = label.partition("_")
    if naming == "glasser":
        base = GLASSER[base]
    return f"{base} {side}".strip()
