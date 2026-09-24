import os
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
import matplotlib
matplotlib.use("Agg")  # headless background process -- force non-interactive backend defensively
import bombcell

KS_DIR = r"D:\Gil\Shamir\20260901_085606.rec\20260901_085606.kilosort\kilosort4\kilosort_original"
RAW_FILE = r"D:\Gil\Shamir\20260901_085606.rec\20260901_085606.kilosort\20260901_085606.probe1.dat"
SAVE_PATH = r"D:\Gil\spike_sorting_agent\outputs\bombcell_results_gaincorrected"

param = bombcell.get_default_parameters(
    kilosort_path=KS_DIR,
    raw_file=RAW_FILE,
    kilosort_version=4,
    gain_to_uV=0.018311105685598315,  # true system gain, found in 20260901_084658_shamir.trodesconf
                                       # (spikeScalingToUv, uniform across every SpikeNTrode entry -- see conversation).
                                       # The first run used 1.0 (raw passthrough), which made bombcell's
                                       # amplitude/SNR-based MUA thresholds (minAmplitude=40uV, minSNR=5) nearly
                                       # meaningless since raw ADC counts are already in the hundreds-thousands range.
)
param["nChannels"] = 384       # this session's probe1.dat has no sync channel (verified in step 1: 384*2 bytes divides the file size exactly)
param["nSyncChannels"] = 0
param["plotGlobal"] = False    # the metrics computation finished (401/401) on the first attempt but the
                                # process then hung indefinitely (flat CPU time) somewhere past that point --
                                # almost certainly the global summary plotting step (matplotlib GUI backend
                                # in a headless background process). We only need quality_metrics/unit_type,
                                # not the plots, so skip plotting entirely rather than debug the hang.

print("Running bombcell...")
quality_metrics, param, unit_type, unit_type_string, = bombcell.run_bombcell(
    KS_DIR, SAVE_PATH, param, save_figures=False, return_figures=False,
)

import numpy as np
print("\n=== BOMBCELL SUMMARY ===")
labels, counts = np.unique(unit_type_string, return_counts=True)
for l, c in zip(labels, counts):
    print(f"  {l}: {c}")
