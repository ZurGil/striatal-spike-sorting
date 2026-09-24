"""
Run bombcell quality metrics for one session. Requires the dedicated `D:\\conda_envs\\bombcell`
conda env (NOT `kilosort` -- bombcell isn't installed there), AND PYTHONIOENCODING=utf-8 set
at the SHELL level before launching python (bombcell prints a unicode checkmark on import;
os.environ.setdefault() inside the script is too late -- Windows console encoding is fixed
at interpreter startup). See WORKFLOW.md 1f-pre for the full story.

Usage (from Git Bash / a shell that supports `export`):
  export PYTHONIOENCODING=utf-8
  D:\\conda_envs\\bombcell\\python.exe run_bombcell_session.py <session_id> <rec_root> <gain_to_uV>

<rec_root> is the folder containing <session_id>.kilosort\\, e.g.
  F:\\Gil\\Shamir\\20260911_100049.rec

<gain_to_uV>: NOT universal -- verify per-session. Fastest way: grep the session's own raw
.rec file header for `spikeScalingToUv=` (it's embedded in the Trodes config XML at the top
of the file, uniform across all SpikeNTrode entries) rather than assuming a prior session's
value still applies (it happened to be identical -- 0.018311105685598315 -- for both
20260901_085606 and 20260911_100049, but that's a coincidence of an unchanged rig, not a
given).
"""
import os, sys
os.environ.setdefault("PYTHONIOENCODING", "utf-8")  # harmless leftover; real fix is the shell export, see docstring
import matplotlib
matplotlib.use("Agg")
import bombcell

SESSION_ID = sys.argv[1]
REC_ROOT = sys.argv[2]          # e.g. F:\Gil\Shamir\20260911_100049.rec
GAIN_TO_UV = float(sys.argv[3])

KS_DIR = rf"{REC_ROOT}\{SESSION_ID}.kilosort\kilosort4\kilosort_original"
RAW_FILE = rf"{REC_ROOT}\{SESSION_ID}.kilosort\{SESSION_ID}.probe1.dat"
SAVE_PATH = rf"D:\Gil\spike_sorting_agent\outputs\bombcell_results_{SESSION_ID}"

param = bombcell.get_default_parameters(
    kilosort_path=KS_DIR,
    raw_file=RAW_FILE,
    kilosort_version=4,
    gain_to_uV=GAIN_TO_UV,
)
param["nChannels"] = 384       # verify per-session: probe1.dat size / (384*2) should be an exact
                                # integer, matching kilosort4.log's own "N samples" count
param["nSyncChannels"] = 0
param["plotGlobal"] = False    # known hang risk in headless mode (global summary plotting step
                                # hangs indefinitely after metrics finish -- skip, we only need
                                # quality_metrics/unit_type, not the plots)

print(f"Running bombcell on {SESSION_ID}...")
quality_metrics, param, unit_type, unit_type_string, = bombcell.run_bombcell(
    KS_DIR, SAVE_PATH, param, save_figures=False, return_figures=False,
)

import numpy as np
print("\n=== BOMBCELL SUMMARY ===")
labels, counts = np.unique(unit_type_string, return_counts=True)
for l, c in zip(labels, counts):
    print(f"  {l}: {c}")
