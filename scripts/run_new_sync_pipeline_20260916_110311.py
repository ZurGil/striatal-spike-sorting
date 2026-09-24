"""
Rebuilds 20260916_110311's official synced/ data using the NEW (fixed)
sync_pipeline (D:\\Gil\\sync_pipeline, post-timing-fix), against the
standard (pre-Phy, no verdict_review override needed) kilosort4 folder --
this session has no Phy curation yet, so its cluster_info.tsv is still
the fabricated stand-in from build_cluster_info_from_classification_session.py.

No hard-link workaround needed here (unlike 20260901_085606): the standard
kilosort4 folder's parent IS <session>.kilosort\\, exactly where
<session>.timestamps.dat already lives.

Output goes to a NEW sibling folder (synced_NEWFIX), NOT overwriting the
existing preliminary synced/ built with the old (buggy) pipeline earlier
this session -- that old one is kept as the "OLD" side of the comparison.

Usage: python run_new_sync_pipeline_20260916_110311.py
"""
import sys
sys.path.insert(0, r"D:\Gil")
from sync_pipeline import process_session

REC_FOLDER = r"F:\Gil\Shamir\20260916_110311.rec"
BPOD_FILE = r"Z:\Gil\Shamir_1\bpod\Shamir01_Dual2AFC_nat_Sep16_2026_Session1.mat"
RAT_ROOT = r"Z:\Gil\Shamir_1"
OUTPUT_DIR = r"F:\Gil\Shamir\20260916_110311.rec\20260916_110311.kilosort\synced_NEWFIX"

summary = process_session.process_session(
    rec_folder=REC_FOLDER, bpod_file=BPOD_FILE, rat_root=RAT_ROOT, output_dir=OUTPUT_DIR,
)
for k, v in summary.items():
    print(f"{k}: {v}")
print("\nNEW_SYNC_20260916_DONE")
