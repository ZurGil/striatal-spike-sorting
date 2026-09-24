"""
Runs the NEW (fixed) sync_pipeline (D:\\Gil\\sync_pipeline, post-timing-fix)
against session 20260901_085606's Phy-curated working copy, to a SEPARATE
test output folder (does not touch the existing authoritative
verdict_review\\synced\\ built by the old, buggy pipeline) -- purpose:
compare whether the ephys-timing fix actually changes anything for a
session that's already been manually verdicted and Phy-curated, before
deciding whether to trust/keep it as the new authoritative synced data.

Uses the same monkeypatch as WORKFLOW.md section 2e to point
process_session at the verdict_review/ Phy working copy instead of the
standard kilosort4/ folder.

Usage: python run_new_sync_pipeline_20260901_085606_TEST.py
"""
import sys
sys.path.insert(0, r"D:\Gil")
import dataclasses
from pathlib import Path
from unittest.mock import patch

from sync_pipeline import process_session, session_discovery

REC_FOLDER = r"D:\Gil\Shamir\20260901_085606.rec"
BPOD_FILE = r"Z:\Gil\Shamir_1\bpod\Shamir01_Dual2AFC_nat_Sep01_2026_Session1.mat"
RAT_ROOT = r"Z:\Gil\Shamir_1"
VERDICT_REVIEW_DIR = r"D:\Gil\Shamir\20260901_085606.rec\20260901_085606.kilosort\kilosort4\verdict_review"
OUTPUT_DIR = VERDICT_REVIEW_DIR + r"\synced_NEWFIX_TEST"

_orig = session_discovery.session_paths_from_rec_folder


def _patched(rec_folder, bpod_mat_file):
    paths = _orig(rec_folder, bpod_mat_file)
    return dataclasses.replace(paths, kilosort4_folder=Path(VERDICT_REVIEW_DIR))


with patch.object(session_discovery, "session_paths_from_rec_folder", side_effect=_patched):
    summary = process_session.process_session(
        rec_folder=REC_FOLDER, bpod_file=BPOD_FILE, rat_root=RAT_ROOT,
        output_dir=OUTPUT_DIR,
    )

for k, v in summary.items():
    print(f"{k}: {v}")
print("\nNEW_SYNC_TEST_DONE")
