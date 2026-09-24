"""
Builds pipeline_review_data.json for one session -- the data file the published HTML
review tool reads. See WORKFLOW.md section 1f-pre for the full field-by-field source
map and what's still missing (cluster_points, sep_vs_noise/other, co_located_units,
burst_curve, isi_amp_scatter/corr -- not computed by this script yet).

Usage: python build_pipeline_review_data.py <session_id> <rec_root>
  <rec_root> e.g. F:\Gil\Shamir\20260911_100049.rec
Run with the `kilosort` conda env. Prerequisite: corrected_classification_session.py
must have already produced outputs/corrected_classification_<session_id>_with_waveforms.json.
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from striatal_agent.acg_tools import compute_acg_hist

SESSION_ID = sys.argv[1]
REC_ROOT = sys.argv[2]
CLASSIFICATION_JSON = rf"D:\Gil\spike_sorting_agent\outputs\corrected_classification_{SESSION_ID}_with_waveforms.json"
KS_DIR = rf"{REC_ROOT}\{SESSION_ID}.kilosort\kilosort4\kilosort_original"
OUT_PATH = rf"D:\Gil\spike_sorting_agent\outputs\pipeline_review_data_{SESSION_ID}.json"

FS = 30000.0
ACG_MAX_LAG_MS = 30.0
ACG_BIN_MS = 0.5
ACG_REFRACTORY_MS = 1.5
ACG_SHOULDER_LO_MS = 5.0
ACG_SHOULDER_HI_MS = 25.0


def main():
    with open(CLASSIFICATION_JSON) as f:
        records = json.load(f)
    print(f"loaded {len(records)} unit records from {CLASSIFICATION_JSON}")

    spike_times = np.load(os.path.join(KS_DIR, "spike_times.npy")).ravel()
    spike_clusters = np.load(os.path.join(KS_DIR, "spike_clusters.npy")).ravel()

    out_records = []
    for i, rec in enumerate(records):
        uid = rec["unit_id"]
        st = spike_times[spike_clusters == uid]
        hist, edges, total_pairs = compute_acg_hist(st, FS, max_lag_ms=ACG_MAX_LAG_MS, bin_ms=ACG_BIN_MS)

        out = dict(rec)
        # waveform_top4 is literally corrected_classification's own "ks_template" --
        # same top-4-channel-by-amplitude slice of templates.npy, just renamed to
        # match the review tool's expected field name (see WORKFLOW.md 1f-pre).
        out["waveform_top4"] = out.pop("ks_template")
        out["acg_hist"] = hist.tolist()
        out["acg_bin_ms"] = ACG_BIN_MS
        out["acg_max_lag_ms"] = ACG_MAX_LAG_MS
        out["acg_refractory_ms"] = ACG_REFRACTORY_MS
        out["acg_shoulder_lo_ms"] = ACG_SHOULDER_LO_MS
        out["acg_shoulder_hi_ms"] = ACG_SHOULDER_HI_MS
        out_records.append(out)

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(records)}")

    with open(OUT_PATH, "w") as f:
        json.dump(out_records, f)
    print(f"\nwrote {len(out_records)} units to {OUT_PATH}")
    print("NOTE: cluster_points, sep_vs_noise/sep_vs_other, co_located_units, "
          "burst_curve, isi_amp_scatter/corr are NOT included yet -- see WORKFLOW.md 1f-pre.")


if __name__ == "__main__":
    main()
