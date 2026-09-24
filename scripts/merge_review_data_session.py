"""
Merges cluster_points, sep_vs_noise/sep_vs_other (+vectors, n_valid_*, co_located_units),
burst_curve/burst_shape/isi_amp_corr_all/isi_amp_corr_burst, AND trodes_channel into
pipeline_review_data_<session_id>.json (in place). This is the final assembly step
before build_review_html.py -- see WORKFLOW.md 1f-pre for the full field-source map.

BUG FIXED 2026-09-16: this script originally merged everything EXCEPT
trodes_channel, even though build_trodes_channel_lookup_session.py's own output
CSV was sitting right there and correctly computed (values ~1086-1466, the real
Trodes ntrode numbering, not the 0-383 Kilosort channel index) -- the merge step
for it was just never written. Every unit in the first published
20260911_100049 review tool showed trodes_channel=null and silently fell back
to displaying the 0-383 KS channel instead, which is what the user actually
spotted ("channel number should be much larger, about 1083 to 1466"). Always
re-derive this table's completeness check (grep the built HTML/JSON for
`"trodes_channel": null` if in doubt) rather than assuming a lookup CSV having
been built means it was actually merged in.

isi_amp_scatter, n_true_bursts, n_burst_pairs are DELIBERATELY NOT added -- confirmed
by grepping build_review_html.py's JS that nothing renders them (dead fields even in
the original session's data). Do not add them back without re-checking the JS first.

sep_vs_noise / sep_vs_other scalars and the "p80" label in the UI are the 80th
percentile (linear interpolation, numpy default) of that unit's 10-chunk vector,
excluding NaN entries -- reverse-engineered and confirmed against the original
session's data (unit 0: p80([0.686,0.900,1.481,0.772,1.049,1.227,1.203,0.429,0.737,
1.019]) = 1.2080 -> matches stored sep_vs_noise=1.2078; unit 5 vs_other confirmed too).
co_located_units is exactly raw_separation's own "other_units" list, unmodified --
confirmed unit 0 (other_units=[] -> co_located_units=[]) and unit 5
(other_units=[7,8] -> co_located_units=[7,8]).

Usage: python merge_review_data_session.py <session_id>
Prerequisites (all session-parameterized scripts, run first):
  build_pipeline_review_data.py, raw_cluster_points_session.py,
  raw_separation_metric_session.py, build_burst_isi_metrics_session.py
"""
import json, os, sys
import numpy as np
import pandas as pd

SESSION_ID = sys.argv[1]
OUT = r"D:\Gil\spike_sorting_agent\outputs"
REVIEW_JSON = rf"{OUT}\pipeline_review_data_{SESSION_ID}.json"
CLUSTER_POINTS_JSON = rf"{OUT}\raw_cluster_points_full_{SESSION_ID}.json"
SEPARATION_NPY = rf"{OUT}\raw_separation_full_{SESSION_ID}.npy"
BURST_ISI_JSON = rf"{OUT}\burst_isi_metrics_{SESSION_ID}.json"
TRODES_LUT_CSV = rf"{OUT}\unit_trodes_channel_lookup_{SESSION_ID}.csv"


def p80(vals):
    valid = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not valid:
        return None, 0
    return round(float(np.percentile(valid, 80)), 4), len(valid)


def clean_vec(vals):
    return [None if (v is None or (isinstance(v, float) and np.isnan(v))) else round(float(v), 4) for v in vals]


def main():
    with open(REVIEW_JSON) as f:
        records = json.load(f)
    print(f"loaded {len(records)} unit records")

    with open(CLUSTER_POINTS_JSON) as f:
        cluster_points = json.load(f)  # keys are string unit ids

    sep_data = np.load(SEPARATION_NPY, allow_pickle=True).item()  # keys are int unit ids

    with open(BURST_ISI_JSON) as f:
        burst_data = json.load(f)  # keys are string unit ids

    trodes_lut = pd.read_csv(TRODES_LUT_CSV).set_index("unit_id")["trodes_ntrode_id"]

    n_missing_cp, n_missing_sep, n_missing_burst, n_missing_trodes = 0, 0, 0, 0
    for rec in records:
        uid = rec["unit_id"]

        if uid in trodes_lut.index and not pd.isna(trodes_lut.loc[uid]):
            rec["trodes_channel"] = int(trodes_lut.loc[uid])
        else:
            rec["trodes_channel"] = None
            n_missing_trodes += 1

        cp = cluster_points.get(str(uid))
        rec["cluster_points"] = cp
        if cp is None:
            n_missing_cp += 1

        sep = sep_data.get(uid)
        if sep is None:
            rec["co_located_units"] = []
            rec["sep_vs_noise"] = None
            rec["sep_vs_other"] = None
            rec["sep_vs_noise_vector"] = None
            rec["sep_vs_other_vector"] = None
            rec["n_valid_noise"] = 0
            rec["n_valid_other"] = 0
            n_missing_sep += 1
        else:
            rec["co_located_units"] = sep["other_units"]
            sv, n_v = p80(sep["sep_vs_noise"])
            so, n_o = p80(sep["sep_vs_others"])
            rec["sep_vs_noise"] = sv
            rec["sep_vs_other"] = so
            rec["sep_vs_noise_vector"] = clean_vec(sep["sep_vs_noise"])
            rec["sep_vs_other_vector"] = clean_vec(sep["sep_vs_others"])
            rec["n_valid_noise"] = n_v
            rec["n_valid_other"] = n_o

        b = burst_data.get(str(uid))
        if b is None:
            rec["burst_curve"] = []
            rec["burst_shape"] = "insufficient_data"
            rec["isi_amp_corr_all"] = None
            rec["isi_amp_corr_burst"] = None
            n_missing_burst += 1
        else:
            rec["burst_curve"] = b["burst_curve"]
            rec["burst_shape"] = b["burst_shape"]
            rec["isi_amp_corr_all"] = b["isi_amp_corr_all"]
            rec["isi_amp_corr_burst"] = b["isi_amp_corr_burst"]

    with open(REVIEW_JSON, "w") as f:
        json.dump(records, f)
    print(f"wrote {len(records)} merged units to {REVIEW_JSON}")
    print(f"missing: cluster_points={n_missing_cp}, separation={n_missing_sep}, burst={n_missing_burst}, trodes_channel={n_missing_trodes}")


if __name__ == "__main__":
    main()
