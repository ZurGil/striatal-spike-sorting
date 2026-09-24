import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from striatal_agent.session import discover_session
from striatal_agent.acg_tools import compute_acg_hist
from striatal_agent import output_writer as ow

SESSION_ROOT = r"D:\Gil\Shamir\20260901_085606.rec"
UNITS = [71, 200, 202, 223, 256, 302, 315, 380]
SCREEN_JSON = r"D:\Gil\Shamir\20260901_085606.rec\20260901_085606.kilosort\kilosort4\agent_processed\structural_screen.json"
OUT_JSON = r"D:\Gil\spike_sorting_agent\outputs\random_pile_set.json"

paths = discover_session(SESSION_ROOT)
ks_orig = ow.build_kilosort_original(paths)
spike_times_all = ks_orig["spike_times"]
spike_clusters_all = ks_orig["spike_clusters"]
templates = ks_orig["templates"]

with open(SCREEN_JSON) as f:
    screen = {d["unit_id"]: d for d in json.load(f)}

import pandas as pd
ks_dir = paths.ks4_output_dir
contam = pd.read_csv(os.path.join(ks_dir, "cluster_ContamPct.tsv"), sep="\t").set_index("cluster_id")["ContamPct"].to_dict()
amp_tsv = pd.read_csv(os.path.join(ks_dir, "cluster_Amplitude.tsv"), sep="\t").set_index("cluster_id")["Amplitude"].to_dict()
kslabel = pd.read_csv(os.path.join(ks_dir, "cluster_KSLabel.tsv"), sep="\t").set_index("cluster_id")["KSLabel"].to_dict()

fs = 30000.0
results = []
for uid in UNITS:
    st = spike_times_all[spike_clusters_all == uid]
    templ_full = templates[uid]
    ptp = templ_full.max(axis=0) - templ_full.min(axis=0)
    order = np.argsort(-ptp)
    top4 = order[:4]
    peak_chan = int(order[0])
    ks_templ = templ_full[:, top4]
    peak_ch_templ = templ_full[:, peak_chan]
    peak_idx, trough_idx = int(np.argmax(peak_ch_templ)), int(np.argmin(peak_ch_templ))
    width_ms = abs(trough_idx - peak_idx) / fs * 1000
    amplitude = float(peak_ch_templ.max() - peak_ch_templ.min())
    trough_first = trough_idx < peak_idx

    hist, edges, total_pairs = compute_acg_hist(st, fs, max_lag_ms=30.0, bin_ms=0.5)

    s = screen[uid]
    results.append(dict(
        unit_id=uid, n_spikes=len(st), mean_rate_hz=s["mean_rate_hz"],
        ks_contam_pct=contam.get(uid), ks_amplitude=amp_tsv.get(uid), ks_label=kslabel.get(uid),
        structural_score=s["structural_score"], components=s["components"],
        likely_shared_artifact=s["likely_shared_artifact"], cross_unit_acg_info=s["cross_unit_acg_info"],
        footprint_concentration_ratio=s["footprint_concentration_ratio"],
        peak_channel=peak_chan, waveform_channels=top4.tolist(),
        acg_bin_centers_ms=((edges[:-1] + edges[1:]) / 2).tolist(), acg_counts=hist.tolist(),
        acg_violation_ratio=s["components"]["ccg"],
        ks_template=ks_templ.tolist(), ks_peak_trough_width_ms=width_ms,
        ks_template_amplitude=amplitude, ks_trough_before_peak=trough_first,
    ))
    print(f"unit {uid}: score={s['structural_score']:.2f}, width={width_ms:.2f}ms, amp={amplitude:.2f}, "
          f"best_match={s['cross_unit_acg_info']['best_match_unit']} r={s['cross_unit_acg_info']['correlation']:.2f} "
          f"dist={s['cross_unit_acg_info']['distance_um']:.0f}um")

with open(OUT_JSON, "w") as f:
    json.dump(results, f, indent=2)
print("wrote", OUT_JSON)
