import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from striatal_agent.session import discover_session
from striatal_agent.acg_tools import compute_acg_hist
from striatal_agent import output_writer as ow

SESSION_ROOT = r"D:\Gil\Shamir\20260901_085606.rec"
IN_JSON = r"D:\Gil\spike_sorting_agent\outputs\random_pile_set.json"
OUT_JSON = r"D:\Gil\spike_sorting_agent\outputs\random_pile_set_with_match.json"
fs = 30000.0

with open(IN_JSON) as f:
    units = json.load(f)

paths = discover_session(SESSION_ROOT)
ks_orig = ow.build_kilosort_original(paths)
spike_times_all = ks_orig["spike_times"]
spike_clusters_all = ks_orig["spike_clusters"]
templates = ks_orig["templates"]

def own_peak_template(uid):
    t = templates[uid]
    ptp = t.max(axis=0) - t.min(axis=0)
    peak_chan = int(np.argmax(ptp))
    order = np.argsort(-ptp)[:4]
    return t[:, peak_chan], t[:, order], peak_chan, order

for u in units:
    uid = u["unit_id"]
    match_id = u["cross_unit_acg_info"]["best_match_unit"]

    own_peak, own_top4, own_peak_chan, own_top4_idx = own_peak_template(uid)
    match_peak, match_top4, match_peak_chan, match_top4_idx = own_peak_template(match_id)

    # shape-only comparison: z-score each unit's own-peak-channel waveform, correlate
    own_z = (own_peak - own_peak.mean()) / (own_peak.std() + 1e-9)
    match_z = (match_peak - match_peak.mean()) / (match_peak.std() + 1e-9)
    shape_corr = float(np.corrcoef(own_z, match_z)[0, 1])

    st_match = spike_times_all[spike_clusters_all == match_id]
    hist, edges, _ = compute_acg_hist(st_match, fs, max_lag_ms=30.0, bin_ms=0.5)

    peak_idx, trough_idx = int(np.argmax(match_peak)), int(np.argmin(match_peak))
    width_ms = abs(trough_idx - peak_idx) / fs * 1000
    amplitude = float(match_peak.max() - match_peak.min())

    u["waveform_shape_corr_with_match"] = shape_corr
    u["match_unit_id"] = match_id
    u["match_n_spikes"] = int(len(st_match))
    u["match_ks_template"] = match_top4.tolist()
    u["match_acg_bin_centers_ms"] = ((edges[:-1] + edges[1:]) / 2).tolist()
    u["match_acg_counts"] = hist.tolist()
    u["match_ks_peak_trough_width_ms"] = width_ms
    u["match_ks_template_amplitude"] = amplitude
    u["match_ks_trough_before_peak"] = trough_idx < peak_idx
    u["match_peak_channel"] = match_peak_chan

    print(f"unit {uid} (w={u['ks_peak_trough_width_ms']:.2f}ms, amp={u['ks_template_amplitude']:.2f}) "
          f"vs match {match_id} (w={width_ms:.2f}ms, amp={amplitude:.2f}): "
          f"waveform shape corr = {shape_corr:.3f}")

with open(OUT_JSON, "w") as f:
    json.dump(units, f, indent=2)
print("wrote", OUT_JSON)
