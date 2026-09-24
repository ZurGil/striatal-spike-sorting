import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from striatal_agent.session import discover_session
from striatal_agent.raw_io import footprint_channels, ChannelGeometry
from striatal_agent import output_writer as ow

SESSION_ROOT = r"D:\Gil\Shamir\20260901_085606.rec"
FLAGGED = [2, 64, 66, 137, 140, 142, 151, 152, 153, 156, 158, 161, 183, 257, 264, 272, 282, 293, 385]
OUT_JSON = r"D:\Gil\spike_sorting_agent\outputs\waveform_only_set.json"
fs = 30000.0

paths = discover_session(SESSION_ROOT)
ks_orig = ow.build_kilosort_original(paths)
templates = ks_orig["templates"]
geometry = ChannelGeometry.from_json(paths.channel_map_json)

results = []
for uid in FLAGGED:
    chans, peak_chan = footprint_channels(templates, uid, 10, geometry, 100.0)
    t = templates[uid]
    ptp_all = t.max(axis=0) - t.min(axis=0)
    order_all = np.argsort(-ptp_all)
    top4 = order_all[:4]
    ks_templ = t[:, top4]
    peak_ch_templ = t[:, peak_chan]

    peak_idx, trough_idx = int(np.argmax(peak_ch_templ)), int(np.argmin(peak_ch_templ))
    width_ms = abs(trough_idx - peak_idx) / fs * 1000
    amplitude = float(peak_ch_templ.max() - peak_ch_templ.min())
    trough_first = trough_idx < peak_idx

    footprint_ptp = t[:, chans].max(axis=0) - t[:, chans].min(axis=0)
    sorted_ptp = np.sort(footprint_ptp)[::-1]
    concentration = float(sorted_ptp[1] / sorted_ptp[0]) if sorted_ptp[0] > 0 else 0.0

    # crude asymmetry check: compare the "fast phase" (time from the earlier extremum to
    # the later one) against the "return phase" (time from the later extremum back to
    # near-baseline) -- real APs are asymmetric (fast phase, slower return); noise/sine
    # tends toward symmetric
    lo, hi = min(peak_idx, trough_idx), max(peak_idx, trough_idx)
    fast_phase_n = hi - lo
    baseline = peak_ch_templ[0]
    post = peak_ch_templ[hi:]
    return_n = len(post)
    thresh = baseline + 0.1 * (peak_ch_templ[hi] - baseline)
    crossing = np.where(np.sign(post - thresh) != np.sign(post[0] - thresh))[0]
    return_n = int(crossing[0]) if len(crossing) else len(post)
    asymmetry_ratio = float(return_n / max(fast_phase_n, 1))

    results.append(dict(
        unit_id=uid, peak_channel=peak_chan, waveform_channels=top4.tolist(),
        ks_template=ks_templ.tolist(), ks_peak_trough_width_ms=width_ms,
        ks_template_amplitude=amplitude, ks_trough_before_peak=trough_first,
        footprint_concentration_ratio=concentration, asymmetry_ratio=asymmetry_ratio,
    ))
    print(f"unit {uid}: width={width_ms:.2f}ms amp={amplitude:.2f} concentration={concentration:.2f} "
          f"asymmetry(return/fast)={asymmetry_ratio:.2f}")

with open(OUT_JSON, "w") as f:
    json.dump(results, f, indent=2)
print("wrote", OUT_JSON)
