"""
Who exactly did the footprint gate throw out?

The gated scan reported that unit 440's detections collapsed from 474 to 1
once the footprint gate was applied, while unit 408's barely moved (311 ->
246). Both numbers are aggregates. This script breaks them down by WHICH
neighbouring unit each detection was attributable to, before and after the
gate, so the aggregate becomes interpretable.

It also fixes a flaw in how the gated scan chose its impostor: it picked the
spatially NEAREST competing unit, which for unit 440 was unit 438 (same peak
channel, 0 um) -- and unit 438 turned out to be rejected by the R2 gate alone
(0.4% pass rate). Distance does not identify the hardest impostor. The unit
that actually contaminates the detections does. That unit is found here
empirically, from the attribution counts themselves.

Reads only the saved candidate CSVs plus the Kilosort spike list -- no raw
voltage, so this is fast.

Usage: python demo_footprint_gate_attribution_breakdown.py
"""
import os
import numpy as np
import pandas as pd

KS = r"F:\Gil\Shamir\20260916_110311.rec\20260916_110311.kilosort\kilosort4"
OUT = r"D:\Gil\spike_sorting_agent\outputs"
RADIUS_UM, JITTER_WINDOW = 60.0, 15

templates = np.load(KS + r"\templates.npy")
spike_times = np.load(KS + r"\spike_times.npy").ravel()
spike_clusters = np.load(KS + r"\spike_clusters.npy").ravel()
channel_positions = np.load(KS + r"\channel_positions.npy")
n_templ = templates.shape[0]
peak_ch_all = np.argmax(templates.max(axis=1) - templates.min(axis=1), axis=1)

order = np.argsort(spike_times)
st_sorted, sc_sorted = spike_times[order], spike_clusters[order]


def attributions(uid, samples):
    """For each sample, every unit within RADIUS_UM holding a spike within
    JITTER_WINDOW. Returns a Counter-like dict unit -> count."""
    mp = channel_positions[peak_ch_all[uid]]
    counts = {}
    lo = np.searchsorted(st_sorted, np.asarray(samples) - JITTER_WINDOW, side="left")
    hi = np.searchsorted(st_sorted, np.asarray(samples) + JITTER_WINDOW, side="right")
    for a, b in zip(lo, hi):
        for oc in set(sc_sorted[a:b].tolist()) - {uid}:
            if oc >= n_templ:
                continue
            d = np.sqrt(((channel_positions[peak_ch_all[oc]] - mp) ** 2).sum())
            if d <= RADIUS_UM:
                counts[int(oc)] = counts.get(int(oc), 0) + 1
    return counts


rows = []
for uid in (440, 408):
    path = os.path.join(OUT, f"fpgate_unit{uid}_candidates.csv")
    cdf = pd.read_csv(path)
    before = cdf.loc[cdf["pass_r2"], "sample"].to_numpy()
    after = cdf.loc[cdf["pass_both"], "sample"].to_numpy()
    a_before = attributions(uid, before)
    a_after = attributions(uid, after)

    mp = channel_positions[peak_ch_all[uid]]
    print(f"\n{'='*78}")
    print(f"unit {uid} (peak ch{peak_ch_all[uid]}): {len(before)} detections pass R2, "
          f"{len(after)} also pass footprint")
    print(f"  {'neighbour':>10}{'peak ch':>9}{'dist um':>9}"
          f"{'attributed BEFORE':>19}{'AFTER':>8}{'removed':>9}")
    allu = sorted(set(a_before) | set(a_after),
                  key=lambda u: -a_before.get(u, 0))
    for oc in allu:
        nb, na = a_before.get(oc, 0), a_after.get(oc, 0)
        d = np.sqrt(((channel_positions[peak_ch_all[oc]] - mp) ** 2).sum())
        print(f"  {oc:>10}{peak_ch_all[oc]:>9}{d:>9.0f}{nb:>19,}{na:>8,}{nb-na:>9,}")
        rows.append(dict(unit=uid, neighbour=oc, neighbour_peak_ch=int(peak_ch_all[oc]),
                         dist_um=round(float(d), 1), attributed_before=nb,
                         attributed_after=na, removed=nb - na))
    if allu:
        worst = max(allu, key=lambda u: a_before.get(u, 0))
        print(f"  -> the real contaminating impostor is unit {worst} "
              f"({a_before.get(worst,0)} of {len(before)} detections = "
              f"{a_before.get(worst,0)/max(len(before),1):.1%}), "
              f"NOT the nearest-by-distance unit the scan auto-picked")

pd.DataFrame(rows).to_csv(os.path.join(OUT, "fpgate_attribution_breakdown.csv"), index=False)
print(f"\nsaved fpgate_attribution_breakdown.csv")
