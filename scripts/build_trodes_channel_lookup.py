import re, json
import numpy as np
import pandas as pd

TRODESCONF = r"D:\Gil\Shamir\20260901_084658_shamir.trodesconf"
CHANMAP_JSON = r"D:\Gil\Shamir\20260901_085606.rec\20260901_085606.kilosort\20260901_085606_custom_neuropixels_map.json"
TEMPLATES = r"D:\Gil\Shamir\20260901_085606.rec\20260901_085606.kilosort\kilosort4\kilosort_original\templates.npy"
SPIKE_CLUSTERS = r"D:\Gil\Shamir\20260901_085606.rec\20260901_085606.kilosort\kilosort4\kilosort_original\spike_clusters.npy"
OUT_CSV = r"D:\Gil\spike_sorting_agent\outputs\unit_trodes_channel_lookup.csv"

# parse every SpikeNTrode + its SpikeChannel from the Trodes config
with open(TRODESCONF, encoding="utf-8", errors="ignore") as f:
    conf = f.read()

ntrode_blocks = re.findall(r'<SpikeNTrode[^>]*id="(\d+)"[^>]*>\s*<SpikeChannel[^>]*coord_ap="(-?\d+)"[^>]*coord_dv="(-?\d+)"[^>]*coord_ml="(-?\d+)"[^>]*hwChan="(\d+)"', conf)
print(f"Parsed {len(ntrode_blocks)} SpikeNTrode/SpikeChannel entries from Trodes config")

coord_to_trodes = {}
for trodes_id, ap, dv, ml, hwchan in ntrode_blocks:
    coord_to_trodes[(int(ml), int(dv))] = dict(trodes_id=int(trodes_id), hwChan=int(hwchan), coord_ap=int(ap))

with open(CHANMAP_JSON) as f:
    cm = json.load(f)
xc, yc = np.array(cm["xc"]), np.array(cm["yc"])

chan_to_trodes = {}
missing = []
for ch in range(len(xc)):
    key = (int(xc[ch]), int(yc[ch]))
    if key in coord_to_trodes:
        chan_to_trodes[ch] = coord_to_trodes[key]
    else:
        missing.append(ch)
print(f"Matched {len(chan_to_trodes)}/{len(xc)} channels to a Trodes SpikeNTrode id ({len(missing)} unmatched)")

templates = np.load(TEMPLATES)
spike_clusters = np.load(SPIKE_CLUSTERS).ravel()
uniq, counts = np.unique(spike_clusters, return_counts=True)
count_map = dict(zip(uniq.tolist(), counts.tolist()))

rows = []
for u in range(templates.shape[0]):
    ptp = templates[u].max(axis=0) - templates[u].min(axis=0)
    peak_chan = int(np.argmax(ptp))
    info = chan_to_trodes.get(peak_chan)
    rows.append(dict(
        unit_id=u, peak_channel_index=peak_chan,
        trodes_ntrode_id=info["trodes_id"] if info else None,
        trodes_hwChan=info["hwChan"] if info else None,
        peak_amplitude=float(ptp.max()),
        n_spikes=count_map.get(u, 0),
    ))

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)
print(f"Wrote {len(df)} units to {OUT_CSV}")
print()
print("Units of interest from this conversation:")
units_of_interest = [285,295,139,187,156,151,272,183,142,152,153,257,264,282,293,229,30,32,69]
print(df[df.unit_id.isin(units_of_interest)].sort_values('unit_id').to_string(index=False))
