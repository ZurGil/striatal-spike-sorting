"""
Independent second-opinion sort using Mountainsort5 (a completely different algorithm
family than Kilosort4 -- density-based/isosplit clustering vs. template-matching), via
SpikeInterface. Run on a time slice (not the full 3hr/250GB session) to keep this
tractable -- 20 minutes gives ~1200 spikes for a 1Hz unit, ~12000 for a 10Hz unit,
enough for real clustering, while finishing in a reasonable time.
"""
import json
import numpy as np
import spikeinterface.full as si
import mountainsort5 as ms5

PROBE_DAT = r"D:\Gil\Shamir\20260901_085606.rec\20260901_085606.kilosort\20260901_085606.probe1.dat"
CHANMAP_JSON = r"D:\Gil\Shamir\20260901_085606.rec\20260901_085606.kilosort\20260901_085606_custom_neuropixels_map.json"
OUT_DIR = r"D:\Gil\spike_sorting_agent\outputs\mountainsort5_results"
FS = 30000.0
N_CHAN = 384
SLICE_MINUTES = 20

print("Loading recording...")
rec = si.read_binary(file_paths=[PROBE_DAT], sampling_frequency=FS, dtype="int16", num_channels=N_CHAN)

with open(CHANMAP_JSON) as f:
    cm = json.load(f)
locations = np.array(list(zip(cm["xc"], cm["yc"])), dtype=float)
rec.set_channel_locations(locations)

n_frames_slice = int(SLICE_MINUTES * 60 * FS)
rec_slice = rec.frame_slice(start_frame=0, end_frame=min(n_frames_slice, rec.get_num_frames()))
print(f"Using first {SLICE_MINUTES} min = {rec_slice.get_num_frames()} frames")

print("Preprocessing (bandpass 300-6000Hz + common median reference)...")
rec_f = si.bandpass_filter(rec_slice, freq_min=300, freq_max=6000)
rec_cmr = si.common_reference(rec_f, reference="global", operator="median")

print("Running Mountainsort5 (scheme2)...")
params = ms5.Scheme2SortingParameters(
    phase1_detect_channel_radius=150,
    detect_channel_radius=50,
    detect_threshold=5.5,
    phase1_detect_threshold=5.5,
)
sorting = ms5.sorting_scheme2(recording=rec_cmr, sorting_parameters=params)

print(f"\nFound {len(sorting.unit_ids)} units")
sorting.save(folder=OUT_DIR + "_sorting", overwrite=True)

# quick per-unit summary: spike count, firing rate over the slice
rows = []
for uid in sorting.unit_ids:
    st = sorting.get_unit_spike_train(uid)
    n = len(st)
    rate = n / (rec_slice.get_num_frames() / FS)
    rows.append(dict(ms5_unit_id=int(uid), n_spikes=int(n), rate_hz=float(rate)))

import pandas as pd
df = pd.DataFrame(rows).sort_values("n_spikes", ascending=False)
df.to_csv(r"D:\Gil\spike_sorting_agent\outputs\mountainsort5_units.csv", index=False)
print(df.to_string(index=False))
print(f"\nTotal spikes: {df['n_spikes'].sum()}")
