"""
Does unit 42's waveform look like noise because of a real data issue,
or because Phy's default rendering (opaque, solid, stacked lines) hides
a jittery-but-real spike shape that only shows up with alpha blending?
Re-render the SAME 60 spikes both ways, side by side.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VR = r"D:\Gil\Shamir\20260901_085606.rec\20260901_085606.kilosort\kilosort4\verdict_review"
DAT_PATH = r"D:\Gil\Shamir\20260901_085606.rec\20260901_085606.kilosort\20260901_085606.probe1.dat"
OUT = r"D:\Gil\spike_sorting_agent\outputs"
GAIN_TO_UV = 0.018311105685598315
N_CHAN_BIN = 384
FS = 30000.0
NT, NT0MIN = 61, 20
UID = 42

info = pd.read_csv(VR + r"\cluster_info.tsv", sep="\t").set_index("cluster_id")
chan = int(info.loc[UID, "ch"])
spike_times_samples = np.load(VR + r"\spike_times.npy").ravel()
spike_clusters = np.load(VR + r"\spike_clusters.npy").ravel()
st_samples = spike_times_samples[spike_clusters == UID]

rng = np.random.default_rng(0)
sample = rng.choice(st_samples, size=60, replace=False)
wfs = []
itemsize = 2
with open(DAT_PATH, "rb") as f:
    for s in sample:
        lo = int(s) - NT0MIN
        f.seek(lo * N_CHAN_BIN * itemsize)
        raw = f.read(NT * N_CHAN_BIN * itemsize)
        if len(raw) != NT * N_CHAN_BIN * itemsize:
            continue
        block = np.frombuffer(raw, dtype=np.int16).reshape(NT, N_CHAN_BIN)
        wfs.append(block[:, chan].astype(np.float64))
wfs = np.array(wfs) * GAIN_TO_UV
t_ms = (np.arange(NT) - NT0MIN) / FS * 1000

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

ax = axes[0]
for wf in wfs:
    ax.plot(t_ms, wf, color="#3b7ab3", alpha=1.0, lw=1.0)
ax.set_title("Phy-like: opaque solid lines, no alpha")
ax.set_xlabel("ms"); ax.set_ylabel("uV")

ax = axes[1]
for wf in wfs:
    ax.plot(t_ms, wf, color="#3b7ab3", alpha=0.25, lw=0.7)
ax.plot(t_ms, wfs.mean(0), color="black", lw=1.8)
ax.set_title("Alpha-blended (my earlier plot)")
ax.set_xlabel("ms")

ax = axes[2]
# widened y-limits to a "typical" fixed probe-wide scale, e.g. +/-150uV,
# which is what a display auto-scaled to a much bigger-amplitude cluster
# elsewhere on the shank could look like
for wf in wfs:
    ax.plot(t_ms, wf, color="#3b7ab3", alpha=1.0, lw=1.0)
ax.set_ylim(-150, 150)
ax.set_title("Opaque + compressed to +/-150uV scale")
ax.set_xlabel("ms")

plt.suptitle(f"unit 42 (ch{chan}) -- same 60 raw spikes, three render styles")
plt.tight_layout()
out_path = os.path.join(OUT, "unit42_rendering_comparison_20260901_085606.png")
plt.savefig(out_path, dpi=115, bbox_inches="tight")
print("saved", out_path)
