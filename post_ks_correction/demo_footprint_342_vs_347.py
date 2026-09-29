"""
First real validation of pillar 1c: do units 342 and 347 (SAME peak
channel, established many turns ago) actually have distinguishable
spatial footprints across the surrounding channels? And does that let us
tell a real 342 spike apart from a real 347 spike using ONLY spatial
information -- something neither nuisance_model.py nor
wavelet_features.py can do at all (both are single-channel).

Usage: python demo_footprint_342_vs_347.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from spatial_footprint import unit_footprint, spatial_energy_vector, footprint_similarity

VR = r"D:\Gil\Shamir\20260901_085606.rec\20260901_085606.kilosort\kilosort4\verdict_review"
DAT_PATH = r"D:\Gil\Shamir\20260901_085606.rec\20260901_085606.kilosort\20260901_085606.probe1.dat"
OUT = r"D:\Gil\spike_sorting_agent\outputs"
GAIN_TO_UV = 0.018311105685598315
N_CHAN_BIN = 384
FS = 30000.0
NT0MIN = 20
NT = 61
ITEMSIZE = 2

templates = np.load(VR + r"\templates.npy")
channel_positions = np.load(VR + r"\channel_positions.npy")
spike_times = np.load(VR + r"\spike_times.npy").ravel()
spike_clusters = np.load(VR + r"\spike_clusters.npy").ravel()

fp_342 = unit_footprint(templates, 342, channel_positions, radius_um=60.0)
fp_347 = unit_footprint(templates, 347, channel_positions, radius_um=60.0)
print(f"unit 342: peak_channel={fp_342['peak_channel']}, {len(fp_342['channels'])} channels in footprint")
print(f"unit 347: peak_channel={fp_347['peak_channel']}, {len(fp_347['channels'])} channels in footprint")

# use the UNION of both units' channel sets so both footprints and both
# observed spikes are compared on the exact same axes
channels_union = np.union1d(fp_342["channels"], fp_347["channels"])
print(f"union of channels: {len(channels_union)}")


def reindex_footprint(fp, channels_union):
    full = np.zeros(len(channels_union))
    for i, ch in enumerate(fp["channels"]):
        idx = np.searchsorted(channels_union, ch)
        full[idx] = fp["footprint"][i]
    norm = np.linalg.norm(full)
    return full / norm if norm > 0 else full


footprint_342 = reindex_footprint(fp_342, channels_union)
footprint_347 = reindex_footprint(fp_347, channels_union)

# how similar are the two units' TEMPLATE footprints to each other?
templ_vs_templ_sim = footprint_similarity(footprint_342, footprint_347)
print(f"\ntemplate-vs-template footprint similarity (342 vs 347): {templ_vs_templ_sim:.3f}")


def get_multichannel_snippet(spike_sample, channels):
    lo_sample = int(spike_sample) - NT0MIN
    with open(DAT_PATH, "rb") as f:
        f.seek(lo_sample * N_CHAN_BIN * ITEMSIZE)
        raw = f.read(NT * N_CHAN_BIN * ITEMSIZE)
    if len(raw) != NT * N_CHAN_BIN * ITEMSIZE:
        return None
    block = np.frombuffer(raw, dtype=np.int16).reshape(NT, N_CHAN_BIN)
    return block[:, channels].astype(np.float64) * GAIN_TO_UV


st_342 = np.sort(spike_times[spike_clusters == 342])
st_347 = np.sort(spike_times[spike_clusters == 347])

rng = np.random.default_rng(3)
sample_342 = rng.choice(st_342, size=min(60, len(st_342)), replace=False)
sample_347 = rng.choice(st_347, size=min(60, len(st_347)), replace=False)

sims_342_vs_342template, sims_347_vs_342template = [], []
sims_342_vs_347template, sims_347_vs_347template = [], []
for s in sample_342:
    snip = get_multichannel_snippet(s, channels_union)
    if snip is None:
        continue
    obs = spatial_energy_vector(snip)
    sims_342_vs_342template.append(footprint_similarity(obs, footprint_342))
    sims_342_vs_347template.append(footprint_similarity(obs, footprint_347))
for s in sample_347:
    snip = get_multichannel_snippet(s, channels_union)
    if snip is None:
        continue
    obs = spatial_energy_vector(snip)
    sims_347_vs_342template.append(footprint_similarity(obs, footprint_342))
    sims_347_vs_347template.append(footprint_similarity(obs, footprint_347))

print(f"\nreal 342 spikes (n={len(sims_342_vs_342template)}) vs 342's own footprint: "
      f"mean={np.mean(sims_342_vs_342template):.3f}")
print(f"real 342 spikes vs 347's footprint: mean={np.mean(sims_342_vs_347template):.3f}")
print(f"real 347 spikes (n={len(sims_347_vs_347template)}) vs 347's own footprint: "
      f"mean={np.mean(sims_347_vs_347template):.3f}")
print(f"real 347 spikes vs 342's footprint: mean={np.mean(sims_347_vs_342template):.3f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

ax = axes[0]
xpos = channel_positions[channels_union]
ax.scatter(xpos[:, 0], xpos[:, 1], s=footprint_342 * 800, color="#1f77b4", alpha=0.5, label="unit 342 footprint")
ax.scatter(xpos[:, 0], xpos[:, 1], s=footprint_347 * 800, color="#d62728", alpha=0.5, marker="^", label="unit 347 footprint")
ax.set_xlabel("x (um)")
ax.set_ylabel("y (um)")
ax.set_title(f"Spatial footprints (dot size = relative amplitude)\ntemplate-vs-template similarity = {templ_vs_templ_sim:.2f}", fontsize=10.5)
ax.legend(fontsize=8.5)

ax = axes[1]
ax.hist(sims_342_vs_342template, bins=15, alpha=0.6, color="#1f77b4", label="342 spikes vs 342 footprint")
ax.hist(sims_347_vs_342template, bins=15, alpha=0.6, color="#d62728", label="347 spikes vs 342 footprint")
ax.set_xlabel("cosine similarity to unit 342's expected footprint")
ax.set_ylabel("count")
ax.set_title("Can spatial footprint alone tell 342\nand 347 spikes apart?", fontsize=10.5)
ax.legend(fontsize=8.5)

plt.suptitle("Pillar 1c first real test: units 342 vs 347 (same peak channel) via spatial footprint", fontsize=13, fontweight="bold")
plt.tight_layout()
out_path = os.path.join(OUT, "wavelet_34_footprint_342_vs_347.png")
plt.savefig(out_path, dpi=115, bbox_inches="tight")
print("\nsaved", out_path)
