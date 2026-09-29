"""
Ideal illustrative pair for pillar 1c, found by scanning real units:
381 and 397 -- nearly IDENTICAL single-channel waveform shape (corr=0.98),
similar location (peak channels 80um apart), but essentially UNRELATED
spatial footprints (cosine sim=0.0016). This is the case that most
clearly shows what pillar 1c adds beyond single-channel shape analysis:
these two units would be indistinguishable by waveform alone, but their
spatial footprints separate them cleanly.

Usage: python demo_footprint_ideal_pair.py
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
U1, U2 = 381, 397

templates = np.load(VR + r"\templates.npy")
channel_positions = np.load(VR + r"\channel_positions.npy")
spike_times = np.load(VR + r"\spike_times.npy").ravel()
spike_clusters = np.load(VR + r"\spike_clusters.npy").ravel()

templ_all_1, templ_all_2 = templates[U1], templates[U2]
pc1 = int(np.argmax(templ_all_1.max(axis=0) - templ_all_1.min(axis=0)))
pc2 = int(np.argmax(templ_all_2.max(axis=0) - templ_all_2.min(axis=0)))
wf1, wf2 = templ_all_1[:, pc1], templ_all_2[:, pc2]
shape_corr = np.corrcoef(wf1, wf2)[0, 1]

fp1 = unit_footprint(templates, U1, channel_positions, radius_um=60.0)
fp2 = unit_footprint(templates, U2, channel_positions, radius_um=60.0)
channels_union = np.union1d(fp1["channels"], fp2["channels"])


def reindex(fp, channels_union):
    full = np.zeros(len(channels_union))
    for i, ch in enumerate(fp["channels"]):
        idx = np.searchsorted(channels_union, ch)
        full[idx] = fp["footprint"][i]
    norm = np.linalg.norm(full)
    return full / norm if norm > 0 else full


footprint_1 = reindex(fp1, channels_union)
footprint_2 = reindex(fp2, channels_union)
fp_sim_templates = footprint_similarity(footprint_1, footprint_2)
dist_um = np.sqrt(((channel_positions[pc1] - channel_positions[pc2]) ** 2).sum())

print(f"unit {U1}: peak_ch={pc1}, unit {U2}: peak_ch={pc2}, {dist_um:.0f}um apart")
print(f"waveform shape correlation: {shape_corr:.4f}")
print(f"template footprint similarity: {fp_sim_templates:.4f}")


def get_multichannel_snippet(spike_sample, channels):
    lo_sample = int(spike_sample) - NT0MIN
    with open(DAT_PATH, "rb") as f:
        f.seek(lo_sample * N_CHAN_BIN * ITEMSIZE)
        raw = f.read(NT * N_CHAN_BIN * ITEMSIZE)
    if len(raw) != NT * N_CHAN_BIN * ITEMSIZE:
        return None
    block = np.frombuffer(raw, dtype=np.int16).reshape(NT, N_CHAN_BIN)
    return block[:, channels].astype(np.float64) * GAIN_TO_UV


st1 = np.sort(spike_times[spike_clusters == U1])
st2 = np.sort(spike_times[spike_clusters == U2])
rng = np.random.default_rng(3)
sample1 = rng.choice(st1, size=min(60, len(st1)), replace=False)
sample2 = rng.choice(st2, size=min(60, len(st2)), replace=False)

sims1_vs_fp1, sims2_vs_fp1 = [], []
for s in sample1:
    snip = get_multichannel_snippet(s, channels_union)
    if snip is not None:
        sims1_vs_fp1.append(footprint_similarity(spatial_energy_vector(snip), footprint_1))
for s in sample2:
    snip = get_multichannel_snippet(s, channels_union)
    if snip is not None:
        sims2_vs_fp1.append(footprint_similarity(spatial_energy_vector(snip), footprint_1))

print(f"real unit-{U1} spikes vs unit-{U1}'s footprint: mean={np.mean(sims1_vs_fp1):.3f}")
print(f"real unit-{U2} spikes vs unit-{U1}'s footprint: mean={np.mean(sims2_vs_fp1):.3f}")

# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(19, 5.8))
t_ms = (np.arange(NT) - NT0MIN) / FS * 1000

ax = axes[0]
ax.plot(t_ms, wf1, color="#1f77b4", lw=2, label=f"unit {U1} (ch{pc1})")
ax.plot(t_ms, wf2, color="#d62728", lw=2, ls="--", label=f"unit {U2} (ch{pc2})")
ax.axhline(0, color="grey", lw=0.4)
ax.set_xlabel("time (ms)")
ax.set_ylabel("template amplitude (a.u.)")
ax.set_title(f"WAVEFORM SHAPE (single channel each)\ncorrelation = {shape_corr:.3f} -- nearly IDENTICAL\n"
             f"(single-channel view alone can't tell these apart)", fontsize=10.5)
ax.legend(fontsize=9)

ax = axes[1]
xpos = channel_positions[channels_union]
ax.scatter(xpos[:, 0], xpos[:, 1], s=footprint_1 * 1200, color="#1f77b4", alpha=0.5, label=f"unit {U1} footprint")
ax.scatter(xpos[:, 0], xpos[:, 1], s=footprint_2 * 1200, color="#d62728", alpha=0.5, marker="^", label=f"unit {U2} footprint")
ax.scatter(*channel_positions[pc1], marker="*", s=200, color="#1f77b4", edgecolor="k", zorder=5)
ax.scatter(*channel_positions[pc2], marker="*", s=200, color="#d62728", edgecolor="k", zorder=5)
ax.set_xlabel("x (um)")
ax.set_ylabel("y (um)")
ax.set_title(f"SPATIAL FOOTPRINT (dot size = amplitude)\ntemplate-vs-template similarity = {fp_sim_templates:.4f}\n"
             f"peak channels {dist_um:.0f}um apart -- SEPARATED energy", fontsize=10.5)
ax.legend(fontsize=9)

ax = axes[2]
ax.hist(sims1_vs_fp1, bins=15, alpha=0.65, color="#1f77b4", label=f"real unit-{U1} spikes\n(n={len(sims1_vs_fp1)})")
ax.hist(sims2_vs_fp1, bins=15, alpha=0.65, color="#d62728", label=f"real unit-{U2} spikes\n(n={len(sims2_vs_fp1)})")
ax.set_xlabel(f"cosine similarity to unit {U1}'s expected footprint")
ax.set_ylabel("count")
ax.set_title(f"REAL spikes scored against unit {U1}'s\nfootprint -- clean separation despite\nnearly identical waveform shape", fontsize=10.5)
ax.legend(fontsize=8.5)

plt.suptitle(f"Pillar 1c, ideal case: units {U1} & {U2} -- same shape, different location, separated footprint energy",
             fontsize=14, fontweight="bold")
plt.tight_layout()
out_path = os.path.join(OUT, "wavelet_35_footprint_ideal_pair.png")
plt.savefig(out_path, dpi=115, bbox_inches="tight")
print("\nsaved", out_path)
