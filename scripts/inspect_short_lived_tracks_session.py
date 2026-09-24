"""
Inspect the SHORT-LIVED tracks from independent_clustering_windowed_overlap_session.py
-- the ~half of channel 129's spikes that did NOT link into one of the 2
session-persistent tracks. Question: are these mostly noise-like
(low amplitude, inconsistent waveform shape, no refractory structure) or
real-but-unstable spikes (adequate amplitude, spike-shaped, some
refractory structure) that just failed to link across windows?

Groups short-lived tracks into lifetime tiers (since a track present in
12/22 windows is a very different animal from one present in only 1/22),
and for each tier shows: mean waveform +/- SEM, p2p/Teager scatter vs the
persistent tracks, ISI histogram (computed per-track then pooled, so
merging tracks doesn't create fake short ISIs), and a sample of individual
raw waveform traces for visual inspection.

Reuses all previously-saved npz outputs -- no raw-file re-read.

Usage: python inspect_short_lived_tracks_session.py <session_id> <chan>
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SESSION_ID = sys.argv[1]
CHAN = int(sys.argv[2])
OUT = r"D:\Gil\spike_sorting_agent\outputs"

FS = 30000.0
NT_CENTER = 20

d = np.load(rf"{OUT}\independent_clustering_{SESSION_ID}_ch{CHAN}.npz")
snips = d["snips"]
spike_times = d["spike_times"]

de = np.load(rf"{OUT}\independent_clustering_extra_{SESSION_ID}_ch{CHAN}.npz")
p2p = de["p2p"]
psi = de["psi_trough"]

do = np.load(rf"{OUT}\independent_clustering_windowed_overlap_{SESSION_ID}_ch{CHAN}.npz")
track_labels = do["track_labels"]
core_w_of = do["core_w_of"]
long_lived = set(do["long_lived_tracks"].tolist())
n_windows = int(do["n_windows"])

all_tids = np.unique(track_labels)
short_tids = [t for t in all_tids if t not in long_lived]

# lifetime per track
lifetime = {}
for t in short_tids:
    m = track_labels == t
    lifetime[t] = len(set(core_w_of[m].tolist()))

tiers = {
    "long-lived fragment (>=6/22 windows)": [t for t in short_tids if lifetime[t] >= 6],
    "medium (2-5/22 windows)": [t for t in short_tids if 2 <= lifetime[t] < 6],
    "single-window (1/22)": [t for t in short_tids if lifetime[t] == 1],
}
for name, tids in tiers.items():
    n_spk = sum((track_labels == t).sum() for t in tids)
    print(f"{name}: {len(tids)} tracks, {n_spk} spikes")

persist_tids = sorted(long_lived, key=lambda t: -(track_labels == t).sum())
print(f"\npersistent tracks for comparison: {persist_tids}, "
      f"sizes={[int((track_labels==t).sum()) for t in persist_tids]}")

t_ms = (np.arange(snips.shape[1]) - NT_CENTER) / FS * 1000
colors = {"persistent": "#0b7a75", "long-lived fragment (>=6/22 windows)": "#c1440e",
          "medium (2-5/22 windows)": "#d9a441", "single-window (1/22)": "#8a8f87"}

fig, axes = plt.subplots(2, 3, figsize=(18, 11))

# 1. mean waveform +/- SEM per group
ax = axes[0, 0]
for t in persist_tids:
    m = track_labels == t
    mw, sw = snips[m].mean(0), snips[m].std(0) / np.sqrt(m.sum())
    ax.plot(t_ms, mw, color=colors["persistent"], lw=2, label=f"persistent trk{t} (n={m.sum()})")
    ax.fill_between(t_ms, mw - sw, mw + sw, color=colors["persistent"], alpha=0.15)
for name, tids in tiers.items():
    if not tids:
        continue
    m = np.isin(track_labels, tids)
    mw, sw = snips[m].mean(0), snips[m].std(0) / np.sqrt(m.sum())
    ax.plot(t_ms, mw, color=colors[name], lw=1.5, ls="--", label=f"{name} (n={m.sum()})")
    ax.fill_between(t_ms, mw - sw, mw + sw, color=colors[name], alpha=0.12)
ax.set_xlabel("ms from threshold crossing")
ax.set_ylabel("uV")
ax.set_title("mean waveform +/- SEM, persistent vs short-lived tiers")
ax.legend(fontsize=7)

# 2. p2p vs Teager scatter
ax = axes[0, 1]
for t in persist_tids:
    m = track_labels == t
    ax.scatter(p2p[m], psi[m], s=3, color=colors["persistent"], alpha=0.4, label=f"persistent (n={m.sum()})" if t == persist_tids[0] else None)
for name, tids in tiers.items():
    if not tids:
        continue
    m = np.isin(track_labels, tids)
    ax.scatter(p2p[m], psi[m], s=3, color=colors[name], alpha=0.35, label=f"{name} (n={m.sum()})")
ax.set_xlabel("peak-to-peak amplitude (uV)")
ax.set_ylabel("Teager energy at trough")
ax.set_title("amplitude/energy: persistent vs short-lived")
ax.legend(fontsize=7)

# 3. ISI histogram, per-track computed then pooled
ax = axes[0, 2]
def pooled_isi_ms(tids):
    vals = []
    for t in tids:
        st = np.sort(spike_times[track_labels == t]) / FS
        if len(st) > 1:
            vals.append(np.diff(st) * 1000)
    return np.concatenate(vals) if vals else np.array([])

isi_persist = pooled_isi_ms(persist_tids)
ax.hist(isi_persist[isi_persist < 50], bins=50, histtype="step", color=colors["persistent"],
        lw=2, density=True, label=f"persistent (n={len(isi_persist)} isis)")
for name, tids in tiers.items():
    isi = pooled_isi_ms(tids)
    if len(isi) == 0:
        continue
    ax.hist(isi[isi < 50], bins=50, histtype="step", color=colors[name], lw=1.5, ls="--",
             density=True, label=f"{name} (n={len(isi)} isis)")
ax.axvline(1.5, color="k", lw=0.7, ls=":")
ax.set_xlabel("ISI (ms)")
ax.set_ylabel("density")
ax.set_title("ISI distribution (density-normalized): persistent vs short-lived")
ax.legend(fontsize=7)

# 4-6: sample individual raw traces (not mean) for persistent + 2 short-lived tiers
rng = np.random.default_rng(0)
for ax, (name, tids, color) in zip(
    axes[1],
    [("persistent", persist_tids, colors["persistent"]),
     ("long-lived fragment (>=6/22 windows)", tiers["long-lived fragment (>=6/22 windows)"], colors["long-lived fragment (>=6/22 windows)"]),
     ("single-window (1/22)", tiers["single-window (1/22)"], colors["single-window (1/22)"])],
):
    if name == "persistent":
        m = np.isin(track_labels, tids)
    else:
        m = np.isin(track_labels, tids)
    idx = np.where(m)[0]
    if len(idx) == 0:
        continue
    sample = rng.choice(idx, size=min(40, len(idx)), replace=False)
    for i in sample:
        ax.plot(t_ms, snips[i], color=color, alpha=0.25, lw=0.7)
    mw = snips[m].mean(0)
    ax.plot(t_ms, mw, color="black", lw=1.8, label="mean")
    ax.set_xlabel("ms")
    ax.set_ylabel("uV")
    ax.set_title(f"{name}: 40 individual raw traces (n={m.sum()} total)")
    ax.legend(fontsize=7)

plt.suptitle(f"Session {SESSION_ID}, ch {CHAN} -- what's in the non-persistent tracks?")
plt.tight_layout()
fig_path = rf"{OUT}\short_lived_track_inspection_{SESSION_ID}_ch{CHAN}.png"
plt.savefig(fig_path, dpi=120)
print(f"\nsaved {fig_path}")

# quantitative refractory-violation check (<1.5ms) per group
def viol_rate(tids):
    isi = pooled_isi_ms(tids)
    return (isi < 1.5).sum(), len(isi)

for name, tids in [("persistent", persist_tids)] + list(tiers.items()):
    v, n = viol_rate(tids)
    print(f"{name}: {v}/{n} ISIs < 1.5ms ({100*v/n if n else 0:.2f}%)")
