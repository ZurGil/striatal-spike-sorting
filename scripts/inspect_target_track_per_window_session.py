"""
Show one specific persistent track (by default: the higher-amplitude of the
2 session-persistent tracks from independent_clustering_windowed_overlap_session.py)
window by window, compared against every other spike detected in that same
window -- to see whether it stays visually separated from the rest of the
local cloud, and whether its own waveform shape stays consistent, across
the whole session.

Produces two montages:
  1. p2p-amplitude vs Teager-energy scatter, one panel per core window,
     target track highlighted against all other spikes in that window (gray).
  2. Raw waveform overlay (up to 40 traces), one panel per core window,
     target track's OWN spikes only, to check shape stability over time.

Reuses previously-saved npz outputs -- no raw-file re-read.

Usage: python inspect_target_track_per_window_session.py <session_id> <chan> [track_id]
       (if track_id omitted, picks the higher-mean-amplitude persistent track)
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
long_lived = do["long_lived_tracks"]
n_windows = int(do["n_windows"])
step_s = float(do["step_s"])
t0 = spike_times.min() / FS

if len(sys.argv) > 3:
    target = int(sys.argv[3])
else:
    means = {int(t): p2p[track_labels == t].mean() for t in long_lived}
    target = max(means, key=means.get)
    print(f"auto-picked higher-amplitude persistent track: {target} "
          f"(mean p2p={means[target]:.1f}uV vs others {means})")

m_target_all = track_labels == target
print(f"track {target}: {m_target_all.sum()} spikes total, "
      f"present in {len(set(core_w_of[m_target_all].tolist()))}/{n_windows} windows")

t_ms = (np.arange(snips.shape[1]) - NT_CENTER) / FS * 1000

# global axis ranges for comparability across panels
p2p_lim = (0, np.percentile(p2p, 99.5))
psi_lim = (min(0, psi.min()), np.percentile(psi, 99.5))
wf_lim = (np.percentile(snips[m_target_all], 0.5) - 20, np.percentile(snips[m_target_all], 99.5) + 20)

ncols, nrows = 5, int(np.ceil(n_windows / 5))

# --- figure 1: scatter, target vs everyone else, per window ---
fig1, axes1 = plt.subplots(nrows, ncols, figsize=(ncols * 3.4, nrows * 3.0))
axes1 = axes1.flatten()
for w in range(n_windows):
    ax = axes1[w]
    mw = core_w_of == w
    mt = mw & m_target_all
    mo = mw & ~m_target_all
    ax.scatter(p2p[mo], psi[mo], s=3, color="#c9cfc9", alpha=0.35, label=f"other (n={mo.sum()})")
    ax.scatter(p2p[mt], psi[mt], s=5, color="#0b7a75", alpha=0.6, label=f"trk{target} (n={mt.sum()})")
    t_start = t0 + w * step_s
    ax.set_title(f"w{w}: {t_start/60:.0f}-{(t_start+step_s)/60:.0f}min", fontsize=9)
    ax.set_xlim(*p2p_lim)
    ax.set_ylim(*psi_lim)
    ax.tick_params(labelsize=7)
    if w == 0:
        ax.set_xlabel("p2p (uV)", fontsize=8)
        ax.set_ylabel("Teager", fontsize=8)
for w in range(n_windows, len(axes1)):
    axes1[w].axis("off")
fig1.suptitle(f"Session {SESSION_ID}, ch {CHAN} -- track {target} (higher-amplitude persistent unit) "
              f"vs all other spikes, window by window", fontsize=13)
plt.tight_layout()
fig1_path = rf"{OUT}\track{target}_vs_others_per_window_{SESSION_ID}_ch{CHAN}.png"
plt.savefig(fig1_path, dpi=110)
print(f"saved {fig1_path}")

# --- figure 2: raw waveform overlay, target track only, per window ---
rng = np.random.default_rng(0)
fig2, axes2 = plt.subplots(nrows, ncols, figsize=(ncols * 3.4, nrows * 3.0))
axes2 = axes2.flatten()
for w in range(n_windows):
    ax = axes2[w]
    idx = np.where((core_w_of == w) & m_target_all)[0]
    t_start = t0 + w * step_s
    ax.set_title(f"w{w}: {t_start/60:.0f}-{(t_start+step_s)/60:.0f}min (n={len(idx)})", fontsize=9)
    if len(idx) == 0:
        ax.axis("off")
        continue
    sample = rng.choice(idx, size=min(40, len(idx)), replace=False)
    for i in sample:
        ax.plot(t_ms, snips[i], color="#0b7a75", alpha=0.25, lw=0.6)
    ax.plot(t_ms, snips[idx].mean(0), color="black", lw=1.5)
    ax.set_ylim(*wf_lim)
    ax.tick_params(labelsize=7)
    if w == 0:
        ax.set_xlabel("ms", fontsize=8)
        ax.set_ylabel("uV", fontsize=8)
for w in range(n_windows, len(axes2)):
    axes2[w].axis("off")
fig2.suptitle(f"Session {SESSION_ID}, ch {CHAN} -- track {target} own waveform shape, window by window "
              f"(black=mean, up to 40 raw traces)", fontsize=13)
plt.tight_layout()
fig2_path = rf"{OUT}\track{target}_waveform_stability_per_window_{SESSION_ID}_ch{CHAN}.png"
plt.savefig(fig2_path, dpi=110)
print(f"saved {fig2_path}")
