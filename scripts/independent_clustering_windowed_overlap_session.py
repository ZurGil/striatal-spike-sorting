"""
Overlapping-window variant of independent_clustering_windowed_session.py.

Instead of non-overlapping 12min windows linked by mean-waveform correlation,
use windows of length WINDOW_S stepped by STEP_S < WINDOW_S, so consecutive
windows share an overlap zone of (WINDOW_S - STEP_S) seconds. Every spike in
the overlap zone gets clustered as part of BOTH windows' independent local
fits. That gives a much more direct linking signal than correlation: for
each pair of adjacent windows, build a contingency table of "which local
cluster did this SAME spike fall into in window w" vs "...in window w+1",
over just the overlap-zone spikes, and link whichever pair of clusters
shares the most spikes (normalized by the smaller side, like a Jaccard
score). Overlap spikes are used only to help fit + link; final per-spike
track attribution uses each spike's own non-overlapping "core" window
(STEP_S-sized partition) exactly once, so nothing is double-counted.

Reuses the spike_times + snips already saved by
independent_channel_clustering_session.py -- no raw-file re-read.

Usage: python independent_clustering_windowed_overlap_session.py <session_id> <chan> [window_s] [step_s] [link_frac_thresh]
"""
import sys
import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

SESSION_ID = sys.argv[1]
CHAN = int(sys.argv[2])
WINDOW_S = float(sys.argv[3]) if len(sys.argv) > 3 else 900.0   # 15 min clustering window
STEP_S = float(sys.argv[4]) if len(sys.argv) > 4 else 600.0     # 10 min core step -> 5min overlap
LINK_FRAC_THRESH = float(sys.argv[5]) if len(sys.argv) > 5 else 0.5

OUT = r"D:\Gil\spike_sorting_agent\outputs"
NPZ_ALLONCE = rf"{OUT}\independent_clustering_{SESSION_ID}_ch{CHAN}.npz"
NPZ_EXTRA = rf"{OUT}\independent_clustering_extra_{SESSION_ID}_ch{CHAN}.npz"

FS = 30000.0
MAX_K = 6
MIN_SPIKES_FOR_K = 30
NT_CENTER = 20  # NT0MIN, for time axis in plots


def zscore(x):
    s = x.std()
    return (x - x.mean()) / s if s > 0 else x - x.mean()


def cluster_window(snips_w, p2p_w, psi_w):
    n = len(snips_w)
    if n < MIN_SPIKES_FOR_K:
        return np.zeros(n, dtype=int), 1
    pca = PCA(n_components=3)
    local_scores = pca.fit_transform(snips_w - snips_w.mean(0))
    feats = np.column_stack([
        zscore(local_scores[:, 0]), zscore(local_scores[:, 1]), zscore(local_scores[:, 2]),
        zscore(p2p_w), zscore(psi_w),
    ])
    max_k = min(MAX_K, max(1, n // MIN_SPIKES_FOR_K))
    bics, models = [], {}
    for k in range(1, max_k + 1):
        gmm = GaussianMixture(n_components=k, n_init=3, random_state=0)
        gmm.fit(feats)
        bics.append(gmm.bic(feats))
        models[k] = gmm
    best_k = int(np.argmin(bics)) + 1
    return models[best_k].predict(feats), best_k


def main():
    d = np.load(NPZ_ALLONCE)
    snips = d["snips"]
    spike_times = d["spike_times"]
    old_labels = d["labels"]
    de = np.load(NPZ_EXTRA)
    extra_labels = de["new_labels"]
    p2p_all = de["p2p"]
    psi_all = de["psi_trough"]

    t_sec = spike_times / FS
    t0, t1 = t_sec.min(), t_sec.max()
    n_windows = int(np.ceil((t1 - t0) / STEP_S))
    print(f"{len(spike_times)} spikes, {t1 - t0:.0f}s span, {n_windows} windows "
          f"(len={WINDOW_S:.0f}s, step={STEP_S:.0f}s, overlap={WINDOW_S - STEP_S:.0f}s)")

    # per-window: indices into global spike arrays + local cluster labels
    win_idx, win_labels, win_bestk = {}, {}, {}
    win_meanwf, win_n = {}, {}

    for w in range(n_windows):
        cs = t0 + w * STEP_S
        ce = cs + WINDOW_S
        idx = np.where((t_sec >= cs) & (t_sec < ce))[0]
        if len(idx) == 0:
            win_idx[w] = idx
            win_labels[w] = np.array([], dtype=int)
            win_bestk[w] = 0
            continue
        labels_w, best_k = cluster_window(snips[idx], p2p_all[idx], psi_all[idx])
        win_idx[w] = idx
        win_labels[w] = labels_w
        win_bestk[w] = best_k
        for k in range(best_k):
            mk = labels_w == k
            win_meanwf[(w, k)] = snips[idx[mk]].mean(0)
            win_n[(w, k)] = mk.sum()
        print(f"  window {w} [{cs:.0f},{ce:.0f})s: {len(idx)} spikes, k={best_k}")

    # link adjacent windows using the OVERLAP zone's dual cluster membership
    track_id_of = {}
    next_track = 0
    for w in range(n_windows):
        for k in range(win_bestk[w]):
            if (w, k) not in track_id_of:
                track_id_of[(w, k)] = next_track
                next_track += 1
        if w + 1 >= n_windows or win_bestk[w] == 0 or win_bestk[w + 1] == 0:
            continue
        # overlap zone spikes = intersection of window w's and window w+1's index sets
        set_w = {gi: li for li, gi in enumerate(win_idx[w])}
        set_w1 = {gi: li for li, gi in enumerate(win_idx[w + 1])}
        overlap_gis = set(set_w) & set(set_w1)
        if not overlap_gis:
            print(f"    window {w}->{w+1}: NO overlap spikes (gap?)")
            continue
        cnt = np.zeros((win_bestk[w], win_bestk[w + 1]), dtype=int)
        for gi in overlap_gis:
            kw = win_labels[w][set_w[gi]]
            kw1 = win_labels[w + 1][set_w1[gi]]
            cnt[kw, kw1] += 1
        row_tot = cnt.sum(axis=1, keepdims=True)
        col_tot = cnt.sum(axis=0, keepdims=True)
        denom = np.minimum(row_tot, col_tot)
        denom[denom == 0] = 1
        frac = cnt / denom
        row, col = linear_sum_assignment(-frac)
        f_vals = [frac[i, j] for i, j in zip(row, col)]
        print(f"    window {w}->{w+1}: {len(overlap_gis)} overlap spikes, "
              f"best-match agreement fractions: {[f'{f:.2f}' for f in sorted(f_vals, reverse=True)]}")
        for i, j in zip(row, col):
            if frac[i, j] > LINK_FRAC_THRESH:
                track_id_of[(w + 1, j)] = track_id_of[(w, i)]
            else:
                track_id_of[(w + 1, j)] = next_track
                next_track += 1

    # final per-spike attribution: each spike's own non-overlapping CORE window, once
    n_spikes = len(spike_times)
    track_labels = np.full(n_spikes, -1, dtype=int)
    core_w_of = np.clip(np.floor((t_sec - t0) / STEP_S).astype(int), 0, n_windows - 1)
    for w in range(n_windows):
        idx = win_idx[w]
        if len(idx) == 0:
            continue
        core_mask_local = core_w_of[idx] == w
        for li in np.where(core_mask_local)[0]:
            gi = idx[li]
            k = win_labels[w][li]
            track_labels[gi] = track_id_of[(w, k)]
    assert (track_labels >= 0).all(), "every spike must get exactly one core-window label"

    n_tracks = next_track
    track_windows = {}
    track_spikes = {}
    for gi in range(n_spikes):
        tid = track_labels[gi]
        track_windows.setdefault(tid, set()).add(core_w_of[gi])
        track_spikes[tid] = track_spikes.get(tid, 0) + 1
    lifetimes = {tid: len(ws) for tid, ws in track_windows.items()}
    print(f"\n{n_tracks} total tracks, lifetimes (core windows) sorted desc: "
          f"{sorted(lifetimes.values(), reverse=True)}")

    long_lived = [tid for tid, lf in lifetimes.items() if lf >= 0.7 * n_windows]
    print(f"long-lived (>=70% of {n_windows} windows): {len(long_lived)}")
    for tid in sorted(long_lived, key=lambda t: -track_spikes[t]):
        rate = track_spikes[tid] / (t1 - t0)
        print(f"  track {tid}: {lifetimes[tid]}/{n_windows} windows, {track_spikes[tid]} spikes, {rate:.2f}Hz")
    short_lived = [tid for tid in lifetimes if tid not in long_lived]
    print(f"short-lived: {len(short_lived)} tracks, {sum(track_spikes[t] for t in short_lived)} spikes")

    n_pca = old_labels.max() + 1
    n_extra = extra_labels.max() + 1
    print(f"\ncross-tab: long-lived overlap-windowed tracks vs allonce_pca (k={n_pca})")
    print("        " + "".join(f"pca{c:>6d}" for c in range(n_pca)))
    for tid in sorted(long_lived, key=lambda t: -track_spikes[t]):
        m = track_labels == tid
        row = "".join(f"{(old_labels[m] == c).sum():>9d}" for c in range(n_pca))
        print(f"trk{tid:>2d}  {row}")

    np.savez(rf"{OUT}\independent_clustering_windowed_overlap_{SESSION_ID}_ch{CHAN}.npz",
              spike_times=spike_times, core_w_of=core_w_of, track_labels=track_labels,
              label_allonce_pca=old_labels, label_allonce_extra=extra_labels,
              n_windows=n_windows, window_s=WINDOW_S, step_s=STEP_S, n_tracks=n_tracks,
              long_lived_tracks=np.array(long_lived))

    # figure: persistence scatter + mean waveform per long-lived track
    t_ms = (np.arange(snips.shape[1]) - NT_CENTER) / FS * 1000
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for tid in sorted(track_windows):
        ws = sorted(track_windows[tid])
        color = "C0" if tid in long_lived else "lightgray"
        axes[0].scatter(ws, [tid] * len(ws), s=8, color=color)
    axes[0].set_xlabel("core window index")
    axes[0].set_ylabel("track id")
    axes[0].set_title(f"track persistence (blue=long-lived, n={len(long_lived)}/{n_tracks} total)")

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(long_lived), 1)))
    for c, tid in enumerate(sorted(long_lived, key=lambda t: -track_spikes[t])):
        m = track_labels == tid
        mean_wf = snips[m].mean(0)
        sem_wf = snips[m].std(0) / np.sqrt(m.sum())
        axes[1].plot(t_ms, mean_wf, color=colors[c], label=f"track {tid} (n={m.sum()}, {track_spikes[tid]/(t1-t0):.2f}Hz)")
        axes[1].fill_between(t_ms, mean_wf - sem_wf, mean_wf + sem_wf, color=colors[c], alpha=0.2)
    axes[1].set_xlabel("ms from threshold crossing")
    axes[1].set_ylabel("uV")
    axes[1].legend(fontsize=8)
    axes[1].set_title("mean waveform, long-lived (session-persistent) tracks")

    bestk_arr = [win_bestk[w] for w in range(n_windows)]
    axes[2].bar(range(n_windows), bestk_arr)
    axes[2].set_xlabel("window index")
    axes[2].set_ylabel("local k")
    axes[2].set_title("local cluster count per window")

    plt.suptitle(f"Session {SESSION_ID}, ch {CHAN} -- OVERLAP-linked windowed clustering "
                 f"(win={WINDOW_S:.0f}s, step={STEP_S:.0f}s)")
    plt.tight_layout()
    fig_path = rf"{OUT}\independent_clustering_windowed_overlap_{SESSION_ID}_ch{CHAN}.png"
    plt.savefig(fig_path, dpi=120)
    print(f"\nsaved {fig_path}")


if __name__ == "__main__":
    main()
