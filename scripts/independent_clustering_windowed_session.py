"""
Windowed version of the channel-129 independent clustering: instead of one
static PCA+GMM fit over the whole ~3.5h session (which is exposed to
electrode drift and slow amplitude change inflating cluster spread), split
the ALREADY-DETECTED spikes into non-overlapping time windows (WINDOW_S
each), recompute PCA + z-scored p2p/Teager features LOCALLY per window, and
cluster each window independently (k chosen by BIC, same as before). Then
link clusters across consecutive windows into persistent "tracks" by
Pearson correlation of their mean raw-uV waveform (correlation is
scale-invariant, so a track survives amplitude drift/burst decrement as
long as the waveform SHAPE stays consistent).

Reuses the spike_times + snips already saved by
independent_channel_clustering_session.py -- no raw-file re-read.

Every spike ends up with THREE labels for comparison:
  - label_allonce_pca       : original single-shot PCA-only clustering (k=5)
  - label_allonce_extra     : single-shot PCA+p2p+Teager clustering (k=6)
  - label_windowed_track    : persistent track ID from this windowed+linked approach

Usage: python independent_clustering_windowed_session.py <session_id> <chan> [window_s]
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
WINDOW_S = float(sys.argv[3]) if len(sys.argv) > 3 else 720.0  # 12 min default

OUT = r"D:\Gil\spike_sorting_agent\outputs"
NPZ_ALLONCE = rf"{OUT}\independent_clustering_{SESSION_ID}_ch{CHAN}.npz"
NPZ_EXTRA = rf"{OUT}\independent_clustering_extra_{SESSION_ID}_ch{CHAN}.npz"

FS = 30000.0
MAX_K = 6
MIN_SPIKES_FOR_K = 30  # need at least this many spikes per candidate cluster to try higher k
CORR_LINK_THRESH = float(sys.argv[4]) if len(sys.argv) > 4 else 0.92


def zscore(x):
    s = x.std()
    return (x - x.mean()) / s if s > 0 else x - x.mean()


def cluster_window(snips_w, p2p_w, psi_w):
    """Local PCA + z-scored p2p/Teager, GMM with BIC-selected k."""
    n = len(snips_w)
    if n < MIN_SPIKES_FOR_K:
        return np.zeros(n, dtype=int), 1  # too few spikes: treat as one cluster
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
    spike_times = d["spike_times"]  # in samples
    old_labels = d["labels"]
    de = np.load(NPZ_EXTRA)
    extra_labels = de["new_labels"]
    p2p_all = de["p2p"]
    psi_all = de["psi_trough"]

    t_sec = spike_times / FS
    t0, t1 = t_sec.min(), t_sec.max()
    n_windows = int(np.ceil((t1 - t0) / WINDOW_S))
    print(f"{len(spike_times)} spikes, {t1 - t0:.0f}s span, {n_windows} windows of {WINDOW_S:.0f}s")

    window_idx = np.floor((t_sec - t0) / WINDOW_S).astype(int)
    window_idx = np.clip(window_idx, 0, n_windows - 1)

    local_labels = np.full(len(spike_times), -1, dtype=int)
    window_cluster_meanwf = {}   # (w, local_k) -> mean waveform
    window_cluster_n = {}
    window_best_k = np.zeros(n_windows, dtype=int)

    for w in range(n_windows):
        m = window_idx == w
        n = m.sum()
        if n == 0:
            continue
        labels_w, best_k = cluster_window(snips[m], p2p_all[m], psi_all[m])
        local_labels[m] = labels_w
        window_best_k[w] = best_k
        for k in range(best_k):
            mk = m.copy()
            mk[m] = labels_w == k
            window_cluster_meanwf[(w, k)] = snips[mk].mean(0)
            window_cluster_n[(w, k)] = mk.sum()
        print(f"  window {w}: {n} spikes, k={best_k}")

    # link clusters across consecutive windows by mean-waveform correlation
    track_id_of = {}  # (w, k) -> track id
    next_track = 0
    for w in range(n_windows):
        ks_this = [k for (ww, k) in window_cluster_meanwf if ww == w]
        for k in ks_this:
            if (w, k) not in track_id_of:
                track_id_of[(w, k)] = next_track
                next_track += 1
        if w + 1 >= n_windows:
            continue
        ks_next = [k for (ww, k) in window_cluster_meanwf if ww == w + 1]
        if not ks_this or not ks_next:
            continue
        cost = np.ones((len(ks_this), len(ks_next)))
        for i, ki in enumerate(ks_this):
            wf_i = window_cluster_meanwf[(w, ki)]
            for j, kj in enumerate(ks_next):
                wf_j = window_cluster_meanwf[(w + 1, kj)]
                r = np.corrcoef(wf_i, wf_j)[0, 1]
                cost[i, j] = 1 - r
        row, col = linear_sum_assignment(cost)
        r_vals = [1 - cost[i, j] for i, j in zip(row, col)]
        print(f"    window {w}->{w+1} best-match r values: {[f'{r:.3f}' for r in sorted(r_vals, reverse=True)]}")
        for i, j in zip(row, col):
            r = 1 - cost[i, j]
            if r > CORR_LINK_THRESH:
                track_id_of[(w + 1, ks_next[j])] = track_id_of[(w, ks_this[i])]
            else:
                track_id_of[(w + 1, ks_next[j])] = next_track
                next_track += 1

    track_labels = np.full(len(spike_times), -1, dtype=int)
    for (w, k), tid in track_id_of.items():
        m = (window_idx == w) & (local_labels == k)
        track_labels[m] = tid

    n_tracks = next_track
    track_n_windows = {}
    track_total_spikes = {}
    for (w, k), tid in track_id_of.items():
        track_n_windows.setdefault(tid, set()).add(w)
        track_total_spikes[tid] = track_total_spikes.get(tid, 0) + window_cluster_n[(w, k)]

    lifetimes = {tid: len(ws) for tid, ws in track_n_windows.items()}
    print(f"\nall track lifetimes (windows spanned), sorted desc: {sorted(lifetimes.values(), reverse=True)}")
    print(f"windows-set for top-3 longest tracks:")
    for tid in sorted(lifetimes, key=lambda t: -lifetimes[t])[:3]:
        print(f"  track {tid}: windows={sorted(track_n_windows[tid])}")
    long_lived = [tid for tid, lf in lifetimes.items() if lf >= 0.7 * n_windows]
    print(f"\n{n_tracks} total tracks found across {n_windows} windows")
    print(f"tracks present in >=70% of windows (long-lived, likely real units): {len(long_lived)}")
    for tid in sorted(long_lived, key=lambda t: -track_total_spikes[t]):
        print(f"  track {tid}: present in {lifetimes[tid]}/{n_windows} windows, "
              f"{track_total_spikes[tid]} total spikes, "
              f"rate={track_total_spikes[tid]/(t1-t0):.2f}Hz")
    short_lived = [tid for tid in lifetimes if tid not in long_lived]
    print(f"short-lived tracks (<70% windows, likely fragments/noise/drift artifacts): {len(short_lived)}, "
          f"{sum(track_total_spikes[t] for t in short_lived)} spikes total")

    np.savez(rf"{OUT}\independent_clustering_windowed_{SESSION_ID}_ch{CHAN}.npz",
              spike_times=spike_times, window_idx=window_idx, local_labels=local_labels,
              track_labels=track_labels, label_allonce_pca=old_labels, label_allonce_extra=extra_labels,
              n_windows=n_windows, window_s=WINDOW_S, n_tracks=n_tracks,
              long_lived_tracks=np.array(long_lived))

    # comparison table: long-lived tracks vs allonce_pca (k=5) vs allonce_extra (k=6)
    print(f"\nsummary: allonce_pca k=5, allonce_extra k=6, windowed long-lived tracks k={len(long_lived)}")

    n_pca = old_labels.max() + 1
    n_extra = extra_labels.max() + 1
    print(f"\ncross-tab: long-lived windowed tracks vs allonce_pca (k={n_pca})")
    print("        " + "".join(f"pca{c:>6d}" for c in range(n_pca)))
    for tid in sorted(long_lived, key=lambda t: -track_total_spikes[t]):
        m = track_labels == tid
        row = "".join(f"{(old_labels[m] == c).sum():>9d}" for c in range(n_pca))
        print(f"trk{tid:>2d}  {row}")

    print(f"\ncross-tab: long-lived windowed tracks vs allonce_extra (k={n_extra})")
    print("        " + "".join(f"ext{c:>6d}" for c in range(n_extra)))
    for tid in sorted(long_lived, key=lambda t: -track_total_spikes[t]):
        m = track_labels == tid
        row = "".join(f"{(extra_labels[m] == c).sum():>9d}" for c in range(n_extra))
        print(f"trk{tid:>2d}  {row}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for tid in sorted(track_n_windows):
        ws = sorted(track_n_windows[tid])
        color = "C0" if tid in long_lived else "lightgray"
        sizes = [window_cluster_n.get((w, k), 10) for w in ws
                 for k in [kk for (ww, kk) in track_id_of if ww == w and track_id_of[(ww, kk)] == tid]]
        axes[0].scatter(ws, [tid] * len(ws), s=8, color=color)
    axes[0].set_xlabel("window index (12min each)")
    axes[0].set_ylabel("track id")
    axes[0].set_title(f"track persistence over session (blue=long-lived, n={len(long_lived)})")

    axes[1].bar(range(n_windows), window_best_k)
    axes[1].set_xlabel("window index")
    axes[1].set_ylabel("local k (clusters found)")
    axes[1].set_title("local cluster count per window")

    plt.suptitle(f"Session {SESSION_ID}, channel {CHAN} -- windowed clustering ({WINDOW_S:.0f}s windows) + track linking")
    plt.tight_layout()
    fig_path = rf"{OUT}\independent_clustering_windowed_{SESSION_ID}_ch{CHAN}.png"
    plt.savefig(fig_path, dpi=120)
    print(f"\nsaved {fig_path}")


if __name__ == "__main__":
    main()
