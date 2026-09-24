"""
Rerun the independent channel-129 clustering (see
independent_channel_clustering_session.py) with two extra features added
alongside the 3 PCA components: peak-to-peak amplitude and the Teager-Kaiser
energy operator evaluated at each snippet's trough (psi_trough), both
z-scored so they're on comparable footing with the PC scores. Purpose:
check whether these amplitude/energy-sensitive features change the cluster
count or boundaries found by PCA-shape alone, and specifically whether they
artificially split any cluster via burst-related amplitude decrement.

Reuses the snippets already saved by independent_channel_clustering_session.py
(no re-read of the raw .dat file needed).

Usage: python independent_clustering_extra_features_session.py <session_id> <chan>
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

SESSION_ID = sys.argv[1]
CHAN = int(sys.argv[2])
OUT = r"D:\Gil\spike_sorting_agent\outputs"
NPZ_IN = rf"{OUT}\independent_clustering_{SESSION_ID}_ch{CHAN}.npz"

MAX_K = 6
FS = 30000.0


def teager(x):
    """psi[n] = x[n]^2 - x[n-1]*x[n+1], full array, edges = nan."""
    psi = np.full_like(x, np.nan)
    psi[1:-1] = x[1:-1] ** 2 - x[:-2] * x[2:]
    return psi


def zscore(x):
    return (x - x.mean()) / x.std()


def main():
    d = np.load(NPZ_IN)
    snips = d["snips"]  # (n_spikes, NT), already in uV
    scores_pca = d["scores"]  # (n_spikes, 3)
    spike_times = d["spike_times"]
    old_labels = d["labels"]
    old_best_k = int(d["best_k"])
    n_spikes, NT = snips.shape
    print(f"loaded {n_spikes} spikes, NT={NT}, old best_k={old_best_k}")

    # peak-to-peak amplitude
    p2p = snips.max(axis=1) - snips.min(axis=1)

    # Teager energy at each snippet's own trough sample
    trough_idx = snips.argmin(axis=1)
    trough_idx = np.clip(trough_idx, 1, NT - 2)
    psi_trough = np.array([
        snips[i, t] ** 2 - snips[i, t - 1] * snips[i, t + 1]
        for i, t in enumerate(trough_idx)
    ])

    print(f"p2p amplitude: mean={p2p.mean():.1f}uV, std={p2p.std():.1f}uV")
    print(f"psi_trough: mean={psi_trough.mean():.1f}, std={psi_trough.std():.1f}")

    # z-score PCA scores too, so all 5 dims are comparable scale
    feats = np.column_stack([
        zscore(scores_pca[:, 0]),
        zscore(scores_pca[:, 1]),
        zscore(scores_pca[:, 2]),
        zscore(p2p),
        zscore(psi_trough),
    ])

    bics = []
    models = {}
    for k in range(1, MAX_K + 1):
        gmm = GaussianMixture(n_components=k, n_init=3, random_state=0)
        gmm.fit(feats)
        bic = gmm.bic(feats)
        bics.append(bic)
        models[k] = gmm
        print(f"  k={k}: BIC={bic:.0f}")
    best_k = int(np.argmin(bics)) + 1
    print(f"\nbest k by BIC (5D w/ p2p+Teager): {best_k}  (was {old_best_k} with PCA-only)")
    new_labels = models[best_k].predict(feats)

    print("\nnew clusters:")
    for k in range(best_k):
        m = new_labels == k
        rate = m.sum() / ((spike_times.max() - spike_times.min()) / FS)
        print(f"  cluster {k}: n={m.sum()}, rate={rate:.2f}Hz, "
              f"mean_p2p={p2p[m].mean():.1f}uV, mean_psi={psi_trough[m].mean():.1f}")

    # cross-tab old (PCA-only) vs new (PCA+p2p+Teager) labels
    crosstab = np.zeros((old_best_k, best_k), dtype=int)
    for o, n in zip(old_labels, new_labels):
        crosstab[o, n] += 1
    print("\ncross-tab: rows=old PCA-only cluster, cols=new (PCA+p2p+Teager) cluster")
    header = "        " + "".join(f"new{n:>7d}" for n in range(best_k))
    print(header)
    for o in range(old_best_k):
        row = "".join(f"{crosstab[o, n]:>10d}" for n in range(best_k))
        print(f"old{o:>2d}  {row}")

    np.savez(rf"{OUT}\independent_clustering_extra_{SESSION_ID}_ch{CHAN}.npz",
              spike_times=spike_times, p2p=p2p, psi_trough=psi_trough,
              feats=feats, new_labels=new_labels, old_labels=old_labels,
              best_k=best_k, bics=np.array(bics), crosstab=crosstab)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    colors_new = plt.cm.tab10(np.linspace(0, 1, best_k))
    colors_old = plt.cm.tab10(np.linspace(0, 1, old_best_k))

    axes[0, 0].plot(range(1, MAX_K + 1), bics, "o-")
    axes[0, 0].axvline(best_k, color="r", ls="--")
    axes[0, 0].set_xlabel("k")
    axes[0, 0].set_ylabel("BIC")
    axes[0, 0].set_title(f"BIC, 5D (PCA+p2p+Teager), best k={best_k}")

    for k in range(old_best_k):
        m = old_labels == k
        axes[0, 1].scatter(scores_pca[m, 0], scores_pca[m, 1], s=4, color=colors_old[k], alpha=0.5, label=f"old {k}")
    axes[0, 1].set_xlabel("PC1")
    axes[0, 1].set_ylabel("PC2")
    axes[0, 1].set_title(f"ORIGINAL clustering (PCA-only, k={old_best_k})")
    axes[0, 1].legend(fontsize=7)

    for k in range(best_k):
        m = new_labels == k
        axes[0, 2].scatter(scores_pca[m, 0], scores_pca[m, 1], s=4, color=colors_new[k], alpha=0.5, label=f"new {k}")
    axes[0, 2].set_xlabel("PC1")
    axes[0, 2].set_ylabel("PC2")
    axes[0, 2].set_title(f"NEW clustering (PCA+p2p+Teager, k={best_k}), same PC1/PC2 axes")
    axes[0, 2].legend(fontsize=7)

    for k in range(best_k):
        m = new_labels == k
        axes[1, 0].scatter(p2p[m], psi_trough[m], s=4, color=colors_new[k], alpha=0.5, label=f"new {k}")
    axes[1, 0].set_xlabel("peak-to-peak amplitude (uV)")
    axes[1, 0].set_ylabel("Teager energy at trough")
    axes[1, 0].set_title("new clusters in p2p/Teager space")
    axes[1, 0].legend(fontsize=7)

    im = axes[1, 1].imshow(crosstab, cmap="viridis", aspect="auto")
    axes[1, 1].set_xlabel("new cluster")
    axes[1, 1].set_ylabel("old cluster")
    axes[1, 1].set_xticks(range(best_k))
    axes[1, 1].set_yticks(range(old_best_k))
    for o in range(old_best_k):
        for n in range(best_k):
            axes[1, 1].text(n, o, crosstab[o, n], ha="center", va="center", color="w", fontsize=8)
    axes[1, 1].set_title("cross-tab: old vs new cluster membership")
    plt.colorbar(im, ax=axes[1, 1])

    t_ms = (np.arange(NT) - 20) / FS * 1000
    for k in range(best_k):
        m = new_labels == k
        mean_wf = snips[m].mean(0)
        axes[1, 2].plot(t_ms, mean_wf, color=colors_new[k], label=f"new {k} (n={m.sum()})")
    axes[1, 2].set_xlabel("ms from threshold crossing")
    axes[1, 2].set_ylabel("uV")
    axes[1, 2].legend(fontsize=7)
    axes[1, 2].set_title("mean waveform per NEW cluster")

    plt.suptitle(f"Session {SESSION_ID}, channel {CHAN} -- PCA-only vs PCA+p2p+Teager clustering")
    plt.tight_layout()
    fig_path = rf"{OUT}\independent_clustering_extra_{SESSION_ID}_ch{CHAN}.png"
    plt.savefig(fig_path, dpi=120)
    print(f"\nsaved {fig_path}")


if __name__ == "__main__":
    main()
