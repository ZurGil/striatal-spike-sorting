"""
Independent (non-Kilosort) spike detection + clustering for ONE raw channel,
for comparison against Kilosort4's own output on the same channel. Built to
diagnose whether known-responsive channels are genuinely not showing
task responses, or whether something in the Kilosort/pipeline chain is
losing them.

Method (deliberately simple/classical, NOT the same machinery as Kilosort,
so it's a genuinely independent check):
  1. Read the FULL session's continuous trace for one raw channel index
     (already-extracted probe1.dat, i.e. the .rec file's own data,
     demultiplexed -- see raw_channel_check.py precedent).
  2. Highpass filter (same cfg.highpass_hz as the rest of the pipeline).
  3. Threshold spike detection: MAD-based threshold (4.5x MAD, same
     convention as raw_cluster_points_session.py), negative-going
     crossings only, with a refractory dead-time to avoid double-counting
     one spike's multiple threshold crossings.
  4. Extract NT=61, NT0MIN=20 waveform snippets (matches Kilosort4's own
     'nt'/'nt0min' ops for this session, for a fair shape comparison).
  5. PCA (top N_PCS components) on the snippet waveforms.
  6. GMM clustering on the PC scores, with the number of components chosen
     by BIC over a range (1..MAX_K) -- reports the full BIC curve, not just
     the argmin, so a human can sanity-check the automatic choice.
  7. Outputs: per-cluster spike times (saved), mean waveform per cluster,
     PCA scatter colored by cluster, ISI histogram per cluster (refractory-
     violation sanity check), and firing rates.

Usage: python independent_channel_clustering_session.py <session_id> <rec_root> <raw_channel_index> <gain_to_uV>

MEMORY-SAFETY REWRITE (2026-09-17): the original version used
`striatal_agent.raw_io.RawReader`, which memory-maps the ENTIRE probe1.dat
(hundreds of GB) once and keeps that mapping alive for the reader's whole
lifetime -- every `read_window()` call is a slice into that one persistent
memmap. A full sequential scan of a file much larger than RAM (this one's
~270GB against 64GB RAM) through a single persistent memmap lets Windows'
page cache (standby list) balloon to consume nearly all physical memory by
the end of the scan -- technically reclaimable, but enough to trip a
low-memory kill TWICE on this exact script (once concurrently with
Kilosort4, once running completely alone -- ruling out concurrency as the
cause; the memmap access pattern itself is the problem). Fixed by reading
each chunk with plain buffered file I/O (`open()` + `seek()` + `read()`)
instead of a persistent memmap: each chunk's bytes are read into an ordinary
buffer, converted, and then explicitly released (`del` + periodic
`gc.collect()`) before the next chunk, so no single long-lived object ever
holds the whole file's cache footprint. Chunk size also cut from 300s to 60s
to keep the peak per-iteration footprint small. If this still causes memory
pressure, the next fallback is restricting to a shorter time window rather
than the full session.
"""
import os, sys, gc, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from striatal_agent.config import AgentConfig
from striatal_agent.raw_io import highpass_filter

SESSION_ID = sys.argv[1]
REC_ROOT = sys.argv[2]
CHAN = int(sys.argv[3])
GAIN_TO_UV = float(sys.argv[4])

DAT_PATH = rf"{REC_ROOT}\{SESSION_ID}.kilosort\{SESSION_ID}.probe1.dat"
OUT = r"D:\Gil\spike_sorting_agent\outputs"

FS = 30000.0
NT, NT0MIN = 61, 20
DEAD_SAMPLES = int(0.001 * FS)
THRESH_MAD = 4.5
N_PCS = 3
MAX_K = 6
CHUNK_S = 60  # process in 1-min chunks to bound memory (reduced from 300s)
N_CHAN_BIN = 384
DTYPE = np.int16


def read_chunk_raw(f, start, end, chan, margin):
    """Read exactly [start-margin, end+margin) samples for ONE channel via
    plain buffered file I/O -- no persistent memmap, no lingering OS cache
    tied to a long-lived object. Returns (col_float32, actual_start)."""
    itemsize = np.dtype(DTYPE).itemsize
    lo = max(0, start - margin)
    hi = end + margin  # caller clamps to n_samples via n_bytes check below
    f.seek(lo * N_CHAN_BIN * itemsize)
    n_rows_requested = hi - lo
    raw_bytes = f.read(n_rows_requested * N_CHAN_BIN * itemsize)
    n_rows = len(raw_bytes) // (N_CHAN_BIN * itemsize)  # last chunk may be short
    raw = np.frombuffer(raw_bytes, dtype=DTYPE, count=n_rows * N_CHAN_BIN).reshape(n_rows, N_CHAN_BIN)
    col = raw[:, chan].astype(np.float32).copy()
    del raw, raw_bytes
    return col, lo, n_rows


def detect_and_snip(n_samples, chan, cfg):
    chunk_len = int(CHUNK_S * FS)
    all_times, all_snips = [], []
    margin = 200
    t0 = time.time()
    with open(DAT_PATH, "rb") as f:
        for start in range(0, n_samples, chunk_len):
            end = min(start + chunk_len, n_samples)
            block1d, s2, n_rows = read_chunk_raw(f, start, end, chan, margin)
            e2 = s2 + n_rows
            filt = highpass_filter(block1d[:, None], FS, cfg.highpass_hz, axis=0)[:, 0]
            del block1d
            lo = start - s2
            hi = end - s2
            trace = filt[lo:hi]
            mad = np.median(np.abs(trace - np.median(trace))) * 1.4826
            thresh = THRESH_MAD * mad
            below = trace < -thresh
            crossings = np.where(below & ~np.roll(below, 1))[0]
            keep, last = [], -DEAD_SAMPLES
            for c in crossings:
                if c - last >= DEAD_SAMPLES:
                    keep.append(c)
                    last = c
            crossings = np.array(keep)
            crossings = crossings[(crossings + lo > NT0MIN + 2) & (crossings + lo < len(filt) - (NT - NT0MIN) - 2)]
            for c in crossings:
                abs_idx = c + lo
                wf = filt[abs_idx - NT0MIN: abs_idx - NT0MIN + NT]
                if len(wf) == NT:
                    all_times.append(abs_idx + s2)
                    all_snips.append(wf)
            del filt, trace
            if (start // chunk_len) % 20 == 0:
                elapsed = time.time() - t0
                frac = max(start / n_samples, 1e-6)
                eta = elapsed / frac - elapsed
                print(f"  {start/FS:.0f}/{n_samples/FS:.0f}s, {len(all_times)} spikes so far "
                      f"(wall {elapsed:.0f}s elapsed, ETA {eta:.0f}s)")
                gc.collect()
    return np.array(all_times), np.array(all_snips)


def main():
    t_wall_start = time.time()
    cfg = AgentConfig()
    itemsize = np.dtype(DTYPE).itemsize
    n_bytes = os.path.getsize(DAT_PATH)
    if n_bytes % (N_CHAN_BIN * itemsize) != 0:
        raise ValueError(f"{DAT_PATH}: size not divisible by n_chan*itemsize")
    n_samples = n_bytes // (N_CHAN_BIN * itemsize)
    print(f"session {SESSION_ID}, channel {CHAN}, {n_samples/FS:.1f}s total")

    spike_times, snips = detect_and_snip(n_samples, CHAN, cfg)
    print(f"\ndetected {len(spike_times)} spikes (threshold={THRESH_MAD}x MAD)")

    snips_uv = snips * GAIN_TO_UV
    pca = PCA(n_components=N_PCS)
    scores = pca.fit_transform(snips_uv - snips_uv.mean(0))
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_.round(3)}")

    bics = []
    models = {}
    for k in range(1, MAX_K + 1):
        gmm = GaussianMixture(n_components=k, n_init=3, random_state=0)
        gmm.fit(scores)
        bic = gmm.bic(scores)
        bics.append(bic)
        models[k] = gmm
        print(f"  k={k}: BIC={bic:.0f}")
    best_k = int(np.argmin(bics)) + 1
    print(f"\nbest k by BIC: {best_k}")
    labels = models[best_k].predict(scores)

    np.savez(rf"{OUT}\independent_clustering_{SESSION_ID}_ch{CHAN}.npz",
              spike_times=spike_times, snips=snips_uv, scores=scores, labels=labels, best_k=best_k, bics=np.array(bics))

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    colors = plt.cm.tab10(np.linspace(0, 1, best_k))

    axes[0, 0].plot(range(1, MAX_K + 1), bics, "o-")
    axes[0, 0].axvline(best_k, color="r", ls="--")
    axes[0, 0].set_xlabel("k (clusters)")
    axes[0, 0].set_ylabel("BIC")
    axes[0, 0].set_title("GMM model selection")

    for k in range(best_k):
        m = labels == k
        axes[0, 1].scatter(scores[m, 0], scores[m, 1], s=4, color=colors[k], label=f"cluster {k} (n={m.sum()})", alpha=0.5)
    axes[0, 1].set_xlabel("PC1")
    axes[0, 1].set_ylabel("PC2")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].set_title(f"PCA scores, k={best_k}")

    t_ms = (np.arange(NT) - NT0MIN) / FS * 1000
    for k in range(best_k):
        m = labels == k
        mean_wf = snips_uv[m].mean(0)
        sem_wf = snips_uv[m].std(0) / np.sqrt(m.sum())
        axes[1, 0].plot(t_ms, mean_wf, color=colors[k], label=f"cluster {k}")
        axes[1, 0].fill_between(t_ms, mean_wf - sem_wf, mean_wf + sem_wf, color=colors[k], alpha=0.2)
    axes[1, 0].set_xlabel("ms from threshold crossing")
    axes[1, 0].set_ylabel("uV")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].set_title("mean waveform per cluster")

    for k in range(best_k):
        m = labels == k
        st = np.sort(spike_times[m]) / FS
        isi_ms = np.diff(st) * 1000
        rate_hz = m.sum() / (n_samples / FS)
        axes[1, 1].hist(isi_ms[isi_ms < 50], bins=50, histtype="step", color=colors[k],
                         label=f"cluster {k}: {rate_hz:.2f}Hz")
    axes[1, 1].axvline(1.5, color="k", lw=0.7, ls="--")
    axes[1, 1].set_xlabel("ISI (ms)")
    axes[1, 1].set_ylabel("count")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].set_title("ISI histogram per cluster (<50ms)")

    plt.suptitle(f"Session {SESSION_ID} -- independent clustering, raw channel {CHAN}")
    plt.tight_layout()
    fig_path = rf"{OUT}\independent_clustering_{SESSION_ID}_ch{CHAN}.png"
    plt.savefig(fig_path, dpi=120)
    print(f"\nsaved {fig_path}")
    for k in range(best_k):
        m = labels == k
        print(f"cluster {k}: n={m.sum()}, rate={m.sum()/(n_samples/FS):.2f}Hz")
    print(f"\nTOTAL_WALL_TIME_S={time.time() - t_wall_start:.1f}")


if __name__ == "__main__":
    main()
