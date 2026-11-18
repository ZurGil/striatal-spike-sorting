# -*- coding: utf-8 -*-
"""
Point clouds for the raw-data scatter, split into time windows and labelled per unit.

WHAT THIS REPLACES, AND WHY
---------------------------
raw_cluster_points_session.py detects events on each unit's peak channel in five
120 s chunks, KEEPS THE ONE CHUNK with the most of that unit's own spikes, throws the
other four away, and labels every point as own / other / noise -- "other" pooled, with
no record of which unit each point came from. So the panel shows one 2-minute window
of a two-hour session, a different window for each unit, and a green cloud that cannot
be read.

Three changes, each one needed by something the panel is being asked to do:

  EVERY WINDOW IS KEPT, not just the best one, so the panel can offer one checkbox per
  window: tick more to add points, untick to look at one stretch of the session.

  EVERY POINT CARRIES THE UNIT IT MATCHED, so each neighbour gets its own colour and
  its own legend entry instead of all of them being one green.

  ONE PCA BASIS PER CHANNEL, fitted once and stored, instead of a fresh SVD per chunk.
  This one is not cosmetic. Measured on 20260922_144404, the per-chunk PC1 of two
  windows on the same channel gave dot products of +0.81, -0.13, -0.99 and +0.99 --
  the axis flips sign and sometimes turns into a different axis altogether. Per-chunk
  PCA means two windows are not drawn in the same coordinates, so overlaying them
  would be meaningless and a region drawn on one could not be applied to another.

The basis goes in its own file (cluster_basis_<session>.json) rather than the page
payload: nothing in the page needs it, and the step that later applies a hand-drawn
region to every spike does, so it has to be reproducible rather than redrawn.

HOW THE RAW DATA IS READ, which is the whole cost
-------------------------------------------------
probe1.dat is sample-major: one channel's 120 s costs the same read as all 384
channels' 120 s, because the stride pulls the span either way. Measured on Z: 4.1 s
for one channel of 20 s, 4.5 s for all 384 of the same 20 s. So this reads
WINDOW-MAJOR -- each window's span is read once, in 10 s slices, and every channel is
detected from it -- rather than the old channel-major loop, which paid for the same
span once per channel. That is the difference between ~3 hours and ~8 minutes per
session.

Two passes, because the basis has to exist before anything can be projected onto it:
pass A reads two windows (the first and the middle) to fit each channel's basis, pass
B reads every window and builds the records.

Detection is otherwise unchanged -- 4.5 MAD on the high-passed channel, 1 ms dead
time, 60-sample snippets starting 20 before the crossing. One deliberate difference:
the noise level is now estimated per 10 s slice instead of per 120 s chunk, which is a
more local estimate of the noise a spike was actually detected against. On 600k
samples of striatal background the two agree closely.

Usage:
  python build_cluster_windows.py <session_id> [--rec-root Z:/Gil/Shamir_1]
                                  [--windows 10] [--window-sec 120]
                                  [--cap-own 60] [--cap-other 90] [--cap-noise 50]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from striatal_agent.config import AgentConfig          # noqa: E402
from striatal_agent.session import discover_session    # noqa: E402
from striatal_agent.raw_io import RawReader, highpass_filter  # noqa: E402
from striatal_agent.paths import out_path              # noqa: E402

FS = 30000
NT, NT0MIN = 60, 20
DEAD_SAMPLES = int(0.001 * FS)
MATCH_WINDOW = int(0.0005 * FS)
NOISE = -1                      # not 0: unit 0 is a real unit
FIELDS = ("pc1", "pc2", "ptp", "teo_max", "psi_trough", "teo2")
SLICE_SEC = 10                  # how much of a window is held in memory at once
PAD = 200                       # filter padding, trimmed off after filtering
BASIS_SNIPS_PER_WINDOW = 1500   # cap on what one window contributes to a channel's SVD

T0 = time.time()


def log(msg):
    print("[%6.1fs] %s" % (time.time() - T0, msg), flush=True)


def detect_trace(trace, start_abs):
    """Threshold crossings in one filtered trace, as raw_cluster_points_session does."""
    mad = np.median(np.abs(trace - np.median(trace))) * 1.4826
    if not np.isfinite(mad) or mad <= 0:
        return None
    below = trace < -4.5 * mad
    crossings = np.where(below & ~np.roll(below, 1))[0]
    keep, last = [], -DEAD_SAMPLES
    for c in crossings:
        if c - last >= DEAD_SAMPLES:
            keep.append(c)
            last = c
    crossings = np.array(keep, dtype=np.int64)
    crossings = crossings[(crossings > NT0MIN + 2)
                          & (crossings < len(trace) - (NT - NT0MIN) - 2)]
    if len(crossings) == 0:
        return None
    snips = np.array([trace[c - NT0MIN:c - NT0MIN + NT] for c in crossings],
                     dtype=np.float32)
    return crossings + start_abs, snips


def scan_window(reader, cfg, channels, start, dur, n_samples):
    """Detect on every channel across one window, reading the span once per slice.

    Returns {channel: (times, snippets)} with the window's detections concatenated
    across slices. Crossings within a snippet's length of a slice edge are dropped,
    the same way the original dropped them at a chunk edge -- a handful of spikes per
    window boundary, on a cloud that is subsampled anyway.
    """
    acc_t = dict((ch, []) for ch in channels)
    acc_s = dict((ch, []) for ch in channels)
    step = SLICE_SEC * FS
    s = start
    while s < start + dur:
        s1 = min(start + dur, s + step)
        a, b = max(0, s - PAD), min(n_samples, s1 + PAD)
        # The memmap slice straight from the reader: one sequential span, converted a
        # channel at a time so the float32 copy is 2.4 MB rather than 1.8 GB.
        blk = np.asarray(reader._arr[a:b, :])
        off = s - a
        for ch in channels:
            filt = highpass_filter(blk[:, ch].astype(np.float32), FS, cfg.highpass_hz,
                                   axis=0)
            got = detect_trace(filt[off:off + (s1 - s)], s)
            if got is None:
                continue
            acc_t[ch].append(got[0])
            acc_s[ch].append(got[1])
        del blk
        s = s1
    out = {}
    for ch in channels:
        if acc_t[ch]:
            out[ch] = (np.concatenate(acc_t[ch]), np.vstack(acc_s[ch]))
    return out


def shape_features(snips):
    """The four basis-free axes. Absolute, so they are comparable across windows."""
    ptp = snips.max(axis=1) - snips.min(axis=1)
    teo_max = np.array([(s[1:-1] ** 2 - s[:-2] * s[2:]).max() for s in snips])
    trough = np.clip(np.argmin(snips, axis=1), 1, NT - 2)
    psi_trough = np.array([s[i] ** 2 - s[i - 1] * s[i + 1]
                           for s, i in zip(snips, trough)])
    teo2 = np.array([(s[2:-2] ** 2 - s[:-4] * s[4:]).max() for s in snips])
    return ptp, teo_max, psi_trough, teo2


def fit_basis(snip_pool):
    """One PCA basis for a channel, with a sign convention so it is reproducible.

    SVD fixes the axes only up to sign, and numpy will hand back -v just as readily
    tomorrow. The convention: make each component negative at the mean snippet's
    trough sample. The trough is the one feature every spike on this channel shares,
    so it is a stable thing to orient against.
    """
    mean = snip_pool.mean(axis=0)
    _, _, Vt = np.linalg.svd(snip_pool - mean, full_matrices=False)
    basis = Vt[:2].copy()
    trough = int(np.argmin(mean))
    for k in range(2):
        if basis[k][trough] > 0:
            basis[k] = -basis[k]
    return basis, mean, trough


def fmt(v):
    """Compact enough to publish, precise enough for a 300 px scatter.

    The Teager axes run to 1e9 and would spend 11 characters each on precision no
    screen can show; they are stored in thousands (the axis label says so). Everything
    else is in the thousands already, so an integer is well under a pixel.
    """
    v = float(v)
    return int(round(v)) if abs(v) >= 1000 else round(v, 1)


TEO_SCALE = 1000.0
SCALED = ("teo_max", "psi_trough", "teo2")


def pack(values, field):
    if field in SCALED:
        return [fmt(v / TEO_SCALE) for v in values]
    return [fmt(v) for v in values]


def subsample(mask, rng, cap):
    idx = np.where(mask)[0]
    if len(idx) > cap:
        idx = np.sort(rng.choice(idx, size=cap, replace=False))
    return idx


def match_units(times, spike_times, spike_clusters, lo, hi):
    """Which sorted unit each detection is, within half a millisecond, or NOISE."""
    who = np.full(len(times), NOISE, dtype=np.int64)
    j0, j1 = np.searchsorted(spike_times, [lo, hi])
    st, sc = spike_times[j0:j1], spike_clusters[j0:j1]
    if not len(st):
        return who
    pos = np.searchsorted(st, times)
    for k, t in enumerate(times):
        best, bestd = NOISE, MATCH_WINDOW + 1
        for m in range(max(0, pos[k] - 2), min(len(st), pos[k] + 2)):
            dt = abs(int(st[m]) - int(t))
            if dt <= MATCH_WINDOW and dt < bestd:
                best, bestd = int(sc[m]), dt
        who[k] = best
    return who


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session_id")
    ap.add_argument("--rec-root", default=r"Z:\Gil\Shamir_1")
    ap.add_argument("--windows", type=int, default=10)
    ap.add_argument("--window-sec", type=float, default=120.0)
    ap.add_argument("--cap-own", type=int, default=80)
    ap.add_argument("--cap-other", type=int, default=70)
    ap.add_argument("--cap-noise", type=int, default=40)
    ap.add_argument("--max-channels", type=int, default=0,
                    help="stop after this many channels -- for checking the format "
                         "without paying for a whole session")
    ap.add_argument("--suffix", default="",
                    help="appended to the output names, so a trial run cannot "
                         "overwrite a real one")
    args = ap.parse_args()

    sid = args.session_id
    sfx = args.suffix
    root = os.path.join(args.rec_root, sid + ".rec")
    out_points = out_path("cluster_windows_%s%s.json" % (sid, sfx))
    out_basis = out_path("cluster_basis_%s%s.json" % (sid, sfx))
    progress = out_path("cluster_windows_%s%s_progress.txt" % (sid, sfx))

    cfg = AgentConfig()
    paths = discover_session(root)
    ks = paths.kilosort_original_dir
    log("clustering from %s" % ks)
    spike_times = np.load(os.path.join(ks, "spike_times.npy")).ravel().astype(np.int64)
    spike_clusters = np.load(os.path.join(ks, "spike_clusters.npy")).ravel()
    order = np.argsort(spike_times)
    spike_times, spike_clusters = spike_times[order], spike_clusters[order]

    with open(out_path("corrected_classification_%s_with_waveforms.json" % sid),
              encoding="utf-8") as f:
        rows = json.load(f)
    peak_of = dict((u["unit_id"], int(u["peak_channel"])) for u in rows)
    units_on = {}
    for uid, ch in peak_of.items():
        units_on.setdefault(ch, []).append(uid)
    log("%d units on %d distinct peak channels (%.2f per channel)"
        % (len(peak_of), len(units_on), len(peak_of) / float(len(units_on))))

    reader = RawReader(paths.probe_dat, cfg.n_chan)
    dur = int(args.window_sec * FS)
    starts = np.linspace(0, reader.n_samples - dur, args.windows).astype(np.int64)
    log("probe1.dat %.1f min, %d windows of %.0fs at %s min"
        % (reader.n_samples / FS / 60.0, args.windows, args.window_sec,
           ", ".join("%.0f" % (s / FS / 60.0) for s in starts)))

    windows_meta = [{"i": i, "start_sample": int(s),
                     "t0_s": round(float(s) / FS, 1),
                     "t1_s": round(float(s + dur) / FS, 1)}
                    for i, s in enumerate(starts)]

    channels = sorted(units_on)
    if args.max_channels:
        channels = channels[:args.max_channels]
        log("TRIAL RUN: first %d channels only" % len(channels))

    rng = np.random.default_rng(0)

    # ---------------------------------------------------- pass A: one basis per channel
    fit_windows = sorted({0, len(starts) // 2})
    pool = dict((ch, []) for ch in channels)
    for wi in fit_windows:
        log("pass A: window %d for the PCA basis" % wi)
        got = scan_window(reader, cfg, channels, int(starts[wi]), dur, reader.n_samples)
        for ch, (_, snips) in got.items():
            take = min(BASIS_SNIPS_PER_WINDOW, len(snips))
            pool[ch].append(snips[np.sort(rng.choice(len(snips), take, replace=False))])

    basis_out, basis = {}, {}
    for ch in channels:
        if not pool[ch]:
            continue
        stack = np.vstack(pool[ch])
        if len(stack) < 20:
            continue
        b, mean, trough = fit_basis(stack)
        basis[ch] = (b, mean)
        basis_out[str(ch)] = {
            "vt": [[round(float(v), 6) for v in comp] for comp in b],
            "mean_snippet": [round(float(v), 4) for v in mean],
            "trough_sample": trough,
            "n_snippets_fitted": int(len(stack)),
        }
    pool = None
    log("pass A done: a basis for %d of %d channels" % (len(basis), len(channels)))

    # ------------------------------------------------------------ pass B: the records
    units_out = dict((str(u), {"peak_channel": peak_of[u], "windows": {}})
                     for u in peak_of if peak_of[u] in channels)
    for wi, start in enumerate(starts):
        got = scan_window(reader, cfg, channels, int(start), dur, reader.n_samples)
        lo, hi = int(start), int(start) + dur
        n_pts = 0
        for ch, (times, snips) in got.items():
            if ch not in basis:
                continue
            b, mean = basis[ch]
            centred = snips - mean
            feats = {"pc1": centred @ b[0], "pc2": centred @ b[1]}
            (feats["ptp"], feats["teo_max"], feats["psi_trough"],
             feats["teo2"]) = shape_features(snips)
            who = match_units(times, spike_times, spike_clusters, lo, hi)

            # "Other" means a unit that PEAKS ON THIS CHANNEL, as it always has. A
            # detection matching a unit 500 um away -- which happens constantly, a
            # distant spike still crosses threshold here -- is background, and counting
            # it as a neighbour would put 58 units in a legend meant to name the two or
            # three clusters actually sitting on this channel.
            co = set(units_on[ch])
            for uid in units_on[ch]:
                own_m = who == uid
                other_m = np.array([w in co and w != uid for w in who], dtype=bool)
                noise_m = ~own_m & ~other_m
                i_own = subsample(own_m, rng, args.cap_own)
                i_other = subsample(other_m, rng, args.cap_other)
                i_noise = subsample(noise_m, rng, args.cap_noise)
                if not (len(i_own) or len(i_other) or len(i_noise)):
                    continue
                n_pts += len(i_own) + len(i_other) + len(i_noise)
                units_out[str(uid)]["windows"][str(wi)] = {
                    "own": dict((f, pack(feats[f][i_own], f)) for f in FIELDS),
                    "other": dict((f, pack(feats[f][i_other], f)) for f in FIELDS),
                    "noise": dict((f, pack(feats[f][i_noise], f)) for f in FIELDS),
                    # Which neighbour each coloured point belongs to -- the whole
                    # reason the neighbours can stop being one undifferentiated green.
                    "other_unit": [int(x) for x in who[i_other]],
                    # Totals BEFORE subsampling, so the panel can say how many spikes a
                    # hand-drawn region would really take, not how many dots it covered.
                    "n_own": int(own_m.sum()),
                    "n_other": int(other_m.sum()),
                    "n_noise": int(noise_m.sum()),
                }
        el = time.time() - T0
        msg = ("window %d/%d done (%d channels, %d points kept), %.0fs elapsed, "
               "~%.0fs left" % (wi + 1, len(starts), len(got), n_pts, el,
                                el / (wi + 1) * (len(starts) - wi - 1)))
        log(msg)
        with open(progress, "w") as f:
            f.write(msg + "\n")

    doc = {
        "format": "cluster_windows/2",
        "session_id": sid,
        "built": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "fs_hz": FS,
        "window_sec": args.window_sec,
        "windows": windows_meta,
        "fields": list(FIELDS),
        "scaled_fields": {f: TEO_SCALE for f in SCALED},
        "caps": {"own": args.cap_own, "other": args.cap_other, "noise": args.cap_noise},
        "detection": {"threshold_mad": 4.5, "dead_ms": 1.0, "nt": NT, "nt0min": NT0MIN,
                      "match_window_ms": 1000.0 * MATCH_WINDOW / FS,
                      "noise_estimated_over_s": SLICE_SEC},
        "n_units": len(units_out),
        "n_channels": len(basis_out),
        "units": units_out,
    }
    with open(out_points, "w", encoding="utf-8") as f:
        json.dump(doc, f, separators=(",", ":"))
    with open(out_basis, "w", encoding="utf-8") as f:
        json.dump({"format": "cluster_basis/1", "session_id": sid,
                   "built": doc["built"], "nt": NT, "nt0min": NT0MIN,
                   "fs_hz": FS, "windows": windows_meta,
                   "detection": doc["detection"], "channels": basis_out}, f,
                  separators=(",", ":"))

    with_pts = sum(1 for v in units_out.values() if v["windows"])
    log("%d of %d units have points" % (with_pts, len(units_out)))
    log("wrote %s (%.1f MiB) and %s (%.1f MiB)"
        % (os.path.basename(out_points), os.path.getsize(out_points) / 1048576.0,
           os.path.basename(out_basis), os.path.getsize(out_basis) / 1048576.0))
    with open(progress, "w") as f:
        f.write("DONE: %d units, %d channels, %.0fs\n"
                % (with_pts, len(basis_out), time.time() - T0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
