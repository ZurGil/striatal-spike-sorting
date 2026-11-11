# -*- coding: utf-8 -*-
"""Save each unit's raw mean waveform, in two halves, for cross-session matching.

    python extract_unit_waveforms.py --ks-dir <kilosort4 out> --data <session.probe1.dat> \
        --out <dir> [--n-spikes 500] [--window 82] [--pre 31] [--no-car]

WHY THIS RUNS ON THE CLUSTER, IN THE SAME JOB AS KILOSORT
---------------------------------------------------------
Following a neuron from one session to the next is done on the shape of its average
spike as seen across all the sites it reaches -- a small picture of the cell on the
probe. Producing that picture needs the RAW data, which exists on cluster scratch only
while the sort is running. Doing it here costs minutes and ~120 MB; doing it afterwards
costs a ~250 GB re-upload per session. So it is not a separate job, and it is not
optional.

Kilosort's own `templates.npy` is NOT a substitute. Those templates live in whitened,
drift-corrected space and are fit from a shared basis of universal templates, so two
different neurons can end up with suspiciously similar templates and a single neuron's
template is not its measured average waveform.

WHY TWO HALVES
--------------
This is the whole reason the method can be calibrated instead of argued about. Split a
unit's spikes in two and you have two average waveforms from the SAME neuron, by
construction. The distribution of similarity over those pairs is the within-neuron
distribution; pairs of different units in the same session give the between-neuron
distribution; the decision threshold and both error rates follow from the overlap. No
judgement, and no need to believe anybody's published threshold.

The split is by TIME -- first half of the unit's spikes against the second half -- and
not by alternating spikes. Alternating spikes would give a flattering answer, because
both halves then span the same drift. Splitting by time lets within-session drift into
the within-neuron distribution, which is the conservative choice and matches what the
across-session comparison will actually face. (UnitMatch splits the same way.)

WHAT IS AND IS NOT DONE TO THE SNIPPETS
---------------------------------------
Per snippet, per channel: the baseline (mean of the first `--baseline` samples) is
subtracted, so slow LFP offsets do not contaminate the amplitude pattern. Then, unless
`--no-car` is given, the median across channels is subtracted per time sample -- the
same common-average reference Kilosort ran with (`do_CAR=true`), so the footprints are
comparable to the sort that produced the spike times. No high-pass filter is applied:
filtering is a choice a matcher can make later from this file, but it cannot be undone
if baked in here. Everything that was done is recorded in the output metadata.

OUTPUT (<out>/unit_waveforms.npz, plus a small JSON twin for provenance)
    unit_ids      (n_units,)              int64, the cluster ids
    mean_wf       (n_units, 2, nt, nchan) float32, microvolts; [:,0] = first half
    n_spikes      (n_units, 2)            int32, spikes averaged into each half
    peak_chan     (n_units,)              int32, from the full-unit mean
    centroid      (n_units, 2, 2)         float32, (x, y) per half, amplitude-weighted
    channel_pos   (nchan, 2)              float32, copied from the sort
A unit with fewer than `--min-spikes` in either half is still written, with whatever
spikes it has and the true counts, because "too few spikes to trust" is a judgement for
the matcher to make with the counts in hand -- not something to silently drop here.
"""
import argparse
import json
import os
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks-dir", required=True, help="Kilosort4 output directory")
    ap.add_argument("--data", required=True, help="raw flat int16 <session>.probe1.dat")
    ap.add_argument("--out", required=True, help="output directory (created)")
    ap.add_argument("--n-chan", type=int, default=384)
    ap.add_argument("--fs", type=float, default=30000.0)
    ap.add_argument("--gain-to-uv", type=float, default=0.018311105685598315,
                    help="spikeScalingToUv from THIS session's .rec header -- verify it, "
                         "do not inherit it from another session")
    ap.add_argument("--n-spikes", type=int, default=500,
                    help="max spikes averaged per half; bounds the I/O, 500 is already "
                         "far past where the mean stops moving")
    ap.add_argument("--window", type=int, default=82, help="samples per snippet")
    ap.add_argument("--pre", type=int, default=31, help="samples before the spike time")
    ap.add_argument("--baseline", type=int, default=12,
                    help="leading samples used as the per-channel baseline")
    ap.add_argument("--min-spikes", type=int, default=20,
                    help="units with fewer spikes in a half are flagged, not dropped")
    ap.add_argument("--no-car", action="store_true",
                    help="skip the median-across-channels subtraction")
    args = ap.parse_args()

    nt, pre, nchan = args.window, args.pre, args.n_chan
    post = nt - pre

    st = np.load(os.path.join(args.ks_dir, "spike_times.npy")).ravel().astype(np.int64)
    sc = np.load(os.path.join(args.ks_dir, "spike_clusters.npy")).ravel().astype(np.int64)
    if st.size != sc.size:
        raise SystemExit("spike_times (%d) and spike_clusters (%d) disagree" % (st.size, sc.size))

    chan_pos = np.load(os.path.join(args.ks_dir, "channel_positions.npy")).astype(np.float32)
    if chan_pos.shape[0] != nchan:
        raise SystemExit("channel_positions has %d rows but --n-chan is %d"
                         % (chan_pos.shape[0], nchan))

    nbytes = os.path.getsize(args.data)
    if nbytes % (2 * nchan):
        raise SystemExit("%s is not a whole number of %d-channel int16 samples"
                         % (args.data, nchan))
    n_samples = nbytes // (2 * nchan)
    print("raw %s" % args.data)
    print("    %d bytes = %d samples x %d channels = %.3f hours"
          % (nbytes, n_samples, nchan, n_samples / args.fs / 3600.0))

    # memmap, not read: we touch a few tens of GB of scattered snippets out of 250 GB
    raw = np.memmap(args.data, dtype=np.int16, mode="r", shape=(n_samples, nchan))

    units = np.unique(sc)
    print("%d units, %d spikes total" % (units.size, st.size))

    mean_wf = np.zeros((units.size, 2, nt, nchan), dtype=np.float32)
    n_used = np.zeros((units.size, 2), dtype=np.int32)
    peak_chan = np.zeros(units.size, dtype=np.int32)
    centroid = np.full((units.size, 2, 2), np.nan, dtype=np.float32)
    thin = []

    rng = np.random.default_rng(0)  # fixed, so a re-run reproduces the same file

    for i, uid in enumerate(units):
        t = st[sc == uid]
        # Spikes too near either edge cannot give a whole snippet. Dropping them is
        # correct and must happen BEFORE the halves are cut, or the two halves end up
        # drawn from different stretches of the recording.
        t = t[(t >= pre) & (t + post <= n_samples)]
        if t.size == 0:
            thin.append((int(uid), 0, 0))
            continue
        t.sort()
        halves = np.array_split(t, 2)

        for h, th in enumerate(halves):
            if th.size == 0:
                continue
            take = th if th.size <= args.n_spikes else rng.choice(th, args.n_spikes, replace=False)
            take = np.sort(take)
            acc = np.zeros((nt, nchan), dtype=np.float64)
            for s in take:
                snip = raw[s - pre:s - pre + nt, :].astype(np.float32)
                snip -= snip[:args.baseline, :].mean(axis=0, keepdims=True)
                if not args.no_car:
                    snip -= np.median(snip, axis=1, keepdims=True)
                acc += snip
            acc /= float(take.size)
            mean_wf[i, h] = (acc * args.gain_to_uv).astype(np.float32)
            n_used[i, h] = take.size

        if n_used[i].min() < args.min_spikes:
            thin.append((int(uid), int(n_used[i, 0]), int(n_used[i, 1])))

        full = mean_wf[i].mean(axis=0)                       # nt x nchan, in uV
        amp = full.max(axis=0) - full.min(axis=0)            # peak-to-peak per channel
        peak_chan[i] = int(np.argmax(amp))
        for h in range(2):
            a = mean_wf[i, h].max(axis=0) - mean_wf[i, h].min(axis=0)
            # Weight by amplitude above a floor, so the centroid is set by the sites the
            # cell actually reaches and not dragged to the probe's middle by 300 silent
            # channels each contributing a little noise.
            w = np.clip(a - 0.3 * a.max(), 0, None)
            if w.sum() > 0:
                centroid[i, h] = (chan_pos * w[:, None]).sum(axis=0) / w.sum()

        if (i + 1) % 25 == 0 or i + 1 == units.size:
            print("  %d/%d units" % (i + 1, units.size), flush=True)

    os.makedirs(args.out, exist_ok=True)
    npz = os.path.join(args.out, "unit_waveforms.npz")
    np.savez_compressed(
        npz,
        unit_ids=units.astype(np.int64),
        mean_wf=mean_wf,
        n_spikes=n_used,
        peak_chan=peak_chan,
        centroid=centroid,
        channel_pos=chan_pos,
    )

    meta = {
        "data": args.data,
        "ks_dir": args.ks_dir,
        "n_units": int(units.size),
        "n_chan": nchan,
        "fs": args.fs,
        "gain_to_uv": args.gain_to_uv,
        "window_samples": nt,
        "pre_samples": pre,
        "baseline_samples": args.baseline,
        "max_spikes_per_half": args.n_spikes,
        "split": "by time: first half of each unit's spikes vs second half",
        "car_applied": not args.no_car,
        "highpass_applied": False,
        "units_in_uV": True,
        "npz_bytes": os.path.getsize(npz),
        "units_below_min_spikes": thin,
    }
    with open(os.path.join(args.out, "unit_waveforms.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("wrote %s (%.1f MB)" % (npz, os.path.getsize(npz) / 1e6))
    if thin:
        print("%d unit(s) with fewer than %d spikes in a half -- written with true counts:"
              % (len(thin), args.min_spikes))
        for uid, a, b in thin[:15]:
            print("   unit %-6d halves %d / %d" % (uid, a, b))


if __name__ == "__main__":
    main()
