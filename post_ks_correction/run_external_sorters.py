"""Run third-party sorters on our benchmark, for the first external reference.

Every sorting number in this project so far compares Kilosort to our own
modifications of Kilosort. There has never been an outside reference point, so
"Kilosort gets 0.882 recall on these placements" has had nothing to be
calibrated against -- we do not know whether that is good, average or poor for
this data.

The si_ms5 environment on D: has five installed sorters, and two of them
matter especially:

  spykingcircus2   mature iterative template matching WITH DECONVOLUTION
  lupin            also template-matching based
  mountainsort5    ISO-SPLIT clustering, a genuinely different approach:
                   non-parametric, no k, splits only where a projection fails
                   a unimodality test
  tridesclous2     template matching with a different feature pipeline
  simple           a deliberately minimal reference

The spykingcircus2 / lupin results answer, empirically and cheaply, the
question that would otherwise take a few hundred lines of our own
deconvolution code to answer: DOES iterative template subtraction actually beat
Kilosort on THIS recording? 5ax measured that subtraction is the one component
LOCDEN lacks, but if a mature implementation of it does not beat Kilosort
here, then building our own will not either.

Output goes to hybrid_v2_rep{r}/ks_{name}/ as spike_times.npy and
spike_clusters.npy, so run_v2_comparison.score() scores these by the identical
rule as every Kilosort configuration -- same tolerance, same fragment
definition, same placements.

ALIGNMENT IS CHECKED, NOT ASSUMED. Truth times are trough samples and
Kilosort agrees to a median of 0 and +/-1 sample at 5-95%. A sorter that
reports peak-aligned or template-start-aligned times would fail the 10-sample
matcher for a bookkeeping reason and look far worse than it is, so the median
offset is measured and reported for each sorter. It is NOT silently corrected:
a correction would be a thumb on the scale, so it is reported and left to a
deliberate decision.

Usage:
  python run_external_sorters.py <rep> <sorter> [<sorter> ...]
  python run_external_sorters.py --check <rep> <name>
"""
import os
import sys
import json
import time
import shutil

import numpy as np

ROOT = r"D:\Gil\spike_sorting_agent"
SCRATCH = (r"C:\Users\Adam\AppData\Local\Temp\claude\D--Gil-spike-sorting-agent"
           r"\8186c809-76fa-41ff-bbca-59d44125e5fc\scratchpad\si_work")
FS = 30000.0
N_CHAN_BIN = 384
GAIN_TO_UV = 0.018311105685598315

# Defaults left alone wherever possible: the point is an external reference,
# not a tuned competitor. Only what the data format forces is set.
PARAMS = {
    "mountainsort5": dict(),
    "spykingcircus2": dict(),
    "tridesclous2": dict(),
    "lupin": dict(),
    "simple": dict(),
}

# Aliases: a tuned run of an installed sorter, written to its own ks_ folder.
# The defaults run stays untouched so both are reportable -- "MountainSort5 at
# defaults" and "MountainSort5 given a fair shot" are different claims and
# conflating them would be unfair to the sorter.
ALIAS = {
    # MS5's defaults contain the two settings this project has MEASURED to be
    # wrong for this data:
    #   detect_threshold 5.5 -- the injected units average a 3.06 sigma
    #     spike-triggered-average trough on their own best channel (5ax), and
    #     only 11.7% reach 5 sigma, so 5.5 cannot see most of them
    #   detect_time_radius_msec 0.5 -- a 15-sample exclusion radius, which is
    #     exactly the occlusion trap that pinned our own detector's recall at
    #     0.476 regardless of threshold until the footprint was shrunk to
    #     +/-3 samples (5ax)
    "ms5_tuned": ("mountainsort5",
                  dict(detect_threshold=4.0, detect_time_radius_msec=0.2,
                       scheme2_detect_channel_radius=75)),
    "ms5_sensitive": ("mountainsort5",
                      dict(detect_threshold=3.5, detect_time_radius_msec=0.15,
                           scheme2_detect_channel_radius=75)),
    # SpyKING CIRCUS 2 defaults to motion correction, whose dredge method
    # needs torch, which si_ms5 does not have -- that is the only reason the
    # first run failed. Disabling it is also the MATCHED setting: every
    # Kilosort run in this project uses nblocks=0, i.e. no drift correction,
    # and the benchmark is a single 120 s window where drift is negligible.
    "sc2": ("spykingcircus2", dict(apply_motion_correction=False)),
    "lupin_": ("lupin", dict(apply_motion_correction=False)),
}


def data_dir(rep):
    return os.path.join(ROOT, "hybrid_v2_rep%d" % rep)


def build_recording(rep):
    import spikeinterface.extractors as se
    import spikeinterface.preprocessing as sp
    from probeinterface import Probe

    with open(os.path.join(data_dir(rep), "probe.json")) as f:
        p = json.load(f)
    cm = np.asarray(p["chanMap"], dtype=int)
    xc = np.asarray(p["xc"], dtype=float)
    yc = np.asarray(p["yc"], dtype=float)

    rec = se.read_binary(
        os.path.join(data_dir(rep), "hybrid.bin"), sampling_frequency=FS,
        dtype="int16", num_channels=N_CHAN_BIN, gain_to_uV=GAIN_TO_UV,
        offset_to_uV=0.0)

    # keep only the mapped channels, in the file's own channel order, so a
    # returned unit's channel index still means the same physical site
    ids = rec.channel_ids
    keep = [ids[c] for c in cm]
    rec = (rec.select_channels(keep) if hasattr(rec, "select_channels")
           else rec.channel_slice(keep))

    pr = Probe(ndim=2, si_units="um")
    pr.set_contacts(positions=np.column_stack([xc, yc]), shapes="square",
                    shape_params={"width": 12})
    pr.set_device_channel_indices(np.arange(len(cm)))
    rec = rec.set_probe(pr)

    rec = sp.bandpass_filter(rec, freq_min=300.0, freq_max=6000.0)
    rec = sp.common_reference(rec, reference="global", operator="median")
    return rec


def run(rep, name):
    import spikeinterface.sorters as ss

    sorter, extra = ALIAS.get(name, (name, PARAMS.get(name, {})))
    rec = build_recording(rep)
    work = os.path.join(SCRATCH, "rep%d_%s" % (rep, name))
    if os.path.isdir(work):
        shutil.rmtree(work, ignore_errors=True)
    os.makedirs(SCRATCH, exist_ok=True)

    print("[%s] starting on rep%d (%d channels, %.0f s) sorter=%s params=%s"
          % (name, rep, rec.get_num_channels(),
             rec.get_num_samples() / FS, sorter, extra), flush=True)
    t0 = time.time()
    sort = ss.run_sorter(sorter, rec, folder=work, remove_existing_folder=True,
                         verbose=True, **extra)
    dt = time.time() - t0

    st, cl = [], []
    for u in sort.unit_ids:
        s = sort.get_unit_spike_train(u)
        st.append(np.asarray(s, dtype=np.int64))
        cl.append(np.full(len(s), int(u), dtype=np.int64))
    if not st:
        print("[%s] produced NO units" % name)
        return
    st = np.concatenate(st)
    cl = np.concatenate(cl)
    o = np.argsort(st)
    _, cl = np.unique(cl[o], return_inverse=True)

    out = os.path.join(data_dir(rep), "ks_" + name)
    os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, "spike_times.npy"), st[o])
    np.save(os.path.join(out, "spike_clusters.npy"), cl.astype(np.int32))
    print("[%s] %d spikes, %d units, %.0f s  -> ks_%s"
          % (name, len(st), len(np.unique(cl)), dt, name), flush=True)
    check(rep, name)


def check(rep, name):
    """Measure the time-alignment convention against the ground truth."""
    out = os.path.join(data_dir(rep), "ks_" + name)
    st = np.load(os.path.join(out, "spike_times.npy")).ravel().astype(np.int64)
    cl = np.load(os.path.join(out, "spike_clusters.npy")).ravel()
    T = np.load(os.path.join(data_dir(rep), "hybrid_truth.npz"),
                allow_pickle=True)
    tt, lab = T["times"].ravel(), T["labels"].ravel()
    from collections import Counter
    offs = []
    for u in np.unique(lab):
        t = tt[lab == u]
        c = Counter()
        for x in t:
            c.update(cl[np.abs(st - x) <= 30].tolist())
        if not c:
            continue
        s = np.sort(st[cl == c.most_common(1)[0][0]])
        for x in t:
            j = np.searchsorted(s, x)
            k = [s[m] for m in (j - 1, j) if 0 <= m < len(s)]
            if k:
                d = min(k, key=lambda y: abs(y - x))
                if abs(d - x) <= 30:
                    offs.append(d - x)
    if not offs:
        print("  [align] no matches within 30 samples -- cannot check")
        return
    offs = np.array(offs)
    print("  [align] %s: median offset %+.1f samples, 5-95%% [%+.0f, %+.0f], "
          "n=%d" % (name, np.median(offs), np.percentile(offs, 5),
                    np.percentile(offs, 95), len(offs)))
    if abs(np.median(offs)) > 3:
        print("  [align] WARNING: median offset exceeds 3 samples. This sorter")
        print("          does not use the trough convention, so its score is")
        print("          depressed for a bookkeeping reason, not a real one.")


def realign(rep, name, out_name=None, radius=25):
    """Enforce the trough convention using ONLY the recording.

    MountainSort5 reports times a median of +14 samples (5-95%: +6 to +20)
    from the trough, which is outside the scorer's 10-sample tolerance, so its
    recall would read near zero for a pure bookkeeping reason. Correcting that
    is legitimate; doing it with the ground truth would not be.

    So this uses no truth at all. For each cluster it builds the mean filtered
    waveform on that cluster's own peak channel, finds where that template's
    trough sits relative to the reported time, and shifts every spike of the
    cluster by that one integer offset. Per cluster rather than globally,
    because the wide spread above shows the offset is not one constant -- each
    unit's template is aligned differently.
    """
    from scipy.signal import butter, sosfiltfilt

    src = os.path.join(data_dir(rep), "ks_" + name)
    st = np.load(os.path.join(src, "spike_times.npy")).ravel().astype(np.int64)
    cl = np.load(os.path.join(src, "spike_clusters.npy")).ravel()

    with open(os.path.join(data_dir(rep), "probe.json")) as f:
        cmap = np.asarray(json.load(f)["chanMap"], dtype=int)
    raw = np.memmap(os.path.join(data_dir(rep), "hybrid.bin"), dtype=np.int16,
                    mode="r").reshape(-1, N_CHAN_BIN)
    n_samples = raw.shape[0]
    sos = butter(3, [300.0, 6000.0], btype="bandpass", fs=FS, output="sos")

    NT = 2 * radius + 1
    new = st.copy()
    shifts = {}
    for c in np.unique(cl):
        s = st[cl == c]
        s = s[(s > radius + 100) & (s < n_samples - radius - 100)]
        if len(s) < 20:
            shifts[c] = 0
            continue
        pick = s[np.random.default_rng(0).permutation(len(s))[:200]]
        acc = np.zeros((NT, N_CHAN_BIN))
        for x in pick:
            seg = raw[int(x) - radius - 50:int(x) + radius + 51, :]
            seg = sosfiltfilt(sos, seg.astype(np.float64), axis=0)
            acc += seg[50:50 + NT, :]
        mw = acc / len(pick) * GAIN_TO_UV
        mw -= mw.mean(axis=0, keepdims=True)
        pk = int(np.argmax(mw.max(axis=0) - mw.min(axis=0)))
        off = int(np.argmin(mw[:, pk])) - radius
        shifts[c] = off
        new[cl == c] = st[cl == c] + off

    med = int(np.median(list(shifts.values())))
    print("  [realign] %s: per-cluster trough offsets, median %+d, "
          "range %+d..%+d over %d clusters"
          % (name, med, min(shifts.values()), max(shifts.values()),
             len(shifts)))
    keep = (new > 0) & (new < n_samples)
    out_name = out_name or (name + "_aligned")
    out = os.path.join(data_dir(rep), "ks_" + out_name)
    os.makedirs(out, exist_ok=True)
    o = np.argsort(new[keep])
    np.save(os.path.join(out, "spike_times.npy"), new[keep][o])
    np.save(os.path.join(out, "spike_clusters.npy"),
            cl[keep][o].astype(np.int32))
    print("  [realign] wrote ks_%s (%d spikes)" % (out_name, int(keep.sum())))
    check(rep, out_name)


if __name__ == "__main__":
    if sys.argv[1] == "--realign":
        realign(int(sys.argv[2]), sys.argv[3])
    elif sys.argv[1] == "--check":
        check(int(sys.argv[2]), sys.argv[3])
    else:
        rep = int(sys.argv[1])
        for nm in sys.argv[2:]:
            try:
                run(rep, nm)
            except Exception as e:
                print("[%s] FAILED: %s: %s" % (nm, type(e).__name__, e),
                      flush=True)
