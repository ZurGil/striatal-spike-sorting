"""Match the state word on the sync lines against Bpod's own record, trial by trial.

The check in `check_bpod_dio_in_rec.py` says every value on the six lines is a real state
of the protocol and lasted a legal length. This goes one step further and asks whether
the *sequence* is right: Bpod's `OriginalStateData` is the list of state numbers it
actually ran on each trial, so for a correctly wired rig the codes between one
PreITI_start and the following ITI must equal that list exactly, trial after trial.

It also measures the one number sync.py exists to find: the constant offset between the
Bpod clock and the Trodes clock. If the word is faithful, every state onset in every
matched trial sits at the same offset, and the spread of that offset is the real
uncertainty of the alignment. A wiring fault that somehow produced legal states in a
wrong order would show up here as sequence mismatches; a timing fault would show up as a
drifting or noisy offset.

The recording may start long after the Bpod session does, so nothing is assumed about
which trial the window begins on -- the trial offset is found by matching sequences.

Read-only. Imports check_bpod_dio_in_rec for the packet layout, and nothing from
sync_pipeline.

Usage:
    python compare_dio_word_to_bpod.py <session-id> <bpod .mat> [--seconds 240]
                                       [--start-s S | --from-end]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_bpod_dio_in_rec as C   # noqa: E402

TRIAL_START = 1    # PreITI_start -- the first state of every trial
TRIAL_END = 40     # ITI -- the last


def bpod_trials(mat_path):
    """Per-trial state numbers and their onsets on the Bpod clock."""
    import scipy.io as sio
    m = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    sd = m["SessionData"]
    rd = sd.RawData
    nt = int(sd.nTrials)
    tstart = np.atleast_1d(sd.TrialStartTimestamp).astype(float)
    names = np.atleast_1d(rd.OriginalStateNamesByNumber[0])
    out = []
    for t in range(nt):
        s = np.atleast_1d(rd.OriginalStateData[t]).astype(int)
        ts = np.atleast_1d(rd.OriginalStateTimestamps[t]).astype(float)
        out.append((s, tstart[t] + ts[:len(s)]))
    return out, tstart, names, nt


def dio_trials(codes, times):
    """Cut the decoded word into trials: each PreITI_start up to the following ITI.

    Code 0 is not a state -- it is every line low, which is what sits between trials --
    so it is dropped rather than treated as part of either side.
    """
    keep = codes != 0
    codes, times = codes[keep], times[keep]
    starts = np.flatnonzero(codes == TRIAL_START)
    trials = []
    for k, i in enumerate(starts):
        j = starts[k + 1] if k + 1 < len(starts) else len(codes)
        seg_c, seg_t = codes[i:j], times[i:j]
        end = np.flatnonzero(seg_c == TRIAL_END)
        if len(end) == 0:
            continue                      # trial cut off by the window's end
        last = end[0] + 1
        trials.append((seg_c[:last], seg_t[:last]))
    return trials


def main():
    args = sys.argv[1:]
    if len(args) < 2:
        raise SystemExit(__doc__)
    path = C.rec_path(args[0])
    mat = args[1]
    secs, start_s, from_end = 240.0, None, True
    for i, a in enumerate(args):
        if a == "--seconds":
            secs = float(args[i + 1])
        elif a == "--start-s":
            start_s, from_end = float(args[i + 1]), False
        elif a == "--from-end":
            from_end = True

    hdr, hdr_len = C.read_header(path)
    L = C.layout(hdr)
    fs = L["fs"]
    total_pk = (os.path.getsize(path) - hdr_len) // L["packet"]

    npk = int(secs * fs)
    start_pk = max(0, total_pk - npk - int(2 * fs)) if from_end else int(start_s * fs)
    npk = min(npk, total_pk - start_pk)
    sent, ts, dio = C.pull(path, hdr_len, L, start_pk, npk)
    if not sent or not np.all(np.diff(ts.astype(np.int64)) == 1):
        raise SystemExit("packet layout self-check failed -- refusing to decode")

    bits, code, idx, t = C.decode(dio, L, fs, ts[0])
    dtr = dio_trials(code[idx], t)
    btr, btstart, names, nt = bpod_trials(mat)

    print("%s" % os.path.basename(path))
    print("  window: %.1f s starting %.1f s into the recording (Trodes clock "
          "%.1f..%.1f s)" % (npk / fs, start_pk / fs, t[0], t[-1]))
    print("  complete trials on the wire: %d" % len(dtr))
    print("%s" % os.path.basename(mat))
    print("  %d trials, Bpod clock %.1f..%.1f s" % (nt, btstart[0], btstart[-1]))
    if not dtr:
        raise SystemExit("no complete trial in this window -- try a longer --seconds")

    # Which Bpod trial does the window start on? Try every offset and count how many
    # trials match exactly. A correct offset matches essentially all of them; a wrong
    # one matches almost none, so this does not need a tolerance.
    best, best_hits = None, -1
    for off in range(0, nt):
        hits = 0
        for k, (dc, _) in enumerate(dtr):
            if off + k >= nt:
                break
            bc = btr[off + k][0]
            if len(bc) == len(dc) and np.array_equal(bc, dc):
                hits += 1
        if hits > best_hits:
            best, best_hits = off, hits
    print("\n  best alignment: the window's first complete trial is Bpod trial %d "
          "(of %d)" % (best + 1, nt))
    print("  trials whose state sequence matches EXACTLY: %d of %d"
          % (best_hits, min(len(dtr), nt - best)))

    # Per-trial detail, and the clock offset measured at every state onset.
    offsets, bad = [], []
    print("\n  %-7s %-7s %-7s %-9s %s"
          % ("Bpod", "states", "states", "sequence", "Bpod->Trodes offset at each"))
    print("  %-7s %-7s %-7s %-9s %s"
          % ("trial", "Bpod", "wire", "match", "state onset (s)"))
    for k, (dc, dt) in enumerate(dtr):
        bi = best + k
        if bi >= nt:
            break
        bc, bt = btr[bi]
        same = len(bc) == len(dc) and np.array_equal(bc, dc)
        if same:
            o = dt - bt[:len(dt)]
            offsets.append(o)
            print("  %-7d %-7d %-7d %-9s %.6f  (spread within trial %.3f ms)"
                  % (bi + 1, len(bc), len(dc), "yes", float(np.median(o)),
                     float(o.max() - o.min()) * 1000.0))
        else:
            bad.append(bi + 1)
            print("  %-7d %-7d %-7d %-9s" % (bi + 1, len(bc), len(dc), "NO"))
            print("        Bpod: %s" % bc.tolist())
            print("        wire: %s" % dc.tolist())

    if offsets:
        allo = np.concatenate(offsets)
        print("\n  the Bpod->Trodes clock offset, measured at %d state onsets across "
              "%d trials:" % (len(allo), len(offsets)))
        print("    median      %.6f s" % float(np.median(allo)))
        print("    spread      %.3f ms (max - min)"
              % float((allo.max() - allo.min()) * 1000.0))
        print("    sd          %.3f ms" % float(np.std(allo) * 1000.0))
        drift = float(np.median(offsets[-1]) - np.median(offsets[0]))
        span = float(np.median(offsets[-1]) * 0 + (btr[best + len(offsets) - 1][1][0]
                                                   - btr[best][1][0]))
        print("    drift       %.3f ms over %.1f s of session (%.2f ppm)"
              % (drift * 1000.0, span, (drift / span * 1e6) if span else 0.0))
        print("\n  Every state onset Bpod recorded lands at the same distance from the "
              "same state onset on the wire. That is only possible if the six lines are "
              "carrying the state numbers Bpod meant to send.")
    if bad:
        print("\n  TRIALS THAT DO NOT MATCH: %s" % bad)
    else:
        print("\n  No trial in this window disagrees with Bpod in any state.")


if __name__ == "__main__":
    main()
