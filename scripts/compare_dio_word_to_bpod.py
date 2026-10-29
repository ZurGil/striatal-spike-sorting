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

    # Which Bpod trial is which trial on the wire? Matching state sequences alone is not
    # enough: a short generic trial like [1, 2, 3, 40] occurs dozens of times in a
    # session, so counting exact matches can settle on an alignment that is off by
    # several trials and still "match" the short ones. The test that cannot be fooled is
    # timing -- in a true pairing, every state onset in the trial sits at the SAME
    # distance from Bpod's own onset, so the offset's spread within the trial is a few
    # tenths of a millisecond. A coincidental pairing is out by hundreds.
    MAX_SPREAD_MS = 5.0
    pair = []
    for k, (dc, dt) in enumerate(dtr):
        cands = []
        for bi in range(nt):
            bc, bt = btr[bi]
            if len(bc) != len(dc) or not np.array_equal(bc, dc):
                continue
            o = dt - bt[:len(dt)]
            spread = float(o.max() - o.min()) * 1000.0
            if spread <= MAX_SPREAD_MS:
                cands.append((spread, bi, o))
        cands.sort()
        pair.append((k, cands, dc, dt))

    # Within-trial agreement still is not enough on its own. A bare [1, 2, 3, 40] trial
    # -- light on, wait, time out -- has the SAME internal timing in every such trial,
    # because every state in it is a fixed duration, so it agrees perfectly with dozens
    # of them. Those trials carry no identity of their own and can only be placed by the
    # clock: the offset is shared by the whole session and drifts slowly, so a trial
    # whose offset disagrees with its neighbours' by seconds is not that trial.
    informative = [c[0][2] for _, c, dc, _ in pair if len(c) == 1 and len(dc) >= 8]
    if not informative:
        informative = [c[0][2] for _, c, _, _ in pair if c]
    consensus = float(np.median([np.median(o) for o in informative])) if informative \
        else None
    MAX_FROM_CONSENSUS_S = 2.0

    resolved = []
    for k, cands, dc, dt in pair:
        pick = None
        if cands and consensus is not None:
            near = [c for c in cands if abs(float(np.median(c[2])) - consensus)
                    <= MAX_FROM_CONSENSUS_S]
            near.sort(key=lambda c: abs(float(np.median(c[2])) - consensus))
            pick = near[0] if near else None
        elif cands:
            pick = cands[0]
        resolved.append((k, pick, dc, dt))
    pair = resolved

    matched = [(k, c) for k, c, _, _ in pair if c is not None]
    print("\n  pairing each trial on the wire to a Bpod trial by sequence AND by timing")
    print("  (a pairing counts only if every state onset in it agrees to within %g ms)"
          % MAX_SPREAD_MS)
    if matched:
        idxs = [c[1] for _, c in matched]
        print("  paired %d of %d complete trials, Bpod trials %d..%d of %d"
              % (len(matched), len(dtr), min(idxs) + 1, max(idxs) + 1, nt))
        steps = np.diff(idxs)
        print("  consecutive on the wire and consecutive in Bpod: %s"
              % ("yes" if np.all(steps == 1) else "no, index steps %s" % steps.tolist()))

    offsets, bad = [], []
    print("\n  %-7s %-7s %-7s %-9s %s"
          % ("Bpod", "states", "states", "paired", "Bpod->Trodes offset at each"))
    print("  %-7s %-7s %-7s %-9s %s"
          % ("trial", "Bpod", "wire", "", "state onset (s)"))
    for k, c, dc, dt in pair:
        if c is None:
            bad.append(k)
            print("  %-7s %-7s %-7d %-9s" % ("?", "-", len(dc), "NO"))
            print("        wire: %s" % dc.tolist())
            continue
        spread, bi, o = c
        bc, _ = btr[bi]
        offsets.append(o)
        print("  %-7d %-7d %-7d %-9s %.6f  (spread within trial %.3f ms)"
              % (bi + 1, len(bc), len(dc), "yes", float(np.median(o)), spread))

    if offsets:
        allo = np.concatenate(offsets)
        print("\n  the Bpod->Trodes clock offset, measured at %d state onsets across "
              "%d trials:" % (len(allo), len(offsets)))
        print("    median      %.6f s" % float(np.median(allo)))
        print("    spread      %.3f ms (max - min)"
              % float((allo.max() - allo.min()) * 1000.0))
        print("    sd          %.3f ms" % float(np.std(allo) * 1000.0))
        first_bi, last_bi = matched[0][1][1], matched[-1][1][1]
        drift = float(np.median(offsets[-1]) - np.median(offsets[0]))
        span = float(btr[last_bi][1][0] - btr[first_bi][1][0])
        print("    drift       %.3f ms over %.1f s of session (%.0f ppm -- normal here, "
              "and what sync.py fits)"
              % (drift * 1000.0, span, (drift / span * 1e6) if span else 0.0))
        print("\n  Every state onset Bpod recorded lands at the same distance from the "
              "same state onset on the wire. That is only possible if the six lines are "
              "carrying the state numbers Bpod meant to send.")
    if bad:
        print("\n  %d trial(s) on the wire were not paired. Before reading that as a "
              "fault, check the obvious cause: a Bpod file copied mid-session ends at a "
              "trial, and everything the recording saw after that cannot be in it. The "
              "sequences above are printed so they can be read by eye." % len(bad))
    else:
        print("\n  Every complete trial on the wire was paired with a Bpod trial, state "
              "numbers and timing both.")


if __name__ == "__main__":
    main()
