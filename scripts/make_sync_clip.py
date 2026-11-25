# -*- coding: utf-8 -*-
"""Cut chosen Bpod trials out of the session video, with the task events on the frames.

The point is the chain, not the clip. Three links have to agree before a frame can be
named:

    Bpod clock  ->  Trodes 30 kHz clock   the 6-bit state word on Din1..Din6
    Trodes      ->  camera frame index    the exposure pulses on Din8
    frame index ->  video frame           record.py's own per-grab CSV

If "CENTER POKE" lands on the frame where the rat's nose enters the centre port, all
three links are right. That is a check no QC number can give you, and it is why this
exists: run it after the first session on a rig, after any rewiring, and whenever a
sync result looks too good or too strange to trust.

Two things it does NOT do, deliberately:

  * It does not fit a line from the Bpod clock to the Trodes clock. Inside one trial the
    two agree to about 0.2 ms, but the offset STEPS by 21-36 ms from trial to trial
    regardless of how long the trial ran -- a fixed per-trial overhead Bpod leaves out of
    its own timestamps, not a rate error. A least-squares line over a dozen trials was
    out by 14 ms sd and 35 ms peak, which is two frames, and it puts labels on the wrong
    frames without ever looking wrong. Each trial gets its own measured offset.
  * It does not trust the whole Din8 line. A camera that was started, stopped and
    restarted leaves two pulse trains in one recording; only the last one wrote the
    video that survives. Pulse 1 of that train is frame 1.

Usage:
    python make_sync_clip.py <session-id> <trial> [<trial> ...]
                             [--bpod <.mat>]    default: found by date under the rat
                             [--video <.mp4>]   default: the session's camera_recording
                             [--out <dir>]      default: the scratch beside this repo
                             [--line 8]         which Din carries the camera pulses
                             [--probe-s S]      where to look first for the sync word
                             [--dry-run]        resolve paths and stop
"""
import os
import re
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_bpod_dio_in_rec as C                                     # noqa: E402
from camera_frame_times import rec_path                               # noqa: E402
from compare_dio_word_to_bpod import bpod_trials, dio_trials          # noqa: E402

FFMPEG_CANDIDATES = (
    r"C:\Users\Adam\anaconda3\envs\DEEPLABCUT\Library\bin\ffmpeg.exe",
    r"C:\Users\Adam\anaconda3\Library\bin\ffmpeg.exe",
    "ffmpeg",
)
FONT = "C\\:/Windows/Fonts/arialbd.ttf"
MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")

VIDEO_FPS = 60.0          # the container's declared timebase, not the true rate
WIN_SECS = 180.0          # raw packets decoded per look for the sync word
PRE_S, POST_S = 2.0, 2.0  # padding around the chosen trials

# Dual2AFC_nat state numbers
CENTER, CUE, CUE_END, SKIPPED, ITI = 5, 6, 11, 38, 40
CHOICE = (12, 14)
REWARD = (28, 29)
IN_PORT = (17, 20, 23, 26)        # nose in the side port

COL = {"CENTER POKE": "0xffd166", "CUE START": "0x8ecae6", "CUE END": "0x8ecae6",
       "CHOICE POKE": "0xa7e8a0", "REWARD": "0x90e0c0", "LEFT PORT": "0xffa0a0",
       "NO REWARD": "0xff8080", "LEAVING": "0xd8b4e2"}
ROW = ["CENTER POKE", "CUE START", "CUE END", "CHOICE POKE", "REWARD",
       "LEFT PORT", "NO REWARD", "LEAVING"]


def say(s):
    print(s, flush=True)


# --------------------------------------------------------------- finding things
def find_ffmpeg():
    for c in FFMPEG_CANDIDATES:
        if c == "ffmpeg" or os.path.isfile(c):
            return c
    raise SystemExit("no ffmpeg found -- pass one on PATH or install it in an env")


def find_bpod(rec, sid):
    """The Bpod file for this session, by date. Bpod names by day, not by session id."""
    rat_dir = os.path.dirname(os.path.dirname(rec))          # .../<Rat>
    m = re.match(r"(\d{4})(\d{2})(\d{2})_", sid)
    if not m:
        raise SystemExit("cannot read a date out of the session id %r" % sid)
    y, mo, d = m.group(1), MONTHS[int(m.group(2)) - 1], m.group(3)
    stamp = "%s%s_%s" % (mo, d, y)
    roots = [os.path.join(rat_dir, "bpod")]
    # the archive always has it even when the sort is staged on scratch
    roots.append(os.path.join("Z:\\Gil", os.path.basename(rat_dir), "bpod"))
    for r in roots:
        if not os.path.isdir(r):
            continue
        hits = sorted(f for f in os.listdir(r) if stamp in f and f.endswith(".mat"))
        if hits:
            if len(hits) > 1:
                say("  %d Bpod files for %s; taking %s" % (len(hits), stamp, hits[0]))
            return os.path.join(r, hits[0])
    raise SystemExit("no Bpod .mat matching %s under %s -- pass --bpod"
                     % (stamp, " or ".join(roots)))


def find_video(rec, sid):
    rat_dir = os.path.dirname(os.path.dirname(rec))
    for base in (rat_dir, os.path.join("Z:\\Gil", os.path.basename(rat_dir))):
        p = os.path.join(base, "camera_recording", sid, sid + ".rec.mp4")
        if os.path.isfile(p):
            return p
    raise SystemExit("no video at <rat>/camera_recording/%s/%s.rec.mp4 -- pass --video"
                     % (sid, sid))


def find_din(rec, sid, line):
    d = os.path.join(os.path.dirname(rec), sid + ".DIO")
    p = os.path.join(d, "%s.dio_Controller_Din%d.dat" % (sid, line))
    return p if os.path.isfile(p) else None


# ------------------------------------------------------------------ the clocks
def offsets_in_window(rec, mat, win_start_s):
    """Each trial's own Bpod->Trodes offset, from the state word in one window."""
    hdr, hdr_len = C.read_header(rec)
    L = C.layout(hdr)
    fs = L["fs"]
    total = (os.path.getsize(rec) - hdr_len) // L["packet"]
    start_pk = max(0, min(int(win_start_s * fs), total - 1))
    npk = min(int(WIN_SECS * fs), total - start_pk)
    sent, ts, dio = C.pull(rec, hdr_len, L, start_pk, npk)
    if not sent or not np.all(np.diff(ts.astype(np.int64)) == 1):
        raise SystemExit("packet layout self-check failed -- refusing to decode")
    _, code, idx, t = C.decode(dio, L, fs, ts[0])
    btr, tstart, _, nt = bpod_trials(mat)

    off, spread = {}, {}
    for dc, dt in dio_trials(code[idx], t):
        for bi in range(nt):
            bc, bt = btr[bi]
            if len(bc) != len(dc) or not np.array_equal(bc, dc):
                continue
            o = dt - bt[:len(dt)]
            sp = float(o.max() - o.min()) * 1000.0
            if sp <= 5.0:
                off[bi + 1], spread[bi + 1] = float(np.median(o)), sp
                break
    return off, spread, tstart


def locate(rec, mat, trials, probe_s):
    """Find each wanted trial's offset, looking again nearer it if the first look missed.

    The recording may start long after the Bpod session, and the gap is tens of minutes,
    so there is no way to guess the window. One probe gives a rough offset; that places
    the wanted trials to within a second, and the second look lands on them.
    """
    off, spread, tstart = offsets_in_window(rec, mat, probe_s)
    if not off:
        raise SystemExit("no trial paired in the probe window at %.0f s -- try "
                         "--probe-s somewhere the task was running" % probe_s)
    if all(t in off for t in trials):
        return off, spread, tstart

    rough = float(np.median(list(off.values())))
    want = tstart[trials[0] - 1] + rough
    say("  probe at %.0f s paired Bpod trials %d..%d; trial %d sits near Trodes %.0f s, "
        "looking again there" % (probe_s, min(off), max(off), trials[0], want))
    off2, spread2, _ = offsets_in_window(rec, mat, max(want - 60.0, 0.0))
    off.update(off2)
    spread.update(spread2)
    return off, spread, tstart


# ----------------------------------------------------- Trodes -> the video frame
def pulse_train(din_path, rec, line):
    """Rising-edge sample numbers of the LAST contiguous pulse train on this line."""
    if din_path:
        size = os.path.getsize(din_path)
        with open(din_path, "rb") as fh:
            hdr = fh.read(2000).find(b"<End settings>") + len(b"<End settings>\n")
        n = (size - hdr) // 5
        with open(din_path, "rb") as fh:
            fh.seek(hdr)
            r = np.frombuffer(fh.read(n * 5),
                              dtype=np.dtype([("t", "<u4"), ("s", "u1")]))
        rise = r["t"][r["s"] == 1].astype(np.int64)
    else:
        from camera_frame_times import layout, read_line
        L = layout(rec)
        b, t0, _ = read_line(rec, L, line - 1)
        rise = (np.flatnonzero(np.diff(b) == 1) + 1).astype(np.int64) + t0
    if rise.size < 3:
        raise SystemExit("Din%d carries no pulse train" % line)
    cut = np.flatnonzero(np.diff(rise) > 5 * np.median(np.diff(rise)))
    if cut.size:
        say("  Din%d holds %d separate runs; using the last (%d pulses) -- the earlier "
            "ones wrote no surviving video" % (line, cut.size + 1, rise.size - cut[-1] - 1))
        rise = rise[cut[-1] + 1:]
    return rise


def to_frame(t_s, rise):
    k = int(np.searchsorted(rise, t_s * 30000.0, side="right")) - 1
    if k < 0 or k >= rise.size:
        raise SystemExit("Trodes time %.3f s is outside the pulse train (%.1f..%.1f s)"
                         % (t_s, rise[0] / 30000.0, rise[-1] / 30000.0))
    return k


# ----------------------------------------------------------------- Bpod events
def trial_events(sd, tn):
    s = np.atleast_1d(sd.RawData.OriginalStateData[tn - 1]).astype(int)
    t = np.atleast_1d(sd.RawData.OriginalStateTimestamps[tn - 1]).astype(float)[:len(s)]

    def first(want):
        w = want if isinstance(want, tuple) else (want,)
        i = np.flatnonzero(np.isin(s, w))
        return float(t[i[0]]) if i.size else None

    out = []
    for lab, want in (("CENTER POKE", CENTER), ("CUE START", CUE),
                      ("CUE END", CUE_END), ("CHOICE POKE", CHOICE)):
        v = first(want)
        if v is not None:
            out.append((lab, v))
    rew = first(REWARD)
    if rew is not None:
        out.append(("REWARD", rew))
    else:
        inp = np.flatnonzero(np.isin(s, IN_PORT))
        if inp.size and inp[-1] + 1 < len(t):
            out.append(("LEFT PORT", float(t[inp[-1] + 1])))
        sk = first(SKIPPED)
        if sk is not None:
            out.append(("NO REWARD", sk))
    it = first(ITI)
    if it is not None:
        out.append(("LEAVING", it))
    return out


# ------------------------------------------------------------------- the clip
def build_filter(marks, waits, t0, f0, f1, trials, path):
    parts = ["[0:v]trim=start=%.6f:end=%.6f,setpts=PTS-STARTPTS[v0]"
             % ((f0 - 0.5) / VIDEO_FPS, (f1 + 0.5) / VIDEO_FPS)]
    state = {"cur": "v0", "n": 0}

    def add(expr):
        state["n"] += 1
        parts.append("[%s]%s[v%d]" % (state["cur"], expr, state["n"]))
        state["cur"] = "v%d" % state["n"]

    def text(txt, col, size, x, y, a, b):
        add("drawtext=fontfile='%s':text='%s':fontcolor=%s:fontsize=%d:x=%s:y=%s:"
            "box=1:boxcolor=0x000000@0.55:boxborderw=8:"
            "enable='between(t\\,%.4f\\,%.4f)'" % (FONT, txt, col, size, x, y, a, b))

    for tn in trials:
        mk = [m for m in marks if m[3] == tn]
        text("TRIAL %d" % tn, "0xffffff", 34, "28", "24",
             mk[0][0] - t0, mk[-1][0] - t0 + 1.5)

    for vt, k, lab, tn, rel, tro in marks:
        until = max(m[0] for m in marks if m[3] == tn) - t0 + 1.5
        text("%s   %.3f s" % (lab, rel), COL[lab], 30, "28",
             str(78 + ROW.index(lab) * 44), vt - t0, until)
        add("drawtext=fontfile='%s':text='>>> %s':fontcolor=%s:fontsize=46:"
            "x=(w-text_w)/2:y=h-110:box=1:boxcolor=0x000000@0.55:boxborderw=8:"
            "enable='between(t\\,%.4f\\,%.4f)'"
            % (FONT, lab, COL[lab], vt - t0, vt - t0 + 0.45))

    # the wait counting up on the frames, which is the check that catches slow drift
    for tn, ca, cb in waits:
        add("drawtext=fontfile='%s':text='WAITING AT PORT   %%{eif\\:t-%.4f\\:f\\:2} s':"
            "fontcolor=0xffe9a0:fontsize=38:x=(w-text_w)/2:y=h-180:box=1:"
            "boxcolor=0x000000@0.6:boxborderw=10:enable='between(t\\,%.4f\\,%.4f)'"
            % (FONT, ca - t0, ca - t0, cb - t0))

    add("drawtext=fontfile='%s':text='frame %%{eif\\:n+%d\\:d}':fontcolor=0xcccccc:"
        "fontsize=26:x=w-text_w-24:y=h-44:box=1:boxcolor=0x000000@0.55:boxborderw=6"
        % (FONT, f0))

    parts[-1] = parts[-1].rsplit("[v%d]" % state["n"], 1)[0] + "[out]"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(";\n".join(parts))
    return state["n"]


def main():
    args = sys.argv[1:]
    if not args:
        raise SystemExit(__doc__)

    def opt(flag, default=None, cast=str):
        return cast(args[args.index(flag) + 1]) if flag in args else default

    flags = {"--bpod", "--video", "--out", "--line", "--probe-s"}
    positional, skip = [], False
    for i, a in enumerate(args):
        if skip:
            skip = False
        elif a in flags:
            skip = True
        elif not a.startswith("--"):
            positional.append(a)

    if len(positional) < 2:
        raise SystemExit("give a session id and at least one trial number\n" + __doc__)
    sid = os.path.basename(positional[0].rstrip("\\/")).replace(".rec", "")
    trials = sorted(int(p) for p in positional[1:])
    line = opt("--line", 8, int)
    probe_s = opt("--probe-s", 600.0, float)

    rec = rec_path(positional[0])
    mat = opt("--bpod") or find_bpod(rec, sid)
    video = opt("--video") or find_video(rec, sid)
    outdir = opt("--out") or os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "outputs")
    din = find_din(rec, sid, line)

    say("session %s, trials %s" % (sid, trials))
    say("  rec   : %s" % rec)
    say("  bpod  : %s" % mat)
    say("  video : %s" % video)
    say("  Din%-2d : %s" % (line, din or "(no DIO export -- reading raw packets)"))
    say("  out   : %s" % outdir)
    if "--dry-run" in args:
        return

    import scipy.io as sio
    sd = sio.loadmat(mat, squeeze_me=True, struct_as_record=False)["SessionData"]
    tstart_all = np.atleast_1d(sd.TrialStartTimestamp).astype(float)
    if trials[-1] > len(tstart_all):
        raise SystemExit("trial %d is past the end of the session (%d trials)"
                         % (trials[-1], len(tstart_all)))

    say("\nplacing the clocks from the sync word on Din1..Din6")
    off, spread, tstart = locate(rec, mat, trials, probe_s)
    missing = [t for t in trials if t not in off]
    if missing:
        raise SystemExit("trials %s never paired on the wire -- they may be outside the "
                         "recording, or the state word is wrong" % missing)
    for t in trials:
        say("  trial %d: offset %.6f s, states agree within %.3f ms"
            % (t, off[t], spread[t]))

    rise = pulse_train(din, rec, line)
    say("  camera train: %d pulses, %.1f..%.1f s on the Trodes clock"
        % (rise.size, rise[0] / 30000.0, rise[-1] / 30000.0))

    marks, waits = [], []
    for tn in trials:
        evs = trial_events(sd, tn)
        for lab, rel in evs:
            tro = float(tstart[tn - 1] + rel) + off[tn]
            k = to_frame(tro, rise)
            marks.append((k / VIDEO_FPS, k, lab, tn, rel, tro))
        d = dict(evs)
        stop = d.get("REWARD", d.get("LEFT PORT"))
        if "CHOICE POKE" in d and stop is not None:
            a = to_frame(float(tstart[tn - 1] + d["CHOICE POKE"]) + off[tn], rise)
            b = to_frame(float(tstart[tn - 1] + stop) + off[tn], rise)
            waits.append((tn, a / VIDEO_FPS, b / VIDEO_FPS))

    marks.sort()
    f0 = int(marks[0][1] - PRE_S * VIDEO_FPS)
    f1 = int(marks[-1][1] + POST_S * VIDEO_FPS)
    t0 = f0 / VIDEO_FPS

    say("\n  %-6s %-13s %12s %12s %10s" % ("trial", "event", "in trial (s)",
                                           "Trodes (s)", "frame"))
    for vt, k, lab, tn, rel, tro in marks:
        say("  %-6d %-13s %12.3f %12.3f %10d" % (tn, lab, rel, tro, k))
    for tn, a, b in waits:
        say("  trial %d: %.3f s at the side port" % (tn, b - a))

    os.makedirs(outdir, exist_ok=True)
    filt = os.path.join(outdir, "sync_clip_filter_%s.txt" % sid)
    n = build_filter(marks, waits, t0, f0, f1, trials, filt)
    out = os.path.join(outdir, "sync_clip_%s_trial_%s.mp4"
                       % (sid, "_".join(str(t) for t in trials)))
    say("\n  clip: frames %d..%d = %.1f s, %d text layers"
        % (f0, f1, (f1 - f0) / VIDEO_FPS, n))

    cmd = [find_ffmpeg(), "-y", "-copyts", "-ss", "%.3f" % max(t0 - 4.0, 0),
           "-to", "%.3f" % (f1 / VIDEO_FPS + 1.0), "-i", video,
           "-filter_complex_script", filt, "-map", "[out]",
           "-c:v", "libx264", "-crf", "20", "-preset", "veryfast",
           "-pix_fmt", "yuv420p", "-an", out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        say(r.stderr[-4000:])
        raise SystemExit("ffmpeg failed")
    say("  wrote %s (%.1f MB)" % (out, os.path.getsize(out) / 1e6))


if __name__ == "__main__":
    main()
