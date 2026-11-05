"""Which Trodes digital input is the camera strobe actually on? Read it out of the .rec.

Why this exists: the physical pin mapping of the 20-pin controller connector is not
documented for us, so the camera's ExposureActive line gets wired to a best guess. This
script does not need the guess to be right. It reads all eight DIO bits and asks which
one, if any, looks like a free-running frame strobe -- a near-constant period with a
near-constant pulse width -- and then names the pin. A wrong guess therefore costs one
short recording instead of a datasheet hunt.

It is also the verification step for every later session: pulse count must agree with the
camera's own frame count, and the measured pulse width must agree with the configured
exposure time. Width disagreeing while period agrees means the line is inverted; nothing
agreeing means the pulse is on a different pin or absent.

Everything about the packet layout is MEASURED from the recording's own XML header rather
than assumed, the same way check_bpod_dio_in_rec.py does it, because a wrong byte offset
would otherwise produce a confident wrong answer about which pin is which.

Unlike check_bpod_dio_in_rec.py this does NOT abort on a timestamp discontinuity. A
Trodes acquisition dropout is a real and survivable event (one was measured at 419 ms on
2026-11-05), and refusing to look would be worse than looking carefully: time is taken
from the packet TIMESTAMP rather than the packet index, so a dropout cannot fabricate a
long pulse, and any interval spanning a gap is held out of the statistics instead of
silently polluting them.

Nothing in sync_pipeline/ is imported or touched.

Usage:
    python find_camera_strobe_in_rec.py <session-id or .rec path>
                                        [--fps 60] [--exposure-ms 10]
                                        [--frames 215971]
                                        [--start-s S] [--seconds N]
"""
import os
import re
import sys

import numpy as np

SENTINEL = 0x55
FS_FALLBACK = 30000.0
BPOD_LINES = 6          # Din1..Din6 carry the Bpod state word; a strobe must not be here


def rec_path(arg):
    if arg.lower().endswith(".rec") and os.path.isfile(arg):
        return arg
    sid = os.path.basename(arg.rstrip("\\/")).replace(".rec", "")
    for root in (r"F:\Gil\Shamir", r"Z:\Gil\Shamir_1"):
        p = os.path.join(root, sid + ".rec", sid + ".rec")
        if os.path.isfile(p):
            return p
    raise SystemExit("cannot find a .rec for %r" % arg)


def read_header(path):
    with open(path, "rb") as f:
        probe = f.read(4 << 20)
    i = probe.find(b"</Configuration>")
    if i < 0:
        raise SystemExit("no </Configuration> in the first 4 MiB -- not a Trodes .rec?")
    n = i + len(b"</Configuration>") + 1
    return probe[:n].decode("utf-8", "replace"), n


def layout(hdr):
    """Packet geometry and the DIO bit map, straight out of the header."""
    nch = int(re.search(r'numChannels="(\d+)"', hdr).group(1))
    fs = float(re.search(r'samplingRate="([\d.]+)"', hdr).group(1)) or FS_FALLBACK
    devs = []
    for tag in re.findall(r"<Device\b[^>]*>", hdr):
        a = dict(re.findall(r'(\w+)="([^"]*)"', tag))
        if a.get("available") != "1":
            continue
        devs.append((int(a.get("packetOrderPreference", 0)), a.get("name", "?"),
                     int(a.get("numBytes", 0))))
    devs.sort()
    off, dev_off = 1, {}
    for _, name, nb in devs:
        dev_off[name] = off
        off += nb
    ts_off = off
    m = re.search(r'<Device\b[^>]*name="([^"]*DIO[^"]*)"[^>]*>(.*?)</Device>', hdr, re.S)
    if not m:
        raise SystemExit("no DIO device in this header")
    lines = {}
    for tag in re.findall(r"<Channel\b[^>]*/>", m.group(2)):
        a = dict(re.findall(r'(\w+)="([^"]*)"', tag))
        if a.get("dataType") == "digital":
            lines[a["id"]] = (dev_off[m.group(1)] + int(a["startByte"]), int(a["bit"]))
    return dict(nch=nch, fs=fs, packet=ts_off + 4 + 2 * nch, ts_off=ts_off,
                dio=lines, dio_name=m.group(1))


def line_no(name):
    return int(re.search(r"(\d+)$", name).group(1))


def describe(bit_vals, ts, fs, gap_edges):
    """Pulse statistics for one bit, with gap-spanning intervals held out.

    Returns None for a line that never changes. A free-running strobe shows a tight
    period and a tight width; anything else (a behavioural state line, a noisy input)
    does not.
    """
    d = np.diff(bit_vals.astype(np.int8))
    rise = np.flatnonzero(d == 1) + 1
    fall = np.flatnonzero(d == -1) + 1
    if len(rise) < 2:
        return None
    t_rise = ts[rise]
    per = np.diff(t_rise).astype(float)
    # an interval containing a dropout is not a measurement of the signal
    clean = np.ones(len(per), dtype=bool)
    if len(gap_edges):
        k = np.searchsorted(t_rise, gap_edges, side="right") - 1
        clean[np.unique(k[(k >= 0) & (k < len(per))])] = False
    per_ok = per[clean]
    # width: each rise paired with the next fall after it
    j = np.searchsorted(fall, rise)
    pair = j < len(fall)
    wid = (ts[fall[j[pair]]] - ts[rise[pair]]).astype(float)
    return dict(n_rise=len(rise), n_fall=len(fall),
                per_med=np.median(per_ok) if len(per_ok) else np.nan,
                per_iqr=(np.percentile(per_ok, 75) - np.percentile(per_ok, 25))
                if len(per_ok) else np.nan,
                per_min=per_ok.min() if len(per_ok) else np.nan,
                per_max=per_ok.max() if len(per_ok) else np.nan,
                n_gap_spanning=int((~clean).sum()),
                wid_med=np.median(wid) if len(wid) else np.nan,
                wid_min=wid.min() if len(wid) else np.nan,
                wid_max=wid.max() if len(wid) else np.nan,
                frac_high=float(bit_vals.mean()), fs=fs)


def main():
    args = sys.argv[1:]
    if not args:
        raise SystemExit(__doc__)

    def opt(flag, cast, default=None):
        return cast(args[args.index(flag) + 1]) if flag in args else default

    path = rec_path(args[0])
    fps = opt("--fps", float, 60.0)
    exp_ms = opt("--exposure-ms", float, None)
    frames = opt("--frames", int, None)
    start_s = opt("--start-s", float, 0.0)
    secs = opt("--seconds", float, None)

    hdr, hdr_len = read_header(path)
    L = layout(hdr)
    fs = L["fs"]
    total_pk = (os.path.getsize(path) - hdr_len) // L["packet"]

    print("%s" % path)
    print("  header %d bytes (measured), packet %d bytes, %d channels at %g Hz"
          % (hdr_len, L["packet"], L["nch"], fs))
    print("  %d whole packets = %.1f s" % (total_pk, total_pk / fs))

    start_pk = max(0, int(start_s * fs))
    npk = total_pk - start_pk if secs is None else min(int(secs * fs), total_pk - start_pk)
    if npk <= 0:
        raise SystemExit("that window is past the end of the file")

    with open(path, "rb") as f:
        f.seek(hdr_len + start_pk * L["packet"])
        a = np.frombuffer(f.read(npk * L["packet"]), dtype=np.uint8)
    n = len(a) // L["packet"]
    a = a[:n * L["packet"]].reshape(n, L["packet"])
    sent_ok = bool(np.all(a[:, 0] == SENTINEL))
    ts = np.ascontiguousarray(a[:, L["ts_off"]:L["ts_off"] + 4]).view(np.uint32).ravel()
    ts = ts.astype(np.int64)
    dio_byte = sorted({b for b, _ in L["dio"].values()})[0]
    dio = a[:, dio_byte].copy()
    del a

    # the layout self-check that matters: the sentinel. A timestamp gap is a dropout,
    # which is reported rather than treated as a parsing failure.
    dts = np.diff(ts)
    gaps = np.flatnonzero(dts != 1)
    lost = int(dts[gaps].sum() - len(gaps)) if len(gaps) else 0
    print("\n  sentinel 0x55 on every packet : %s" % ("yes" if sent_ok else "NO"))
    if not sent_ok:
        raise SystemExit("  REFUSING to read the DIO byte: the packet layout does not "
                         "check out, so any answer about which pin is which would be an "
                         "artefact of mis-parsing.")
    if len(gaps):
        print("  timestamp gaps: %d, losing %d samples (%.1f ms) in total; first at "
              "%.3f s, last at %.3f s" % (len(gaps), lost, lost * 1000.0 / fs,
                                          (ts[gaps[0]] - ts[0]) / fs,
                                          (ts[gaps[-1]] - ts[0]) / fs))
        print("  (intervals spanning a gap are held out of the statistics below)")
    else:
        print("  timestamp increments by 1     : yes, no dropouts")

    gap_edges = ts[gaps] if len(gaps) else np.array([], dtype=np.int64)
    exp_per = fs / fps
    print("\n  looking for a strobe with period %.1f samples (%.3f ms at %g fps)"
          % (exp_per, 1000.0 / fps, fps))
    if exp_ms is not None:
        print("  and pulse width %.1f samples (%.2f ms, the configured exposure)"
              % (exp_ms * fs / 1000.0, exp_ms))

    print("\n  %-6s %-9s %-10s %-22s %-20s %s"
          % ("line", "rises", "frac high", "period (samples)", "width (samples)",
             "verdict"))
    best, rows = None, {}
    for name in sorted(L["dio"], key=lambda k: line_no(k)):
        j = line_no(name)
        b = ((dio >> L["dio"][name][1]) & 1).astype(np.int8)
        st = describe(b, ts, fs, gap_edges)
        rows[j] = st
        if st is None:
            print("  %-6s %-9s %-10.4f %-22s %-20s %s"
                  % ("Din%d" % j, "0", float(b.mean()), "-", "-", "idle, no transitions"))
            continue
        # a strobe is a tight period AND a tight width at the expected rate
        per_err = abs(st["per_med"] - exp_per) / exp_per
        tight = (st["per_iqr"] <= max(2.0, 0.02 * exp_per)
                 and (st["wid_max"] - st["wid_min"]) <= max(4.0, 0.05 * exp_per))
        is_strobe = per_err < 0.05 and tight
        verdict = "STROBE (period matches %g fps)" % fps if is_strobe else (
            "state word bit %d" % (j - 1) if j <= BPOD_LINES else "active, not a strobe")
        print("  %-6s %-9d %-10.4f %-22s %-20s %s"
              % ("Din%d" % j, st["n_rise"], st["frac_high"],
                 "%.1f med, %.1f-%.1f" % (st["per_med"], st["per_min"], st["per_max"]),
                 "%.1f med, %.1f-%.1f" % (st["wid_med"], st["wid_min"], st["wid_max"]),
                 verdict))
        if is_strobe and (best is None or st["per_iqr"] < rows[best]["per_iqr"]):
            best = j

    print("")
    if best is None:
        print("  NO LINE LOOKS LIKE A FRAME STROBE in this window.")
        print("  Either the pulse is not reaching the controller at all (check the pin, "
              "the pull-up, and that the camera was running), or it is on a line that is "
              "also carrying something else. Lines with transitions are listed above -- "
              "if one of Din1..Din6 suddenly has far more activity than the state word "
              "should produce, the camera is very likely wired onto a Bpod pin.")
        raise SystemExit(1)

    st = rows[best]
    print("  ==> the camera strobe is on Din%d (bit %d)." % (best, best - 1))
    if best <= BPOD_LINES:
        print("      WARNING: Din%d also carries Bpod state word bit %d. The camera is on "
              "the wrong pin and this recording's behavioural decode is corrupted. Move "
              "the wire before recording anything real." % (best, best - 1))
    meas_fps = fs / st["per_med"]
    print("      %d pulses, implied frame rate %.4f fps (%.0f ppm from nominal %g)"
          % (st["n_rise"], meas_fps, (meas_fps - fps) / fps * 1e6, fps))
    print("      pulse width %.2f ms median (%.2f-%.2f ms)"
          % (st["wid_med"] * 1000.0 / fs, st["wid_min"] * 1000.0 / fs,
             st["wid_max"] * 1000.0 / fs))
    if st["n_gap_spanning"]:
        print("      %d interval(s) held out for spanning a dropout" % st["n_gap_spanning"])

    if exp_ms is not None:
        w = st["wid_med"] * 1000.0 / fs
        if abs(w - exp_ms) <= max(0.1, 0.1 * exp_ms):
            print("      width agrees with the configured %.2f ms exposure." % exp_ms)
        elif abs((1000.0 / fps - w) - exp_ms) <= max(0.1, 0.1 * exp_ms):
            print("      width is %.2f ms, which is the frame period MINUS the %.2f ms "
                  "exposure: THE LINE IS INVERTED. Count falling edges, or invert the "
                  "GPIO line's output polarity." % (w, exp_ms))
        else:
            print("      width %.2f ms does NOT match the configured %.2f ms exposure. "
                  "Either the exposure actually in use differs from the one reported, or "
                  "this is not ExposureActive (FrameTriggerWait and ExposureEnd have "
                  "different widths)." % (w, exp_ms))

    if frames is not None:
        d = st["n_rise"] - frames
        print("\n  pulses %d vs camera frame count %d -> difference %+d"
              % (st["n_rise"], frames, d))
        if d in (0, 1):
            print("      Agrees. A single extra pulse is expected when the last exposure "
                  "fires as the script stops and that frame never reaches the container.")
        elif d > 1:
            print("      %d MORE pulses than frames written: the camera exposed frames "
                  "that never reached disk. Encoder or write-path loss at the tail, or "
                  "throughout -- check the frame CSV for FrameID gaps to tell which." % d)
        else:
            print("      %d FEWER pulses than frames: pulses were missed, not frames. "
                  "The usual cause is an exposure close enough to the frame period that "
                  "consecutive pulses merge -- at %g fps the period is only %.1f samples, "
                  "so keep the exposure well below %.2f ms."
                  % (-d, fps, exp_per, 1000.0 / fps))


if __name__ == "__main__":
    main()
