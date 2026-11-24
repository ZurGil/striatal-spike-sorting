"""Turn the camera's frame pulses into a Trodes timestamp for every video frame.

Why this exists: the FLIR's Line1 opto output drives one MCU digital input, holding it
high for the duration of each exposure. Trodes stamps every one of those edges on the
same 30 kHz clock that stamps the spikes, so the alignment between video and neural data
needs no clock model at all -- frame i of the video is pulse i of the train. That is
strictly better than the Bpod sync, which has to fit and interpolate because the Bpod
free-runs on its own clock and only reports at state transitions.

Two things it will not do silently:

  * It refuses to emit a table unless the pulse train is clean. A dropped pulse, or a
    train that starts before the recording does, would shift every later frame against
    the video and nothing downstream could detect it.
  * With --frames it checks the pulse count against the number of frames the camera
    actually wrote. That comparison is the only thing that catches a frame the writer
    dropped, because such a frame leaves no trace whatsoever on the neural side.

Like check_bpod_dio_in_rec.py this reads the raw packets, so it works on a recording that
is still open and needs no trodesexport pass. The packet layout is measured from the
recording's own XML header, and the 0x55 sentinel plus a timestamp incrementing by
exactly 1 are both verified before any DIO byte is believed -- a wrong offset would
otherwise produce a confident, wrong answer.

The emitted time is the Trodes timestamp carried in the packet, not the packet index, so
it is directly comparable with spike times.

Usage:
    python camera_frame_times.py <session-id or .rec path>
                                 [--line 8]            which Din carries the pulses
                                 [--frames N]          frames the camera wrote
                                 [--frames-csv <path>] record.py's per-grab CSV, or the
                                                       directory holding them (then the
                                                       right one is found by fingerprint)
                                 [--polarity low|high] which phase is the exposure
                                 [--out <dir>]         default: the session's own folder
                                 [--no-figure]
"""
import os
import re
import sys

import numpy as np

SENTINEL = 0x55
FS_FALLBACK = 30000.0

# Every rat, local scratch first then the archive, so the caller never has to know which
# rat a recording belongs to. Kept in step with check_bpod_dio_in_rec.py.
RAT_ROOTS = (r"F:\Gil\Shamir", r"F:\Gil\Popper", r"F:\Gil\Begin",
             r"Z:\Gil\Shamir_1", r"Z:\Gil\Popper", r"Z:\Gil\Begin_1")


def rec_path(arg):
    if arg.lower().endswith(".rec") and os.path.isfile(arg):
        return arg
    sid = os.path.basename(arg.rstrip("\\/")).replace(".rec", "")
    for root in RAT_ROOTS:
        p = os.path.join(root, sid + ".rec", sid + ".rec")
        if os.path.isfile(p):
            return p
    raise SystemExit("no .rec found for %r under any rat root" % arg)


def layout(path):
    """Header length and packet geometry, measured from the embedded XML.

    A Trodes packet is: 0x55 sentinel, then each configured device's bytes in
    packetOrderPreference order, then a uint32 timestamp, then nch int16 samples.
    """
    with open(path, "rb") as f:
        blob = f.read(400000)
    m = re.search(b"</Configuration>", blob)
    if not m:
        raise SystemExit("no </Configuration> in the first 400 kB -- not a Trodes .rec?")
    xml = blob[:m.end()].decode("latin-1")
    nch = int(re.search(r'numChannels="(\d+)"', xml).group(1))
    fs = float(re.search(r'samplingRate="(\d+)"', xml).group(1)) or FS_FALLBACK

    devs = []
    for d in re.finditer(r"<Device\b([^>]*)>", xml):
        a = d.group(1)
        order = re.search(r'packetOrderPreference="(-?\d+)"', a)
        name = re.search(r'name="([^"]*)"', a)
        nb = re.search(r'numBytes="(\d+)"', a)
        devs.append((int(order.group(1)) if order else 0,
                     name.group(1) if name else "?",
                     int(nb.group(1)) if nb else 0))
    devs.sort()
    off, dio = 1, None
    for _, name, nb in devs:
        if "DIO" in name:
            dio = off
        off += nb
    if dio is None:
        raise SystemExit("no DIO device in the packet -- nothing to read")
    ts_off = off
    packet = ts_off + 4 + 2 * nch

    # The header can be padded; slide forward until the sentinel grid locks in and the
    # timestamps increment by one. Both must hold or the offset is wrong.
    size = os.path.getsize(path)
    for pad in range(64):
        h = m.end() + pad
        n = min(2000, (size - h) // packet)
        if n < 100:
            continue
        with open(path, "rb") as f:
            f.seek(h)
            g = np.frombuffer(f.read(n * packet), dtype=np.uint8).reshape(n, packet)
        if not np.all(g[:, 0] == SENTINEL):
            continue
        ts = g[:, ts_off:ts_off + 4].copy().view(np.uint32).ravel()
        if np.all(np.diff(ts.astype(np.int64)) == 1):
            return dict(hdr=h, packet=packet, nch=nch, fs=fs, dio=dio, ts_off=ts_off)
    raise SystemExit("could not lock the packet grid: sentinel or timestamp check failed")


def read_line(path, L, bit):
    """The bit plane for one Din line, plus the Trodes timestamp of every packet."""
    npk = int((os.path.getsize(path) - L["hdr"]) // L["packet"])
    mm = np.memmap(path, dtype=np.uint8, mode="r", offset=L["hdr"],
                   shape=(npk, L["packet"]))
    # Verify the grid across the whole file, not just the probe window at the front.
    step = max(1, npk // 200000)
    if not np.all(mm[::step, 0] == SENTINEL):
        raise SystemExit("the 0x55 sentinel fails somewhere in the file -- refusing to read")
    dio = np.ascontiguousarray(mm[:, L["dio"]])
    t0 = int(np.frombuffer(mm[0, L["ts_off"]:L["ts_off"] + 4].tobytes(), np.uint32)[0])
    return ((dio >> bit) & 1).astype(np.int8), t0, npk


def resolve_csv(arg, npulse):
    """Find record.py's CSV for this recording, by fingerprint rather than by name.

    The session name is typed by hand on the camera PC's command line (record.py's
    --session), so the video and the recording can end up named inconsistently. That is
    a bookkeeping problem, not a data problem: the ChunkFrameID span is a fingerprint.
    An hour at 60 Hz is ~216,000 exposures, and two different sessions agree on that
    count only if they were started and stopped on the very same frame, so a span equal
    to the pulse count identifies the CSV belonging to this recording whatever it is
    called. Prefer passing the directory and letting this pick the file: a name can be
    mistyped, a count of a quarter of a million cannot be mistyped into agreement.
    """
    import glob as _glob
    if os.path.isfile(arg):
        return arg
    cands = sorted(_glob.glob(os.path.join(arg, "*_frames.csv")) if os.path.isdir(arg)
                   else _glob.glob(arg))
    if not cands:
        raise SystemExit("no *_frames.csv found at %s" % arg)
    if len(cands) == 1:
        return cands[0]

    import pandas as pd
    hits, table = [], []
    for c in cands:
        try:
            fid = pd.to_numeric(pd.read_csv(c, usecols=["frame_id"])["frame_id"],
                                errors="coerce")
            span = (int(fid.max()) - int(fid.min()) + 1) if fid.notna().any() else 0
        except Exception as e:
            table.append((c, "unreadable (%s)" % e))
            continue
        table.append((c, "span %d" % span))
        if span == npulse:
            hits.append(c)

    print("\nidentifying the camera CSV by fingerprint: %d candidates, %d pulses on the wire"
          % (len(cands), npulse))
    for c, s in table:
        print("  %-46s %s%s" % (os.path.basename(c), s,
                                "   <== MATCH" if c in hits else ""))
    if len(hits) == 1:
        print("  exactly one span equals the pulse count, so this CSV is this "
              "recording's regardless of what it is named")
        return hits[0]
    if not hits:
        raise SystemExit(
            "no CSV has a ChunkFrameID span equal to the %d pulses in this recording. "
            "Either the video for this session is not in %s, or frames were exposed "
            "outside the recording window -- check the start/stop order before "
            "trusting any pairing." % (npulse, arg))
    raise SystemExit("%d CSVs match the pulse count, so the fingerprint cannot single "
                     "one out -- pass the file path explicitly" % len(hits))


def join_frames(pulses, csv_path, npulse):
    """Attach a Trodes timestamp to every row of record.py's per-grab CSV.

    The CSV has one row per grab ATTEMPT: grab_index, status, frame_id, timestamp_ticks.
    ChunkFrameID is assigned by the camera at exposure, so it indexes the same events the
    pulses do; grab_index only indexes what the host was handed, and a frame lost in
    transport consumes no grab_index while still producing a pulse. So the key is
    frame_id, and grab_index is kept only to break the one tie frame_id cannot.

    The whole verification is one comparison:

        npulse == f_last - f_first + 1

    If it holds, no exposure happened outside the observed span, so pulse i is the frame
    with ChunkFrameID f_first + i - 1 and the mapping is exact however many frames were
    dropped and wherever they fell.
    """
    import pandas as pd
    cam = pd.read_csv(csv_path)
    need = {"grab_index", "status", "frame_id"}
    missing = need - set(cam.columns)
    if missing:
        return pulses, ["%s lacks column(s) %s" % (csv_path, ", ".join(sorted(missing)))]

    fid = pd.to_numeric(cam["frame_id"], errors="coerce")
    if fid.notna().sum() == 0:
        return pulses, ["%s has no usable frame_id values" % csv_path]
    f_first, f_last = int(fid.min()), int(fid.max())
    span = f_last - f_first + 1
    written = int((cam["status"] == "written").sum())
    incomplete = int((cam["status"] != "written").sum())
    print("\ncamera CSV %s" % os.path.basename(csv_path))
    print("  %d grabs logged: %d written, %d incomplete" % (len(cam), written, incomplete))
    print("  ChunkFrameID %d .. %d  -> span %d exposures" % (f_first, f_last, span))
    print("  pulses on the wire: %d" % npulse)

    if span != npulse:
        d = npulse - span
        hint = ""
        if incomplete and fid.isna().any():
            lost = cam.loc[fid.isna(), "grab_index"].tolist()
            hint = (" grab_index %s logged no frame_id, so %s"
                    % (lost[:5],
                       "they account for the difference and precede f_first"
                       if len(lost) == abs(d) else
                       "they only partly account for it"))
        return pulses, [
            "%d pulses but ChunkFrameID spans %d exposures (difference %+d) -- the camera "
            "exposed frames the host never observed, so the offset between pulse 1 and "
            "the first logged frame is not determined.%s" % (npulse, span, d, hint)]

    # Exact: pulse i is ChunkFrameID f_first + i - 1.
    pulses = pulses.copy()
    pulses["frame_id"] = f_first + pulses["pulse_index"].to_numpy() - 1
    out = cam.merge(pulses, on="frame_id", how="left")
    unmatched = int(out["trodes_sample_exposure_start"].isna().sum())
    if unmatched:
        return pulses, ["%d logged grabs got no pulse -- frame_id outside the train"
                        % unmatched]
    print("  every logged grab matched a pulse; mapping is exact")
    return out, []


def main():
    args = sys.argv[1:]
    if not args:
        raise SystemExit(__doc__)

    def opt(flag, default=None, cast=str):
        return cast(args[args.index(flag) + 1]) if flag in args else default

    path = rec_path(args[0])
    line = opt("--line", 8, int)
    nframes = opt("--frames", None, int)
    frames_csv = opt("--frames-csv", None, str)
    polarity = opt("--polarity", "high", str)
    outdir = opt("--out", os.path.dirname(path), str)
    figure = "--no-figure" not in args
    bit = line - 1

    L = layout(path)
    fs = L["fs"]
    print(path)
    print("  packet %d B, DIO byte %d, timestamp byte %d, %d channels at %g Hz"
          % (L["packet"], L["dio"], L["ts_off"], L["nch"], fs))

    b, t0, npk = read_line(path, L, bit)
    print("  %d whole packets = %.1f s, first Trodes timestamp %d" % (npk, npk / fs, t0))
    print("  sentinel 0x55 verified across the file")

    d = np.diff(b)
    rise = np.flatnonzero(d == 1) + 1
    fall = np.flatnonzero(d == -1) + 1
    print("\nDin%d: %d rising, %d falling, high %.4f of the time"
          % (line, rise.size, fall.size, b.mean()))
    if rise.size < 3:
        raise SystemExit("Din%d is not carrying a pulse train -- nothing to do" % line)

    # Exposure start is the edge that begins the active phase. Measured on this rig the
    # HIGH phase is the exposure: the camera reported ExposureTime = 10002 us while the
    # high phase ran 9.9585 ms and the low phase 6.7095 ms, so high matches to 43 us and
    # low is out by 3.3 ms. The 43 us is about what an optocoupler's turn-on/turn-off
    # asymmetry costs. So rising edge opens the exposure, falling edge closes it.
    # Do not infer this from the opto being open-collector -- that reasoning gives the
    # wrong answer here, which is why the exposure time is the thing to check it against.
    if polarity == "low":
        start, end = fall, rise
    elif polarity == "high":
        start, end = rise, fall
    else:
        raise SystemExit("--polarity must be low or high")

    # Pair each start with the next end, and drop a half pulse at either boundary.
    j = np.searchsorted(end, start)
    keep = j < end.size
    start, end = start[keep], end[j[keep]]
    print("  %d complete pulses after trimming partial ones at the edges" % start.size)

    iv = np.diff(start) / fs * 1000.0
    med = float(np.median(iv))
    width = (end - start) / fs * 1000.0
    long_gaps = int(np.sum(iv > 1.5 * med))
    print("\nframe interval   mean %.4f ms (%.4f Hz)   median %.4f   std %.4f (jitter)"
          % (iv.mean(), 1000.0 / iv.mean(), med, iv.std()))
    print("                 min %.4f   max %.4f   intervals > 1.5x median: %d"
          % (iv.min(), iv.max(), long_gaps))
    print("exposure (%s phase)   mean %.4f ms   std %.4f   min %.4f   max %.4f"
          % (polarity, width.mean(), width.std(), width.min(), width.max()))
    ppm = (1000.0 / iv.mean() / (1000.0 / med) - 1.0) * 1e6
    print("camera clock vs Trodes: %.4f Hz measured; %+.0f ppm against the modal interval"
          % (1000.0 / iv.mean(), ppm))

    problems = []
    if long_gaps:
        problems.append("%d interval(s) longer than 1.5x the median -- pulses are missing"
                        % long_gaps)

    # Was the camera already running when the recording opened? Do not try to answer that
    # from the line's level: the pull-up holds the input high when the opto is idle, and
    # high is also the exposure-active level, so an idle camera and an exposing one read
    # the same instantaneously. Only the absence of transitions distinguishes them, so
    # the test is whether there is a quiet lead-in before the train starts.
    all_edges = np.sort(np.concatenate([rise, fall]))
    lead_in = float(all_edges[0]) / fs
    tail = float(npk - all_edges[-1]) / fs
    period = med / 1000.0
    print("\nquiet lead-in before the first edge : %.3f s (%.1f frame periods)"
          % (lead_in, lead_in / period))
    print("quiet tail after the last edge      : %.3f s (%.1f frame periods)"
          % (tail, tail / period))
    if lead_in < 2 * period:
        problems.append("the pulse train is already running at the first packet (lead-in "
                        "%.3f s) -- the camera was acquiring before the recording started, "
                        "so pulse 1 is not frame 1 and the count cannot be trusted without "
                        "ChunkFrameID" % lead_in)
    if tail < 2 * period:
        problems.append("the train runs to the last packet (tail %.3f s) -- the recording "
                        "stopped before the camera did, so the final frames may have no "
                        "pulse recorded" % tail)
    if nframes is not None and nframes != start.size:
        problems.append("camera wrote %d frames but the train has %d pulses (difference "
                        "%+d) -- give --frames-csv so ChunkFrameID can locate them"
                        % (nframes, start.size, nframes - start.size))

    # Trodes timestamps, directly comparable with spike times.
    ts_start = t0 + start.astype(np.int64)
    ts_end = t0 + end.astype(np.int64)
    import pandas as pd
    df = pd.DataFrame({
        "pulse_index": np.arange(1, start.size + 1, dtype=np.int64),
        "trodes_sample_exposure_start": ts_start,
        "trodes_sample_exposure_end": ts_end,
        "t_exposure_start_s": (ts_start - t0) / fs,
        "t_exposure_end_s": (ts_end - t0) / fs,
        "t_exposure_mid_s": ((ts_start + ts_end) / 2.0 - t0) / fs,
    })

    if frames_csv:
        df, extra = join_frames(df, resolve_csv(frames_csv, start.size), start.size)
        problems.extend(extra)
    else:
        print("\nNOTE: no --frames-csv, so nothing here can catch a frame the camera "
              "exposed but never wrote. Such a frame still makes a pulse and leaves no "
              "trace on the neural side; only the camera's ChunkFrameID reveals it. The "
              "table below is pulses, not video frames.")

    sid = os.path.basename(path)[:-4]
    os.makedirs(outdir, exist_ok=True)
    if problems:
        print("\nREFUSING to write frame_times -- the train is not clean:")
        for p in problems:
            print("  * %s" % p)
    else:
        out = os.path.join(outdir, sid + "_frame_times.parquet")
        df.to_parquet(out, index=False)
        print("\nall checks passed; %d rows -> %s" % (len(df), out))

    if figure:
        draw(b, iv, width, start, fs, line, sid, npk, outdir, med)


def draw(b, iv, width, start, fs, line, sid, npk, outdir, med):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.1),
                           gridspec_kw=dict(width_ratios=[1.5, 1, 1]))
    t0 = int(start[max(1, start.size // 2)]) - int(0.010 * fs)
    seg = b[t0:t0 + int(0.150 * fs)]
    ax[0].step(np.arange(seg.size) / fs * 1000.0, seg, where="post",
               color="#1f6feb", lw=1.6)
    ax[0].set_ylim(-0.15, 1.3)
    ax[0].set_yticks([0, 1]); ax[0].set_yticklabels(["low", "high"])
    ax[0].set_xlabel("ms"); ax[0].set_title("Din%d, 150 ms of the recording" % line)

    ax[1].hist(iv, bins=80, color="#1f6feb", edgecolor="none")
    ax[1].axvline(med, color="#d1242f", lw=1.4,
                  label="median %.3f ms\n%.3f Hz" % (med, 1000.0 / med))
    ax[1].set_xlabel("frame interval (ms)"); ax[1].set_ylabel("frames")
    ax[1].set_title("interval, jitter %.1f us" % (iv.std() * 1000.0))
    ax[1].legend(frameon=False, fontsize=9)

    ax[2].hist(width, bins=80, color="#8250df", edgecolor="none")
    ax[2].axvline(np.median(width), color="#d1242f", lw=1.4,
                  label="median %.3f ms" % np.median(width))
    ax[2].set_xlabel("exposure (ms)"); ax[2].set_ylabel("frames")
    ax[2].set_title("exposure width"); ax[2].legend(frameon=False, fontsize=9)

    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
        a.tick_params(labelsize=9)
    fig.suptitle("Camera frame pulses on Din%d  --  %s  --  %d frames in %.1f s"
                 % (line, sid, start.size, npk / fs), fontsize=11)
    fig.tight_layout()
    out = os.path.join(outdir, sid + "_frame_pulses.png")
    fig.savefig(out, dpi=135)
    print("figure -> %s" % out)


if __name__ == "__main__":
    main()
