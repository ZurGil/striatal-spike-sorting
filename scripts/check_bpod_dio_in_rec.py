"""Is the Bpod sync word arriving at the MCU intact? Read it out of the live .rec.

Why this exists: when the Bpod-to-MCU wiring is in question, the thing you want to see
is the 6-bit state word the way the sync pipeline will later see it -- but that pipeline
reads `trodesexport`-ed Din{1..6}.dat files, which do not exist until the recording is
closed and exported. The word is already in the .rec, one byte per packet, so it can be
read while the recording is still running.

Nothing here touches sync_pipeline. It re-derives the same decode (Din{j} = bit j-1,
zero-order hold) from the raw packets, read-only, so the two can be compared without
making the pipeline depend on a diagnostic.

Everything about the packet layout is MEASURED from the recording's own XML header, not
assumed: the header length, the packet size, which byte holds the DIO device, and which
bit each Din line sits on. Two self-checks must pass before any DIO byte is believed --
every packet's 0x55 sentinel, and a timestamp that increments by exactly 1 -- because a
wrong offset would otherwise produce a confident, wrong answer about the wiring.

Usage:
    python check_bpod_dio_in_rec.py <session-id or .rec path> [--seconds 60]
                                    [--start-s S | --from-end]
                                    [--compare <session-id of a known-good session>]
"""
import collections
import os
import re
import sys

import numpy as np

FS_FALLBACK = 30000.0
SENTINEL = 0x55
NLINES = 6          # only Din1..Din6 carry the Bpod state word
MIN_DWELL_MS = 1.0  # a code held for less than this is a transition transient, not a state


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
    """Header text plus its exact length. The marker is the only reliable end."""
    with open(path, "rb") as f:
        probe = f.read(4 << 20)
    i = probe.find(b"</Configuration>")
    if i < 0:
        raise SystemExit("no </Configuration> in the first 4 MiB -- not a Trodes .rec?")
    n = i + len(b"</Configuration>") + 1
    return probe[:n].decode("utf-8", "replace"), n


def layout(hdr):
    """Packet geometry and the DIO bit map, straight out of the header.

    A Trodes packet is: 0x55 sentinel, then each configured device's bytes in
    packetOrderPreference order, then a uint32 timestamp, then nch int16 samples.
    """
    nch = int(re.search(r'numChannels="(\d+)"', hdr).group(1))
    fs = float(re.search(r'samplingRate="([\d.]+)"', hdr).group(1))
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
    packet = ts_off + 4 + 2 * nch

    # the DIO device's own channel list: id -> (byte, bit)
    m = re.search(r'<Device\b[^>]*name="([^"]*DIO[^"]*)"[^>]*>(.*?)</Device>', hdr, re.S)
    if not m:
        raise SystemExit("no DIO device in this header")
    dio_name, body = m.group(1), m.group(2)
    lines = {}
    for tag in re.findall(r"<Channel\b[^>]*/>", body):
        a = dict(re.findall(r'(\w+)="([^"]*)"', tag))
        if a.get("dataType") != "digital":
            continue
        lines[a["id"]] = (dev_off[dio_name] + int(a["startByte"]), int(a["bit"]))
    return dict(nch=nch, fs=fs, packet=packet, ts_off=ts_off, dio=lines, devs=devs,
                dio_name=dio_name)


def line_no(name):
    return int(re.search(r"(\d+)$", name).group(1))


def pull(path, hdr_len, L, start_pk, npk):
    """Read npk whole packets and return (sentinel ok, timestamps, dio bytes)."""
    pk = L["packet"]
    with open(path, "rb") as f:
        f.seek(hdr_len + start_pk * pk)
        buf = f.read(pk * npk)
    n = len(buf) // pk
    if n == 0:
        raise SystemExit("no whole packets at that offset -- is the window past the end?")
    a = np.frombuffer(buf[:n * pk], dtype=np.uint8).reshape(n, pk)
    sent = bool(np.all(a[:, 0] == SENTINEL))
    ts = np.ascontiguousarray(a[:, L["ts_off"]:L["ts_off"] + 4]).view(np.uint32).ravel()
    dio_byte = sorted({b for b, _ in L["dio"].values()})[0]
    return sent, ts, a[:, dio_byte].copy()


def decode(dio, L, fs, t0_samp):
    """Per-line bit planes and the combined 6-bit code, sampled at every packet."""
    bits = {}
    for name, (_, bit) in L["dio"].items():
        bits[name] = ((dio >> bit) & 1).astype(np.int8)
    code = np.zeros(len(dio), dtype=np.int64)
    for name, b in bits.items():
        j = line_no(name)
        if j <= NLINES:
            code += b.astype(np.int64) << (j - 1)
    idx = np.concatenate([[0], np.flatnonzero(np.diff(code) != 0) + 1])
    return bits, code, idx, (t0_samp + idx) / fs


def alternation(b):
    """A clean digital line alternates 0,1,0,1 at its own change points."""
    chg = np.flatnonzero(np.diff(b) != 0) + 1
    if len(chg) == 0:
        return 0, True, None
    vals = b[chg]
    ok = bool(np.all(np.diff(vals.astype(np.int16)) != 0))
    dwell = float(np.min(np.diff(chg))) if len(chg) > 1 else None
    return len(chg), ok, dwell


def main():
    args = sys.argv[1:]
    if not args:
        raise SystemExit(__doc__)
    path = rec_path(args[0])
    secs, start_s, from_end = 60.0, None, True
    for i, a in enumerate(args):
        if a == "--seconds":
            secs = float(args[i + 1])
        elif a == "--start-s":
            start_s, from_end = float(args[i + 1]), False
        elif a == "--from-end":
            from_end = True

    hdr, hdr_len = read_header(path)
    L = layout(hdr)
    fs = L["fs"] or FS_FALLBACK
    size = os.path.getsize(path)
    total_pk = (size - hdr_len) // L["packet"]

    print("%s" % path)
    print("  header %d bytes (measured), packet %d bytes, %d channels at %g Hz"
          % (hdr_len, L["packet"], L["nch"], fs))
    print("  devices in packet order: %s"
          % ", ".join("%s(%dB)" % (n, nb) for _, n, nb in L["devs"]))
    dio_byte = sorted({b for b, _ in L["dio"].values()})[0]
    print("  DIO device %r -> packet byte %d; %s"
          % (L["dio_name"], dio_byte,
             ", ".join("%s=bit%d" % (k.split("_")[-1], v[1])
                       for k, v in sorted(L["dio"].items(), key=lambda kv: line_no(kv[0])))))
    print("  %d whole packets on disk = %.1f s of recording" % (total_pk, total_pk / fs))

    npk = int(secs * fs)
    if from_end:
        start_pk = max(0, total_pk - npk - int(2 * fs))   # stay clear of the write head
    else:
        start_pk = int(start_s * fs)
    start_pk = min(start_pk, max(0, total_pk - 1))
    npk = min(npk, total_pk - start_pk)

    sent, ts, dio = pull(path, hdr_len, L, start_pk, npk)
    dts = np.diff(ts.astype(np.int64))
    print("\nself-checks on the %d packets read (%.1f s, starting %.1f s into the file)"
          % (len(dio), len(dio) / fs, start_pk / fs))
    print("  sentinel 0x55 on every packet : %s" % ("yes" if sent else "NO"))
    print("  timestamp increments by 1     : %s"
          % ("yes" if np.all(dts == 1) else "NO (%d breaks)" % int(np.sum(dts != 1))))
    if not sent or not np.all(dts == 1):
        print("\n  REFUSING to read the DIO byte: the packet layout does not check out, so "
              "any answer about the wiring would be an artefact of mis-parsing, not a "
              "measurement.")
        raise SystemExit(2)

    bits, code, idx, t = decode(dio, L, fs, ts[0])
    print("\nthe input lines, as they sit in this window")
    print("  %-6s %-6s %-12s %-10s %-15s %s"
          % ("line", "idle", "transitions", "frac high", "shortest pulse", "role"))
    dead, nonalt = [], []
    for name in sorted(bits, key=line_no):
        b = bits[name]
        n, ok, dwell = alternation(b)
        short = "-" if dwell is None else "%.3f ms" % (dwell * 1000.0 / fs)
        j = line_no(name)
        role = "state word bit %d (value %d)" % (j - 1, 1 << (j - 1)) if j <= NLINES \
            else "not part of the state word"
        print("  %-6s %-6s %-12d %-10.4f %-15s %s"
              % (name.split("_")[-1], "high" if b[0] else "low", n, float(b.mean()),
                 short, role))
        if j <= NLINES:
            if n == 0:
                dead.append(name.split("_")[-1])
            if not ok:
                nonalt.append(name.split("_")[-1])
    for nm in nonalt:
        print("  %s does not alternate cleanly -- two changes in a row to the same state"
              % nm)

    dwell_ms = np.diff(np.concatenate([idx, [len(code)]])) * 1000.0 / fs
    stable = dwell_ms >= MIN_DWELL_MS
    vals = code[idx]
    print("\nthe decoded 6-bit word")
    print("  %d changes in %.1f s; %d held >= %g ms (real states), %d shorter "
          "(transients while several lines move)"
          % (len(idx), len(dio) / fs, int(stable.sum()), MIN_DWELL_MS,
             int((~stable).sum())))
    print("  value range %d..%d (a 6-bit word can only be 0..63)"
          % (int(vals.min()), int(vals.max())))
    cs = collections.Counter(vals[stable].tolist())
    ct = collections.Counter(vals[~stable].tolist())
    print("  states held >= %g ms : %s" % (MIN_DWELL_MS, dict(sorted(cs.items()))))
    if ct:
        print("  transient values     : %s" % dict(sorted(ct.items())))
    if dead:
        print("\n  LINE(S) WITH NO ACTIVITY AT ALL: %s. If the Bpod is running trials, "
              "that is a dead or disconnected input, and every state number carrying that "
              "bit comes out wrong." % ", ".join(dead))

    print("\n  stable states in order, with how long each was held:")
    sv, st_, sd = vals[stable], t[stable], dwell_ms[stable]
    for k in range(min(50, len(sv))):
        print("    %10.3f s   code %2d   %9.2f ms" % (st_[k], sv[k], sd[k]))
    if len(sv) > 50:
        print("    ... %d more" % (len(sv) - 50))

    if "--compare" in args:
        ref = args[args.index("--compare") + 1]
        d = os.path.join(r"F:\Gil\Shamir", ref + ".rec", ref + ".DIO")
        if not os.path.isdir(d):
            print("\n(no exported .DIO for %s on F, so no comparison)" % ref)
            return
        sys.path.insert(0, r"D:\Gil")
        from sync_pipeline import dio_decoder as dd   # read-only; never modified
        rc, rt = dd.extract_ttls(d)
        rdw = np.append(np.diff(rt), np.inf) * 1000.0
        rs = collections.Counter(rc[rdw >= MIN_DWELL_MS].tolist())
        print("\nfor comparison, %s decoded by sync_pipeline's own dio_decoder:" % ref)
        print("  %d events over %.0f s" % (len(rc), rt[-1] - rt[0]))
        print("  states held >= %g ms : %s" % (MIN_DWELL_MS, dict(sorted(rs.items()))))
        new = sorted(set(cs) - set(rs))
        print("  state values in this window that %s never produced: %s"
              % (ref, new if new else "none"))


if __name__ == "__main__":
    main()
