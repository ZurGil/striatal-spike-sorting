# -*- coding: utf-8 -*-
"""Wait for trodesexport to finish a session, check its output, copy it to Z, verify.

    python scripts/archive_session_when_exported.py <session_id> [--pid N]

Three jobs, in order, and it stops at the first one that fails rather than carrying a
half-truth forward:

  1. WAIT. Polls until the trodesexport process is gone AND the output files have
     stopped growing. A dead process alone is not enough -- it could have crashed.
  2. CHECK the export is complete, from the data itself rather than from arithmetic
     done in advance: probe1.dat must be a whole number of 384-channel int16 samples,
     timestamps.dat must agree on that same packet count, and both must agree with the
     packet count implied by the .rec's own length. A truncated export fails the third
     check and a disk-full one usually fails the first.
  3. COPY to the archive and verify: identical path/size lists, then sampled 1 MiB
     blocks hashed on both sides for the three big files. A size match is never on its
     own evidence that a copy finished -- robocopy preallocates the destination, so
     watching the destination grow tells you nothing about progress.

Nothing is deleted from F. Freeing space stays a human decision.
"""
import hashlib
import os
import subprocess
import sys
import time

B = chr(92)
SID = None
PIDS = []          # --pid may be given more than once: Trodes can be told to export
                   # the same session twice, and then BOTH writers must be gone before
                   # the output can be judged complete.
args = sys.argv[1:]
i = 0
while i < len(args):
    if args[i] == "--pid":
        PIDS.append(int(args[i + 1]))
        i += 2
    else:
        SID = args[i]
        i += 1
if not SID:
    raise SystemExit(__doc__)

SRC = "F:" + B + "Gil" + B + "Shamir" + B + SID + ".rec"
DST = "Z:" + B + "Gil" + B + "Shamir_1" + B + SID + ".rec"
KS = os.path.join(SRC, SID + ".kilosort")
REC = os.path.join(SRC, SID + ".rec")
PROBE = os.path.join(KS, SID + ".probe1.dat")
TS = os.path.join(KS, SID + ".timestamps.dat")

NCH = 384
PACKET = 856          # bytes per .rec packet on this rig
REC_MARKER = b"</Configuration>"   # end of the .rec XML header
TS_MARKER = b"<End settings>"      # end of a Trodes text header
POLL = 60
BLK = 1024 * 1024
NBLOCKS = 32


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg))
    sys.stdout.flush()


def size(p):
    try:
        return os.path.getsize(p)
    except OSError:
        return -1


def running(pids):
    """True while ANY of the watched pids is still alive."""
    for pid in pids or []:
        out = subprocess.run(["tasklist", "/FI", "PID eq %d" % pid, "/NH"],
                             capture_output=True, text=True).stdout
        if str(pid) in out:
            return True
    return False


def header_len(path, marker, probe=2 * 1024 * 1024):
    """Length of a Trodes text header: up to and including the byte after `marker`.

    MEASURED per file, never assumed. These headers are not a fixed width -- the
    `timestamps.dat` header was 609 bytes in September and 612 in October, because a
    field inside it grew. A hard-coded length makes a perfectly good export read as
    corrupt, which is exactly what happened on the first run of this script.
    """
    with open(path, "rb") as f:
        head = f.read(probe)
    i = head.find(marker)
    if i < 0:
        return None
    return i + len(marker) + 1


def sampled_hashes(path, nbytes):
    """The header block, NBLOCKS spread through the body, and the tail."""
    offsets = [0] + [int(k * (nbytes - BLK) / float(NBLOCKS)) for k in range(1, NBLOCKS)]
    offsets.append(nbytes - BLK)
    out = []
    with open(path, "rb") as f:
        for off in offsets:
            f.seek(off)
            out.append((off, hashlib.sha256(f.read(BLK)).hexdigest()))
    return out


# --- 1. wait ------------------------------------------------------------------------
log("waiting for the export of %s" % SID)
log("  watching  %s" % PROBE)
log("  process   %s" % (", ".join("pid %d" % p for p in PIDS) if PIDS
                        else "none given, size-only"))
if PIDS and not running(PIDS):
    # Refuse to "watch" something that is already gone. A wait whose condition can only
    # be false is not a wait -- it is an instant, false report of completion.
    log("none of the given pids is alive -- either the export already finished or the "
        "pids are wrong. Check before trusting this, and re-run with the right ones.")
    raise SystemExit(1)
last = (-1, -1)
stable = 0
while True:
    cur = (size(PROBE), size(TS))
    alive = running(PIDS)
    stable = stable + 1 if cur == last else 0
    log("probe1=%d ts=%d  alive=%s  unchanged_polls=%d" % (cur[0], cur[1], alive, stable))
    if not alive and stable >= 2:
        log("every export process is gone and nothing has grown for %d polls" % stable)
        break
    last = cur
    time.sleep(POLL)

# --- 2. check -----------------------------------------------------------------------
rec_n, probe_n, ts_n = size(REC), size(PROBE), size(TS)
log("")
log("checking the export is complete")
log("  .rec            %d" % rec_n)
log("  probe1.dat      %d" % probe_n)
log("  timestamps.dat  %d" % ts_n)

rec_hdr = header_len(REC, REC_MARKER)
ts_hdr = header_len(TS, TS_MARKER)
log("  headers: rec=%s timestamps=%s (measured, not assumed)" % (rec_hdr, ts_hdr))

problems = []
if rec_hdr is None:
    problems.append("no %r in the first 2 MiB of the .rec -- not a Trodes recording?"
                    % REC_MARKER)
if ts_hdr is None:
    problems.append("no %r at the head of timestamps.dat" % TS_MARKER)

packets_probe = packets_ts = packets_rec = -1
if not problems:
    if probe_n % (NCH * 2) != 0:
        problems.append("probe1.dat is not a whole number of %d-channel samples "
                        "(%d bytes, %d left over)"
                        % (NCH, probe_n, probe_n % (NCH * 2)))
    packets_probe = probe_n // (NCH * 2)

    if (ts_n - ts_hdr) % 4 != 0:
        problems.append("timestamps.dat is not a whole number of 4-byte stamps past "
                        "its %d-byte header (%d bytes, %d left over)"
                        % (ts_hdr, ts_n, (ts_n - ts_hdr) % 4))
    packets_ts = (ts_n - ts_hdr) // 4
    if packets_probe != packets_ts:
        problems.append("probe1.dat holds %d packets but timestamps.dat holds %d"
                        % (packets_probe, packets_ts))

    body = rec_n - rec_hdr
    packets_rec = body // PACKET
    trailing = body % PACKET
    if trailing:
        # normal: recording stopped mid-packet. Worth printing, not a failure.
        log("  the .rec ends %d bytes into a packet (recording stopped mid-packet)"
            % trailing)
    if abs(packets_probe - packets_rec) > 1:
        problems.append("the .rec implies %d packets but the export holds %d -- short "
                        "by %.1f GiB"
                        % (packets_rec, packets_probe,
                           (packets_rec - packets_probe) * NCH * 2 / 1024.0 ** 3))
    log("  packets: probe1=%d timestamps=%d rec=%d"
        % (packets_probe, packets_ts, packets_rec))
    log("  duration: %.2f h at 30 kHz" % (packets_probe / 30000.0 / 3600.0))

if problems:
    log("")
    for p in problems:
        log("INCOMPLETE: %s" % p)
    log("")
    log("NOT copying to the archive. The .rec is intact, so a re-export is possible.")
    raise SystemExit(2)
log("  the export is complete and internally consistent")

# --- 3. copy ------------------------------------------------------------------------
log("")
log("copying to %s" % DST)
t0 = time.time()
cmd = ["robocopy", SRC, DST, "/E", "/J", "/R:2", "/W:5", "/NFL", "/NDL", "/NP", "/NJH"]
r = subprocess.run(cmd, capture_output=True, text=True)
log("robocopy exit code %d (0-7 means success, 8 and up means failure)" % r.returncode)
for line in (r.stdout or "").strip().split("\n")[-12:]:
    if line.strip():
        log("  | %s" % line.strip())
if r.returncode >= 8:
    log("robocopy FAILED -- the archive is not trustworthy, nothing deleted")
    raise SystemExit(3)
log("the copy took %.1f min" % ((time.time() - t0) / 60.0))

# --- 4. verify ----------------------------------------------------------------------
log("")
log("verifying the archived copy")


def walk(root):
    out = {}
    for dirpath, _d, files in os.walk(root):
        for name in files:
            full = os.path.join(dirpath, name)
            out[os.path.relpath(full, root)] = os.path.getsize(full)
    return out


fm, zm = walk(SRC), walk(DST)
log("  F: %d files, %.2f GiB" % (len(fm), sum(fm.values()) / 1024.0 ** 3))
log("  Z: %d files, %.2f GiB" % (len(zm), sum(zm.values()) / 1024.0 ** 3))
bad = [k for k in sorted(fm) if k not in zm or zm[k] != fm[k]]
for k in bad:
    log("  MISSING OR WRONG SIZE ON Z: %s" % k)
if bad:
    log("the archive is incomplete -- nothing deleted")
    raise SystemExit(4)
log("  every file is present on Z at the same size")

for name, path in (("rec", REC), ("probe1.dat", PROBE), ("timestamps.dat", TS)):
    zpath = os.path.join(DST, os.path.relpath(path, SRC))
    n = os.path.getsize(path)
    log("  sampling %s (%.1f GiB)" % (name, n / 1024.0 ** 3))
    a, b = sampled_hashes(path, n), sampled_hashes(zpath, n)
    diff = [(o, h1, h2) for (o, h1), (_o, h2) in zip(a, b) if h1 != h2]
    if diff:
        for off, h1, h2 in diff:
            log("    MISMATCH @ %d  F=%s Z=%s" % (off, h1[:16], h2[:16]))
        log("the archived content differs -- nothing deleted")
        raise SystemExit(5)
    log("    %d x 1 MiB blocks identical on both sides" % len(a))

# --- 5. verify again, with the cache bypassed ---------------------------------------
# Everything above read Z seconds after robocopy stopped writing, which the local SMB
# client can answer from its own cache -- so on its own it confirms what was SENT, not
# what landed. verify_archive.py re-reads the big files with FILE_FLAG_NO_BUFFERING at
# fresh offsets, hashes the small files in full, and checks the archived .rec against
# itself. It is a separate script so it can be re-run later without redoing the copy.
log("")
log("re-verifying with the Windows cache bypassed")
v = os.path.join(os.path.dirname(os.path.abspath(__file__)), "verify_archive.py")
r = subprocess.run([sys.executable, v, SID], capture_output=True, text=True)
for line in (r.stdout or "").strip().split("\n"):
    log("  | %s" % line)
if (r.stderr or "").strip():
    for line in r.stderr.strip().split("\n")[-6:]:
        log("  ! %s" % line)
if r.returncode == 1:
    log("")
    log("the independent check FAILED -- treat the archive as untrustworthy, and note "
        "that F still holds the original")
    raise SystemExit(6)
if r.returncode != 0:
    log("")
    log("the independent check could not finish (exit %d). The copy itself passed "
        "every check above, but the big files are not independently confirmed -- run "
        "scripts/verify_archive.py %s by hand before deleting anything from F."
        % (r.returncode, SID))
    raise SystemExit(7)

log("")
log("DONE. %s is archived at %s and verified twice: once against F, and once more "
    "with the cache bypassed." % (SID, DST))
log("Nothing was deleted from F -- that stays a human decision.")
