# -*- coding: utf-8 -*-
"""Verify an archived session on Z against F, with the Windows cache bypassed.

    python scripts/verify_archive.py <session_id>

This exists because the verification inside archive_session_when_exported.py runs
SECONDS after robocopy stops writing, and a read that soon can be served from the local
SMB client cache -- so it can confirm what was *sent* rather than what *landed*. Three
checks the first pass cannot make:

  1. Full sha256 on every file that is not one of the three big ones. Small files are
     cheap to hash completely, so there is no reason to sample them.
  2. Sampled 1 MiB blocks on the three big files, read with FILE_FLAG_NO_BUFFERING so
     every read goes to the device, at DIFFERENT offsets from the first pass -- the
     point is to cover blocks nothing has looked at yet.
  3. The archived .rec read from Z and checked for internal completeness on its own
     terms: its header measured, its body a whole number of packets, and that packet
     count times 384 channels times 2 bytes equal to the archived probe1.dat. This is
     the only check that does not take F's copy as the reference.

Reads only. Deletes nothing, writes nothing. Exit 0 means every check passed, 1 means a
real mismatch, 6 means the cache could not be bypassed so check 2 is inconclusive.
"""
import ctypes
import hashlib
import msvcrt
import os
import sys

B = chr(92)
SID = sys.argv[1] if len(sys.argv) > 1 else None
if not SID:
    raise SystemExit(__doc__)

F = "F:" + B + "Gil" + B + "Shamir" + B + SID + ".rec"
Z = "Z:" + B + "Gil" + B + "Shamir_1" + B + SID + ".rec"
KS = SID + ".kilosort"
BIG = (".rec", ".probe1.dat", ".timestamps.dat")
SMALL_MAX = 10 * 1024 * 1024
NCH = 384
PACKET = 856
REC_MARKER = b"</Configuration>"
MiB = 1024 * 1024
NBLOCKS = 16          # + first and last, so 18 offsets per file

GENERIC_READ = 0x80000000
FILE_SHARE_RW = 0x00000003
OPEN_EXISTING = 3
FILE_FLAG_NO_BUFFERING = 0x20000000
INVALID_HANDLE = ctypes.c_void_p(-1).value


def log(msg):
    print(msg)
    sys.stdout.flush()


def open_unbuffered(path):
    """A read handle that never answers from the cache.

    FILE_FLAG_NO_BUFFERING demands sector-aligned offsets and lengths; every offset and
    length here is a multiple of 1 MiB, which satisfies any sector size in use.
    """
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.restype = ctypes.c_void_p
    kernel32.CreateFileW.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32,
                                     ctypes.c_uint32, ctypes.c_void_p,
                                     ctypes.c_uint32, ctypes.c_uint32,
                                     ctypes.c_void_p]
    h = kernel32.CreateFileW(path, GENERIC_READ, FILE_SHARE_RW, None,
                             OPEN_EXISTING, FILE_FLAG_NO_BUFFERING, None)
    if h is None or h == INVALID_HANDLE:
        raise ctypes.WinError(ctypes.get_last_error())
    fd = msvcrt.open_osfhandle(h, os.O_RDONLY | os.O_BINARY)
    return os.fdopen(fd, "rb", buffering=0)


def block_hash_uncached(path, offset, length=MiB):
    with open_unbuffered(path) as f:
        f.seek(offset)
        buf = bytearray()
        while len(buf) < length:
            chunk = f.read(length - len(buf))
            if not chunk:
                break
            buf.extend(chunk)
    if len(buf) != length:
        raise IOError("short read at %d (%d of %d) in %s"
                      % (offset, len(buf), length, path))
    return hashlib.sha256(bytes(buf)).hexdigest()


def sha_full(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(4 * 1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def rel_all(root):
    out = []
    for dirpath, _d, files in os.walk(root):
        for name in files:
            out.append(os.path.relpath(os.path.join(dirpath, name), root))
    return sorted(out)


problems = []
inconclusive = []

for root in (F, Z):
    if not os.path.isdir(root):
        raise SystemExit("no such session folder: %s" % root)

# --- 1. the small files, hashed in full ---------------------------------------------
log("-- full sha256 on every file that is not one of the three big ones")
n_small = 0
for rel in rel_all(F):
    fp = os.path.join(F, rel)
    if rel.endswith(BIG) and os.path.getsize(fp) > SMALL_MAX:
        continue
    zp = os.path.join(Z, rel)
    if not os.path.exists(zp):
        problems.append("missing on Z: %s" % rel)
        log("  MISSING ON Z  %s" % rel)
        continue
    hf, hz = sha_full(fp), sha_full(zp)
    n_small += 1
    if hf != hz:
        problems.append("sha256 mismatch: %s" % rel)
    log("  %s %-46s %9d bytes  %s"
        % ("ok " if hf == hz else "BAD", os.path.basename(rel),
           os.path.getsize(fp), hf[:12]))

# --- 2. the big files, sampled with the cache bypassed ------------------------------
log("")
log("-- sampled 1 MiB blocks on the big files, read with the cache bypassed")
pairs = [
    ("rec", os.path.join(F, SID + ".rec"), os.path.join(Z, SID + ".rec")),
    ("probe1.dat", os.path.join(F, KS, SID + ".probe1.dat"),
     os.path.join(Z, KS, SID + ".probe1.dat")),
    ("timestamps.dat", os.path.join(F, KS, SID + ".timestamps.dat"),
     os.path.join(Z, KS, SID + ".timestamps.dat")),
]
n_blocks = 0
for name, fp, zp in pairs:
    if not (os.path.exists(fp) and os.path.exists(zp)):
        problems.append("%s is missing on one side" % name)
        continue
    fn, zn = os.path.getsize(fp), os.path.getsize(zp)
    log("  %-15s F=%d Z=%d %s"
        % (name, fn, zn, "sizes match" if fn == zn else "SIZES DIFFER"))
    if fn != zn:
        problems.append("%s differs in size: F=%d Z=%d" % (name, fn, zn))
        continue
    if fn < 2 * MiB:
        log("    smaller than 2 MiB, already hashed in full above")
        continue

    # offsets at 1 MiB alignment, deliberately offset from the first pass's spacing
    last = (fn - MiB) // MiB
    offsets = [0]
    offsets += [(last * k // (NBLOCKS + 1)) * MiB for k in range(1, NBLOCKS + 1)]
    offsets.append(last * MiB)
    offsets = sorted(set(offsets))
    try:
        diff = []
        for off in offsets:
            hf = block_hash_uncached(fp, off)
            hz = block_hash_uncached(zp, off)
            n_blocks += 1
            if hf != hz:
                diff.append(off)
                log("    MISMATCH @ %d  F=%s Z=%s" % (off, hf[:16], hz[:16]))
        if diff:
            problems.append("%s differs at %d sampled offset(s)" % (name, len(diff)))
        else:
            log("    %d x 1 MiB blocks identical, every read taken from the device"
                % len(offsets))
    except (OSError, IOError) as exc:
        # Could not bypass the cache. Say so rather than quietly downgrading to a
        # cached read and calling the result verified.
        inconclusive.append("%s: %s" % (name, exc))
        log("    COULD NOT BYPASS THE CACHE: %s" % exc)

# --- 3. the archived .rec, judged on its own ----------------------------------------
log("")
log("-- the archived .rec, read from Z: is it internally complete?")
zrec = os.path.join(Z, SID + ".rec")
zprobe = os.path.join(Z, KS, SID + ".probe1.dat")
size = os.path.getsize(zrec)
with open(zrec, "rb") as f:
    head = f.read(2 * MiB)
i = head.find(REC_MARKER)
if i < 0:
    problems.append("no %r in the first 2 MiB of the archived .rec" % REC_MARKER)
else:
    hdr = i + len(REC_MARKER) + 1
    body = size - hdr
    packets, trailing = divmod(body, PACKET)
    pn = os.path.getsize(zprobe)
    log("  size            %d" % size)
    log("  header          %d  (marker at %d, measured not assumed)" % (hdr, i))
    log("  whole packets   %d" % packets)
    log("  trailing bytes  %d" % trailing)
    log("  packets x %d x 2 = %d" % (NCH, packets * NCH * 2))
    log("  archived probe1.dat = %d" % pn)
    log("  duration        %.2f h at 30 kHz" % (packets / 30000.0 / 3600.0))
    if trailing:
        log("  the .rec ends %d bytes into a packet (recording stopped mid-packet)"
            % trailing)
    if abs(packets * NCH * 2 - pn) > NCH * 2:
        problems.append("packets x 384 x 2 (%d) != archived probe1.dat (%d)"
                        % (packets * NCH * 2, pn))
    else:
        log("  MATCHES the archived probe1.dat")

log("")
if problems:
    for p in problems:
        log("PROBLEM: %s" % p)
    log("the archive is NOT verified -- nothing should be deleted from F")
    raise SystemExit(1)
if inconclusive:
    for p in inconclusive:
        log("INCONCLUSIVE: %s" % p)
    log("%d small files hashed in full and identical, and the archived .rec is "
        "internally complete -- but the cache could not be bypassed, so the big files "
        "are not independently confirmed" % n_small)
    raise SystemExit(6)
log("VERIFIED. %d small files hashed in full, %d x 1 MiB blocks read from the device "
    "and identical, and the archived .rec is internally complete on its own terms."
    % (n_small, n_blocks))
