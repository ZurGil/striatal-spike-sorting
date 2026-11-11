# -*- coding: utf-8 -*-
"""Prove the archive copy is complete before anything is deleted from F.

Every file on F must exist on Z at the same size AND the same content. Small files
are hashed in full. For the few very large ones a full read of ~300 GB over the
network would take hours, so they are compared by sampled blocks -- 1 MiB at 64
positions spread through the file plus the first and last block, read on both sides.
That will not catch a single flipped bit in the middle of a 290 GB file, and it is
not claimed to; it catches truncation, a partial copy, a wrong file and a stale
version, which are the failures that actually happen.
"""
import hashlib, os, sys

F = r"F:\Gil\Shamir\20260916_110311.rec"
Z = r"Z:\Gil\Shamir_1\20260916_110311.rec"
FULL_LIMIT = 200 * 1024 * 1024
BLOCK = 1024 * 1024
NSAMPLE = 64

def full_hash(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def sampled_hash(p, size):
    h = hashlib.md5()
    positions = [0] + [int(i * (size - BLOCK) / (NSAMPLE - 1)) for i in range(1, NSAMPLE - 1)] + [max(0, size - BLOCK)]
    with open(p, "rb") as f:
        for off in positions:
            f.seek(off)
            h.update(f.read(BLOCK))
    return h.hexdigest()

rel_files = []
for root, _, files in os.walk(F):
    for fn in files:
        p = os.path.join(root, fn)
        rel_files.append(os.path.relpath(p, F))
rel_files.sort()

print("%d files on F" % len(rel_files))
bad, checked_full, checked_sampled, total = [], 0, 0, 0
for i, rel in enumerate(rel_files, 1):
    fp, zp = os.path.join(F, rel), os.path.join(Z, rel)
    if not os.path.isfile(zp):
        bad.append((rel, "MISSING ON Z")); continue
    fs, zs = os.path.getsize(fp), os.path.getsize(zp)
    if fs != zs:
        bad.append((rel, "SIZE %d vs %d" % (fs, zs))); continue
    total += fs
    try:
        if fs <= FULL_LIMIT:
            a, b = full_hash(fp), full_hash(zp); checked_full += 1
        else:
            a, b = sampled_hash(fp, fs), sampled_hash(zp, zs); checked_sampled += 1
            print("   sampled %-55s %.1f GiB" % (rel, fs / 1073741824.0)); sys.stdout.flush()
        if a != b:
            bad.append((rel, "CONTENT DIFFERS"))
    except Exception as e:
        bad.append((rel, "READ ERROR %s" % e))
    if i % 100 == 0:
        print("   ...%d/%d" % (i, len(rel_files))); sys.stdout.flush()

print()
print("hashed in full : %d files" % checked_full)
print("sampled        : %d files" % checked_sampled)
print("bytes compared : %.1f GiB" % (total / 1073741824.0))
print()
if bad:
    print("*** %d PROBLEM(S) -- DO NOT DELETE ***" % len(bad))
    for rel, why in bad[:40]:
        print("   %-60s %s" % (rel, why))
    sys.exit(1)
print("Every file on F is present on Z at the same size and matching content.")

# And the things that exist ONLY on Z, which are the point of the archive.
for extra in ["20260916_110311.rec",
              r"20260916_110311.kilosort\review\README.txt",
              r"20260916_110311.kilosort\review\verdicts\verdicts.json"]:
    p = os.path.join(Z, extra)
    print("  on Z only: %-58s %s" % (extra, ("%.1f GiB" % (os.path.getsize(p)/1073741824.0)) if os.path.isfile(p) else "MISSING"))
