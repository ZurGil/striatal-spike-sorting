#!/usr/bin/env bash
# Upload a batch of sessions from the archive to RIS scratch, verify each one, and
# submit the two chained cluster jobs. Resumable, and it refuses to submit anything
# it has not verified.
#
#   bash scripts/ris_upload_batch.sh [--rat Shamir] [--submit] <sid> [<sid> ...]
#
# WHY NOT stage_session_to_ris.ps1
# -------------------------------
# That script copies to a MOUNTED RIS share. There is no RIS drive mapped on this PC
# and the UNC paths are not reachable, so the route that actually works is the one
# measured on 2026-11-12: `dd bs=1M | ssh compute1 "cat > ..."`, which sustained
# 173 MB/s from F. Reading from Z instead caps it at the 109 MB/s Z gives.
# The probe map still has to be built on Windows -- the compute node has no Trodes
# reader -- and D:\Gil\probe_maps\<rat>\ already holds one per session, verified
# byte-identical across every map-A session.
#
# WHAT GOES UP, AND WHY EACH ONE
#   <sid>.probe1.dat             the recording; what Kilosort sorts
#   <sid>.timestamps.dat         NOT optional. discover_session() refuses to describe
#                                a session without it, and all six post-KS steps start
#                                by describing the session. Its absence cost a whole
#                                job on 20260915_084812_103710_concat, and LSF still
#                                said DONE (RESEARCH_LOG 5bd.46).
#   <sid>.channelmap_probe1.dat  the map's source, so the map can be re-audited there
#   <sid>_custom_neuropixels_map.json   the map itself, built here
#   dio_Controller_Din{1..6}.dat six small files, for sync later
#
# VERIFICATION, before a single job is submitted
#   exact byte size on both sides, and md5 of nine 1 MiB blocks spread through the
#   file INCLUDING THE LAST ONE -- a truncated transfer matches everywhere except the
#   end, so the last block is the one that earns its keep.
#
# RESUME
#   a partial file is truncated back to a 1 MiB boundary and appended from there, so
#   an interrupted upload never leaves a torn block in the middle.
#
# Nothing is deleted anywhere. The archive is read-only throughout. Only this
# account's own scratch directory is written to.
set -uo pipefail

RAT="Shamir"
SUBMIT=0
SIDS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --rat) RAT="$2"; shift 2 ;;
    --submit) SUBMIT=1; shift ;;
    -h|--help) sed -n '2,45p' "$0"; exit 0 ;;
    *) SIDS+=("$1"); shift ;;
  esac
done
[ ${#SIDS[@]} -gt 0 ] || { echo "usage: $0 [--rat Shamir] [--submit] <sid> [...]"; exit 2; }

case "$RAT" in
  Shamir) ARCH="/z/Gil/Shamir_1" ;;
  Popper) ARCH="/z/Gil/Popper" ;;
  Begin)  ARCH="/z/Gil/Begin_1" ;;
  *) echo "unknown rat $RAT"; exit 2 ;;
esac
MAPS="/d/Gil/probe_maps/${RAT}"
REMOTE="/scratch1/fs1/akepecs/zur"
CODE="/storage2/fs1/akepecs/Active/Gil_Zur/code"
LOG="/d/Gil/spike_sorting_agent/outputs/ris_upload_batch_$(date +%Y%m%d_%H%M).log"

PY="${PY:-/c/Users/Adam/anaconda3/envs/kilosort4/python.exe}"

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# md5 of nine 1 MiB blocks spread through a file, last block included.
local_blocks() {
  local f="$1" n
  n=$(stat -c%s "$f")
  "$PY" - "$f" "$n" <<'PY'
import hashlib, sys
f, n = sys.argv[1], int(sys.argv[2])
B = 1024 * 1024
offs = [0] if n <= B else [int(k * (n - B) / 8.0) for k in range(9)]
out = []
with open(f, "rb") as fh:
    for o in offs:
        fh.seek(o)
        out.append(hashlib.md5(fh.read(min(B, n))).hexdigest()[:12])
print(" ".join(out))
PY
}

remote_blocks() {
  local d="$1" n="$2"
  ssh -o BatchMode=yes compute1 "python3 - '$d' '$n'" <<'PY'
import hashlib, sys
f, n = sys.argv[1], int(sys.argv[2])
B = 1024 * 1024
offs = [0] if n <= B else [int(k * (n - B) / 8.0) for k in range(9)]
out = []
with open(f, "rb") as fh:
    for o in offs:
        fh.seek(o)
        out.append(hashlib.md5(fh.read(min(B, n))).hexdigest()[:12])
print(" ".join(out))
PY
}

upload_one() {
  local src="$1" dst="$2" name
  name="$(basename "$src")"
  local lsize rsize
  lsize=$(stat -c%s "$src")
  rsize=$(ssh -o BatchMode=yes compute1 "stat -c%s '$dst' 2>/dev/null || echo 0" | tr -d '\r')
  rsize=${rsize:-0}

  if [ "$rsize" = "$lsize" ]; then
    say "    $name already there at $lsize bytes, checking blocks"
    local lb rb
    lb=$(local_blocks "$src"); rb=$(remote_blocks "$dst" "$lsize")
    if [ "$lb" = "$rb" ]; then say "    $name verified, skipping"; return 0; fi
    say "    $name SAME SIZE BUT DIFFERENT BLOCKS -- re-sending from scratch"
    rsize=0
  fi

  local skip=0
  if [ "$rsize" -gt 0 ] && [ "$rsize" -lt "$lsize" ]; then
    skip=$(( rsize / 1048576 ))
    local keep=$(( skip * 1048576 ))
    say "    $name resuming: $rsize bytes there, truncating to $keep (a 1 MiB boundary)"
    ssh -o BatchMode=yes compute1 "truncate -s $keep '$dst'" || return 1
  fi

  local t0=$SECONDS
  if [ "$skip" -gt 0 ]; then
    dd if="$src" bs=1M skip="$skip" status=none | \
      ssh -o BatchMode=yes compute1 "cat >> '$dst'" || return 1
  else
    dd if="$src" bs=1M status=none | \
      ssh -o BatchMode=yes compute1 "cat > '$dst'" || return 1
  fi
  local dt=$(( SECONDS - t0 )); [ "$dt" -lt 1 ] && dt=1
  say "    $name sent in ${dt}s ($(( (lsize - rsize) / 1048576 / dt )) MB/s)"

  rsize=$(ssh -o BatchMode=yes compute1 "stat -c%s '$dst'" | tr -d '\r')
  if [ "$rsize" != "$lsize" ]; then
    say "    $name SIZE MISMATCH after upload: $rsize vs $lsize"
    return 1
  fi
  local lb rb
  lb=$(local_blocks "$src"); rb=$(remote_blocks "$dst" "$lsize")
  if [ "$lb" != "$rb" ]; then
    say "    $name BLOCK MISMATCH after upload"
    say "      local  $lb"
    say "      remote $rb"
    return 1
  fi
  say "    $name verified ($lsize bytes, 9 blocks match including the last)"
  return 0
}

say "=============================================================="
say "batch upload: ${#SIDS[@]} session(s) of $RAT  ->  $REMOTE"
say "archive $ARCH (read-only)   maps $MAPS"
say "log $LOG"
say "=============================================================="

OKS=(); BAD=()
for sid in "${SIDS[@]}"; do
  say ""
  say "--- $sid"
  KS="$ARCH/${sid}.rec/${sid}.kilosort"
  DIO="$ARCH/${sid}.rec/${sid}.DIO"
  MAP="$MAPS/${sid}_custom_neuropixels_map.json"
  files=("$KS/${sid}.probe1.dat" "$KS/${sid}.timestamps.dat"
         "$KS/${sid}.channelmap_probe1.dat" "$MAP")
  for i in 1 2 3 4 5 6; do files+=("$DIO/${sid}.dio_Controller_Din${i}.dat"); done

  missing=0
  for f in "${files[@]}"; do
    [ -s "$f" ] || { say "    MISSING locally: $f"; missing=1; }
  done
  if [ "$missing" = 1 ]; then say "    skipping $sid"; BAD+=("$sid"); continue; fi

  gb=$(( $(stat -c%s "$KS/${sid}.probe1.dat") / 1073741824 ))
  say "    probe1.dat ${gb} GiB, plus timestamps, channel map, probe map and 6 DIO"
  ssh -o BatchMode=yes compute1 "mkdir -p '$REMOTE/$sid'" || { BAD+=("$sid"); continue; }

  failed=0
  for f in "${files[@]}"; do
    upload_one "$f" "$REMOTE/$sid/$(basename "$f")" || { failed=1; break; }
  done
  if [ "$failed" = 1 ]; then
    say "    $sid NOT fully uploaded -- nothing submitted for it"
    BAD+=("$sid"); continue
  fi
  say "    $sid complete and verified"
  OKS+=("$sid")

  if [ "$SUBMIT" = 1 ]; then
    say "    submitting the two chained jobs"
    ssh -o BatchMode=yes compute1 "bash $CODE/ris_submit_all.sh $sid" 2>&1 | tee -a "$LOG" | tail -6
  fi
done

say ""
say "=============================================================="
say "uploaded and verified: ${#OKS[@]}  ${OKS[*]:-}"
say "not done:              ${#BAD[@]}  ${BAD[*]:-}"
if [ "$SUBMIT" = 0 ] && [ ${#OKS[@]} -gt 0 ]; then
  say "nothing was submitted (--submit not given). To submit:"
  for s in "${OKS[@]}"; do say "  ssh compute1 'bash $CODE/ris_submit_all.sh $s'"; done
fi
say "=============================================================="
[ ${#BAD[@]} -eq 0 ] || exit 1
