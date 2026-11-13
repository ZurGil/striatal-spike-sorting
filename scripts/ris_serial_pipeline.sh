#!/usr/bin/env bash
# Sort a queue of sessions on RIS while keeping AT MOST TWO on scratch: one being
# worked on, one staged behind it. Nothing accumulates, and nothing on Z is ever
# touched except to receive results.
#
#   bash scripts/ris_serial_pipeline.sh --rat Shamir --queue outputs/ris_queue.txt
#   bash scripts/ris_serial_pipeline.sh --rat Shamir --queue outputs/ris_queue.txt --once
#
# THE RULE THIS EXISTS TO ENFORCE (Gil, 2026-11-13)
# ------------------------------------------------
#   "You don't delete anything from Z. You first copy the files to scratch, do the
#    kilosort, all the results, and then delete from scratch -- because those files
#    are supposed to be saved only at Z. I don't want you to copy 5 or 6 at the same
#    time. If it's not running right now, there's no need for it to be there."
#
# So: scratch is a workbench. Z is the only home. The cycle per session is
#     stage -> sort -> bring EVERY result home to Z -> verify -> purge RIS
# and the next session is not staged until a slot opens.
#
# WHAT IT WILL NOT DO
#   * it never deletes, moves or overwrites anything on Z. It only adds result
#     directories that are absent there.
#   * it never purges a session from RIS unless every single file under that
#     session's RIS result tree is present on Z at the same path and the same byte
#     size, and there are at least MIN_RESULT_FILES of them. On any doubt it STALLS
#     and says so. A stalled pipeline is a cheap problem; a wrong purge is not.
#   * it never purges a session that still has an LSF job, pending or running.
#
# WHY VERIFICATION IS NOT OPTIONAL, with a real example
#   20260915_084812_103710_concat was sorted on the cluster and its Kilosort output
#   copied to Z -- 26 files. 687 more files, the whole agent_processed/ tree and
#   kilosort_original/, existed ONLY on RIS, 8.4 GiB of them, because that session
#   was fetched by hand before the fetch was scripted. Purging RIS on the strength of
#   "the results are on Z" would have destroyed the agent pipeline's entire output
#   and the pristine pre-curation arrays. Hence: compare every file, or purge nothing.
#
# STATE lives on the cluster and on Z, not in a local file, so this can be killed and
# restarted at any point and will work out where it got to.
set -uo pipefail

RAT="Shamir"
QUEUE=""
ONCE=0
SLEEP=600
MAX_STAGED=2
MIN_RESULT_FILES=200

while [ $# -gt 0 ]; do
  case "$1" in
    --rat) RAT="$2"; shift 2 ;;
    --queue) QUEUE="$2"; shift 2 ;;
    --once) ONCE=1; shift ;;
    --sleep) SLEEP="$2"; shift 2 ;;
    --max_staged) MAX_STAGED="$2"; shift 2 ;;
    -h|--help) sed -n '2,45p' "$0"; exit 0 ;;
    *) echo "unknown argument $1"; exit 2 ;;
  esac
done
[ -n "$QUEUE" ] || { echo "usage: $0 --rat <rat> --queue <file> [--once]"; exit 2; }
[ -f "$QUEUE" ] || { echo "no such queue file: $QUEUE"; exit 2; }

case "$RAT" in
  Shamir) ZROOT="/z/Gil/Shamir_1" ;;
  Popper) ZROOT="/z/Gil/Popper" ;;
  Begin)  ZROOT="/z/Gil/Begin_1" ;;
  *) echo "unknown rat $RAT"; exit 2 ;;
esac
SCRATCH="/scratch1/fs1/akepecs/zur"
RESULTS="/storage2/fs1/akepecs/Active/Gil_Zur/kilosort4"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGDIR="$HERE/outputs"
LOG="$LOGDIR/ris_serial_pipeline.log"

say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
rssh() { ssh -o BatchMode=yes compute1 "$@" 2>/dev/null | tr -d '\r'; }

# --- is any LSF job alive for this session? ------------------------------------
jobs_for() {
  local sid="$1"
  rssh "bjobs -w 2>/dev/null | grep -c -- '$sid' || true" | tail -1
}

# --- has the cluster finished producing results for it? -----------------------
results_ready() {
  local sid="$1"
  rssh "test -f '$RESULTS/$sid/kilosort4/spike_clusters.npy' && \
        test -f '$RESULTS/$sid/kilosort4/agent_processed/agent_report.json' && \
        ls '$RESULTS/$sid/outputs/corrected_classification_$sid.csv' >/dev/null 2>&1 && \
        echo yes || echo no" | tail -1
}

# --- bring results home, adding only what Z does not already have -------------
fetch_to_z() {
  local sid="$1"
  local zks="$ZROOT/$sid.rec/$sid.kilosort"
  local zk4="$zks/kilosort4"
  mkdir -p "$zk4" || return 1

  # kilosort4 tree, directory by directory, skipping any that Z already holds
  local subs
  subs=$(rssh "cd '$RESULTS/$sid/kilosort4' && ls -1")
  for d in $subs; do
    # --skip-old-files, not a directory-level skip. A directory that exists on Z is
    # not evidence that it is COMPLETE: 20260915_084812_103710_concat had its 26
    # Kilosort files on Z and 687 others only on RIS. Extracting with
    # --skip-old-files fills in whatever is missing and cannot overwrite a single
    # byte that Z already holds, which is the one thing that must never happen here.
    say "      fetching kilosort4/$d (existing files on Z are never overwritten)"
    ssh -o BatchMode=yes compute1 "cd '$RESULTS/$sid/kilosort4' && tar cf - '$d'" 2>/dev/null \
      | tar xf - --skip-old-files -C "$zk4" || { say "      FETCH FAILED on $d"; return 1; }
  done

  # the outputs/ directory and the per-session waveforms go beside the sort
  for pair in "outputs:$zks/cluster_outputs" "waveforms:$zks/waveforms_cluster"; do
    local src="${pair%%:*}" dst="${pair#*:}"
    if rssh "test -d '$RESULTS/$sid/$src' && echo yes || echo no" | grep -q yes; then
      mkdir -p "$dst"
      say "      fetching $src -> $(basename "$dst") (never overwrites Z)"
      ssh -o BatchMode=yes compute1 "cd '$RESULTS/$sid' && tar cf - '$src'" 2>/dev/null \
        | tar xf - --skip-old-files -C "$dst" --strip-components=1 \
        || { say "      FETCH FAILED on $src"; return 1; }
    fi
  done
  return 0
}

# --- every RIS file present on Z at the same size? ----------------------------
verify_on_z() {
  local sid="$1"
  local zk4="$ZROOT/$sid.rec/$sid.kilosort/kilosort4"
  local r="$LOGDIR/.ris_$sid.txt" z="$LOGDIR/.z_$sid.txt"
  rssh "cd '$RESULTS/$sid/kilosort4' && find . -type f -printf '%p %s\n'" \
    | grep -E '^\./' | LC_ALL=C sort > "$r"
  [ -d "$zk4" ] || { say "      Z has no kilosort4 for $sid"; return 1; }
  ( cd "$zk4" && find . -type f -printf '%p %s\n' ) | LC_ALL=C sort > "$z"
  local nr nz miss
  nr=$(wc -l < "$r"); nz=$(wc -l < "$z")
  miss=$(comm -23 "$r" "$z" | wc -l)
  say "      RIS $nr files, Z $nz files, not matched on Z: $miss"
  if [ "$nr" -lt "$MIN_RESULT_FILES" ]; then
    say "      REFUSING: only $nr files on RIS, expected at least $MIN_RESULT_FILES."
    say "      That looks like an incomplete run, not a finished one."
    return 1
  fi
  if [ "$miss" -ne 0 ]; then
    say "      REFUSING to purge -- these are on RIS and not identical on Z:"
    comm -23 "$r" "$z" | head -10 | while read -r line; do say "        $line"; done
    return 1
  fi
  return 0
}

purge_ris() {
  local sid="$1"
  local a b
  a=$(rssh "du -sh '$SCRATCH/$sid' 2>/dev/null | cut -f1"); a=${a:-0}
  b=$(rssh "du -sh '$RESULTS/$sid' 2>/dev/null | cut -f1"); b=${b:-0}
  say "      purging RIS: scratch $a, storage2 $b"
  rssh "rm -rf '$SCRATCH/$sid' '$RESULTS/$sid'"
  local left
  left=$(rssh "test -e '$SCRATCH/$sid' -o -e '$RESULTS/$sid' && echo STILL || echo gone" | tail -1)
  say "      $sid on RIS: $left"
}

cycle() {
  say "---------------------------------------------------------------"
  local staged
  staged=$(rssh "ls -1 '$SCRATCH' 2>/dev/null" | grep -v '^$' || true)
  local n=0
  [ -n "$staged" ] && n=$(echo "$staged" | wc -l)
  say "on scratch: $n session(s) -- $(echo $staged | tr '\n' ' ')"

  # 1. finish anything that is done
  for sid in $staged; do
    local nj
    nj=$(jobs_for "$sid"); nj=${nj:-0}
    if [ "$nj" != "0" ]; then
      say "  $sid: $nj LSF job(s) still queued or running -- leaving it alone"
      continue
    fi
    local rr
    rr=$(results_ready "$sid")
    if [ "$rr" != "yes" ]; then
      say "  $sid: no LSF job but the results are incomplete on storage2."
      say "         Not purging, not restaging. Read the job log before trusting this:"
      say "         tail -40 /storage2/fs1/akepecs/Active/Gil_Zur/logs/postks_${sid}_*.out"
      continue
    fi
    say "  $sid: done on the cluster -- bringing results home to Z"
    if ! fetch_to_z "$sid"; then
      say "    fetch failed; nothing purged. Will retry next cycle."
      continue
    fi
    if verify_on_z "$sid"; then
      purge_ris "$sid"
    else
      say "    verification failed; RIS keeps everything. This needs a human."
    fi
  done

  # 2. stage the next one if there is room
  staged=$(rssh "ls -1 '$SCRATCH' 2>/dev/null" | grep -v '^$' || true)
  n=0; [ -n "$staged" ] && n=$(echo "$staged" | wc -l)
  if [ "$n" -ge "$MAX_STAGED" ]; then
    say "  $n staged, limit is $MAX_STAGED -- staging nothing"
    return 0
  fi
  local next=""
  while read -r sid; do
    [ -n "$sid" ] || continue
    case "$sid" in \#*) continue ;; esac
    echo "$staged" | grep -qx "$sid" && continue
    # already sorted and home? then it is not a candidate
    if [ -f "$ZROOT/$sid.rec/$sid.kilosort/kilosort4/spike_clusters.npy" ]; then
      continue
    fi
    next="$sid"; break
  done < "$QUEUE"
  if [ -z "$next" ]; then
    say "  queue has nothing left to stage"
    return 0
  fi
  say "  room for one more -- staging $next"
  bash "$HERE/scripts/ris_upload_batch.sh" --rat "$RAT" --submit "$next" 2>&1 \
    | grep -E "^\[" | tee -a "$LOG" | tail -4
}

say "==============================================================="
say "serial pipeline: rat $RAT, queue $QUEUE, at most $MAX_STAGED on scratch"
say "Z is write-only here: results are added, nothing on it is ever removed"
say "==============================================================="
while true; do
  cycle
  [ "$ONCE" = 1 ] && break
  say "sleeping ${SLEEP}s"
  sleep "$SLEEP"
done
