#!/usr/bin/env bash
# Build the review tool for several sessions, one after another.
#
#     bash scripts/build_review_backfill.sh 20260917_100043 20260918_100248 ...
#
# Which sessions need it is the `tool ready` column of post_ks_correction/SESSION_STATUS.md.
#
# WHY THIS EXISTS
# ---------------
# Gil, 2026-11-17: "please make sure that all session already have it, so next time
# I'll ask you for session, you won't have to build anything."
#
# From now on the cluster computes the raw-bound half as part of the sort
# (ris_run_postks.sh steps 7-11), so a newly sorted session arrives ready. The sessions
# sorted BEFORE that change have only the four original cluster outputs, and the rest
# has to be computed here, off Z -- about 25-40 minutes each, almost all of it
# build_merge_candidates.py reading raw waveforms over SMB. That is a one-off backfill,
# and this runs it unattended.
#
# SERIAL ON PURPOSE. Two of these at once both memory-map a ~300 GB probe1.dat on Z and
# contend for the same link; on 2026-11-12 four concurrent raw readers took this machine
# into memory pressure and the harness reaped them. One at a time is barely slower.
#
# A failure does not stop the run: the remaining sessions still get built, and the
# summary at the end names what failed and the command to repeat it.
set -u

PYEXE="C:/Users/Adam/anaconda3/envs/kilosort4/python.exe"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTDIR="$HERE/outputs"

cd "$HERE" || exit 1

sids=("$@")
if [ "${#sids[@]}" -eq 0 ]; then
    echo "usage: $0 <session_id> [<session_id> ...]" >&2
    exit 2
fi

started=$(date +%s)
ok=(); failed=()

for sid in "${sids[@]}"; do
    log="$OUTDIR/build_review_${sid}.log"
    echo
    echo "=============================================================="
    echo "$(date +%H:%M:%S)  building the review tool for $sid"
    echo "    log: $log"
    echo "=============================================================="
    t0=$(date +%s)
    "$PYEXE" scripts/build_review_for_session.py "$sid" > "$log" 2>&1
    rc=$?
    t1=$(date +%s)
    if [ "$rc" -eq 0 ]; then
        ok+=("$sid")
        echo "    done in $(( (t1 - t0) / 60 )) min"
        tail -3 "$log" | sed 's/^/    /'
    else
        failed+=("$sid")
        echo "    FAILED (exit $rc) after $(( (t1 - t0) / 60 )) min. Last lines:"
        tail -12 "$log" | sed 's/^/    /'
    fi
done

echo
echo "=============================================================="
echo "finished in $(( ($(date +%s) - started) / 60 )) min"
echo "built  : ${#ok[@]}  ${ok[*]:-}"
if [ "${#failed[@]}" -gt 0 ]; then
    echo "FAILED : ${#failed[@]}  ${failed[*]}"
    for sid in "${failed[@]}"; do
        echo "    re-run: \"$PYEXE\" scripts/build_review_for_session.py $sid"
    done
fi
echo "Every build that succeeded is saved on Z and needs only to be PUBLISHED,"
echo "which only Claude can do (Artifact tool)."
echo "=============================================================="
[ "${#failed[@]}" -eq 0 ]
