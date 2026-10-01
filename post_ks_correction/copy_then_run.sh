#!/bin/bash
# Wait for the 730 GB session copy to finish, VERIFY it, then launch the
# Kilosort sweep. Gil asked for the sweep to start as soon as the copy is done,
# and this chains the two so neither needs watching.
#
# The verification is the point: robocopy with /J pre-allocates each file at
# full size, so the destination LOOKS complete within seconds of starting. File
# size alone therefore proves nothing until robocopy has exited. Only after it
# exits does comparing total bytes and file count against the source mean
# anything, and the sweep is not launched unless that comparison passes.

SRC="F:/Gil/Shamir/20261001_094335.rec"
DST="Z:/Gil/Shamir_1/20261001_094335.rec"
LOG="D:/Gil/spike_sorting_agent/outputs/copy_then_run.log"

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "waiting for robocopy to finish..."
while tasklist //FI "IMAGENAME eq robocopy.exe" 2>/dev/null | grep -qi robocopy; do
  sleep 60
done
say "robocopy exited"

src_n=$(find "$SRC" -type f 2>/dev/null | wc -l)
dst_n=$(find "$DST" -type f 2>/dev/null | wc -l)
src_b=$(find "$SRC" -type f -printf '%s\n' 2>/dev/null | awk '{s+=$1} END{print s+0}')
dst_b=$(find "$DST" -type f -printf '%s\n' 2>/dev/null | awk '{s+=$1} END{print s+0}')

say "source      : $src_n files, $src_b bytes"
say "destination : $dst_n files, $dst_b bytes"

if [ "$src_n" -ne "$dst_n" ] || [ "$src_b" -ne "$dst_b" ]; then
  say "COPY VERIFICATION FAILED -- NOT launching Kilosort."
  say "re-run the robocopy; it skips files already copied identically."
  exit 1
fi

say "COPY VERIFIED: $src_n files, $(awk -v b=$src_b 'BEGIN{printf "%.2f", b/1073741824}') GB match exactly"
say "launching the Kilosort sweep"
cd "D:/Gil/spike_sorting_agent/post_ks_correction" || exit 1
bash run_all_v2.sh >> "D:/Gil/spike_sorting_agent/outputs/run_all_v2.log" 2>&1
say "sweep finished"
