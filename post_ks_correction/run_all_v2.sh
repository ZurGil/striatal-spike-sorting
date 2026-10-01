#!/bin/bash
# Run every configuration against every v2 replicate.
#
# ONE PROCESS PER (config, replicate). ks_patches.enable() swaps a module-level
# function, so sharing a process between configurations would let a patch leak
# into the next run.
#
# PRE-FLIGHT GPU GUARD. Trodes and Kilosort cannot share this 6 GB card: during
# an acquisition Trodes holds ~5.9 GB of the 6.1 GB and sits at 100% util, which
# leaves Kilosort nothing and -- far worse -- risks interfering with a live
# recording. So this refuses to start unless the GPU is actually free. Override
# with FORCE=1 only if you know nothing is recording.
#
# RESUMABLE. Finished runs are skipped, so this can be stopped and restarted.
#
# Usage:  bash run_all_v2.sh            # runs everything still outstanding
#         FORCE=1 bash run_all_v2.sh    # skip the GPU guard

set -u
PY="C:/Users/Adam/anaconda3/envs/kilosort4/python.exe"
ROOT="D:/Gil/spike_sorting_agent"
CFGS="vanilla vanilla_repeat subsample_align coarse_then_align \
amplitude_normalize align_and_amp_norm coarse_align_amp_norm \
footprint_cluster footprint_cluster_strong"
REPS="0 1 2 3"
NEED_FREE_MIB=4000

if [ "${FORCE:-0}" != "1" ]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    free=$((total - used))
    echo "GPU: ${used} MiB used of ${total} MiB (${free} MiB free)"
    if nvidia-smi --query-compute-apps=process_name --format=csv,noheader 2>/dev/null \
         | grep -qi trodes; then
      echo "REFUSING TO START: Trodes is using the GPU." >&2
      echo "A recording may be in progress. Re-run when Trodes is closed," >&2
      echo "or FORCE=1 bash run_all_v2.sh if you are sure." >&2
      exit 1
    fi
    if [ "$free" -lt "$NEED_FREE_MIB" ]; then
      echo "REFUSING TO START: only ${free} MiB free, need ${NEED_FREE_MIB}." >&2
      exit 1
    fi
  fi
fi

for r in $REPS; do
  if [ ! -f "$ROOT/hybrid_v2_rep$r/hybrid.bin" ]; then
    echo "MISSING dataset for replicate $r -- run: python build_tiered_v2.py $r" >&2
    exit 1
  fi
done

total=0; done_already=0
for r in $REPS; do for c in $CFGS; do
  total=$((total + 1))
  [ -f "$ROOT/hybrid_v2_rep$r/ks_$c/spike_times.npy" ] && done_already=$((done_already + 1))
done; done
echo "$done_already of $total runs already complete; starting at $(date +%H:%M:%S)"

for r in $REPS; do
  for c in $CFGS; do
    out="$ROOT/hybrid_v2_rep$r/ks_$c/spike_times.npy"
    if [ -f "$out" ]; then
      echo "[skip] $c rep$r"
      continue
    fi
    echo "=== $c rep$r   $(date +%H:%M:%S) ==="
    "$PY" run_v2_comparison.py --run "$c" "$r" 2>&1 | tail -3
    if [ ! -f "$out" ]; then
      echo "[FAILED] $c rep$r produced no output" >&2
    fi
  done
done

echo "ALL RUNS COMPLETE $(date +%H:%M:%S)"
echo "now score with:"
echo "  python run_v2_comparison.py --compare $CFGS"
