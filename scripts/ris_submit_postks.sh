#!/usr/bin/env bash
# Submit one session's POST-KILOSORT run to WashU RIS Compute1: the agent pipeline,
# bombcell, the corrected classification, and the two raw_* metrics -- everything in the
# review-tool pipeline that has to read the raw .dat.
#
#   Run from a compute1 client host, never from the Neuropixels PC:
#       ssh compute1
#       bash /storage2/fs1/akepecs/Active/Gil_Zur/code/ris_submit_postks.sh 20260915_103710
#
#   Arguments: <session-id> [gain_to_uV] [after-job-id]
#
# The third argument chains this job behind the sort: pass the Kilosort job id and LSF
# holds this one until that job finishes successfully (-w "done(<id>)"). That is how
# ris_submit_all.sh submits both at once -- sort, then post-KS, one stage of the data,
# no human in between. Without it the job runs as soon as a slot frees, which is what
# you want when the sort is already done and you are only adding the missing half.
#
# WHY A SEPARATE JOB AND NOT A THIRD STEP IN ris_run_session.sh
# -------------------------------------------------------------
# The sort needs a GPU host; nothing here touches a GPU. Folding these hours into the
# Kilosort job would hold a V100 idle for all of them, on a cluster where GPU hosts are
# the scarce resource and CPU slots are not. Two jobs also means a post-KS failure costs
# only the post-KS job, with the sort already safe on storage2.
#
# SIZING, MEASURED ON THE RIG AND THEN SCALED BY SESSION LENGTH
# -------------------------------------------------------------
# Each agent-pipeline batch process peaks at 16-20 GB RSS (measured 2026-11-12 on
# 20260915_084812_103710_concat, a 180 GB / 2.17 h joint session, at 10 units per
# process). PARALLEL x 24 GB is therefore the memory that session actually needed, and
# asking for it is the whole difference from running it on the 64 GB rig, where four at
# once got reaped.
#
# But that number is not a constant: what a batch process holds is the raw snippets for
# its units' spikes, so it grows with the session's DURATION as well as with the units
# per process. A 4.7 h session has roughly twice the spikes per unit of a 2.2 h one, and
# 10 units per process there would want ~40 GB each -- 160 GB for four, against a request
# of 112. The job would be reaped hours in, after the GPU half had already been paid for.
#
# So the two knobs are derived from the session's own length instead of hardcoded. The
# invariant is UNIT-HOURS PER PROCESS: 10 units x 2.17 h = 21.7 is the product that
# measured 16-20 GB, so BATCH is set to keep that product roughly constant. Note what
# this gives for the session it was measured on -- 22/2.17 = 10 units, the batch size
# that was actually proven there -- which is the property to preserve when editing this.
#
# PER_PROC_GB is deliberately 32 and not 20: above 2.2 h the linear scaling is INFERRED,
# not measured, so the headroom covers being wrong by half. Memory is not the scarce
# resource here -- 37 of the 41 exec hosts carry 502 GB or more (lshosts, 2026-11-12), so
# a 144 GB request costs nothing in queue time, while an OOM costs the whole job.

set -euo pipefail

SESSION="${1:?usage: $0 <session-id> [gain_to_uV] [after-job-id]}"
GAIN="${2:-0.018311105685598315}"
AFTER="${3:-}"

GROUP="compute-akepecs"
IMAGE="zurgil/spike-postks:bc0.89-py311"
QUEUE="general"

PARALLEL=4                     # concurrent batch processes
CORES=$(( PARALLEL + 1 ))      # one spare for bombcell's threads and the shell itself
PER_PROC_GB=32                 # peak RSS allowed per batch process; see SIZING above
UNIT_HOURS=22                  # units x session-hours per process -- the measured invariant
WALLTIME=2880                  # minutes

MEM="$(( PARALLEL * PER_PROC_GB + 16 ))GB"

SCRATCH="/scratch1/fs1/akepecs/zur"
STORAGE="/storage2/fs1/akepecs/Active/Gil_Zur"
CODE="${STORAGE}/code"

DATA="${SCRATCH}/${SESSION}/${SESSION}.probe1.dat"
PROBE="${SCRATCH}/${SESSION}/${SESSION}_custom_neuropixels_map.json"
SDIR="${STORAGE}/kilosort4/${SESSION}"
LOGDIR="${STORAGE}/logs"

mkdir -p "${LOGDIR}"

for f in "${DATA}" "${PROBE}" "${CODE}/ris_run_postks.sh" \
         "${CODE}/run_pipeline_batch_session.py" "${CODE}/gather_worker_batches.py" \
         "${CODE}/consolidate_batches_session.py" "${CODE}/run_bombcell_session.py" \
         "${CODE}/corrected_classification_session.py" \
         "${CODE}/build_trodes_channel_lookup_session.py" \
         "${CODE}/raw_cluster_points_session.py" \
         "${CODE}/raw_separation_metric_session.py"; do
    [ -f "${f}" ] || { echo "MISSING: ${f}" >&2; exit 1; }
done

# Without a sort there is nothing to score. When chaining behind the Kilosort job this
# file does not exist yet, which is exactly what -w "done(...)" is for -- so only check
# it when running unchained.
if [ -z "${AFTER}" ] && [ ! -s "${SDIR}/kilosort4/spike_times.npy" ]; then
    echo "REFUSING: no finished sort at ${SDIR}/kilosort4" >&2
    echo "Run the sort first, or pass its job id as the third argument to chain behind it." >&2
    exit 1
fi

# The Trodes config, needed by step 5 (the channel lookup) and therefore by step 6b,
# which reads step 5's CSV. stage_session_to_ris.ps1 extracts it from the .rec header and
# ships it -- but a CONCATENATED session has no .rec at all, so for those it has to be
# taken from either part and dropped in by hand. Warn rather than refuse: the other four
# steps, bombcell included, do not need it and are worth running regardless.
if [ ! -s "${SDIR}/${SESSION}.rec/${SESSION}.trodesconf" ] \
   && [ ! -s "${SCRATCH}/${SESSION}/${SESSION}.trodesconf" ]; then
    echo "WARNING: no ${SESSION}.trodesconf anywhere." >&2
    echo "  Steps 5 (trodes channel lookup) and 6b (raw separation metric, which reads" >&2
    echo "  step 5's CSV) WILL FAIL. The other four steps are unaffected." >&2
    echo "  For a joint session, extract it from either part's .rec and put it at" >&2
    echo "    ${SDIR}/${SESSION}.rec/${SESSION}.trodesconf" >&2
    echo "  (the two parts of one day share a config; checked for 20260915 on 2026-11-12)." >&2
    echo >&2
fi

# Same truncation check as the sort's submit script: the raw file must be a whole number
# of 384-channel int16 samples. A short file scores every unit against the wrong data.
BYTES=$(stat -c %s "${DATA}")
if [ $(( BYTES % (2 * 384) )) -ne 0 ]; then
    echo "REFUSING: ${DATA} is ${BYTES} bytes, not a whole number of 384-ch int16 samples." >&2
    exit 1
fi

# Units per batch process, from the session's own duration -- see SIZING at the top.
# The raw file IS the clock here: 384 channels x int16 at 30 kHz, so its length gives the
# recorded hours without needing the .rec or the timestamps file.
SAMPLES=$(( BYTES / (2 * 384) ))
HOURS_X10=$(( SAMPLES * 10 / (30000 * 3600) ))      # tenths of an hour, integer shell only
[ "${HOURS_X10}" -lt 1 ] && HOURS_X10=1
BATCH=$(( UNIT_HOURS * 10 / HOURS_X10 ))
[ "${BATCH}" -gt 10 ] && BATCH=10                   # 10 is the largest batch ever measured
[ "${BATCH}" -lt 2 ] && BATCH=2                     # below 2, per-process startup dominates

export LSF_DOCKER_VOLUMES="${STORAGE}:${STORAGE} /scratch1/fs1/akepecs:/scratch1/fs1/akepecs"

DEPEND=()
if [ -n "${AFTER}" ]; then
    DEPEND=(-w "done(${AFTER})")
fi

echo "submitting post-KS for ${SESSION}"
echo "  image      ${IMAGE}"
echo "  queue      ${QUEUE}  cores ${CORES}  mem ${MEM}  walltime ${WALLTIME} min  no gpu"
echo "  data       ${DATA}  (${BYTES} bytes)"
echo "  session    ${SDIR}"
echo "  gain_to_uV ${GAIN}"
echo "  length     ${SAMPLES} samples = $(( HOURS_X10 / 10 )).$(( HOURS_X10 % 10 )) h"
echo "  parallel   ${PARALLEL} batch processes x ${BATCH} units each (${PER_PROC_GB} GB budget apiece)"
[ -n "${AFTER}" ] && echo "  chained    waits for done(${AFTER})"
echo "  steps      1) agent pipeline  2) consolidate  3) bombcell"
echo "             4) corrected classification  5) trodes lookup  6) raw_* metrics"
echo

bsub \
    -G "${GROUP}" \
    -q "${QUEUE}" \
    -J "postks_${SESSION}" \
    -n "${CORES}" \
    -R "rusage[mem=${MEM}] span[hosts=1]" \
    -M "${MEM}" \
    -W "${WALLTIME}" \
    "${DEPEND[@]}" \
    -o "${LOGDIR}/postks_${SESSION}_%J.out" \
    -e "${LOGDIR}/postks_${SESSION}_%J.err" \
    -a "docker(${IMAGE})" \
    bash "${CODE}/ris_run_postks.sh" \
        "${SESSION}" "${DATA}" "${PROBE}" "${SDIR}" "${GAIN}" "${PARALLEL}" "${BATCH}"

echo
echo "queued. watch it with:"
echo "  bjobs -l"
echo "  tail -f ${LOGDIR}/postks_${SESSION}_*.out"
echo
echo "when it is done, bring home ${SDIR}/outputs/ and the session's agent_processed/"
echo "-- and only then purge ${SCRATCH}/${SESSION}."
