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
# SIZING, MEASURED ON THE RIG
# ---------------------------
# Each agent-pipeline batch process peaks at 16-20 GB RSS (measured 2026-11-12 on
# 20260915_084812_103710_concat, a 180 GB / 2.17 h joint session). PARALLEL x 24 GB is
# therefore the memory the job actually needs, and asking for it here is the whole
# difference from running it on the 64 GB rig, where four at once got reaped.

set -euo pipefail

SESSION="${1:?usage: $0 <session-id> [gain_to_uV] [after-job-id]}"
GAIN="${2:-0.018311105685598315}"
AFTER="${3:-}"

GROUP="compute-akepecs"
IMAGE="zurgil/spike-postks:bc0.89-py311"
QUEUE="general"

PARALLEL=4                     # concurrent batch processes
CORES=$(( PARALLEL + 1 ))      # one spare for bombcell's threads and the shell itself
MEM="112GB"                    # PARALLEL x 24 GB + headroom; see SIZING above
WALLTIME=2880                  # minutes

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
echo "  parallel   ${PARALLEL} batch processes"
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
        "${SESSION}" "${DATA}" "${PROBE}" "${SDIR}" "${GAIN}" "${PARALLEL}"

echo
echo "queued. watch it with:"
echo "  bjobs -l"
echo "  tail -f ${LOGDIR}/postks_${SESSION}_*.out"
echo
echo "when it is done, bring home ${SDIR}/outputs/ and the session's agent_processed/"
echo "-- and only then purge ${SCRATCH}/${SESSION}."
