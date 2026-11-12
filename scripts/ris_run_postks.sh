#!/usr/bin/env bash
# The cluster-side POST-KILOSORT run for ONE session, executed INSIDE the postks
# container by LSF. The companion to ris_run_session.sh, which does the sort itself.
#
# WHY THIS EXISTS AT ALL
# ----------------------
# Because "stage -> sort -> build review data -> copy back -> purge" is the unit of work
# (RESEARCH_LOG 5bd.20), and until now only the first two steps ran here. Everything
# below READS THE RAW .dat, which lives on scratch for a few days and is then purged:
#
#   agent pipeline   memory-maps it, one fresh process per 10 units
#   bombcell         reads raw waveforms from it
#   raw_cluster_points / raw_separation_metric   detect spikes in it directly
#
# Run them on the Neuropixels PC instead and each session costs a ~250 GB download,
# then competes with Trodes for the rig's memory and disk -- which is exactly what
# happened to 20260915_084812_103710_concat on 2026-11-11: four workers on a 180 GB
# joint session on a 64 GB machine, reaped for memory pressure, and then blocked behind
# a live recording. Here it is a second LSF job on a CPU host.
#
# The six steps are ordered by what feeds what:
#   1 agent pipeline batches  -> agent_processed/batch_reports/   (raw)
#   2 consolidate             -> agent_processed/agent_report.json + cluster_*.tsv
#   3 bombcell                -> $OUTPUTS/bombcell_results_<sid>/ (raw)
#   4 corrected classification-> $OUTPUTS/corrected_classification_<sid>{.csv,_with_waveforms.json}
#   5 trodes channel lookup   -> $OUTPUTS/unit_trodes_channel_lookup_<sid>.csv
#   6 raw cluster points + raw separation metric                   (raw)
#
# Steps 3-6 are each allowed to fail WITHOUT killing the run, for the same reason the
# waveform extractor is in ris_run_session.sh: by the time they run, hours of work sit
# in the earlier outputs and must still be collectable. Each failure is reported loudly
# and repeated in the summary at the end, with the exact command to re-run it while the
# raw file is still on scratch.
#
# The display half of the pipeline -- session timeline, merge candidates, the review
# HTML -- is deliberately NOT here. It needs the DIO and the Bpod file, it is seconds of
# CPU, and it ends in an artifact published from the PC. Only the raw-bound steps have a
# reason to be on the cluster.
#
# Called by ris_submit_postks.sh; not meant to be run by hand.
#
#   ris_run_postks.sh <session> <data.dat> <probe.json> <session-dir> <gain_to_uv> [parallel] [batch]

set -uo pipefail

SESSION="${1:?session}"
DATA="${2:?data (the staged <session>.probe1.dat)}"
PROBE="${3:?probe json}"
SDIR="${4:?session dir -- where kilosort4/ already is, and where the work tree is built}"
GAIN="${5:?gain_to_uv}"
PARALLEL="${6:-4}"
BATCH="${7:-10}"

CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "${CODE}")"
PY=python3

# Everything the scripts would write to D:\Gil\spike_sorting_agent\outputs goes here
# instead. striatal_agent/paths.py reads this; unset, the scripts behave exactly as they
# always have on the PC.
export SPIKE_AGENT_OUTPUTS="${SDIR}/outputs"
# On the PC the scripts find striatal_agent by walking one directory up from
# scripts/. On the cluster there is no repo -- code/ is flat and striatal_agent/ sits
# INSIDE it (see ris_build_postks_image.sh step 0) -- so that walk lands one level too
# high. Both roots go on PYTHONPATH and the import works either way.
export PYTHONPATH="${CODE}:${REPO}:${PYTHONPATH:-}"
export PYTHONIOENCODING=utf-8
export MPLBACKEND=Agg
mkdir -p "${SPIKE_AGENT_OUTPUTS}"

# Two caches that both want a writable home, and neither fails loudly without one:
# numba (bombcell's @njit(cache=True)) and bombcell's own cachecache, which prints
# "Path '/home/<user>/.bombcell' not writable - caching ... will be skipped" and then
# recomputes everything every run. The job's home is not writable from inside the
# container, so both are pointed at the session's work tree.
export HOME="${SDIR}/.home"
export NUMBA_CACHE_DIR="${SDIR}/.numba_cache"
mkdir -p "${HOME}" "${NUMBA_CACHE_DIR}"

# ---------------------------------------------------------------------------
# The session-shaped work tree.
#
# Every post-KS script finds its inputs through discover_session(), which expects
#     <root>/<sid>.rec/<sid>.kilosort/{<sid>.probe1.dat, map json, kilosort4/}
# Scratch holds a flat staging directory instead. So build that shape with SYMLINKS --
# no copy, the raw file is 300 GB -- and point it at what is already there.
# ---------------------------------------------------------------------------
REC_ROOT="${SDIR}/${SESSION}.rec"
KSIN="${REC_ROOT}/${SESSION}.kilosort"
mkdir -p "${KSIN}"

ln -sfn "${DATA}"  "${KSIN}/${SESSION}.probe1.dat"
ln -sfn "${PROBE}" "${KSIN}/${SESSION}_custom_neuropixels_map.json"
ln -sfn "${SDIR}/kilosort4" "${KSIN}/kilosort4"

# timestamps.dat is required by discover_session for an ordinary session and legitimately
# absent for a concatenated one, which instead proves itself with its concat_*.json
# provenance file (see striatal_agent/session.py). Link whichever the staging shipped.
for extra in "${SESSION}.timestamps.dat" "${SESSION}.channelmap_probe1.dat" \
             "${SESSION}.trodesconf" concat_*.json; do
    src="$(dirname "${DATA}")/${extra}"
    for s in ${src}; do
        [ -e "${s}" ] && ln -sfn "${s}" "${KSIN}/$(basename "${s}")"
    done
done
# the .trodesconf belongs beside the .rec, not inside .kilosort/ -- that is where
# build_trodes_channel_lookup_session.py looks for it
tconf="$(dirname "${DATA}")/${SESSION}.trodesconf"
[ -e "${tconf}" ] && ln -sfn "${tconf}" "${REC_ROOT}/${SESSION}.trodesconf"

N_UNITS=$(${PY} -c "
import numpy as np, sys
st = np.load(sys.argv[1]).ravel()
print(len(np.unique(st)))" "${SDIR}/kilosort4/spike_templates.npy") || exit 1
N_BATCHES=$(( (N_UNITS + BATCH - 1) / BATCH ))

echo "================================================================"
echo "session    ${SESSION}"
echo "code       ${CODE}"
echo "data       ${DATA}"
echo "session    ${SDIR}"
echo "outputs    ${SPIKE_AGENT_OUTPUTS}"
echo "units      ${N_UNITS}  ->  ${N_BATCHES} batches of ${BATCH}, ${PARALLEL} at a time"
echo "started    $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "================================================================"

FAILURES=()

# ---------------------------------------------------------------------------
# STEP 1  the agent pipeline, in batches, in parallel worker roots
#
# One batch per FRESH PROCESS is not a style choice: run_pipeline() reliably crashes
# around unit 20-24 inside a long-lived process (WORKFLOW.md 1e). And run_pipeline()
# writes agent_processed/agent_report.json at a FIXED name, so two batches in one root
# would silently archive each other's units -- hence a root per worker, exactly the
# layout gather_worker_batches.py already knows how to collect.
# ---------------------------------------------------------------------------
echo
echo "### STEP 1/6  agent pipeline, ${N_BATCHES} batches"
PAR_ROOT="${SDIR}/par"
for w in $(seq 1 "${PARALLEL}"); do
    wks="${PAR_ROOT}/w${w}/${SESSION}.rec/${SESSION}.kilosort"
    mkdir -p "${wks}/kilosort4"
    for f in "${KSIN}"/*; do
        [ -f "${f}" ] || [ -L "${f}" ] || continue
        ln -sfn "$(readlink -f "${f}")" "${wks}/$(basename "${f}")"
    done
    # kilosort4 is COPIED, not linked: the pipeline writes kilosort_original/ and
    # agent_processed/ underneath it, and those must not cross between workers.
    cp -f "${SDIR}"/kilosort4/*.npy "${SDIR}"/kilosort4/*.tsv "${SDIR}"/kilosort4/params.py \
          "${wks}/kilosort4/" 2>/dev/null
done

b=0
while [ "${b}" -lt "${N_BATCHES}" ]; do
    pids=()
    for w in $(seq 1 "${PARALLEL}"); do
        [ "${b}" -lt "${N_BATCHES}" ] || break
        wrec="${PAR_ROOT}/w${w}/${SESSION}.rec"
        log="${SPIKE_AGENT_OUTPUTS}/batch_${b}_run.log"
        # run_pipeline_batch_session.py takes the unit INDEX range, not the batch size
        s=$(( b * BATCH )); e=$(( s + BATCH ))
        ${PY} "${CODE}/run_pipeline_batch_session.py" "${SESSION}" "${wrec}" "${b}" "${s}" "${e}" \
            > "${log}" 2>&1 &
        pids+=("$!")
        b=$(( b + 1 ))
    done
    for p in "${pids[@]}"; do wait "${p}" || true; done
    echo "  $(date -u +%H:%M:%S)  ${b}/${N_BATCHES} batches dispatched"
done

echo
echo "### STEP 2/6  gather + consolidate"
${PY} "${CODE}/gather_worker_batches.py" "${SESSION}" "${REC_ROOT}" \
      --par "${PAR_ROOT}" --workers "${PARALLEL}" --batch_size "${BATCH}" \
   && ${PY} "${CODE}/consolidate_batches_session.py" "${SESSION}" "${REC_ROOT}"
RC=$?
[ "${RC}" -ne 0 ] && FAILURES+=("consolidate (exit ${RC}) -- the per-batch archives are in ${PAR_ROOT}/w*/")

# ---------------------------------------------------------------------------
# STEPS 3-6: each one reports and records its own failure instead of exiting, so a
# break in one does not strand the hours already spent in the others.
# ---------------------------------------------------------------------------
run_step () {   # run_step <label> <rerun hint> <cmd...>
    local label="$1"; shift
    local hint="$1"; shift
    echo
    echo "### ${label}"
    "$@"
    local rc=$?
    if [ "${rc}" -ne 0 ]; then
        echo "${label} FAILED (exit ${rc})" >&2
        echo "  re-run WHILE THE RAW FILE IS STILL ON SCRATCH:  ${hint}" >&2
        FAILURES+=("${label} (exit ${rc}) -- re-run: ${hint}")
    fi
    return 0
}

run_step "STEP 3/6  bombcell" \
    "${PY} ${CODE}/run_bombcell_session.py ${SESSION} ${REC_ROOT} ${GAIN}" \
    ${PY} "${CODE}/run_bombcell_session.py" "${SESSION}" "${REC_ROOT}" "${GAIN}"

run_step "STEP 4/6  corrected classification" \
    "${PY} ${CODE}/corrected_classification_session.py ${SESSION} ${REC_ROOT}" \
    ${PY} "${CODE}/corrected_classification_session.py" "${SESSION}" "${REC_ROOT}"

run_step "STEP 5/6  trodes channel lookup" \
    "${PY} ${CODE}/build_trodes_channel_lookup_session.py ${SESSION} ${REC_ROOT}" \
    ${PY} "${CODE}/build_trodes_channel_lookup_session.py" "${SESSION}" "${REC_ROOT}"

run_step "STEP 6a/6 raw cluster points" \
    "${PY} ${CODE}/raw_cluster_points_session.py ${SESSION} ${REC_ROOT}" \
    ${PY} "${CODE}/raw_cluster_points_session.py" "${SESSION}" "${REC_ROOT}"

run_step "STEP 6b/6 raw separation metric" \
    "${PY} ${CODE}/raw_separation_metric_session.py ${SESSION} ${REC_ROOT}" \
    ${PY} "${CODE}/raw_separation_metric_session.py" "${SESSION}" "${REC_ROOT}"

echo
echo "================================================================"
echo "finished   $(date -u +%Y-%m-%dT%H:%M:%SZ)"
if [ "${#FAILURES[@]}" -eq 0 ]; then
    echo "all six steps completed"
else
    echo "${#FAILURES[@]} STEP(S) FAILED -- everything else is complete and collectable:"
    for f in "${FAILURES[@]}"; do echo "  * ${f}"; done
fi
echo
echo "outputs to bring home (${SPIKE_AGENT_OUTPUTS}):"
du -sh "${SPIKE_AGENT_OUTPUTS}"/* 2>/dev/null
echo "and the session's own agent_processed:"
du -sh "${KSIN}/kilosort4/agent_processed" 2>/dev/null
echo "================================================================"

# The worker roots are throwaway: their batch archives were gathered into the session
# root in step 2, and each holds a symlink to the raw file, not a copy. Removing them
# keeps the scratch footprint honest for whoever looks at the quota next.
rm -rf "${PAR_ROOT}"

exit 0
