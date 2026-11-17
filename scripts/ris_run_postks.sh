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
# The steps are ordered by what feeds what:
#   1 agent pipeline batches  -> agent_processed/batch_reports/   (raw)
#   2 consolidate             -> agent_processed/agent_report.json + cluster_*.tsv
#   3 bombcell                -> $OUTPUTS/bombcell_results_<sid>/ (raw)
#   4 corrected classification-> $OUTPUTS/corrected_classification_<sid>{.csv,_with_waveforms.json}
#   5 trodes channel lookup   -> $OUTPUTS/unit_trodes_channel_lookup_<sid>.csv
#   6 raw cluster points + raw separation metric                   (raw)
#   7 review data             -> $OUTPUTS/pipeline_review_data_<sid>.json
#   8 burst and ISI metrics   -> $OUTPUTS/burst_isi_metrics_<sid>.json
#   9 merge 4-9 into the review data (no new computation, just the join)
#  10 session timeline        -> $OUTPUTS/session_timeline_<sid>.json        (DIO)
#  11 merge candidates + raw spike trace snippets                  (raw)
#
# STEPS 7-11 WERE ADDED ON 2026-11-17, AND WHY MATTERS
# ----------------------------------------------------
# They were originally left out as "the display half": seconds of CPU, and they end in
# an artifact published from the PC. That was wrong about two of them. Step 11 reads the
# RAW probe1.dat -- merge candidates pulls 150 waveforms per unit for every co-located
# pair and the snippets cut 16 spikes per unit out of the recording -- so on the PC it
# costs 15-20 minutes per session reading 300 GB over SMB from Z, and it can only be
# done at all while the .rec still exists. Here the file is on local scratch.
#
# Gil's standard, 2026-11-17: "merge candidates should happen as part of the building of
# the review tool in the cluster ... so next time I'll ask you for session, you won't
# have to build anything." After this, a sorted session arrives on Z with every input
# the page needs already computed, and the PC only assembles the HTML and publishes it.
#
# Step 10 needs the DIO, which IS staged (stage_session_to_ris.ps1 copies <sid>.DIO/
# into the staging directory as DIO/), and step 9 needs nothing but steps 4-8's output.
#
# Steps 3-11 are each allowed to fail WITHOUT killing the run, for the same reason the
# waveform extractor is in ris_run_session.sh: by the time they run, hours of work sit
# in the earlier outputs and must still be collectable. Each failure is reported loudly
# and repeated in the summary at the end, with the exact command to re-run it while the
# raw file is still on scratch.
#
# What is still NOT here: assembling the review HTML and publishing it. The first is
# seconds of CPU on the PC against files this job has already produced, and the second
# can only be done by Claude through the Artifact tooling. Nothing else that the page
# needs is left for the PC.
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

# The DIO, for the session timeline (step 10). Staging copies <sid>.DIO/ in as DIO/;
# build_session_timeline_data.py looks for <rec_root>/<sid>.DIO beside the .kilosort.
dio="$(dirname "${DATA}")/DIO"
if [ -d "${dio}" ]; then
    ln -sfn "${dio}" "${REC_ROOT}/${SESSION}.DIO"
else
    echo "NOTE: no DIO/ in the staging directory -- the session timeline (step 10) and"
    echo "      therefore the merge candidates (step 11a) will not run here."
fi

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
echo "### STEP 1/11  agent pipeline, ${N_BATCHES} batches"
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
echo "### STEP 2/11  gather + consolidate"
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

run_step "STEP 3/11  bombcell" \
    "${PY} ${CODE}/run_bombcell_session.py ${SESSION} ${REC_ROOT} ${GAIN}" \
    ${PY} "${CODE}/run_bombcell_session.py" "${SESSION}" "${REC_ROOT}" "${GAIN}"

run_step "STEP 4/11  corrected classification" \
    "${PY} ${CODE}/corrected_classification_session.py ${SESSION} ${REC_ROOT}" \
    ${PY} "${CODE}/corrected_classification_session.py" "${SESSION}" "${REC_ROOT}"

run_step "STEP 5/11  trodes channel lookup" \
    "${PY} ${CODE}/build_trodes_channel_lookup_session.py ${SESSION} ${REC_ROOT}" \
    ${PY} "${CODE}/build_trodes_channel_lookup_session.py" "${SESSION}" "${REC_ROOT}"

run_step "STEP 6a/11 raw cluster points" \
    "${PY} ${CODE}/raw_cluster_points_session.py ${SESSION} ${REC_ROOT}" \
    ${PY} "${CODE}/raw_cluster_points_session.py" "${SESSION}" "${REC_ROOT}"

run_step "STEP 6b/11 raw separation metric" \
    "${PY} ${CODE}/raw_separation_metric_session.py ${SESSION} ${REC_ROOT}" \
    ${PY} "${CODE}/raw_separation_metric_session.py" "${SESSION}" "${REC_ROOT}"

# ---------------------------------------------------------------------------
# STEPS 7-11  everything the review page needs, computed while the raw file is here.
#
# Order is not free: 9 joins 4+6+8 into 7's file, and 11a reads both 7's file (after
# the join) and 10's timeline. 11a and 11b are the raw-bound pair and the reason the
# whole block moved here.
# ---------------------------------------------------------------------------
run_step "STEP 7/11  review data" \
    "${PY} ${CODE}/build_pipeline_review_data.py ${SESSION} ${REC_ROOT}" \
    ${PY} "${CODE}/build_pipeline_review_data.py" "${SESSION}" "${REC_ROOT}"

run_step "STEP 8/11  burst and ISI metrics" \
    "${PY} ${CODE}/build_burst_isi_metrics_session.py ${SESSION} ${REC_ROOT}" \
    ${PY} "${CODE}/build_burst_isi_metrics_session.py" "${SESSION}" "${REC_ROOT}"

run_step "STEP 9/11  merge cluster points, separation, bursts and channels in" \
    "${PY} ${CODE}/merge_review_data_session.py ${SESSION}" \
    ${PY} "${CODE}/merge_review_data_session.py" "${SESSION}"

run_step "STEP 10/11 session timeline from the DIO" \
    "${PY} ${CODE}/build_session_timeline_data.py ${SESSION} ${REC_ROOT}" \
    ${PY} "${CODE}/build_session_timeline_data.py" "${SESSION}" "${REC_ROOT}"

# --max_pairs 3000, not the old default of 150: RESEARCH_LOG 5bd.49 measured that 150
# discarded 94 % of co-located pairs, and a unit missing from the merge panel reads as
# "checked, nothing found" when it means "never compared".
# --uv_per_bit is the session's own gain, passed in by the submitter from the .rec
# header. The 0.195 default is a SpikeGLX Neuropixels figure, 10.6x too large here.
run_step "STEP 11a/11 merge candidates" \
    "${PY} ${CODE}/build_merge_candidates.py ${SESSION} --rec_root ${SDIR} --max_pairs 3000 --uv_per_bit ${GAIN}" \
    ${PY} "${CODE}/build_merge_candidates.py" "${SESSION}" --rec_root "${SDIR}" \
          --max_pairs 3000 --uv_per_bit "${GAIN}"

run_step "STEP 11b/11 raw spike trace snippets" \
    "${PY} ${CODE}/compute_raw_trace_snippets_session.py ${SESSION} ${REC_ROOT} ${GAIN}" \
    ${PY} "${CODE}/compute_raw_trace_snippets_session.py" "${SESSION}" "${REC_ROOT}" "${GAIN}"

echo
echo "================================================================"
echo "finished   $(date -u +%Y-%m-%dT%H:%M:%SZ)"
if [ "${#FAILURES[@]}" -eq 0 ]; then
    echo "all eleven steps completed"
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
