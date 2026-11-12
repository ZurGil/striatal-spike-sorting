#!/usr/bin/env bash
# Submit EVERYTHING the cluster can do for one session, in one command:
#
#   job 1 (GPU)  kilosort4  ->  raw split-half waveforms          ris_run_session.sh
#   job 2 (CPU)  agent pipeline -> consolidate -> bombcell ->
#                corrected classification -> trodes lookup ->
#                raw cluster points + raw separation metric       ris_run_postks.sh
#
# Job 2 is held by LSF until job 1 finishes successfully (-w "done(<id>)"), so the two
# run back to back against ONE staging of the raw file and no human has to be awake in
# between.
#
#   ssh compute1
#   bash /storage2/fs1/akepecs/Active/Gil_Zur/code/ris_submit_all.sh 20260915_103710
#
#   Arguments: <session-id> [Th_universal] [Th_learned] [gain_to_uV]
#
# WHY THIS IS THE DEFAULT WAY TO SUBMIT A SESSION
# -----------------------------------------------
# The unit of work is stage -> sort -> build review data -> copy back -> purge, not
# "sort everything, review later" (RESEARCH_LOG 5bd.20). Everything in job 2 reads the
# raw .dat, which exists on scratch for a few days. Deferring it to the Neuropixels PC
# costs a ~250 GB download per session and then runs on a 64 GB machine that is usually
# also recording. Submitting the sort WITHOUT the post-KS job is the exception, not the
# normal case -- so it needs its own command (ris_submit_session.sh), and this one is
# what a session is submitted with.
#
# After both jobs: bring the outputs home, THEN purge scratch. The display half of the
# pipeline (session timeline, merge candidates, review HTML) runs on the PC, because it
# needs the DIO and the Bpod file and ends in a published artifact.

set -euo pipefail

SESSION="${1:?usage: $0 <session-id> [Th_universal] [Th_learned] [gain_to_uV]}"
TH_UNIVERSAL="${2:-9.0}"
TH_LEARNED="${3:-8.0}"
GAIN="${4:-0.018311105685598315}"

CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=============================================================="
echo "JOB 1 of 2 -- Kilosort4 + waveforms"
echo "=============================================================="
OUT1="$(bash "${CODE}/ris_submit_session.sh" "${SESSION}" "${TH_UNIVERSAL}" "${TH_LEARNED}" "${GAIN}")"
echo "${OUT1}"

# bsub says: Job <759350> is submitted to queue <general>.
JOB1="$(printf '%s\n' "${OUT1}" | sed -n 's/^Job <\([0-9][0-9]*\)>.*/\1/p' | head -1)"
if [ -z "${JOB1}" ]; then
    echo >&2
    echo "Could not read a job id out of the sort submission, so the post-KS job was NOT" >&2
    echo "submitted -- an unchained one would start immediately and find no sort." >&2
    echo "Submit it by hand once the sort is queued:" >&2
    echo "  bash ${CODE}/ris_submit_postks.sh ${SESSION} ${GAIN} <job-id>" >&2
    exit 1
fi

echo
echo "=============================================================="
echo "JOB 2 of 2 -- post-Kilosort review pipeline, chained to ${JOB1}"
echo "=============================================================="
bash "${CODE}/ris_submit_postks.sh" "${SESSION}" "${GAIN}" "${JOB1}"

echo
echo "=============================================================="
echo "both queued for ${SESSION}.  bjobs -w   to watch them."
echo "job 2 stays PEND with a DEPEND reason until job 1 is DONE -- that is correct,"
echo "not a stuck job. If job 1 EXITs, job 2 never runs and LSF removes it."
echo "=============================================================="
