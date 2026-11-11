#!/usr/bin/env bash
# Submit one session's full cluster-side run to WashU RIS Compute1: Kilosort4 followed
# by the raw split-half waveform extraction, in a single job.
#
# This supersedes ris_submit_kilosort.sh, which submitted the sort alone. The only real
# differences are that it runs ris_run_session.sh instead of the Python runner directly,
# and that the thresholds now DEFAULT to Kilosort's stock values.
#
#   Run from a compute1 client host, never from the Neuropixels PC:
#       ssh compute1
#       bash /storage2/fs1/akepecs/Active/Gil_Zur/code/ris_submit_session.sh 20260915_103710
#
#   Arguments: <session-id> [Th_universal] [Th_learned] [gain_to_uV]
#
# WHY THE THRESHOLDS NOW HAVE DEFAULTS, WHEN THE RUNNER REFUSES TO
# ----------------------------------------------------------------
# run_kilosort4_session.py still requires them, and should: a default buried in the
# runner is how a wrong pair gets used silently. But the batch-wide decision (Gil,
# 2026-11-11) is Kilosort4's STOCK defaults for every session -- Th_universal=9,
# Th_learned=8, confirmed against kilosort.parameters.DEFAULT_SETTINGS -- so that the
# 31 sessions are comparable by construction and no hand-tuned number has to be
# defended later. The submit script is the right place for a batch-wide constant.
# 20260901 and 20260910 were sorted at 9.2/7.0 and are therefore NOT batch-consistent.
#
# WHY gain_to_uV IS AN ARGUMENT AND NOT A CONSTANT
# ------------------------------------------------
# It has been 0.018311105685598315 for every session checked so far, but that is an
# unchanged rig rather than a law. Read it from the session's OWN .rec header:
#   head -c 2000000 <session>.rec | tr -d '\0' | grep -o 'spikeScalingToUv="[0-9.e-]*"'

set -euo pipefail

SESSION="${1:?usage: $0 <session-id> [Th_universal] [Th_learned] [gain_to_uV]}"
TH_UNIVERSAL="${2:-9.0}"
TH_LEARNED="${3:-8.0}"
GAIN="${4:-0.018311105685598315}"

GROUP="compute-akepecs"
IMAGE="zurgil/kilosort4:4.0.24-cu121"
QUEUE="general"

CORES=4
MEM="64GB"
WALLTIME=2880     # minutes; 48 h, far over the few hours this takes and far under the
                  # queue's 28-day limit

SCRATCH="/scratch1/fs1/akepecs/zur"
STORAGE="/storage2/fs1/akepecs/Active/Gil_Zur"
CODE="${STORAGE}/code"

DATA="${SCRATCH}/${SESSION}/${SESSION}.probe1.dat"
PROBE="${SCRATCH}/${SESSION}/${SESSION}_custom_neuropixels_map.json"
OUT="${STORAGE}/kilosort4/${SESSION}"
LOGDIR="${STORAGE}/logs"

mkdir -p "${LOGDIR}" "${STORAGE}/kilosort4"

for f in "${DATA}" "${PROBE}" "${CODE}/run_kilosort4_session.py" \
         "${CODE}/extract_unit_waveforms.py" "${CODE}/ris_run_session.sh"; do
    [ -f "${f}" ] || { echo "MISSING: ${f}" >&2; exit 1; }
done

# A truncated upload is the failure that wastes the most time, because the sort succeeds
# and the result is quietly wrong at the end. The raw file must be a whole number of
# 384-channel int16 samples.
BYTES=$(stat -c %s "${DATA}")
if [ $(( BYTES % (2 * 384) )) -ne 0 ]; then
    echo "REFUSING: ${DATA} is ${BYTES} bytes, not a whole number of 384-ch int16 samples." >&2
    echo "That means the upload is incomplete. Re-check it against the size on Z." >&2
    exit 1
fi
HOURS=$(awk "BEGIN{printf \"%.3f\", ${BYTES}/2/384/30000/3600}")

if [ -e "${OUT}/kilosort4/spike_times.npy" ]; then
    echo "REFUSING: ${OUT} already holds a finished sort. Move it aside first." >&2
    exit 1
fi

export LSF_DOCKER_VOLUMES="${STORAGE}:${STORAGE} /scratch1/fs1/akepecs:/scratch1/fs1/akepecs"

echo "submitting ${SESSION}"
echo "  image      ${IMAGE}"
echo "  queue      ${QUEUE}  cores ${CORES}  mem ${MEM}  walltime ${WALLTIME} min  gpu 1"
echo "  data       ${DATA}"
echo "             ${BYTES} bytes = ${HOURS} h of 384-channel data"
echo "  probe      ${PROBE}"
echo "  out        ${OUT}"
echo "  thresholds Th_universal=${TH_UNIVERSAL}  Th_learned=${TH_LEARNED}  (Kilosort stock)"
echo "  gain_to_uV ${GAIN}"
echo "  steps      1) kilosort4   2) raw split-half waveforms"
echo

bsub \
    -G "${GROUP}" \
    -q "${QUEUE}" \
    -J "ks4wf_${SESSION}" \
    -n "${CORES}" \
    -R "rusage[mem=${MEM}] span[hosts=1] select[gpuhost]" \
    -gpu "num=1:gmem=8G" \
    -M "${MEM}" \
    -W "${WALLTIME}" \
    -o "${LOGDIR}/ks4wf_${SESSION}_%J.out" \
    -e "${LOGDIR}/ks4wf_${SESSION}_%J.err" \
    -a "docker(${IMAGE})" \
    bash "${CODE}/ris_run_session.sh" \
        "${SESSION}" "${DATA}" "${PROBE}" "${OUT}" \
        "${TH_UNIVERSAL}" "${TH_LEARNED}" "${GAIN}" "${CORES}"

echo
echo "queued. watch it with:"
echo "  bjobs -l"
echo "  tail -f ${LOGDIR}/ks4wf_${SESSION}_*.out"
