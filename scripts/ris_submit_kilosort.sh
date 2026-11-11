#!/usr/bin/env bash
# Submit one session's Kilosort4 sort to WashU RIS Compute1 (LSF + Docker, one GPU).
#
# Every value in here is now measured or confirmed, not guessed:
#   group    compute-akepecs      (bsub -G; the AD group Charlotte granted on 2026-11-10)
#   queue    general              RUNLIMIT 40320 min = 28 days, JL/U 500
#   image    zurgil/kilosort4:4.0.24-cu121   built on the cluster, proven on a V100
#   gpu      one TeslaV100_SXM2 (32 GB) -- compute1-exec-195..196 carry four each
#   scratch  /scratch1/fs1/akepecs/zur   inputs live here; 53 TB free, 4.0 GB/s
#   storage  /storage2/fs1/akepecs/Active/Gil_Zur   results land here (persistent)
#
# The image deliberately does NOT contain the lab scripts -- that is what keeps it
# publishable on Docker Hub. run_kilosort4_session.py is mounted from storage2 at
# run time, so editing it there changes the next job with no rebuild.
#
# Run from a compute1 client host, never from the Neuropixels PC:
#     ssh compute1
#     bash ris_submit_kilosort.sh 20260916_110311 9.0 8.0
#
# Arguments: <session-id> <Th_universal> <Th_learned>
# The thresholds are per session on purpose -- see run_kilosort4_session.py.

set -euo pipefail

SESSION="${1:?usage: $0 <session-id> <Th_universal> <Th_learned>}"
TH_UNIVERSAL="${2:?missing Th_universal -- it is session-specific, do not guess}"
TH_LEARNED="${3:?missing Th_learned -- it is session-specific, do not guess}"

GROUP="compute-akepecs"
IMAGE="zurgil/kilosort4:4.0.24-cu121"
QUEUE="general"

# CPU side of a GPU job: KS4 still does its file I/O and some preprocessing on the
# host. 4 cores is plenty; asking for more only lengthens the queue wait.
CORES=4
MEM="64GB"       # the 120 s CPU benchmark peaked at 7.89 GB and is NOT a guide --
                 # KS4's host memory grows with the spike count, not the file size
WALLTIME=2880    # MINUTES. 48 h, comfortably over the ~4 h a GPU sort should take,
                 # and far under the queue's 28-day limit.

SCRATCH="/scratch1/fs1/akepecs/zur"
STORAGE="/storage2/fs1/akepecs/Active/Gil_Zur"
CODE="${STORAGE}/code"

DATA="${SCRATCH}/${SESSION}/${SESSION}.probe1.dat"
PROBE="${SCRATCH}/${SESSION}/${SESSION}_custom_neuropixels_map.json"
OUT="${STORAGE}/kilosort4/${SESSION}"
LOGDIR="${STORAGE}/logs"

mkdir -p "${LOGDIR}" "${STORAGE}/kilosort4"

# All three must exist before a multi-hour job is queued. The probe JSON especially:
# it is generated on the Windows box by make_probe_map.py (which needs the Trodes
# reader) and has to be copied across WITH the data. A missing or stale map is the
# single most likely staging mistake, and it silently produces a plausible-looking
# sort at the wrong depths.
for f in "${DATA}" "${PROBE}" "${CODE}/run_kilosort4_session.py"; do
    [ -f "${f}" ] || { echo "MISSING: ${f}" >&2; exit 1; }
done

# Refuse to overwrite a finished sort. Re-running is fine, but it has to be asked for.
if [ -e "${OUT}/spike_times.npy" ]; then
    echo "REFUSING: ${OUT} already holds a finished sort. Move it aside first." >&2
    exit 1
fi

# LSF only mounts what it is told to. Without this the container sees neither the
# staged data, nor the runner script, nor anywhere to write results.
export LSF_DOCKER_VOLUMES="${STORAGE}:${STORAGE} /scratch1/fs1/akepecs:/scratch1/fs1/akepecs"

echo "submitting ${SESSION}"
echo "  image     ${IMAGE}"
echo "  queue     ${QUEUE}  cores ${CORES}  mem ${MEM}  walltime ${WALLTIME} min  gpu 1"
echo "  data      ${DATA}  ($(stat -c %s "${DATA}") bytes)"
echo "  probe     ${PROBE}"
echo "  out       ${OUT}"
echo "  Th_universal=${TH_UNIVERSAL}  Th_learned=${TH_LEARNED}"
echo

bsub \
    -G "${GROUP}" \
    -q "${QUEUE}" \
    -J "ks4_${SESSION}" \
    -n "${CORES}" \
    -R "rusage[mem=${MEM}] span[hosts=1] select[gpuhost]" \
    -gpu "num=1:gmodel=TeslaV100_SXM2_32GB" \
    -M "${MEM}" \
    -W "${WALLTIME}" \
    -o "${LOGDIR}/ks4_${SESSION}_%J.out" \
    -e "${LOGDIR}/ks4_${SESSION}_%J.err" \
    -a "docker(${IMAGE})" \
    python3 "${CODE}/run_kilosort4_session.py" \
        --data "${DATA}" \
        --probe "${PROBE}" \
        --out "${OUT}" \
        --th-universal "${TH_UNIVERSAL}" \
        --th-learned "${TH_LEARNED}" \
        --device cuda \
        --threads "${CORES}"

echo
echo "queued. watch it with:"
echo "  bjobs -l"
echo "  tail -f ${LOGDIR}/ks4_${SESSION}_*.out"
