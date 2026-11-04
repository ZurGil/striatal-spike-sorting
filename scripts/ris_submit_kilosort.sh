#!/usr/bin/env bash
# Submit one session's Kilosort4 sort to WashU RIS Compute1 (LSF + Docker).
#
# TEMPLATE. The values marked PENDING are allocation-specific and must come from
# RIS / Sarah / Unmukt, not from a guess -- a wrong group or queue name fails
# immediately, which is fine, but a wrong walltime or memory request fails six
# hours in, which is not.
#
# Run this from a compute1 client host (NOT from the Neuropixels PC):
#     ssh <wustlkey>@compute1-client-1.ris.wustl.edu
#     bash ris_submit_kilosort.sh 20261021_111842 9.2 7.0
#
# Arguments: <session-id> <Th_universal> <Th_learned>
# The thresholds are per session on purpose -- see run_kilosort4_session.py.

set -euo pipefail

SESSION="${1:?usage: $0 <session-id> <Th_universal> <Th_learned>}"
TH_UNIVERSAL="${2:?missing Th_universal -- it is session-specific, do not guess}"
TH_LEARNED="${3:?missing Th_learned -- it is session-specific, do not guess}"

# ---------------------------------------------------------------------------
# PENDING -- fill these in from the allocation
# ---------------------------------------------------------------------------
GROUP="compute-PENDING"                 # bsub -G, the compute group
PI="PENDING"                            # PI key id, used in the storage paths
IMAGE="PENDING/kilosort4:4.0.24-cpu"    # registry path of the pushed image
QUEUE="general"                         # check `bqueues` for what exists
CORES=8                                 # see the scaling note below
MEM="64GB"                              # from the benchmark's peak RSS + headroom
WALLTIME=10080                          # MINUTES. 10080 = 7 days. Check the queue max.
# ---------------------------------------------------------------------------

STORAGE="/storage1/fs1/${PI}/Active"
SCRATCH="/scratch1/fs1/${PI}"

# Inputs are staged on scratch (fast, purged) and results land on storage
# (persistent, quota'd). Keeping the 300+ GB input off storage is what makes the
# 2 TB allocation hold more than a couple of sessions at a time.
DATA="${SCRATCH}/${SESSION}/${SESSION}.probe1.dat"
PROBE="${SCRATCH}/${SESSION}/${SESSION}_custom_neuropixels_map.json"
OUT="${STORAGE}/kilosort4/${SESSION}"
LOGDIR="${STORAGE}/logs"

mkdir -p "${LOGDIR}"

# Both must exist before a multi-day job is queued. The probe JSON especially:
# it is generated on the Windows box by make_probe_map.py (which needs the Trodes
# reader) and has to be copied across WITH the data. A missing map is the single
# most likely staging mistake.
for f in "${DATA}" "${PROBE}"; do
    [ -f "${f}" ] || { echo "MISSING: ${f}" >&2; exit 1; }
done

# A Kerberos ticket that expires mid-job breaks access to RIS storage, and a
# multi-day CPU sort is well past the default ticket lifetime. Renew before
# submitting; for jobs longer than the maximum renewable lifetime, write results
# to scratch and copy them to storage in a separate short job.
if command -v klist >/dev/null 2>&1; then
    klist -s || { echo "No valid Kerberos ticket -- run 'kinit' first." >&2; exit 1; }
fi

# LSF only mounts what it is told to. Without this the container sees neither
# the staged data nor anywhere to write results.
export LSF_DOCKER_VOLUMES="${STORAGE}:${STORAGE} ${SCRATCH}:${SCRATCH}"

echo "submitting ${SESSION}"
echo "  image     ${IMAGE}"
echo "  queue     ${QUEUE}  cores ${CORES}  mem ${MEM}  walltime ${WALLTIME} min"
echo "  data      ${DATA}"
echo "  out       ${OUT}"
echo "  Th_universal=${TH_UNIVERSAL}  Th_learned=${TH_LEARNED}"

bsub \
    -G "${GROUP}" \
    -q "${QUEUE}" \
    -J "ks4_${SESSION}" \
    -n "${CORES}" \
    -R "rusage[mem=${MEM}] span[hosts=1]" \
    -M "${MEM}" \
    -W "${WALLTIME}" \
    -o "${LOGDIR}/ks4_${SESSION}_%J.out" \
    -e "${LOGDIR}/ks4_${SESSION}_%J.err" \
    -a "docker(${IMAGE})" \
    python /opt/spike_sorting_agent/scripts/run_kilosort4_session.py \
        --data "${DATA}" \
        --probe "${PROBE}" \
        --out "${OUT}" \
        --th-universal "${TH_UNIVERSAL}" \
        --th-learned "${TH_LEARNED}" \
        --device cpu \
        --threads "${CORES}"

echo
echo "queued. watch it with:  bjobs -l    and    tail -f ${LOGDIR}/ks4_${SESSION}_*.out"
echo
cat <<'EOF'
Notes on CORES and span[hosts=1]:
  Kilosort4 on CPU is threaded within one process, so the job must land on ONE
  host -- hence span[hosts=1]. More cores help, but sub-linearly; measure with
  scripts/benchmark_kilosort_cpu.py on the node at a couple of thread counts
  before requesting a large allocation, because a wide request queues longer for
  a gain that may not be there.

Notes on throughput:
  The binding constraint is not cores, it is the storage allocation. At ~325 GiB
  of input plus ~20 GiB of output per session, 2 TB holds about 5 sessions in
  flight. Running more jobs than that means staging in waves: copy in, sort,
  copy results back, delete the input, copy the next one.
EOF
