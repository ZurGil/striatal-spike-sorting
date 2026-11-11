#!/usr/bin/env bash
# Extract raw split-half mean waveforms for a session whose sort ALREADY EXISTS, while its
# raw .dat is still on scratch.
#
# ris_submit_session.sh chains the extraction onto a fresh sort, which is the normal path.
# This script exists for the sessions sorted BEFORE the extractor was written: their raw
# data is still staged on scratch, and that is the only window in which the waveforms are
# cheap. After scratch is purged the same file costs a 250-300 GB re-upload.
#
# WHY --ks-dir IS AN ARGUMENT AND NOT DERIVED FROM THE SESSION
# -----------------------------------------------------------
# A session can have more than one sort, and the waveforms are only useful if their unit
# numbering matches the sort whose units are being matched or curated. 20260916_110311 has
# two: the curated one from 2026-09-17 (Kilosort 4.1.7, on Z, the one Gil's verdicts are
# keyed to) and the cluster's own from 2026-11-10 (4.0.24). Pointing this at the wrong one
# produces a file that silently describes different units.
#
#   usage: ris_submit_waveforms.sh <session> <ks-dir> <out-dir> [gain_to_uV]
#
# No GPU: the extraction is memmap reads and averaging. Asking for one would queue behind
# the GPU hosts and hold a card doing nothing.

set -euo pipefail

SESSION="${1:?usage: $0 <session> <ks-dir> <out-dir> [gain_to_uV]}"
KSDIR="${2:?ks-dir}"
OUT="${3:?out-dir}"
GAIN="${4:-0.018311105685598315}"

GROUP="compute-akepecs"
IMAGE="zurgil/kilosort4:4.0.24-cu121"
QUEUE="general"
CORES=2
MEM="32GB"
WALLTIME=240

SCRATCH="/scratch1/fs1/akepecs/zur"
STORAGE="/storage2/fs1/akepecs/Active/Gil_Zur"
CODE="${STORAGE}/code"
DATA="${SCRATCH}/${SESSION}/${SESSION}.probe1.dat"
LOGDIR="${STORAGE}/logs"

mkdir -p "${LOGDIR}"

[ -f "${DATA}" ] || { echo "MISSING RAW: ${DATA}" >&2
                      echo "Scratch has been purged for this session; the waveforms now need a re-upload." >&2
                      exit 1; }

for f in spike_times.npy spike_clusters.npy channel_positions.npy; do
    [ -s "${KSDIR}/${f}" ] || { echo "MISSING: ${KSDIR}/${f}" >&2; exit 1; }
done

[ -e "${OUT}/unit_waveforms.npz" ] && { echo "REFUSING: ${OUT}/unit_waveforms.npz already exists." >&2; exit 1; }

BYTES=$(stat -c %s "${DATA}")
if [ $(( BYTES % (2 * 384) )) -ne 0 ]; then
    echo "REFUSING: ${DATA} is ${BYTES} bytes, not a whole number of 384-ch int16 samples." >&2
    exit 1
fi

export LSF_DOCKER_VOLUMES="${STORAGE}:${STORAGE} /scratch1/fs1/akepecs:/scratch1/fs1/akepecs"

echo "submitting waveform extraction for ${SESSION}"
echo "  ks-dir  ${KSDIR}"
echo "  data    ${DATA}  (${BYTES} bytes)"
echo "  out     ${OUT}"
echo "  gain    ${GAIN}"
echo

bsub \
    -G "${GROUP}" \
    -q "${QUEUE}" \
    -J "wf_${SESSION}" \
    -n "${CORES}" \
    -R "rusage[mem=${MEM}] span[hosts=1]" \
    -M "${MEM}" \
    -W "${WALLTIME}" \
    -o "${LOGDIR}/wf_${SESSION}_%J.out" \
    -e "${LOGDIR}/wf_${SESSION}_%J.err" \
    -a "docker(${IMAGE})" \
    python3 "${CODE}/extract_unit_waveforms.py" \
        --ks-dir "${KSDIR}" \
        --data "${DATA}" \
        --out "${OUT}" \
        --gain-to-uv "${GAIN}"
