#!/usr/bin/env bash
# The whole cluster-side run for ONE session, executed INSIDE the container by LSF.
#
# Two steps, in this order, in one job on purpose:
#
#   1. Kilosort4                -> $OUT/kilosort4
#   2. raw split-half waveforms -> $OUT/waveforms
#
# Step 2 shares the job with step 1 because it is the only moment the raw .dat is on
# scratch. Run later it costs a ~250 GB re-upload per session; run here it costs a few
# minutes. See scripts/extract_unit_waveforms.py for why those waveforms are the input
# to following a neuron across sessions, and why Kilosort's templates.npy is not.
#
# Step 2 is deliberately NOT allowed to fail the job: if the extraction breaks, the sort
# it just spent hours on must still be there to collect. It reports loudly instead.
#
# Called by ris_submit_session.sh; not meant to be run by hand.
#
#   ris_run_session.sh <session> <data.dat> <probe.json> <out-dir> <th_universal> <th_learned> <gain_to_uv> <threads>

set -uo pipefail

SESSION="${1:?session}"
DATA="${2:?data}"
PROBE="${3:?probe}"
OUT="${4:?out}"
TH_UNIVERSAL="${5:?th_universal}"
TH_LEARNED="${6:?th_learned}"
GAIN="${7:?gain_to_uv}"
THREADS="${8:-4}"

CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "================================================================"
echo "session   ${SESSION}"
echo "code      ${CODE}"
echo "data      ${DATA}"
echo "probe     ${PROBE}"
echo "out       ${OUT}"
echo "started   $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "================================================================"

mkdir -p "${OUT}"

echo
echo "### STEP 1/2  Kilosort4"
python3 "${CODE}/run_kilosort4_session.py" \
    --data "${DATA}" \
    --probe "${PROBE}" \
    --out "${OUT}/kilosort4" \
    --th-universal "${TH_UNIVERSAL}" \
    --th-learned "${TH_LEARNED}" \
    --device cuda \
    --threads "${THREADS}"
KS_RC=$?

if [ "${KS_RC}" -ne 0 ]; then
    echo "KILOSORT FAILED (exit ${KS_RC}) -- not attempting the waveform extraction," >&2
    echo "because it has nothing to extract from." >&2
    exit "${KS_RC}"
fi

# A zero exit from Kilosort is necessary but not sufficient -- check it actually wrote
# the two files step 2 reads, so a silent partial write is caught here and not three
# steps downstream.
for f in spike_times.npy spike_clusters.npy channel_positions.npy; do
    [ -s "${OUT}/kilosort4/${f}" ] || { echo "Kilosort exited 0 but ${f} is missing or empty" >&2; exit 1; }
done

echo
echo "### STEP 2/2  raw split-half mean waveforms"
python3 "${CODE}/extract_unit_waveforms.py" \
    --ks-dir "${OUT}/kilosort4" \
    --data "${DATA}" \
    --out "${OUT}/waveforms" \
    --gain-to-uv "${GAIN}"
WF_RC=$?

if [ "${WF_RC}" -ne 0 ]; then
    echo
    echo "WAVEFORM EXTRACTION FAILED (exit ${WF_RC})." >&2
    echo "The sort in ${OUT}/kilosort4 is complete and must still be collected." >&2
    echo "Re-run step 2 on its own WHILE THE RAW FILE IS STILL ON SCRATCH:" >&2
    echo "  python3 ${CODE}/extract_unit_waveforms.py --ks-dir ${OUT}/kilosort4 \\" >&2
    echo "      --data ${DATA} --out ${OUT}/waveforms --gain-to-uv ${GAIN}" >&2
    echo "Once scratch is purged this is a 250 GB re-upload, so do not defer it." >&2
fi

echo
echo "================================================================"
echo "finished  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "kilosort  exit ${KS_RC}"
echo "waveforms exit ${WF_RC}"
du -sh "${OUT}"/* 2>/dev/null
echo "================================================================"

exit "${KS_RC}"
