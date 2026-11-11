#!/usr/bin/env bash
# Pull a finished cluster sort's comparison arrays down and score them against
# the sort already on F.
#
# Only three files are needed to judge agreement -- spike_times, spike_clusters
# and the KSLabel table, about 300 MB in total. The big ones (pc_features is
# 6 GB on its own) stay on the cluster until the sort is accepted; there is no
# point moving 20 GB to find out whether the 300 MB agrees.
#
#     bash fetch_and_compare_cluster_sort.sh 20260916_110311 Shamir
#
# Arguments: <session-id> <rat>   (rat picks the reference path on F)

set -euo pipefail

SESSION="${1:?usage: $0 <session-id> <rat>}"
RAT="${2:?missing rat -- F: and Z: name the animals differently, see archive_session_when_exported.py}"

case "$RAT" in
    Shamir) F_DIR="Shamir" ;;
    Popper) F_DIR="Popper" ;;
    Begin)  F_DIR="Begin"  ;;
    *) echo "unknown rat $RAT" >&2; exit 1 ;;
esac

REMOTE="/storage2/fs1/akepecs/Active/Gil_Zur/kilosort4/${SESSION}"
LOCAL="D:/Gil/cluster_sorts/${SESSION}"
# kilosort_original/ holds the pristine arrays, before any curation or merges --
# which is the only fair thing to compare a fresh sort against. The reference
# phy dir itself has had verdicts applied to cluster_info.tsv since September.
REF="F:/Gil/${F_DIR}/${SESSION}.rec/${SESSION}.kilosort/kilosort4/kilosort_original"
OUT="D:/Gil/spike_sorting_agent/outputs/cluster_vs_local_${SESSION}.csv"

PY="/c/Users/Adam/anaconda3/python.exe"
export PATH="/c/Users/Adam/anaconda3:/c/Users/Adam/anaconda3/Library/bin:$PATH"

mkdir -p "$LOCAL"

echo "=== checking the sort actually finished ==="
# spike_times.npy is written last-ish, but ops.npy and the TSVs only appear once
# KS4 has saved everything -- so require the whole set rather than just one file.
for f in spike_times.npy spike_clusters.npy cluster_KSLabel.tsv; do
    n=$(ssh -o BatchMode=yes compute1 "stat -c %s ${REMOTE}/${f} 2>/dev/null" 2>/dev/null || true)
    if [ -z "$n" ]; then
        echo "  MISSING on the cluster: ${f}" >&2
        echo "  the sort has not finished writing. Nothing fetched." >&2
        exit 2
    fi
    echo "  ${f}  ${n} bytes"
done

echo
echo "=== fetching ==="
for f in spike_times.npy spike_clusters.npy cluster_KSLabel.tsv; do
    scp -o BatchMode=yes -q "compute1:${REMOTE}/${f}" "${LOCAL}/${f}"
    r=$(ssh -o BatchMode=yes compute1 "stat -c %s ${REMOTE}/${f}" 2>/dev/null)
    l=$(stat -c %s "${LOCAL}/${f}")
    if [ "$r" != "$l" ]; then
        echo "  SIZE MISMATCH on ${f}: remote=${r} local=${l}" >&2
        exit 3
    fi
    echo "  ${f} ok (${l} bytes)"
done

echo
echo "=== comparing against the sort on F ==="
echo "  reference ${REF}"
echo "  test      ${LOCAL}"
echo
[ -d "$REF" ] || { echo "reference sort not found: $REF" >&2; exit 4; }

"$PY" D:/Gil/spike_sorting_agent/scripts/compare_two_sorts.py \
    --ref "$REF" --test "$LOCAL" --out "$OUT"

echo
echo "per-unit CSV: $OUT"
