#!/usr/bin/env bash
# One pipeline worker: runs a contiguous range of batches in its OWN session root,
# each batch a fresh python process (WORKFLOW.md 1e -- the pipeline dies around unit
# 20-24 inside a single long-lived process regardless of free RAM).
#
#   run_worker.sh <worker-index> <first-batch> <last-batch>
#
# Paths use FORWARD slashes on purpose. Windows Python accepts them, and the previous
# version of this script was written through a bash heredoc, which halved the doubled
# backslashes and left "\${SESSION}" -- an escaped dollar -- so every batch was handed
# a root path containing the literal text ${SESSION} and failed instantly.
#
# Per-batch stdout goes to a log, not /dev/null: the first run of this hid the real
# error behind an exit code.
set -uo pipefail

W="${1:?worker index}"
B0="${2:?first batch}"
B1="${3:?last batch}"

SESSION=20260915_103710
ROOT="F:/Gil/Shamir/par/w${W}/${SESSION}.rec"
PY="/d/conda_envs/kilosort/python.exe"
SCRIPT="D:/Gil/spike_sorting_agent/scripts/run_pipeline_batch_session.py"
LOGDIR="C:/Users/Adam/AppData/Local/Temp/claude/D--Gil-spike-sorting-agent/8186c809-76fa-41ff-bbca-59d44125e5fc/scratchpad/worker_logs"

mkdir -p "${LOGDIR}"
echo "worker ${W}: batches ${B0}..${B1}"
echo "  root ${ROOT}"

FAILED=""
for B in $(seq "${B0}" "${B1}"); do
    S=$(( B * 10 ))
    E=$(( S + 10 ))
    printf '%s  w%s batch %s units[%s:%s]  ' "$(date -u +%H:%M:%SZ)" "${W}" "${B}" "${S}" "${E}"
    "${PY}" "${SCRIPT}" "${SESSION}" "${ROOT}" "${B}" "${S}" "${E}" \
        > "${LOGDIR}/w${W}_batch${B}.log" 2>&1
    RC=$?
    if [ "${RC}" -eq 0 ]; then
        echo "ok"
    else
        echo "FAILED rc=${RC}  see ${LOGDIR}/w${W}_batch${B}.log"
        FAILED="${FAILED} ${B}"
    fi
done

N=$(ls "F:/Gil/Shamir/par/w${W}/${SESSION}.rec/${SESSION}.kilosort/kilosort4/agent_processed/batch_reports/" 2>/dev/null | grep -c report)
echo "worker ${W} DONE $(date -u +%H:%M:%SZ) -- ${N} batch reports in this root"
if [ -n "${FAILED}" ]; then
    echo "worker ${W} FAILED BATCHES:${FAILED}"
fi
exit 0
