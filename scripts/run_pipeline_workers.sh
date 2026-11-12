#!/usr/bin/env bash
# One pipeline worker: runs a contiguous range of batches in its OWN session root,
# each batch a fresh python process (WORKFLOW.md 1e -- the pipeline dies around unit
# 20-24 inside a single long-lived process regardless of free RAM).
#
#   run_pipeline_workers.sh <worker-index> <first-batch> <last-batch> <session-id>
#
# The session id is REQUIRED and has no default. It used to be hardcoded to
# 20260915_103710, which is the kind of default that quietly runs one session's
# batches against another session's raw file.
#
# Paths use FORWARD slashes on purpose. Windows Python accepts them, and an early
# version of this script was written through a bash heredoc, which halved the doubled
# backslashes and left "\${SESSION}" -- an escaped dollar -- so every batch was handed
# a root path containing the literal text ${SESSION} and failed instantly.
#
# Per-batch stdout goes to a log, not /dev/null: the first run of this hid the real
# error behind an exit code. Override LOGDIR in the environment to put them elsewhere.
set -uo pipefail

W="${1:?worker index}"
B0="${2:?first batch}"
B1="${3:?last batch}"
SESSION="${4:?session id, e.g. 20260915_084812_103710_concat}"

PAR="${PAR:-F:/Gil/Shamir/par}"
ROOT="${PAR}/w${W}/${SESSION}.rec"
PY="${PY:-/d/conda_envs/kilosort/python.exe}"
SCRIPT="D:/Gil/spike_sorting_agent/scripts/run_pipeline_batch_session.py"
LOGDIR="${LOGDIR:-D:/Gil/spike_sorting_agent/outputs/worker_logs/${SESSION}}"

if [ ! -d "${ROOT}" ]; then
    echo "worker ${W}: no such root ${ROOT} -- run setup_pipeline_workers.ps1 -Session ${SESSION} first"
    exit 2
fi

mkdir -p "${LOGDIR}"
echo "worker ${W}: batches ${B0}..${B1}"
echo "  root ${ROOT}"
echo "  logs ${LOGDIR}"

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

N=$(ls "${ROOT}/${SESSION}.kilosort/kilosort4/agent_processed/batch_reports/" 2>/dev/null | grep -c report)
echo "worker ${W} DONE $(date -u +%H:%M:%SZ) -- ${N} batch reports in this root"
if [ -n "${FAILED}" ]; then
    echo "worker ${W} FAILED BATCHES:${FAILED}"
fi
exit 0
