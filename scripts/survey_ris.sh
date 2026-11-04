#!/usr/bin/env bash
# Survey the WashU RIS allocation from a compute1 client host.
#
# This replaces the generic-node survey: RIS is an LSF + Docker cluster, so the
# questions are answered by LSF commands (bqueues, lshosts, busers) rather than
# by poking at a machine's hardware. The login/client host is NOT where compute
# happens, so its own CPU and RAM are irrelevant -- what matters is what the
# scheduler will give us.
#
# Read-only. It submits nothing and changes nothing. Every command is an
# inspection command; the worst case is that one is unavailable and prints a
# note.
#
# Usage, once there is access:
#     ssh <wustlkey>@compute1-client-1.ris.wustl.edu 'bash -s' < scripts/survey_ris.sh
#
# Piping it in leaves nothing behind on the cluster.
#
# This answers, by measurement, questions 2-9 of
# docs/ris_questions_for_unmukt.md. Questions 1 (can we get an allocation before
# the 1 Jan 2027 cutoff) and 10 (is this a bad fit, in your experience) are for
# a human and cannot be measured.

set -u

line() { printf '\n=== %s ===\n' "$1"; }
have() { command -v "$1" >/dev/null 2>&1; }
note() { printf '    (%s)\n' "$1"; }

line "who am I, and which groups"
echo "host : $(hostname -f 2>/dev/null || hostname)"
echo "user : $(whoami)"
echo "date : $(date)"
id
# The compute group is what bsub -G needs, and the storage- group is what gates
# the SMB mount from the Windows PC.
echo "-- groups that look like RIS compute/storage groups --"
id -Gn 2>/dev/null | tr ' ' '\n' | grep -iE 'compute|storage' || note "none matched -- ask which group to use for bsub -G"

line "LSF cluster identity"
if have lsid; then lsid; else note "lsid not found -- is this really a compute1 client host?"; fi

line "QUESTION 6: how long may a job run, and how big may it be"
# RIS documentation says the general queue allows up to 28 days and
# general-interactive 24 h. Confirm that here rather than trusting the doc, and
# capture the core and memory ceilings at the same time.
if have bqueues; then
    echo "-- all queues we can see --"
    bqueues 2>/dev/null
    for q in general general-interactive; do
        echo
        echo "-- bqueues -l $q : RUNLIMIT is the walltime cap, look also for MAX/PROCLIMIT --"
        bqueues -l "$q" 2>/dev/null | grep -iE \
            'QUEUE|RUNLIMIT|CORELIMIT|PROCLIMIT|MEMLIMIT|USERS|HOSTS|MAX|JL/U|Maximum' \
            || note "queue $q not visible to this account"
    done
    echo
    echo "-- any queue whose name suggests a GPU --"
    bqueues 2>/dev/null | awk 'NR==1 || tolower($1) ~ /gpu/'
else
    note "bqueues not found"
fi

line "QUESTION 5: is there a GPU we can actually use"
# Kilosort4 needs only ~3.3 GB of VRAM, so any of these cards is more than
# enough -- the question is availability, not capability.
if have lshosts; then
    echo "-- hosts advertising a GPU resource --"
    lshosts -gpu 2>/dev/null | head -40 || note "lshosts -gpu unsupported here"
    echo
    echo "-- GPU models present, counted --"
    lshosts -gpu 2>/dev/null | awk 'NR>1{print $4, $5, $6}' | sort | uniq -c | sort -rn | head -20
fi
if have bhosts; then
    echo
    echo "-- how busy are the GPU hosts right now --"
    bhosts -gpu 2>/dev/null | head -30 || note "bhosts -gpu unsupported here"
fi
if have bmgroup; then
    echo
    echo "-- host groups (which nodes back which queue) --"
    bmgroup 2>/dev/null | grep -iE 'gpu|general' | head -10
fi
note "to request one later: -R 'gpuhost' -gpu \"num=1\"  on a GPU queue"

line "QUESTION 3: how many jobs may we run at once"
if have busers; then
    echo "-- MAX is the cap on running jobs for this user; JL/P is per-processor --"
    busers all 2>/dev/null | head -10
    busers 2>/dev/null
fi
if have bqueues; then
    echo "-- per-queue job limits for one user (JL/U) --"
    bqueues -l general 2>/dev/null | grep -iE 'JL/U|MAX|Maximum' || true
fi
if have bjobs; then
    echo "-- anything of ours already queued or running --"
    bjobs -u "$(whoami)" 2>/dev/null || note "no jobs"
fi

line "QUESTION 4: how many cores can one job get"
# Kilosort4 on CPU is threaded inside ONE process, so it cannot span hosts --
# the ceiling that matters is cores on a single node, not total cluster cores.
if have lshosts; then
    echo "-- the biggest single hosts, by core count (ncpus) --"
    lshosts 2>/dev/null | head -1
    lshosts 2>/dev/null | awk 'NR>1 && $5 ~ /^[0-9]+$/' | sort -k5 -rn | head -8
    note "Kilosort cannot span hosts -- span[hosts=1] -- so one node's core count is the cap"
fi

line "QUESTION 2: where does the 325 GB input file go"
# This is the one that decides how many sessions we can sort at once.
echo "-- home --"
echo "HOME=$HOME"
df -h "$HOME" 2>/dev/null
echo
echo "-- what storage is mounted --"
df -h 2>/dev/null | grep -iE 'storage1|scratch1|Filesystem' || note "no storage1/scratch1 mounts visible from this host"
echo
for root in /storage1/fs1 /scratch1/fs1; do
    echo "-- $root --"
    if [ -d "$root" ]; then
        ls -1 "$root" 2>/dev/null | head -20
        for d in "$root"/*/; do
            [ -d "$d" ] || continue
            printf '   %-44s writable=%s  %s\n' "$d" \
                "$([ -w "$d" ] && echo yes || echo NO)" \
                "$(df -h "$d" 2>/dev/null | awk 'NR==2{print $4" free of "$2}')"
        done
    else
        echo "   does not exist or not visible from this host"
    fi
done
echo
echo "-- scratch purge policy and usage, which RIS writes INTO the scratch dir --"
for f in /scratch1/fs1/*/RIS_Usage_Report.txt /scratch1/fs1/*/RIS_cleanup_policy*; do
    [ -f "$f" ] && { echo "--- $f ---"; head -25 "$f"; }
done 2>/dev/null
note "if those files exist they state the real purge window -- no need to ask"
echo
echo "-- quota, if the command exists --"
for c in "quota -s" "mmlsquota" "bquota"; do
    if have "${c%% *}"; then echo "\$ $c"; $c 2>/dev/null | head -15; fi
done

line "QUESTION 9: Kerberos ticket lifetime vs a multi-day job"
if have klist; then
    klist 2>/dev/null || note "no ticket right now"
    echo
    note "compare 'renew until' against a 5-6 day CPU job; if it is shorter, write"
    note "results to scratch and copy to storage in a separate short job"
else
    note "klist not found -- may not be a Kerberos-mediated storage setup after all"
fi

line "QUESTION 8: can we build or pull a Docker image"
echo "-- LSF docker integration variables currently set --"
env | grep -iE '^LSF_DOCKER' || note "none set (normal -- we set LSF_DOCKER_VOLUMES at submit time)"
echo "-- is a local docker client present on the client host? --"
for c in docker podman; do
    if have "$c"; then echo "$c -> $($c --version 2>&1 | head -1)"; else echo "$c -> absent (expected; RIS runs it for us via bsub -a docker())"; fi
done
note "if RIS must build the image, it is 'bsub -a docker_build' -- confirm the registry"

line "software already available (do we even need our own image?)"
for p in python3 python conda; do
    if have "$p"; then echo "$p -> $($p --version 2>&1 | head -1)"; else echo "$p -> absent"; fi
done
python3 -c "import torch,kilosort" 2>/dev/null \
    && echo "torch AND kilosort importable on the client host" \
    || note "no system kilosort -- our container is the right approach"

line "what still needs a human"
cat <<'EOF'
Measured above: questions 2-9 of docs/ris_questions_for_unmukt.md.
Still for Unmukt or Sarah:
  Q1  Can we get an allocation before Compute1 stops taking new General Access
      users on 1 January 2027? (And is ours General Access or a condo?)
  Q10 From your experience: does anything here fail in practice -- do long CPU
      jobs really survive, is the GPU queue usable, is Globus the only sane way
      to move 325 GB?
Plus, for the Windows side: which VPN, and can it run alongside the lab network
without disturbing the acquisition link.
EOF
