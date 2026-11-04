#!/usr/bin/env bash
# Survey a lab compute node before installing anything on it.
#
# Read-only: this script inspects and prints, and changes nothing. Run it first
# thing after the first successful ssh, so the environment is known before any
# install decision is made. It answers most of the "Ask Sarah" list in
# docs/remote_kilosort_setup.md by measurement instead of by asking.
#
# Usage, from the Neuropixels PC:
#     ssh compute1 'bash -s' < scripts/survey_compute_node.sh
#
# Piping it in avoids having to copy the file over first, and leaves nothing
# behind on the node.

set -u   # not -e: a missing tool is an answer, not a reason to stop

line() { printf '\n== %s ==\n' "$1"; }
have() { command -v "$1" >/dev/null 2>&1; }

line "identity"
echo "host    : $(hostname -f 2>/dev/null || hostname)"
echo "user    : $(whoami)"
echo "uid     : $(id)"
echo "date    : $(date)"
echo "uptime  : $(uptime 2>/dev/null)"

line "os and kernel"
cat /etc/os-release 2>/dev/null | grep -E '^(NAME|VERSION)=' || echo "no /etc/os-release"
echo "kernel  : $(uname -r)"
echo "arch    : $(uname -m)"

line "cpu and memory"
if have lscpu; then lscpu | grep -E '^(Model name|CPU\(s\)|Socket|Thread|Core)' ; fi
free -h 2>/dev/null || echo "free unavailable"

line "gpu"
if have nvidia-smi; then
    nvidia-smi --query-gpu=index,name,memory.total,driver_version,compute_cap \
               --format=csv 2>/dev/null || nvidia-smi
    echo "-- current load (is anyone else using it?) --"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>/dev/null
else
    echo "nvidia-smi NOT FOUND -- either no GPU or the driver is not visible to this account"
fi

line "cuda toolkit"
if have nvcc; then nvcc --version | tail -2; else echo "nvcc not on PATH (fine -- pip torch ships its own CUDA runtime)"; fi

line "python and conda already present"
for p in python3 python conda mamba; do
    if have "$p"; then echo "$p -> $(command -v $p) : $($p --version 2>&1 | head -1)"; else echo "$p -> absent"; fi
done
python3 -c "import torch; print('system torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available())" 2>/dev/null \
    || echo "no system-wide torch (expected; we install our own)"

line "containers"
for c in docker apptainer singularity podman; do
    if have "$c"; then echo "$c -> $(command -v $c) : $($c --version 2>&1 | head -1)"; else echo "$c -> absent"; fi
done
# Being in the docker group is what decides whether docker is usable without root.
echo "groups  : $(groups 2>/dev/null)"

line "scheduler"
for s in sinfo squeue sbatch qsub bsub; do
    if have "$s"; then echo "$s -> $(command -v $s)"; else echo "$s -> absent"; fi
done
if have sinfo; then
    echo "-- partitions --"; sinfo -o '%P %a %l %D %G %N' 2>/dev/null | head -20
fi

line "disk: home, scratch, tmp"
echo "home    : $HOME"
df -h "$HOME" /tmp /scratch /local /data /mnt 2>/dev/null | sort -u
echo "-- quota, if enforced --"
quota -s 2>/dev/null || echo "quota command unavailable or no quota set"
echo "-- plausible scratch locations --"
for d in /scratch /local /localscratch /tmp /data /work "$HOME/scratch"; do
    [ -d "$d" ] && printf '%-20s writable=%s  %s\n' "$d" \
        "$([ -w "$d" ] && echo yes || echo no)" \
        "$(df -h "$d" 2>/dev/null | awk 'NR==2{print $4" free of "$2}')"
done

line "the archive share -- the decisive question"
# Kilosort needs probe1.dat, which already lives on this share. If the node can
# reach it, no data ever has to be pushed from the Neuropixels PC.
echo "-- is it already mounted somewhere? --"
mount 2>/dev/null | grep -i -E 'cifs|smb|nfs|172\.20\.22\.203|neuropixels' || echo "no SMB/NFS mount of it found"
echo "-- can the node even see the share host on 445? --"
if have nc; then
    nc -z -w 4 172.20.22.203 445 && echo "172.20.22.203:445 REACHABLE" || echo "172.20.22.203:445 not reachable"
elif have timeout; then
    timeout 4 bash -c 'cat < /dev/null > /dev/tcp/172.20.22.203/445' 2>/dev/null \
        && echo "172.20.22.203:445 REACHABLE" || echo "172.20.22.203:445 not reachable"
else
    echo "no nc and no timeout -- cannot test from here"
fi
echo "-- is an SMB client installed at all? --"
for t in mount.cifs smbclient; do
    if have "$t"; then echo "$t present"; else echo "$t absent"; fi
done

line "network back to the Neuropixels PC"
echo "-- can the node reach 172.20.22.115 (this PC)? useful for pull-style copies --"
ping -c 2 -W 2 172.20.22.115 2>/dev/null | tail -2 || echo "ping failed or blocked"

line "summary of what still needs a human"
cat <<'EOF'
Read the output above against docs/remote_kilosort_setup.md:
  - GPU + VRAM          -> answers Q6
  - scheduler section   -> answers Q7
  - containers/groups   -> answers Q8
  - scratch + quota     -> answers Q9, Q10
  - os/driver           -> answers Q11, and picks the torch CUDA build
  - archive share       -> answers Q5, which decides the whole data plan
Anything still blank after this is a genuine question for Sarah, not a guess.
EOF
