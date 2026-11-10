#!/usr/bin/env bash
# Build the Kilosort4 container ON the RIS cluster and push it to Docker Hub.
#
# There is no docker client on compute1-client-1 (measured 2026-11-10), and
# there is none on the Neuropixels PC either, so neither machine can run
# `docker build`. RIS's own LSF integration is the way: esub.docker_build is
# present on the cluster and honours LSB_DOCKER_LOGIN_ONLY, LSB_DOCKER_OPTIONS
# and LSB_DOCKER_PUSH_URL.
#
# WashU has no private container registry, so the destination is Docker Hub.
# The image carries nothing but pinned open-source packages -- the lab's own
# scripts are mounted at run time, not baked in -- so a PUBLIC repository is
# fine and the cluster needs no credentials to pull it later.
#
# This script does NOT run anything by itself. It prints the two commands, in
# order, for a human to run. Step 1 needs a password typed at a terminal and
# cannot be automated; step 2 is a real LSF submission and therefore needs
# Gil's go-ahead.

set -euo pipefail

: "${DOCKERHUB_USER:?set DOCKERHUB_USER to the Docker Hub account name}"

IMAGE="${DOCKERHUB_USER}/kilosort4:4.0.24-cu121"
GROUP="compute-akepecs"
CODE_DIR="/storage2/fs1/akepecs/Active/Gil_Zur/code"
BUILD_DIR="/storage2/fs1/akepecs/Active/Gil_Zur/build"

cat <<EOF
Target image : ${IMAGE}
Compute group: ${GROUP}
Build context: ${BUILD_DIR}

-----------------------------------------------------------------------------
STEP 0 -- put the build context on the cluster (run from this PC)
-----------------------------------------------------------------------------
Only the Dockerfile is needed; the image copies nothing else in.

    ssh compute1 "mkdir -p ${BUILD_DIR} ${CODE_DIR}"
    scp docker/Dockerfile.kilosort4.gpu compute1:${BUILD_DIR}/Dockerfile
    scp scripts/run_kilosort4_session.py compute1:${CODE_DIR}/

-----------------------------------------------------------------------------
STEP 1 -- log in to Docker Hub, once (interactive, needs a terminal)
-----------------------------------------------------------------------------
This stores a credential on the cluster so step 2 can push. It builds nothing.
It must be interactive (-Is) because docker login prompts for a password, and
that password is the DOCKER HUB one, not the WUSTL Key.

    ssh -t compute1
    export LSB_DOCKER_LOGIN_ONLY=1
    bsub -Is -G ${GROUP} -q general-interactive -a 'docker_build' -- --help

-----------------------------------------------------------------------------
STEP 2 -- build and push (a real LSF submission -- needs Gil's go-ahead)
-----------------------------------------------------------------------------
Note the bare '--': without it bsub tries to parse the docker build arguments
as its own and fails. Everything after it belongs to docker build.

    ssh compute1
    bsub -G ${GROUP} -q general \\
         -n 4 -R 'rusage[mem=16GB] span[hosts=1]' \\
         -o ${BUILD_DIR}/build.%J.log \\
         -a 'docker_build' \\
         -- --tag ${IMAGE} ${BUILD_DIR}

Watch it with:    ssh compute1 bjobs
Read the log:     ssh compute1 cat ${BUILD_DIR}/build.<jobid>.log

The build asserts its own pins (kilosort 4.0.24, torch 2.4.0+cu121, numpy
1.26.4) and FAILS the build if any of them drifted. A green build is therefore
a real guarantee, not just an absence of errors.

-----------------------------------------------------------------------------
STEP 3 -- prove the image runs, and that it can see a GPU
-----------------------------------------------------------------------------
    ssh compute1
    bsub -Is -G ${GROUP} -q general-interactive \\
         -R 'gpuhost rusage[mem=8GB]' -gpu "num=1" \\
         -a "docker(${IMAGE})" \\
         python -c "import torch, kilosort; print(kilosort.__version__, torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

Expected:  4.0.24 2.4.0+cu121 True Tesla V100-SXM2-32GB

If torch.cuda.is_available() is False the job landed somewhere without a GPU,
or the -gpu request was dropped. Do not proceed -- a silent CPU fallback turns
4.5 hours into 6 days and reports nothing wrong.

-----------------------------------------------------------------------------
Running a real sort then looks like this (scripts mounted, not baked in):
-----------------------------------------------------------------------------
    export LSF_DOCKER_VOLUMES="${CODE_DIR}:/opt/spike_sorting_agent /scratch1/fs1/akepecs/zur:/data"
    bsub -G ${GROUP} -q general \\
         -R 'gpuhost rusage[mem=64GB] span[hosts=1]' -gpu "num=1" \\
         -a "docker(${IMAGE})" \\
         python /opt/spike_sorting_agent/run_kilosort4_session.py ...

EOF
