#!/usr/bin/env bash
# Build the POST-KILOSORT container on the RIS cluster and push it to Docker Hub.
#
# Same route and same reasoning as ris_build_image.sh (which builds the Kilosort image):
# there is no docker client on the submission host or on the Neuropixels PC, so RIS's own
# esub.docker_build does the build on the cluster. The image carries nothing but pinned
# open-source packages -- the lab's scripts are mounted at run time -- so the Docker Hub
# repository is public and the cluster needs no stored credential to pull it.
#
# Why a second image rather than one: the two halves cannot share an interpreter.
# Kilosort 4.0.24 pins python 3.9 / numpy 1.26.4; bombcell 0.89 and pandas 3.0.5 need
# python 3.11 / numpy 2.4.6. Both pins are load-bearing for comparability with the
# sessions already processed on the PC. The full argument is in docker/Dockerfile.postks.
#
# As with the Kilosort image this script RUNS NOTHING. It prints the commands in order:
# step 1 needs a password typed at a terminal, step 2 is a real LSF submission.

set -euo pipefail

DOCKERHUB_USER="${DOCKERHUB_USER:-zurgil}"

IMAGE="${DOCKERHUB_USER}/spike-postks:bc0.89-py311"
GROUP="compute-akepecs"
CODE_DIR="/storage2/fs1/akepecs/Active/Gil_Zur/code"
BUILD_DIR="/storage2/fs1/akepecs/Active/Gil_Zur/build_postks"

cat <<EOF
Target image : ${IMAGE}
Compute group: ${GROUP}
Build context: ${BUILD_DIR}

-----------------------------------------------------------------------------
STEP 0 -- put the build context and the code on the cluster (run from this PC)
-----------------------------------------------------------------------------
The image copies nothing in, so only the Dockerfile goes to the build dir. The
pipeline's own code is mounted at run time and therefore has to be up to date in
${CODE_DIR} -- including striatal_agent/, which the post-KS scripts import.

    ssh compute1 "mkdir -p ${BUILD_DIR} ${CODE_DIR}/striatal_agent"
    scp docker/Dockerfile.postks compute1:${BUILD_DIR}/Dockerfile
    scp scripts/ris_run_postks.sh scripts/ris_submit_postks.sh scripts/ris_submit_all.sh \\
        scripts/run_pipeline_batch_session.py scripts/gather_worker_batches.py \\
        scripts/consolidate_batches_session.py scripts/run_bombcell_session.py \\
        scripts/corrected_classification_session.py \\
        scripts/build_trodes_channel_lookup_session.py \\
        scripts/raw_cluster_points_session.py scripts/raw_separation_metric_session.py \\
        compute1:${CODE_DIR}/
    scp striatal_agent/*.py compute1:${CODE_DIR}/striatal_agent/

-----------------------------------------------------------------------------
STEP 1 -- Docker Hub login, once per cluster account (interactive)
-----------------------------------------------------------------------------
Already done for the Kilosort image on 2026-11-10; the credential is stored on the
cluster and covers this push too. Only needed again if it was cleared.

    ssh -t compute1
    export LSB_DOCKER_LOGIN_ONLY=1
    bsub -Is -G ${GROUP} -q general-interactive -a 'docker_build' -- --help

-----------------------------------------------------------------------------
STEP 2 -- build and push (a real LSF submission)
-----------------------------------------------------------------------------
The bare '--' matters: without it bsub parses the docker build arguments as its own.

    ssh compute1
    bsub -G ${GROUP} -q general \\
         -n 4 -R 'rusage[mem=16GB] span[hosts=1]' \\
         -o ${BUILD_DIR}/build.%J.log \\
         -a 'docker_build' \\
         -- --tag ${IMAGE} ${BUILD_DIR}

Watch it:     ssh compute1 bjobs
Read the log: ssh compute1 cat ${BUILD_DIR}/build.<jobid>.log

The build asserts its own pins (python 3.11, numpy 2.4.6, scipy 1.17.1, pandas 3.0.5,
matplotlib 3.11.1) and imports bombcell, so a green build is a guarantee rather than
just an absence of errors. These are exactly the versions in D:\\conda_envs\\bombcell on
the Neuropixels PC, read off it on 2026-11-12 -- bombcell's metrics are thresholded
downstream against values calibrated on 20260901_085606, so a different bombcell moves
every verdict without saying so.

-----------------------------------------------------------------------------
STEP 3 -- prove the image runs
-----------------------------------------------------------------------------
    ssh compute1
    bsub -Is -G ${GROUP} -q general-interactive -R 'rusage[mem=8GB]' \\
         -a "docker(${IMAGE})" \\
         python -c "import bombcell, numpy, pandas; print(bombcell.__name__, numpy.__version__, pandas.__version__)"

-----------------------------------------------------------------------------
Then a session's post-KS half is one command:
-----------------------------------------------------------------------------
    bash ${CODE_DIR}/ris_submit_postks.sh <session-id> [gain] [after-job-id]

or, for a session not yet sorted, both halves chained:

    bash ${CODE_DIR}/ris_submit_all.sh <session-id>

EOF
