#!/bin/bash

set -e

ENV_NAME=$1

if [ -z "$ENV_NAME" ]; then
    >&2 echo "Pass environment name as first argument"
    exit 1
fi

# delete environment if it exists, noop if not
conda remove -y --name ${ENV_NAME} --all || :

# create, activate, and install ilastik in environment
conda create -y -n ${ENV_NAME} python=2.7 ipython
source activate ${ENV_NAME}
conda install -y -c ilastik ilastik-everything

# install other requirements
easy_install -U pip
STARTING_DIR=$(realpath $(dirname $0))
pip install -r ${STARTING_DIR}/requirements.txt

# fix GUI ilastik bugs
rm ${CONDA_PREFIX}/lib/libstdc++.*
REORDER_AXES_PATH=${CONDA_PREFIX}/ilastik-meta/lazyflow/lazyflow/operators
cd ${REORDER_AXES_PATH}
cp opReorderAxes.py opReorderAxes.pyBACKUP
patch -i ${STARTING_DIR}/opReorderAxes.patch
