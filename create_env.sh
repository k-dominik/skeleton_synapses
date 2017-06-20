#!/bin/bash

set -e

ENV_NAME=$1

conda create -n ${ENV_NAME} python=2.7 ipython
source activate ${ENV_NAME}
conda install -c ilastik ilastik-everything

easy_install -U pip
pip install -r $(realpath $(dirname $0))/requirements.txt
