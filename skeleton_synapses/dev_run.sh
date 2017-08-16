#!/bin/bash

set -e

project_dir=$(realpath "../projects-2017/L1-CNS")
cred_path="credentials_dev.json"
stack_id=1
# skel_id=18531735  # small test skeleton only on CLB's local instance
skel_id=11524047 # real skeleton
force=1

source ./set_env_vars.sh
./locate_syn_catmaid.py ${cred_path} ${stack_id} ${project_dir} ${skel_id} -f ${force}
