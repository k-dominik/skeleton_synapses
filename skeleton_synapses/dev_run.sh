#!/bin/bash

set -e

project_dir="../projects-2017/L1-CNS"
cred_path="credentials_dev.json"
stack_id=1
skel_id=11524047  #18531735  # small test skeleton only on CLB's local instance
force=1

timestamp=$(date +"%Y-%m-%d_%H:%M:%S")

log_dir=${project_dir}/logs/${timestamp}
mkdir -p ${log_dir}

source ./set_env_vars.sh

script_path=`realpath $0`
cp ${script_path} ${log_dir}/run_script.sh
cp ./set_env_vars.sh ${log_dir}/set_env_vars.sh

echo `git rev-parse HEAD` > ${log_dir}/version.txt

./locate_syn_catmaid.py ${cred_path} ${stack_id} ${skel_id} ${project_dir} -f ${force} 2>&1 | tee \
${log_dir}/locate_synapses.txt;

grep 'PERFORMANCE_LOGGER' ${log_dir}/locate_synapses.txt > ${log_dir}/timing.txt

echo "FINISHED"