#!/bin/bash

set -e

project_dir=$(realpath "../projects-2017/L1-CNS")
cred_path="credentials_dev.json"
stack_id=1
#skel_id=18531745  # single-node skeleton on de moivre tile [9485, 7413, 582] on CLB's local instance
# skel_id=18531735  # small test skeleton only on CLB's local instance
skel_id=11524047 # real skeleton
force=1

timestamp=$(date +"%Y-%m-%d_%H:%M:%S")

log_root=${project_dir}/logs
log_dir=${log_root}/${timestamp}
mkdir -p ${log_dir}
ln -sfn ${log_dir} ${log_root}/latest

source ./set_env_vars.sh

script_path=`realpath $0`
cp ${script_path} ${log_dir}/run_script.sh
cp ./set_env_vars.sh ${log_dir}/set_env_vars.sh

echo `git rev-parse HEAD` > ${log_dir}/version.txt

./locate_syn_catmaid.py ${cred_path} ${stack_id} ${skel_id} ${project_dir} -f ${force} 2>&1 | tee \
${log_dir}/locate_synapses.txt;

grep 'PERFORMANCE' ${log_dir}/locate_synapses.txt > ${log_dir}/timing.txt

echo "FINISHED"
