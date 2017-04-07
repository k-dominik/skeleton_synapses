#!/bin/bash
project_dir="../projects-2017/L1-CNS"
cred_path="credentials_real.json"
stack_id=1
skel_id=11524047
force=1

timestamp=$(date +"%Y-%m-%d_%H:%M:%S")

log_dir=${project_dir}/logs/${timestamp}
mkdir -p ${log_dir}

source ./set_env_vars.sh

script_path=`realpath $0`
cp ${script_path} ${log_dir}/run_script.sh
cp ./set_env_vars.sh ${log_dir}/set_env_vars.sh

echo `git rev-parse HEAD` > ${log_dir}/version.txt

echo "Started at ${timestamp}" > ${log_dir}/time.txt

./locate_synapses.py ${cred_path} ${stack_id} ${skel_id} ${project_dir} -f ${force} 2>&1 | tee ${log_dir}/locate_synapses.txt;

echo "Segmentation finished at $(date +"%Y-%m-%d_%H:%M:%S")" >> ${log_dir}/time.txt

./results_to_volume.py ${cred_path} ${stack_id} ${skel_id} ${project_dir}/skeletons \
${project_dir}/synapse_volume.hdf5 -f ${force} 2>&1 | tee ${log_dir}/results_to_volume.txt;

echo "Data committing finished at $(date +"%Y-%m-%d_%H:%M:%S")" >> ${log_dir}/time.txt

echo "FINISHED"