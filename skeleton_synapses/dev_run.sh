#!/bin/bash
project_dir="../projects-2017/L1-CNS"
cred_path="credentials_dev.json"
stack_id=1
skel_id=11524047

./locate_synapses.py ${cred_path} ${stack_id} ${skel_id} ${project_dir};

./results_to_volume.py ${cred_path} ${stack_id} ${skel_id} ${project_dir}/skeletons \
${project_dir}/synapse_volume.hdf5;