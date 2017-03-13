#!/bin/bash
project_dir="../projects-2017/L1-CNS"
stack_id=1
skel_id=11524047

./locate_synapses.py credentials_real.json ${stack_id} ${skel_id} ${project_dir}/projects/full-vol-autocontext.ilp \
${project_dir}/projects/multicut/L1-CNS-multicut.ilp ${project_dir}/skeletons;

./results_to_volume.py credentials_real.json ${stack_id} ${project_dir}/skeletons/${skel_id}/synapse_cc.h5 \
${project_dir}/synapse_volume.hdf5;