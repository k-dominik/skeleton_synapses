#!/usr/bin/env python
from __future__ import division

import argparse
import logging

import numpy as np
import psutil
import signal
import sys
from catpy import CatmaidClient
from tqdm import tqdm

from helpers.files import hash_algorithm
from skeleton_synapses.catmaid_interface import CatmaidSynapseSuggestionAPI
from skeleton_synapses.constants import DEFAULT_ROI_RADIUS, TQDM_KWARGS, DEBUG, LOG_LEVEL, THREADS
from skeleton_synapses.dto import NeuronSegmenterInput
from skeleton_synapses.helpers.files import ensure_list, Paths, get_algo_notes, TILE_SIZE
from skeleton_synapses.helpers.logging_ss import setup_logging, Timestamper
from skeleton_synapses.helpers.roi import nodes_to_tile_indexes, roi_around_synapse
from skeleton_synapses.parallel.process import DetectorProcess, NeuronSegmenterProcess, ProcessRunner
from skeleton_synapses.parallel.queues import (
    commit_tilewise_results_from_queue, commit_node_association_results_from_queue,
    populate_tile_input_queue, populate_synapse_queue
)

logger = logging.getLogger(__name__)


def main(
        credentials_path, stack_id, skeleton_ids, input_file_dir, output_file_dir,
        roi_radius_px=DEFAULT_ROI_RADIUS, force=False
):
    logger.info("STARTING TILEWISE")

    catmaid = CatmaidSynapseSuggestionAPI(CatmaidClient.from_json(credentials_path), stack_id)
    stack_info = catmaid.get_stack_info(stack_id)

    skeleton_ids = ensure_list(skeleton_ids)

    paths = Paths(input_file_dir, output_file_dir)
    paths.initialise(catmaid, stack_info, skeleton_ids, force)

    algo_notes = get_algo_notes(paths.projects_dir)

    if force:
        logger.info('Using random hash')
        algo_hash = hash(np.random.random())
    else:
        algo_hash = hash_algorithm(paths.autocontext_ilp, paths.multicut_ilp)

    for skeleton_id in tqdm(skeleton_ids, desc='Skeleton processing', unit='skeletons', **TQDM_KWARGS):
        locate_synapses(catmaid, paths, stack_info, skeleton_id, roi_radius_px, algo_hash, algo_notes)


def detect_synapses(catmaid, workflow_id, paths, stack_info, skeleton_id, roi_radius_px):
    node_infos = catmaid.get_node_infos(skeleton_id, stack_info['sid'])

    tile_queue, tile_count = populate_tile_input_queue(catmaid, roi_radius_px, workflow_id, node_infos)

    if tile_count:
        logger.info('Classifying pixels in {} tiles'.format(tile_count))

        detector_setup_args = paths, paths.skeleton_output_dir(skeleton_id), TILE_SIZE

        with ProcessRunner(tile_queue, DetectorProcess, detector_setup_args, min(THREADS, tile_count)) as runner:
            commit_tilewise_results_from_queue(
                runner.output_queue, paths.output_hdf5, tile_count, TILE_SIZE, workflow_id, catmaid
            )

    else:
        logger.debug('No tiles found (probably already processed)')
        tile_queue.close()


def associate_skeletons(catmaid, workflow_id, paths, stack_info, skeleton_id, roi_radius_px, algo_hash, algo_notes):
    project_workflow_id = catmaid.get_project_workflow_id(
        workflow_id, algo_hash, association_notes=algo_notes['skeleton_association']
    )

    synapse_queue, synapse_count = populate_synapse_queue(
        catmaid, roi_radius_px, project_workflow_id, stack_info, skeleton_id
    )

    # timestamper.log('finished getting synapse planes'.format(synapse_count))

    if synapse_count:
        logger.info('Segmenting {} synapse windows'.format(synapse_count))

        seg_setup_args = paths, catmaid

        with ProcessRunner(synapse_queue, NeuronSegmenterProcess, seg_setup_args, min(THREADS, synapse_count)) as runner:
            commit_node_association_results_from_queue(runner.output_queue, synapse_count, project_workflow_id, catmaid)

    else:
        logger.debug('No synapses required re-segmenting')
        synapse_queue.close()


def locate_synapses(catmaid, paths, stack_info, skeleton_id, roi_radius_px, algo_hash, algo_notes):
    """

    Parameters
    ----------
    autocontext_project_path : str
        .ilp file path
    multicut_project : str
        .ilp file path
    input_filepath : str
        Stack description JSON file
    output_file_dir : str
    skeleton : Skeleton
    roi_radius_px : int
        Default 150

    Returns
    -------

    """
    # todo: test (needs refactors)
    workflow_id = catmaid.get_workflow_id(
        stack_info['sid'], algo_hash, TILE_SIZE, detection_notes=algo_notes['synapse_detection']
    )

    logger.info('Populating tile queue')

    timestamper = Timestamper()

    timestamper.log('started detecting synapses')

    detect_synapses(catmaid, workflow_id, paths, stack_info, skeleton_id, roi_radius_px)

    timestamper.log('finished detecting synapses; started associating skeletons')

    associate_skeletons(catmaid, workflow_id, paths, stack_info, skeleton_id, roi_radius_px, algo_hash, algo_notes)

    logger.info("DONE with skeleton.")


def kill_child_processes(signum=None, frame=None):
    current_proc = psutil.Process()
    killed = []
    for child_proc in current_proc.children(recursive=True):
        logger.debug('Killing process: {} with status {}'.format(child_proc.name(), child_proc.status()))
        child_proc.kill()
        killed.append(child_proc.pid)
    logger.debug('Killed {} processes'.format(len(killed)))
    return killed


if __name__ == "__main__":
    if DEBUG:
        print("USING DEBUG ARGUMENTS")

        input_dir = "../projects-2017/L1-CNS"
        output_dir = input_dir
        cred_path = "credentials_dev.json"
        stack_id = 1
        skel_ids = [18531735]  # small test skeleton only on CLB's local instance

        force = 1

        args_list = [
            cred_path, stack_id, input_dir, skel_ids
        ]
        kwargs_dict = {'force': force}
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument('credentials_path',
                            help='Path to a JSON file containing CATMAID credentials (see credentials/example.json)')
        parser.add_argument('stack_id',
                            help='ID or name of image stack in CATMAID')
        parser.add_argument('input_dir', help="A directory containing project files.")
        parser.add_argument('skeleton_ids', nargs='+',
                            help="Skeleton IDs in CATMAID")
        parser.add_argument('-o', '--output_dir', default=None,
                            help='A directory containing output files')
        parser.add_argument('-r', '--roi_radius_px', default=DEFAULT_ROI_RADIUS,
                            help='The radius (in pixels) around each skeleton node to search for synapses')
        parser.add_argument('-f', '--force', type=int, default=0,
                            help="Whether to delete all prior results for a given skeleton: pass 1 for true or 0")

        args = parser.parse_args()
        output_dir = args.output_dir or args.input_file_dir
        args_list = [
            args.credentials_path, args.stack_id, args.skeleton_ids, args.input_file_dir, output_dir,
            args.roi_radius_px, args.force
        ]
        kwargs_dict = {}  # must be empty

    log_listener = setup_logging(output_dir, args_list, kwargs_dict, LOG_LEVEL)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.NOTSET)

    logger.info('STARTING CATMAID-COMPATIBLE DETECTION')

    signal.signal(signal.SIGTERM, kill_child_processes)

    exit_code = 1
    try:
        main(*args_list, **kwargs_dict)
        exit_code = 0
    except Exception as e:
        logger.exception('Errored, killing all child processes and exiting')
        kill_child_processes()
        raise
    finally:
        log_listener.stop()
        sys.exit(exit_code)
