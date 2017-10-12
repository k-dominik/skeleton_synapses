#!/usr/bin/env python
from __future__ import division

import argparse
import hashlib
import logging
import multiprocessing as mp
import os
import signal
import subprocess
import sys
import time

import numpy as np
import psutil
from tqdm import tqdm
from catpy import CatmaidClient

from skeleton_synapses.catmaid_interface import CatmaidSynapseSuggestionAPI
from skeleton_synapses.constants import DEFAULT_ROI_RADIUS, TQDM_KWARGS, DEBUG, LOG_LEVEL, ALGO_HASH
from skeleton_synapses.dto import NeuronSegmenterInput
from skeleton_synapses.helpers.files import ensure_list, setup_files, ensure_hdf5, HDF5_NAME, TILE_SIZE
from skeleton_synapses.helpers.logging_ss import setup_logging, Timestamper
from skeleton_synapses.helpers.roi import nodes_to_tile_indexes
from skeleton_synapses.parallel.commit import commit_tilewise_results_from_queue, commit_node_association_results_from_queue
from skeleton_synapses.parallel.process import CaretakerProcess, DetectorProcess, NeuronSegmenterProcess


logger = logging.getLogger(__name__)


def main(credentials_path, stack_id, skeleton_ids, input_file_dir, output_file_dir, roi_radius_px=150, force=False):
    logger.info("STARTING TILEWISE")

    catmaid = CatmaidSynapseSuggestionAPI(CatmaidClient.from_json(credentials_path), stack_id)
    stack_info = catmaid.get_stack_info(stack_id)

    ensure_hdf5(stack_info, output_file_dir, force=force)

    skeleton_ids = ensure_list(skeleton_ids)

    autocontext_project, multicut_project, volume_description_path, skel_output_dirs, algo_notes = setup_files(
        credentials_path, stack_id, skeleton_ids, input_file_dir, force, output_file_dir if DEBUG else None
    )

    if force:
        logger.info('Using random hash')
        algo_hash = hash(np.random.random())
    else:
        algo_hash = hash_algorithm(autocontext_project, multicut_project)

    for skeleton_id in tqdm(skeleton_ids, desc='Skeleton processing', unit='skeletons', **TQDM_KWARGS):
        locate_synapses_catmaid(
            autocontext_project,
            multicut_project,
            volume_description_path,
            output_file_dir,
            skeleton_id,
            roi_radius_px,
            stack_info,
            algo_hash,
            algo_notes,
            catmaid
        )


def locate_synapses_catmaid(
        autocontext_project_path,
        multicut_project,
        input_filepath,
        output_file_dir,
        skeleton_id,
        roi_radius_px,
        stack_info,
        algo_hash,
        algo_notes,
        catmaid
):
    # todo: test (needs refactors)
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
    # NODES_PER_PROCESS = int(os.getenv('SYNAPSE_DETECTION_NODES_PER_PROCESS', 500))
    threads = int(os.getenv('SYNAPSE_DETECTION_THREADS', 3))
    logger.debug('Parallelising over {} threads'.format(threads))
    ram_MB_per_process = int(os.getenv('SYNAPSE_DETECTION_RAM_MB_PER_PROCESS', 5000))
    logger.debug('Will terminate subprocesses at {}MB of RAM'.format(ram_MB_per_process))

    hdf5_path = os.path.join(output_file_dir, HDF5_NAME)
    # debug
    # skel_output_dir = os.path.join(output_file_dir, 'skeletons', str(skeleton_id))
    skel_output_dir = None

    workflow_id = catmaid.get_workflow_id(
        stack_info['sid'], algo_hash, TILE_SIZE, detection_notes=algo_notes['synapse_detection']
    )

    logger.info('Populating tile queue')

    timestamper = Timestamper()

    timestamper.log('started getting tiles')

    node_infos = catmaid.get_node_infos(skeleton_id, stack_info['sid'])

    tile_index_set = nodes_to_tile_indexes(node_infos, TILE_SIZE, roi_radius_px)

    addressed_tiles = catmaid.get_detected_tiles(workflow_id)

    tile_queue, tile_result_queue = mp.Queue(), mp.Queue()
    tile_count = 0
    for tile_idx in tqdm(tile_index_set, desc='Populating tile queue', unit='tiles', **TQDM_KWARGS):
        if (tile_idx.x_idx, tile_idx.y_idx, tile_idx.z_idx) in addressed_tiles:
            logger.debug("Tile %s has been addressed by this algorithm, skipping", repr(tile_idx))
        else:
            logger.debug("Tile %s has not been addressed, adding to queue", repr(tile_idx))
            tile_count += 1
            tile_queue.put(tile_idx)

    timestamper.log('finished getting tiles')

    if tile_count:
        logger.info('Classifying pixels in {} tiles'.format(tile_count))

        detector_containers = [
            CaretakerProcess(
                DetectorProcess, tile_queue, ram_MB_per_process,
                (tile_result_queue, input_filepath, autocontext_project_path, skel_output_dir, TILE_SIZE),
                name='CaretakerProcess{}'.format(idx)
            )
            for idx in range(min(threads, tile_count))
        ]

        logger.debug('{} tiles queued'.format(tile_count))

        timestamper.log('started synapse detection ({} tiles, 512x512 each, {} threads)'.format(
            tile_count, len(detector_containers)
        ))

        while tile_queue.qsize() < min(threads, tile_count):
            logger.debug('Waiting for tile queue to populate...')
            time.sleep(1)

        for detector_container in detector_containers:
            detector_container.start()

        commit_tilewise_results_from_queue(tile_result_queue, hdf5_path, tile_count, TILE_SIZE, workflow_id, catmaid)

        for detector_container in detector_containers:
            detector_container.join()

        timestamper.log('finished synapse detection')
    else:
        logger.debug('No tiles found (probably already processed)')

    timestamper.log('started getting nodes')

    project_workflow_id = catmaid.get_project_workflow_id(
        workflow_id, algo_hash, association_notes=algo_notes['skeleton_association']
    )

    synapse_queue, synapse_result_queue = mp.Queue(), mp.Queue()
    synapse_count = 0

    roi_radius_nm = roi_radius_px * stack_info['resolution']['x']  # assumes XY isotropy
    logger.debug('Getting synapses spatially near skeleton {}'.format(skeleton_id))
    synapses_near_skeleton = catmaid.get_synapses_near_skeleton(skeleton_id, project_workflow_id, roi_radius_nm)
    nodes_of_interest = {node_info.id for node_info in node_infos}
    logger.debug('Found {} synapse planes near skeleton {}'.format(len(synapses_near_skeleton), skeleton_id))
    slice_id_tuples = set()
    for synapse in tqdm(synapses_near_skeleton, desc='Populating synapse plane queue', unit='synapse planes', **TQDM_KWARGS):
        if int(synapse['treenode_id']) not in nodes_of_interest:
            continue

        slice_id_tuple = tuple(synapse['synapse_slice_ids'])
        if slice_id_tuple in slice_id_tuples:
            continue

        slice_id_tuples.add(slice_id_tuple)

        radius = np.array([[-roi_radius_px, -roi_radius_px, 0], [roi_radius_px, roi_radius_px, 1]])

        # synapse plane bounds + buffer
        roi_xyz = (radius + np.array([
            synapse['synapse_bounds_s'][:2] + [synapse['synapse_z_s']],  # xmin, ymin, zmin
            synapse['synapse_bounds_s'][2:] + [synapse['synapse_z_s']]  # xmax, ymax, zmax
        ])).astype(int)

        logger.debug('Getting treenodes in roi {}'.format(roi_xyz))
        item = NeuronSegmenterInput(roi_xyz, slice_id_tuple)
        logger.debug('Adding {} to neuron segmentation queue'.format(item))
        synapse_queue.put(item)
        synapse_count += 1

    timestamper.log('finished getting synapse planes'.format(synapse_count))

    if synapse_count:
        logger.info('Segmenting {} synapse windows'.format(synapse_count))

        neuron_seg_containers = [
            CaretakerProcess(
                NeuronSegmenterProcess, synapse_queue, ram_MB_per_process,
                (synapse_result_queue, input_filepath, autocontext_project_path, multicut_project, hdf5_path, catmaid),
                name='CaretakerProcess{}'.format(idx)
            )
            for idx in range(min(threads, synapse_count))
            # for _ in range(1)
        ]

        timestamper.log('started segmenting neurons ({} items, {} threads)'.format(
            synapse_count, len(neuron_seg_containers)
        ))

        while synapse_queue.qsize() < min(threads, synapse_count):
            logger.debug('Waiting for node queue to populate...')
            time.sleep(1)

        for neuron_seg_container in neuron_seg_containers:
            neuron_seg_container.start()
            assert neuron_seg_container.is_alive()

        commit_node_association_results_from_queue(synapse_result_queue, synapse_count, project_workflow_id, catmaid)

        for neuron_seg_container in neuron_seg_containers:
            neuron_seg_container.join()

        timestamper.log('finished segmenting neurons')
    else:
        logger.debug('No synapses required re-segmenting')

    logger.info("DONE with skeleton.")


def hash_algorithm(*paths):
    """
    Calculate a combined hash of the algorithm. Included for hashing are the commit hash of this repo, the hashes of
    any files whose paths are given, and the git commit hash inside any directories whose paths are given.

    Parameters
    ----------
    paths
        Paths to files or directories outside of this git repo which affect the algorithm

    Returns
    -------
    str
    """
    logger.info('Hashing algorithm...')
    commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip()

    md5 = hashlib.md5(commit_hash)

    for path in sorted(paths):
        if os.path.isdir(path):
            logger.debug('Getting git commit hash of directory %s', path)
            try:
                output = subprocess.check_output(['git', '-C', path, 'rev-parse', 'HEAD']).strip()
                md5.update(output)
            except subprocess.CalledProcessError:
                logger.exception('Error encountered while finding git hash of directory %s', path)
        elif os.path.isfile(path):
            logger.debug('Getting hash of file %s', path)
            with open(path, 'rb') as f:
                for chunk in iter(lambda: f.read(128 * md5.block_size), b''):
                    md5.update(chunk)
        else:
            logger.warning('No file, symlink or directory found at %s', path)

    digest = md5.hexdigest()

    # todo: remove this
    if ALGO_HASH is not None:
        digest = ALGO_HASH
        logger.warning('Ignoring real algorithm hash, using hardcoded value'.format(ALGO_HASH))
    logger.debug('Algorithm hash is %s', digest)
    return digest


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
        parser.add_argument('input_file_dir', help="A directory containing project files.")
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
