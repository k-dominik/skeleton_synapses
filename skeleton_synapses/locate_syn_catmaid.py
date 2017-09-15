#!/usr/bin/env python
from __future__ import division
import logging
import multiprocessing as mp
from collections import namedtuple
import argparse
import sys
import os
import json
from itertools import product
import time
import hashlib
import subprocess
import signal

import psutil
from datetime import datetime

import h5py
import numpy as np
from six.moves import range
from skimage.morphology import skeletonize
from skimage.measure import find_contours
import vigra

from lazyflow.utility import Timer
from lazyflow.request import Request

from catpy import CatmaidClient

from catmaid_interface import CatmaidSynapseSuggestionAPI
from locate_synapses import (
    # constants/singletons
    DEFAULT_ROI_RADIUS, LOGGER_FORMAT,
    # functions
    setup_files, setup_classifier, setup_classifier_and_multicut, ensure_list, mkdir_p,
    fetch_raw_and_predict_for_node, raw_data_for_roi, labeled_synapses_for_node, segmentation_for_node,
    # classes
    CaretakerProcess, LeakyProcess,
    segmentation_for_img, cached_synapses_predictions_for_roi)
from skeleton_utils import roi_around_node


# def addapt_numpy_float64(numpy_float64):
#     return AsIs(numpy_float64)
# register_adapter(np.float64, addapt_numpy_float64)

LOG_LEVEL = logging.DEBUG

TIMESTAMP = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

HDF5_PATH = "../projects-2017/L1-CNS/tilewise_image_store.hdf5"

UNKNOWN_LABEL = 0  # should be smallest label
BACKGROUND_LABEL = 1  # should be smaller than synapse labels

LABEL_DTYPE = np.int64
PIXEL_PREDICTION_DTYPE = np.float32

TILE_SIZE = 512

THREADS = int(os.getenv('SYNAPSE_DETECTION_THREADS', 3))
# NODES_PER_PROCESS = int(os.getenv('SYNAPSE_DETECTION_NODES_PER_PROCESS', 500))
RAM_MB_PER_PROCESS = int(os.getenv('SYNAPSE_DETECTION_RAM_MB_PER_PROCESS', 5000))

DEBUG = False

ALGO_HASH = None

catmaid = None

last_event = time.time()
def log_timestamp(msg):
    global last_event
    now = time.time()
    performance_logger.info('{}: {}'.format(now - last_event, msg))
    last_event = now


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


def main(credentials_path, stack_id, skeleton_ids, project_dir, roi_radius_px=150, force=False):
    global catmaid

    logger.info("STARTING TILEWISE")

    log_timestamp('started setup')

    catmaid = CatmaidSynapseSuggestionAPI(CatmaidClient.from_json(credentials_path), stack_id)
    stack_info = catmaid.get_stack_info(stack_id)

    ensure_hdf5(stack_info, force=force)

    log_timestamp('finished setup')
    skeleton_ids = ensure_list(skeleton_ids)

    autocontext_project, multicut_project, volume_description_path, skel_output_dirs, algo_notes = setup_files(
        credentials_path, stack_id, skeleton_ids, project_dir, force
    )

    if force:
        logger.info('Using random hash')
        algo_hash = hash(np.random.random())
    else:
        algo_hash = hash_algorithm(autocontext_project, multicut_project)

    performance_logger.info('{}: finished setup')

    for skeleton_id, skel_output_dir in zip(skeleton_ids, skel_output_dirs):
        locate_synapses_catmaid(
            autocontext_project,
            multicut_project,
            volume_description_path,
            skel_output_dir,
            skeleton_id,
            roi_radius_px,
            stack_info,
            algo_hash,
            algo_notes
        )


def create_label_volume(stack_info, hdf5_file, name, tile_size=TILE_SIZE, dtype=LABEL_DTYPE, extra_dim=None):
    dimension = [stack_info['dimension'][dim] for dim in 'zyx']
    chunksize = (1, tile_size, tile_size)

    if extra_dim is not None:
        dimension += [extra_dim]
        chunksize += (extra_dim, )

    labels = hdf5_file.create_dataset(
        name,
        dimension,  # zyx(c)
        chunks=chunksize,  # zyx(c)
        fillvalue=0,
        dtype=dtype
    )

    for key in ['translation', 'dimension', 'resolution']:
        labels.attrs[key] = json.dumps(stack_info[key])

    return labels


def ensure_hdf5(stack_info, force=False):
    if force or not os.path.isfile(HDF5_PATH):
        if os.path.isfile(HDF5_PATH):
            os.rename(HDF5_PATH, '{}BACKUP{}'.format(HDF5_PATH, datetime.now().strftime('%Y-%m-%d_%H:%M:%S')))
        logger.info('Creating HDF5 volumes in %s', HDF5_PATH)
        with h5py.File(HDF5_PATH) as f:
            # f.attrs['workflow_id'] = workflow_id  # todo
            f.attrs['source_stack_id'] = stack_info['sid']

            create_label_volume(stack_info, f, 'slice_labels', TILE_SIZE)
            # create_label_volume(stack_info, f, 'object_labels')  # todo?
            create_label_volume(stack_info, f, 'pixel_predictions', TILE_SIZE, dtype=PIXEL_PREDICTION_DTYPE,
                                extra_dim=3)

            f.flush()


TileIndex = namedtuple('TileIndex', 'z_idx y_idx x_idx')


def nodes_to_tile_indexes(node_infos, tile_size, minimum_radius=DEFAULT_ROI_RADIUS):
    """

    Parameters
    ----------
    node_infos : skeleton_utils.NodeInfo
    tile_size
    minimum_radius

    Returns
    -------
    set of TileIndex
    """
    tile_set = set()
    tile_size_xyz = np.array([tile_size, tile_size, 1])

    logger.info('Getting tile set for nodes')

    for node_info in node_infos:
        bounds_xyz = roi_around_node(node_info, minimum_radius)
        tile_idxs = (bounds_xyz / tile_size_xyz).astype(int)
        tile_set.update(TileIndex(*idxs) for idxs in product(
            [node_info.z_px],  # z index
            range(tile_idxs[0, 1], tile_idxs[1, 1] + 1),  # all tile y indices
            range(tile_idxs[0, 0], tile_idxs[1, 0] + 1)  # all tile x indices
        ))

    return tile_set


def tile_index_to_bounds(tile_index, tile_size):
    """

    Parameters
    ----------
    tile_index : skeleton_utils.TileIndex
    tile_size : int

    Returns
    -------
    numpy.ndarray
        [[min_x, min_y, min_z], [max_x, max_y, max_z]] in pixel coordinates
    """
    tile_size_xyz = np.array([tile_size, tile_size, 1])
    topleft = np.array([tile_index.x_idx, tile_index.y_idx, tile_index.z_idx], dtype=int)
    return np.stack((topleft, topleft+1)) * tile_size_xyz  # todo: might need to -1 to bottom row


def square_bounds(roi_xyz):
    """Convert a rectangular ROI array into the minimum square in which the original ROI is centered"""
    shape = np.diff(roi_xyz[:, :2], axis=0).squeeze()
    size_diff = shape[0] - shape[1]
    if size_diff == 0:
        return roi_xyz
    elif size_diff > 0:
        half_diff = float(size_diff) / 2
        smaller_dim = 1
    else:
        half_diff = float(size_diff) / -2
        smaller_dim = 0

    roi_xyz[0, smaller_dim] -= np.floor(half_diff)
    roi_xyz[1, smaller_dim] += np.ceil(half_diff)

    return roi_xyz


def locate_synapses_catmaid(
        autocontext_project_path,
        multicut_project,
        input_filepath,
        skel_output_dir,
        skeleton_id,
        roi_radius_px,
        stack_info,
        algo_hash,
        algo_notes
):
    """

    Parameters
    ----------
    autocontext_project_path : str
        .ilp file path
    multicut_project : str
        .ilp file path
    input_filepath : str
        Stack description JSON file
    skel_output_dir : str
        {project_dir}/skeletons/{skel_id}/
    skeleton : Skeleton
    roi_radius_px : int
        Default 150

    Returns
    -------

    """
    global catmaid

    workflow_id = catmaid.get_workflow_id(
        stack_info['sid'], algo_hash, TILE_SIZE, detection_notes=algo_notes['synapse_detection'])

    logger.info('Populating tile queue')

    log_timestamp('started getting tiles')

    node_infos = catmaid.get_treenode_locations(skeleton_id, stack_info['sid'])

    tile_index_set = nodes_to_tile_indexes(node_infos, TILE_SIZE, roi_radius_px)

    addressed_tiles = catmaid.get_detected_tiles(workflow_id)

    tile_queue, tile_result_queue = mp.Queue(), mp.Queue()
    tile_count = 0
    for tile_idx in tile_index_set:
        if (tile_idx.x_idx, tile_idx.y_idx, tile_idx.z_idx) in addressed_tiles:
            logging.debug("Tile %s has been addressed by this algorithm, skipping", repr(tile_idx))
        else:
            logging.debug("Tile %s has not been addressed, adding to queue", repr(tile_idx))
            tile_count += 1
            tile_queue.put(tile_idx)

    log_timestamp('finished getting tiles')

    # don't save out individual tiles
    skel_output_dir = skel_output_dir if DEBUG else None

    if tile_count:
        logger.info('Classifying pixels in tilewise')

        detector_containers = [
            CaretakerProcess(
                DetectorProcess, tile_queue, RAM_MB_PER_PROCESS,
                (tile_result_queue, input_filepath, autocontext_project_path, skel_output_dir, TILE_SIZE),
                name='CaretakerProcess{}'.format(idx)
            )
            for idx in range(min(THREADS, tile_count))
        ]

        logger.debug('{} tiles queued'.format(tile_count))

        log_timestamp('started synapse detection ({} tiles, 512x512 each, {} threads)'.format(tile_count, THREADS))

        while tile_queue.qsize() < min(THREADS, tile_count):
            logger.debug('Waiting for tile queue to populate...')
            time.sleep(1)

        for detector_container in detector_containers:
            detector_container.start()

        commit_tilewise_results_from_queue(tile_result_queue, HDF5_PATH, tile_count, TILE_SIZE, workflow_id)

        for detector_container in detector_containers:
            detector_container.join()

        log_timestamp('finished synapse detection')
    else:
        logger.debug('No tiles found (probably already processed)')

    log_timestamp('started getting nodes')

    project_workflow_id = catmaid.get_project_workflow_id(
        workflow_id, algo_hash, association_notes=algo_notes['skeleton_association']
    )

    synapse_queue, synapse_result_queue = mp.Queue(), mp.Queue()
    node_count = 0

    roi_radius_nm = roi_radius_px * stack_info['resolution']['x']  # assumes XY isotropy
    logger.debug('Getting synapses spatially near skeleton {}'.format(skeleton_id))
    synapses_near_skeleton = catmaid.get_synapses_near_skeleton(skeleton_id, project_workflow_id, roi_radius_nm)
    logger.debug('Found {} synapse planes near skeleton {}'.format(len(synapses_near_skeleton), skeleton_id))
    slice_id_tuples = set()
    for synapse in synapses_near_skeleton:
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

        # make it into a square
        # roi_xyz = square_bounds(roi_xyz)

        # synapse plane centroid + buffer
        # centroid_xyz = np.array([
        #     synapse['synapse_bounds_s'][:2] + [synapse['synapse_z_s']],
        #     synapse['synapse_bounds_s'][2:] + [synapse['synapse_z_s']]
        # ]).mean(axis=0)
        # roi_xyz = (radius + centroid_xyz).astype(int)

        logger.debug('Getting treenodes in roi {}'.format(roi_xyz))
        # node_locations = catmaid.get_nodes_in_roi(roi_xyz, stack_info['sid'])
        item = NeuronSegmenterInput(roi_xyz, slice_id_tuple)
        logger.debug('Adding {} to neuron segmentation queue'.format(item))
        synapse_queue.put(item)
        node_count += 1

    log_timestamp('finished getting nodes')

    if node_count:
        logger.info('Segmenting synapse windows')

        neuron_seg_containers = [
            CaretakerProcess(
                NeuronSegmenterProcess, synapse_queue, RAM_MB_PER_PROCESS,
                (synapse_result_queue, input_filepath, autocontext_project_path, multicut_project, skel_output_dir),
                name='CaretakerProcess{}'.format(idx)
            )
            for idx in range(min(THREADS, node_count))
            # for _ in range(1)
        ]

        log_timestamp('started segmenting neurons ({} items, {} threads)'.format(node_count, THREADS))

        while synapse_queue.qsize() < min(THREADS, node_count):
            logger.debug('Waiting for node queue to populate...')
            time.sleep(1)

        for neuron_seg_container in neuron_seg_containers:
            neuron_seg_container.start()
            assert neuron_seg_container.is_alive()

        commit_node_association_results_from_queue(synapse_result_queue, node_count, project_workflow_id)

        for neuron_seg_container in neuron_seg_containers:
            neuron_seg_container.join()

        log_timestamp('finished segmenting neurons')
    else:
        logger.debug('No synapses required re-segmenting')

    logger.info("DONE with skeleton.")


DetectorOutput = namedtuple('DetectorOutput', 'tile_idx predictions_xyc synapse_cc_xyc')


class DetectorProcess(LeakyProcess):
    def __init__(
            self, input_queue, max_ram_MB, output_queue, description_file, autocontext_project_path, skel_output_dir,
            tile_size, debug=False, name=None
    ):
        super(DetectorProcess, self).__init__(input_queue, max_ram_MB, debug, name)
        self.output_queue = output_queue

        self.skel_output_dir = skel_output_dir
        self.tile_size = tile_size

        self.opPixelClassification = None

        self.setup_args = (description_file, autocontext_project_path)

    def setup(self):
        self.opPixelClassification = setup_classifier(*self.setup_args)
        Request.reset_thread_pool(1)  # todo: set to 0?

    def execute(self):
        tile_idx = self.input_queue.get()

        self.inner_logger.debug("Addressing tile {}; {} tiles remaining".format(tile_idx, self.input_queue.qsize()))

        with Timer() as timer:
            roi_xyz = tile_index_to_bounds(tile_idx, self.tile_size)

            # GET AND CLASSIFY PIXELS
            predictions_xyc = fetch_raw_and_predict_for_node(
                None, roi_xyz, self.skel_output_dir, self.opPixelClassification
            )
            # DETECT SYNAPSES
            synapse_cc_xy = labeled_synapses_for_node(None, roi_xyz, self.skel_output_dir, predictions_xyc)
            logging.getLogger(self.inner_logger.name + '.timing').info("NODE TIMER: {}".format(timer.seconds()))

        self.inner_logger.debug(
            "Detected synapses in tile {}; {} tiles remaining".format(tile_idx, self.input_queue.qsize())
        )

        self.output_queue.put(DetectorOutput(tile_idx, np.array(predictions_xyc), np.array(synapse_cc_xy)))


NeuronSegmenterInput = namedtuple('NeuronSegmenterInput', ['roi_xyz', 'synapse_slice_ids'])
NeuronSegmenterOutput = namedtuple('NeuronSegmenterOutput', ['node_id', 'synapse_slice_id', 'contact_px'])


# class NeuronSegmenterProcess(DebuggableProcess):
class NeuronSegmenterProcess(LeakyProcess):
    """
    Process which creates its own pixel classifier and multicut workflow, pulls jobs from one queue and returns
    outputs to another queue.
    """
    def __init__(
            self, input_queue, max_ram_MB, output_queue, description_file, autocontext_project_path, multicut_project,
            skel_output_dir, debug=False, name=None
    ):
        """

        Parameters
        ----------
        input_queue : mp.Queue
        output_queue : mp.Queue
        description_file : str
            path
        autocontext_project_path
        multicut_project : str
            path
        skel_output_dir : str
        debug : bool
            Whether to instantiate a serial version for debugging purposes
        """
        super(NeuronSegmenterProcess, self).__init__(input_queue, max_ram_MB, debug, name)
        # super(NeuronSegmenterProcess, self).__init__(debug)
        self.input_queue = input_queue
        self.output_queue = output_queue

        self.skel_output_dir = skel_output_dir

        self.opPixelClassification = self.multicut_shell = None
        self.setup_args = (description_file, autocontext_project_path, multicut_project)

    # for debugging
    # def run(self):
    #     self.setup()
    #
    #     while not self.input_queue.empty():
    #         self.execute()

    def setup(self):
        self.inner_logger.debug('Setting up opPixelClassification and multicut_shell...')
        self.opPixelClassification, self.multicut_shell = setup_classifier_and_multicut(
            *self.setup_args
        )
        self.inner_logger.debug('opPixelClassification and multicut_shell set up')

        Request.reset_thread_pool(1)

    def execute(self):
        roi_xyz, synapse_slice_ids = self.input_queue.get()
        self.inner_logger.debug("Addressing ROI {}; {} ROIs remaining".format(roi_xyz, self.input_queue.qsize()))

        with Timer() as node_timer:
            raw_xy = raw_data_for_roi(roi_xyz, None, self.opPixelClassification)
            synapse_cc_xy, predictions_xyc = cached_synapses_predictions_for_roi(roi_xyz, HDF5_PATH)

            log_str = 'Image shapes: \n\tRaw {}\n\tSynapse_cc {}\n\tPredictions {}'.format(
                raw_xy.shape, synapse_cc_xy.shape, predictions_xyc.shape
            )
            self.inner_logger.debug(log_str)
            logger.debug(log_str)
            print(log_str)

            segmentation_xy = segmentation_for_img(raw_xy, predictions_xyc, self.multicut_shell.workflow)

            overlapping_segments = dict()
            for synapse_slice_id in synapse_slice_ids:
                # todo: need to cast some types?
                segments = np.unique(segmentation_xy[synapse_cc_xy == synapse_slice_id])
                self.inner_logger.debug('Synapse slice {} overlaps with segments {}'.format(synapse_slice_id, segments))
                for overlapping_segment in segments:
                    if overlapping_segment not in overlapping_segments:
                        overlapping_segments[overlapping_segment] = set()
                    overlapping_segments[overlapping_segment].add(synapse_slice_id)

            if len(overlapping_segments) < 2:  # synapse is only in 1 segment
                self.inner_logger.debug(
                    'Synapse slice IDs {} in ROI {} are only in 1 neuron'.format(synapse_slice_ids, roi_xyz)
                )
                return

            node_locations = catmaid.get_nodes_in_roi(roi_xyz, catmaid.stack_id)
            node_locations_arr = node_locations_to_array(synapse_cc_xy, node_locations)

            not_nans = ~np.isnan(node_locations_arr)
            for segment, node_id in zip(segmentation_xy[not_nans], node_locations_arr[not_nans]):
                for synapse_slice_id in overlapping_segments[segment]:
                    contact_px = skeletonize((synapse_cc_xy == synapse_slice_id) * (segmentation_xy == segment)).sum()
                    self.output_queue.put(NeuronSegmenterOutput(node_id, synapse_slice_id, contact_px))

            logging.getLogger(self.inner_logger.name + '.timing').info("TILE TIMER: {}".format(node_timer.seconds()))

        self.inner_logger.debug('Adding segmentation output of ROI {} to output queue; {} nodes remaining'.format(
            roi_xyz, self.input_queue.qsize()
        ))


def node_locations_to_array(template_array_xy, node_locations):
    if not isinstance(template_array_xy, vigra.VigraArray):
        template_array_xy = vigra.taggedView(template_array_xy, axistags='xy')
    arr_xy = template_array_xy.copy().fill(np.nan)
    for node_location in node_locations.values():
        coords = node_location['coords']
        arr_xy[coords['x'], coords['y']] = int(node_location['treenode_id'])

    return arr_xy


def coords_to_multipoint_wkt_str(x_coords, y_coords):
    """
    Convert arrays of coordinates into a WKT string describing a MultiPoint geometry of those coordinates,
    where point i has the coordinates (x_coords[i], y_coords[i]).

    x_coords, y_coords = np.where(binary_array)

    Parameters
    ----------
    x_coords : array-like
        Array of x coordinates
    y_coords : array-like
        Array of y coordinates

    Returns
    -------
    str
        MultiPoint geometry expressed as a WKT string
    """
    coords_str = ','.join('{} {}'.format(x_coord, y_coord) for x_coord, y_coord in zip(x_coords, y_coords))
    # could use scipy to find contour to reduce workload on database, or SQL concave hull, or SQL simplify geometry
    return "MULTIPOINT({})".format(coords_str)


def iterate_queue(queue, final_size, queue_name=None):
    if queue_name is None:
        queue_name = repr(queue)
    for idx in range(final_size):
        logger.debug('Waiting for item {} from queue {} (expect {} more)'.format(idx, queue_name, final_size - idx))
        item = queue.get()
        logger.debug('Got item {} from queue {}: {} (expect {} more)'.format(idx, queue_name, item, final_size - idx))
        yield item


def coords_to_polygon_wkt_str(x_coords, y_coords):
    """
    x_coords and y_coords must be in anti-clockwise order (e.g. output of
    skimage.measure.find_contours(binary_im, 0.5)[0]
    )

    Returns a POLYGON wkt string

    Parameters
    ----------
    x_coords : array-like
    y_coords : array-like

    Returns
    -------
    str
    """
    coords_str = ','.join('{} {}'.format(x_coord, y_coord) for x_coord, y_coord in zip(x_coords, y_coords))
    coords_str += ',{} {}'.format(x_coords[0], y_coords[0])
    return 'POLYGON(({}))'.format(coords_str)


def simplify_image(array_xy, x_offset, y_offset):
    """
    Return wkt polygon string of binary image

    Parameters
    ----------
    array_xy
    x_offset
    y_offset

    Returns
    -------
    str
    """
    outline_coords_xy = find_contours(array_xy, 0.5)[0]
    return coords_to_polygon_wkt_str(outline_coords_xy[:, 0] + x_offset, outline_coords_xy[:, 1] + y_offset)


def commit_tilewise_results_from_queue(
        tile_result_queue, output_path, total_tiles, tile_size, workflow_id
):
    global catmaid
    result_iterator = iterate_queue(tile_result_queue, total_tiles, 'tile_result_queue')

    logger.info('Starting to commit tile classification results')

    with h5py.File(output_path, 'r+') as f:
        pixel_predictions_zyx = f['pixel_predictions']
        slice_labels_zyx = f['slice_labels']

        for tile_count, (tile_idx, predictions_xyc, synapse_cc_xy) in enumerate(result_iterator):
            tilename = 'z{}-y{}-x{}'.format(*tile_idx)
            logger.debug('Committing results from tile {}, {} of {}'.format(tilename, tile_count, total_tiles))
            bounds_xyz = tile_index_to_bounds(tile_idx, tile_size)

            pixel_predictions_zyx[
                bounds_xyz[0, 2], bounds_xyz[0, 1]:bounds_xyz[1, 1], bounds_xyz[0, 0]:bounds_xyz[1, 0], :
            ] = predictions_xyc.transpose((1, 0, 2))  # xyc to yxc

            synapse_cc_yx = synapse_cc_xy.T

            log_prefix = 'Tile {} ({}/{}): '.format(tilename, tile_count, total_tiles)

            synapse_slices = []

            local_label_set = set(np.unique(synapse_cc_yx)[1:].astype(int))
            for local_label in local_label_set:
                slice_prefix = log_prefix + '[{}] '.format(local_label)

                logger.debug('%sProcessing slice label'.format(local_label), slice_prefix)

                binary_arr_xy = synapse_cc_xy == local_label

                syn_pixel_coords_xy = np.where(binary_arr_xy)
                size_px = len(syn_pixel_coords_xy[0])
                y_centroid_px = np.average(syn_pixel_coords_xy[1]) + bounds_xyz[0, 1]
                x_centroid_px = np.average(syn_pixel_coords_xy[0]) + bounds_xyz[0, 0]

                # Determine average uncertainty
                # Get probabilities for this synapse's pixels
                flat_predictions = predictions_xyc[synapse_cc_xy == local_label]
                # Sort along channel axis
                flat_predictions.sort(axis=-1)
                # What's the difference between the highest and second-highest class?
                certainties = flat_predictions[:, -1] - flat_predictions[:, -2]
                avg_certainty = np.mean(certainties)
                uncertainty = 1.0 - avg_certainty

                wkt_str = simplify_image(binary_arr_xy, bounds_xyz[0, 0], bounds_xyz[0, 1])

                synapse_slices.append({
                    'id': int(local_label),
                    'wkt_str': wkt_str,
                    'size_px': int(size_px),
                    'xs_centroid': int(x_centroid_px),
                    'ys_centroid': int(y_centroid_px),
                    'uncertainty': uncertainty
                })

            id_mapping = catmaid.add_synapse_slices_to_tile(workflow_id, synapse_slices, tile_idx)
            logger.debug('Got ID mapping from CATMAID:\n{}'.format(id_mapping))

            returned_keys = {int(key) for key in id_mapping.keys()}
            if returned_keys != local_label_set:
                logger.error(
                    'Returned keys are not the same as sent keys:\n\t{}\n\t{}'.format(returned_keys, local_label_set)
                )

            mapped_synapse_cc_yx = np.ones(synapse_cc_yx.shape, LABEL_DTYPE)
            for local_label, synapse_id in id_mapping.items():
                logger.debug('Addressing ID mapping pair: {}, {}'.format(local_label, synapse_id))
                mapped_synapse_cc_yx[synapse_cc_yx == int(local_label)] = synapse_id

            slice_labels_zyx[
                bounds_xyz[0, 2], bounds_xyz[0, 1]:bounds_xyz[1, 1], bounds_xyz[0, 0]:bounds_xyz[1, 0]
            ] = mapped_synapse_cc_yx

            catmaid.agglomerate_synapses(id_mapping.values())  # maybe do this per larger block?


def commit_node_association_results_from_queue(node_result_queue, total_nodes, project_workflow_id):
    global catmaid

    logger.debug('Committing node association results')

    result_generator = iterate_queue(node_result_queue, total_nodes, 'node_result_queue')

    logger.debug('Getting node association results')
    assoc_tuples = []
    for result in result_generator:
        assoc_tuple = (result.synapse_slice_id, result.node_id, result.contact_px)
        logger.debug('Appending segmentation result to args: %s', repr(assoc_tuple))
        assoc_tuples.append(assoc_tuple)

    logger.debug('Node association results are\n%s', repr(assoc_tuples))
    logger.info('Inserting new slice:treenode mappings')

    catmaid.add_synapse_treenode_associations(assoc_tuples, project_workflow_id)


def setup_logging(project_dir, args, kwargs, level=logging.NOTSET):

    # set up the log files and symlinks
    latest_ln = os.path.join(project_dir, 'logs', 'latest')
    os.remove(latest_ln)
    log_dir = os.path.join(project_dir, 'logs', TIMESTAMP)
    mkdir_p(log_dir)
    os.symlink(log_dir, latest_ln)
    log_file = os.path.join(log_dir, 'locate_synapses.txt')

    # set up the root logger
    root = logging.getLogger()
    formatter = logging.Formatter(LOGGER_FORMAT)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level)
    root.addHandler(file_handler)
    root.addHandler(stream_handler)
    root.setLevel(level)

    # set up the performance logger
    performance_formatter = logging.Formatter('%(asctime)s: elapsed %(message)s')
    performance_handler = logging.FileHandler(os.path.join(log_dir, 'timing.txt'))
    performance_handler.setFormatter(performance_formatter)
    performance_handler.setLevel(logging.INFO)
    performance_logger = logging.getLogger('PERFORMANCE_LOGGER')
    performance_logger.addHandler(performance_handler)
    performance_logger.propagate = True

    # write version information
    commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip()
    git_diff = subprocess.check_output(['git', 'diff']).strip()
    version_string = 'Commit hash: {}\n\nCurrent diff:\n{}'.format(commit_hash, git_diff)
    with open(os.path.join(log_dir, 'version.txt'), 'w') as f:
        f.write(version_string)

    # write argument information
    with open(os.path.join(log_dir, 'arguments.txt'), 'w') as f:
        f.write('Arguments:\n\t{}\nKeyword arguments:\n\t{}'.format(args, kwargs))


def kill_child_processes(signum=None, frame=None):
    current_proc = psutil.Process()
    for child_proc in current_proc.children(recursive=True):
        child_proc.kill()


if __name__ == "__main__":
    if DEBUG:
        print("USING DEBUG ARGUMENTS")

        project_dir = "../projects-2017/L1-CNS"
        cred_path = "credentials_dev.json"
        stack_id = 1
        skel_ids = [18531735]  # small test skeleton only on CLB's local instance

        force = 1

        args_list = [
            cred_path, stack_id, skel_ids, project_dir
        ]
        kwargs_dict = {'force': force}
        setup_logging(project_dir, args_list, kwargs_dict, LOG_LEVEL)
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument('--roi-radius-px', default=DEFAULT_ROI_RADIUS,
                            help='The radius (in pixels) around each skeleton node to search for synapses')
        parser.add_argument('credentials_path',
                            help='Path to a JSON file containing CATMAID credentials (see credentials.jsonEXAMPLE)')
        parser.add_argument('stack_id',
                            help='ID or name of image stack in CATMAID')
        parser.add_argument('project_dir',
                            help="A directory containing project files in ./projects, and which output files will be "
                                 "dropped into.")
        parser.add_argument('skeleton_ids', nargs='+',
                            help="Skeleton IDs in CATMAID")
        parser.add_argument('-f', '--force', type=int, default=0,
                            help="Whether to delete all prior results for a given skeleton: pass 1 for true or 0")

        args = parser.parse_args()
        args_list = [
            args.credentials_path, args.stack_id, args.skeleton_ids, args.project_dir, args.roi_radius_px, args.force
        ]
        kwargs_dict = {}  # must be empty
        setup_logging(args.project_dir, args_list, kwargs_dict, LOG_LEVEL)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.NOTSET)

    performance_logger = logging.getLogger('PERFORMANCE_LOGGER')

    logger.info('STARTING CATMAID-COMPATIBLE DETECTION')
    logger.debug('Parallelising over {} threads'.format(THREADS))
    logger.debug('Will terminate subprocesses at {}MB of RAM'.format(RAM_MB_PER_PROCESS))

    signal.signal(signal.SIGTERM, kill_child_processes)

    exit_code = 1
    try:
        main(*args_list, **kwargs_dict)
        exit_code = 0
    except Exception as e:
        logger.exception('Errored, killing all child processes and exiting')
        kill_child_processes()
    finally:
        sys.exit(exit_code)
