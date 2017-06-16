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

import h5py
import numpy as np
from six.moves import range
from skimage.morphology import skeletonize
from skimage.measure import find_contours

from lazyflow.utility import Timer
from lazyflow.request import Request

from catpy import CatmaidClient

from catmaid_interface import CatmaidSynapseSuggestionAPI
from locate_synapses import (
    # constants/singletons
    DEFAULT_ROI_RADIUS, LOGGER_FORMAT,
    # functions
    setup_files, setup_classifier, setup_classifier_and_multicut,
    fetch_raw_and_predict_for_node, raw_data_for_node, labeled_synapses_for_node, segmentation_for_node,
    # classes
    CaretakerProcess, LeakyProcess
)
from skeleton_utils import roi_around_node


# def addapt_numpy_float64(numpy_float64):
#     return AsIs(numpy_float64)
# register_adapter(np.float64, addapt_numpy_float64)

logging.basicConfig(level=0, format=LOGGER_FORMAT)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

performance_logger = logging.getLogger('PERFORMANCE_LOGGER')

logger.info('STARTING CATMAID-COMPATIBLE DETECTION')

HDF5_PATH = "../projects-2017/L1-CNS/tilewise_image_store.hdf5"
STACK_PATH = "../projects-2017/L1-CNS/synapse_volume.hdf5"

UNKNOWN_LABEL = 0  # should be smallest label
BACKGROUND_LABEL = 1  # should be smaller than synapse labels

TILE_SIZE = 512

THREADS = int(os.getenv('SYNAPSE_DETECTION_THREADS', 3))
logger.debug('Parallelising over {} threads'.format(THREADS))
# NODES_PER_PROCESS = int(os.getenv('SYNAPSE_DETECTION_NODES_PER_PROCESS', 500))
RAM_MB_PER_PROCESS = int(os.getenv('SYNAPSE_DETECTION_RAM_MB_PER_PROCESS', 5000))
logger.debug('Will terminate subprocesses at {}MB of RAM'.format(RAM_MB_PER_PROCESS))

DEBUG = False

ALGO_HASH = '1'

catmaid = None

last_event = time.time()
def log_timestamp(msg):
    global last_event
    now = time.time()
    performance_logger.info('{}: {}'.format(now - last_event, msg))
    last_event = now


def hash_algorithm(*paths):
    """
    Calculate an MD5 sum of the current git commit hash and the contents of some arbitrary files.

    Parameters
    ----------
    paths
        Paths to files outside of git control which affect the algorithm

    Returns
    -------
    str
    """
    logger.info('Hashing algorithm...')
    commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip()

    md5 = hashlib.md5(commit_hash)
    for path in paths:
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(128*md5.block_size), b''):
                md5.update(chunk)

    digest = md5.hexdigest()
    logger.debug('Algorithm hash is %s', digest)
    # todo: remove this
    logger.warning('Ignoring real algorithm hash, using {}'.format(ALGO_HASH))
    digest = ALGO_HASH
    return digest


def main(credentials_path, stack_id, skeleton_id, project_dir, roi_radius_px=150, force=False):
    autocontext_project, multicut_project, volume_description_path, skel_output_dir = setup_files(
        credentials_path, stack_id, skeleton_id, project_dir, force
    )
    global catmaid

    logger.info("STARTING TILEWISE")

    log_timestamp('started setup')

    catmaid = CatmaidSynapseSuggestionAPI(CatmaidClient.from_json(credentials_path))
    stack_info = catmaid.get_stack_info(stack_id)

    ensure_hdf5(stack_info, force=force)

    log_timestamp('finished setup')

    performance_logger.info('{}: finished setup')

    locate_synapses_catmaid(
        autocontext_project,
        multicut_project,
        volume_description_path,
        skel_output_dir,
        skeleton_id,
        roi_radius_px,
        stack_info
    )

    link_images(force=True)


def link_images(existing_path=HDF5_PATH, new_path=STACK_PATH, force=False):
    if force or not os.path.isfile(new_path):
        os.remove(new_path)
        os.link(existing_path, new_path)
        # with h5py.File(existing_path, 'r+') as f:
        #     f['volume'] = f['slice_labels']


def create_label_volume(stack_info, hdf5_file, name, tile_size=TILE_SIZE, dtype=np.float64, extra_dim=None):
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
    if force and os.path.isfile(HDF5_PATH):
        logger.warning('FORCE detected; deleting existing HDF5 file')
        os.remove(HDF5_PATH)

    if not os.path.isfile(HDF5_PATH):
        logger.info('Creating HDF5 volumes in %s', HDF5_PATH)
        with h5py.File(HDF5_PATH, 'w') as f:
            # f.attrs['workflow_id'] = workflow_id  # todo
            f.attrs['source_stack_id'] = stack_info['sid']

            create_label_volume(stack_info, f, 'slice_labels', TILE_SIZE, dtype=np.int64)
            # create_label_volume(stack_info, f, 'object_labels')  # todo?
            create_label_volume(stack_info, f, 'pixel_predictions', TILE_SIZE, dtype=np.float32, extra_dim=3)

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


def locate_synapses_catmaid(
        autocontext_project_path,
        multicut_project,
        input_filepath,
        skel_output_dir,
        skeleton_id,
        roi_radius_px,
        stack_info
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

    algo_hash = hash_algorithm(autocontext_project_path, multicut_project)
    workflow_id = catmaid.get_workflow_id(stack_info['sid'], algo_hash, TILE_SIZE)

    logger.info('Populating tile queue')

    log_timestamp('started getting tiles')

    node_infos = catmaid.get_treenode_locations(skeleton_id, stack_info['sid'])

    tile_index_set = nodes_to_tile_indexes(node_infos, TILE_SIZE, roi_radius_px)

    addressed_tiles = catmaid.get_detected_tiles(workflow_id)

    tile_queue, tile_result_queue = mp.Queue(), mp.Queue()

    for tile_idx in tile_index_set:
        if (tile_idx.x_idx, tile_idx.y_idx, tile_idx.z_idx) in addressed_tiles:
            logging.debug("Tile %s has been addressed by this algorithm, skipping", repr(tile_idx))
        else:
            logging.debug("Tile %s has not been addressed, adding to queue", repr(tile_idx))
            tile_queue.put(tile_idx)

    log_timestamp('finished getting tiles')

    # don't save out individual tiles
    skel_output_dir = skel_output_dir if DEBUG else None

    if not tile_queue.empty():
        logger.info('Classifying pixels in tilewise')

        detector_containers = [
            CaretakerProcess(
                DetectorProcess, tile_queue, RAM_MB_PER_PROCESS,
                (tile_result_queue, input_filepath, autocontext_project_path, skel_output_dir, TILE_SIZE),
                name='CaretakerProcess{}'.format(idx)
            )
            for idx in range(THREADS)
        ]

        total_tiles = tile_queue.qsize()
        logger.debug('{} tiles queued'.format(total_tiles))

        log_timestamp('started synapse detection ({} tiles, 512x512 each, {} threads)'.format(total_tiles, THREADS))

        for detector_container in detector_containers:
            detector_container.start()

        commit_tilewise_results_from_queue(tile_result_queue, HDF5_PATH, total_tiles, TILE_SIZE, workflow_id)

        for detector_container in detector_containers:
            detector_container.join()

        log_timestamp('finished synapse detection')
    else:
        logger.debug('No tiles found (probably already processed)')

    log_timestamp('started getting nodes')

    project_workflow_id = catmaid.get_project_workflow_id(workflow_id, algo_hash)
    treenode_slice_mappings = catmaid.get_treenode_synapse_associations(project_workflow_id)
    associated_treenodes = {int(pair[0]) for pair in treenode_slice_mappings}

    node_queue, node_result_queue = mp.Queue(), mp.Queue()
    for node_info in node_infos:
        if int(node_info.id) not in associated_treenodes:
            node_queue.put(NeuronSegmenterInput(node_info, roi_radius_px))

    log_timestamp('finished getting nodes')

    if not node_queue.empty():
        logger.info('Segmenting node windows')

        neuron_seg_containers = [
            CaretakerProcess(
                NeuronSegmenterProcess, node_queue, RAM_MB_PER_PROCESS,
                (node_result_queue, input_filepath, autocontext_project_path, multicut_project, skel_output_dir),
                name='CaretakerProcess{}'.format(idx)
            )
            for idx in range(THREADS)
            # for _ in range(1)
        ]

        total_nodes = node_queue.qsize()

        log_timestamp('started segmenting neurons ({} nodes, {} threads)'.format(total_nodes, THREADS))

        for neuron_seg_container in neuron_seg_containers:
            neuron_seg_container.start()
            assert neuron_seg_container.is_alive()

        commit_node_association_results_from_queue(node_result_queue, total_nodes, project_workflow_id)

        for neuron_seg_container in neuron_seg_containers:
            neuron_seg_container.join()

        log_timestamp('finished segmenting neurons')
    else:
        logger.debug('No nodes required re-segmenting')

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

        self.output_queue.put(DetectorOutput(tile_idx, predictions_xyc, synapse_cc_xy))


NeuronSegmenterInput = namedtuple('NeuronSegmenterInput', ['node_info', 'roi_radius_px'])
NeuronSegmenterOutput = namedtuple('NeuronSegmenterOutput', ['node_info', 'synapse_slice_id', 'contact_px'])


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
        node_info, roi_radius_px = self.input_queue.get()

        self.inner_logger.debug("Addressing node {}; {} nodes remaining".format(node_info.id, self.input_queue.qsize()))

        with Timer() as node_timer:
            roi_xyz = roi_around_node(node_info, roi_radius_px)
            raw_xy = raw_data_for_node(node_info, roi_xyz, None, self.opPixelClassification)

            # convert roi into a tuple of slice objects which can be used by numpy for indexing
            roi_slices = (roi_xyz[0, 2], slice(roi_xyz[0, 1], roi_xyz[1, 1]), slice(roi_xyz[0, 0], roi_xyz[1, 0]))

            # N.B. might involve parallel reads - consider a single reader process
            with h5py.File(HDF5_PATH, 'r') as f:
                synapse_cc_xy = np.array(f['slice_labels'][roi_slices]).T
                predictions_xyc = np.array(f['pixel_predictions'][roi_slices]).transpose((1, 0, 2))

            segmentation_xy = segmentation_for_node(
                node_info, roi_xyz, self.skel_output_dir, self.multicut_shell.workflow, raw_xy, predictions_xyc
            )

            center_coord = np.array(segmentation_xy.shape) // 2
            node_segment = segmentation_xy[tuple(center_coord)]

            synapse_overlaps = synapse_cc_xy * (segmentation_xy == node_segment)
            outputs = []
            slice_ids = np.unique(synapse_overlaps[synapse_overlaps > 1])
            self.inner_logger.debug('Found overlapping IDs {} in roi_xyz {}', slice_ids, roi_xyz)
            for syn_id in slice_ids:
                contact_px = skeletonize(synapse_overlaps == syn_id).sum()  # todo: improve this?
                outputs.append(NeuronSegmenterOutput(node_info, syn_id, contact_px))

            logging.getLogger(self.inner_logger.name + '.timing').info("TILE TIMER: {}".format(node_timer.seconds()))

        self.inner_logger.debug('Adding segmentation output of node {} to output queue; {} nodes remaining'.format(
            node_info.id, self.input_queue.qsize()
        ))

        self.output_queue.put(tuple(outputs))


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


def simplify_image(array, x_offset, y_offset):
    """
    Return wkt polygon string of binary image

    Parameters
    ----------
    array
    x_offset
    y_offset

    Returns
    -------
    str
    """
    outline_coords_yx = find_contours(array, 0.5)[0]
    return coords_to_polygon_wkt_str(outline_coords_yx[:, 1] + x_offset, outline_coords_yx[:, 0] + y_offset)


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
            synapse_ids = []
            tilename = 'z{}-y{}-x{}'.format(*tile_idx)
            logger.debug('Committing results from tile {}, {} of {}'.format(tilename, tile_count, total_tiles))
            bounds_xyz = tile_index_to_bounds(tile_idx, tile_size)

            pixel_predictions_zyx[
                bounds_xyz[0, 2], bounds_xyz[0, 1]:bounds_xyz[1, 1], bounds_xyz[0, 0]:bounds_xyz[1, 0], :
            ] = predictions_xyc.transpose((1, 0, 2))  # xyc to yxc

            synapse_cc_yx = synapse_cc_xy.T

            log_prefix = 'Tile {} ({}/{}): '.format(tilename, tile_count, total_tiles)

            synapse_slices = []

            slice_label_set = set(np.unique(synapse_cc_yx)[1:].astype(int))
            for slice_label in slice_label_set:
                slice_prefix = log_prefix + '[{}] '.format(slice_label)

                logger.debug('%sProcessing slice label'.format(slice_label), slice_prefix)

                binary_arr = synapse_cc_xy == slice_label

                syn_pixel_coords = np.where(binary_arr)
                size_px = len(syn_pixel_coords[0])
                y_centroid_px = np.average(syn_pixel_coords[0]) + bounds_xyz[0, 1]
                x_centroid_px = np.average(syn_pixel_coords[1]) + bounds_xyz[0, 0]

                # Determine average uncertainty
                # Get probabilities for this synapse's pixels
                flat_predictions = predictions_xyc[synapse_cc_xy == slice_label]
                # Sort along channel axis
                flat_predictions.sort(axis=-1)
                # What's the difference between the highest and second-highest class?
                certainties = flat_predictions[:, -1] - flat_predictions[:, -2]
                avg_certainty = np.mean(certainties)
                uncertainty = 1.0 - avg_certainty

                wkt_str = simplify_image(binary_arr, bounds_xyz[0, 0], bounds_xyz[0, 1])

                synapse_slices.append({
                    'id': int(slice_label),
                    'wkt_str': wkt_str,
                    'size_px': int(size_px),
                    'xs_centroid': int(x_centroid_px),
                    'ys_centroid': int(y_centroid_px),
                    'uncertainty': uncertainty
                })

            id_mapping = catmaid.add_synapse_slices_to_tile(workflow_id, synapse_slices, tile_idx)

            assert set(id_mapping.keys()) == slice_label_set

            synapse_cc_yx[synapse_cc_yx == 0] = 1
            mapped_synapse_cc_yx = synapse_cc_yx / synapse_cc_yx
            for slice_label, synapse_id in id_mapping.items():
                mapped_synapse_cc_yx[synapse_cc_yx == slice_label] = synapse_id

            slice_labels_zyx[
                bounds_xyz[0, 2], bounds_xyz[0, 1]:bounds_xyz[1, 1], bounds_xyz[0, 0]:bounds_xyz[1, 0]
            ] = synapse_cc_yx

            catmaid.agglomerate_synapses(id_mapping.values())  # maybe do this per larger block?


def commit_node_association_results_from_queue(node_result_queue, total_nodes, project_workflow_id):
    global catmaid

    logger.debug('Committing node association results')

    result_generator = iterate_queue(node_result_queue, total_nodes, 'node_result_queue')

    logger.debug('Getting node association results')
    assoc_tuples = []
    for node_result in result_generator:
        for result in node_result:
            assoc_tuple = (result.synapse_slice_id, result.node_info.id, result.contact_px)
            logger.debug('Appending segmentation result to args: %s', repr(assoc_tuple))
            assoc_tuples.append(assoc_tuple)

    logger.debug('Node association results are\n%s', repr(assoc_tuples))
    logger.info('Inserting new slice:treenode mappings')

    catmaid.add_synapse_treenode_associations(assoc_tuples, project_workflow_id)


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
        skel_id = 18531735  # small test skeleton only on CLB's local instance
        force = 0

        args_list = [
            cred_path, stack_id, skel_id, project_dir
        ]
        kwargs_dict = {'force': force}
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument('--roi-radius-px', default=DEFAULT_ROI_RADIUS,
                            help='The radius (in pixels) around each skeleton node to search for synapses')
        parser.add_argument('credentials_path',
                            help='Path to a JSON file containing CATMAID credentials (see credentials.jsonEXAMPLE)')
        parser.add_argument('stack_id',
                            help='ID or name of image stack in CATMAID')
        parser.add_argument('skeleton_id',
                            help="A skeleton ID in CATMAID")
        parser.add_argument('project_dir',
                            help="A directory containing project files in ./projects, and which output files will be "
                                 "dropped into.")
        parser.add_argument('-f', '--force', type=int, default=0,
                            help="Whether to delete all prior results for a given skeleton: pass 1 for true or 0")

        args = parser.parse_args()
        args_list = [
            args.credentials_path, args.stack_id, args.skeleton_id, args.project_dir, args.roi_radius_px, args.force
        ]
        kwargs_dict = {}  # must be empty

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
