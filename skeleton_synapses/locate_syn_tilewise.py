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
from types import StringTypes
from string import Formatter

import psycopg2
# from psycopg2.extensions import register_adapter, AsIs
import h5py
import numpy as np
from six.moves import range
import networkx as nx
from skimage.morphology import skeletonize

from lazyflow.utility import Timer
from lazyflow.request import Request

from catmaid_interface import CatmaidAPI
from locate_synapses import (
    # constants/singletons
    DEFAULT_ROI_RADIUS,
    # functions
    setup_files, setup_classifier, setup_classifier_and_multicut, roi_around_node,
    fetch_raw_and_predict_for_node, raw_data_for_node, labeled_synapses_for_node, segmentation_for_node,
    # classes
    CaretakerProcess, LeakyProcess
)


# def addapt_numpy_float64(numpy_float64):
#     return AsIs(numpy_float64)
# register_adapter(np.float64, addapt_numpy_float64)

logging.basicConfig(level=0, format='%(levelname)s %(processName)s(%(process)d) %(name)s: %(message)s')

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

logger.info('STARTING TILEWISE')

HDF5_PATH = "../projects-2017/L1-CNS/tilewise_image_store.hdf5"
STACK_PATH = "../projects-2017/L1-CNS/synapse_volume.hdf5"

POSTGRES_CREDENTIALS = {
    'dbname': os.getenv('SYNSUGG_DB'),
    'password': os.getenv('SYNSUGG_DB_PASSWORD'),
    'user': os.getenv('SYNSUGG_DB_USER'),
    'host': os.getenv('SYNSUGG_DB_HOST', '127.0.0.1'),
    'port': int(os.getenv('SYNSUGG_DB_PORT', 5432))
}

UNKNOWN_LABEL = 0  # should be smallest label
BACKGROUND_LABEL = 1  # should be smaller than synapse labels
MIRROR_ID = 5  # todo: change this to 4 for production

ALGO_VERSION = 1

THREADS = int(os.getenv('SYNAPSE_DETECTION_THREADS', 3))
logger.debug('Parallelising over {} threads'.format(THREADS))
# NODES_PER_PROCESS = int(os.getenv('SYNAPSE_DETECTION_NODES_PER_PROCESS', 500))
RAM_MB_PER_PROCESS = int(os.getenv('SYNAPSE_DETECTION_RAM_MB_PER_PROCESS', 5000))
logger.debug('Will terminate subprocesses at {}MB of RAM'.format(RAM_MB_PER_PROCESS))

DEBUG = False


def main(credentials_path, stack_id, skeleton_id, project_dir, roi_radius_px=150, force=False):
    autocontext_project, multicut_project, volume_description_path, skel_output_dir, skeleton = setup_files(
        credentials_path, stack_id, skeleton_id, project_dir, force
    )

    logger.info("STARTING TILEWISE")

    catmaid = CatmaidAPI.from_json(credentials_path)
    stack_info = catmaid.get_stack_info(stack_id)

    ensure_tables(force)
    ensure_hdf5(stack_info, force)

    locate_synapses_tilewise(
        autocontext_project,
        multicut_project,
        volume_description_path,
        skel_output_dir,
        skeleton,
        roi_radius_px,
        ALGO_VERSION,  # todo: make dynamic
        stack_info
    )

    link_images(force=True)


def link_images(existing_path=HDF5_PATH, new_path=STACK_PATH, force=False):
    if force or not os.path.isfile(new_path):
        os.remove(new_path)
        os.link(existing_path, new_path)
        # with h5py.File(existing_path, 'r+') as f:
        #     f['volume'] = f['slice_labels']


def ensure_tables(force=False):
    if force:
        # todo: deprecate this
        logger.info('Dropping and recreating database %s', POSTGRES_CREDENTIALS['dbname'])
        cred = POSTGRES_CREDENTIALS.copy()
        cred['dbname'] = 'postgres'
        conn = psycopg2.connect(**cred)
        cursor = conn.cursor()
        conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
        cursor.execute('DROP DATABASE IF EXISTS {};'.format(POSTGRES_CREDENTIALS['dbname']))  # TODO: CHANGE THIS
        conn.commit()
        cursor.execute('CREATE DATABASE {};'.format(POSTGRES_CREDENTIALS['dbname']))  # TODO: CHANGE THIS
        conn.commit()
        cursor.close()
        conn.close()

    conn = psycopg2.connect(**POSTGRES_CREDENTIALS)
    cursor = conn.cursor()

    logger.debug('Ensuring tables and indices exist')
    # todo: postgres optimisation (index statistics, shared memory etc.)
    cursor.execute("""
        CREATE EXTENSION IF NOT EXISTS postgis;
        
        CREATE TABLE IF NOT EXISTS synapse_slice (
          id BIGSERIAL PRIMARY KEY,
          x_tile_idx INTEGER NOT NULL,
          y_tile_idx INTEGER NOT NULL,
          z_tile_idx INTEGER NOT NULL,
          convex_hull_2d GEOMETRY NOT NULL,
          algo_version INTEGER NOT NULL,
          size_px INTEGER NOT NULL,
          uncertainty REAL NOT NULL,
          x_centroid_px INTEGER NOT NULL,
          y_centroid_px INTEGER NOT NULL
        );
        
        CREATE INDEX IF NOT EXISTS synapse_slice_z_tile_idx ON synapse_slice (z_tile_idx);
        CREATE INDEX IF NOT EXISTS synapse_slice_convex_hull_2d ON synapse_slice USING GIST (convex_hull_2d);
        
        -- set ID sequence to start at 1, or 1 + the highest ID
        SELECT setval(
          'synapse_slice_id_seq',(SELECT GREATEST(MAX(id)+1,nextval('synapse_slice_id_seq')) FROM synapse_slice)
        );
        
        CREATE TABLE IF NOT EXISTS synapse_object (
          id BIGSERIAL PRIMARY KEY
        );
        
        SELECT setval(
          'synapse_object_id_seq',(SELECT GREATEST(MAX(id)+1,nextval('synapse_object_id_seq')) FROM synapse_object)
        );
        
        CREATE TABLE IF NOT EXISTS synapse_slice_synapse_object (
          id BIGSERIAL PRIMARY KEY,
          synapse_slice_id BIGINT UNIQUE REFERENCES synapse_slice (id) ON DELETE CASCADE,
          synapse_object_id BIGINT REFERENCES synapse_object (id) ON DELETE CASCADE
        );
        
        -- index synapse_slice_synapse_object_synapse_slice_id?
        
        CREATE TABLE IF NOT EXISTS synapse_slice_skeleton (
          id BIGSERIAL PRIMARY KEY,
          synapse_slice_id BIGINT REFERENCES synapse_slice (id) ON DELETE CASCADE,
          skeleton_id BIGINT,
          node_id BIGINT,
          algo_version INTEGER,
          contact_px INTEGER
        );
    """)

    conn.commit()
    cursor.close()
    conn.close()


def get_stack_mirror(stack_info, mirror_id=MIRROR_ID):
    if DEBUG:
        mirror_id = 5
    for mirror in stack_info['mirrors']:
        if mirror['id'] == mirror_id:
            return mirror


def create_label_volume(stack_info, hdf5_file, name, dtype=np.float64, extra_dim=None):
    mirror = get_stack_mirror(stack_info)

    dimension = [stack_info['dimension'][dim] for dim in 'zyx']
    chunksize = (1, mirror['tile_height'], mirror['tile_width'])

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


def get_tile_counts_zyx(stack_info):
    mirror = get_stack_mirror(stack_info)
    tile_size = {'z': 1}
    tile_size['y'], tile_size['x'] = [mirror['tile_{}'.format(dim)] for dim in ['height', 'width']]
    return [int(stack_info['dimension'][dim] / tile_size[dim]) for dim in 'zyx']


def ensure_hdf5(stack_info, force=False):
    if force or not os.path.isfile(HDF5_PATH):
        logger.info('Creating HDF5 volumes in %s', HDF5_PATH)
        with h5py.File(HDF5_PATH, 'w') as f:
            f.attrs['project_id'] = stack_info['pid']
            f.attrs['source_stack_id'] = stack_info['sid']
            f.attrs['source_mirror_id'] = get_stack_mirror(stack_info)['id']

            create_label_volume(stack_info, f, 'slice_labels', dtype=np.int64)
            f['volume'] = f['slice_labels']  # for compatibility with _parallel and _serial
            # create_label_volume(stack_info, f, 'object_labels')  # todo?
            create_label_volume(stack_info, f, 'pixel_predictions', dtype=np.float32, extra_dim=3)

            f.create_dataset(
                'algo_versions',
                get_tile_counts_zyx(stack_info),
                fillvalue=0,
                dtype=int
            )

            f.flush()


TileIndex = namedtuple('TileIndex', 'z_idx y_idx x_idx')


def skeleton_to_tiles(skeleton, mirror_info, minimum_radius=DEFAULT_ROI_RADIUS):
    """
    
    Parameters
    ----------
    skeleton
    mirror_info
    minimum_radius

    Returns
    -------

    """
    tile_set = set()
    tile_size_xyz = [mirror_info['tile_{}'.format(dim)] for dim in ['width', 'height']] + [1]

    logger.info('Getting tile set for %s', skeleton.skeleton_id)

    for branch in skeleton.branches:
        for node_info in branch:
            bounds_xyz = roi_around_node(node_info, minimum_radius)
            tile_idxs = (bounds_xyz / tile_size_xyz).astype(int)
            tile_set.update(TileIndex(*idxs) for idxs in product(
                [node_info.z_px],  # z index
                range(tile_idxs[0, 1], tile_idxs[1, 1] + 1),  # all tile y indices
                range(tile_idxs[0, 0], tile_idxs[1, 0] + 1)  # all tile x indices
            ))

    return tile_set


def tile_index_to_bounds(tile_index, mirror_info):
    tile_size_xyz = [mirror_info['tile_{}'.format(dim)] for dim in ['width', 'height']] + [1]
    topleft = np.array([tile_index.x_idx, tile_index.y_idx, tile_index.z_idx], dtype=int)
    return np.stack((topleft, topleft+1)) * tile_size_xyz  # todo: might need to -1 to bottom row


def flatten(arg):
    elements = []
    if isinstance(arg, StringTypes):
        elements.append(arg)
    else:
        try:
            for item in arg:
                elements.extend(flatten(item))
        except TypeError:
            elements.append(arg)
    return elements


def list_into_query(query, arg_lst, fmt='%s'):
    """
    Convert simple query with list of arguments into mogrifier-friendly form
    
    Parameters
    ----------
    query : str
        A string with a single {} in it
    arg_lst : array-like
        List of arguments to supply to SQL
    fmt : str
        Placeholder to use for each element (e.g. use this to wrap stuff in brackets), or to account for tuples

    Returns
    -------
    (str, array-like)
        The two arguments to pass to cursor.execute
        
    Examples
    --------
    >>> list_into_query("DELETE FROM table_name WHERE id IN ({})", [1, 2, 3])
    >>> ("DELETE FROM table_name WHERE id IN (%s, %s, %s)", (1, 2, 3))
    
    >>> list_into_query("INSERT INTO table_name (a, b) VALUES ({})", [[1, 2], [3, 4]], fmt='(%s, %s)')
    >>> ("INSERT INTO table_name (a, b) VALUES ((%s, %s), (%s, %s))", (1, 2, 3, 4))
    """
    assert set(fmt).issubset(',()%s ')

    arg_str = ', '.join(fmt for _ in arg_lst)
    final_query = query.format(arg_str)
    final_args = tuple(flatten(arg_lst))

    logger.debug('Preparing SQL query for form \n%s with arguments %s', final_query, str(final_args))
    return final_query, final_args


def list_into_query_multi(query, fmt=None, **kwargs):
    """
    Convert complex query with several lists of arguments into mogrifier-friendly form
    
    Parameters
    ----------
    query : str
        Format string using keyword format, e.g. 'Hi, my name is {name} and I am {age} years old'
    fmt : dict
        Mapping from keywords to SQL-friendly format strings (defaults to '%s' for everything)
    kwargs : dict
        Mapping from keywords to argument lists

    Returns
    -------
    (str, array-like)
        The two arguments to pass to cursor.execute
        
    Examples
    --------
    >>> query = "INSERT INTO table_name1 (a, b) VALUES ({first}); INSERT INTO table_name2 (a, b) VALUES ({second});"
    >>> list_into_query_multi(query, fmt={'second': '(%s, %s)'}, first=[1, 2, 3], second=[[1,2], [3,4]])
    >>> ('INSERT INTO table_name1 (a, b) VALUES (%s, %s, %s); INSERT INTO table_name2 (a, b) VALUES ((%s, %s), (%s, %s));',
    >>>     [1, 2, 3, 1, 2, 3, 4])
    """
    if fmt is None:
        fmt = dict()
    assert all(set(value).issubset(',()%s ') for value in fmt.values())

    formatter = Formatter()
    arg_order = [arg_name for _, arg_name, _, _ in formatter.parse(query) if arg_name]
    arg_strs = {
        arg_name: ', '.join(fmt.get(arg_name, '%s') for _ in arg_list)
        for arg_name, arg_list in kwargs.items()
    }
    final_args = flatten(kwargs[arg_name] for arg_name in arg_order)
    final_query = query.format(**arg_strs)

    logger.debug('Preparing SQL query for form \n%s \nwith arguments \n%s', final_query, str(final_args))
    return final_query, tuple(final_args)


def trim_skeleton(skeleton, max_size=20):
    """In place"""
    first_branch = skeleton.branches[0]
    try:
        trimmed_branch = first_branch[:max_size]
    except IndexError:
        trimmed_branch = first_branch
    skeleton.branches = [trimmed_branch]
    logger.warning('Trimmed skeleton for debug purposes, to {} edges'.format(len(trimmed_branch)))
    return skeleton


def locate_synapses_tilewise(
        autocontext_project_path,
        multicut_project,
        input_filepath,
        skel_output_dir,
        skeleton,
        roi_radius_px,
        algo_version,
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
    algo_version : int

    Returns
    -------

    """
    tile_queue, tile_result_queue = mp.Queue(), mp.Queue()

    mirror_info = get_stack_mirror(stack_info)

    logger.info('Populating tile queue')

    # trim_skeleton(skeleton)

    tile_set = skeleton_to_tiles(skeleton, mirror_info, roi_radius_px)
    with h5py.File(HDF5_PATH, 'r') as f:
        algo_versions = f['algo_versions']
        for tile_idx in tile_set:
            if algo_versions[tile_idx.z_idx, tile_idx.y_idx, tile_idx.x_idx] != algo_version:
                logging.debug("Tile %s has not been addressed, adding to queue", repr(tile_idx))
                tile_queue.put(tile_idx)
            else:
                logging.debug("Tile %s has been addressed by this algorithm, skipping", repr(tile_idx))

    tile_queue.close()

    if not tile_queue.empty():
        logger.info('Classifying pixels in tilewise')
        mirror_info = get_stack_mirror(stack_info)

        detector_containers = [
            CaretakerProcess(
                DetectorProcess, tile_queue, RAM_MB_PER_PROCESS,
                (tile_result_queue, input_filepath, autocontext_project_path, skel_output_dir, mirror_info),
                name='CaretakerProcess{}'.format(idx)
            )
            for idx in range(THREADS)
        ]

        total_tiles = tile_queue.qsize()
        logger.debug('{} tiles queued'.format(total_tiles))

        for detector_container in detector_containers:
            detector_container.start()

        commit_tilewise_results_from_queue(tile_result_queue, HDF5_PATH, total_tiles, mirror_info, algo_version)

        for detector_container in detector_containers:
            detector_container.join()
    else:
        logger.debug('No tiles found (probably already processed)')

    node_queue, node_result_queue = mp.Queue(), mp.Queue()
    node_id_to_seg_input = dict()
    for branch in skeleton.branches:
        for node_info in branch:
            node_id_to_seg_input[node_info.id] = NeuronSegmenterInput(node_info, roi_radius_px)

    conn = psycopg2.connect(**POSTGRES_CREDENTIALS)
    cursor = conn.cursor()

    logger.debug('Finding which node windows do not need re-segmenting')
    # fetch nodes which have been addressed by this algorithm
    # todo: check this
    query, cursor_args = list_into_query_multi("""
        SELECT synapse_slice_skeleton.node_id FROM synapse_slice_skeleton
          INNER JOIN ( VALUES {node_ids} ) node (id) ON (node.id = synapse_slice_skeleton.node_id)
          WHERE synapse_slice_skeleton.algo_version = {algo_version};
    """, node_ids=list(node_id_to_seg_input), algo_version=[algo_version], fmt={'node_ids': '(%s)'})
    cursor.execute(query, cursor_args)

    nodes_to_exclude = cursor.fetchall()

    cursor.close()
    conn.close()

    for node_tup in nodes_to_exclude:
        del node_id_to_seg_input[node_tup[0]]

    if node_id_to_seg_input:
        logger.info('Segmenting node windows')

        for seg_input in node_id_to_seg_input.values():
            node_queue.put(seg_input)

        node_queue.close()

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

        for neuron_seg_container in neuron_seg_containers:
            neuron_seg_container.start()
            assert neuron_seg_container.is_alive()

        commit_node_association_results_from_queue(node_result_queue, skeleton.skeleton_id, total_nodes, algo_version)

        for neuron_seg_container in neuron_seg_containers:
            neuron_seg_container.join()
    else:
        logger.debug('No nodes required re-segmenting')

    logger.info("DONE with skeleton.")


DetectorOutput = namedtuple('DetectorOutput', 'tile_idx predictions_xyc synapse_cc_xyc')


class DetectorProcess(LeakyProcess):
    def __init__(
            self, input_queue, max_ram_MB, output_queue, description_file, autocontext_project_path, skel_output_dir,
            mirror_info, debug=False, name=None
    ):
        super(DetectorProcess, self).__init__(input_queue, max_ram_MB, debug, name)
        self.output_queue = output_queue

        self.logger = logging.getLogger('{}.{}'.format(__name__, self.name))
        self.logger.setLevel(logging.DEBUG)
        self.logger.debug('Detector process instantiated')

        self.timing_logger = logging.getLogger(self.logger.name + '.timing')
        self.timing_logger.setLevel(logging.INFO)

        self.skel_output_dir = skel_output_dir
        self.mirror_info = mirror_info

        self.opPixelClassification = None

        self.setup_args = (description_file, autocontext_project_path)

    def setup(self):
        self.opPixelClassification = setup_classifier(*self.setup_args)
        Request.reset_thread_pool(1)  # todo: set to 0?

    def execute(self):
        tile_idx = self.input_queue.get()

        self.logger.debug("Addressing tile {}; {} tiles remaining".format(tile_idx, self.input_queue.qsize()))

        with Timer() as timer:
            roi_xyz = tile_index_to_bounds(tile_idx, self.mirror_info)

            # GET AND CLASSIFY PIXELS
            predictions_xyc = fetch_raw_and_predict_for_node(
                None, roi_xyz, self.skel_output_dir, self.opPixelClassification
            )
            # DETECT SYNAPSES
            synapse_cc_xy = labeled_synapses_for_node(None, roi_xyz, self.skel_output_dir, predictions_xyc)
            self.timing_logger.info("NODE TIMER: {}".format(timer.seconds()))

        self.logger.debug("Detected synapses in tile {}; {} tiles remaining".format(tile_idx, self.input_queue.qsize()))

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

        self.logger_name = '{}.{}'.format(__name__, self.name)
        logger = logging.getLogger(self.logger_name)
        logger.setLevel(logging.DEBUG)
        logger.debug('Segmenter process instantiated')

        self.timing_logger = logging.getLogger(self.logger_name + '.timing')
        self.timing_logger.setLevel(logging.INFO)

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
        logger = logging.getLogger(self.logger_name)
        logger.debug('Setting up opPixelClassification and multicut_shell...')
        self.opPixelClassification, self.multicut_shell = setup_classifier_and_multicut(
            *self.setup_args
        )
        logger.debug('opPixelClassification and multicut_shell set up')

        Request.reset_thread_pool(1)

    def execute(self):
        node_info, roi_radius_px = self.input_queue.get()

        logger = logging.getLogger(self.logger_name)

        logger.debug("Addressing node {}; {} nodes remaining".format(node_info.id, self.input_queue.qsize()))

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
            outputs = tuple()
            for syn_id in np.unique(synapse_overlaps[synapse_overlaps > 1]):
                contact_px = skeletonize(synapse_overlaps == syn_id).sum()  # todo: improve this?
                outputs += (NeuronSegmenterOutput(node_info, syn_id, contact_px), )

            self.timing_logger.info("TILE TIMER: {}".format(node_timer.seconds()))

        logger.debug('Adding segmentation output of node {} to output queue; {} nodes remaining'.format(
            node_info.id, self.input_queue.qsize()
        ))

        self.output_queue.put(outputs)


def syn_coords_to_wkt_str(x_coords, y_coords):
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


def commit_tilewise_results_from_queue(tile_result_queue, output_path, total_tiles, mirror_info, algo_version=ALGO_VERSION):
    conn = psycopg2.connect(**POSTGRES_CREDENTIALS)
    cursor = conn.cursor()

    geometry_adjacencies = nx.Graph()

    result_iterator = iterate_queue(tile_result_queue, total_tiles, 'tile_result_queue')

    logger.info('Starting to commit tile classification results')

    with h5py.File(output_path, 'r+') as f:
        pixel_predictions_zyx = f['pixel_predictions']
        slice_labels_zyx = f['slice_labels']
        algo_versions_zyx = f['algo_versions']
        # object_labels_zyx = f['object_labels']  # todo

        for tile_count, (tile_idx, predictions_xyc, synapse_cc_xy) in enumerate(result_iterator):
            tilename = 'z{}-y{}-x{}'.format(*tile_idx)
            logger.debug('Committing results from tile {}, {} of {}'.format(tilename, tile_count, total_tiles))
            bounds_xyz = tile_index_to_bounds(tile_idx, mirror_info)

            pixel_predictions_zyx[
                bounds_xyz[0, 2], bounds_xyz[0, 1]:bounds_xyz[1, 1], bounds_xyz[0, 0]:bounds_xyz[1, 0], :
            ] = predictions_xyc.transpose((1, 0, 2))  # xyc to yxc

            algo_versions_zyx[tile_idx.z_idx, tile_idx.y_idx, tile_idx.x_idx] = algo_version

            synapse_cc_yx = synapse_cc_xy.T

            log_prefix = 'Tile {} ({}/{}): '.format(tilename, tile_count, total_tiles)

            logger.debug('%sDeleting synapse slices associated with old tile', log_prefix)
            cursor.execute("""
                DELETE FROM synapse_slice 
                  WHERE z_tile_idx = %s AND y_tile_idx = %s AND x_tile_idx = %s;
            """, tile_idx)

            conn.commit()

            for slice_label in np.unique(synapse_cc_yx)[1:]:
                slice_prefix = log_prefix + '[{}] '.format(slice_label)

                logger.debug('%sProcessing slice label'.format(slice_label), slice_prefix)

                syn_pixel_coords = np.where(synapse_cc_xy == slice_label)
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

                wkt_str = syn_coords_to_wkt_str(
                    syn_pixel_coords[1] + bounds_xyz[0, 0], syn_pixel_coords[0] + bounds_xyz[0, 1]
                )

                logger.debug('%sGetting intersecting geometries', slice_prefix)
                # todo: temporary table?
                cursor.execute("""
                    SELECT id FROM synapse_slice
                      WHERE z_tile_idx BETWEEN %s AND %s 
                      AND
                      ST_DWithin(ST_ConvexHull(ST_GeomFromText(%s)), convex_hull_2d, 1.1);
                """, (tile_idx.z_idx - 1, tile_idx.z_idx + 1, wkt_str))

                intersecting_id_tups = cursor.fetchall()

                logger.debug('%sInserting new row', slice_prefix)
                cursor.execute("""
                    INSERT INTO synapse_slice (
                      x_tile_idx,
                      y_tile_idx,
                      z_tile_idx,
                      convex_hull_2d,
                      algo_version,
                      size_px,
                      uncertainty,
                      x_centroid_px,
                      y_centroid_px
                    ) VALUES (
                      %s, %s, %s, ST_ConvexHull(ST_GeomFromText(%s)), %s, %s, %s, %s, %s
                    ) RETURNING id;
                """,
                (
                    tile_idx.x_idx, tile_idx.y_idx, tile_idx.z_idx, wkt_str, algo_version, size_px,
                    uncertainty, x_centroid_px, y_centroid_px
                )
                )

                conn.commit()

                new_id = cursor.fetchone()[0]
                synapse_cc_yx[synapse_cc_yx == slice_label] = new_id

                logger.debug('%sUpdating adjacency graph', slice_prefix)
                geometry_adjacencies.add_node(new_id)  # in case there are no edges

                for intersecting_id_tup in intersecting_id_tups:
                    geometry_adjacencies.add_edge(new_id, intersecting_id_tup[0])

            logger.debug('%sInserting new label data into volume', log_prefix)
            synapse_cc_yx[synapse_cc_yx == 0] = 1
            slice_labels_zyx[
                bounds_xyz[0, 2], bounds_xyz[0, 1]:bounds_xyz[1, 1], bounds_xyz[0, 0]:bounds_xyz[1, 0]
            ] = synapse_cc_yx

    logger.info('Getting slice:object mapping')
    # get existing mapping of synapse slices associated with new synapse slices, to synapse objects
    query, cursor_args = list_into_query("""
        SELECT sl_obj.synapse_slice_id, sl_obj.synapse_object_id FROM synapse_slice_synapse_object sl_obj
          INNER JOIN (VALUES {}) syn_sl (id) ON (syn_sl.id = sl_obj.synapse_slice_id);
    """, geometry_adjacencies.nodes(), fmt='(%s)')

    cursor.execute(query, cursor_args)

    existing_syn_slice_to_obj = dict(cursor.fetchall())

    new_mappings = []
    obsolete_objects = []

    for syn_slice_group in nx.connected_components(geometry_adjacencies):
        syn_obj_ids = {
            existing_syn_slice_to_obj[slice_id] for slice_id in syn_slice_group if slice_id in existing_syn_slice_to_obj
        }

        if len(syn_obj_ids) == 0:
            logger.debug('No object found for slice IDs %s, creating new one', repr(sorted(syn_slice_group)))
            # create new synapse object
            cursor.execute("""
                INSERT INTO synapse_object DEFAULT VALUES RETURNING id;
            """)
            obj_id = cursor.fetchone()[0]
            new_mappings.extend((slice_id, obj_id) for slice_id in syn_slice_group)
        else:
            # decide which synapse object ID to use
            obj_id = min(syn_obj_ids)
            other_obj_ids = [other_obj_id for other_obj_id in syn_obj_ids if other_obj_id != obj_id]

            if other_obj_ids:
                # other_obj_ids_str = ', '.join(other_obj_ids)
                obsolete_objects.extend(other_obj_ids)

                logger.debug('New synapse slice joins >1 previous synapse objects: merging...')
                # remap all synapse slices mapped to objects with larger IDs
                query, cursor_args = list_into_query_multi("""
                    UPDATE synapse_slice_synapse_object
                      SET synapse_object_id = {obj_id} WHERE synapse_object_id IN ({other_obj_ids});
                """, obj_id=[obj_id], other_obj_ids=other_obj_ids)

                cursor.execute(query, cursor_args)

            new_mappings.extend(
                (slice_id, obj_id)
                for slice_id in syn_slice_group
                if obj_id != existing_syn_slice_to_obj.get(slice_id, None)
            )

    if obsolete_objects:  # todo: do this in SQL
        logger.info('Deleting obsolete synapse objects')
        query, cursor_args = list_into_query("DELETE FROM synapse_object WHERE id IN ({});", obsolete_objects)
        cursor.execute(query, cursor_args)

    logger.info('Inserting new slice:object mappings')
    query, cursor_args = list_into_query("""
        INSERT INTO synapse_slice_synapse_object (synapse_slice_id, synapse_object_id)
          VALUES {};
    """, new_mappings, fmt='(%s, %s)')

    cursor.execute(query, cursor_args)

    conn.commit()
    cursor.close()
    conn.close()


def commit_node_association_results_from_queue(node_result_queue, skeleton_id, total_nodes, algo_version):
    conn = psycopg2.connect(**POSTGRES_CREDENTIALS)
    cursor = conn.cursor()

    logger.debug('Committing node association results')

    result_generator = iterate_queue(node_result_queue, total_nodes, 'node_result_queue')

    logger.debug('Getting node association results')
    # values_to_insert = [(result.synapse_slice_id, skeleton_id, result.node_info.id, algo_version) for result in result_generator]
    values_to_insert = []
    for node_result in result_generator:
        for result in node_result:
            result_tuple = (result.synapse_slice_id, skeleton_id, result.node_info.id, algo_version, result.contact_px)
            logger.debug('Appending segmentation result to args: %s', repr(result_tuple))
            values_to_insert.append(result_tuple)

    logger.debug('Node association results are\n%s', repr(values_to_insert))

    query, cursor_args = list_into_query("""
        INSERT INTO synapse_slice_skeleton (
          synapse_slice_id,
          skeleton_id,
          node_id,
          algo_version,
          contact_px
        ) VALUES {};
    """, values_to_insert, fmt='(%s, %s, %s, %s, %s)')

    logger.info('Inserting new slice:skeleton mappings')

    cursor.execute(query, cursor_args)

    conn.commit()
    cursor.close()
    conn.close()


def get_synapse_objects_for_skel(skel_id=11524047):
    conn = psycopg2.connect(**POSTGRES_CREDENTIALS)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT ss_so.synapse_object_id FROM synapse_slice_synapse_object ss_so
          JOIN synapse_slice_skeleton ss_s ON ss_so.synapse_slice_id = ss_s.synapse_slice_id
          WHERE ss_s.skeleton_id = %s;
    """, (skel_id, ))

    results = [tup[0] for tup in cursor.fetchall()]

    conn.commit()
    cursor.close()
    conn.close()

    return results


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

    sys.exit( main(*args_list, **kwargs_dict) )