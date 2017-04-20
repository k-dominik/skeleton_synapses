#!/usr/bin/env python
from __future__ import division
import logging
import csv
import multiprocessing as mp
from collections import namedtuple
import argparse
import sys
import os
import json
from itertools import product

import psycopg2
import h5py
import numpy as np
from six.moves import range
import networkx as nx

from lazyflow.utility import Timer
from lazyflow.request import Request

from catmaid_interface import CatmaidAPI
from locate_synapses import (
    # constants/singletons
    DEFAULT_ROI_RADIUS, logger,
    # functions
    setup_files, setup_classifier, setup_classifier_and_multicut, roi_around_node, get_and_print_env,
    fetch_raw_and_predict_for_node, raw_data_for_node, labeled_synapses_for_node, segmentation_for_node,
    # classes
    CaretakerProcess, LeakyProcess
)

HDF5_PATH = 'image_store.hdf5'

POSTGRES_USER = 'cbarnes'
POSTGRES_DB = 'synapse_detection'

DTYPE = np.uint32  # can only store up to 2^24
UNKNOWN_LABEL = 0  # should be smallest label
BACKGROUND_LABEL = 1  # should be smaller than synapse labels
MIRROR_ID = 4

ALGO_VERSION = 1

THREADS = get_and_print_env('SYNAPSE_DETECTION_THREADS', 3, int)
NODES_PER_PROCESS = get_and_print_env('SYNAPSE_DETECTION_NODES_PER_PROCESS', 500, int)
RAM_MB_PER_PROCESS = get_and_print_env('SYNAPSE_DETECTION_RAM_MB_PER_PROCESS', 5000, int)


def main(credentials_path, stack_id, skeleton_id, project_dir, roi_radius_px=150, force=False):
    autocontext_project, multicut_project, volume_description_path, skel_output_dir, skeleton = setup_files(
        credentials_path, stack_id, skeleton_id, project_dir, force
    )

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


def ensure_tables(force=False):
    if force:
        conn = psycopg2.connect("dbname={} user={}".format('postgres', POSTGRES_USER), autocommit=True)
        cursor = conn.cursor()
        cursor.execute('DROP DATABASE %s;', POSTGRES_DB)
        cursor.execute('CREATE DATABASE %s;', POSTGRES_DB)
        cursor.close()
        conn.close()

    conn = psycopg2.connect("dbname={} user={}".format(POSTGRES_DB, POSTGRES_USER), autocommit=True)
    cursor = conn.cursor()

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
          algo_version INTEGER
        );
    """)

    cursor.close()
    conn.close()


def get_stack_mirror(stack_info, mirror_id=MIRROR_ID):
    for mirror in stack_info['mirrors']:
        if mirror['id'] == mirror_id:
            return mirror


def create_label_volume(stack_info, hdf5_file, name, extra_dim=None):
    dimension = [stack_info['dimension'][dim] for dim in 'zyx']
    if extra_dim is not None:
        dimension += [extra_dim]

    mirror = get_stack_mirror(stack_info)

    labels = hdf5_file.create_dataset(
        name,
        dimension,  # zyx
        chunks=[1, mirror['tile_height'], mirror['tile_width']],  # zyx
        fillvalue=0,
        dtype=DTYPE
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
        with h5py.File(HDF5_PATH) as f:
            f.attrs['project_id'] = stack_info['pid']
            f.attrs['source_stack_id'] = stack_info['sid']
            f.attrs['source_mirror_id'] = get_stack_mirror(stack_info)['id']

            create_label_volume(stack_info, f, 'slice_labels')
            # create_label_volume(stack_info, f, 'object_labels')  # todo
            create_label_volume(stack_info, f, 'pixel_predictions', 3)

            f.create_dataset(
                'algo_versions',
                get_tile_counts_zyx(stack_info),
                fillvalue=0,
                dtype=DTYPE
            )


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

    for branch in skeleton.branches:
        for node_info in branch:
            bounds_xyz = roi_around_node(node_info, minimum_radius)
            tile_idxs = (bounds_xyz / tile_size_xyz).astype(int)
            tile_set.update(TileIndex(idxs) for idxs in product(
                [node_info.z_px],  # z index
                range(tile_idxs[0, 1], tile_idxs[1, 1] + 1),  # all tile y indices
                range(tile_idxs[0, 0], tile_idxs[1, 0] + 1)  # all tile x indices
            ))

    return tile_set


def tile_index_to_bounds(tile_index, mirror_info):
    tile_size_xyz = [mirror_info['tile_{}'.format(dim)] for dim in ['width', 'height']] + [1]
    topleft = np.array([tile_index.x_index, tile_index.y_index, tile_index.z_index], dtype=int)
    return np.stack((topleft, topleft+1)) * tile_size_xyz  # todo: might need to -1 to bottom row


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

    tile_set = skeleton_to_tiles(skeleton, stack_info, roi_radius_px)
    with h5py.File(HDF5_PATH, 'r') as f:
        algo_versions = f['algo_versions']
        for tile_idx in tile_set:
            if algo_versions[tile_idx.z_idx, tile_idx.y_idx, tile_idx.x_idx] != algo_version:
                tile_queue.put(tile_idx)

    total_tiles = tile_queue.qsize()
    logger.debug('{} tiles queued'.format(total_tiles))

    if total_tiles:
        mirror_info = get_stack_mirror(stack_info)

        detector_containers = [
            CaretakerProcess(
                DetectorProcess, tile_queue, RAM_MB_PER_PROCESS,
                (tile_result_queue, input_filepath, autocontext_project_path, skel_output_dir, mirror_info)
            )
            for _ in range(THREADS)
        ]

        for detector_container in detector_containers:
            detector_container.start()

        commit_tilewise_results_from_queue(tile_result_queue, HDF5_PATH, total_tiles, mirror_info, algo_version)

        for detector_container in detector_containers:
            detector_container.join()

    node_queue, node_result_queue = mp.Queue(), mp.Queue()
    node_id_to_seg_input = dict()
    node_overall_index = -1
    for branch_index, branch in enumerate(skeleton.branches):
        for node_index_in_branch, node_info in enumerate(branch):
            node_overall_index += 1
            node_id_to_seg_input[node_info.id] = NeuronSegmenterInput(node_info, roi_radius_px)

    conn = psycopg2.connect("dbname={} user={}".format(POSTGRES_DB, POSTGRES_USER), autocommit=True)
    cursor = conn.cursor()
    node_ids_str = ', '.join('({})'.format(node_id) for node_id in node_id_to_seg_input)

    # fetch nodes which have been addressed by this algorithm
    # todo: check this
    cursor.execute("""
        SELECT synapse_slice_skeleton.node_id FROM synapse_slice_skeleton
          INNER JOIN ( VALUES %s ) node (id) ON (node.id = synapse_slice_skeleton.node_id)
          WHERE synapse_slice_skeleton.algo_version = %s;
    """, (node_ids_str, algo_version))

    nodes_to_exclude = cursor.fetchall()

    cursor.close()
    conn.close()

    for node_tup in nodes_to_exclude:
        del node_id_to_seg_input[node_tup[0]]

    if node_id_to_seg_input:
        for seg_input in node_id_to_seg_input.values():
            node_queue.put(seg_input)

        total_nodes = node_queue.qsize()

        neuron_seg_containers = [
            CaretakerProcess(
                NeuronSegmenterProcess, tile_queue, RAM_MB_PER_PROCESS,
                (node_result_queue, input_filepath, autocontext_project_path, multicut_project, skel_output_dir)
            )
            for _ in range(THREADS)
        ]

        for neuron_seg_container in neuron_seg_containers:
            neuron_seg_container.start()

        commit_node_association_results_from_queue(node_result_queue, skeleton.skeleton_id, total_nodes, algo_version)

        for neuron_seg_container in neuron_seg_containers:
            neuron_seg_container.join()

    logger.info("DONE with skeleton.")


DetectorOutput = namedtuple('DetectorOutput', 'tile_idx predictions_xyc synapse_cc_xyc')


class DetectorProcess(LeakyProcess):
    def __init__(
            self, input_queue, max_ram_MB, output_queue, description_file, autocontext_project_path, skel_output_dir,
            mirror_info, debug=False
    ):
        super(DetectorProcess, self).__init__(input_queue, max_ram_MB, debug)
        self.output_queue = output_queue

        logger.debug('Detector process {} instantiated'.format(self.name))

        self.timing_logger = logging.getLogger(__name__ + '.timing')
        self.timing_logger.setLevel(logging.INFO)

        self.skel_output_dir = skel_output_dir
        self.mirror_info = mirror_info

        self.opPixelClassification = None

        self.setup_args = (description_file, autocontext_project_path)

    def setup(self):
        self.opPixelClassification = setup_classifier
        Request.reset_thread_pool(1)

    def execute(self):
        tile_idx = self.input_queue.get()

        logger.debug("{} PROGRESS: addressing tile {}, {} tiles remaining"
                     .format(self.name.upper(), tile_idx, self.input_queue.qsize()))

        with Timer() as timer:
            roi_xyz = tile_index_to_bounds(tile_idx, self.mirror_info)

            # GET AND CLASSIFY PIXELS
            predictions_xyc = fetch_raw_and_predict_for_node(
                None, roi_xyz, self.skel_output_dir, self.opPixelClassification
            )
            # DETECT SYNAPSES
            synapse_cc_xy = labeled_synapses_for_node(None, roi_xyz, self.skel_output_dir, predictions_xyc)
            self.timing_logger.info("NODE TIMER: {}".format(timer.seconds()))

        logger.debug("{} PROGRESS: detected synapses in tile {}, {} tiles remaining"
                     .format(self.name.upper(), tile_idx, self.input_queue.qsize()))

        self.output_queue.put(DetectorOutput(tile_idx, predictions_xyc, synapse_cc_xy))


NeuronSegmenterInput = namedtuple('NeuronSegmenterInput', ['node_info', 'roi_radius_px'])
NeuronSegmenterOutput = namedtuple('NeuronSegmenterOutput', 'node_info synapse_slice_id')


class NeuronSegmenterProcess(LeakyProcess):
    """
    Process which creates its own pixel classifier and multicut workflow, pulls jobs from one queue and returns
    outputs to another queue.
    """
    def __init__(
            self, input_queue, max_ram_MB, output_queue, description_file, autocontext_project_path, multicut_project,
            skel_output_dir, debug=False
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
        super(NeuronSegmenterProcess, self).__init__(input_queue, max_ram_MB, debug)
        self.output_queue = output_queue

        logger.debug('Segmenter process {} instantiated'.format(self.name))

        self.timing_logger = logging.getLogger(__name__ + '.timing')
        self.timing_logger.setLevel(logging.INFO)

        self.skel_output_dir = skel_output_dir

        self.opPixelClassification = self.multicut_shell = None
        self.setup_args = (description_file, autocontext_project_path, multicut_project)

    def setup(self):
        self.opPixelClassification, self.multicut_shell = setup_classifier_and_multicut(
            *self.setup_args
        )

        Request.reset_thread_pool(1)

    def execute(self):
        node_info, roi_radius_px = self.input_queue.get()

        logger.debug("{} PROGRESS: addressing node {}, {} nodes remaining"
                     .format(self.name.upper(), node_info.id, self.input_queue.qsize()))

        with Timer() as node_timer:
            roi_xyz = roi_around_node(node_info, roi_radius_px)
            raw_xy = raw_data_for_node(node_info, roi_xyz, None, self.opPixelClassification)

            # convert roi into a tuple of slice objects which can be used by numpy for indexing
            roi_slices = (roi_xyz[0, 2], slice(roi_xyz[0, 1], roi_xyz[1, 1]), slice(roi_xyz[0, 2], roi_xyz[1, 2]))

            # N.B. might involve parallel reads - consider a single reader process
            with h5py.File(HDF5_PATH, 'r') as f:
                synapse_cc_xy = np.array(f['slice_labels'][roi_slices]).T
                predictions_xyc = np.array(f['pixel_predictions'][roi_slices]).transpose((1, 0, 2))

            segmentation_xy = segmentation_for_node(
                node_info, roi_xyz, self.skel_output_dir, self.multicut_shell.workflow, raw_xy, predictions_xyc
            )

            center_coord = np.array(segmentation_xy.shape) // 2
            node_segment = segmentation_xy[tuple(center_coord)]
            for syn_id in np.unique(synapse_cc_xy)[1:]:
                overlapping_segments = np.unique(segmentation_xy[synapse_cc_xy == syn_id])
                if node_segment in overlapping_segments:
                    self.output_queue.put(NeuronSegmenterOutput(node_info, syn_id))

            self.timing_logger.info("NODE TIMER: {}".format(node_timer.seconds()))

        logger.debug("{} PROGRESS: segmented area around node {}, {} nodes remaining"
                     .format(self.name.upper(), node_info.id, self.input_queue.qsize()))


def syn_coords_to_geom_str(x_coords, y_coords):
    """
    String which PostGIS will interpret as a 2D convex hull of the coordinates
    
    Parameters
    ----------
    syn_coords : array-like
        Result of np.where(synapse_cc_xy == slice_label) plus top left corner of panel

    Returns
    -------
    str
        String to be interpolated into SQL query
    """
    coords_str = ','.join('{} {}'.format(x_coord, y_coord) for x_coord, y_coord in zip(x_coords, y_coords))
    # could use scipy to find contour to reduce workload on database, or SQL concave hull, or SQL simplify geometry
    return "ST_ConvexHull(ST_GeomFromText('MULTIPOINT({})'))".format(coords_str)


def commit_tilewise_results_from_queue(tile_result_queue, output_path, total_tiles, mirror_info, algo_version=ALGO_VERSION):
    conn = psycopg2.connect("dbname={} user={}".format(POSTGRES_DB, POSTGRES_USER), autocommit=True)
    cursor = conn.cursor()

    geometry_adjacencies = nx.Graph()

    with h5py.File(output_path, 'r+') as f:
        pixel_predictions_zyx = f['pixel_predictions']
        slice_labels_zyx = f['slice_labels']
        algo_versions_zyx = f['algo_versions']
        # object_labels_zyx = f['object_labels']  # todo

        for _ in range(total_tiles):
            tile_idx, predictions_xyc, synapse_cc_xy = tile_result_queue.get()
            bounds_xyz = tile_index_to_bounds(tile_idx, mirror_info)

            pixel_predictions_zyx[
                bounds_xyz[0, 2], bounds_xyz[0, 1]:bounds_xyz[1, 1], bounds_xyz[0, 0]:bounds_xyz[1, 0], :
            ] = predictions_xyc.transpose((1, 0, 2))  # xyc to yxc

            algo_versions_zyx[tile_idx.z_idx, tile_idx.y_idx, tile_idx.x_idx] = algo_version

            synapse_cc_yx = synapse_cc_xy.T

            cursor.execute("""
                DELETE FROM synapse_slice 
                  WHERE z_tile_idx = %s AND y_tile_idx = %s AND x_tile_idx = %s;
            """, tile_idx)

            for slice_label in np.unique(synapse_cc_yx)[1:]:
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

                geom_str = syn_coords_to_geom_str(
                    syn_pixel_coords[1] + bounds_xyz[0, 0], syn_pixel_coords[0] + bounds_xyz[0, 1]
                )

                # todo: temporary table?
                cursor.execute("""
                    SELECT id FROM synapse_slice
                      WHERE z_tile_idx BETWEEN %s AND %s 
                      AND
                      ST_DWithin(%s, convex_hull_2d, 1.1);
                """, (tile_idx.z_idx - 1, tile_idx.z_idx + 1, geom_str))

                intersecting_id_tups = cursor.fetchall()

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
                      %s, %s, %s, %s, %s, %s, %s, %s, %s
                    ) RETURNING id;
                """,
                (
                    tile_idx.x_idx, tile_idx.y_idx, tile_idx.z_idx, geom_str, algo_version, size_px,
                    uncertainty, x_centroid_px, y_centroid_px
                )
                )

                new_id = cursor.fetchone()[0]
                synapse_cc_yx[synapse_cc_yx == slice_label] = new_id
                geometry_adjacencies.add_node(new_id)  # in case there are no edges

                for intersecting_id_tup in intersecting_id_tups:
                    geometry_adjacencies.add_edge(new_id, intersecting_id_tup[0])

            slice_labels_zyx[
                bounds_xyz[0, 2], bounds_xyz[0, 1]:bounds_xyz[1, 1], bounds_xyz[0, 0]:bounds_xyz[1, 0]
            ] = synapse_cc_yx

        # get existing mapping of synapse slices associated with new synapse slices, to synapse objects
        syn_slice_str = ', '.join('({})'.format(syn_sl_id) for syn_sl_id in geometry_adjacencies.nodes_iter())
        cursor.execute("""
            SELECT sl_obj.synapse_slice_id, sl_obj.synapse_object_id FROM synapse_slice_synapse_object sl_obj
              INNER JOIN (VALUES %s) syn_sl (id) ON (syn_sl.id = sl_obj.synapse_slice_id);
        """, (syn_slice_str, ))
        syn_slice_to_obj = dict(*cursor.fetchall())

        new_mappings = []
        obsolete_objects_strs = []

        for syn_slice_group in nx.connected_components(geometry_adjacencies):
            syn_obj_ids = {syn_slice_to_obj[slice_id] for slice_id in syn_slice_group if slice_id in syn_slice_to_obj}

            if len(syn_obj_ids) == 0:
                # create new synapse object
                cursor.execute("""
                    INSERT INTO synapse_object DEFAULT VALUES RETURNING id;
                """)
                obj_id = cursor.fetchone()[0]
            else:
                # decide which synapse object ID to use
                obj_id = min(syn_obj_ids)
                other_obj_ids = [other_obj_id for other_obj_id in syn_obj_ids if other_obj_id != obj_id]

                if other_obj_ids:
                    other_obj_ids_str = ', '.join(other_obj_ids)
                    obsolete_objects_strs.append(other_obj_ids_str)

                    # remap all synapse slices mapped to objects with larger IDs, and delete those objects
                    cursor.execute("""
                        UPDATE synapse_slice_synapse_object
                          SET synapse_object_id = %s WHERE synapse_object_id IN (%s);
                    """, (obj_id, other_obj_ids_str))

            new_mappings.extend((slice_id, obj_id) for slice_id in syn_obj_ids)

        # delete obsolete objects and insert slice -> object mappings for all new synapse slices
        # todo: delete objects with no slices?
        cursor.execute("""
            DELETE FROM synapse_object
              WHERE id IN (%s);
            INSERT INTO synapse_slice_synapse_object (synapse_slice_id, synapse_object_id)
              VALUES %s;
        """, (
            ', '.join(obsolete_objects_strs),
            ', '.join('({}, {})'.format(slice_id, obj_id) for slice_id, obj_id in new_mappings)
        ))

        cursor.close()
        conn.close()


def commit_node_association_results_from_queue(node_result_queue, skeleton_id, total_nodes, algo_version):
    conn = psycopg2.connect("dbname={} user={}".format(POSTGRES_DB, POSTGRES_USER), autocommit=True)
    cursor = conn.cursor()

    result_generator = (node_result_queue.get() for _ in range(total_nodes))
    values_to_insert = ', '.join(
        '({}, {}, {}, {})'.format(result.synapse_slice_id, skeleton_id, result.node_info.id, algo_version)
        for result in result_generator
    )

    cursor.execute("""
        INSERT INTO synapse_slice_skeleton (
          synapse_slice_id,
          skeleton_id,
          node_id,
          algo_version
        ) VALUES %s;
    """, (values_to_insert, ))

    cursor.close()
    conn.close()


if __name__ == "__main__":
    DEBUGGING = False
    if DEBUGGING:
        from os.path import dirname, abspath
        print("USING DEBUG ARGUMENTS")

        SKELETON_ID = '11524047'
        # SKELETON_ID = '18531735'
        L1_CNS = abspath( dirname(__file__) + '/../projects-2017/L1-CNS' )
        args_list = ['credentials_dev.json', 1, SKELETON_ID, L1_CNS]
        kwargs_dict = {'force': True}
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