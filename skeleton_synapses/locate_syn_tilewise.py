#!/usr/bin/env python
from __future__ import division
import logging
import csv
import multiprocessing as mp
from collections import namedtuple
import argparse
import sys
import os
import psycopg2
import h5py
import numpy as np
import json
from six.moves import range
from itertools import product

from lazyflow.utility import Timer
from lazyflow.request import Request

from skeleton_utils import CSV_FORMAT
from catmaid_interface import CatmaidAPI
from locate_synapses import (
    # constants/singletons
    OUTPUT_COLUMNS, DEFAULT_ROI_RADIUS, logger,
    # functions
    setup_files, setup_classifier, setup_multicut, roi_around_node, write_synapses, perform_segmentation,
    get_and_print_env, write_output_image, search_queue, fetch_raw_and_predict_for_node,
    labeled_synapses_for_node,
    # classes
    SynapseSliceRelabeler, CaretakerProcess, LeakyProcess
)

HDF5_PATH = 'image_store.hdf5'

POSTGRES_USER = 'cbarnes'
POSTGRES_DB = 'synapse_detection'

MIRROR = 0

DTYPE = np.uint32  # can only store up to 2^24
UNKNOWN_LABEL = 0  # should be smallest label
BACKGROUND_LABEL = 1  # should be smaller than synapse labels
TILE_SIDE = 512  # todo: get this from catmaid
CHUNK_SHAPE = (1, TILE_SIDE, TILE_SIDE)  # yxz

ALGO_VERSION = 1

THREADS = get_and_print_env('SYNAPSE_DETECTION_THREADS', 3, int)
NODES_PER_PROCESS = get_and_print_env('SYNAPSE_DETECTION_NODES_PER_PROCESS', 500, int)
RAM_MB_PER_PROCESS = get_and_print_env('SYNAPSE_DETECTION_RAM_MB_PER_PROCESS', 5000, int)


SegmenterInput = namedtuple('SegmenterInput', ['node_overall_index', 'node_info', 'roi_radius_px'])
SegmenterOutput = namedtuple('SegmenterOutput', ['node_overall_index', 'node_info', 'roi_radius_px', 'predictions_xyc',
                                                 'synapse_cc_xy', 'segmentation_xy'])


def main(credentials_path, stack_id, skeleton_id, project_dir, roi_radius_px=150, force=False):
    autocontext_project, multicut_project, volume_description_path, skel_output_dir, skeleton = setup_files(
        credentials_path, stack_id, skeleton_id, project_dir, force
    )

    catmaid = CatmaidAPI.from_json(credentials_path)
    stack_info = catmaid.get_stack_info(stack_id)

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
        conn = psycopg2.connect("dbname={} user={}".format('postgres', POSTGRES_USER))
        cursor = conn.cursor()
        cursor.execute('DROP DATABASE %s;', POSTGRES_DB)
        cursor.execute('CREATE DATABASE %s;', POSTGRES_DB)
        conn.commit()
        cursor.close()
        conn.close()

    conn = psycopg2.connect("dbname={} user={}".format(POSTGRES_DB, POSTGRES_USER))
    cursor = conn.cursor()
    cursor.execute("""
        CREATE EXTENSION IF NOT EXISTS postgis;
        
        CREATE TABLE IF NOT EXISTS synapse_slice (
          id SERIAL PRIMARY KEY,
          x_tile_idx INTEGER NOT NULL,
          y_tile_idx INTEGER NOT NULL,
          z_tile_idx INTEGER NOT NULL,
          bounding_box GEOMETRY NOT NULL,
          algo_version INTEGER NOT NULL,
          size_px INTEGER NOT NULL,
          uncertainty REAL NOT NULL,
          x_centroid_px INTEGER NOT NULL,
          y_centroid_px INTEGER NOT NULL
        );
        
        -- set ID sequence to start at 1, or 1 + the highest ID
        SELECT setval(
          'synapse_slice_id_seq',(SELECT GREATEST(MAX(id)+1,nextval('synapse_slice_id_seq')) FROM synapse_slice)
        );
        
        CREATE TABLE IF NOT EXISTS synapse_object (
          id SERIAL PRIMARY KEY
        );
        
        SELECT setval(
          'synapse_object_id_seq',(SELECT GREATEST(MAX(id)+1,nextval('synapse_object_id_seq')) FROM synapse_object)
        );
        
        CREATE TABLE IF NOT EXISTS synapse_slice_synapse_object (
          id SERIAL PRIMARY KEY,
          synapse_slice_id INTEGER UNIQUE REFERENCES synapse_slice (id) ON DELETE CASCADE,
          synapse_object_id INTEGER REFERENCES synapse_object (id) ON DELETE CASCADE
        );
        
        CREATE TABLE IF NOT EXISTS synapse_slice_skeleton (
          id SERIAL PRIMARY KEY,
          synapse_slice_id INTEGER REFERENCES synapse_slice (id) ON DELETE CASCADE,
          skeleton INTEGER
        );
    """)
    conn.commit()
    cursor.close()
    conn.close()


def create_label_volume(stack_info, hdf5_file, name, extra_dim=None):
    dimension = [stack_info['dimension'][dim] for dim in 'zyx']
    if extra_dim is not None:
        dimension += [extra_dim]

    labels = hdf5_file.create_dataset(
        name,
        dimension,  # zyx
        chunks=CHUNK_SHAPE,
        fillvalue=0,
        dtype=DTYPE
    )

    for key in ['translation', 'dimension', 'resolution']:
        labels.attrs[key] = json.dumps(stack_info[key])

    return labels


def get_tile_counts_zyx(stack_info):
    tile_size = {'z': 1}
    tile_size['y'], tile_size['x'] = [
        stack_info['mirrors'][MIRROR]['tile_{}'.format(dim)] for dim in ['height', 'width']
    ]
    return [int(stack_info['dimension'][dim] / tile_size[dim]) for dim in 'zyx']


def ensure_hdf5(stack_info, force=False):
    if force or not os.path.isfile(HDF5_PATH):
        with h5py.File(HDF5_PATH) as f:
            create_label_volume(stack_info, f, 'slice_labels')
            create_label_volume(stack_info, f, 'object_labels')
            create_label_volume(stack_info, f, 'pixel_predictions', 3)
            f.create_dataset(
                'algo_versions',
                get_tile_counts_zyx(stack_info),
                fillvalue=0,
                dtype=DTYPE
            )

TileIndex = namedtuple('TileIndex', 'z_idx y_idx x_idx')


def node_info_to_tiles(skeleton, mirror_info, minimum_radius=DEFAULT_ROI_RADIUS):
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

    tile_set = node_info_to_tiles(skeleton, stack_info, roi_radius_px)
    with h5py.File(HDF5_PATH, 'r') as f:
        algo_versions = f['algo_versions']
        for tile_idx in tile_set:
            if algo_versions[tile_idx.z_idx, tile_idx.y_idx, tile_idx.x_idx] != algo_version:
                tile_queue.put(tile_idx)

    total_tiles = tile_queue.qsize()
    logger.debug('{} tiles queued'.format(total_tiles))

    if total_tiles:
        mirror_info = stack_info['mirrors'][MIRROR]

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

        for detector_containers in detector_containers:
            detector_containers.join()

    # todo: segmentation

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


def write_synapses_from_queue(queue, output_path, skeleton, last_node_idx, synapse_output_dir, relabeler=None):
    """
    Relabel synapses if they are multi-labelled, write out the HDF5 of the synapse segmentation, and write synapses
    to a CSV.

    Parameters
    ----------
    queue
    output_path
    skeleton : Skeleton
    last_node_idx
    synapse_output_dir
    relabeler : SynapseSliceRelabeler

    """
    sought_node_idx = 0
    with open(output_path, "w") as fout:
        csv_writer = csv.DictWriter(fout, OUTPUT_COLUMNS, **CSV_FORMAT)
        csv_writer.writeheader()

        while sought_node_idx <= last_node_idx:
            node_overall_index, node_info, roi_radius_px, predictions_xyc, synapse_cc_xy, segmentation_xy = search_queue(
                queue, lambda x: x.node_overall_index == sought_node_idx
            )

            roi_xyz = roi_around_node(node_info, roi_radius_px)

            roi_name = "x{}-y{}-z{}".format(*roi_xyz[0])
            if relabeler:
                synapse_cc_xy = relabeler.normalize_synapse_ids(synapse_cc_xy, roi_xyz)
            write_output_image(synapse_output_dir, synapse_cc_xy[..., None], "synapse_cc", roi_name, mode="slices")

            write_synapses(
                csv_writer, skeleton, node_info, roi_xyz, synapse_cc_xy, predictions_xyc, segmentation_xy,
                node_overall_index
            )

            logger.debug('PROGRESS: Written CSV for node {} of {}'.format(
                sought_node_idx, last_node_idx
            ))

            sought_node_idx += 1

            fout.flush()


def roi_to_wkt(roi_xyz):
    raise NotImplementedError


def commit_tilewise_results_from_queue(tile_result_queue, output_path, total_tiles, mirror_info, algo_version=ALGO_VERSION):
    conn = psycopg2.connect("dbname={} user={}".format(POSTGRES_DB, POSTGRES_USER))
    cursor = conn.cursor()
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

            for slice_label in np.unique(synapse_cc_yx):
                if slice_label == 0:
                    continue

                syn_pixel_coords = np.where(synapse_cc_xy == slice_label)
                size_px = len(syn_pixel_coords[0])
                x_centroid_px = np.average(syn_pixel_coords[1]) + bounds_xyz[0, 0]
                y_centroid_px = np.average(syn_pixel_coords[0]) + bounds_xyz[0, 1]

                # Determine average uncertainty
                # Get probabilities for this synapse's pixels
                flat_predictions = predictions_xyc[synapse_cc_xy == slice_label]
                # Sort along channel axis
                flat_predictions.sort(axis=-1)
                # What's the difference between the highest and second-highest class?
                certainties = flat_predictions[:, -1] - flat_predictions[:, -2]
                avg_certainty = np.mean(certainties)
                uncertainty = 1.0 - avg_certainty


                cursor.execute("""
                    DELETE FROM synapse_slice 
                      WHERE z_tile_idx = %s AND y_tile_idx = %s AND x_tile_idx = %s;
                """, tile_idx)

                cursor.execute("""
                    INSERT INTO synapse_slice (
                      x_tile_idx,
                      y_tile_idx,
                      z_tile_idx,
                      bounding_box,
                      algo_version,
                      size_px,
                      uncertainty,
                      x_centroid_px,
                      y_centroid_px
                    ) VALUES (
                      %s, %s, %s, ST_GeomFromEWKT(%s), %s, %s, %s, %s, %s
                    ) RETURNING id;
                """,
                (
                    tile_idx.x_idx, tile_idx.y_idx, tile_idx.z_idx, roi_to_wkt(bounds_xyz), algo_version, size_px,
                    uncertainty, x_centroid_px, y_centroid_px
                )
                )

                conn.commit()

                new_id = cursor.fetchone()

                synapse_cc_yx[synapse_cc_yx == slice_label] = new_id

            slice_labels_zyx[
                bounds_xyz[0, 2], bounds_xyz[0, 1]:bounds_xyz[1, 1], bounds_xyz[0, 0]:bounds_xyz[1, 0]
            ] = synapse_cc_yx

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