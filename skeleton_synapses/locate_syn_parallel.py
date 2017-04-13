#!/usr/bin/env python
import logging
import csv
import os
import multiprocessing as mp
from collections import namedtuple
import warnings
import psutil
import argparse
import sys

from lazyflow.utility import Timer
from lazyflow.request import Request

from skeleton_utils import CSV_FORMAT
from locate_synapses import (
    # constants/singletons
    OUTPUT_COLUMNS, PROJECT_NAME, DEFAULT_ROI_RADIUS, logger,
    # functions
    setup_files, setup_classifier_and_multicut, roi_around_node, write_synapses, perform_segmentation,
    get_and_print_env, write_output_image, search_queue,
    # classes
    SynapseSliceRelabeler, DebuggableProcess
)

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

    locate_synapses_parallel(
        autocontext_project,
        multicut_project,
        volume_description_path,
        skel_output_dir,
        skeleton,
        roi_radius_px
    )


def locate_synapses_parallel(
        autocontext_project_path,
        multicut_project,
        input_filepath,
        skel_output_dir,
        skeleton,
        roi_radius_px
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
    output_path = skel_output_dir + "/skeleton-{}-synapses.csv".format(skeleton.skeleton_id)

    node_queue, result_queue = mp.Queue(), mp.Queue()

    node_overall_index = -1
    for branch_index, branch in enumerate(skeleton.branches):
        for node_index_in_branch, node_info in enumerate(branch):
            node_overall_index += 1

            node_queue.put(SegmenterInput(node_overall_index, node_info, roi_radius_px))

    logger.debug('{} nodes queued'.format(node_overall_index))

    segmenter_containers = [
        SegmenterCaretaker(
            node_queue, result_queue, input_filepath, autocontext_project_path, multicut_project,
            skel_output_dir, max_nodes=NODES_PER_PROCESS, max_ram_MB=RAM_MB_PER_PROCESS, debug=False
        )
        for _ in range(THREADS)
    ]

    for idx, segmenter_container in enumerate(segmenter_containers):
        segmenter_container.start()

    relabeler = SynapseSliceRelabeler()
    write_synapses_from_queue(result_queue, output_path, skeleton, node_overall_index, skel_output_dir, relabeler)

    for segmenter_container in segmenter_containers:
        segmenter_container.join()

    logger.info("DONE with skeleton.")


class SegmenterCaretaker(DebuggableProcess):
    """
    Process which takes care of spawning a SegmenterProcess, pruning it when it terminates, and starting a new one if
    there are still items remaining in the input queue.
    """
    def __init__(
            self, input_queue, output_queue, description_file, autocontext_project_path, multicut_project,
            skel_output_dir, max_nodes=0, max_ram_MB=0, debug=False
    ):
        super(SegmenterCaretaker, self).__init__(debug)
        self.segmenter_args = (input_queue, output_queue, description_file, autocontext_project_path, multicut_project,
            skel_output_dir, max_nodes, max_ram_MB, debug)

        self.input_queue = input_queue

    def run(self):
        while not self.input_queue.empty():
            segmenter = SegmenterProcess(*self.segmenter_args)
            logger.debug('Starting {}'.format(segmenter.name))
            segmenter.start()
            segmenter.join()
            del segmenter


class SegmenterProcess(DebuggableProcess):
    """
    Process which creates its own pixel classifier and multicut workflow, pulls jobs from one queue and returns
    outputs to another queue.
    """
    def __init__(
            self, input_queue, output_queue, description_file, autocontext_project_path, multicut_project,
            skel_output_dir, max_nodes=0, max_ram_MB=0, debug=False
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
        super(SegmenterProcess, self).__init__(debug)
        self.input_queue = input_queue
        self.output_queue = output_queue

        logger.debug('Segmenter process {} instantiated'.format(self.name))

        self.timing_logger = logging.getLogger(__name__ + '.timing')
        self.timing_logger.setLevel(logging.INFO)

        self.skel_output_dir = skel_output_dir

        self.count = float('-inf')

        self.count = 0
        self.max_nodes = max_nodes
        self.max_ram_MB = max_ram_MB

        self.opPixelClassification = self.multicut_shell = None
        self.setup_args = (description_file, autocontext_project_path, multicut_project)

        if not max_nodes and not max_ram_MB:
            warnings.warn('If segmenter processes are not pruned using the max_nodes or max_ram_MB argument, '
                          'the program may crash due to a memory leak in ilastik')

        self.psutil_process = None

    def run(self):
        """
        Pull node information from the input queue, generate pixel predictions, synapse labels and cell
        segmentations, and put them on the output queue.
        """
        self.opPixelClassification, self.multicut_shell = setup_classifier_and_multicut(
            *self.setup_args
        )

        self.psutil_process = [proc for proc in psutil.process_iter() if proc.pid == self.pid][0]
        Request.reset_thread_pool(1)
        while not self.input_queue.empty():
            if self.needs_pruning():
                return

            node_overall_index, node_info, roi_radius_px = self.input_queue.get()

            logger.debug("{} PROGRESS: addressing node {}, {} nodes remaining"
                         .format(self.name.upper(), node_overall_index, self.input_queue.qsize()))

            with Timer() as node_timer:
                predictions_xyc, synapse_cc_xy, segmentation_xy = perform_segmentation(
                    node_info, roi_radius_px, self.skel_output_dir, self.opPixelClassification,
                    self.multicut_shell.workflow
                )
                self.timing_logger.info("NODE TIMER: {}".format(node_timer.seconds()))

            logger.debug("{} PROGRESS: segmented area around node {}, {} nodes remaining"
                         .format(self.name.upper(), node_overall_index, self.input_queue.qsize()))

            self.output_queue.put(SegmenterOutput(node_overall_index, node_info, roi_radius_px, predictions_xyc,
                                                  synapse_cc_xy, segmentation_xy))

    def needs_pruning(self):
        """
        Return True if either the maximum number of nodes is exceeded, or the total RAM usage is

        Returns
        -------
        bool
            Whether the process should gracefully quit
        """
        self.count += 1
        if self.max_nodes and self.count > self.max_nodes:
            return True
        elif self.max_ram_MB and self.psutil_process.memory_info().rss >= self.max_ram_MB * 1024 * 1024:
            return True
        else:
            return False


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