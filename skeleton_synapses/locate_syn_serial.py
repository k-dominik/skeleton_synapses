#!/usr/bin/env python
import logging
import csv
import os
import argparse

from lazyflow.utility import Timer
from progress_server import ProgressInfo, ProgressServer

from skeleton_utils import CSV_FORMAT
from locate_synapses import (
    # constants/singletons
    OUTPUT_COLUMNS, PROJECT_NAME, DEFAULT_ROI_RADIUS, logger,
    # functions
    setup_files, setup_classifier_and_multicut, roi_around_node, write_synapses, perform_segmentation,
    # classes
    SynapseSliceRelabeler
)


def main(credentials_path, stack_id, skeleton_id, project_dir, roi_radius_px=150, progress_port=None, force=False):
    volume_description_path, skel_output_dir, skeleton = setup_files(
        credentials_path, stack_id, skeleton_id, project_dir, force
    )

    progress_server = None
    progress_callback = lambda p: None
    if progress_port is not None:
        # Start a server for others to poll progress.
        progress_server = ProgressServer.create_and_start( "localhost", progress_port )
        progress_callback = progress_server.update_progress
    try:
        autocontext_project = os.path.join(project_dir, 'projects', 'full-vol-autocontext.ilp')
        multicut_project = os.path.join(project_dir, 'projects', 'multicut', PROJECT_NAME + '-multicut.ilp')

        locate_synapses_serial(
            autocontext_project,
            multicut_project,
            volume_description_path,
            skel_output_dir,
            skeleton,
            roi_radius_px,
            progress_callback
        )
    finally:
        if progress_server:
            progress_server.shutdown()


def locate_synapses_serial(autocontext_project_path,
                    multicut_project,
                    input_filepath,
                    skel_output_dir,
                    skeleton,
                    roi_radius_px,
                    progress_callback=lambda p: None):
    """
    autocontext_project_path: Path to .ilp file.  Must use axis order 'xytc'.
    """
    output_path = skel_output_dir + "/skeleton-{}-synapses.csv".format(skeleton.skeleton_id)
    skeleton_branch_count = len(skeleton.branches)
    skeleton_node_count = sum(map(len, skeleton.branches))

    opPixelClassification, multicut_shell = setup_classifier_and_multicut(
        input_filepath, autocontext_project_path, multicut_project
    )

    timing_logger = logging.getLogger(__name__ + '.timing')
    timing_logger.setLevel(logging.INFO)

    relabeler = SynapseSliceRelabeler()

    with open(output_path, "w") as fout:
        csv_writer = csv.DictWriter(fout, OUTPUT_COLUMNS, **CSV_FORMAT)
        csv_writer.writeheader()

        node_overall_index = -1
        for branch_index, branch in enumerate(skeleton.branches):
            for node_index_in_branch, node_info in enumerate(branch):
                with Timer() as node_timer:
                    node_overall_index += 1
                    roi_xyz = roi_around_node(node_info, roi_radius_px)
                    skeleton_coord = (node_info.x_px, node_info.y_px, node_info.z_px)
                    logger.debug("skeleton point: {}".format( skeleton_coord ))

                    predictions_xyc, synapse_cc_xy, segmentation_xy = perform_segmentation(
                        node_info, roi_radius_px, skel_output_dir, opPixelClassification,
                        multicut_shell.workflow
                    )

                    write_synapses(
                        csv_writer, skeleton, node_info, roi_xyz, synapse_cc_xy, predictions_xyc, segmentation_xy,
                        node_overall_index
                    )
                    fout.flush()

                timing_logger.info( "NODE TIMER: {}".format( node_timer.seconds() ) )

                progress = 100*float(node_overall_index)/skeleton_node_count
                logger.debug("PROGRESS: node {}/{} ({:.1f}%) ({} detections)"
                             .format(node_overall_index, skeleton_node_count, progress, relabeler.max_label))

                # Progress: notify client
                progress_callback( ProgressInfo( node_overall_index,
                                                 skeleton_node_count,
                                                 branch_index,
                                                 skeleton_branch_count,
                                                 node_index_in_branch,
                                                 len(branch),
                                                 relabeler.max_label ) )
    logger.info("DONE with skeleton.")


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
        parser.add_argument('progress_port', nargs='?', type=int, default=0,
                            help="An http server will be launched on the given port (if nonzero), "
                                 "which can be queried to give information about progress.")
        parser.add_argument('-f', '--force', type=int, default=0,
                            help="Whether to delete all prior results for a given skeleton: pass 1 for true or 0")

        args = parser.parse_args()
        args_list = [
            args.credentials_path, args.stack_id, args.skeleton_id, args.project_dir, args.roi_radius_px,
            args.progress_port, args.force
        ]
        kwargs_dict = {}  # must be empty

    sys.exit( main(*args_list, **kwargs_dict) )