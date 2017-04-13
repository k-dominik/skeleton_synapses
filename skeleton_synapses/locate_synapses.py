#!/usr/bin/env python
import os
import sys
import shutil
import csv
import errno
import signal
import logging
import argparse
import tempfile
import warnings
from itertools import starmap
from collections import OrderedDict, namedtuple
import json
import datetime
import multiprocessing as mp
import psutil

# Don't warn about duplicate python bindings for opengm
# (We import opengm twice, as 'opengm' 'opengm_with_cplex'.)
warnings.filterwarnings("ignore", message='.*second conversion method ignored.', category=RuntimeWarning)

# Start with a NullHandler to avoid logging configuration
# warnings before we actually configure logging below.
logging.getLogger().addHandler(logging.NullHandler())

import numpy as np
import h5py
import vigra
from vigra.analysis import unique
from scipy.spatial.distance import euclidean

from lazyflow.graph import Graph
from lazyflow.utility import PathComponents, isUrl

import ilastik_main
from ilastik.shell.headless.headlessShell import HeadlessShell
from ilastik.applets.dataSelection import DataSelectionApplet
from ilastik.applets.dataSelection.opDataSelection import DatasetInfo
from ilastik.applets.thresholdTwoLevels import OpThresholdTwoLevels
from ilastik.applets.pixelClassification.opPixelClassification import OpPixelClassification
from ilastik.applets.edgeTrainingWithMulticut.opEdgeTrainingWithMulticut import OpEdgeTrainingWithMulticut
from ilastik.workflows.newAutocontext.newAutocontextWorkflow import NewAutocontextWorkflowBase
from ilastik.workflows.edgeTrainingWithMulticut import EdgeTrainingWithMulticutWorkflow

from skeleton_utils import Skeleton, roi_around_node
from catmaid_interface import CatmaidAPI

# Import requests in advance so we can silence its log messages.
import requests
logging.getLogger("requests").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

signal.signal(signal.SIGINT, signal.SIG_DFL) # Quit on Ctrl+C

MEMBRANE_CHANNEL = 0
SYNAPSE_CHANNEL = 2

INFINITE_DISTANCE = 99999.0

PROJECT_NAME = 'L1-CNS'  # todo: remove dependency on this

OUTPUT_COLUMNS = [ "synapse_id", "skeleton_id", "overlaps_node_segment",
                   "x_px", "y_px", "z_px", "size_px",
                   "tile_x_px", "tile_y_px", "tile_index",
                   "distance_to_node_px",
                   "detection_uncertainty",
                   "node_id", "node_x_px", "node_y_px", "node_z_px",
                   'xmin', 'xmax', 'ymin', 'ymax']

DEFAULT_ROI_RADIUS = 150


def get_and_print_env(name, default, constructor=str):
    """

    Parameters
    ----------
    name
    default
    constructor : callable

    Returns
    -------

    """
    val = os.getenv(name)
    if val is None:
        print('{} environment variable not set, using default.'.format(name, default))
        val = default
    print('{} value is {}'.format(name[len('SYNAPSE_DETECTION_'):], val))
    return constructor(val)


THREADS = get_and_print_env('SYNAPSE_DETECTION_THREADS', 3, int)
NODES_PER_PROCESS = get_and_print_env('SYNAPSE_DETECTION_NODES_PER_PROCESS', 500, int)
RAM_MB_PER_PROCESS = get_and_print_env('SYNAPSE_DETECTION_RAM_MB_PER_PROCESS', 5000, int)


def setup_files(credentials_path, stack_id, skeleton_id, project_dir, force=False):
    """
    
    Parameters
    ----------
    credentials_path
    stack_id
    skeleton_id
    project_dir
    force

    Returns
    -------
    tuple of (str, str, skeleton_utils.Skeleton)
        (volume_description_path, skel_output_dir, skeleton)
    """
    catmaid = CatmaidAPI.from_json(credentials_path)

    include_offset = False

    volume_description_path = os.path.join(
        project_dir, PROJECT_NAME + '-description{}.json'.format('' if include_offset else '-NO-OFFSET')
    )

    if not os.path.isfile(volume_description_path):
        volume_description_dict = catmaid.get_stack_description(stack_id, include_offset=include_offset)
        with open(volume_description_path, 'w') as f:
            json.dump(volume_description_dict, f, sort_keys=True, indent=2)

    # Name the output directory with the skeleton id
    skel_output_dir = os.path.join(project_dir, 'skeletons', skeleton_id)
    if force:
        shutil.rmtree(skel_output_dir, ignore_errors=True)
    mkdir_p(skel_output_dir)

    skel_path = os.path.join(skel_output_dir, 'tree_geometry.json')
    skeleton = Skeleton.from_catmaid(catmaid, skeleton_id, stack_id, skel_path)

    return volume_description_path, skel_output_dir, skeleton


def perform_segmentation(node_info, roi_radius_px, skel_output_dir, opPixelClassification, multicut_workflow,
                         relabeler=None):
    """
    Run raw_data_for_node, predictions_for_node, and segmentation_for_node and return their results

    Parameters
    ----------
    node_info
    roi_radius_px
    skel_output_dir
    opPixelClassification
    multicut_workflow
        multicut_shell.workflow

    Returns
    -------
    tuple
        predictions_xyc, synapse_cc_xy, segmentation_xy
    """
    roi_xyz = roi_around_node(node_info, roi_radius_px)

    # GET AND CLASSIFY PIXELS
    raw_xy = raw_data_for_node(node_info, roi_xyz, skel_output_dir, opPixelClassification)
    predictions_xyc = predictions_for_node(node_info, roi_xyz, skel_output_dir, opPixelClassification)
    # DETECT SYNAPSES
    synapse_cc_xy = labeled_synapses_for_node(node_info, roi_xyz, skel_output_dir, predictions_xyc, relabeler)
    # SEGMENT
    segmentation_xy = segmentation_for_node(node_info, roi_xyz, skel_output_dir, multicut_workflow, raw_xy,
                                            predictions_xyc)

    return predictions_xyc, synapse_cc_xy, segmentation_xy


def setup_classifier_and_multicut(description_file, autocontext_project_path, multicut_project):
    """
    Boilerplate for getting the requisite ilastik interface objects and sanity-checking them

    Parameters
    ----------
    description_file : str
        Path to JSON stack description file
    autocontext_project_path : str
    multicut_project : str
        path

    Returns
    -------
    (OpPixelClassification, HeadlessShell)
        opPixelClassification, multicut_shell
    """
    autocontext_shell = open_project(autocontext_project_path, init_logging=True)
    assert isinstance(autocontext_shell, HeadlessShell)
    assert isinstance(autocontext_shell.workflow, NewAutocontextWorkflowBase)

    append_lane(autocontext_shell.workflow, description_file, 'xyt')

    # We only use the final stage predictions
    opPixelClassification = autocontext_shell.workflow.pcApplets[-1].topLevelOperator

    # Sanity checks
    assert isinstance(opPixelClassification, OpPixelClassification)
    assert opPixelClassification.Classifier.ready()
    assert opPixelClassification.HeadlessPredictionProbabilities[-1].meta.drange == (0.0, 1.0)

    multicut_shell = open_project(multicut_project, init_logging=False)
    assert isinstance(multicut_shell, HeadlessShell)
    assert isinstance(multicut_shell.workflow, EdgeTrainingWithMulticutWorkflow)

    return opPixelClassification, multicut_shell


SegmenterInput = namedtuple('SegmenterInput', ['node_overall_index', 'node_info', 'roi_radius_px'])
SegmenterOutput = namedtuple('SegmenterOutput', ['node_overall_index', 'node_info', 'roi_radius_px', 'predictions_xyc',
                                                 'synapse_cc_xy', 'segmentation_xy'])


class DebuggableProcess(mp.Process):
    """
    Classes inheriting from this instead of multiprocessing.Process can use a `debug` parameter to run in serial 
    instead of spawning a new python process, for easier debugging.
    """
    def __init__(self, debug=False):
        super(DebuggableProcess, self).__init__()
        self.debug = debug

    def start(self):
        if self.debug:
            self.run()
        else:
            super(DebuggableProcess, self).start()


def search_queue(queue, criterion, timeout=None):
    """
    Get the first item of the queue for which criterion(item) is truthy: items found before this one are returned
    to the back of the queue.

    Parameters
    ----------
    queue
    criterion : callable
        A callable which returns a truthy value for wanted items and a falsey value for unwanted items.
    timeout
        The timeout in seconds for each separate get operation while the queue is spooling through to
        find a wanted item.
    """
    while True:
        item = queue.get(timeout=timeout)
        if criterion(item):
            return item
        else:
            queue.put(item)


def raw_data_for_node(node_info, roi_xyz, output_dir, opPixelClassification):
    roi_name = "x{}-y{}-z{}".format(*roi_xyz[0])
    raw_xyzc = opPixelClassification.InputImages[-1](list(roi_xyz[0]) + [0], list(roi_xyz[1]) + [1]).wait()
    raw_xyzc = vigra.taggedView(raw_xyzc, 'xyzc')
    write_output_image(output_dir, raw_xyzc[:,:,0,:], "raw", roi_name, 'slices')
    raw_xy = raw_xyzc[:,:,0,0]
    return raw_xy


def predictions_for_node(node_info, roi_xyz, output_dir, opPixelClassification):
    """
    Run classification on the given node with the given operator.
    Returns: predictions_xyc
    """
    roi_name = "x{}-y{}-z{}".format(*roi_xyz[0])
    skeleton_coord = (node_info.x_px, node_info.y_px, node_info.z_px)
    logger.debug("skeleton point: {}".format( skeleton_coord ))

    # Predict
    num_classes = opPixelClassification.HeadlessPredictionProbabilities[-1].meta.shape[-1]
    roi_xyzc = np.append(roi_xyz, [[0],[num_classes]], axis=1)
    predictions_xyzc = opPixelClassification.HeadlessPredictionProbabilities[-1](*roi_xyzc).wait()
    predictions_xyzc = vigra.taggedView( predictions_xyzc, "xyzc" )
    predictions_xyc = predictions_xyzc[:,:,0,:]
    write_output_image(output_dir, predictions_xyc, "predictions", roi_name, mode='slices')
    return predictions_xyc


# opThreshold is global so we don't waste time initializing it repeatedly.
opThreshold = OpThresholdTwoLevels(graph=Graph())


def labeled_synapses_for_node(node_info, roi_xyz, output_dir, predictions_xyc, relabeler=None):
    roi_name = "x{}-y{}-z{}".format(*roi_xyz[0])
    skeleton_coord = (node_info.x_px, node_info.y_px, node_info.z_px)
    logger.debug("skeleton point: {}".format( skeleton_coord ))

    # Threshold synapses
    opThreshold.Channel.setValue(SYNAPSE_CHANNEL)
    opThreshold.SingleThreshold.setValue(0.5)
    opThreshold.SmootherSigma.setValue({'x': 3.0, 'y': 3.0, 'z': 1.0})
    opThreshold.MinSize.setValue(100)
    opThreshold.MaxSize.setValue(5000) # This is overshooting a bit.
    opThreshold.InputImage.setValue(predictions_xyc)
    opThreshold.InputImage.meta.drange = (0.0, 1.0)
    synapse_cc_xy = opThreshold.Output[:].wait()[...,0]
    synapse_cc_xy = vigra.taggedView(synapse_cc_xy, 'xy')

    # Relabel for consistency with previous slice
    if relabeler:
        synapse_cc_xy = relabeler.normalize_synapse_ids(synapse_cc_xy, roi_xyz)
        write_output_image(output_dir, synapse_cc_xy[..., None], "synapse_cc", roi_name, mode="slices")
    return synapse_cc_xy


def segmentation_for_node(node_info, roi_xyz, output_dir, multicut_workflow, raw_xy, predictions_xyc):
    roi_name = "x{}-y{}-z{}".format(*roi_xyz[0])
    skeleton_coord = (node_info.x_px, node_info.y_px, node_info.z_px)
    logger.debug("skeleton point: {}".format( skeleton_coord ))

    opEdgeTrainingWithMulticut = multicut_workflow.edgeTrainingWithMulticutApplet.topLevelOperator
    assert isinstance(opEdgeTrainingWithMulticut, OpEdgeTrainingWithMulticut)

    opDataExport = multicut_workflow.dataExportApplet.topLevelOperator
    opDataExport.OutputAxisOrder.setValue('xy')

    role_data_dict = OrderedDict([ ("Raw Data", [ DatasetInfo(preloaded_array=raw_xy) ]),
                                   ("Probabilities", [ DatasetInfo(preloaded_array=predictions_xyc) ])])
    batch_results = multicut_workflow.batchProcessingApplet.run_export(role_data_dict, export_to_array=True)
    assert len(batch_results) == 1
    segmentation_xy = batch_results[0]
    write_output_image(output_dir, segmentation_xy[:,:,None], "segmentation", roi_name, 'slices')
    return segmentation_xy


def open_project( project_path, init_logging=True ):
    """
    Open a project file and return the HeadlessShell instance.
    """
    parsed_args = ilastik_main.parser.parse_args([])
    parsed_args.headless = True
    parsed_args.project = project_path
    parsed_args.readonly = True

    shell = ilastik_main.main( parsed_args, init_logging=init_logging )
    return shell


def append_lane(workflow, input_filepath, axisorder=None):
    """
    Add a lane to the project file for the given input file.

    If axisorder is given, override the default axisorder for
    the file and force the project to use the given one.
    
    Globstrings are supported, in which case the files are converted to HDF5 first.
    """
    # If the filepath is a globstring, convert the stack to h5
    input_filepath = DataSelectionApplet.convertStacksToH5( [input_filepath], tempfile.mkdtemp() )[0]

    info = DatasetInfo()
    info.location = DatasetInfo.Location.FileSystem
    info.filePath = input_filepath

    comp = PathComponents(input_filepath)

    # Convert all (non-url) paths to absolute
    # (otherwise they are relative to the project file, which probably isn't what the user meant)
    if not isUrl(input_filepath):
        comp.externalPath = os.path.abspath(comp.externalPath)
        info.filePath = comp.totalPath()
    info.nickname = comp.filenameBase
    if axisorder:
        info.axistags = vigra.defaultAxistags(axisorder)

    logger.debug( "adding lane: {}".format( info ) )

    opDataSelection = workflow.dataSelectionApplet.topLevelOperator

    # Add a lane
    num_lanes = len( opDataSelection.DatasetGroup )+1
    logger.debug( "num_lanes: {}".format( num_lanes ) )
    opDataSelection.DatasetGroup.resize( num_lanes )

    # Configure it.
    role_index = 0 # raw data
    opDataSelection.DatasetGroup[-1][role_index].setValue( info )


initialized_files = set()


def write_output_image(output_dir, image_xyc, name, name_prefix="", mode="stacked"):
    """
    Write the given image to an hdf5 file.
    
    If mode is "slices", create a new file for the image.
    If mode is "stacked", create a new file with 'name' if it doesn't exist yet,
    or append to it if it does.
    """
    global initialized_files
    if not output_dir:
        return

    # Insert a Z-axis
    image_xyzc = vigra.taggedView(image_xyc[:,:,None,:], 'xyzc')

    if mode == "slices":
        output_subdir = os.path.join(output_dir, name)
        mkdir_p(output_subdir)
        if not name_prefix:
            name_prefix = datetime.datetime.now().isoformat()
        filepath = os.path.join(output_subdir, name_prefix + '.h5')
        with h5py.File(filepath, 'w') as f:
            f.create_dataset("data", data=image_xyzc)

    elif mode == "stacked":
        # If the file exists from a previous (failed) run,
        # delete it and start from scratch.
        filepath = os.path.join(output_dir, name + '.h5')
        if filepath not in initialized_files:
            try:
                os.unlink(filepath)
            except OSError as ex:
                if ex.errno != errno.ENOENT:
                    raise
            initialized_files.add(filepath)

        # Also append to an HDF5 stack
        with h5py.File(filepath) as f:
            if 'data' in f:
                # Add room for another z-slice
                z_size = f['data'].shape[2]
                f['data'].resize(z_size+1, 2)
            else:
                maxshape = np.array(image_xyzc.shape)
                maxshape[2] = 100000
                f.create_dataset('data', shape=image_xyzc.shape, maxshape=tuple(maxshape), dtype=image_xyzc.dtype)
                f['data'].attrs['axistags'] = image_xyzc.axistags.toJSON()
                f['data'].attrs['slice-names'] = []

            # Write onto the end of the stack.
            f['data'][:, :, -1:, :] = image_xyzc

            # Maintain an attribute 'slice-names' to list each slice's name
            z_size = f['data'].shape[2]
            names = list(f['data'].attrs['slice-names'])
            names += ["{}: {}".format(z_size-1, name_prefix)]
            del f['data'].attrs['slice-names']
            f['data'].attrs['slice-names'] = names

    else:
        raise ValueError('Image write mode {} not recognised.'.format(repr(mode)))

    return filepath


class SynapseSliceRelabeler(object):
    def __init__(self):
        self.max_label = 0
        self.previous_slice = None
        self.previous_roi = None

    def normalize_synapse_ids(self, current_slice, current_roi):
        """
        When the same synapse appears in two neighboring slices,
        we want it to have the same ID in both slices.
        
        This function will relabel the synapse labels in 'current_slice'
        to be consistent with those in self.previous_slice.
        
        It is not assumed that the two slices are aligned:
        the slices' positions are given by current_roi and self.previous_roi.
        
        Returns:
            (relabeled_slice, new_max_label)
            
        """
        current_roi = np.array(current_roi)
        intersection_roi = None
        if self.previous_roi is not None:
            previous_roi = self.previous_roi
            current_roi_2d = current_roi[:, :-1]
            previous_roi_2d = previous_roi[:, :-1]
            intersection_roi, current_intersection_roi, prev_intersection_roi = intersection( current_roi_2d, previous_roi_2d )

        current_unique_labels = unique(current_slice)
        assert current_unique_labels[0] == 0, "This function assumes that not all pixels belong to detections."
        if len(current_unique_labels) == 1:
            # No objects in this slice.
            self.previous_slice = None
            self.previous_roi = None
            return current_slice

        if intersection_roi is None or self.previous_slice is None or abs(int(current_roi[0,2]) - int(previous_roi[0,2])) > 1:
            # We want our synapse ids to be consecutive, so we do a proper relabeling.
            # If we could guarantee that the input slice was already consecutive, we could do this:
            # relabeled_current = np.where( current_slice, current_slice+previous_max_label, 0 )
            # ... but that's not the case.

            max_current_label = current_unique_labels[-1]
            relabel = np.zeros( (max_current_label+1,), dtype=np.uint32 )
            new_max_label = self.max_label + len(current_unique_labels)-1
            relabel[(current_unique_labels[1:],)] = np.arange( self.max_label+1, new_max_label+1, dtype=np.uint32 )
            relabeled_slice = relabel[current_slice]
            self.max_label = new_max_label
            self.previous_roi = current_roi
            self.previous_slice = relabeled_slice
            return relabeled_slice

        # Extract the intersecting region from the current/prev slices,
        #  so its easy to compare corresponding pixels
        current_intersection_slice = current_slice[slicing(current_intersection_roi)]
        prev_intersection_slice = self.previous_slice[slicing(prev_intersection_roi)]

        # omit label 0
        previous_slice_objects = unique(self.previous_slice)[1:]
        current_slice_objects = unique(current_slice)[1:]
        max_current_object = max(0, *current_slice_objects)
        relabel = np.zeros((max_current_object+1,), dtype=np.uint32)

        for cc in previous_slice_objects:
            current_labels = unique(current_intersection_slice[prev_intersection_slice==cc])
            for cur_label in current_labels:
                relabel[cur_label] = cc

        new_max_label = self.max_label
        for cur_object in current_slice_objects:
            if relabel[cur_object] == 0:
                relabel[cur_object] = new_max_label+1
                new_max_label = new_max_label+1

        # Relabel the entire current slice
        relabel[0] = 0
        relabeled_slice = relabel[current_slice]

        self.max_label = new_max_label
        self.previous_roi = current_roi
        self.previous_slice = relabeled_slice
        return relabeled_slice


def write_synapses(csv_writer, skeleton, node_info, roi_xyz, synapse_cc_xy, predictions_xyc, segmentation_xy, node_overall_index):
    """
    Given a slice of synapse segmentation and prediction images,
    append a CSV row (using the given writer) for each synapse detection in the slice. 
    """
    # Node is always located in the middle pixel, by definition.
    center_coord = np.array(segmentation_xy.shape) / 2
    node_segment = segmentation_xy[tuple(center_coord)]

    synapseIds = unique(synapse_cc_xy)
    for sid in synapseIds[1:]: # skip 0
        # find the pixel positions of this synapse
        syn_pixel_coords = np.where(synapse_cc_xy == sid)
        synapse_size = len( syn_pixel_coords[0] )
        syn_average_x = np.average(syn_pixel_coords[0])+roi_xyz[0,0]
        syn_average_y = np.average(syn_pixel_coords[1])+roi_xyz[0,1]

        # For now, we just compute euclidean distance to the node (in pixels).
        distance_euclidean = euclidean((syn_average_x, syn_average_y), (node_info.x_px, node_info.y_px))

        # Determine average uncertainty
        # Get probabilities for this synapse's pixels
        flat_predictions = predictions_xyc[synapse_cc_xy == sid]
        # Sort along channel axis
        flat_predictions.sort(axis=-1)
        # What's the difference between the highest and second-highest class?
        certainties = flat_predictions[:,-1] - flat_predictions[:,-2]
        avg_certainty = np.mean(certainties)
        avg_uncertainty = 1.0 - avg_certainty

        overlapping_segments = unique(segmentation_xy[synapse_cc_xy == sid])

        fields = {}
        fields["synapse_id"] = int(sid)
        fields["x_px"] = int(syn_average_x + 0.5)
        fields["y_px"] = int(syn_average_y + 0.5)
        fields["z_px"] = roi_xyz[0,2]

        fields["size_px"] = synapse_size
        fields["distance_to_node_px"] = distance_euclidean
        fields["detection_uncertainty"] = avg_uncertainty
        fields["skeleton_id"] = int(skeleton.skeleton_id)
        fields["overlaps_node_segment"] = {True: "true", False: "false"}[node_segment in overlapping_segments]

        fields["tile_x_px"] = int(syn_average_x + 0.5) - node_info.x_px + center_coord[0]
        fields["tile_y_px"] = int(syn_average_y + 0.5) - node_info.y_px + center_coord[1]
        fields["tile_index"] = node_overall_index

        fields["node_id"] = node_info.id
        fields["node_x_px"] = node_info.x_px
        fields["node_y_px"] = node_info.y_px
        fields["node_z_px"] = node_info.z_px

        fields["xmin"] = np.min(syn_pixel_coords[0]) + roi_xyz[0, 0]
        fields["xmax"] = np.max(syn_pixel_coords[0]) + roi_xyz[0, 0]
        fields["ymin"] = np.min(syn_pixel_coords[1]) + roi_xyz[0, 1]
        fields["ymax"] = np.max(syn_pixel_coords[1]) + roi_xyz[0, 1]

        assert len(fields) == len(OUTPUT_COLUMNS)
        csv_writer.writerow( fields )


def intersection(roi_a, roi_b):
    """
    Compute the intersection (overlap) of the two rois A and B.

    Returns the intersection roi in three forms (as a tuple):
        - in global coordinates
        - in coordinates relative to A
        - in coordinates relative to B
    
    If they don't overlap at all, returns (None, None, None).
    """
    roi_a = np.asarray(roi_a)
    roi_b = np.asarray(roi_b)
    assert roi_a.shape == roi_b.shape
    assert roi_a.shape[0] == 2

    out = roi_a.copy()
    out[0] = np.maximum( roi_a[0], roi_b[0] )
    out[1] = np.minimum( roi_a[1], roi_b[1] )

    if not (out[1] > out[0]).all():
        # No intersection; rois are disjoint
        return None, None, None

    out_within_a = out - roi_a[0]
    out_within_b = out - roi_b[0]
    return out, out_within_a, out_within_b


def slicing(roi):
    """
    Convert the roi to a slicing that can be used with ndarray.__getitem__()
    """
    return tuple( starmap( slice, zip(*roi) ) )


def mkdir_p(path):
    """
    Like the bash command 'mkdir -p'
    """
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
