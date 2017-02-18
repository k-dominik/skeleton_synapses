import os
import sys
import csv
import errno    
import signal
import shutil
import logging
import argparse
import tempfile
import warnings
from itertools import starmap

# Don't warn about duplicate python bindings for opengm
# (We import opengm twice, as 'opengm' 'opengm_with_cplex'.)
warnings.filterwarnings("ignore", message='.*to-Python converter for .*opengm.*', category=RuntimeWarning)

import numpy as np
import h5py
import vigra
from vigra.analysis import unique
from scipy.spatial.distance import euclidean

from lazyflow.graph import Graph
from lazyflow.request import Request
from lazyflow.utility import PathComponents, isUrl, Timer
from lazyflow.utility.io_util import TiledVolume

import ilastik_main
from ilastik.shell.headless.headlessShell import HeadlessShell
from ilastik.applets.dataSelection import DataSelectionApplet
from ilastik.applets.dataSelection.opDataSelection import DatasetInfo 
from ilastik.applets.thresholdTwoLevels import OpThresholdTwoLevels
from ilastik.workflows.newAutocontext.newAutocontextWorkflow import NewAutocontextWorkflowBase
from ilastik.applets.pixelClassification.opPixelClassification import OpPixelClassification

from skeleton_synapses.skeleton_utils import Skeleton, roi_around_node
from skeleton_synapses.progress_server import ProgressInfo, ProgressServer
from skeleton_utils import CSV_FORMAT

# Import requests in advance so we can silence its log messages.
import requests
logging.getLogger("requests").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

timing_logger = logging.getLogger(__name__ + '.timing')
timing_logger.setLevel(logging.INFO)

signal.signal(signal.SIGINT, signal.SIG_DFL) # Quit on Ctrl+C


MEMBRANE_CHANNEL = 0
SYNAPSE_CHANNEL = 2

# FIXME: This shouldn't be hard-coded.
ROI_RADIUS = 150

INFINITE_DISTANCE = 99999.0 

DEBUG_OUTPUT_DIR = ""   # Set this to enable debug images
                        # WARNING: The contents of this directory will be DELETED ON STARTUP!
                        #          (E.g. don't set it to your home directory...)

OUTPUT_COLUMNS = [ "synapse_id", "x_px", "y_px", "z_px", "size_px",
                   "detection_uncertainty",
                   "distance",
                   "node_id", "node_x_px", "node_y_px", "node_z_px",
                   "node_connector_id", "node_connector_distance_nm",
                   "node_connector_x_nm", "node_connector_y_nm", "node_connector_z_nm" ]

def main():
    parser = argparse.ArgumentParser() 
    parser.add_argument('skeleton_json')
    parser.add_argument('autocontext_project')
    parser.add_argument('volume_description')
    parser.add_argument('output_file')
    parser.add_argument('progress_port', nargs='?', type=int, default=8000)
    
    args = parser.parse_args()

    if DEBUG_OUTPUT_DIR:
        shutil.rmtree(DEBUG_OUTPUT_DIR, ignore_errors=True)
        mkdir_p(DEBUG_OUTPUT_DIR)
    
    # Read the volume resolution
    volume_description = TiledVolume.readDescription(args.volume_description)
    z_res, y_res, x_res = volume_description.resolution_zyx
    
    skeleton = Skeleton(args.skeleton_json, (x_res, y_res, z_res))
        
    
    # Start a server for others to poll progress.
    progress_server = ProgressServer.create_and_start( "localhost", args.progress_port )
    try:
        locate_synapses( args.autocontext_project, 
                         args.volume_description, 
                         args.output_file,
                         skeleton,
                         ROI_RADIUS,
                         
                         progress_callback=progress_server.update_progress )
    finally:
        progress_server.shutdown()


def locate_synapses( autocontext_project_path, 
                     input_filepath,
                     output_path, 
                     skeleton,
                     roi_radius_px,
                     progress_callback=lambda p: None ):
    """
    autocontext_project_path: Path to .ilp file.  Must use axis order 'xytc'.
    """    
    skeleton_branch_count = len(skeleton.branches)
    skeleton_node_count = sum( map(len, skeleton.branches) )

    shell = open_project(autocontext_project_path)
    assert isinstance(shell, HeadlessShell)
    assert isinstance(shell.workflow, NewAutocontextWorkflowBase)

    append_lane(shell.workflow, input_filepath, 'xyt')

    # We only use the final stage predictions
    opPixelClassification = shell.workflow.pcApplets[-1].topLevelOperator

    # Sanity checks
    assert isinstance(opPixelClassification, OpPixelClassification)
    assert opPixelClassification.Classifier.ready()
    assert opPixelClassification.HeadlessPredictionProbabilities[-1].meta.drange == (0.0, 1.0)
    axes = opPixelClassification.HeadlessPredictionProbabilities[-1].meta.getAxisKeys()
    assert axes == list('xytc'), \
        "Project {} has unexpected axis ordering: {}".format(autocontext_project_path, axes)

    relabeler = SynapseSliceRelabeler()

    with open(output_path, "w") as fout:
        csv_writer = csv.DictWriter(fout, OUTPUT_COLUMNS, **CSV_FORMAT)
        csv_writer.writeheader()

        node_overall_index = -1
        for branch_index, branch in enumerate(skeleton.branches):
            branch_node_count = len(branch)

            for node_index_in_branch, node_info in enumerate(branch):
                roi_xyz = roi_around_node(node_info, roi_radius_px)
                with Timer() as node_timer:
                    synapse_cc_xyz, predictions_xyzc = \
                        detections_for_node(opPixelClassification, relabeler, node_info, roi_xyz)
                    
                    # Write to csv
                    write_synapses( csv_writer, skeleton, node_info, roi_xyz, synapse_cc_xyz, predictions_xyzc )
                    fout.flush()

                    # Progress update (notify client)    
                    node_overall_index += 1
                    logger.debug("PROGRESS: node {}/{} ({:.1f}%) ({} detections)"
                                 .format(node_overall_index, skeleton_node_count,
                                         float(node_overall_index)/skeleton_node_count, relabeler.max_label))

                    progress_callback( ProgressInfo( node_overall_index, 
                                                     skeleton_node_count, 
                                                     branch_index, 
                                                     skeleton_branch_count, 
                                                     node_index_in_branch, 
                                                     branch_node_count,
                                                     relabeler.max_label ) )

                    timing_logger.debug( "NODE TIMER: {}".format( node_timer.seconds() ) )

    logger.info("DONE with skeleton.")


# opThreshold is global so we don't waste time initializing it repeatedly.
opThreshold = OpThresholdTwoLevels(graph=Graph())

def detections_for_node(opPixelClassification, relabeler, node_info, roi_xyz):
    roi_name = "x{}-y{}-z{}".format(*roi_xyz[0])
    skeleton_coord = (node_info.x_px, node_info.y_px, node_info.z_px)
    logger.debug("skeleton point: {}".format( skeleton_coord ))

    # Raw image (for debug output)
    raw_req = opPixelClassification.InputImages[-1](list(roi_xyz[0]) + [0], list(roi_xyz[1]) + [1])
    write_debug_image(raw_req, "raw", roi_name)

    # Predict
    num_classes = opPixelClassification.HeadlessPredictionProbabilities[-1].meta.shape[-1]
    roi_xyzc = np.append(roi_xyz, [[0],[num_classes]], axis=1)
    predictions_xyzc = opPixelClassification.HeadlessPredictionProbabilities[-1](*roi_xyzc).wait()
    predictions_xyzc = vigra.taggedView( predictions_xyzc, "xyzc" )
    write_debug_image(predictions_xyzc, "predictions", roi_name)

    # Threshold synapses
    opThreshold.Channel.setValue(SYNAPSE_CHANNEL)
    opThreshold.SingleThreshold.setValue(0.5)
    opThreshold.SmootherSigma.setValue({'x': 3.0, 'y': 3.0, 'z': 1.0})
    opThreshold.InputImage.setValue(predictions_xyzc)
    opThreshold.InputImage.meta.drange = (0.0, 1.0)
    synapse_cc_xyz = opThreshold.Output[:].wait()[...,0]
    
    # Relabel for consistency with previous slice
    synapse_cc_xyz = relabeler.normalize_synapse_ids(synapse_cc_xyz, roi_xyz)
    write_debug_image(synapse_cc_xyz[...,None], "synapse_cc", roi_name)
    
    return synapse_cc_xyz, predictions_xyzc


def open_project( project_path ):
    """
    Open a project file and return the HeadlessShell instance.
    """
    parsed_args = ilastik_main.parser.parse_args([])
    parsed_args.headless = True
    parsed_args.project = project_path

    shell = ilastik_main.main( parsed_args )
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

def write_debug_image(image_xyzc, name, name_prefix="", mode="stacked"):
    if not DEBUG_OUTPUT_DIR:
        return

    if isinstance(image_xyzc, Request):
        # Caller may provide a request instead of an image,
        # in which case we need to execute it now.
        image_xyzc = image_xyzc.wait()

    image_xyzc = vigra.taggedView(image_xyzc, 'xyzc')
    
    if mode == "slices":
        slice_name = name
        if name_prefix:
            slice_name = name_prefix + '-' + name
        with h5py.File(DEBUG_OUTPUT_DIR + "/" + slice_name + ".h5", 'w') as f:
            f.create_dataset("data", data=image_xyzc)

    elif mode == "stacked":    
        # Also append to an HDF5 stack
        with h5py.File(DEBUG_OUTPUT_DIR + "/" + name + ".h5") as f:
            if 'data' in f:
                # Add room for another z-slice
                z_size = f['data'].shape[2]
                f['data'].resize(z_size+1, 2)
            else:
                maxshape = np.array(image_xyzc.shape)
                maxshape[2] = 100000
                f.create_dataset('data', shape=image_xyzc.shape, maxshape=tuple(maxshape), dtype=image_xyzc.dtype)
                f['data'].attrs['axistags'] = vigra.defaultAxistags('xyzc').toJSON()
                f['data'].attrs['slice-names'] = []
    
            # Write onto the end of the stack.
            f['data'][:, :, -1:, :] = image_xyzc

            # Maintain an attribute 'slice-names' to list each slice's name
            z_size = f['data'].shape[2]
            names = list(f['data'].attrs['slice-names'])
            names += ["{}: {}".format(z_size-1, name_prefix)]
            del f['data'].attrs['slice-names']
            f['data'].attrs['slice-names'] = names

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
        to be consistent with those in 'previous_slice'.
        
        It is not assumed that the two slices are aligned:
        the slices' positions are given by current_roi and previous_roi.
        
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
    
        current_unique_labels = np.unique(current_slice)
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
            current_labels = np.unique(current_intersection_slice[prev_intersection_slice==cc].flat)
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


def write_synapses(csv_writer, skeleton, node_info, roi_xyz, synapse_cc_xyz, predictions_xyzc):
    synapseIds = set(synapse_cc_xyz.flat)
    synapseIds.remove(0)
    for sid in synapseIds:
        # find the pixel positions of this synapse
        syn_pixel_coords = np.where(synapse_cc_xyz[...,0] == sid)
        synapse_size = len( syn_pixel_coords[0] )
        syn_average_x = np.average(syn_pixel_coords[0])+roi_xyz[0,0]
        syn_average_y = np.average(syn_pixel_coords[1])+roi_xyz[0,1]

        # For now, we just compute euclidean distance to the node.
        distance_euclidean = euclidean((syn_average_x, syn_average_y), (node_info.x_px, node_info.y_px))

        # Determine average uncertainty
        # Get probabilities for this synapse's pixels
        flat_predictions = predictions_xyzc[synapse_cc_xyz == sid]
        # Sort along channel axis
        flat_predictions.sort(axis=-1)
        # What's the difference between the highest and second-highest class?
        certainties = flat_predictions[:,-1] - flat_predictions[:,-2]
        avg_certainty = np.mean(certainties)
        avg_uncertainty = 1.0 - avg_certainty

        fields = {}
        fields["synapse_id"] = int(sid)
        fields["x_px"] = int(syn_average_x + 0.5)
        fields["y_px"] = int(syn_average_y + 0.5)
        fields["z_px"] = roi_xyz[0,2]
        fields["size_px"] = synapse_size
        fields["distance"] = distance_euclidean
        fields["detection_uncertainty"] = avg_uncertainty
        fields["node_id"] = node_info.id
        fields["node_x_px"] = node_info.x_px
        fields["node_y_px"] = node_info.y_px
        fields["node_z_px"] = node_info.z_px

        try:
            node_connectors = skeleton.node_to_connector[node_info.id]
        except KeyError:
            fields["node_connector_id"] = -1
            fields["node_connector_distance_nm"] = INFINITE_DISTANCE
            fields["node_connector_x_nm"] = -1
            fields["node_connector_y_nm"] = -1
            fields["node_connector_z_nm"] = -1
        else:
            # FIXME: This assumes that there's only one connector associated with this node!
            connector_info = skeleton.connector_infos[node_connectors[0]]
            
            syn_x_nm = syn_average_x * skeleton.x_res
            syn_y_nm = syn_average_x * skeleton.y_res
            distance_nm = euclidean((syn_x_nm, syn_y_nm), (connector_info.x_nm, connector_info.y_nm))

            fields["node_connector_id"] = connector_info.id
            fields["node_connector_distance_nm"] = distance_nm
            fields["node_connector_x_nm"] = connector_info.x_nm
            fields["node_connector_y_nm"] = connector_info.y_nm
            fields["node_connector_z_nm"] = connector_info.z_nm
        
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
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

if __name__=="__main__":
    DEBUGGING = False
    if DEBUGGING:
        print "USING DEBUG ARGUMENTS"

        SKELETON_ID = '11524047'

        DEBUG_OUTPUT_DIR = "/magnetic/workspace/skeleton_synapses/L1-CNS-skeletons/{}/debug-images".format(SKELETON_ID) # See warning above. Be careful!

        autocontext_project = '/magnetic/workspace/skeleton_synapses/projects-2017/autocontext.ilp'
        volume_description = '/magnetic/workspace/skeleton_synapses/example/L1-CNS-description.json'
        skeleton_json = '/magnetic/workspace/skeleton_synapses/L1-CNS-skeletons/{}/tree_geometry.json'.format(SKELETON_ID)
        output_file = '/magnetic/workspace/skeleton_synapses/L1-CNS-skeletons/{}/{}-detections.csv'.format(SKELETON_ID, SKELETON_ID)

        sys.argv.append(skeleton_json)
        sys.argv.append(autocontext_project)
        sys.argv.append(volume_description)
        sys.argv.append(output_file)

    sys.exit( main() )
