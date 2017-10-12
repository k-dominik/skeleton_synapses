#!/usr/bin/env python
import os
import shutil
import errno
import signal
import logging
import warnings
from itertools import starmap
from collections import OrderedDict, namedtuple
import json
import datetime
import multiprocessing as mp
import psutil
import tempfile

# Don't warn about duplicate python bindings for opengm
# (We import opengm twice, as 'opengm' 'opengm_with_cplex'.)
warnings.filterwarnings("ignore", message='.*second conversion method ignored.', category=RuntimeWarning)

# Start with a NullHandler to avoid logging configuration
# warnings before we actually configure logging below.
# logging.getLogger().addHandler(logging.NullHandler())

import six

import numpy as np
import h5py
import vigra

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
from catpy import CatmaidClient

from catmaid_interface import CatmaidSynapseSuggestionAPI
from skeleton_utils import roi_around_node

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

TQDM_KWARGS = {
    'ncols': 50,
}
ROOT_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..')


def ensure_list(value):
    """Ensure that a given value is a non-string sequence (making it the sole element of a list if not).

    Used for compatibility purposes.
    """
    try:
        len(value)
        if isinstance(value, six.string_types):
            raise TypeError
        return value
    except TypeError:
        return [value]


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
        logger.info('{} environment variable not set, using default.'.format(name, default))
        val = default
    logger.info('{} value is {}'.format(name[len('SYNAPSE_DETECTION_'):], val))
    return constructor(val)


def ensure_description_file(catmaid, description_path, stack_id, include_offset=False, force=False):
    """

    Parameters
    ----------
    catmaid : CatmaidSynapseSuggestionAPI
    description_path : str
    stack_id : int
    include_offset : bool

    Returns
    -------
    bool
        Whether a new volume description was created
    """
    if force or not os.path.isfile(description_path):
        volume_description_dict = catmaid.get_stack_description(stack_id, include_offset=include_offset)
        with open(description_path, 'w') as f:
            json.dump(volume_description_dict, f, sort_keys=True, indent=2)
        return True


def ensure_skel_output_dirs(output_file_dir, skel_ids, catmaid_ss, stack_id, force=False):
    skel_output_dirs = []
    for skeleton_id in ensure_list(skel_ids):
        # Name the output directory with the skeleton id
        skel_output_dir = os.path.join(output_file_dir, 'skeletons', str(skeleton_id))
        if force:
            try:
                shutil.rmtree(skel_output_dir, ignore_errors=True)
            except OSError:
                pass

        mkdir_p(skel_output_dir)

        skel_path = os.path.join(skel_output_dir, 'tree_geometry.json')
        skel_data = catmaid_ss.get_transformed_treenode_and_connector_geometry(stack_id, skeleton_id)
        with open(skel_path, 'w') as f:
            json.dump(skel_data, f)

        skel_output_dirs.append(skel_output_dir)

    return skel_output_dirs


def get_algo_notes(projects_dir):
    try:
        with open(os.path.join(projects_dir, 'algorithm_notes.json')) as f:
            algo_notes = json.load(f)
    except IOError:
        logger.warning('Algorithm notes not found, using empty strings')
        algo_notes = {'synapse_detection': '', 'skeleton_association': ''}

    return algo_notes


def setup_files(
        credentials_path, stack_id, skeleton_ids, input_file_dir, force=False, output_file_dir=None
):
    """

    Parameters
    ----------
    credentials_path
    stack_id
    skeleton_ids
    input_file_dir
    force
    output_file_dir

    Returns
    -------
    tuple of (str, str, str, list of str, dict of {str:str})
    """
    catmaid = CatmaidSynapseSuggestionAPI(CatmaidClient.from_json(credentials_path), stack_id)
    projects_dir = os.path.join(input_file_dir, 'projects')

    autocontext_project = os.path.join(projects_dir, 'full-vol-autocontext.ilp')
    multicut_project = os.path.join(projects_dir, 'multicut', PROJECT_NAME + '-multicut.ilp')

    volume_description_path = os.path.join(
        input_file_dir, PROJECT_NAME + '-description-NO-OFFSET.json'
    )
    ensure_description_file(catmaid, volume_description_path, stack_id, include_offset=False)

    skeleton_ids = ensure_list(skeleton_ids)
    if output_file_dir:
        skel_output_dirs = ensure_skel_output_dirs(output_file_dir, skeleton_ids, catmaid, stack_id, force)
    else:
        skel_output_dirs = [None for _ in skeleton_ids]

    algo_notes = get_algo_notes(projects_dir)

    return autocontext_project, multicut_project, volume_description_path, skel_output_dirs, algo_notes


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


def setup_classifier(description_file, autocontext_project_path):
    logger.debug('Setting up opPixelClassification')
    autocontext_shell = open_project(autocontext_project_path, init_logging=False)
    assert isinstance(autocontext_shell, HeadlessShell)
    assert isinstance(autocontext_shell.workflow, NewAutocontextWorkflowBase)

    append_lane(autocontext_shell.workflow, description_file, 'xyt')

    # We only use the final stage predictions
    opPixelClassification = autocontext_shell.workflow.pcApplets[-1].topLevelOperator

    # Sanity checks
    assert isinstance(opPixelClassification, OpPixelClassification)
    assert opPixelClassification.Classifier.ready()
    assert opPixelClassification.HeadlessPredictionProbabilities[-1].meta.drange == (0.0, 1.0)

    return opPixelClassification


def setup_multicut(multicut_project):
    logger.debug('Setting up multicut_shell')
    multicut_shell = open_project(multicut_project, init_logging=False)
    assert isinstance(multicut_shell, HeadlessShell)
    assert isinstance(multicut_shell.workflow, EdgeTrainingWithMulticutWorkflow)

    return multicut_shell


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
    logger.debug('Setting up opPixelClassification and multicut_shell')
    opPixelClassification = setup_classifier(description_file, autocontext_project_path)
    multicut_shell = setup_multicut(multicut_project)

    return opPixelClassification, multicut_shell


SegmenterInput = namedtuple('SegmenterInput', ['node_overall_index', 'node_info', 'roi_radius_px'])
SegmenterOutput = namedtuple('SegmenterOutput', ['node_overall_index', 'node_info', 'roi_radius_px', 'predictions_xyc',
                                                 'synapse_cc_xy', 'segmentation_xy'])


def fetch_raw_and_predict_for_node(node_info, roi_xyz, output_dir, opPixelClassification):
    """

    Parameters
    ----------
    node_info : NodeInfo
        Optional, for logging purposes
    roi_xyz
    output_dir
    opPixelClassification

    Returns
    -------
    array-like
        Pixel predictions, xyc
    """
    raw_data_for_node(None, roi_xyz, output_dir, opPixelClassification)
    return predictions_for_node(node_info, roi_xyz, output_dir, opPixelClassification)


def raw_data_for_node(node_info, roi_xyz, output_dir, opPixelClassification):
    """
    DEPRECATED. This should only be called through fetch_raw_and_predict_for_node. Left for compatibility purposes.

    Parameters
    ----------
    node_info : None
        Not required
    roi_xyz
    output_dir
    opPixelClassification

    Returns
    -------

    """
    return raw_data_for_roi(roi_xyz, output_dir, opPixelClassification)


def raw_data_for_roi(roi_xyz, output_dir, opPixelClassification):
    roi_name = "x{}-y{}-z{}".format(*roi_xyz[0])
    raw_xyzc = opPixelClassification.InputImages[-1](list(roi_xyz[0]) + [0], list(roi_xyz[1]) + [1]).wait()
    raw_xyzc = vigra.taggedView(raw_xyzc, 'xyzc')
    if output_dir:
        write_output_image(output_dir, raw_xyzc[:, :, 0, :], "raw", roi_name, 'slices')
    raw_xy = raw_xyzc[:, :, 0, 0]
    return raw_xy


def cached_synapses_predictions_for_roi(roi_xyz, hdf5_path, squeeze=True):
    """

    Parameters
    ----------
    roi_xyz
    hdf5_path
    squeeze

    Returns
    -------
    (vigra.VigraArray, vigra.VigraArray)
    """
    # convert roi into a tuple of slice objects which can be used by numpy for indexing
    roi_slices = roi_xyz[0, 2], slice(roi_xyz[0, 1], roi_xyz[1, 1]), slice(roi_xyz[0, 0], roi_xyz[1, 0])

    with h5py.File(hdf5_path, 'r') as f:
        synapse_cc_xy = vigra.taggedView(
            f['slice_labels'], axistags=f['slice_labels'].attrs['axistags']
        )[roi_slices].transposeToOrder('V')
        predictions_xyc = vigra.taggedView(
            f['pixel_predictions'], axistags=f['pixel_predictions'].attrs['axistags']
        )[roi_slices].transposeToOrder('V')

    # if squeeze:
    #     return synapse_cc_xy.squeeze(), predictions_xyc.squeeze()
    # else:
    return synapse_cc_xy, predictions_xyc


def predictions_for_node(node_info, roi_xyz, output_dir, opPixelClassification):
    """
    DEPRECATED. This should only be called through fetch_raw_and_predict_for_node.

    Run classification on the given node with the given operator.

    Parameters
    ----------
    node_info : NodeInfo
        Optional, for logging purposes
    roi_xyz : array-like
    output_dir : str
        Directory in which data should be dumped
    opPixelClassification

    Returns
    -------
    array-like
        Pixel predictions, xyc
    """
    roi_name = "x{}-y{}-z{}".format(*roi_xyz[0])
    if node_info:
        skeleton_coord = (node_info.x_px, node_info.y_px, node_info.z_px)
        logger.debug("skeleton point: {}".format( skeleton_coord ))
    else:
        logger.debug('roi name: {}'.format(roi_name))

    # Predict
    num_classes = opPixelClassification.HeadlessPredictionProbabilities[-1].meta.shape[-1]
    roi_xyzc = np.append(roi_xyz, [[0],[num_classes]], axis=1)
    predictions_xyzc = opPixelClassification.HeadlessPredictionProbabilities[-1](*roi_xyzc).wait()
    predictions_xyzc = vigra.taggedView( predictions_xyzc, "xyzc" )
    predictions_xyc = predictions_xyzc[:,:,0,:]
    if output_dir:
        write_output_image(output_dir, predictions_xyc, "predictions", roi_name, mode='slices')
    return predictions_xyc


# opThreshold is global so we don't waste time initializing it repeatedly.
opThreshold = OpThresholdTwoLevels(graph=Graph())


def labeled_synapses_for_node(node_info, roi_xyz, output_dir, predictions_xyc, relabeler=None):
    """

    Parameters
    ----------
    node_info : NodeInfo
        Optional, for logging purposes
    roi_xyz : array-like
    output_dir : str
        Directory in which data should be dumped
    predictions_xyc
    relabeler : SynapseSliceRelabeler

    Returns
    -------
    array-like
        Numpy array of synapse labels, xy
    """
    roi_name = "x{}-y{}-z{}".format(*roi_xyz[0])
    if node_info:
        skeleton_coord = (node_info.x_px, node_info.y_px, node_info.z_px)
        logger.debug("skeleton point: {}".format( skeleton_coord ))
    else:
        logger.debug('roi name: {}'.format(roi_name))

    # Threshold synapses
    opThreshold.Channel.setValue(SYNAPSE_CHANNEL)
    opThreshold.LowThreshold.setValue(0.5)
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

    if output_dir:
        write_output_image(output_dir, synapse_cc_xy[..., None], "synapse_cc", roi_name, mode="slices")
    return synapse_cc_xy


def segmentation_for_node(node_info, roi_xyz, output_dir, multicut_workflow, raw_xy, predictions_xyc):
    """

    Parameters
    ----------
    node_info : NodeInfo
        Optional, for logging purposes
    roi_xyz
    output_dir
    multicut_workflow
    raw_xy : vigra.VigraArray
    predictions_xyc : vigra.VigraArray

    Returns
    -------

    """
    roi_name = "x{}-y{}-z{}".format(*roi_xyz[0])
    skeleton_coord = (node_info.x_px, node_info.y_px, node_info.z_px)
    logger.debug("skeleton point: {}".format( skeleton_coord ))

    segmentation_xy = segmentation_for_img(raw_xy, predictions_xyc, multicut_workflow)

    if output_dir:
        write_output_image(output_dir, segmentation_xy[:, :, None], "segmentation", roi_name, 'slices')
    return segmentation_xy


def are_same_xy(*args):
    if not args:
        return True

    return all(
        args[0].shape[:2] == arg.shape[:2] and tuple(args[0].axistags)[:2] == tuple(arg.axistags)[:2] for arg in args
    )


def segmentation_for_img(raw_xy, predictions_xyc, multicut_workflow):
    """

    Parameters
    ----------
    raw_xy : vigra.VigraArray
    predictions_xyc : vigra.VigraArray
    multicut_workflow

    Returns
    -------

    """
    assert are_same_xy(raw_xy, predictions_xyc)

    # move these into setup_multicut?
    #####
    opEdgeTrainingWithMulticut = multicut_workflow.edgeTrainingWithMulticutApplet.topLevelOperator
    assert isinstance(opEdgeTrainingWithMulticut, OpEdgeTrainingWithMulticut)

    opDataExport = multicut_workflow.dataExportApplet.topLevelOperator
    opDataExport.OutputAxisOrder.setValue('xy')
    #####

    role_data_dict = OrderedDict([
        ("Raw Data", [DatasetInfo(preloaded_array=raw_xy)]),
        ("Probabilities", [DatasetInfo(preloaded_array=predictions_xyc)])
    ])
    batch_results = multicut_workflow.batchProcessingApplet.run_export(role_data_dict, export_to_array=True)

    assert len(batch_results) == 1
    segmentation_xy = vigra.taggedView(batch_results[0], axistags='xy')
    assert are_same_xy(segmentation_xy, raw_xy, predictions_xyc)
    return segmentation_xy


def open_project( project_path, init_logging=False ):
    """
    Open a project file and return the HeadlessShell instance.
    """
    # todo: do not init logging?
    parsed_args = ilastik_main.parser.parse_args([])
    parsed_args.headless = True
    parsed_args.project = project_path
    parsed_args.readonly = True
    parsed_args.debug = True  # possibly delete this?

    shell = ilastik_main.main( parsed_args, init_logging=init_logging )
    return shell


def append_lane(workflow, input_filepath, axisorder=None):
    """
    Add a lane to the project file for the given input file.

    If axisorder is given, override the default axisorder for
    the file and force the project to use the given one.

    Globstrings are supported, in which case the files are converted to HDF5 first.
    """
    # If the filepath is a globstring, convert the stack to h5  # todo: skip this?
    tmp_dir = tempfile.mkdtemp()
    input_filepath = DataSelectionApplet.convertStacksToH5( [input_filepath], tmp_dir )[0]

    try:
        os.rmdir(tmp_dir)
    except OSError as e:
        if e.errno == 39:
            logger.warning('Temporary directory {} was populated: should be deleted')
        else:
            raise

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
