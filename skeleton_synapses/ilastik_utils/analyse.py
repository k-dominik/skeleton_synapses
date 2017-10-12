import logging
from collections import OrderedDict

import numpy as np
import vigra
from ilastik.applets.dataSelection import DatasetInfo
from ilastik.applets.edgeTrainingWithMulticut.opEdgeTrainingWithMulticut import OpEdgeTrainingWithMulticut
from ilastik.applets.thresholdTwoLevels import OpThresholdTwoLevels
from lazyflow.graph import Graph

from skeleton_synapses.helpers.files import write_output_image
from skeleton_synapses.helpers.images import are_same_xy
from skeleton_synapses.helpers.roi import roi_around_node


SYNAPSE_CHANNEL = 2


logger = logging.getLogger(__name__)


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
    segmentation_xy = segmentation_for_node(
        node_info, roi_xyz, skel_output_dir, multicut_workflow, raw_xy, predictions_xyc
    )

    return predictions_xyc, synapse_cc_xy, segmentation_xy


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
