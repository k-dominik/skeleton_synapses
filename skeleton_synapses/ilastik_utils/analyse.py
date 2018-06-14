import json
import logging
from collections import OrderedDict

import numpy as np
import vigra
from ilastik.applets.dataSelection import DatasetInfo
from ilastik.applets.edgeTrainingWithMulticut.opEdgeTrainingWithMulticut import OpEdgeTrainingWithMulticut
from ilastik.applets.thresholdTwoLevels import OpThresholdTwoLevels
from lazyflow.graph import Graph

from skeleton_synapses.dto import SynapseDetectionOutput
from skeleton_synapses.helpers.files import cached_synapses_predictions_for_roi, dump_images
from skeleton_synapses.helpers.roi import tile_index_to_bounds
from skeleton_synapses.helpers.segmentation import get_synapse_segment_overlaps, get_node_associations

from skeleton_synapses.helpers.images import are_same_xy
from skeleton_synapses.helpers.roi import roi_around_node


SYNAPSE_CHANNEL = 2


logger = logging.getLogger(__name__)


def fetch_predict_detect_segment(roi_xyz, opPixelClassification, multicut_workflow):
    """
    Run raw_data_for_node, predictions_for_node, and segmentation_for_node and return their results

    Parameters
    ----------
    node_info
    roi_radius_px
    opPixelClassification
    multicut_workflow
        multicut_shell.workflow

    Returns
    -------
    tuple
        predictions_xyc, synapse_cc_xy, segmentation_xy
    """

    # GET AND CLASSIFY PIXELS
    raw_xy = raw_data_for_roi(roi_xyz, opPixelClassification)
    predictions_xyc = _predictions_for_roi(roi_xyz, opPixelClassification)
    # DETECT SYNAPSES
    synapse_cc_xy = label_synapses(predictions_xyc)
    # SEGMENT
    segmentation_xy = segmentation_for_img(multicut_workflow, raw_xy, predictions_xyc)

    return raw_xy, predictions_xyc, synapse_cc_xy, segmentation_xy


def fetch_and_predict(roi_xyz, opPixelClassification):
    """
    Fetch raw data and perform pixel classification on it.

    Parameters
    ----------
    roi_xyz : np.array
    opPixelClassification

    Returns
    -------
    (vigra.VigraArray, vigra.VigraArray)
        raw_xy, predictions_xyc
    """
    logger.debug('fetching raw data')
    raw_xy = raw_data_for_roi(roi_xyz, opPixelClassification)
    logger.debug('predicting pixels')
    predictions_xyc = _predictions_for_roi(roi_xyz, opPixelClassification)
    return raw_xy, predictions_xyc


def raw_data_for_roi(roi_xyz, opPixelClassification):
    input_obj = opPixelClassification.InputImages[-1]

    # takes roi as zyxc, returns data in zyxc
    raw_zyxc = input_obj(list(roi_xyz[0][::-1]) + [0], list(roi_xyz[1][::-1]) + [1]).wait()
    raw_zyxc = vigra.taggedView(raw_zyxc, 'zyxc')
    raw_xy = raw_zyxc[0, :, :, 0].transposeToVigraOrder()
    return raw_xy


def _predictions_for_roi(roi_xyz, opPixelClassification):
    """Warning: should only be called when opPixelClassification has been populated with raw data"""
    num_classes = opPixelClassification.HeadlessPredictionProbabilities[-1].meta.shape[-1]
    roi_zyx = [item[::-1] for item in roi_xyz]
    roi_zyxc = np.append(roi_zyx, [[0], [num_classes]], axis=1)
    predictions_zyxc = opPixelClassification.HeadlessPredictionProbabilities[-1](*roi_zyxc).wait()
    predictions_zyxc = vigra.taggedView(predictions_zyxc, "zyxc")
    predictions_xyc = predictions_zyxc[0, :, :, :].transposeToVigraOrder()

    return predictions_xyc


# opThreshold is global so we don't waste time initializing it repeatedly.
opThreshold = OpThresholdTwoLevels(graph=Graph())


def label_synapses(predictions_xyc):
    """

    Parameters
    ----------
    predictions_xyc

    Returns
    -------
    vigra.VigraArray
        Numpy array of synapse labels, xy
    """
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

    return synapse_cc_xy


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


def detect_synapses(tile_size, opPixelClassification, tile_idx):
    """

    Parameters
    ----------
    tile_size : int
        Side length of a square tile
    opPixelClassification
    tile_idx : TileIndex

    Returns
    -------
    SynapseDetectionOutput
    """
    roi_xyz = tile_index_to_bounds(tile_idx, tile_size)
    logger.debug('fetching_and_predict in {}'.format(roi_xyz))
    raw_xy, predictions_xyc = fetch_and_predict(roi_xyz, opPixelClassification)
    logger.debug('label_synapses in {}'.format(roi_xyz))
    synapse_cc_xy = label_synapses(predictions_xyc)

    dump_images("ims_" + json.dumps(roi_xyz.tolist()), raw=raw_xy, predictions=predictions_xyc, synapse_cc=synapse_cc_xy)

    return SynapseDetectionOutput(tile_idx, predictions_xyc, synapse_cc_xy)


def associate_skeletons(hdf5_path, opPixelClassification, multicut_shell, catmaid, skeleton_association_input):
    """

    Parameters
    ----------
    hdf5_path : str or PathLike
    opPixelClassification
    multicut_shell
    catmaid : CatmaidSynapseSuggestionAPI
    skeleton_association_input : SkeletonAssociationInput

    Returns
    -------
    list of SkeletonAssociationOutput
    """
    roi_xyz, synapse_slice_ids, synapse_object_id = skeleton_association_input
    raw_xy = raw_data_for_roi(roi_xyz, opPixelClassification)
    synapse_cc_xy, predictions_xyc = cached_synapses_predictions_for_roi(roi_xyz, hdf5_path)

    logger.debug('Image shapes: \n\tRaw {}\n\tSynapse_cc {}\n\tPredictions {}'.format(
        raw_xy.shape, synapse_cc_xy.shape, predictions_xyc.shape
    ))

    segmentation_xy = segmentation_for_img(raw_xy, predictions_xyc, multicut_shell.workflow)

    overlapping_segments = get_synapse_segment_overlaps(synapse_cc_xy, segmentation_xy, synapse_slice_ids)
    logger.debug('Local segment: synapse slice overlaps found: \n{}'.format(overlapping_segments))

    if len(overlapping_segments) < 2:  # synapse is only in 1 segment
        return []

    node_locations = catmaid.get_nodes_in_roi(roi_xyz, catmaid.stack_id)
    if len(node_locations) == 0:
        logger.debug('ROI {} has no nodes'.format(roi_xyz))

    return get_node_associations(synapse_cc_xy, segmentation_xy, node_locations, overlapping_segments)
