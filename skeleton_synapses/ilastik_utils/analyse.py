import logging
from collections import OrderedDict

import numpy as np
import vigra
from ilastik.applets.dataSelection import DatasetInfo
from ilastik.applets.edgeTrainingWithMulticut.opEdgeTrainingWithMulticut import OpEdgeTrainingWithMulticut
from ilastik.applets.thresholdTwoLevels import OpThresholdTwoLevels
from lazyflow.graph import Graph

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
    raw_xy = raw_data_for_roi(roi_xyz, opPixelClassification)
    predictions_xyc = _predictions_for_roi(roi_xyz, opPixelClassification)
    return raw_xy, predictions_xyc


def raw_data_for_roi(roi_xyz, opPixelClassification):
    raw_xyzc = opPixelClassification.InputImages[-1](list(roi_xyz[0]) + [0], list(roi_xyz[1]) + [1]).wait()
    raw_xyzc = vigra.taggedView(raw_xyzc, 'xyzc')
    raw_xy = raw_xyzc[:, :, 0, 0]
    return raw_xy


def _predictions_for_roi(roi_xyz, opPixelClassification):
    """Warning: should only be called when opPixelClassification has been populated with raw data"""
    num_classes = opPixelClassification.HeadlessPredictionProbabilities[-1].meta.shape[-1]
    roi_xyzc = np.append(roi_xyz, [[0], [num_classes]], axis=1)
    predictions_xyzc = opPixelClassification.HeadlessPredictionProbabilities[-1](*roi_xyzc).wait()
    predictions_xyzc = vigra.taggedView(predictions_xyzc, "xyzc")
    predictions_xyc = predictions_xyzc[:, :, 0, :]

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
