import os
import collections
import numpy
import csv
import vigra
from vigra import graphs
import time

import ilastik_main
from ilastik.workflows.pixelClassification import PixelClassificationWorkflow
from ilastik.applets.dataSelection import DataSelectionApplet
from ilastik.applets.thresholdTwoLevels import OpThresholdTwoLevels
from ilastik.applets.dataSelection.opDataSelection import DatasetInfo 
from lazyflow.operators.vigraOperators import OpPixelFeaturesPresmoothed
from lazyflow.graph import Graph
from lazyflow.utility import PathComponents, isUrl, Timer

from lazyflow.roi import roiToSlice, getIntersection
from lazyflow.utility.io import TiledVolume

from skeleton_synapses.opCombinePredictions import OpCombinePredictions
from skeleton_synapses.opUpsampleByTwo import OpUpsampleByTwo
from skeleton_synapses.skeleton_utils import parse_skeleton_swc, parse_skeleton_json, construct_tree, nodes_and_rois_for_tree
from skeleton_synapses.progress_server import ProgressInfo, ProgressServer
from skeleton_utils import CSV_FORMAT

THRESHOLD = 5
MEMBRANE_CHANNEL = 0
SYNAPSE_CHANNEL = 2

import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

import tempfile
TMP_DIR = tempfile.gettempdir()
import logging

# Import requests in advance so we can silence its log messages.
import requests
logging.getLogger("requests").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

timing_logger = logging.getLogger(__name__ + '.timing')
timing_logger.setLevel(logging.INFO)

OUTPUT_COLUMNS = ["synapse_id", "x_px", "y_px", "z_px", "size_px", "distance", "detection_uncertainty", "node_id", "node_x_px", "node_y_px", "node_z_px"]

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
    # Sanity checks
    assert isinstance(workflow, PixelClassificationWorkflow)
    opPixelClassification = workflow.pcApplet.topLevelOperator
    assert opPixelClassification.Classifier.ready()

    # If the filepath is a globstring, convert the stack to h5
    input_filepath = DataSelectionApplet.convertStacksToH5( [input_filepath], TMP_DIR )[0]

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

    # Sanity check
    assert len(opPixelClassification.InputImages) == num_lanes
    
    return opPixelClassification


def locate_synapses(project3dname, 
                    project2dname, 
                    input_filepath, 
                    output_path, 
                    branchwise_rois, 
                    debug_images=False, 
                    order2d='xyz', 
                    order3d='xyt', 
                    progress_callback=lambda p: None):
    outdir = os.path.split( output_path )[0]
    
    shell3d = open_project(project3dname)
    shell2d = open_project(project2dname)

    opPixelClassification3d = append_lane(shell3d.workflow, input_filepath, order3d) # Z
    logger.debug( "appended 3d lane" )
    opPixelClassification2d = append_lane(shell2d.workflow, input_filepath, order2d) # T
    logger.debug( "appended 2d lane" )
    
    # Combine
    tempGraph = Graph()
    opCombinePredictions = OpCombinePredictions(SYNAPSE_CHANNEL, MEMBRANE_CHANNEL, graph=tempGraph)
    opPixelClassification3d.FreezePredictions.setValue(False)
    opCombinePredictions.SynapsePredictions.connect( opPixelClassification3d.PredictionProbabilities[-1], permit_distant_connection=True )
    opCombinePredictions.MembranePredictions.connect( opPixelClassification2d.HeadlessPredictionProbabilities[-1], permit_distant_connection=True )

    #data_shape_3d = input_data.shape[0:3]    
    opUpsample = OpUpsampleByTwo(graph = tempGraph)
    opUpsample.Input.connect(opCombinePredictions.Output)
    
    opFeatures = OpPixelFeaturesPresmoothed(graph=tempGraph)
    
    # Compute the Hessian slicewise and create gridGraphs
    standard_scales = [0.3, 0.7, 1.0, 1.6, 3.5, 5.0, 10.0]
    standard_feature_ids = ['GaussianSmoothing', 'LaplacianOfGaussian', \
                            'GaussianGradientMagnitude', 'DifferenceOfGaussians', \
                            'StructureTensorEigenvalues', 'HessianOfGaussianEigenvalues']

    opFeatures.Scales.setValue(standard_scales)
    opFeatures.FeatureIds.setValue(standard_feature_ids)
    
    # Select Hessian Eigenvalues at scale 5.0
    scale_index = standard_scales.index(5.0)
    feature_index = standard_feature_ids.index('HessianOfGaussianEigenvalues')
    selection_matrix = numpy.zeros( (6,7), dtype=bool ) # all False
    selection_matrix[feature_index][scale_index] = True
    opFeatures.Matrix.setValue(selection_matrix)
    opFeatures.Input.connect(opUpsample.Output)

    gridGraphs = []
    graphEdges = []
    opThreshold = OpThresholdTwoLevels(graph=tempGraph)
    opThreshold.Channel.setValue(SYNAPSE_CHANNEL)
    opThreshold.SingleThreshold.setValue(0.5) #FIXME: solve the mess with uint8/float in predictions
    opThreshold.SmootherSigma.setValue({'x': 3.0, 'y': 3.0, 'z': 1.0}) #NOTE: two-level is much better. Maybe we can afford it?
    
    previous_slice_objects = None
    previous_slice_roi = None
    maxLabelSoFar = 0

    with open(output_path, "w") as fout:
        csv_writer = csv.DictWriter(fout, OUTPUT_COLUMNS, **CSV_FORMAT)
        csv_writer.writeheader()

        skeleton_branch_count = len(branchwise_rois)
        skeleton_node_count = sum( map(len, branchwise_rois) )

        node_overall_index = -1
        for branch_index, branch_rois in enumerate(branchwise_rois):
            previous_slice_objects = None
            previous_slice_roi = None
            branch_node_count = len(branch_rois)
            for node_index_in_branch, (node_info, roi) in enumerate(branch_rois):
                node_overall_index += 1
                with Timer() as timer:
                    skeletonCoord = (node_info.x_px, node_info.y_px, node_info.z_px)
                    logger.debug("skeleton point: {}".format( skeletonCoord ))
                    #Add channel dimension
                    roi_with_channel = numpy.zeros((2, roi.shape[1]+1), dtype=numpy.uint32)
                    roi_with_channel[:, :-1] = roi[:]
                    roi_with_channel[0, -1] = 0
                    roi_with_channel[1, -1] = 1
                    iz = roi[0][2]
                    roi_hessian = (roi_with_channel[0]*2, roi_with_channel[1]*2-1)
                    for x in range(roi.shape[1]):
                        if roi[0][x] == 0:
                            roi_hessian[0][x] = 0
                    roi_hessian[0][2] = iz
                    roi_hessian[1][2] = iz+1
                    #we need the second eigenvalue
                    roi_hessian[0][-1] = 1
                    roi_hessian[1][-1] = 2
                    
                    if debug_images:
                        outdir1 = outdir+"raw/"
                        try:
                            os.makedirs(outdir1)
                        except os.error:
                            pass
                        outfile = outdir1+"%.02d"%iz + ".png"
                        data = opPixelClassification3d.InputImages[-1](roi_with_channel[0], roi_with_channel[1]).wait()
                        vigra.impex.writeImage(data.squeeze().astype(numpy.uint8), outfile)
                        '''
                        outdir2 = outdir + "synapse_pred/"
                        outfile = outdir2+"%.02d"%iz + ".png"
                        data = opThreshold.InputImage(roi_with_channel[0], roi_with_channel[1]).wait()
                        vigra.impex.writeImage(data.squeeze().astype(numpy.uint8), outfile)
                        '''
                    start_pred = time.time()
                    prediction_roi = numpy.append( roi_with_channel[:,:-1], [[0],[4]], axis=1 )
                    synapse_predictions = opPixelClassification3d.PredictionProbabilities[-1](*prediction_roi).wait()
                    synapse_predictions = vigra.taggedView( synapse_predictions, "xytc" )
                    stop_pred = time.time()
                    timing_logger.debug( "spent in first 3d prediction: {}".format( stop_pred-start_pred ) )
                    opThreshold.InputImage.setValue(synapse_predictions)
                    opThreshold.InputImage.meta.drange = opPixelClassification3d.PredictionProbabilities[-1].meta.drange
                    synapse_cc = opThreshold.Output[:].wait()
                    if debug_images:
                        outdir1 = outdir+"predictions_roi/"
                        try:
                            os.makedirs(outdir1)
                        except os.error:
                            pass
                        outfile = outdir1+"%.02d"%iz + ".tiff"
                        #norm = numpy.where(synapse_cc[:, :, 0, 0]>0, 255, 0)
                        vigra.impex.writeImage(synapse_predictions[...,0,SYNAPSE_CHANNEL], outfile)
        
        
                    if debug_images:
                        outdir1 = outdir+"synapses_roi/"
                        try:
                            os.makedirs(outdir1)
                        except os.error:
                            pass
                        outfile = outdir1+"%.02d"%iz + ".tiff"
                        norm = numpy.where(synapse_cc[:, :, 0, 0]>0, 255, 0)
                        vigra.impex.writeImage(norm.astype(numpy.uint8), outfile)
                    if numpy.sum(synapse_cc)==0:
                        #print "NO SYNAPSES IN THIS SLICE:", iz
                        timing_logger.debug( "ROI TIMER: {}".format( timer.seconds() ) )
                        continue
    
                    start_hess = time.time()
                    eigenValues = opFeatures.Output(roi_hessian[0], roi_hessian[1]).wait()
                    eigenValues = numpy.abs(eigenValues[:, :, 0, 0])
                    stop_hess = time.time()
                    timing_logger.debug( "spent for hessian: {}".format( stop_hess-start_hess ) )
                    shape_x = roi[1][0]-roi[0][0]
                    shape_y =  roi[1][1]-roi[0][1]
                    shape_x = long(shape_x)
                    shape_y = long(shape_y)
                    start_gr = time.time()
                    gridGr = graphs.gridGraph((shape_x, shape_y )) # !on original pixels
                    gridGraphEdgeIndicator = graphs.edgeFeaturesFromInterpolatedImage(gridGr, eigenValues) 
                    gridGraphs.append(gridGr)
                    graphEdges.append(gridGraphEdgeIndicator)
                    stop_gr = time.time()
                    timing_logger.debug( "creating graph: {}".format( stop_gr - start_gr ) )
                    if debug_images:
                        outdir1 = outdir+"hessianUp/"
                        try:
                            os.makedirs(outdir1)
                        except os.error:
                            pass
                        outfile = outdir1+"%.02d"%iz + ".tiff"
                        logger.debug( "saving hessian to file: {}".format( outfile ) )
                        vigra.impex.writeImage(eigenValues, outfile )
                    
            
                
                    instance = vigra.graphs.ShortestPathPathDijkstra(gridGr)
                    relative_coord = [skeletonCoord[0]-roi[0][0], skeletonCoord[1]-roi[0][1]]
                    relative_coord = map(long, relative_coord)
                    sourceNode = gridGr.coordinateToNode(relative_coord)
                    start_dij = time.time()
                    instance.run(gridGraphEdgeIndicator, sourceNode, target=None)
                    
                    distances = instance.distances()
                    stop_dij = time.time()
                    timing_logger.debug( "spent in dijkstra {}".format( stop_dij - start_dij ) )
                    if debug_images:
                        outdir1 = outdir+"distances/"
                        try:
                            os.makedirs(outdir1)
                        except os.error:
                            pass
                        outfile = outdir1+"%.02d"%iz + ".tiff"
                        logger.debug( "saving distances to file:".format( outfile ) )
                        distances[skeletonCoord[0]-roi[0][0], skeletonCoord[1]-roi[0][1]] = numpy.max(distances)
                        vigra.impex.writeImage(distances, outfile )
                    
    
                    synapse_objects_4d, maxLabelCurrent = normalize_synapse_ids(synapse_cc, roi,\
                                                                                  previous_slice_objects, previous_slice_roi,\
                                                                                  maxLabelSoFar)
                    synapse_objects = synapse_objects_4d.squeeze()
    
                    synapseIds = set(synapse_objects.flat)
                    synapseIds.remove(0)
                    for sid in synapseIds:
                        #find the pixel positions of this synapse
                        syn_pixel_coords = numpy.where(synapse_objects == sid)
                        synapse_size = len( syn_pixel_coords[0] )
                        #syn_pixel_coords = numpy.unravel_index(syn_pixels, distances.shape)
                        #FIXME: offset by roi
                        syn_average_x = numpy.average(syn_pixel_coords[0])+roi[0][0]
                        syn_average_y = numpy.average(syn_pixel_coords[1])+roi[0][1]
                        syn_distances = distances[syn_pixel_coords]
                        mindist = numpy.min(syn_distances)

                        # Determine average uncertainty
                        # Get probabilities for this synapse's pixels
                        flat_predictions = synapse_predictions.view(numpy.ndarray)[synapse_objects_4d[...,0] == sid]
                        # Sort along channel axis
                        flat_predictions.sort(axis=-1)
                        # What's the difference between the highest and second-highest class?
                        certainties = flat_predictions[:,-1] - flat_predictions[:,-2]
                        avg_certainty = numpy.average(certainties)
                        avg_uncertainty = 1.0 - avg_certainty                        

                        fields = {}
                        fields["synapse_id"] = int(sid)
                        fields["x_px"] = int(syn_average_x + 0.5)
                        fields["y_px"] = int(syn_average_y + 0.5)
                        fields["z_px"] = iz
                        fields["size_px"] = synapse_size
                        fields["distance"] = mindist
                        fields["detection_uncertainty"] = avg_uncertainty
                        fields["node_id"] = node_info.id
                        fields["node_x_px"] = node_info.x_px
                        fields["node_y_px"] = node_info.y_px
                        fields["node_z_px"] = node_info.z_px

                        csv_writer.writerow( fields )                                                
                        fout.flush()

                    progress_callback( ProgressInfo( node_overall_index, 
                                                     skeleton_node_count, 
                                                     branch_index, 
                                                     skeleton_branch_count, 
                                                     node_index_in_branch, 
                                                     branch_node_count,
                                                     maxLabelCurrent ) )

                    #add this synapse to the exported list
                    previous_slice_objects = synapse_objects
                    previous_slice_roi = roi
                    maxLabelSoFar = maxLabelCurrent
            
                        
                    #Sanity check
                    #outfile = outdir+"hessianUp/"+ "%.02d"%iz + ".tiff"
                    #vigra.impex.writeImage(eigenValues, outfile)
                    #outfile = outdir+"distances/"+ "%.02d"%iz + ".tiff"
                    #vigra.impex.writeImage(distances, outfile)
                timing_logger.debug( "ROI TIMER: {}".format( timer.seconds() ) )

def normalize_synapse_ids(current_slice, current_roi, previous_slice, previous_roi, maxLabel):
    current_roi = numpy.array(current_roi)
    intersection_roi = None
    if previous_roi is not None:
        previous_roi = numpy.array(previous_roi)
        current_roi_2d = current_roi[:, :-1]
        previous_roi_2d = previous_roi[:, :-1]
        intersection_roi = getIntersection( current_roi_2d, previous_roi_2d, assertIntersect=False )

    if intersection_roi is None or previous_slice is None or abs(int(current_roi[0,2]) - int(previous_roi[0,2])) > 1:
        # We want our synapse ids to be consecutive, so we do a proper relabeling.
        # If we could guarantee that the input slice was already consecutive, we could do this:
        # relabeled_current = numpy.where( current_slice, current_slice+maxLabel, 0 )
        # ... but that's not the case.

        current_unique_labels = numpy.unique(current_slice)
        assert current_unique_labels[0] == 0, "This function assumes that not all pixels belong to detections."
        if len(current_unique_labels) == 1:
            # No objects in this slice.
            return current_slice, maxLabel
        max_current_label = current_unique_labels[-1]
        relabel = numpy.zeros( (max_current_label+1,), dtype=numpy.uint32 )
        new_max = maxLabel + len(current_unique_labels)-1
        relabel[(current_unique_labels[1:],)] = numpy.arange( maxLabel+1, new_max+1, dtype=numpy.uint32 )
        return relabel[current_slice], new_max
    
    # Extract the intersecting region from the current/prev slices,
    #  so its easy to compare corresponding pixels
    current_intersection_roi = numpy.subtract(intersection_roi, current_roi_2d[0])
    prev_intersection_roi = numpy.subtract(intersection_roi, previous_roi_2d[0])    
    current_intersection_slice = current_slice[roiToSlice(*current_intersection_roi)]
    prev_intersection_slice = previous_slice[roiToSlice(*prev_intersection_roi)]

    # omit label 0
    previous_slice_objects = numpy.unique(previous_slice)[1:]
    current_slice_objects = numpy.unique(current_slice)[1:]
    max_current_object = max(0, *current_slice_objects)
    relabel = numpy.zeros((max_current_object+1,), dtype=numpy.uint32)
    
    for cc in previous_slice_objects:
        current_labels = numpy.unique(current_intersection_slice[prev_intersection_slice==cc].flat)
        for cur_label in current_labels:
            relabel[cur_label] = cc
    
    for cur_object in current_slice_objects:
        if relabel[cur_object]==0:
            relabel[cur_object] = maxLabel+1
            maxLabel=maxLabel+1

    relabel[0] = 0

    # Relabel the entire current slice
    relabeled_slice_objects = relabel[current_slice]
    return relabeled_slice_objects, maxLabel

def main():
    # FIXME: This shouldn't be hard-coded.
    ROI_RADIUS = 150

    import argparse
    parser = argparse.ArgumentParser() 
    parser.add_argument('skeleton_file')
    parser.add_argument('project3d')
    parser.add_argument('project2d')
    parser.add_argument('volume_description')
    parser.add_argument('output_file')
    parser.add_argument('progress_port', nargs='?', default=8000)
    
    parsed_args = parser.parse_args()
    
    # Read the volume resolution
    volume_description = TiledVolume.readDescription(parsed_args.volume_description)
    z_res, y_res, x_res = volume_description.resolution_zyx
    
    # Parse the swc into a list of nodes
    skeleton_ext = os.path.splitext(parsed_args.skeleton_file)[1]
    if skeleton_ext == '.swc':
        node_infos = parse_skeleton_swc( parsed_args.skeleton_file, x_res, y_res, z_res )
    elif skeleton_ext == '.json':
        node_infos = parse_skeleton_json( parsed_args.skeleton_file, x_res, y_res, z_res )
    else:
        raise Exception("Unknown skeleton file format: " + skeleton_ext)
    
    # Construct a networkx tree
    tree = construct_tree( node_infos )
    
    # Get lists of (coord, roi) for each node, grouped into branches
    tree_nodes_and_rois = nodes_and_rois_for_tree(tree, radius=ROI_RADIUS)

    # Start a server for others to poll progress.
    progress_server = ProgressServer.create_and_start( "localhost", int(parsed_args.progress_port) )

    try:
        locate_synapses( parsed_args.project3d, 
                         parsed_args.project2d, 
                         parsed_args.volume_description, 
                         parsed_args.output_file,
                         tree_nodes_and_rois, 
                         debug_images=False, 
                         order2d='xyt', 
                         order3d='xyz',
                         progress_callback=progress_server.update_progress )
    finally:
        progress_server.shutdown()

if __name__=="__main__":
    import sys
    DEBUGGING = False
    if DEBUGGING:
        print "USING DEBUG ARGUMENTS"

#         project3dname = '/magnetic/workspace/skeleton_synapses/Synapse_Labels3D.ilp'
#         project2dname = '/magnetic/workspace/skeleton_synapses/Synapse_Labels2D.ilp'
#         skeleton_file = '/magnetic/workspace/skeleton_synapses/abd1.5_skeletons/abd1.5_skeleton_2.swc'
#         #skeleton_file = '/magnetic/workspace/skeleton_synapses/example/skeleton_18689.json'
#         volume_description = '/magnetic/workspace/skeleton_synapses/example/example_volume_description_2.json'
#         output_file = '/magnetic/workspace/skeleton_synapses/abd1.5_skeleton_2_detections.csv'

        project3dname = '/magnetic/workspace/skeleton_synapses/projects/Synapse_Labels3D.ilp'
        project2dname = '/magnetic/workspace/skeleton_synapses/projects/Synapse_Labels2D.ilp'
        skeleton_file = '/magnetic/workspace/skeleton_synapses/test_skeletons/skeleton_163751.json'
        volume_description = '/magnetic/workspace/skeleton_synapses/example/example_volume_description_2.json'
        output_file = '/magnetic/workspace/skeleton_synapses/DEBUG2.csv'

        sys.argv.append(skeleton_file)
        sys.argv.append(project3dname)
        sys.argv.append(project2dname)
        sys.argv.append(volume_description)
        sys.argv.append(output_file)

    sys.exit( main() )
