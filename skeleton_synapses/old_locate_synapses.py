import os
import collections
import numpy
import csv
from functools import partial
import vigra
from vigra import graphs
import time
import scipy

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
from lazyflow.request import Request, RequestPool, RequestLock

from skeleton_synapses.opCombinePredictions import OpCombinePredictions
from skeleton_synapses.opNodewiseCache import OpNodewiseCache
from skeleton_synapses.opUpsampleByTwo import OpUpsampleByTwo
from skeleton_synapses.skeleton_utils import parse_skeleton_swc, parse_skeleton_json, \
                                             construct_tree, nodes_and_rois_for_tree, \
                                             parse_connectors
                                             
from skeleton_synapses.progress_server import ProgressInfo, ProgressServer
from skeleton_utils import CSV_FORMAT

THRESHOLD = 5
MEMBRANE_CHANNEL = 0
SYNAPSE_CHANNEL = 2

X_RES = 0
Y_RES = 0
Z_RES = 0

import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

import tempfile
TMP_DIR = tempfile.gettempdir()
import logging

import lazyflow.request
lazyflow.request.Request.reset_thread_pool(0)

# Import requests in advance so we can silence its log messages.
import requests
logging.getLogger("requests").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

timing_logger = logging.getLogger(__name__ + '.timing')
timing_logger.setLevel(logging.INFO)

OUTPUT_COLUMNS = ["synapse_id", "x_px", "y_px", "z_px", "size_px", "distance_hessian", "distance_raw_probs", "detection_uncertainty", "node_id", \
                  "node_x_px", "node_y_px", "node_z_px", "nearest_connector_id", "nearest_connector_distance_nm", \
                  "nearest_connector_x_nm", "nearest_connector_y_nm", "nearest_connector_z_nm"]


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


MEMBRANE_CACHE_FORMAT = "/groups/flyem/home/bergs/workspace/skeleton_synapses/debuggingmembrane/{z_px}-{node_id}.png"
SYNAPSE_CACHE_FORMAT = "/groups/flyem/home/bergs/workspace/skeleton_synapses/debuggingpredictions_roi/{z_px}-{node_id}.tiff"

def roi_to_path( skeleton_id, node_coords_to_ids, path_format, axiskeys, start, stop ):
    """
    path_format: A format string with any of the following format parameters:
                 x y z node_id skeleton_id
    """
    assert len(axiskeys) == len(start) == len(stop)
    start = numpy.asarray(start)
    stop = numpy.asarray(stop)
    roi_center = (stop + start) / 2
    
    tagged_coords = collections.OrderedDict( zip(axiskeys, roi_center) )
    x_px, y_px = tagged_coords['x'], tagged_coords['y']
    
    # Must have either t or z
    try:
        z_px = tagged_coords['z']
        assert 't' not in tagged_coords
    except KeyError:
        z_px = tagged_coords['t']        
    
    try:
        node_id = node_coords_to_ids[(x_px,y_px,z_px)]
    except KeyError:
        return None
    
    format_keys = { "skeleton_id" : skeleton_id,
                    "node_id" : node_id,
                    "x_px" : x_px,
                    "y_px" : y_px,
                    "z_px" : z_px }
    
    tile_path = path_format.format( **format_keys )
    
    if os.path.exists( tile_path ):
        return tile_path
    return None

def locate_synapses(project3dname, 
                    project2dname, 
                    input_filepath, 
                    output_path, 
                    branchwise_rois, 
                    node_to_connector,
                    connector_infos,
                    debug_images=False, 
                    order2d='xyz', 
                    order3d='xyt', 
                    progress_callback=lambda p: None,
                    node_infos=None):
    outdir = os.path.split( output_path )[0]
    
    shell3d = open_project(project3dname)
    shell2d = open_project(project2dname)

    opPixelClassification3d = append_lane(shell3d.workflow, input_filepath, order3d) # Z
    logger.debug( "appended 3d lane" )
    opPixelClassification2d = append_lane(shell2d.workflow, input_filepath, order2d) # T
    logger.debug( "appended 2d lane" )

    assert node_infos
    node_coords_to_ids = { (n.x_px, n.y_px, n.z_px) : n.id for n in node_infos }
    
    tempGraph = Graph()
    opMembranePredictionCache = OpNodewiseCache( graph=tempGraph )
    opMembranePredictionCache.ComputedInput.connect( opPixelClassification2d.HeadlessPredictionProbabilities[-1], permit_distant_connection=True )
    opMembranePredictionCache.RoiToPathFn.setValue( partial( roi_to_path, "", node_coords_to_ids, MEMBRANE_CACHE_FORMAT ) )
    opMembranePredictionCache.TransformFn.setValue( lambda a: numpy.asarray(a, dtype=numpy.float32) / 255.0 )

    opSynapsePredictionCache = OpNodewiseCache( graph=tempGraph )
    opSynapsePredictionCache.ComputedInput.connect( opPixelClassification3d.PredictionProbabilities[-1], permit_distant_connection=True )
    opSynapsePredictionCache.RoiToPathFn.setValue( partial( roi_to_path, "", node_coords_to_ids, SYNAPSE_CACHE_FORMAT ) )
    #opSynapsePredictionCache.TransformFn.setValue( lambda a: numpy.asarray(a, dtype=numpy.float32) / 255.0 )
    
    # Combine
    opCombinePredictions = OpCombinePredictions(SYNAPSE_CHANNEL, MEMBRANE_CHANNEL, graph=tempGraph)
    opPixelClassification3d.FreezePredictions.setValue(False)
    opCombinePredictions.SynapsePredictions.connect( opSynapsePredictionCache.Output )
    opCombinePredictions.MembranePredictions.connect( opMembranePredictionCache.Output )

    #data_shape_3d = input_data.shape[0:3]    
    opUpsample = OpUpsampleByTwo(graph = tempGraph)
    opUpsample.Input.connect(opCombinePredictions.Output)
    
    maxLabelSoFar = [0]

    with open(output_path, "w") as fout:
        csv_writer = csv.DictWriter(fout, OUTPUT_COLUMNS, **CSV_FORMAT)
        csv_writer.writeheader()

        skeleton_branch_count = len(branchwise_rois)
        skeleton_node_count = sum( map(len, branchwise_rois) )

        node_overall_index = [-1]
        f_out_lock = RequestLock()
        max_label_lock = RequestLock()
        
        def process_branch( branch_index, branch_rois ):
            # opFeatures and opThreshold are declared locally so this whole block can be parallelized!
            # (We use Input.setValue() instead of Input.connect() here.)
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
        
            # opFeatures and opThreshold are declared locally so this whole block can be parallelized!
            # (We use Input.setValue() instead of Input.connect() here.)
            opThreshold = OpThresholdTwoLevels(graph=tempGraph)
            opThreshold.Channel.setValue(0) # We select SYNAPSE_CHANNEL before the data is given to opThreshold
            opThreshold.SmootherSigma.setValue({'x': 2.0, 'y': 2.0, 'z': 1.0}) #NOTE: two-level is much better. Maybe we can afford it?

            #opThreshold.CurOperator.setValue(0) # 0==one-level
            #opThreshold.SingleThreshold.setValue(0.4) #FIXME: solve the mess with uint8/float in predictions

            opThreshold.CurOperator.setValue(1) # 1==two-level
            opThreshold.HighThreshold.setValue(0.4)
            opThreshold.LowThreshold.setValue(0.2)
            
            previous_slice_objects = None
            previous_slice_roi = None
            conn_ids = [x.id for x in connector_infos]
            connector_infos_dict = dict(zip(conn_ids, connector_infos))

            branch_node_count = len(branch_rois)
            for node_index_in_branch, (node_info, roi) in enumerate(branch_rois):
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
                    
                    WITH_CONNECTORS_ONLY = True
                    if WITH_CONNECTORS_ONLY:
                        if not node_info.id in node_to_connector.keys():
                            continue
                    
                    if debug_images:
                        outdir1 = outdir+"raw/"
                        try:
                            os.makedirs(outdir1)
                        except os.error:
                            pass
                        outfile = outdir1+"/{}-{}".format( iz, node_info.id ) + ".png"
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
                    synapse_prediction_roi = numpy.append( prediction_roi[:,:-1], [[SYNAPSE_CHANNEL],[SYNAPSE_CHANNEL+1]], axis=1 )
                    membrane_prediction_roi = numpy.append( prediction_roi[:,:-1], [[MEMBRANE_CHANNEL],[MEMBRANE_CHANNEL+1]], axis=1 )
                    
                    #synapse_predictions = opPixelClassification3d.PredictionProbabilities[-1](*prediction_roi).wait()                    
                    synapse_predictions = opSynapsePredictionCache.Output(*synapse_prediction_roi).wait()
                    synapse_predictions = vigra.taggedView( synapse_predictions, "xytc" )

                    if debug_images:
                        outdir1 = outdir+"membrane/"
                        try:
                            os.makedirs(outdir1)
                        except os.error:
                            pass
                        outfile = outdir1+"/{}-{}".format( iz, node_info.id ) + ".png"
                        #membrane_predictions = opPixelClassification2d.HeadlessPredictionProbabilities[-1](*prediction_roi).wait()
                        membrane_predictions = opMembranePredictionCache.Output(*membrane_prediction_roi).wait()
                        vigra.impex.writeImage(membrane_predictions[..., 0].squeeze(), outfile)
                    
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
                        outfile = outdir1+"/{}-{}".format( iz, node_info.id ) + ".tiff"
                        #norm = numpy.where(synapse_cc[:, :, 0, 0]>0, 255, 0)
                        vigra.impex.writeImage(synapse_predictions[...,0,0], outfile)
        
                    if debug_images:
                        outdir1 = outdir+"synapses_roi/"
                        try:
                            os.makedirs(outdir1)
                        except os.error:
                            pass
                        outfile = outdir1+"/{}-{}".format( iz, node_info.id ) + ".tiff"
                        norm = numpy.where(synapse_cc[:, :, 0, 0]>0, 255, 0)
                        vigra.impex.writeImage(norm.astype(numpy.uint8), outfile)
                    
                    if numpy.sum(synapse_cc)==0:
                        print "NO SYNAPSES IN THIS SLICE:", iz
                        timing_logger.debug( "ROI TIMER: {}".format( timer.seconds() ) )
                        continue
                    
                    # Distances over Hessian
                    start_hess = time.time()
                    roi_hessian = ( tuple(map(long, roi_hessian[0])), tuple(map(long, roi_hessian[1])) )
                    upsampled_combined_membranes = opUpsample.Output(*roi_hessian).wait()
                    upsampled_combined_membranes = vigra.taggedView(upsampled_combined_membranes, opUpsample.Output.meta.axistags )
                    opFeatures.Input.setValue(upsampled_combined_membranes)
                    eigenValues = opFeatures.Output[...,1:2].wait() #we need the second eigenvalue
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
                    #gridGraphs.append(gridGr)
                    #graphEdges.append(gridGraphEdgeIndicator)
                    stop_gr = time.time()
                    timing_logger.debug( "creating graph: {}".format( stop_gr - start_gr ) )
                    if debug_images:
                        outdir1 = outdir+"hessianUp/"
                        try:
                            os.makedirs(outdir1)
                        except os.error:
                            pass
                        outfile = outdir1+"/{}-{}".format( iz, node_info.id ) + ".tiff"
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
                        outfile = outdir1+"/{}-{}".format( iz, node_info.id ) + ".tiff"
                        logger.debug( "saving distances to file:".format( outfile ) )
                        # Create a "white" pixel at the source node
                        distances[skeletonCoord[0]-roi[0][0], skeletonCoord[1]-roi[0][1]] = numpy.max(distances)
                        vigra.impex.writeImage(distances, outfile )

                    # Distances over raw membrane probabilities
                    roi_upsampled_membrane = numpy.asarray( roi_hessian )
                    roi_upsampled_membrane[:, -1] = [0,1]
                    roi_upsampled_membrane = (map(long, roi_upsampled_membrane[0]), map(long, roi_upsampled_membrane[1]))
                    connector_distances = None
                    connector_coords = None
                    if node_info.id in node_to_connector.keys():
                        connectors = node_to_connector[node_info.id]
                        connector_info = connector_infos_dict[connectors[0]]
                        #Convert to pixels
                        con_x_px = int(connector_info.x_nm / float(X_RES))
                        con_y_px = int(connector_info.y_nm / float(Y_RES))
                        con_z_px = int(connector_info.z_nm / float(Z_RES))
                        connector_coords = (con_x_px-roi[0][0], con_y_px-roi[0][1])
                        if con_x_px>roi[0][0] and con_x_px<roi[1][0] and con_y_px>roi[0][1] and con_y_px<roi[1][1]:
                            #this connector is inside our prediction roi, compute the distance field                                                                                                        "
                            con_relative = [long(con_x_px-roi[0][0]), long(con_y_px-roi[0][1])]
    
                            sourceNode = gridGr.coordinateToNode(con_relative)
                            instance.run(gridGraphEdgeIndicator, sourceNode, target=None)
                            connector_distances = instance.distances()
                        else:
                            connector_distances = None
                    
                    upsampled_membrane_probabilities = opUpsample.Output(*roi_upsampled_membrane).wait().squeeze()
                    upsampled_membrane_probabilities = vigra.filters.gaussianSmoothing(upsampled_membrane_probabilities, sigma=1.0)
                    #print "UPSAMPLED MEMBRANE SHAPE: {} MAX: {} MIN: {}".format( upsampled_membrane_probabilities.shape, upsampled_membrane_probabilities.max(), upsampled_membrane_probabilities.min() )
                    gridGrRaw = graphs.gridGraph((shape_x, shape_y )) # !on original pixels
                    gridGraphRawEdgeIndicator = graphs.edgeFeaturesFromInterpolatedImage(gridGrRaw, upsampled_membrane_probabilities) 
                    #gridGraphs.append(gridGrRaw)
                    #graphEdges.append(gridGraphRawEdgeIndicator)
                    instance_raw = vigra.graphs.ShortestPathPathDijkstra(gridGrRaw)
                    sourceNode = gridGrRaw.coordinateToNode(relative_coord)
                    instance_raw.run(gridGraphRawEdgeIndicator, sourceNode, target=None)
                    distances_raw = instance_raw.distances()

                    stop_dij = time.time()
                    timing_logger.debug( "spent in dijkstra (raw probs) {}".format( stop_dij - start_dij ) )
                    if debug_images:
                        outdir1 = outdir+"distances_raw/"
                        try:
                            os.makedirs(outdir1)
                        except os.error:
                            pass
                        outfile = outdir1+"/{}-{}".format( iz, node_info.id ) + ".tiff"
                        logger.debug( "saving distances (raw probs) to file:".format( outfile ) )
                        # Create a "white" pixel at the source node
                        distances_raw[skeletonCoord[0]-roi[0][0], skeletonCoord[1]-roi[0][1]] = numpy.max(distances_raw)
                        vigra.impex.writeImage(distances_raw, outfile )
    
                    if numpy.sum(synapse_cc)==0:
                        continue
                    

                    with max_label_lock:                    
                        synapse_objects_4d, maxLabelCurrent = normalize_synapse_ids( synapse_cc, 
                                                                                     roi,
                                                                                     previous_slice_objects, 
                                                                                     previous_slice_roi,
                                                                                     maxLabelSoFar[0] )
    
                        maxLabelSoFar[0] = maxLabelCurrent
                        synapse_objects = synapse_objects_4d.squeeze()

                        #add this synapse to the exported list
                        previous_slice_objects = synapse_objects
                        previous_slice_roi = roi
                    '''
                    if numpy.sum(synapse_cc)==0:
                        print "NO SYNAPSES IN THIS SLICE:", iz
                        timing_logger.debug( "ROI TIMER: {}".format( timer.seconds() ) )
                        continue
                    '''
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
                        
                        syn_distances_raw = distances_raw[syn_pixel_coords]
                        mindist_raw = numpy.min(syn_distances_raw)

                        if connector_distances is not None:
                            syn_distances_connector = connector_distances[syn_pixel_coords]
                            min_conn_distance = numpy.min(syn_distances_connector)
                            
                        elif connector_coords is not None:
                            euclidean_dists = [scipy.spatial.distance.euclidean(connector_coords, xy) for xy in zip(syn_pixel_coords[0], syn_pixel_coords[1])]
                            min_conn_distance = numpy.min(euclidean_dists)
                        else:
                            min_conn_distance = 99999.0
                            
                        # Determine average uncertainty
                        # Get probabilities for this synapse's pixels
                        flat_predictions = synapse_predictions.view(numpy.ndarray)[synapse_objects_4d[...,0] == sid]

                        # If we pulled the data from cache, there may be only one channel.
                        # In that case, we can't quite compute a proper uncertainty, 
                        #  so we'll just pretend there were only two prediction channels to begin with.
                        if flat_predictions.shape[-1] > 1:
                            # Sort along channel axis
                            flat_predictions.sort(axis=-1)
                            # What's the difference between the highest and second-highest class?
                            certainties = flat_predictions[:,-1] - flat_predictions[:,-2]
                        else:
                            # Pretend there were only two channels
                            certainties = flat_predictions[:,0] - (1 - flat_predictions[:,0])
                        avg_certainty = numpy.average(certainties)
                        avg_uncertainty = 1.0 - avg_certainty                        

                        fields = {}
                        fields["synapse_id"] = int(sid)
                        fields["x_px"] = int(syn_average_x + 0.5)
                        fields["y_px"] = int(syn_average_y + 0.5)
                        fields["z_px"] = iz
                        fields["size_px"] = synapse_size
                        fields["distance_hessian"] = mindist
                        fields["distance_raw_probs"] = mindist_raw
                        fields["detection_uncertainty"] = avg_uncertainty
                        fields["node_id"] = node_info.id
                        fields["node_x_px"] = node_info.x_px
                        fields["node_y_px"] = node_info.y_px
                        fields["node_z_px"] = node_info.z_px
                        if min_conn_distance!=99999.0:
                            connectors = node_to_connector[node_info.id]
                            connector_info = connector_infos_dict[connectors[0]]
                            fields["nearest_connector_id"] = connector_info.id
                            fields["nearest_connector_distance_nm"] = min_conn_distance
                            fields["nearest_connector_x_nm"] = connector_info.x_nm
                            fields["nearest_connector_y_nm"] = connector_info.y_nm
                            fields["nearest_connector_z_nm"] = connector_info.z_nm
                        else:
                            fields["nearest_connector_id"] = -1
                            fields["nearest_connector_distance_nm"] = min_conn_distance
                            fields["nearest_connector_x_nm"] = -1
                            fields["nearest_connector_y_nm"] = -1
                            fields["nearest_connector_z_nm"] = -1
                                        
                        

                        with f_out_lock:
                            csv_writer.writerow( fields )                                                
                            fout.flush()

                    with f_out_lock:
                        node_overall_index[0] += 1
                        progress_callback( ProgressInfo( node_overall_index[0], 
                                                         skeleton_node_count, 
                                                         branch_index, 
                                                         skeleton_branch_count, 
                                                         node_index_in_branch, 
                                                         branch_node_count,
                                                         maxLabelCurrent ) )
            
                        
                    #Sanity check
                    #outfile = outdir+"hessianUp/"+ "%.02d"%iz + ".tiff"
                    #vigra.impex.writeImage(eigenValues, outfile)
                    #outfile = outdir+"distances/"+ "%.02d"%iz + ".tiff"
                    #vigra.impex.writeImage(distances, outfile)
                timing_logger.debug( "ROI TIMER: {}".format( timer.seconds() ) )

        request_pool = RequestPool()
        for branch_index, branch_rois in enumerate(branchwise_rois):
            request_pool.add( Request( partial( process_branch, branch_index, branch_rois ) ) )
        
        request_pool.wait()

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
    global Z_RES
    Z_RES = z_res
    global Y_RES 
    Y_RES = y_res
    global X_RES 
    X_RES = x_res
    
    # Parse the swc into a list of nodes
    node_to_connector = None
    skeleton_ext = os.path.splitext(parsed_args.skeleton_file)[1]
    if skeleton_ext == '.swc':
        node_infos = parse_skeleton_swc( parsed_args.skeleton_file, x_res, y_res, z_res )
    elif skeleton_ext == '.json':
        node_infos = parse_skeleton_json( parsed_args.skeleton_file, x_res, y_res, z_res )
        connector_infos, node_to_connector = parse_connectors(parsed_args.skeleton_file)
    else:
        raise Exception("Unknown skeleton file format: " + skeleton_ext)
    
    # Construct a networkx tree
    tree = construct_tree( node_infos )
    
    # Get lists of (coord, roi) for each node, grouped into branches
    tree_nodes_and_rois = nodes_and_rois_for_tree(tree, radius=ROI_RADIUS)

    SPECIAL_DEBUG = False
    if SPECIAL_DEBUG:
        nodes_of_interest = [26717, 29219, 28228, 91037, 33173, 31519, 92443, 28010, 91064, 28129, 226935, 90886, 91047, 91063, 94379, 33997, 28626, 36989, 39556, 33870, 91058, 35882, 28260, 36252, 90399, 36892, 21248, 92841, 94203, 29465, 91967, 27937, 28227, 35717, 38656, 19764, 32398, 91026, 90350]
        #nodes_of_interest = [37575]
        nodes_of_interest = set(nodes_of_interest)
        new_tree_nodes_and_rois = []
        for branch_coords_and_rois in tree_nodes_and_rois:
            new_branch = []
            for node_info, roi_around_point in branch_coords_and_rois:
                if node_info.id in nodes_of_interest :
                    new_branch.append( (node_info, roi_around_point) )
            if new_branch:
                new_tree_nodes_and_rois.append( new_branch )
        tree_nodes_and_rois = new_tree_nodes_and_rois

    # Start a server for others to poll progress.
    #progress_server = ProgressServer.create_and_start( "localhost", int(parsed_args.progress_port) )

    try:
        locate_synapses( parsed_args.project3d, 
                         parsed_args.project2d, 
                         parsed_args.volume_description, 
                         parsed_args.output_file,
                         tree_nodes_and_rois, 
                         node_to_connector,
                         connector_infos,
                         debug_images=False,
                         order2d='xyt', 
                         order3d='xyz',
                         #progress_callback=progress_server.update_progress,
                         node_infos=node_infos )
    except:
        raise
    else:
        pass
        #progress_server.shutdown()

if __name__=="__main__":
    import sys
    DEBUGGING = True
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
        skeleton_file = '/magnetic/workspace/skeleton_synapses/test_skeletons/skeleton_18689.json'
        volume_description = '/magnetic/workspace/skeleton_synapses/example/example_volume_description_2.json'
        output_file = '/magnetic/workspace/skeleton_synapses/debugging/connectors_only.csv'
        #output_file = '/magnetic/workspace/skeleton_synapses/cachecheck/raw_output.csv'

        #project3dname = '/home/anna/data/albert/Johannes/for_Janelia/Synapse_Labels3D.ilp'
        #project2dname = '/home/anna/data/albert/Johannes/for_Janelia/Synapse_Labels2D.ilp'
        #skeleton_file = '/home/anna/catmaid_tools/test_skeletons/skeleton_18689.json'
        #volume_description = '/home/anna/catmaid_tools/example/example_volume_description_2.json'
        #output_file = '/tmp/synapses.csv'

        sys.argv.append(skeleton_file)
        sys.argv.append(project3dname)
        sys.argv.append(project2dname)
        sys.argv.append(volume_description)
        sys.argv.append(output_file)

    sys.exit( main() )
