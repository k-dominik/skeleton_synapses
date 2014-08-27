import os
import numpy
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
from skeleton_synapses.swc_rois import parse_swc, construct_tree, coords_and_rois_for_tree

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

def locate_synapses(project3dname, project2dname, input_filepath, output_path, branchwise_rois, debug_images=False, order2d='xyz', order3d='xyt'):
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
    
    previous_slice_objects = None
    previous_slice_roi = None
    maxLabelSoFar = 0

    with open(output_path, "w") as fout:
        for branch_rois in branchwise_rois:
            previous_slice_objects = None
            previous_slice_roi = None
            for skeletonCoord, roi in branch_rois:
                with Timer() as timer:
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
                    
    
                    synapse_objects, maxLabelCurrent = normalize_synapse_ids(synapse_cc, roi,\
                                                                                  previous_slice_objects, previous_slice_roi,\
                                                                                  maxLabelSoFar)
                    synapse_objects = synapse_objects.squeeze()
    
                    synapseIds = set(synapse_objects.flat)
                    synapseIds.remove(0)
                    for sid in synapseIds:
                        #find the pixel positions of this synapse
                        syn_pixel_coords = numpy.where(synapse_objects ==sid)
                        #syn_pixel_coords = numpy.unravel_index(syn_pixels, distances.shape)
                        #FIXME: offset by roi
                        syn_average_x = numpy.average(syn_pixel_coords[0])+roi[0][0]
                        syn_average_y = numpy.average(syn_pixel_coords[1])+roi[0][1]
                        syn_distances = distances[syn_pixel_coords]
                        mindist = numpy.min(syn_distances)
                        str_to_write = str(int(sid)) + "\t" + str(int(syn_average_x)) + "\t" + str(int(syn_average_y)) + \
                                       "\t" + str(iz) + "\t" + str(mindist)+"\n"
                        fout.write(str_to_write)
                        fout.flush()
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

    if intersection_roi is None or previous_slice is None or current_roi[0,2]!=previous_roi[0,2]+1:
        # Relabel from max
        relabeled_current = numpy.where( current_slice, current_slice+maxLabel, 0 )
        return relabeled_current, numpy.max(relabeled_current)
    
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

def merge_synapse_ids(fin, fout):
    all_synapses = {}
    with open(fin) as f:
        for line in f:
            #NOTE: this part is fully dependent on the order, in which synapses are written
            #Be careful, if it changes
            parts = line.split('\t')
            syn_id = parts[0]
            syn_values = [int(x) for x in parts[1:-1]] #coordinates are int
            syn_values.append(float(parts[-1])) #distance is float
            if all_synapses.has_key(syn_id):
                all_synapses[syn_id].extend(syn_values)
            else:
                all_synapses[syn_id] = syn_values
    
    print len(all_synapses)
    with open(fout, "w") as f2:        
        for syn_id, syn_coords in all_synapses.iteritems():
            syn_coord_array = numpy.asarray(syn_coords)
            if len(syn_coord_array.shape)>1:
                syn_coord_array = numpy.average(syn_coord_array, 0)
            str_to_write = syn_id + "\t" + str(syn_coord_array[0]) + "\t" +str(syn_coord_array[1]) + "\t"+str(syn_coord_array[2]) + "\t"+\
                            str(syn_coord_array[3]) + "\n"
            f2.write(str_to_write)
            f2.flush()
        
        
    
    
def main():
    # FIXME: These shouldn't be hard-coded.
    ROI_RADIUS = 150

    import argparse
    parser = argparse.ArgumentParser() 
    parser.add_argument('skeleton_swc')
    parser.add_argument('project3d')
    parser.add_argument('project2d')
    parser.add_argument('volume_description')
    parser.add_argument('output_file')
    
    parsed_args = parser.parse_args()
    
    # Read the volume resolution
    volume_description = TiledVolume.readDescription(parsed_args.volume_description)
    z_res, y_res, x_res = volume_description.resolution_zyx
    
    # Parse the swc into a list of nodes
    node_infos = parse_swc( parsed_args.skeleton_swc, x_res, y_res, z_res )
    
    # Construct a networkx tree
    tree = construct_tree( node_infos )
    
    # Get lists of (coord, roi) for each node, grouped into branches
    tree_coords_and_rois = coords_and_rois_for_tree(tree, radius=ROI_RADIUS)

    locate_synapses( parsed_args.project3d, 
                     parsed_args.project2d, 
                     parsed_args.volume_description, 
                     parsed_args.output_file,
                     tree_coords_and_rois, 
                     debug_images=False, 
                     order2d='xyt', 
                     order3d='xyz' )

if __name__=="__main__":
    import sys
    DEBUGGING = False
    POSTPROCESS = True
    if DEBUGGING:
        project3dname = '/Users/bergs/Desktop/forStuart/Synapse_Labels3D.ilp'
        project2dname = '/Users/bergs/Desktop/forStuart/Synapse_Labels2D.ilp'
        skeleton_swc = '/Users/bergs/Documents/workspace/skeleton_synapses/example/example_skeleton.swc'
        volume_description = '/Users/bergs/Documents/workspace/skeleton_synapses/example/example_volume_description_1.json'
        output_file = '/Users/bergs/Documents/workspace/skeleton_synapses/synapses.csv'

        sys.argv.append(skeleton_swc)
        sys.argv.append(project3dname)
        sys.argv.append(project2dname)
        sys.argv.append(volume_description)
        sys.argv.append(output_file)
    if POSTPROCESS:
        fin = '/home/akreshuk/data/abd1.5_output_synapse_1.csv'
        fout = '/home/akreshuk/data/abd1.5_output_synapse_1_pp.csv'
        merge_synapse_ids(fin, fout)
        sys.exit()

    sys.exit( main() )
