import os
import sys
import numpy
import vigra
from vigra import graphs
import glob
import time

import ilastik_main
from ilastik.workflows.pixelClassification import PixelClassificationWorkflow
from ilastik.applets.dataSelection import DataSelectionApplet
from ilastik.applets.thresholdTwoLevels import OpThresholdTwoLevels
from ilastik.applets.dataSelection.opDataSelection import DatasetInfo 
from lazyflow.operators.vigraOperators import OpPixelFeaturesPresmoothed
from lazyflow.graph import Graph
from lazyflow.utility import PathComponents, isUrl, Timer

from lazyflow.graph import Operator, InputSlot, OutputSlot
from lazyflow.roi import roiToSlice, sliceToRoi, getIntersection

#project3dname = '/home/anna/data/albert/Johannes/for_Janelia/Synapse_Labels3D.ilp'
#project2dname = '/home/anna/data/albert/Johannes/for_Janelia/Synapse_Labels2D.ilp'
#input_dir = '/home/anna/data/connector_archive_2g0y0b/14894406/presynaptic/16592557/'
input_dir = '/home/anna/data/tmp/'

#project3dname = '/Users/bergs/Desktop/forStuart/Synapse_Labels3D.ilp'
#project2dname = '/Users/bergs/Desktop/forStuart/Synapse_Labels2D.ilp'

#project3dname = '/groups/flyem/home/kreshuka/workspace/scripts/fruitfly/Synapse_Labels3D.ilp'
#project2dname = '/groups/flyem/home/kreshuka/workspace/scripts/fruitfly/Synapse_Labels2D.ilp'

#project3dname = '/home/bergs/workspace/anna_scripts/fruitfly/Synapse_Labels3D.ilp'
#project2dname = '/home/bergs/workspace/anna_scripts/fruitfly/Synapse_Labels2D.ilp'

project3dname = '/home/akreshuk/scripts/fruitfly/Synapse_Labels3D.ilp'
project2dname = '/home/akreshuk/scripts/fruitfly/Synapse_Labels2D.ilp'


#input_dir = '/home/anna/data/connector_archive_2g0y0b/14894406/presynaptic/16592557/'
#outdir = "/home/anna/data/tmp/"
outdir = "/tmp/"

THRESHOLD = 5
MEMBRANE_CHANNEL = 0
SYNAPSE_CHANNEL = 2

import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

import tempfile
TMP_DIR = tempfile.gettempdir()
import logging
import requests
logging.getLogger("requests").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

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

def do_stuff(project3dname, project2dname, input_filepath, outdir, branchwise_rois, debug_images=False, order2d='xyz', order3d='xyt'):
    #input_data = volume_from_dir(input_path)
    #print "input data shape:", input_data.shape
    shell3d = open_project(project3dname)
    shell2d = open_project(project2dname)

    #opPixelClassification3d = configure_batch_pipeline(shell3d.workflow, input_data, 'xyzc') # Z
    #opPixelClassification2d = configure_batch_pipeline(shell2d.workflow, input_data, 'xytc') # T
    
    opPixelClassification3d = append_lane(shell3d.workflow, input_filepath, order3d) # Z
    logger.debug( "appended 3d lane" )
    opPixelClassification2d = append_lane(shell2d.workflow, input_filepath, order2d) # T
    logger.debug( "appended 2d lane" )
    
    # Combine
    tempGraph = Graph()
    opCombinePredictions = OpCombinePredictions(graph=tempGraph)
    opPixelClassification3d.FreezePredictions.setValue(False)
    opCombinePredictions.SynapsePredictions.connect( opPixelClassification3d.PredictionProbabilities[-1], permit_distant_connection=True )
    opCombinePredictions.MembranePredictions.connect( opPixelClassification2d.HeadlessPredictionProbabilities[-1], permit_distant_connection=True )

    data_shape_3d = opPixelClassification3d.InputImages[-1].meta.shape[0:3]
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
    fout = open(outdir+"synapses.csv", "w")
    opThreshold = OpThresholdTwoLevels(graph=tempGraph)
    opThreshold.Channel.setValue(SYNAPSE_CHANNEL)
    #opThreshold.InputImage.connect(opPixelClassification3d.HeadlessUint8PredictionProbabilities[-1], permit_distant_connection=True )
    opThreshold.SingleThreshold.setValue(0.5) #FIXME: solve the mess with uint8/float in predictions
    
    previous_slice_objects = None
    previous_slice_roi = None
    maxLabelSoFar = 0
    

    for branch_rois in branchwise_rois:
        previous_slice_objects = None
        previous_slice_roi = None
        for skeletonCoord, roi in branch_rois:
            with Timer() as timer:
                logging.debug("skeleton point: {}".format( skeletonCoord ))
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
                start_pred = time.clock()
                prediction_roi = numpy.append( roi_with_channel[:,:-1], [[0],[4]], axis=1 )
                synapse_predictions = opPixelClassification3d.PredictionProbabilities[-1](*prediction_roi).wait()
                synapse_predictions = vigra.taggedView( synapse_predictions, "xytc" )
                stop_pred = time.clock()
                logger.debug( "spent in first 3d prediction: {}".format( stop_pred-start_pred ) )
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
                    logger.debug( "ROI TIMER: {}".format( timer.seconds() ) )
                    continue
                
                
                
                #roi_list = [slice(None, None, None), slice(None, None, None), slice(iz, iz+1, None), slice(1, 2, None)]
                #roi = sliceToRoi(roi_list, data_shape_3d+(2,))
                start_hess = time.clock()
                eigenValues = opFeatures.Output(roi_hessian[0], roi_hessian[1]).wait()
                eigenValues = numpy.abs(eigenValues[:, :, 0, 0])
                stop_hess = time.clock()
                logger.debug( "spent for hessian: {}".format( stop_hess-start_hess ) )
                shape_x = roi[1][0]-roi[0][0]
                shape_y =  roi[1][1]-roi[0][1]
                shape_x = long(shape_x)
                shape_y = long(shape_y)
                start_gr = time.clock()
                gridGr = graphs.gridGraph((shape_x, shape_y )) # !on original pixels
                gridGraphEdgeIndicator = graphs.edgeFeaturesFromInterpolatedImage(gridGr, eigenValues) 
                gridGraphs.append(gridGr)
                graphEdges.append(gridGraphEdgeIndicator)
                stop_gr = time.clock()
                logger.debug( "creating graph: {}".format( stop_gr - start_gr ) )
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
                start_dij = time.clock()
                instance.run(gridGraphEdgeIndicator, sourceNode, target=None)
                
                distances = instance.distances()
                stop_dij = time.clock()
                logger.debug( "spent in dijkstra {}".format( stop_dij - start_dij ) )
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
                

                synapse_objects, maxLabelCurrent = find_synapses_consistently(synapse_cc, roi,\
                                                                              previous_slice_objects, previous_slice_roi,\
                                                                              maxLabelSoFar)
                synapse_objects = synapse_objects.squeeze()
        
                    
                #synapse_objects_slice = synapse_objects[:, :, iz]
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
                    #add this synapse to the exported list
                previous_slice_objects = synapse_objects
                previous_slice_roi = roi
                maxLabelSoFar = maxLabelCurrent
        
                    
                #Sanity check
                #outfile = outdir+"hessianUp/"+ "%.02d"%iz + ".tiff"
                #vigra.impex.writeImage(eigenValues, outfile)
                #outfile = outdir+"distances/"+ "%.02d"%iz + ".tiff"
                #vigra.impex.writeImage(distances, outfile)
            logger.debug( "ROI TIMER: {}".format( timer.seconds() ) )

def find_synapses_consistently(current_slice, current_roi, previous_slice, previous_roi, maxLabel):
    # Drop Z
    intersection_roi = None
    if previous_roi is not None:
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
    max_current_object = max(current_slice_objects)
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

def test_find_synapses():
    slice1 = numpy.zeros((20, 20), dtype=numpy.uint8)
    slice2 = numpy.zeros((20, 20), dtype=numpy.uint8)
    
    slice1[0:3, 0:3] = 1
    slice1[7:9, 2:3] = 3

    slice2[2:5, 2:5] = 2
    slice2[1:3, 7:9] = 5

    roi1 = [(0,0), (10,10)]
    roi2 = [(2,2), (12,12)]    
    
    extracted_slice1 = slice1[roiToSlice(*roi1)]
    extracted_slice2 = slice2[roiToSlice(*roi2)]
    
    result1, maxLabel = find_synapses_consistently(extracted_slice1, roi1, None, None, 0)
    assert numpy.all(result1==extracted_slice1)
    assert maxLabel==3
    
    result2, maxLabel2 = find_synapses_consistently(extracted_slice2, roi2, extracted_slice1, roi1, maxLabel)
    
    # Copy into the original (big) array for straightforward comparison
    slice2[roiToSlice(*roi2)] = result2

    # Note the modified slicings for comparison: 
    #  we don't care what happened outside the intersection region.
    assert numpy.all(slice2[2:5, 2:5]==1)
    assert numpy.all(slice2[2:3, 7:9]==maxLabel+1)
    assert maxLabel2==4
    
import h5py

def volume_from_dir(dirpattern, offset=0, nfiles=None):
    filelist = glob.glob(dirpattern)
    filelist = sorted(filelist, key=str.lower)
    begin = offset
    if nfiles is not None and offset+nfiles<len(filelist):
        end=offset+nfiles
    else:
        end = len(filelist)
    filelist = filelist[begin:end]
    nx, ny = vigra.readImage(filelist[0]).squeeze().shape
    dt = vigra.readImage(filelist[0]).dtype
    nz = len(filelist)
    volume = numpy.zeros((nx, ny, nz, 1), dtype=dt)
    
    for i in range(len(filelist)):
        volume[:, :, i, 0] = vigra.readImage(filelist[i]).squeeze()[:]
        
    outfile = h5py.File("/home/anna/data/tmp/random_raw_stack.h5", "w")
    outfile.create_dataset("data", data=volume)
    outfile.close()
    return volume
    


class OpUpsampleByTwo( Operator ):
    Input = InputSlot()
    Output = OutputSlot()
    
    def setupOutputs(self):
        assert len(self.Input.meta.shape)==4 #we only take 4D data
        tagged_shape = self.Input.meta.getTaggedShape()
        shape_x = tagged_shape['x']
        shape_y = tagged_shape['y']
        shape_t = tagged_shape['t']
        new_shape_x = shape_x*2 - 1
        new_shape_y = shape_y*2 - 1
        self.Output.meta.assignFrom(self.Input.meta)
        self.Output.meta.shape = (new_shape_x, new_shape_y, shape_t, 1)
        self.Output.meta.axistags = vigra.VigraArray.defaultAxistags('xytc')
        #print "output of upsampling operator:", self.Output.meta
        
        
    def execute(self, slot, subindex, roi, result):
        x_index, y_index, t_index = map(self.Input.meta.axistags.index, 'xyt')
        roi_array = numpy.array([roi.start, roi.stop])
        roi_array[:, x_index] = (roi_array[:, x_index]+1)/2
        roi_array[:, y_index] = (roi_array[:, y_index]+1)/2
        
        roi_up = 4*[slice(None, None, None)]
        shape_up_x = roi.stop[x_index] - roi.start[x_index]
        shape_up_y = roi.stop[y_index] - roi.start[y_index]
        t_start = roi_array[0, t_index]
        t_stop = roi_array[1, t_index]
        for it in range(t_start, t_stop):
            roi_array[0, t_index] = it
            roi_array[1, t_index] = it+1
            
            roi_up[t_index] = slice(it-t_start, it-t_start+1, None)
            roi_up = sliceToRoi(roi_up, result.shape)
            down_image = self.Input(roi_array[0], roi_array[1]).wait().squeeze()
            assert len(down_image.shape)==2
            up_image = vigra.resize(down_image, [shape_up_x, shape_up_y])
            up_image = up_image.reshape(up_image.shape+(1,)+(1,))
            result[roiToSlice(*roi_up)] = up_image
        return result
    
    def propagateDirty(self, slot, subindex, roi):
        #FIXME: do the correct roi
        self.Output.setDirty(roi.start, roi.stop)     

def testUpsample():
    input_data = numpy.random.randint(0, 255, (256, 256, 10, 1)).astype(numpy.float32)
    input_data = vigra.taggedView(input_data, 'xytc')
    graph = Graph()
    op = OpUpsampleByTwo(graph=graph)
    op.Input.setValue(input_data)
    out = op.Output[:].wait()
    print "Done!"

class OpCombinePredictions( Operator ):
    MembranePredictions = InputSlot()
    SynapsePredictions = InputSlot()
    Output = OutputSlot()
    
    def setupOutputs(self):
        assert self.MembranePredictions.meta.shape == self.SynapsePredictions.meta.shape
        self.Output.meta.assignFrom( self.MembranePredictions.meta )
        output_shape = list(self.MembranePredictions.meta.shape)
        output_shape[self.Output.meta.axistags.channelIndex] = 1
        self.Output.meta.shape = tuple(output_shape)
        self.Output.meta.dtype = numpy.float32 #or should we make it uint16?
    
    def execute(self, slot, subindex, roi, result):
        #request the right channel
        start_combine = time.clock()
        def makeNewChannelRoi(oldroi, channelIndex, channelValue, shape):
            roi_slice = list(roiToSlice(oldroi.start, oldroi.stop))
            roi_slice[channelIndex] = slice(channelValue, channelValue+1, None)
            return sliceToRoi(roi_slice, shape)
        
        
        roi_synapses = makeNewChannelRoi(roi, self.SynapsePredictions.meta.axistags.channelIndex, \
                                         SYNAPSE_CHANNEL, self.SynapsePredictions.meta.shape)
        
        
        roi_membranes = makeNewChannelRoi(roi, self.MembranePredictions.meta.axistags.channelIndex, \
                                          MEMBRANE_CHANNEL, self.MembranePredictions.meta.shape)
        
        membrane_req = self.MembranePredictions(roi_membranes[0], roi_membranes[1])
        synapse_req = self.SynapsePredictions(roi_synapses[0], roi_synapses[1])
        membrane_req.submit()
        synapse_req.submit()
        membrane_predictions = membrane_req.wait()
        synapse_predictions = synapse_req.wait()
        
        ''''
        start_3d = time.clock()
        membrane_predictions = self.MembranePredictions(roi_membranes[0], roi_membranes[1]).wait()
        stop_3d = time.clock()
        print "spent in 2d prediction:", stop_3d-start_3d
        start2d = time.clock()
        synapse_predictions = self.SynapsePredictions(roi_synapses[0], roi_synapses[1]).wait()
        stop2d = time.clock()
        print "spent in 3d prediction:", stop2d - start2d
        '''
        #print "provided synapse and membrane predictions"
        #print numpy.sum(membrane_predictions), numpy.sum(synapse_predictions)
        result[:] = membrane_predictions[...]
        result[:] += synapse_predictions[...]
        stop_combine = time.clock()
        logger.debug( "spent for combining predictions:".format( stop_combine-start_combine ) )
        return result

    def propagateDirty(self, slot, subindex, roi):
        self.Output.setDirty(roi.start, roi.stop)        



if __name__=="__main__":
    #testUpsample()
    #test_find_synapses()
    #import sys
    ##sys.path.append("/groups/flyem/home/kreshuka/workspace/scripts")

    SMALL_TEST = False
    if SMALL_TEST:
        # Generate skeleton points. Here we assume that we get at least one point per slice
        skeletons = [(50, 50, 0), (150, 150, 1), (100, 100, 10), (100, 100, 5)] 
        rois = [numpy.array(((0, 0, 0), (100, 100, 1))), \
                numpy.array(((100, 100, 1), (200, 200, 2))), \
                numpy.array(((0, 0, 10), (200, 200, 11))), \
                numpy.array([[0, 0, 5], [200, 200, 6]])]
        
        rois = map( lambda a: a.astype(long), rois )
        
        branchwise_rois = [zip(skeletons, rois)]
        
        #input_filepath = os.path.join( input_dir, "*.tiff" )
        #volume_from_dir(input_filepath)
        input_filepath = "/home/bergs/workspace/anna_scripts/fruitfly/random_raw_stack.h5/data"
        do_stuff(project3dname, project2dname, input_filepath, outdir, branchwise_rois, debug_images=True, order2d='xytc', order3d='xyzc')
    else:
        X_RES = 4.0
        Y_RES = 4.0
        Z_RES = 45.0
         
        ROI_RADIUS = 150
         
        #import sys
        #sys.argv.append('/home/anna/scripts/fruitfly/example_skeleton.swc')
        #sys.argv.append('/Users/bergs/Documents/workspace/anna_scripts/3034133.swc')
         
        #swc_path = '/home/bergs/workspace/anna_scripts/fruitfly/example_skeleton.swc'
        ''''
        swc_path = "/groups/flyem/home/kreshuka/workspace/scripts/fruitfly/15886416.swc"
        from swc_rois import *
        node_infos = parse_swc( swc_path, X_RES, Y_RES, Z_RES )
        tree = construct_tree( node_infos )
        tree_coords_and_rois = coords_and_rois_for_tree(tree, radius=ROI_RADIUS)
        '''
        #for branch_coords_and_rois in tree_coords_and_rois[:5]:
        #    print "NEXT BRANCH"
        #    for coord, roi in branch_coords_and_rois[:5]:
        #        print "coord = {}, roi = {}".format( coord, roi )
     
         
     
        #input_filepath = os.path.join( input_dir, "*.tiff" )
        #volume_from_dir(input_filepath)
        #input_filepath = os.path.join(input_dir, "random_raw_stack.h5/data")
        input_filepath = "/home/akreshuk/scripts/fruitfly/cardona_volume_description.json"
        ''''
        small_branch = [tree_coords_and_rois[0][100:-1]]
        '''
        
        #skeleton_center = numpy.array((12226, 9173, 229))
        skeleton_center = numpy.array((11949, 17487, 2420))
        fake_branch = [[skeleton_center+[0,0,0], numpy.array((skeleton_center - [100,100,0], skeleton_center + [100,100,1]))],
                       [skeleton_center+[0,0,1], numpy.array((skeleton_center - [100,100,-1], skeleton_center + [100,100,2]))],
                       [skeleton_center+[0,0,2], numpy.array((skeleton_center - [100,100,-2], skeleton_center + [100,100,3]))],
                       [skeleton_center+[0,0,3], numpy.array((skeleton_center - [100,100,-3], skeleton_center + [100,100,4]))],
                       [skeleton_center+[0,0,4], numpy.array((skeleton_center - [100,100,-4], skeleton_center + [100,100,5]))],
                       [skeleton_center+[0,0,5], numpy.array((skeleton_center - [100,100,-5], skeleton_center + [100,100,6]))],
                       ]

        #print small_branch
        #print len(tree_coords_and_rois), len(tree_coords_and_rois[0])
        do_stuff(project3dname, project2dname, input_filepath, outdir, [fake_branch], debug_images=False, order2d='xyt', order3d='xyz')
        
    
    
        
        
    
