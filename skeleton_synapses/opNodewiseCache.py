import os
import collections

import numpy
import vigra

from lazyflow.graph import Operator, InputSlot, OutputSlot
from lazyflow.operators.ioOperators import OpInputDataReader

class OpNodewiseCache(Operator):
    ComputedInput = InputSlot()
    RoiToPathFn = InputSlot()   # Provide a function that produces a path from a given ROI.
                                # If the data can't be found, then return None
                                # Signature: fn(axiskeys, start, stop) -> path_str

    TransformFn = InputSlot(optional=True)  # Used to transform the cached data.
                                            # NOT used with the 'computed' data.
    
    Output = OutputSlot()

    def __init__(self, *args, **kwargs):
        super( OpNodewiseCache, self ).__init__(*args, **kwargs)
        self._opReader = OpInputDataReader( parent=self )

    def setupOutputs(self):
        assert self.ComputedInput.meta.getAxisKeys()[-1] == 'c', \
            "This operator assumes channel is always the last axis."
        self.Output.meta.assignFrom( self.ComputedInput.meta )
        
        if self.TransformFn.ready() and self.ComputedInput.meta.drange:
            input_drange = self.ComputedInput.meta.drange
            output_drange = self.TransformFn.value( numpy.array(input_drange) )
            self.Output.meta.drange = tuple( output_drange )
    
    def execute(self, slot, subindex, roi, result):
        roi_shape = roi.stop - roi.start
        assert roi_shape[-1] == 1, "Only single-channel rois are allowed!"
        
        axiskeys = self.Output.meta.getAxisKeys()
        path_fn = self.RoiToPathFn.value
        image_path = path_fn( axiskeys, roi.start, roi.stop )
        
        if image_path is not None:
            self._opReader.FilePath.setValue( image_path )
            # Shape must exactly match (in 2d, that is).
            if self._opReader.Output.meta.shape[:-1] == result.squeeze().shape:
                # Read.                
                # (Can't use writeInto here because dtypes are not guaranteed to match.)
                cached_data = self._opReader.Output[:].wait()
                if self.TransformFn.ready():
                    transform_fn = self.TransformFn.value
                    cached_data = transform_fn(cached_data)

                # Create a 3D view so we can copy from the reader
                result_view = vigra.taggedView( result, self.Output.meta.axistags )
                result_view = result_view.withAxes( *self._opReader.Output.meta.getAxisKeys() )
                result_view[:] = cached_data

                return result 
        
        # If we're here, then we couldn't find the data
        self.ComputedInput(roi.start, roi.stop).writeInto( result ).wait()
        return result
        
    def propagateDirty(self, slot, subindex, roi):
        if slot == self.ComputedInput:
            self.Output.setDirty(roi)
        elif slot == self.RoiToPathFn:
            self.Output.setDirty()
        elif slot != self.TransformFn:
            assert False, "Unknown input slot passed to propagateDirty()"

if __name__ == "__main__":
    TEST_RESOLUTION_XYZ = ( 4.0, 4.0, 45.0 )
    from skeleton_synapses.skeleton_utils import parse_skeleton_json, parse_skeleton_ids
    skeleton_path = '/groups/flyem/home/bergs/workspace/skeleton_synapses/test_skeletons/skeleton_18689.json'

    skeleton_ids = parse_skeleton_ids( skeleton_path )
    assert len(skeleton_ids) == 1, \
        "Found more than one skeleton in {}".format( skeleton_path )
    skeleton_id = skeleton_ids[0]
    
    node_infos = parse_skeleton_json( skeleton_path, *TEST_RESOLUTION_XYZ )
    node_coords_to_ids = { (n.x_px, n.y_px, n.z_px) : n.id for n in node_infos }

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
        x_px, y_px, z_px = tagged_coords['x'], tagged_coords['y'], tagged_coords['z']
        
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
    
    from functools import partial
    import vigra
    from lazyflow.graph import Graph, MetaDict
    
    class OpConstantProvider(Operator):
        """
        A silly little operator that provides output
        arrays filled with a constant value.
        """
        Output = OutputSlot()
        
        # The constant value and metadata of the output is passed in via the constructor.
        def __init__(self, constant_value, meta, *args, **kwargs):
            super(OpConstantProvider, self).__init__(*args, **kwargs)
            self._constant_value = constant_value
            self.Output.meta.assignFrom(meta)
        
        def setupOutputs(self):
            pass
        
        def execute(self, slot, subindex, roi, result):
            result[...] = self._constant_value

    
    RAW_VALUE = -1234.5678
    raw_data = RAW_VALUE * numpy.ones( (301, 301, 1), dtype=numpy.float32 )
    path_format = "/groups/flyem/home/bergs/workspace/skeleton_synapses/debuggingmembrane/{z_px}-{node_id}.png"
    
    graph = Graph()
    
    raw_meta = MetaDict( { 'shape' : (100000, 100000, 2000, 1),
                           'dtype' : numpy.float32,
                           'axistags' : vigra.defaultAxistags('xyzc') } ) 
    opRawProvider = OpConstantProvider( RAW_VALUE, raw_meta, graph=graph )
    
    opNodewiseCache = OpNodewiseCache(graph=graph)
    opNodewiseCache.TransformFn.setValue( lambda a: numpy.asarray(a, dtype=numpy.float32) / 255.0 )
    opNodewiseCache.RoiToPathFn.setValue( partial( roi_to_path, skeleton_id, node_coords_to_ids, path_format ) )
    opNodewiseCache.ComputedInput.connect( opRawProvider.Output )
    
    TEST_RADIUS = 150
    
    test_node_id = 39053
    test_node_coord_xyzc = numpy.array( (14630, 5687, 111, 0) )
    
    test_roi_start = test_node_coord_xyzc - (TEST_RADIUS, TEST_RADIUS, 0, 0)
    test_roi_stop = test_node_coord_xyzc + (TEST_RADIUS+1, TEST_RADIUS+1, 1, 1)
    test_roi = ( tuple(test_roi_start), tuple(test_roi_stop) )
    
    tile_data = opNodewiseCache.Output( *test_roi ).wait()
    assert (tile_data != RAW_VALUE).all(), "tile data was not loaded from file."
    print "DONE."

