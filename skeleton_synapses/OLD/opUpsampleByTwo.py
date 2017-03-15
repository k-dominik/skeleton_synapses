import numpy
import vigra

from lazyflow.graph import Operator, InputSlot, OutputSlot
from lazyflow.roi import sliceToRoi, roiToSlice

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
            up_image = vigra.sampling.resizeImageLinearInterpolation(down_image, [shape_up_x, shape_up_y])
            up_image = up_image.reshape(up_image.shape+(1,)+(1,))
            result[roiToSlice(*roi_up)] = up_image
        return result
    
    def propagateDirty(self, slot, subindex, roi):
        #FIXME: do the correct roi
        self.Output.setDirty(roi.start, roi.stop)
        