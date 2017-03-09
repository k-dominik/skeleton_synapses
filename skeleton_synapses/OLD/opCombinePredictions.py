import numpy

from lazyflow.graph import Operator, InputSlot, OutputSlot
from lazyflow.roi import roiToSlice, sliceToRoi

import logging
logger = logging.getLogger(__name__)

class OpCombinePredictions( Operator ):
    """
    Given two prediction volumes (each with several channels),
    select one channel from each and add them together.  Output a volume with a single channel.
    The indexes of the two channels to add are provided via __init__.
    """
    MembranePredictions = InputSlot()
    SynapsePredictions = InputSlot()
    Output = OutputSlot()

    def __init__(self, synapse_channel, membrane_channel, *args, **kwargs):
        super(OpCombinePredictions, self).__init__(*args, **kwargs)
        self.SYNAPSE_CHANNEL = synapse_channel
        self.MEMBRANE_CHANNEL = membrane_channel
    
    def setupOutputs(self):
        assert self.MembranePredictions.meta.shape == self.SynapsePredictions.meta.shape
        self.Output.meta.assignFrom( self.MembranePredictions.meta )
        output_shape = list(self.MembranePredictions.meta.shape)
        output_shape[self.Output.meta.axistags.channelIndex] = 1
        self.Output.meta.shape = tuple(output_shape)
        self.Output.meta.dtype = numpy.float32 #or should we make it uint16?
    
    def execute(self, slot, subindex, roi, result):
        #request the right channel
        def makeNewChannelRoi(oldroi, channelIndex, channelValue, shape):
            roi_slice = list(roiToSlice(oldroi.start, oldroi.stop))
            roi_slice[channelIndex] = slice(channelValue, channelValue+1, None)
            return sliceToRoi(roi_slice, shape)
        
        
        roi_synapses = makeNewChannelRoi(roi, self.SynapsePredictions.meta.axistags.channelIndex, \
                                         self.SYNAPSE_CHANNEL, self.SynapsePredictions.meta.shape)
        
        
        roi_membranes = makeNewChannelRoi(roi, self.MembranePredictions.meta.axistags.channelIndex, \
                                          self.MEMBRANE_CHANNEL, self.MembranePredictions.meta.shape)
        
        membrane_req = self.MembranePredictions(roi_membranes[0], roi_membranes[1])
        synapse_req = self.SynapsePredictions(roi_synapses[0], roi_synapses[1])
        membrane_req.submit()
        synapse_req.submit()
        membrane_predictions = membrane_req.wait()
        synapse_predictions = synapse_req.wait()
        
        result[:] = membrane_predictions[...]
        result[:] += synapse_predictions[...]
        return result

    def propagateDirty(self, slot, subindex, roi):
        self.Output.setDirty(roi.start, roi.stop)        

