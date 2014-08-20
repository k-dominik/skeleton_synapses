import numpy
from lazyflow.roi import roiToSlice
from skeleton_synapses.locate_synapses import normalize_synapse_ids

def test_normalize_synapse_ids():
    slice1 = numpy.zeros((20, 20, 1), dtype=numpy.uint8)
    slice2 = numpy.zeros((20, 20, 1), dtype=numpy.uint8)
    
    slice1[0:3, 0:3] = 1
    slice1[7:9, 2:3] = 3

    slice2[2:5, 2:5] = 2
    slice2[1:3, 7:9] = 5

    roi1 = [(0,0,0), (10,10,1)]
    roi2 = [(2,2,1), (12,12,2)]
    
    roi1_2d = (roi1[0][:-1], roi1[1][:-1])
    roi2_2d = (roi2[0][:-1], roi1[1][:-1])
    
    extracted_slice1 = slice1[roiToSlice(*roi1_2d)]
    extracted_slice2 = slice2[roiToSlice(*roi2_2d)]
    
    result1, maxLabel = normalize_synapse_ids(extracted_slice1, roi1, None, None, 0)
    assert numpy.all(result1==extracted_slice1)
    assert maxLabel==3
    
    result2, maxLabel2 = normalize_synapse_ids(extracted_slice2, roi2, extracted_slice1, roi1, maxLabel)
    
    # Copy into the original (big) array for straightforward comparison
    slice2[roiToSlice(*roi2_2d)] = result2

    # Note the modified slicings for comparison: 
    #  we don't care what happened outside the intersection region.
    assert numpy.all(slice2[2:5, 2:5]==1)
    assert numpy.all(slice2[2:3, 7:9]==maxLabel+1)
    assert maxLabel2==4


if __name__ == "__main__":
    import sys
    import nose
    sys.argv.append("--nocapture")    # Don't steal stdout.  Show it on the console as usual.
    sys.argv.append("--nologcapture") # Don't set the logging level to DEBUG.  Leave it alone.
    sys.exit(nose.run(defaultTest=__file__))
