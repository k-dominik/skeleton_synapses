import numpy
import vigra
from lazyflow.graph import Graph

from skeleton_synapses.OLD.opUpsampleByTwo import OpUpsampleByTwo


class TestOpUpsampleByTwo(object):    

    def testUpsample(self):
        input_data = numpy.random.randint(0, 255, (256, 256, 10, 1)).astype(numpy.float32)
        input_data = vigra.taggedView(input_data, 'xytc')
        graph = Graph()
        op = OpUpsampleByTwo(graph=graph)
        op.Input.setValue(input_data)
        out = op.Output[:].wait()

        # FIXME: This test doesn't do much...
        # TODO: Check the output...
    
if __name__ == "__main__":
    import sys
    import nose
    sys.argv.append("--nocapture")    # Don't steal stdout.  Show it on the console as usual.
    sys.argv.append("--nologcapture") # Don't set the logging level to DEBUG.  Leave it alone.
    sys.exit( nose.run(defaultTest=__file__) )
