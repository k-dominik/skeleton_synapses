import h5py
import glob
import vigra
from vigra import graphs
import numpy

from lazyflow.graph import Graph
from opUpsampleByTwo import OpUpsampleByTwo
from lazyflow.operators.vigraOperators import OpPixelFeaturesPresmoothed

# inputdir = "/home/akreshuk/data/connector_archive_2g0y0b/distance_tests/"
inputdir = "/home/anna/data/distance_tests/"
# debugdir = "/home/anna/data/distance_tests/debug_distance_images/"
debugdir = inputdir + "/debug_distance_images*/"
d2_pattern = "*_2d_pred*.h5"
d3_pattern = "*_3d_pred*.h5"
marker_pattern = "*_with_markers.*"
raw_pattern = "*_raw.tiff"

debug_images = True


def computeDistanceHessian(upsampledMembraneProbs, sigma, ddir):
    # compute the second Hessian Eigenvalue of the upsampled probability map
    #
    # save the resulting image into ddir, if present

    tempGraph = Graph()
    opFeatures = OpPixelFeaturesPresmoothed(graph=tempGraph)

    # Compute the Hessian slicewise and create gridGraphs
    standard_scales = [0.3, 0.7, 1.0, 1.6, 3.5, 5.0, 10.0]
    standard_feature_ids = ['GaussianSmoothing', 'LaplacianOfGaussian', \
                            'GaussianGradientMagnitude', 'DifferenceOfGaussians', \
                            'StructureTensorEigenvalues', 'HessianOfGaussianEigenvalues']

    opFeatures.Scales.setValue(standard_scales)
    opFeatures.FeatureIds.setValue(standard_feature_ids)

    # Select the sigma from the parameters
    scale_index = standard_scales.index(sigma)
    feature_index = standard_feature_ids.index('HessianOfGaussianEigenvalues')
    selection_matrix = numpy.zeros((6, 7), dtype=bool)  # all False
    selection_matrix[feature_index][scale_index] = True
    opFeatures.Matrix.setValue(selection_matrix)

    opFeatures.Input.setValue(upsampledMembraneProbs)
    eigenValues = opFeatures.Output[..., 1:2].wait()  # we need the second eigenvalue
    eigenValues = numpy.abs(eigenValues[:, :, 0])

    if debug_images:
        outfile = ddir + "/hess.tiff"
        vigra.impex.writeImage(eigenValues, outfile)

    return eigenValues


def computeDistanceRaw(upsampledMembraneProbs, sigma, ddir):
    # smooth the upsampled prob map by Gaussian with sigma
    #
    # save the result as image in ddir

    smoothed = vigra.filters.gaussianSmoothing(upsampledMembraneProbs, sigma)
    if debug_images:
        outfile = ddir + "rawprobs.tiff"
        vigra.impex.writeImage(smoothed, outfile)
    return smoothed


def filter_by_size(upsampledSmoothedMembraneProbs, ddir, threshold_high=0.9, threshold_low=0.1, minSize=1000):
    # remove small connected components from the probability map (hoping to remove noise in the cytoplasm)
    # use 2-level thresholding as usual

    thresholded = upsampledSmoothedMembraneProbs > threshold_high
    cc = vigra.analysis.labelImageWithBackground(thresholded.astype(numpy.uint8))

    counts = numpy.bincount(cc.flat)
    counts[counts < minSize] = 0
    counts[0] = 0
    cc_filtered = counts[cc]
    returnMaps = numpy.zeros(upsampledSmoothedMembraneProbs.shape, upsampledSmoothedMembraneProbs.dtype)
    thresholded_low = upsampledSmoothedMembraneProbs > threshold_low

    cc_low = vigra.analysis.labelImageWithBackground(thresholded_low.astype(numpy.uint8))
    indices_low = numpy.zeros((numpy.max(cc_low) + 1,), dtype=numpy.uint32)

    cc_low_filtered = cc_low[cc_filtered > 0]
    counts_low = numpy.bincount(cc_low_filtered.flat)
    indices_low[0:counts_low.shape[0]] = counts_low[:]

    cc_low_filtered_full = indices_low[cc_low]

    indices = cc_low_filtered_full > 0
    returnMaps[indices] = upsampledSmoothedMembraneProbs[indices]

    if debug_images:
        outfile = ddir + "cc_filtered.tiff"
        vigra.impex.writeImage(cc_filtered, outfile)
        outfile2 = ddir + "map_filtered.tiff"
        vigra.impex.writeImage(returnMaps, outfile2)
    return returnMaps


def extractMarkedNodes(filename):
    # extracts markers from the raw images
    # returns a dict, with color as key and a list
    # of marker center coords as value

    im = vigra.readImage(filename)
    colored1 = im[..., 0] != im[..., 1]
    colored2 = im[..., 1] != im[..., 2]
    colored3 = im[..., 2] != im[..., 0]
    colored = numpy.logical_or(colored1, colored2)
    colored = numpy.logical_or(colored, colored3)
    cc = vigra.analysis.labelImageWithBackground(colored.astype(numpy.uint8))
    # take the center pixel for each colored square
    feats = vigra.analysis.extractRegionFeatures(colored.astype(numpy.float32), cc, ["RegionCenter"])
    center_coords = feats["RegionCenter"][1:][:].astype(numpy.uint32)
    center_coords_list = [center_coords[:, 0], center_coords[:, 1]]
    im_centers = numpy.asarray(im[center_coords_list])
    # print im_centers
    # struct = im_centers.view(dtype='f4, f4, f4')

    # colors, indices = numpy.unique(struct, return_inverse=True)
    # print colors, indices, colors.shape
    centers_by_color = {}
    for iindex in range(center_coords.shape[0]):
        center = (center_coords[iindex][0], center_coords[iindex][1])
        # print center, index
        centers_by_color.setdefault(tuple(im_centers[iindex]), []).append(center)

    print centers_by_color
    return centers_by_color


def calibrate_distance():
    files_2d = glob.glob(inputdir + d2_pattern)
    files_2d = sorted(files_2d, key=str.lower)

    files_3d = glob.glob(inputdir + d3_pattern)
    files_3d = sorted(files_3d, key=str.lower)

    files_markers = glob.glob(inputdir + marker_pattern)
    files_markers = sorted(files_markers, key=str.lower)

    debug_dirs = glob.glob(debugdir)
    debug_dirs = sorted(debug_dirs, key=str.lower)

    files_raw = glob.glob(inputdir + raw_pattern)
    files_raw = sorted(files_raw, key=str.lower)

    print files_2d, files_3d, files_markers, debug_dirs, files_raw

    tempGraph = Graph()
    edgeIndicators = []
    instances = []

    first = 0
    last = 3

    for f2name, f3name, mname, ddir, rawname in zip(files_2d[first:last], files_3d[first:last],
                                                    files_markers[first:last], debug_dirs[first:last],
                                                    files_raw[first:last]):
        print f2name, f3name, mname

        if debug_images:
            rawim = vigra.readImage(rawname)
            vigra.impex.writeImage(rawim, ddir + "/raw.tiff")

        f2 = h5py.File(f2name)
        f3 = h5py.File(f3name)

        d2 = f2["exported_data"][..., 0]
        d3 = f3["exported_data"][5, :, :, 2]
        d3 = d3.swapaxes(0, 1)

        # print d2.shape, d3.shape

        combined = d2 + d3

        markedNodes = extractMarkedNodes(mname)
        # print
        opUpsample = OpUpsampleByTwo(graph=tempGraph)
        combined = numpy.reshape(combined, combined.shape + (1,) + (1,))
        combined = combined.view(vigra.VigraArray)
        combined.axistags = vigra.defaultAxistags('xytc')
        opUpsample.Input.setValue(combined)
        upsampledMembraneProbs = opUpsample.Output[:].wait()
        # get rid of t
        upsampledMembraneProbs = upsampledMembraneProbs[:, :, 0, :]
        upsampledMembraneProbs = upsampledMembraneProbs.view(vigra.VigraArray)
        upsampledMembraneProbs.axistags = vigra.defaultAxistags('xyc')

        # try to filter
        upsampledSmoothedMembraneProbs = computeDistanceRaw(upsampledMembraneProbs, 1.6, ddir)
        filter_by_size(upsampledSmoothedMembraneProbs, ddir)

        edgeIndicators.append(computeDistanceHessian(upsampledMembraneProbs, 5.0, ddir))
        edgeIndicators.append(upsampledSmoothedMembraneProbs)

        gridGr = graphs.gridGraph((d2.shape[0], d2.shape[1]))  # !on original pixels
        for iind, indicator in enumerate(edgeIndicators):
            gridGraphEdgeIndicator = graphs.edgeFeaturesFromInterpolatedImage(gridGr, indicator)
            instance = vigra.graphs.ShortestPathPathDijkstra(gridGr)
            instances.append(instance)
            distances_same = []
            for color, points in markedNodes.iteritems():
                print color
                for i in range(len(points)):
                    node = map(long, points[i])
                    sourceNode = gridGr.coordinateToNode(node)
                    instance.run(gridGraphEdgeIndicator, sourceNode, target=None)
                    distances_all = instance.distances()

                    for j in range(i + 1, len(points)):
                        other_node = map(long, points[j])
                        distances_same.append(distances_all[other_node[0], other_node[1]])
                        targetNode = gridGr.coordinateToNode(other_node)
                        path = instance.run(gridGraphEdgeIndicator, sourceNode).path(pathType='coordinates',
                                                                                     target=targetNode)
                        max_on_path = numpy.max(distances_all[path])
                        min_on_path = numpy.min(distances_all[path])
                        # print max_on_path, min_on_path
                        # print path.shape
                        print "distance b/w", node, other_node, " = ", distances_all[other_node[0], other_node[1]]

                    distances_all[node[0], node[1]] = numpy.max(distances_all)
                    outfile = ddir + "/" + str(node[0]) + "_" + str(node[1]) + "_" + str(iind) + ".tiff"
                    vigra.impex.writeImage(distances_all, outfile)

                    # print distances_same


                    # vigra.impex.writeImage(combined, f2name+"_combined.tiff")
                    # vigra.impex.writeImage(d3, f2name+"_synapse.tiff")
                    # vigra.impex.writeImage(d2, f2name+"_membrane.tiff")


if __name__ == "__main__":
    calibrate_distance()
