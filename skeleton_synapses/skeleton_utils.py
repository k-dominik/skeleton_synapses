import collections
import numpy


# Not  used in this file, but defined for cmd-line utilities to use.
CSV_FORMAT = { 'delimiter' : '\t', 'lineterminator' : '\n' }


NodeInfo = collections.namedtuple('NodeInfo', 'id x_px y_px z_px parent_id')


def roi_around_point(coord_xyz, radius):
    """
    Produce a 3D roi (start, stop) tuple that surrounds the
    node coordinates, with Z-thickness of 1.
    """
    coord_xyz = numpy.array(coord_xyz)
    start = coord_xyz - [radius, radius, 0]
    stop = coord_xyz + [radius+1, radius+1, 1]
    return numpy.array((tuple(start), tuple(stop)))


def roi_around_node(node_info, radius):
    coord_xyz = (node_info.x_px, node_info.y_px, node_info.z_px)
    return roi_around_point(coord_xyz, radius)
