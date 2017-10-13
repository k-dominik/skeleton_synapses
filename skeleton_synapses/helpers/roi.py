import logging
from itertools import product, starmap

import numpy as np

from skeleton_synapses.dto import TileIndex
from skeleton_synapses.constants import DEFAULT_ROI_RADIUS


logger = logging.getLogger(__name__)


def nodes_to_tile_indexes(node_infos, tile_size, minimum_radius=DEFAULT_ROI_RADIUS):
    """

    Parameters
    ----------
    node_infos : list of skeleton_utils.NodeInfo
    tile_size : int
        side length of square tile
    minimum_radius : float or int

    Returns
    -------
    set of TileIndex
    """
    tile_set = set()
    tile_size_xyz = np.array([tile_size, tile_size, 1])

    logger.info('Getting tile set for nodes')

    for node_info in node_infos:
        bounds_xyz = roi_around_node(node_info, minimum_radius)
        tile_idxs = (bounds_xyz / tile_size_xyz).astype(int)
        tile_set.update(TileIndex(*idxs) for idxs in product(
            [node_info.z_px],  # z index
            range(tile_idxs[0, 1], tile_idxs[1, 1] + 1),  # all tile y indices
            range(tile_idxs[0, 0], tile_idxs[1, 0] + 1)  # all tile x indices
        ))

    return tile_set


def tile_index_to_bounds(tile_index, tile_size):
    """

    Parameters
    ----------
    tile_index : skeleton_utils.TileIndex
    tile_size : int

    Returns
    -------
    numpy.ndarray
        [[min_x, min_y, min_z], [max_x, max_y, max_z]] in pixel coordinates
    """
    tile_size_xyz = np.array([tile_size, tile_size, 1])
    topleft = np.array([tile_index.x_idx, tile_index.y_idx, tile_index.z_idx], dtype=int)
    return np.stack((topleft, topleft+1)) * tile_size_xyz  # todo: might need to -1 to bottom row


def square_bounds(roi_xyz):
    """Convert a rectangular ROI array into the minimum square in which the original ROI is centered"""
    roi_xyz = np.array(roi_xyz)
    shape = np.diff(roi_xyz[:, :2], axis=0).squeeze()
    size_diff = shape[0] - shape[1]
    if size_diff == 0:
        return roi_xyz
    elif size_diff > 0:
        half_diff = float(size_diff) / 2
        smaller_dim = 1
    else:
        half_diff = float(size_diff) / -2
        smaller_dim = 0

    roi_xyz[0, smaller_dim] -= np.floor(half_diff)
    roi_xyz[1, smaller_dim] += np.ceil(half_diff)

    return roi_xyz


def roi_around_point(coord_xyz, radius):
    """
    Produce a 3D roi (start, stop) tuple that surrounds the
    node coordinates, with Z-thickness of 1.
    """
    coord_xyz = np.array(coord_xyz)
    start = coord_xyz - [radius, radius, 0]
    stop = coord_xyz + [radius+1, radius+1, 1]
    return np.array((tuple(start), tuple(stop)))


def roi_around_node(node_info, radius):
    coord_xyz = (node_info.x_px, node_info.y_px, node_info.z_px)
    return roi_around_point(coord_xyz, radius)


def roi_around_synapse(synapse, buffer_px):
    """

    Parameters
    ----------
    synapse : dict
        has keys 'synapse_bounds_s' and 'synapse_z_s'
    buffer_px : number

    Returns
    -------
    np.array
    """
    external_buffer = np.array([[-buffer_px, -buffer_px, 0], [buffer_px, buffer_px, 1]])

    # synapse plane bounds + buffer
    return (external_buffer + np.array([
        synapse['synapse_bounds_s'][:2] + [synapse['synapse_z_s']],  # xmin, ymin, zmin
        synapse['synapse_bounds_s'][2:] + [synapse['synapse_z_s']]  # xmax, ymax, zmax
    ])).astype(int)


def slicing(roi):
    """
    Convert the roi to a slicing that can be used with ndarray.__getitem__()
    """
    return tuple( starmap( slice, zip(*roi) ) )
