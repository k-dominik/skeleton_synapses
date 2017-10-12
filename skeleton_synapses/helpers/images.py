import logging

import geojson
import numpy as np
import vigra
from skimage.measure import find_contours

from skeleton_synapses.helpers.files import LABEL_DTYPE


logger = logging.getLogger(__name__)


def image_to_geojson(array_xy, x_offset, y_offset):
    """
    Return geojson polygon string of binary image

    Parameters
    ----------
    array_xy
    x_offset
    y_offset

    Returns
    -------
    str
    """
    contours = find_contours(array_xy, 0.5, positive_orientation='high')
    rings = []
    for contour in contours:
        coords_list = [list(row) for row in contour + [x_offset, y_offset]]
        if coords_list[0] != coords_list[-1]:
            coords_list.append(coords_list[0])
        rings.append(coords_list)

    return geojson.dumps(geojson.Polygon(rings, validate=True), sort_keys=True)


def get_synapse_uncertainty(flat_predictions):
    """

    Parameters
    ----------
    flat_predictions : np.array
        MxN array where M is the number of pixels, and N is the number of colour channels

    Returns
    -------
    float
        Single uncertainty metric for synapse slice
    """
    assert len(flat_predictions.shape) == 2
    # Sort along channel axis
    flat_predictions.sort(axis=-1)
    # What's the difference between the highest and second-highest class?
    certainties = flat_predictions[:, -1] - flat_predictions[:, -2]
    avg_certainty = np.mean(certainties)  # todo: use a better metric
    return 1.0 - avg_certainty


def get_synapse_slice_size_centroid(binary_arr_xy, x_offset, y_offset):
    """

    Parameters
    ----------
    binary_arr_xy : vigra.VigraArray
        Boolean image of synapse slice
    x_offset : int
    y_offset : int

    Returns
    -------
    tuple
        size_px, x_centroid_px, y_centroid_px
    """
    pixel_coords_x, pixel_coords_y = np.where(binary_arr_xy)
    size_px = len(pixel_coords_x)
    x_centroid_px = np.average(pixel_coords_x) + x_offset
    y_centroid_px = np.average(pixel_coords_y) + y_offset
    return size_px, x_centroid_px, y_centroid_px


def synapse_slices_to_data(predictions_xyc, synapse_cc_xy, x_offset, y_offset):
    """

    Parameters
    ----------
    predictions_xyc : vigra.VigraArray
    synapse_cc_xy : vigra.VigraArray
    x_offset : int
    y_offset : int

    Returns
    -------
    list of dict
    """
    data = []

    for local_label in np.unique(synapse_cc_xy)[1:].astype(int):
        binary_arr_xy = synapse_cc_xy == local_label

        size_px, x_centroid_px, y_centroid_px = get_synapse_slice_size_centroid(
            binary_arr_xy, x_offset, y_offset
        )

        data.append({
            'id': int(local_label),
            'geom': image_to_geojson(binary_arr_xy, x_offset, y_offset),
            'size_px': int(size_px),
            'xs_centroid': int(x_centroid_px),
            'ys_centroid': int(y_centroid_px),
            'uncertainty': get_synapse_uncertainty(predictions_xyc[binary_arr_xy])
        })

    return data


def submit_synapse_slice_data(bounds_xyz, predictions_xyc, synapse_cc_xy, tile_idx, catmaid, workflow_id):
    synapse_slices = synapse_slices_to_data(
        predictions_xyc, synapse_cc_xy, bounds_xyz[0, 0], bounds_xyz[0, 1]
    )

    id_mapping = catmaid.add_synapse_slices_to_tile(workflow_id, synapse_slices, tile_idx)
    logger.debug('Got ID mapping from CATMAID:\n{}'.format(id_mapping))

    local_label_set = set(np.unique(synapse_cc_xy)[1:].astype(int))
    returned_keys = set(id_mapping)
    assert returned_keys == local_label_set, 'Returned keys are not the same as sent keys:\n\t{}\n\t{}'.format(
        returned_keys, local_label_set
    )

    return id_mapping


def remap_synapse_slices(synapse_cc_xy, id_mapping):
    """

    Parameters
    ----------
    synapse_cc_xy : vigra.VigraArray
        Synapse image using local connected component labels, background 0
    id_mapping : dict
        Mapping for all nonzero labels

    Returns
    -------
    vigra.VigraArray
        Synapse image using project connected component labels, background 1
    """
    mapped_synapse_cc_xy = vigra.taggedView(np.ones(synapse_cc_xy.shape, LABEL_DTYPE), axistags='xy')
    for local_label, synapse_id in id_mapping.items():
        logger.debug('Addressing ID mapping pair: {}, {}'.format(local_label, synapse_id))
        mapped_synapse_cc_xy[synapse_cc_xy == local_label] = synapse_id

    return mapped_synapse_cc_xy


def are_same_xy(*args):
    if not args:
        return True

    return all(
        args[0].shape[:2] == arg.shape[:2] and tuple(args[0].axistags)[:2] == tuple(arg.axistags)[:2] for arg in args
    )