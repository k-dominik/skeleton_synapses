from skimage.morphology import skeletonize
import numpy as np
import vigra

from skeleton_synapses.dto import NeuronSegmenterOutput


def get_synapse_segment_overlaps(synapse_cc_xy, segmentation_xy, synapse_slice_ids):
    # todo: test
    overlapping_segments = dict()
    for synapse_slice_id in synapse_slice_ids:
        # todo: need to cast some types?
        segments = np.unique(segmentation_xy[synapse_cc_xy == synapse_slice_id])
        for overlapping_segment in segments:
            if overlapping_segment not in overlapping_segments:
                overlapping_segments[overlapping_segment] = set()
            overlapping_segments[overlapping_segment].add(synapse_slice_id)

    return overlapping_segments


def get_node_associations(synapse_cc_xy, segmentation_xy, node_locations, overlapping_segments):
    # todo: test
    node_locations_arr = node_locations_to_array(synapse_cc_xy, node_locations)
    where_nodes_exist = node_locations_arr >= 0

    outputs = []
    for segment, node_id in zip(segmentation_xy[where_nodes_exist], node_locations_arr[where_nodes_exist]):
        for synapse_slice_id in overlapping_segments.get(segment, []):
            contact_px = skeletonize((synapse_cc_xy == synapse_slice_id) * (segmentation_xy == segment)).sum()
            outputs.append(NeuronSegmenterOutput(node_id, synapse_slice_id, contact_px))

    return outputs


def node_locations_to_array(template_array_xy, node_locations):
    """
    Given a vigra image in xy and a dict containing xy coordinates, return a vigra image of the same shape, where nodes
    are represented by their integer ID, and every other pixel is -1.

    Parameters
    ----------
    template_array_xy : vigra.VigraArray
    node_locations : dict
        dict whose values are a dicts containing a 'coords' dict and a 'treenode_id' value

    Returns
    -------
    vigra.VigraArray
    """
    if not isinstance(template_array_xy, vigra.VigraArray):
        template_array_xy = vigra.taggedView(template_array_xy, axistags='xy')
    else:
        assert template_array_xy.axistags.keys() == ['x', 'y']

    arr_xy = template_array_xy.copy()
    arr_xy.fill(-1)
    for node_location in node_locations.values():
        coords = node_location['coords']
        arr_xy[coords['x'], coords['y']] = int(node_location['treenode_id'])

    return arr_xy
