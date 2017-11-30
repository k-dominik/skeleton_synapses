from collections import namedtuple


TileIndex = namedtuple('TileIndex', ['z_idx', 'y_idx', 'x_idx'])
NodeInfo = namedtuple('NodeInfo', ['id', 'x_px', 'y_px', 'z_px', 'parent_id'])

SynapseDetectionOutput = namedtuple('SynapseDetectionOutput', ['tile_idx', 'predictions_xyc', 'synapse_cc_xyc'])

SkeletonAssociationInput = namedtuple('SkeletonAssociationInput', ['roi_xyz', 'synapse_slice_ids', 'synapse_object_id'])
SkeletonAssociationOutput = namedtuple('SkeletonAssociationOutput', ['node_id', 'synapse_slice_id', 'contact_px'])

# todo: delete if nothing breaks
# SegmenterInput = namedtuple('SegmenterInput', ['node_overall_index', 'node_info', 'roi_radius_px'])
# SegmenterOutput = namedtuple('SegmenterOutput', ['node_overall_index', 'node_info', 'roi_radius_px',
#                                                  'predictions_xyc', 'synapse_cc_xy', 'segmentation_xy'])
