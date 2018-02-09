import logging
import multiprocessing as mp

try:
    from queue import Empty
except ImportError:
    from Queue import Empty

from tqdm import tqdm

from skeleton_synapses.dto import SkeletonAssociationInput
from skeleton_synapses.helpers.files import write_predictions_synapses, TILE_SIZE
from skeleton_synapses.helpers.images import submit_synapse_slice_data, remap_synapse_slices
from skeleton_synapses.helpers.roi import tile_index_to_bounds, nodes_to_tile_indexes, roi_around_synapse
from skeleton_synapses.constants import TQDM_KWARGS, RESULTS_TIMEOUT_SECONDS


logger = logging.getLogger(__name__)


def populate_tile_input_queue(catmaid, roi_radius_px, workflow_id, node_infos):
    tile_index_set = nodes_to_tile_indexes(node_infos, TILE_SIZE, roi_radius_px)

    addressed_tiles = catmaid.get_detected_tiles(workflow_id)

    tile_queue, tile_result_queue = mp.Queue(), mp.Queue()
    tile_count = 0
    for tile_idx in tqdm(tile_index_set, desc='Populating tile queue', unit='tiles', **TQDM_KWARGS):
        if (tile_idx.x_idx, tile_idx.y_idx, tile_idx.z_idx) in addressed_tiles:
            logger.debug("Tile %s has been addressed by this algorithm, skipping", repr(tile_idx))
        else:
            logger.debug("Tile %s has not been addressed, adding to queue", repr(tile_idx))
            tile_count += 1
            tile_queue.put(tile_idx)

    return tile_queue, tile_count


def populate_synapse_queue(catmaid, roi_radius_px, project_workflow_id, stack_info, skeleton_id):
    synapse_queue = mp.Queue()
    synapse_count = 0

    roi_radius_nm = roi_radius_px * stack_info['resolution']['x']  # assumes XY isotropy
    logger.debug('Getting synapses spatially near skeleton {}'.format(skeleton_id))
    synapses_near_skeleton = catmaid.get_synapses_near_skeleton(skeleton_id, project_workflow_id, roi_radius_nm)
    logger.debug('Found {} synapse planes near skeleton {}'.format(len(synapses_near_skeleton), skeleton_id))
    slice_id_tuples = set()
    for synapse in tqdm(synapses_near_skeleton, desc='Populating synapse plane queue', unit='synapse planes',
                        **TQDM_KWARGS):
        slice_id_tuple = tuple(synapse['synapse_slice_ids'])
        if slice_id_tuple in slice_id_tuples:
            continue

        slice_id_tuples.add(slice_id_tuple)
        roi_xyz = roi_around_synapse(synapse, roi_radius_px)

        logger.debug('Getting treenodes in roi {}'.format(roi_xyz))
        item = SkeletonAssociationInput(roi_xyz, slice_id_tuple, synapse['synapse_object_id'])
        logger.debug('Adding {} to neuron segmentation queue'.format(item))
        synapse_queue.put(item)
        synapse_count += 1

    return synapse_queue, synapse_count


def iterate_queue(queue, final_size, queue_name=None, timeout=RESULTS_TIMEOUT_SECONDS):
    if queue_name is None:
        queue_name = repr(queue)
    for idx in range(final_size):
        logger.debug('Waiting for item {} from queue {} (expect {} more)'.format(idx, queue_name, final_size - idx))
        try:
            item = queue.get(timeout=timeout)
        except Empty:
            logger.exception('Result queue timed out after {} seconds'.format(timeout))
            raise
        logger.debug('Got item {} from queue {}: {} (expect {} more)'.format(idx, queue_name, item, final_size-idx-1))
        yield item
    assert queue.empty(), 'More enqueued items in {} than expected'.format(queue_name)


def commit_tilewise_results_from_queue(
        tile_result_queue, output_path, total_tiles, tile_size, workflow_id, catmaid
):
    result_iterator = tqdm(
        iterate_queue(tile_result_queue, total_tiles, 'tile_result_queue'),
        desc='Synapse detection', unit='tiles', total=total_tiles, **TQDM_KWARGS
    )

    logger.info('Starting to commit tile classification results')

    for tile_count, (tile_idx, predictions_xyc, synapse_cc_xy) in enumerate(result_iterator):
        tilename = 'z{}-y{}-x{}'.format(*tile_idx)
        logger.debug('Committing results from tile {}, {} of {}'.format(tilename, tile_count, total_tiles))
        bounds_xyz = tile_index_to_bounds(tile_idx, tile_size)

        id_mapping = submit_synapse_slice_data(
            bounds_xyz, predictions_xyc, synapse_cc_xy, tile_idx, catmaid, workflow_id
        )

        catmaid.agglomerate_synapses(id_mapping.values())

        logger.debug('Got ID mapping from CATMAID:\n{}'.format(id_mapping))

        mapped_synapse_cc_xy = remap_synapse_slices(synapse_cc_xy, id_mapping)
        write_predictions_synapses(output_path, predictions_xyc, mapped_synapse_cc_xy, bounds_xyz)


def commit_node_association_results_from_queue(node_result_queue, total_nodes, project_workflow_id, catmaid):
    logger.debug('Committing node association results')

    result_list_generator = tqdm(
        iterate_queue(node_result_queue, total_nodes, 'node_result_queue'),
        desc='Synapse-treenode association', unit='synapse planes', total=total_nodes, **TQDM_KWARGS
    )

    logger.debug('Getting node association results')
    assoc_tuples = []
    for result_list in result_list_generator:
        for result in result_list:
            assoc_tuple = (result.synapse_slice_id, result.node_id, result.contact_px)
            logger.debug('Appending segmentation result to args: %s', repr(assoc_tuple))
            assoc_tuples.append(assoc_tuple)

    logger.debug('Node association results are\n%s', repr(assoc_tuples))
    logger.info('Inserting new slice:treenode mappings')

    catmaid.add_synapse_treenode_associations(assoc_tuples, project_workflow_id)
