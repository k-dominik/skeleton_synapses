import logging
from Queue import Empty

from tqdm import tqdm

from skeleton_synapses.helpers.files import write_predictions_synapses
from skeleton_synapses.helpers.images import submit_synapse_slice_data, remap_synapse_slices
from skeleton_synapses.helpers.roi import tile_index_to_bounds
from skeleton_synapses.constants import TQDM_KWARGS, RESULTS_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)


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
