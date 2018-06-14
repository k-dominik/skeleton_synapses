import os
import multiprocessing as mp
import time

try:
    from queue import Empty
except ImportError:
    from Queue import Empty

try:
    from unittest import mock
except ImportError:
    import mock

import pytest

from tests.context import skeleton_synapses

from skeleton_synapses.dto import SkeletonAssociationOutput
from skeleton_synapses.parallel.queues import (
    iterate_queue, commit_node_association_results_from_queue, QueueOverpopulatedException
)


@pytest.fixture
def catmaid():
    catmaid = mock.Mock()
    catmaid.add_synapse_treenode_associations = mock.Mock()
    return catmaid


def populate_queue(items, poll_interval=0.01):
    """

    Parameters
    ----------
    items : sequence or int
    poll_interval : float

    Returns
    -------

    """
    try:
        len(items)
    except TypeError:
        items = [1 for _ in range(items)]
    queue = mp.Queue()
    for item in items:
        queue.put(item)
    while queue.qsize() < len(items):
        time.sleep(poll_interval)

    return queue


def test_iterate_queue():
    item_count = final_size = 5
    queue = populate_queue(item_count)

    results = list(iterate_queue(queue, final_size, timeout=0.5))
    assert len(results) == final_size
    assert sum(results) == final_size


def test_iterate_queue_underpopulated():
    item_count = 3
    final_size = 5
    queue = populate_queue(item_count)

    with pytest.raises(Empty):
        for idx, result in enumerate(iterate_queue(queue, final_size, timeout=0.5)):
            assert result
            assert idx < item_count


def test_iterate_queue_overpopulated():
    item_count = 7
    final_size = 5
    queue = populate_queue(item_count)

    with pytest.raises(QueueOverpopulatedException):
        for idx, result in enumerate(iterate_queue(queue, final_size, timeout=0.5)):
            assert result
            assert idx < final_size


def test_commit_node_association_results_from_queue(catmaid):
    item_count = 10
    items = [
        SkeletonAssociationOutput('tnid{}'.format(i), 'ssid{}'.format(i), 'contact{}'.format(i)) for i in range(item_count)
    ]
    item_chunkings = [slice(None, 3), slice(3, 5), slice(5, None)]

    expected_args = [('ssid{}'.format(i), 'tnid{}'.format(i), 'contact{}'.format(i)) for i in range(item_count)]
    expected_args_chunks = [expected_args[chunk] for chunk in item_chunkings]

    item_chunks = [items[chunk] for chunk in item_chunkings]
    queue = populate_queue(item_chunks)
    commit_node_association_results_from_queue(queue, len(item_chunks), None, catmaid)

    for arg_chunk in expected_args_chunks:
        catmaid.add_synapse_treenode_associations.assert_any_call(arg_chunk, None)


if __name__ == '__main__':
    pytest.main(['-v', os.path.realpath(__file__)])
