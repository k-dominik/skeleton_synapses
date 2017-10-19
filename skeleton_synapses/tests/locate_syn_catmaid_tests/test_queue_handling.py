import multiprocessing as mp
import time
from Queue import Empty

import mock
import pytest

from skeleton_synapses.dto import NeuronSegmenterOutput
from skeleton_synapses.parallel.queues import iterate_queue, commit_node_association_results_from_queue


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

    with pytest.raises(AssertionError):
        for idx, result in enumerate(iterate_queue(queue, final_size, timeout=0.5)):
            assert result
            assert idx < final_size


def test_commit_node_association_results_from_queue(catmaid):
    item_count = 10
    items = [
        NeuronSegmenterOutput('tnid{}'.format(i), 'ssid'.format(i), 'contact{}'.format(i)) for i in range(item_count)
    ]
    expected_args = [('ssid'.format(i), 'tnid{}'.format(i), 'contact{}'.format(i)) for i in range(item_count)]

    item_chunks = [items[:3], items[3:5], items[5:]]
    queue = populate_queue(item_chunks)
    commit_node_association_results_from_queue(queue, len(item_chunks), None, catmaid)

    catmaid.add_synapse_treenode_associations.assert_called_once_with(
        expected_args, None
    )


if __name__ == '__main__':
    pytest.main(['-v', os.path.realpath(__file__)])
