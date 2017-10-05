import os

import pytest
import networkx as nx
import numpy as np

from skeleton_synapses.catmaid_interface import (
    get_consecutive, extend_slices, make_tile_url_template, get_nodes_between, get_subarbor_node_infos, in_roi
)
from skeleton_synapses.skeleton_utils import NodeInfo

from common import get_fixture_data


def test_get_consecutive():
    test_input = [2, 4, 1, 5]
    expected_output = [[1, 2], [4, 5]]

    assert get_consecutive(test_input) == expected_output


def test_extend_slices():
    test_input = {'349': 1, '350': 1, '351': 1, '99': 1}
    expected_output = [[98, [99]], [348, [349, 350]], [352, [351]]]

    assert extend_slices(test_input) == expected_output


def test_extend_slices_value_not_one():
    test_input = {'349': 1, '350': 1, '351': 1, '99': 2}
    with pytest.raises(AssertionError) as e:
        extend_slices(test_input)

    assert 'values != 1' in str(e)


def test_make_tile_url_template():
    test_input = 'www.google.com'
    expected_output = "www.google.com/{z_index}/0/{y_index}_{x_index}.jpg"

    assert make_tile_url_template(test_input) == expected_output


@pytest.fixture()
def edgelist():
    return get_fixture_data('small_graph.json')


@pytest.fixture
def nx_graph(edgelist):
    return nx.DiGraph(data=edgelist)


def test_get_nodes_between_root_only(nx_graph):
    expected_output = set(range(2, 10))

    assert get_nodes_between(nx_graph, 2) == expected_output


def test_get_nodes_between_two_leaves(nx_graph):
    expected_output = {2, 3, 4, 7, 8}

    assert get_nodes_between(nx_graph, 2, [4, 8]) == expected_output


def test_get_nodes_between_one_leaf(nx_graph):
    expected_output = {2, 3, 4}

    assert get_nodes_between(nx_graph, 2, [4]) == expected_output


def test_get_subarbor_node_infos(edgelist):
    node_parent = [(node, parent) for parent, node in edgelist]
    coords_xyz = [(node, 0, 0) for node, _ in node_parent]

    expected_output = [
        NodeInfo(node_id, x, y, z, None if parent_id is None else int(parent_id))
        for (node_id, parent_id), (x, y, z) in zip(node_parent, coords_xyz)
    ]

    real_output = get_subarbor_node_infos(node_parent, coords_xyz)

    assert set(expected_output) == set(real_output)


@pytest.fixture
def roi_xyz():
    return np.array([
        [10, 100, 1],
        [20, 200, 2]
    ])


def test_in_roi_central(roi_xyz):
    test_input = [15, 150, 1.5]

    assert in_roi(roi_xyz, test_input)


@pytest.mark.parametrize(['dim_out'], [(0, ), (1, ), (2, )])
@pytest.mark.parametrize(['direction'], [(1, ), (-1, )])
def test_in_roi_one_out(roi_xyz, dim_out, direction):
    test_input = [15, 150, 1.5]
    test_input[dim_out] = 10000 * direction

    assert not in_roi(roi_xyz, test_input)


@pytest.mark.parametrize(['border_on'], [(0, ), (1, )])
def test_in_roi_inclusive(roi_xyz, border_on):
    test_input = roi_xyz[border_on, :]

    assert in_roi(roi_xyz, test_input)


if __name__ == '__main__':
    pytest.main(['-v', os.path.realpath(__file__)])
