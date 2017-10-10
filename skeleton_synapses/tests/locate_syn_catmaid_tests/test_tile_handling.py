import numpy as np

from skeleton_synapses.locate_syn_catmaid import (
    nodes_to_tile_indexes, TileIndex, tile_index_to_bounds
)
from skeleton_synapses.skeleton_utils import NodeInfo

from skeleton_synapses.tests.constants import TILE_SIZE, TILE_CENTRE, MIN_RADIUS


def test_nodes_to_tile_indexes_simple():
    node_infos = [NodeInfo(id=1, z_px=0, y_px=TILE_CENTRE, x_px=TILE_CENTRE, parent_id=None)]

    tile_set = nodes_to_tile_indexes(node_infos, tile_size=TILE_SIZE, minimum_radius=MIN_RADIUS)
    expected_set = {TileIndex(0, 0, 0)}
    assert tile_set == expected_set


def test_nodes_to_tile_indexes_boundary():
    node_infos = [NodeInfo(id=1, z_px=0, y_px=TILE_SIZE, x_px=TILE_SIZE, parent_id=None)]

    tile_set = nodes_to_tile_indexes(node_infos, tile_size=TILE_SIZE, minimum_radius=MIN_RADIUS)
    expected_set = {
        TileIndex(z_idx=0, y_idx=0, x_idx=0),
        TileIndex(z_idx=0, y_idx=0, x_idx=1),
        TileIndex(z_idx=0, y_idx=1, x_idx=0),
        TileIndex(z_idx=0, y_idx=1, x_idx=1)
    }
    assert tile_set == expected_set


def test_nodes_to_tile_indexes_same():
    node_infos = [
        NodeInfo(id=1, z_px=0, y_px=TILE_CENTRE, x_px=TILE_CENTRE, parent_id=None),
        NodeInfo(id=2, z_px=0, y_px=TILE_CENTRE + 1, x_px=TILE_CENTRE + 1, parent_id=1)
    ]

    tile_set = nodes_to_tile_indexes(node_infos, tile_size=TILE_SIZE, minimum_radius=MIN_RADIUS)
    expected_set = {TileIndex(0, 0, 0)}
    assert tile_set == expected_set


def test_tile_index_to_bounds():
    tile_index = TileIndex(z_idx=2, y_idx=3, x_idx=4)
    tile_size = 10
    expected_response = np.array([
        [40, 30, 2],
        [50, 40, 3]
    ])
    output_bounds = tile_index_to_bounds(tile_index, tile_size)
    assert np.allclose(expected_response, output_bounds)