import os
import json
import multiprocessing as mp
from Queue import Empty
import time

import geojson
import h5py
import numpy as np
import vigra
import pytest
import mock

from skeleton_synapses.locate_syn_catmaid import (
    create_label_volume, ensure_hdf5, nodes_to_tile_indexes, tile_index_to_bounds, square_bounds,
    node_locations_to_array, image_to_geojson, iterate_queue, commit_node_association_results_from_queue,
    HDF5_NAME, TileIndex, NeuronSegmenterOutput
)
from skeleton_synapses.skeleton_utils import NodeInfo

from skeleton_synapses.tests.fixtures import tmp_dir


TILE_SIZE = 512
MIN_RADIUS = 150
PROJECT_WORKFLOW = 1

TILE_CENTRE = int(TILE_SIZE / 2)

@pytest.fixture
def catmaid():
    catmaid = mock.Mock()
    catmaid.add_synapse_treenode_associations = mock.Mock()
    return catmaid


@pytest.fixture
def stack_info():
    return {
        'sid': 1,
        'translation': {
            'x': 0,
            'y': 0,
            'z': 0
        },
        'dimension': {
            'x': 10*TILE_SIZE,
            'y': 10*TILE_SIZE,
            'z': 5
        },
        'resolution': {
            'x': 1,
            'y': 1,
            'z': 1
        }
    }


def get_hdf5_path(hdf5_dir):
    return os.path.join(hdf5_dir, HDF5_NAME)


@pytest.fixture
def hdf5_file(tmp_dir):
    with h5py.File(get_hdf5_path(tmp_dir)) as f:
        f.attrs['is_old'] = True
        f.flush()
        yield f


def test_create_label_volume(hdf5_file, stack_info):
    dataset = create_label_volume(stack_info, hdf5_file, 'my_dataset')
    assert set(dataset.attrs).issuperset({'translation', 'dimension', 'resolution', 'axistags'})
    assert dataset.shape == tuple(stack_info['dimension'][key] for key in dataset.attrs['axistags'])


def test_create_label_volume_extra_dim(hdf5_file, stack_info):
    colour_dim = 3
    dataset = create_label_volume(stack_info, hdf5_file, 'my_dataset', colour_channels=colour_dim)
    assert set(dataset.attrs).issuperset({'translation', 'dimension', 'resolution', 'axistags'})
    assert dataset.shape == tuple(stack_info['dimension'].get(key, colour_dim) for key in dataset.attrs['axistags'])


def test_ensure_hdf5_new(stack_info, tmp_dir):
    hdf5_path = ensure_hdf5(stack_info, tmp_dir)
    assert os.path.isfile(hdf5_path)


def test_ensure_hdf5_datasets(stack_info, tmp_dir):
    hdf5_path = ensure_hdf5(stack_info, tmp_dir)
    with h5py.File(hdf5_path) as f:
        assert {'slice_labels', 'pixel_predictions'}.issubset(f)
        assert f.attrs['source_stack_id'] == stack_info['sid']


def test_ensure_hdf5_exists(stack_info, tmp_dir):
    with h5py.File(get_hdf5_path(tmp_dir)) as f:
        f.attrs['is_old'] = True

    hdf5_path = ensure_hdf5(stack_info, tmp_dir)

    with h5py.File(hdf5_path) as f:
        assert f.attrs.get('is_old', False)


def test_ensure_hdf5_force(stack_info, tmp_dir):
    pre_populated_path = get_hdf5_path(tmp_dir)
    with h5py.File(pre_populated_path) as f:
        f.attrs['is_old'] = True
        f.flush()

    hdf5_path = ensure_hdf5(stack_info, tmp_dir, force=True)
    assert hdf5_path == pre_populated_path

    with h5py.File(hdf5_path) as new_file:
        assert not new_file.attrs.get('is_old', False)

    paths = [os.path.join(tmp_dir, fname) for fname in os.listdir(tmp_dir)]
    assert len(paths) == 2
    backup_path = [path for path in paths if path != hdf5_path][0]

    with h5py.File(backup_path) as old_file:
        assert old_file.attrs.get('is_old', False)


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


def test_square_bounds():
    test_roi = np.array([
        [100, 100, 1],
        [150, 200, 1]
    ])
    expected_output = np.array([
        [75, 100, 1],
        [175, 200, 1]
    ])
    results = square_bounds(test_roi)
    assert np.allclose(results, expected_output)


def test_node_locations_to_array():
    tnid = '123'
    x, y = 3, 4
    shape = 5, 5
    node_locations = {
        tnid: {
            'treenode_id': tnid,
            'coords': {
                'x': x,
                'y': y
            }
        }
    }

    template_array = vigra.taggedView(np.random.random(shape), axistags='xy')
    output = node_locations_to_array(template_array, node_locations)

    assert output[x, y] == int(tnid)
    output[x, y] = -1
    assert np.allclose(np.ones(shape) * -1, output)


@pytest.fixture
def img_square():
    img = np.zeros((15, 15))
    img[5:10, 5:10] = 1
    return vigra.taggedView(img, axistags='xy')


def test_image_to_geojson(img_square):
    offset = 0, 0

    geo_str = image_to_geojson(img_square, *offset)
    polygon = geojson.Polygon.to_instance(json.loads(geo_str))

    assert polygon.is_valid
    assert len(polygon.coordinates) == 1

    # sets used so as not to depend on starting point
    coord_xy_set = {tuple(pair) for pair in polygon.coordinates[0]}
    expected_coords = {
        (4.5, 5.0), (4.5, 6.0), (4.5, 7.0), (4.5, 8.0), (4.5, 9.0),
        (5.0, 4.5), (5.0, 9.5),
        (6.0, 4.5), (6.0, 9.5),
        (7.0, 4.5), (7.0, 9.5),
        (8.0, 4.5), (8.0, 9.5),
        (9.0, 4.5), (9.0, 9.5),
        (9.5, 5.0), (9.5, 6.0), (9.5, 7.0), (9.5, 8.0), (9.5, 9.0)
    }
    assert coord_xy_set == expected_coords


def test_image_to_geojson_offset(img_square):
    x_offset, y_offset = 3, 4

    geo_str = image_to_geojson(img_square, x_offset, y_offset)
    polygon = geojson.Polygon.to_instance(json.loads(geo_str))

    assert polygon.is_valid
    assert len(polygon.coordinates) == 1

    # sets used so as not to depend on starting point
    coord_xy_set = {tuple(pair) for pair in polygon.coordinates[0]}
    expected_coords = {
        (7.5, 9.0), (7.5, 10.0), (7.5, 11.0), (7.5, 12.0), (7.5, 13.0),
        (8.0, 8.5), (8.0, 13.5),
        (9.0, 8.5), (9.0, 13.5),
        (10.0, 8.5), (10.0, 13.5),
        (11.0, 8.5), (11.0, 13.5),
        (12.0, 8.5), (12.0, 13.5),
        (12.5, 9.0), (12.5, 10.0), (12.5, 11.0), (12.5, 12.0), (12.5, 13.0)
    }
    assert coord_xy_set == expected_coords


def test_image_to_geojson_check_xy(img_square):
    offsets = 0, 0
    img_rect = img_square.copy()
    img_rect[3:5, 5:10] = 1
    geo_str = image_to_geojson(img_rect, *offsets)
    polygon = geojson.Polygon.to_instance(json.loads(geo_str))

    assert polygon.is_valid
    assert len(polygon.coordinates) == 1

    # sets used so as not to depend on starting point
    coord_xy_set = {tuple(pair) for pair in polygon.coordinates[0]}
    expected_coords = {
        (2.5, 5.0), (2.5, 6.0), (2.5, 7.0), (2.5, 8.0), (2.5, 9.0),
        (3.0, 4.5), (3.0, 9.5),
        (4.0, 4.5), (4.0, 9.5),
        (5.0, 4.5), (5.0, 9.5),
        (6.0, 4.5), (6.0, 9.5),
        (7.0, 4.5), (7.0, 9.5),
        (8.0, 4.5), (8.0, 9.5),
        (9.0, 4.5), (9.0, 9.5),
        (9.5, 5.0), (9.5, 6.0), (9.5, 7.0), (9.5, 8.0), (9.5, 9.0)
    }
    assert coord_xy_set == expected_coords


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
    commit_node_association_results_from_queue(queue, len(item_chunks), PROJECT_WORKFLOW, catmaid)

    catmaid.add_synapse_treenode_associations.assert_called_once_with(
        expected_args, PROJECT_WORKFLOW
    )