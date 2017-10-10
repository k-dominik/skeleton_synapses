from __future__ import division
import json
import os

import mock
import pytest
import geojson
import numpy as np

from skeleton_synapses.locate_syn_catmaid import (
    image_to_geojson, get_synapse_uncertainty, get_synapse_slice_size_centroid, synapse_slices_to_data, submit_synapse_slice_data,
    remap_synapse_slices,
)

from skeleton_synapses.tests.fixtures import get_fixture_data, img_square, img_2, pixel_pred, tmp_dir


ID_MAPPING_2 = {1: 10, 2: 20}


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


def test_get_synapse_uncertainty():
    flat_predictions = np.array([
        [0.1, 0.2, 0.4],
        [0.4, 0.8, 0.5],
        [0.9, 0.8, 0.7]
    ])
    expected_output = 1 - 0.2
    output = get_synapse_uncertainty(flat_predictions)
    assert output == expected_output


def test_get_synapse_slice_size_centroid(img_square):
    x_offset = y_offset = 0
    binary_arr_xy = img_square == img_square.max()
    size_px, x_centroid_px, y_centroid_px = get_synapse_slice_size_centroid(
        binary_arr_xy, x_offset, y_offset
    )
    assert size_px == 25
    assert x_centroid_px == img_square.shape[0]//2
    assert y_centroid_px == img_square.shape[1]//2


def test_get_synapse_slice_size_centroid_offset(img_square):
    x_offset, y_offset = 3, 4
    label = 1
    binary_arr_xy = img_square == label
    size_px, x_centroid_px, y_centroid_px = get_synapse_slice_size_centroid(
        binary_arr_xy, x_offset, y_offset
    )
    assert size_px == 25
    assert x_centroid_px == img_square.shape[0]//2 + x_offset
    assert y_centroid_px == img_square.shape[1]//2 + y_offset


def test_synapse_slices_to_data(img_square, pixel_pred):
    x_offset = y_offset = 0
    data = synapse_slices_to_data(pixel_pred, img_square, x_offset, y_offset)

    assert len(data) == 1
    datum = data[0]
    assert datum['id'] == 1
    assert datum['size_px'] == 25
    assert datum['uncertainty'] < 1


def test_synapse_slices_to_data_multi(img_2, pixel_pred):
    x_offset = y_offset = 0
    data = synapse_slices_to_data(pixel_pred, img_2, x_offset, y_offset)

    assert len(data) == 2
    new_datum = data[1]
    assert new_datum['id'] == 2
    assert new_datum['size_px'] == 9
    assert new_datum['uncertainty'] != data[0]['uncertainty']


@pytest.fixture
def catmaid():
    catmaid = mock.Mock()
    catmaid.add_synapse_slices_to_tile = mock.Mock(return_value=ID_MAPPING_2)
    return catmaid


def test_submit_synapse_slice_data(img_2, pixel_pred, catmaid):
    bounds_xyz = np.array([
        [0, 0, 0],
        [15, 15, 1]
    ])
    id_mapping = submit_synapse_slice_data(bounds_xyz, pixel_pred, img_2, 'tile_idx', catmaid, 'workflow_id')
    catmaid.add_synapse_slices_to_tile.assert_called_once()
    assert id_mapping == ID_MAPPING_2


def test_remap_synapse_slices(img_2):
    output = remap_synapse_slices(img_2, ID_MAPPING_2)

    assert np.allclose(output == 1, img_2 == 0)
    assert np.allclose(output == 10, img_2 == 1)
    assert np.allclose(output == 20, img_2 == 2)


if __name__ == '__main__':
    pytest.main(['-v', os.path.realpath(__file__)])
