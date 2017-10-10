import json

import pytest
import geojson
import numpy as np
import vigra

from skeleton_synapses.locate_syn_catmaid import (
    image_to_geojson, get_synapse_uncertainty, get_synapse_slice_size_centroid, synapse_slices_to_data, submit_synapse_slice_data,
    remap_synapse_slices,
)

from skeleton_synapses.tests.fixtures import tmp_dir


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


def test_get_synapse_uncertainty():
    flat_predictions = np.array([
        [0.1, 0.2, 0.4],
        [0.4, 0.8, 0.5],
        [0.9, 0.8, 0.7]
    ])
    expected_output = 1 - 0.2
    output = get_synapse_uncertainty(flat_predictions)
    assert output == expected_output
