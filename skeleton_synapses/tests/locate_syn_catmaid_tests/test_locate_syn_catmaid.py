import os

import numpy as np
import vigra
import pytest

from skeleton_synapses.helpers.segmentation import node_locations_to_array
from skeleton_synapses.helpers.roi import square_bounds


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


if __name__ == '__main__':
    pytest.main(['-v', os.path.realpath(__file__)])
