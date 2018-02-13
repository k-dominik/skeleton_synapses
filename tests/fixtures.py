import os
import json
import tempfile
import shutil

try:
    import mock
except ImportError:
    from unittest import mock

import pytest
import numpy as np
import vigra

from tests.constants import PROJECT_ID, BASE_URL

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), 'fixture_data')


def get_fixture_data(rel_path, dtype=None, axistags=None):
    path = os.path.join(FIXTURE_DIR, rel_path)
    ext = os.path.splitext(path)[1]

    if ext == '.json':
        with open(path) as f:
            return json.load(f)
    elif ext == '.npy':
        data = np.load(path).astype(dtype)
    elif ext == '.csv':
        data = np.loadtxt(path, dtype=dtype)
    else:
        raise ValueError('Unknown extension {} for fixture at path {}'.format(ext, path))

    if dtype:
        data = data.astype(dtype)
    if axistags:
        data = vigra.taggedView(data, axistags=axistags)
    return data


@pytest.fixture
def tmp_dir(request):
    path = tempfile.mkdtemp(suffix='{}.{}'.format(request.module.__name__, request.function.__name__))
    assert len(os.listdir(path)) == 0
    yield path
    shutil.rmtree(path, True)


@pytest.fixture
def img_square():
    return get_fixture_data('img_square.csv', dtype=int, axistags='xy')


@pytest.fixture
def pixel_pred():
    return get_fixture_data('predictions.npy', axistags='xyc')


@pytest.fixture
def img_2(img_square):
    im = img_square.copy()
    im[11:14, 11:14] = 2
    return im


@pytest.fixture
def compact_detail():
    return get_fixture_data('compact_detail.json')


@pytest.fixture
def synapses_near_skeleton():
    return get_fixture_data('synapses_near_skeleton.json')


@pytest.fixture
def catmaid_mock():
    catmaid = mock.Mock()
    catmaid.project_id = PROJECT_ID
    catmaid.base_url = BASE_URL
    return catmaid
