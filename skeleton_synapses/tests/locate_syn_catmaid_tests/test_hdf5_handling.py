import os

import pytest
import h5py

from skeleton_synapses.locate_syn_catmaid import (
    create_label_volume, ensure_hdf5, HDF5_NAME
)

from skeleton_synapses.tests.fixtures import tmp_dir
from skeleton_synapses.tests.constants import TILE_SIZE


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