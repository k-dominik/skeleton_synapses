import os

import numpy as np
import pytest
import mock
import vigra

from skeleton_synapses.locate_synapses import (
    ensure_list, ensure_description_file, ensure_skel_output_dirs, get_algo_notes,
    are_same_xy, slicing, mkdir_p
)

from fixtures import tmp_dir


def test_ensure_list_list():
    test_input = [1, 2, 3]

    output = ensure_list(test_input)
    assert output == test_input


def test_ensure_list_not_list():
    test_input = 1
    expected_output = [test_input]

    output = ensure_list(test_input)
    assert output == expected_output


def test_ensure_list_str():
    test_input = 'potato'
    expected_output = [test_input]

    output = ensure_list(test_input)
    assert output == expected_output


def assert_contains(path, expected_contents):
    with open(path) as f:
        real_contents = f.read()

    assert expected_contents in real_contents


@pytest.fixture
def description_file(tmp_dir):
    description_path = os.path.join(tmp_dir, 'description.json')
    with open(description_path, 'w') as f:
        f.write('"existing"')
    yield description_path
    os.remove(description_path)


@pytest.fixture
def catmaid():
    catmaid = mock.Mock()
    catmaid.get_stack_description = mock.Mock(return_value='new stack description')
    catmaid.get_transformed_treenode_and_connector_geometry = mock.Mock(return_value='new geometry')
    return catmaid


def test_ensure_description_file_created(catmaid, tmp_dir):
    description_path = os.path.join(tmp_dir, 'description.json')
    replaced = ensure_description_file(catmaid, description_path, None, None, force=False)

    assert replaced
    catmaid.get_stack_description.assert_called_once()
    assert_contains(description_path, 'new')


def test_ensure_description_file_exists(catmaid, description_file):
    replaced = ensure_description_file(catmaid, description_file, None, None, force=False)

    assert not replaced
    catmaid.get_stack_description.assert_not_called()
    assert_contains(description_file, 'existing')


def test_ensure_description_file_overwritten(catmaid, description_file):
    replaced = ensure_description_file(catmaid, description_file, None, None, force=True)

    assert replaced
    catmaid.get_stack_description.assert_called_once()
    assert_contains(description_file, 'new')


def assert_skel_output_dirs(output_dirs, skel_ids, contents='new'):
    assert len(output_dirs) == len(skel_ids)
    for output_dir, skel_id in zip(output_dirs, skel_ids):
        assert output_dir.endswith(str(skel_id))
        assert_contains(os.path.join(output_dir, 'tree_geometry.json'), contents)


@pytest.fixture
def skel_ids():
    return 1, 2


def test_ensure_skel_output_dirs_new(skel_ids, catmaid, tmp_dir):
    output_dirs = ensure_skel_output_dirs(tmp_dir, skel_ids, catmaid, None, force=False)
    assert_skel_output_dirs(output_dirs, skel_ids)


@pytest.fixture
def skel_populated_tmp(skel_ids, tmp_dir):
    skel_path = os.path.join(tmp_dir, 'skeletons')
    os.mkdir(skel_path)
    for skel_id in skel_ids:
        path = os.path.join(skel_path, str(skel_id))
        os.mkdir(path)
        assert os.path.isdir(path)

        geom_path = os.path.join(path, 'tree_geometry.json')
        with open(geom_path, 'w') as f:
            f.write('existing')
        assert os.path.isfile(geom_path)

        other_path = os.path.join(path, 'OTHER')
        with open(other_path, 'w') as f:
            f.write('existing')
        assert os.path.isfile(other_path)

    return tmp_dir


def test_ensure_skel_output_dirs_exist(catmaid, skel_ids, skel_populated_tmp):
    output_dirs = ensure_skel_output_dirs(skel_populated_tmp, skel_ids, catmaid, None, force=False)

    assert_skel_output_dirs(output_dirs, skel_ids)
    for output_dir in output_dirs:
        assert 'OTHER' in os.listdir(output_dir)


def test_ensure_skel_output_dirs_force(catmaid, skel_ids, skel_populated_tmp):
    output_dirs = ensure_skel_output_dirs(skel_populated_tmp, skel_ids, catmaid, None, force=True)

    assert_skel_output_dirs(output_dirs, skel_ids)
    for output_dir in output_dirs:
        assert 'OTHER' not in os.listdir(output_dir)


def test_get_algo_notes_exist(tmp_dir):
    with open(os.path.join(tmp_dir, 'algorithm_notes.json'), 'w') as f:
        f.write('"existing"')

    output = get_algo_notes(tmp_dir)
    assert 'existing' in output


def test_get_algo_notes_none(tmp_dir):
    output = get_algo_notes(tmp_dir)
    assert {'synapse_detection', 'skeleton_association'} == set(output)


@pytest.fixture
def vigra_arrs():
    return [
        vigra.taggedView(np.random.random((5, 5)), axistags='xy')
        for _ in range(3)
    ]


def test_assert_same_xy_same(vigra_arrs):
    assert are_same_xy(*vigra_arrs)


def test_assert_same_xy_add_dim(vigra_arrs):
    vigra_arrs += [
        vigra.taggedView(np.random.random((5, 5, 3)), axistags='xyc'),
        vigra.taggedView(np.random.random((5, 5, 5)), axistags='xyz'),
    ]
    assert are_same_xy(*vigra_arrs)


def test_assert_same_xy_different_shape(vigra_arrs):
    vigra_arrs += [
        vigra.taggedView(np.random.random((5, 6)), axistags='xy')
    ]
    assert not are_same_xy(*vigra_arrs)


def test_assert_same_xy_different_axistags(vigra_arrs):
    vigra_arrs += [
        vigra.taggedView(np.random.random((5, 6)), axistags='yx')
    ]
    assert not are_same_xy(*vigra_arrs)


@pytest.skip('write_output_image used for debugging only')
def test_write_output_image():
    pass


def test_mkdir_p_single(tmp_dir):
    dir_path = os.path.join(tmp_dir, 'my_dir')
    assert not os.path.isdir(dir_path)
    mkdir_p(dir_path)
    assert os.path.isdir(dir_path)


def test_mkdir_p_multi(tmp_dir):
    dir_path = os.path.join(tmp_dir, 'my_outer_dir', 'my_inner_dir')
    assert not os.path.isdir(dir_path)
    mkdir_p(dir_path)
    assert os.path.isdir(dir_path)


def test_mkdir_p_exists(tmp_dir):
    dir_path = os.path.join(tmp_dir, 'my_existing_dir')
    os.mkdir(dir_path)
    assert os.path.isdir(dir_path)
    mkdir_p(dir_path)


def test_slicing():
    input_roi = np.array([
        [10, 100, 1],
        [20, 200, 2]
    ])
    expected_output = (slice(10, 20), slice(100, 200), slice(1, 2))

    slices = slicing(input_roi)
    assert slices == expected_output


if __name__ == '__main__':
    pytest.main(['-v', os.path.realpath(__file__)])
