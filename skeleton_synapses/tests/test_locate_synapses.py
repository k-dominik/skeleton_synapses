import os

import numpy as np
import pytest
import mock

from skeleton_synapses.locate_synapses import (
    ensure_list, ensure_description_file, setup_files, assert_same_xy, write_output_image,
    slicing, mkdir_p
)

from fixtures import get_fixture_data, tmp_dir


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


def assert_contents(path, expected_contents):
    with open(path) as f:
        contents = f.read()

    assert contents == expected_contents


@pytest.fixture
def description_file(tmp_dir):
    description_path = os.path.join(tmp_dir, 'description.json')
    with open(description_path, 'w') as f:
        f.write('existing')
    yield description_path
    os.remove(description_path)


@pytest.fixture
def stack_describer_mock():
    catmaid = mock.Mock()
    catmaid.get_stack_description = mock.Mock(return_value='new')
    return catmaid


def test_ensure_description_file_created(tmp_dir, stack_describer_mock):
    description_path = os.path.join(tmp_dir, 'description.json')
    replaced = ensure_description_file(stack_describer_mock, description_path, None, None, force=False)

    assert replaced
    stack_describer_mock.get_stack_description.assert_called_once()
    assert_contents(description_path, '"new"')


def test_ensure_description_file_exists(description_file, stack_describer_mock):
    replaced = ensure_description_file(stack_describer_mock, description_file, None, None, force=False)

    assert not replaced
    stack_describer_mock.get_stack_description.assert_not_called()
    assert_contents(description_file, 'existing')


def test_ensure_description_file_overwritten(description_file, stack_describer_mock):
    replaced = ensure_description_file(stack_describer_mock, description_file, None, None, force=True)

    assert replaced
    stack_describer_mock.get_stack_description.assert_called_once()
    assert_contents(description_file, '"new"')


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
