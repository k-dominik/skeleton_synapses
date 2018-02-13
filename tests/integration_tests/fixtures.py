import shutil
from itertools import cycle

import pytest

try:
    import mock
except ImportError:
    from unittest import mock

from tests.context import skeleton_synapses
from skeleton_synapses.helpers.files import Paths
from skeleton_synapses.ilastik_utils.projects import setup_classifier, setup_multicut
from skeleton_synapses.parallel.process import SynapseDetectionProcess
from skeleton_synapses.helpers.roi import tile_index_to_bounds
from skeleton_synapses.dto import TileIndex

from tests.integration_tests.constants import STACK_ID, TILE_SIZE, Z_MIN, Z_MAX, Y, X
from tests.fixtures import tmp_dir
from tests.constants import noop


def opt_fixture_factory(option_name):
    @pytest.fixture
    def opt_fixture(request):
        arg = request.config.getoption("--{}".format(option_name))
        if arg is None:
            pytest.skip('No --{} supplied, skipping'.format(option_name))
        else:
            return arg

    return opt_fixture


credentials_path = opt_fixture_factory('credentials_path')
input_dir = opt_fixture_factory('input_dir')


# @pytest.fixture
# def clean_input_dir(input_dir, tmp_dir):
#     shutil.copytree(input_dir, tmp_dir)
#     return tmp_dir


@pytest.fixture
def paths(credentials_path, input_dir, tmp_dir):
    paths_obj = Paths(credentials_path, input_dir, tmp_dir)
    paths_obj.initialise(None, STACK_ID)
    return paths_obj


@pytest.fixture
def opPixelClassification(paths):
    return setup_classifier(paths.description_json, paths.autocontext_ilp)


@pytest.fixture
def multicut_shell(paths):
    return setup_multicut(paths.multicut_ilp)


@pytest.fixture
def synapse_detection_process(paths, opPixelClassification):
    """Only use detect_synapses method on this"""
    p = SynapseDetectionProcess(None, None, paths, TILE_SIZE)
    p.start = noop
    p.execute = noop
    p.run = noop
    p.opPixelClassification = opPixelClassification
    return p


@pytest.fixture
def tile_index_generator():
    def gen(last):
        for idx, z in enumerate(cycle(range(Z_MIN, Z_MAX))):
            if idx >= last:
                raise StopIteration
            yield TileIndex(z, Y, X)

    return gen


@pytest.fixture
def roi_generator(tile_index_generator):
    def gen(last):
        for tile_idx in tile_index_generator(last):
            yield tile_index_to_bounds(tile_idx, TILE_SIZE)

    return gen
