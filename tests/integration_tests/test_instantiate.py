import logging
import psutil
import numpy as np

import pytest

from tests.context import skeleton_synapses

from skeleton_synapses.ilastik_utils.analyse import fetch_and_predict

from tests.integration_tests.fixtures import (
    opPixelClassification, paths, credentials_path, input_dir, tmp_dir, synapse_detection_process,
    tile_index_generator, roi_generator
)


@pytest.mark.skip
def test_instantiate_opPixelClassification(opPixelClassification):
    assert opPixelClassification


class MemoryTracker(object):
    def __init__(self):
        self.process = psutil.Process()
        self.baseline = self.ram_MB()

    @staticmethod
    def B_to_MB(n_bytes):
        return n_bytes / 1024 / 1024

    def ram_MB(self):
        return self.B_to_MB(self.process.memory_info().rss)

    def ram_increase_MB(self):
        return self.ram_MB() - self.baseline


def test_predictor_predicts(opPixelClassification, roi_generator):
    roi_xyz = next(roi_generator(1))
    raw_xy, predictions_xy = fetch_and_predict(roi_xyz, opPixelClassification)
    assert len(np.unique(predictions_xy)) > 1


@pytest.mark.skip
def test_classifier_multiple(synapse_detection_process, tile_index_generator):
    """

    Parameters
    ----------
    synapse_detection_process : skeleton_synapses.parallel.process.SynapseDetectionProcess
    tile_index_generator

    Returns
    -------

    """
    logging.getLogger().level = logging.DEBUG
    mem_tracker = None
    for idx, tile_idx in enumerate(tile_index_generator(10)):
        print(tile_idx)
        output = synapse_detection_process.detect_synapses(tile_idx)
        assert len(np.unique(output.predictions_xyc)) > 1
        if mem_tracker is None:
            mem_tracker = MemoryTracker()
        ram_increase = mem_tracker.ram_increase_MB()
        print(ram_increase)
        assert ram_increase < 200, 'RAM increase was {} after {} iterations'.format(ram_increase, idx)



