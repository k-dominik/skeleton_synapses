import os
import logging
import multiprocessing as mp
import time

from lazyflow.request import Request
from lazyflow.utility.timer import Timer

from skeleton_synapses.helpers.files import cached_synapses_predictions_for_roi, dump_images
from skeleton_synapses.helpers.roi import tile_index_to_bounds
from skeleton_synapses.helpers.segmentation import (
    get_synapse_segment_overlaps, get_node_associations, node_locations_to_array
)
from skeleton_synapses.dto import SynapseDetectionOutput
from skeleton_synapses.ilastik_utils.projects import setup_classifier, setup_classifier_and_multicut
from skeleton_synapses.ilastik_utils.analyse import (
    fetch_and_predict, label_synapses, raw_data_for_roi, segmentation_for_img
)
from skeleton_synapses.parallel.base_classes import DebuggableProcess, LeakyProcess
from skeleton_synapses.parallel.progress_server import QueueMonitorThread, DummyThread


logger = logging.getLogger(__name__)


class CaretakerProcess(DebuggableProcess):
    """
    Process which takes care of spawning a process which may have memory leaks, pruning it when it terminates (for
    example, if it stops itself due to taking up too much memory), and starting a new one if there are still items
    remaining in the input queue.
    """

    def __init__(
            self, constructor, input_queue, args_tuple=(), kwargs_dict=None, debug=False, name=None
    ):
        """

        Parameters
        ----------
        constructor : LeakyProcess constructor
        input_queue : multiprocessing.Queue
        args_tuple : array_like
        kwargs_dict : dict
        debug : bool
        """
        super(CaretakerProcess, self).__init__(debug, name=name)
        self.constructor = constructor
        self.input_queue = input_queue
        self.args_tuple = args_tuple
        self.kwargs_dict = kwargs_dict or dict()
        self.kwargs_dict['debug'] = debug

        self.inner_process_counter = 0

    def run(self):
        logger = self.inner_logger
        while not self.input_queue.empty():
            logger.debug(
                'Starting new inner {} process with {} inputs remaining'.format(
                    self.constructor.__name__, self.input_queue.qsize()
                )
            )
            name = '{}{}({})'.format(self.constructor.__name__, self.inner_process_counter, self.name)
            self.inner_process_counter += 1
            kwargs = self.kwargs_dict.copy()
            kwargs['name'] = name
            inner_process = self.constructor(
                self.input_queue, *self.args_tuple, **kwargs
            )
            inner_process.start()
            logger.debug('Started {} with {} inputs remaining'.format(inner_process.name, self.input_queue.qsize()))
            inner_process.join()
            logger.debug('Stopped {} with {} inputs remaining'.format(inner_process.name, self.input_queue.qsize()))
            del inner_process


class SynapseDetectionProcess(LeakyProcess):
    def __init__(self, input_queue, output_queue, paths, tile_size, debug=False, name=None):
        super(SynapseDetectionProcess, self).__init__(input_queue, debug, name)
        self.output_queue = output_queue

        self.tile_size = tile_size

        self.opPixelClassification = None

        self.paths = paths
        self.setup_args = paths.description_json, paths.autocontext_ilp

    def setup(self):
        self.opPixelClassification = setup_classifier(*self.setup_args)
        Request.reset_thread_pool(1)  # todo: set to 0?

    def execute(self):
        tile_idx = self.input_queue.get()

        self.inner_logger.debug(
            "Addressing tile {}; {} tiles remaining".format(tile_idx, self.input_queue.qsize()))

        with Timer() as timer:
            roi_xyz = tile_index_to_bounds(tile_idx, self.tile_size)

            # GET AND CLASSIFY PIXELS
            raw_xy, predictions_xyc = fetch_and_predict(roi_xyz, self.opPixelClassification)

            # DETECT SYNAPSES
            synapse_cc_xy = label_synapses(predictions_xyc)
            logging.getLogger(self.inner_logger.name + '.timing').info("NODE TIMER: {}".format(timer.seconds()))

        self.inner_logger.debug(
            "Detected synapses in tile {}; {} tiles remaining".format(tile_idx, self.input_queue.qsize())
        )

        self.output_queue.put(SynapseDetectionOutput(tile_idx, predictions_xyc, synapse_cc_xy))

        if int(os.getenv('SS_DEBUG_IMAGES', 0)):
            path = os.path.join(
                    self.paths.debug_tile_dir, 'x{}-y{}-z{}.hdf5'.format(tile_idx.x_idx, tile_idx.y_idx, tile_idx.z_idx)
            )
            dump_images(path, roi_xyz, raw=raw_xy, synapse_cc=synapse_cc_xy, predictions=predictions_xyc)


# class NeuronSegmenterProcess(DebuggableProcess):
class SkeletonAssociationProcess(LeakyProcess):
    """
    Process which creates its own pixel classifier and multicut workflow, pulls jobs from one queue and returns
    outputs to another queue.
    """
    def __init__(self, input_queue, output_queue, paths, catmaid, debug=False, name=None):
        """

        Parameters
        ----------
        input_queue : mp.Queue
        output_queue : mp.Queue
        paths : skeleton_synapses.helpers.files.Paths
        debug : bool
            Whether to instantiate a serial version for debugging purposes
        name : str
        """
        super(SkeletonAssociationProcess, self).__init__(input_queue, debug, name)
        # super(NeuronSegmenterProcess, self).__init__(debug)
        self.input_queue = input_queue
        self.output_queue = output_queue

        self.paths = paths
        self.hdf5_path = paths.output_hdf5

        self.opPixelClassification = self.multicut_shell = None
        self.setup_args = paths.description_json, paths.autocontext_ilp, paths.multicut_ilp
        self.catmaid = catmaid

    # for debugging
    # def run(self):
    #     self.setup()
    #
    #     while not self.input_queue.empty():
    #         self.execute()

    def setup(self):
        self.inner_logger.debug('Setting up opPixelClassification and multicut_shell...')
        # todo: replace opPixelClassification with catpy tile-getter
        self.opPixelClassification, self.multicut_shell = setup_classifier_and_multicut(
            *self.setup_args
        )
        self.inner_logger.debug('opPixelClassification and multicut_shell set up')

        Request.reset_thread_pool(1)

    def execute(self):
        # todo: test (needs refactors)
        logger.debug('Waiting for item')
        roi_xyz, synapse_slice_ids, synapse_object_id = self.input_queue.get()
        self.inner_logger.debug("Addressing ROI {}; {} ROIs remaining".format(roi_xyz, self.input_queue.qsize()))

        with Timer() as node_timer:
            raw_xy = raw_data_for_roi(roi_xyz, self.opPixelClassification)
            synapse_cc_xy, predictions_xyc = cached_synapses_predictions_for_roi(roi_xyz, self.hdf5_path)

            log_str = 'Image shapes: \n\tRaw {}\n\tSynapse_cc {}\n\tPredictions {}'.format(
                raw_xy.shape, synapse_cc_xy.shape, predictions_xyc.shape
            )
            self.inner_logger.debug(log_str)

            segmentation_xy = segmentation_for_img(raw_xy, predictions_xyc, self.multicut_shell.workflow)

            overlapping_segments = get_synapse_segment_overlaps(synapse_cc_xy, segmentation_xy, synapse_slice_ids)
            self.inner_logger.debug('Local segment: synapse slice overlaps found: \n{}'.format(overlapping_segments))

            if len(overlapping_segments) < 2:  # synapse is only in 1 segment
                self.inner_logger.debug(
                    'Synapse slice IDs {} in ROI {} are only in 1 neuron'.format(synapse_slice_ids, roi_xyz)
                )
                self.output_queue.put([])  # outputs will be empty
                return

            node_locations = self.catmaid.get_nodes_in_roi(roi_xyz, self.catmaid.stack_id)
            if len(node_locations) == 0:
                self.inner_logger.debug('ROI {} has no nodes'.format(roi_xyz))

            node_segmenter_outputs = get_node_associations(
                synapse_cc_xy, segmentation_xy, node_locations, overlapping_segments
            )

            self.output_queue.put(node_segmenter_outputs)

            logging.getLogger(self.inner_logger.name + '.timing').info("TILE TIMER: {}".format(node_timer.seconds()))

            if int(os.getenv('SS_DEBUG_IMAGES', 0)):
                node_locations_arr = node_locations_to_array(segmentation_xy.shape, node_locations)
                path = os.path.join(
                    self.paths.debug_synapse_dir, '{}_{}.hdf5'.format(synapse_object_id, roi_xyz[0, 2])
                )
                dump_images(
                    path, roi_xyz, raw=raw_xy, synapse_cc=synapse_cc_xy, predictions=predictions_xyc,
                    segmentation=segmentation_xy, node_locations=node_locations_arr
                )


class ProcessRunner(object):
    def __init__(self, input_queue, constructor, setup_args, threads, monitor_kwargs=None):
        self.input_queue = input_queue
        self.constructor = constructor
        self.setup_args = setup_args
        self.threads = threads

        self.output_queue = mp.Queue()
        self.containers = [
            CaretakerProcess(
                constructor, input_queue, setup_args, name='CaretakerProcess{}'.format(idx)
            )
            for idx in range(threads)
        ]

        if monitor_kwargs:
            if not isinstance(monitor_kwargs, dict):
                monitor_kwargs = dict()

            self.monitor = QueueMonitorThread(input_queue, **monitor_kwargs)
        else:
            self.monitor = DummyThread()

    def __enter__(self):
        while self.input_queue.qsize() < self.threads:
            time.sleep(0.1)

        self.monitor.start()
        for container in self.containers:
            container.start()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for container in self.containers:
            container.join()
        self.monitor.stop()
