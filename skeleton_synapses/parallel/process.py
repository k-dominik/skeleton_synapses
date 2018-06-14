import os
import logging
import multiprocessing as mp
import time

from lazyflow.request import Request
from lazyflow.utility.timer import Timer

from skeleton_synapses.ilastik_utils.analyse import detect_synapses, associate_skeletons
from skeleton_synapses.helpers.files import dump_images
from skeleton_synapses.helpers.roi import tile_index_to_bounds
from skeleton_synapses.ilastik_utils.projects import setup_classifier, setup_classifier_and_multicut
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
            self, constructor, input_queue, output_queue, args_tuple=(), kwargs_dict=None, debug=False, name=None
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
        self.output_queue = output_queue
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
                self.input_queue, self.output_queue, *self.args_tuple, **kwargs
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
        super(SynapseDetectionProcess, self).setup()
        self.opPixelClassification = setup_classifier(*self.setup_args)
        Request.reset_thread_pool(1)  # todo: set to 0?

    def execute(self):
        super(SynapseDetectionProcess, self).execute()
        tile_idx = self.input_queue.get()

        self.inner_logger.debug(
            "Addressing tile {}; {} tiles remaining".format(tile_idx, self.input_queue.qsize()))

        output = self.detect_synapses(tile_idx, debug=bool(int(os.getenv('SS_DEBUG_IMAGES', 0))))

        self.inner_logger.debug(
            "Detected synapses in tile {}; {} tiles remaining".format(tile_idx, self.input_queue.qsize())
        )

        self.output_queue.put(output)

    def detect_synapses(self, tile_idx, debug=False):
        self.inner_logger.debug("detect_synapses called")
        with Timer() as timer:
            output = detect_synapses(self.tile_size, self.opPixelClassification, tile_idx)

        logging.getLogger(self.inner_logger.name + '.timing').info("NODE TIMER: {}".format(timer.seconds()))

        if debug:
            roi_xyz = tile_index_to_bounds(tile_idx, self.tile_size)
            path = os.path.join(
                self.paths.debug_tile_dir, 'x{}-y{}-z{}.hdf5'.format(tile_idx.x_idx, tile_idx.y_idx, tile_idx.z_idx)
            )
            dump_images(path, roi_xyz, synapse_cc=output.synapse_cc_xy, predictions=output.predictions_xyc)

        return output


class SynapseDetectionProcessNew(SynapseDetectionProcess):
    def __init__(self, input_queue, output_queue, paths, tile_size, opPixelClassification, debug=False, name=None):
        super(SynapseDetectionProcessNew, self).__init__(
            input_queue, output_queue, paths, tile_size, debug=debug, name=name
        )
        self.opPixelClassification = opPixelClassification

    def setup(self):
        # Request.reset_thread_pool(1)
        pass


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
        super(SkeletonAssociationProcess, self).setup()
        self.inner_logger.debug('Setting up opPixelClassification and multicut_shell...')
        # todo: replace opPixelClassification with catpy tile-getter
        self.opPixelClassification, self.multicut_shell = setup_classifier_and_multicut(
            *self.setup_args
        )
        self.inner_logger.debug('opPixelClassification and multicut_shell set up')

        Request.reset_thread_pool(1)

    def execute(self):
        super(SkeletonAssociationProcess, self).execute()
        # todo: test (needs refactors)
        logger.debug('Waiting for item')
        skeleton_association_input = self.input_queue.get()
        self.inner_logger.debug("Addressing ROI {}; {} ROIs remaining".format(skeleton_association_input.roi_xyz, self.input_queue.qsize()))

        node_segmenter_outputs = self.associate_skeletons(
            skeleton_association_input, debug=bool(int(os.getenv('SS_DEBUG_IMAGES', 0)))
        )

        self.output_queue.put(node_segmenter_outputs)

    def associate_skeletons(self, skeleton_association_input, debug=False):
        with Timer() as node_timer:
            node_segmenter_outputs = associate_skeletons(
                self.hdf5_path, self.opPixelClassification, self.multicut_shell, self.catmaid,
                skeleton_association_input
            )

        logging.getLogger(self.inner_logger.name + '.timing').info("TILE TIMER: {}".format(node_timer.seconds()))

        # if debug:
        #     node_locations_arr = node_locations_to_array(segmentation_xy.shape, node_locations)
        #     path = os.path.join(
        #         self.paths.debug_synapse_dir, '{}_{}.hdf5'.format(synapse_object_id, roi_xyz[0, 2])
        #     )
        #     dump_images(
        #         path, roi_xyz, raw=raw_xy, synapse_cc=synapse_cc_xy, predictions=predictions_xyc,
        #         segmentation=segmentation_xy, node_locations=node_locations_arr
        #     )

        return node_segmenter_outputs


class ProcessRunner(object):
    def __init__(self, input_queue, constructor, setup_args, threads, monitor_kwargs=None):
        self.input_queue = input_queue
        self.constructor = constructor
        self.setup_args = setup_args
        self.threads = threads

        self.output_queue = mp.Queue()
        self.containers = [
            CaretakerProcess(
                constructor, input_queue, self.output_queue, setup_args, name='CaretakerProcess{}'.format(idx)
            )
            for idx in range(threads)
        ]

        if monitor_kwargs:
            if not isinstance(monitor_kwargs, dict):
                monitor_kwargs = dict()

            self.monitor = QueueMonitorThread(input_queue, **monitor_kwargs)
        else:
            self.monitor = DummyThread()

        logger.debug('ProcessRunner constructor finished')

    def __enter__(self):
        while self.input_queue.qsize() < self.threads:
            time.sleep(0.1)

        logger.debug('ProcessRunner: Starting threads')
        self.monitor.start()
        for container in self.containers:
            container.start()

        logger.debug('ProcessRunner: threads started, returning self')

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for container in self.containers:
            container.join()
        self.monitor.stop()
