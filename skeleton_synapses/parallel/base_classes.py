import logging
import multiprocessing as mp

import psutil

from skeleton_synapses.constants import RAM_MB_PER_PROCESS

class DebuggableProcess(mp.Process):
    """
    Classes inheriting from this instead of multiprocessing.Process can use a `debug` parameter to run in serial
    instead of spawning a new python process, for easier debugging.
    """
    def __init__(self, debug=False, name=None):
        super(DebuggableProcess, self).__init__(name=name)
        self.debug = debug
        self.inner_logger = logging.getLogger(self.name)

    def start(self):
        if self.debug:
            self.run()
        else:
            super(DebuggableProcess, self).start()


class LeakyProcess(DebuggableProcess):
    """
    To be subclassed by actual processes with memory leaks.

    Override methods to determine behaviour:

    setup() is run once per process, on run()
    execute() is run in a while loop for as long as the input queue isn't empty and the RAM limit isn't exceeded
    teardown() is run before the process is shut down, either when the input queue is empty or the RAM limit is exceeded
    """
    def __init__(self, input_queue, debug=False, name=None):
        super(LeakyProcess, self).__init__(debug, name)
        self.input_queue = input_queue
        self.max_ram_MB = RAM_MB_PER_PROCESS
        self.psutil_process = None
        self.execution_counter = 0

        self.size_logger = logging.getLogger(self.name + '.size')

    def run(self):
        self.inner_logger.info('%s started', type(self).__name__)
        self.setup()
        self.psutil_process = [proc for proc in psutil.process_iter() if proc.pid == self.pid][0]
        self.size_logger.debug(self.ram_usage_str())
        while not self.needs_pruning():
            self.execute()
            self.execution_counter += 1
            self.size_logger.debug(self.ram_usage_str())

        self.teardown()

    def setup(self):
        pass

    def execute(self):
        pass

    def teardown(self):
        pass

    @property
    def ram_usage_MB(self):
        try:
            return self.psutil_process.memory_info().rss / 1024 / 1024
        except AttributeError:
            return None

    def ram_usage_str(self):
        return 'RAM usage: {:.03f}MB out of {:.03f}MB'.format(self.ram_usage_MB, self.max_ram_MB)

    def needs_pruning(self):
        if self.input_queue.empty():
            self.size_logger.info('TERMINATING after {} iterations (input queue empty)'.format(self.execution_counter))
            return True

        ram_usage_MB = self.ram_usage_MB

        if self.max_ram_MB and ram_usage_MB >= self.max_ram_MB:
            self.size_logger.info(
                'TERMINATING after {} iterations ({}, {} items remaining)'.format(
                    self.execution_counter, self.ram_usage_str(), self.input_queue.qsize()
                )
            )
            return True

        return False
