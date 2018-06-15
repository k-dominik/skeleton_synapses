import logging
from logging.handlers import QueueHandler, QueueListener
from logging.config import dictConfig
import os
import subprocess
import multiprocessing as mp
import json
from datetime import datetime
import time
import warnings

from skeleton_synapses.constants import PROJECT_ROOT
from skeleton_synapses.helpers.files import mkdir_p

LOGGER_FORMAT = '%(levelname)s %(processName)s %(name)s: %(message)s'


def setup_logging(output_file_dir, args=None, kwargs=None, level=logging.NOTSET):
    """

    Parameters
    ----------
    output_file_dir
        Path of directory which will contain "logs" directory,
        which will contain a timestamped directory containing the log output
    args
        List of arguments given to original script, to be logged
    kwargs
        Dict of keyword arguments given to original script, to be logged
    level

    Returns
    -------
    logging.handlers.QueueListener
    """
    args = args or ()
    kwargs = kwargs or dict()

    # Don't warn about duplicate python bindings for opengm
    # (We import opengm twice, as 'opengm' 'opengm_with_cplex'.)
    warnings.filterwarnings("ignore", message='.*second conversion method ignored.', category=RuntimeWarning)

    # set up the log files and symlinks
    latest_ln = os.path.join(output_file_dir, 'logs', 'latest')
    try:
        os.remove(latest_ln)
    except FileNotFoundError:
        pass
    timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    log_dir = os.path.join(output_file_dir, 'logs', timestamp)
    mkdir_p(log_dir)
    os.symlink(log_dir, latest_ln)
    log_file = os.path.join(log_dir, 'locate_synapses.txt')

    # set up ilastik's default logging (without adding handlers)
    with open(os.path.join(PROJECT_ROOT, 'config', 'ilastik_logging.json')) as f:
        dictConfig(json.load(f))

    # set up handlers
    formatter = logging.Formatter(LOGGER_FORMAT)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)

    # logs should be handled by a single process
    log_queue = mp.Queue()
    queue_handler = QueueHandler(log_queue)
    queue_listener = QueueListener(log_queue, file_handler, console_handler)

    #  set up the root logger
    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(queue_handler)

    # set up the performance logger
    performance_formatter = logging.Formatter('%(asctime)s: elapsed %(message)s')
    performance_handler = logging.FileHandler(os.path.join(log_dir, 'timing.txt'))
    performance_handler.setFormatter(performance_formatter)
    performance_handler.setLevel(logging.INFO)
    performance_logger = logging.getLogger('PERFORMANCE_LOGGER')
    performance_logger.addHandler(performance_handler)
    performance_logger.propagate = True

    # write version information
    commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip()
    git_diff = subprocess.check_output(['git', 'diff']).strip()
    version_string = 'Commit hash: {}\n\nCurrent diff:\n{}'.format(commit_hash, git_diff)
    with open(os.path.join(log_dir, 'version.txt'), 'w') as f:
        f.write(version_string)

    # write argument information
    with open(os.path.join(log_dir, 'arguments.txt'), 'w') as f:
        f.write('Arguments:\n\t{}\nKeyword arguments:\n\t{}'.format(args, kwargs))

    queue_listener.start()

    return queue_listener


class Timestamper(object):
    def __init__(self):
        self.last_event = time.time()
        self.performance_logger = logging.getLogger('PERFORMANCE_LOGGER')

    def log(self, msg):
        now = time.time()
        self.performance_logger.info('{}: {}'.format(now - self.last_event, msg))
        self.last_event = now
