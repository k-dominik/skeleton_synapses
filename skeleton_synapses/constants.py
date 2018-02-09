import logging
import os

DEBUG = bool(int(os.getenv('SS_DEBUG', 0)))
ALGO_HASH = None  # set to fix algorithm hash
LOG_LEVEL = logging.DEBUG

DEFAULT_THREADS = 3
DEFAULT_RAM_MB_PER_PROCESS = 1200

DEFAULT_ROI_RADIUS_PX = 150

DEFAULT_SYNAPSE_DISTANCE_NM = 600

TQDM_KWARGS = {
    'ncols': 50,
}

RESULTS_TIMEOUT_SECONDS = 5*60  # result fetchers time out after 5 minutes

# ROOT_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..')
PACKAGE_ROOT = os.path.realpath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT)

THREADS = int(os.getenv('SYNAPSE_DETECTION_THREADS', DEFAULT_THREADS))
RAM_MB_PER_PROCESS = int(os.getenv('SYNAPSE_DETECTION_RAM_MB_PER_PROCESS', DEFAULT_RAM_MB_PER_PROCESS))
