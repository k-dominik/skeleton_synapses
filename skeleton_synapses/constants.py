import logging
import os
from warnings import warn

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

MONITOR_HOST = 'localhost'
MONITOR_PORT = int(os.getenv('MONITOR_PORT', 8088))
MONITOR_INTERVAL = 10

ILP_RETRAIN = False
ILP_READONLY = True
if ILP_RETRAIN and ILP_READONLY:
    warn('ILP must be writable of it is to retrain. Disabling read-only mode')
    ILP_READONLY = False
