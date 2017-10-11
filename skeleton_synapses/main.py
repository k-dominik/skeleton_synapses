#!/usr/bin/env python
from __future__ import division
import logging
import argparse
import sys
import os
import hashlib
import subprocess
import signal
from datetime import datetime
import multiprocessing as mp

import psutil
import numpy as np
from logutils.queue import QueueHandler, QueueListener

from catpy import CatmaidClient

from catmaid_interface import CatmaidSynapseSuggestionAPI
from locate_synapses import setup_files, LOGGER_FORMAT, ensure_list, mkdir_p, DEFAULT_ROI_RADIUS
from locate_syn_catmaid import ensure_hdf5, locate_synapses_catmaid

DEBUG = False
LOG_LEVEL = logging.DEBUG

ALGO_HASH = None


def hash_algorithm(*paths):
    """
    Calculate a combined hash of the algorithm. Included for hashing are the commit hash of this repo, the hashes of
    any files whose paths are given, and the git commit hash inside any directories whose paths are given.

    Parameters
    ----------
    paths
        Paths to files or directories outside of this git repo which affect the algorithm

    Returns
    -------
    str
    """
    logger.info('Hashing algorithm...')
    commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip()

    md5 = hashlib.md5(commit_hash)

    for path in sorted(paths):
        if os.path.isdir(path):
            logger.debug('Getting git commit hash of directory %s', path)
            try:
                output = subprocess.check_output(['git', '-C', path, 'rev-parse', 'HEAD']).strip()
                md5.update(output)
            except subprocess.CalledProcessError:
                logger.exception('Error encountered while finding git hash of directory %s', path)
        elif os.path.isfile(path):
            logger.debug('Getting hash of file %s', path)
            with open(path, 'rb') as f:
                for chunk in iter(lambda: f.read(128 * md5.block_size), b''):
                    md5.update(chunk)
        else:
            logger.warning('No file, symlink or directory found at %s', path)

    digest = md5.hexdigest()

    # todo: remove this
    if ALGO_HASH is not None:
        digest = ALGO_HASH
        logger.warning('Ignoring real algorithm hash, using hardcoded value'.format(ALGO_HASH))
    logger.debug('Algorithm hash is %s', digest)
    return digest


def main(credentials_path, stack_id, skeleton_ids, input_file_dir, output_file_dir, roi_radius_px=150, force=False):
    logger.info("STARTING TILEWISE")

    catmaid = CatmaidSynapseSuggestionAPI(CatmaidClient.from_json(credentials_path), stack_id)
    stack_info = catmaid.get_stack_info(stack_id)

    ensure_hdf5(stack_info, output_file_dir, force=force)

    skeleton_ids = ensure_list(skeleton_ids)

    autocontext_project, multicut_project, volume_description_path, skel_output_dirs, algo_notes = setup_files(
        credentials_path, stack_id, skeleton_ids, input_file_dir, force, output_file_dir if DEBUG else None
    )

    if force:
        logger.info('Using random hash')
        algo_hash = hash(np.random.random())
    else:
        algo_hash = hash_algorithm(autocontext_project, multicut_project)

    for skeleton_id in skeleton_ids:
        locate_synapses_catmaid(
            autocontext_project,
            multicut_project,
            volume_description_path,
            output_file_dir,
            skeleton_id,
            roi_radius_px,
            stack_info,
            algo_hash,
            algo_notes,
            catmaid
        )


def setup_logging(output_file_dir, args, kwargs, level=logging.NOTSET):
    # set up the log files and symlinks
    latest_ln = os.path.join(output_file_dir, 'logs', 'latest')
    os.remove(latest_ln)
    timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    log_dir = os.path.join(output_file_dir, 'logs', timestamp)
    mkdir_p(log_dir)
    os.symlink(log_dir, latest_ln)
    log_file = os.path.join(log_dir, 'locate_synapses.txt')

    log_queue = mp.Queue()

    # set up handlers
    formatter = logging.Formatter(LOGGER_FORMAT)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level)

    queue_listener = QueueListener(log_queue, file_handler, stream_handler)

    #  set up the root logger
    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(QueueHandler(log_queue))

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


def kill_child_processes(signum=None, frame=None):
    current_proc = psutil.Process()
    killed = []
    for child_proc in current_proc.children(recursive=True):
        logger.debug('Killing process: {} with status {}'.format(child_proc.name(), child_proc.status()))
        child_proc.kill()
        killed.append(child_proc.pid)
    logger.debug('Killed {} processes'.format(len(killed)))
    return killed


logger = None


if __name__ == "__main__":
    if DEBUG:
        print("USING DEBUG ARGUMENTS")

        input_dir = "../projects-2017/L1-CNS"
        output_dir = input_dir
        cred_path = "credentials_dev.json"
        stack_id = 1
        skel_ids = [18531735]  # small test skeleton only on CLB's local instance

        force = 1

        args_list = [
            cred_path, stack_id, input_dir, skel_ids
        ]
        kwargs_dict = {'force': force}
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument('credentials-path',
                            help='Path to a JSON file containing CATMAID credentials (see credentials/example.json)')
        parser.add_argument('stack-id',
                            help='ID or name of image stack in CATMAID')
        parser.add_argument('input-file-dir', help="A directory containing project files.")
        parser.add_argument('skeleton-ids', nargs='+',
                            help="Skeleton IDs in CATMAID")
        parser.add_argument('-o', '--output-dir', default=None,
                            help='A directory containing output files')
        parser.add_argument('-r', '--roi-radius-px', default=DEFAULT_ROI_RADIUS,
                            help='The radius (in pixels) around each skeleton node to search for synapses')
        parser.add_argument('-f', '--force', type=int, default=0,
                            help="Whether to delete all prior results for a given skeleton: pass 1 for true or 0")

        args = parser.parse_args()
        output_dir = args.output_dir or args.input_file_dir
        args_list = [
            args.credentials_path, args.stack_id, args.skeleton_ids, args.input_file_dir, output_dir,
            args.roi_radius_px, args.force
        ]
        kwargs_dict = {}  # must be empty

    logging_listener = setup_logging(output_dir, args_list, kwargs_dict, LOG_LEVEL)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.NOTSET)

    logger.info('STARTING CATMAID-COMPATIBLE DETECTION')

    signal.signal(signal.SIGTERM, kill_child_processes)

    exit_code = 1
    try:
        main(*args_list, **kwargs_dict)
        exit_code = 0
    except Exception as e:
        logger.exception('Errored, killing all child processes and exiting')
        kill_child_processes()
        raise
    finally:
        logging_listener.stop()
        sys.exit(exit_code)
