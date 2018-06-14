import datetime
import errno
import hashlib
import json
import os
import shutil
import subprocess as sp
from datetime import datetime
import logging

import six
import h5py
import numpy as np
import vigra

from skeleton_synapses.catmaid_interface import CatmaidSynapseSuggestionAPI
from skeleton_synapses.constants import PROJECT_ROOT, ALGO_HASH

IMAGE_STORE_NAME = "tilewise_image_store.hdf5"
LABEL_DTYPE = np.int64
PIXEL_PREDICTION_DTYPE = np.float32
TILE_SIZE = 512
PROJECT_NAME = 'L1-CNS'  # todo: remove dependency on this

logger = logging.getLogger(__name__)


def ensure_list(value):
    """Ensure that a given value is a non-string sequence (making it the sole element of a list if not).

    Used for compatibility purposes.
    """
    try:
        len(value)
        if isinstance(value, six.string_types):
            raise TypeError
        return value
    except TypeError:
        return [value]


def ensure_description_file(catmaid, description_path, stack_id, include_offset=False, force=False, throw_on_missing=False):
    """

    Parameters
    ----------
    catmaid : CatmaidSynapseSuggestionAPI
    description_path : str
    stack_id : int
    include_offset : bool

    Returns
    -------
    bool
        Whether a new volume description was created
    """
    if throw_on_missing and not os.path.isfile(description_path):
        raise AssertionError('Description file missing from ' + description_path)

    if force or not os.path.isfile(description_path):
        volume_description_dict = catmaid.get_stack_description(stack_id, include_offset=include_offset)
        with open(description_path, 'w') as f:
            json.dump(volume_description_dict, f, sort_keys=True, indent=2)
        return True


def ensure_dir(path, force=False, throw_on_missing=False):
    if throw_on_missing and not os.path.isdir(path):
        raise AssertionError('Path {} is missing'.format(path))

    if force:
        try:
            shutil.rmtree(path, ignore_errors=True)
        except OSError:
            pass

    mkdir_p(path)


def ensure_skel_output_dir(skel_output_dir, skel_id, catmaid_ss, stack_id, force=False, throw_on_missing=False):
    if skel_output_dir is None:
        return
    # Name the output directory with the skeleton id
    ensure_dir(skel_output_dir, force, throw_on_missing)

    skel_path = os.path.join(skel_output_dir, 'tree_geometry.json')
    skel_data = catmaid_ss.get_transformed_treenode_and_connector_geometry(stack_id, skel_id)
    with open(skel_path, 'w') as f:
        json.dump(skel_data, f)


class Paths(object):
    def __init__(self, credentials_path, input_file_dir, output_file_dir=None, debug_images=False):
        self.root_dir = PROJECT_ROOT

        self.credentials_json = credentials_path
        self.input_dir = input_file_dir
        self.output_dir = output_file_dir or self.input_dir
        self._debug_images = debug_images

        self.projects_dir = os.path.join(input_file_dir, 'projects')

        self.autocontext_ilp = os.path.join(self.projects_dir, 'full-vol-autocontext.ilp')
        self.multicut_ilp = os.path.join(self.projects_dir, 'multicut', PROJECT_NAME + '-multicut.ilp')

        self.output_image_store = os.path.join(output_file_dir, IMAGE_STORE_NAME)
        self.description_json = os.path.join(self.projects_dir, PROJECT_NAME + '-description-NO-OFFSET.json')

        self.debug_dir = os.path.join(self.output_dir, 'debug') if self._debug_images else None

        self.debug_skel_dir = os.path.join(self.debug_dir, 'skeletons') if self._debug_images else None
        self.debug_synapse_dir = os.path.join(self.debug_dir, 'synapses') if self._debug_images else None
        self.debug_tile_dir = os.path.join(self.debug_dir, 'tiles') if self._debug_images else None

        self.initialised = False

    def skeleton_output_dir(self, skeleton_id):
        return os.path.join(self.debug_skel_dir, str(skeleton_id)) if self._debug_images else None

    def initialise(self, catmaid, stack_info, skeleton_ids=None, force=False, throw_on_missing=False):
        """

        Parameters
        ----------
        catmaid : CatmaidSynapseSuggestionAPI or None
            If None, instantiate using given credentials file
        stack_info : dict or int or str
            If dict, assume is the response from catmaid. If int or str, fetch that response
        skeleton_ids
        force
        throw_on_missing

        """
        if self.initialised:
            logger.warning('Paths already initialised, ignoring second initialisation call')
            return

        if catmaid is None:
            catmaid = CatmaidSynapseSuggestionAPI.from_json(self.credentials_json)

        stack_info_dict = stack_info if isinstance(stack_info, dict) else catmaid.get_stack_info(stack_info)

        for dir_path in [self.root_dir, self.input_dir, self.output_dir, self.projects_dir]:
            assert os.path.isdir(dir_path), '{} does not exist as directory'.format(dir_path)
        for file_path in [self.autocontext_ilp, self.multicut_ilp]:
            assert os.path.isfile(file_path), '{} does not exist as file'.format(file_path)

        ensure_image_store(stack_info_dict, self.output_image_store, force, throw_on_missing)
        ensure_description_file(catmaid, self.description_json, stack_info_dict['sid'],
                                force=False, throw_on_missing=throw_on_missing)

        if self._debug_images and skeleton_ids:
            ensure_dir(self.debug_synapse_dir, force, throw_on_missing)
            ensure_dir(self.debug_tile_dir, force, throw_on_missing)
            ensure_dir(self.debug_skel_dir, force, throw_on_missing)
            for skeleton_id in skeleton_ids:
                ensure_skel_output_dir(
                    self.skeleton_output_dir(skeleton_id), skeleton_id, catmaid, stack_info_dict['sid'],
                    force, throw_on_missing
                )

        self.initialised = True

    def __str__(self):
        return 'Paths[{}]'.format(json.dumps(self.__dict__, sort_keys=True, indent=2))


def get_2d_axistags(dataset, default=None, eliminate='z'):
    """

    Parameters
    ----------
    dataset : h5py.Dataset
    default : str
        Axistags to return if they are not stored in the dataset
    eliminate : str
        Axes to eliminate from the dataset in order to make it planar

    Returns
    -------
    str
    """
    axistags = dataset.attrs.get('axistags')
    if axistags is None:
        if default:
            return default
        else:
            raise ValueError('No axistags found in dataset {}, and no defaults given'.format(dataset.name))

    return ''.join(char for char in axistags if char not in eliminate)


def cached_synapses_predictions_for_roi(roi_xyz, image_store_path, synapse_cc=True, predictions=True):
    """
    Will take axistags from image store if they exist, otherwise assume subset of 'zyxct'

    Parameters
    ----------
    roi_xyz
    image_store_path
    squeeze

    Returns
    -------
    (vigra.VigraArray, vigra.VigraArray)
        synapse_cc_xy, predictions_xyc
    """
    # convert roi into a tuple of slice objects which can be used by numpy for indexing
    roi_slices = roi_xyz[0, 2], slice(roi_xyz[0, 1], roi_xyz[1, 1]), slice(roi_xyz[0, 0], roi_xyz[1, 0])

    with h5py.File(image_store_path, 'r') as f:
        synapse_cc_xy = vigra.taggedView(
            f['slice_labels'][roi_slices], axistags=get_2d_axistags(f['slice_labels'], 'yx')
        ).transposeToOrder('V') if synapse_cc else None

        predictions_xyc = vigra.taggedView(
            f['pixel_predictions'][roi_slices], axistags=get_2d_axistags(f['pixel_predictions'], 'yxc')
        ).transposeToOrder('V') if predictions else None

    return synapse_cc_xy, predictions_xyc


initialized_files = set()


def write_output_image(output_dir, image_xyc, name, name_prefix="", mode="stacked"):
    """
    Write the given image to an hdf5 file for debug purposes.

    If mode is "slices", create a new file for the image.
    If mode is "stacked", create a new file with 'name' if it doesn't exist yet,
    or append to it if it does.
    """
    global initialized_files
    if not output_dir:
        return

    # Insert a Z-axis
    image_xyzc = vigra.taggedView(image_xyc[:,:,None,:], 'xyzc')

    if mode == "slices":
        output_subdir = os.path.join(output_dir, name)
        mkdir_p(output_subdir)
        if not name_prefix:
            name_prefix = datetime.datetime.now().isoformat()
        filepath = os.path.join(output_subdir, name_prefix + '.h5')
        with h5py.File(filepath, 'w') as f:
            f.create_dataset("data", data=image_xyzc)

    elif mode == "stacked":
        # If the file exists from a previous (failed) run,
        # delete it and start from scratch.
        filepath = os.path.join(output_dir, name + '.h5')
        if filepath not in initialized_files:
            try:
                os.unlink(filepath)
            except OSError as ex:
                if ex.errno != errno.ENOENT:
                    raise
            initialized_files.add(filepath)

        # Also append to an HDF5 stack
        with h5py.File(filepath) as f:
            if 'data' in f:
                # Add room for another z-slice
                z_size = f['data'].shape[2]
                f['data'].resize(z_size+1, 2)
            else:
                maxshape = np.array(image_xyzc.shape)
                maxshape[2] = 100000
                f.create_dataset('data', shape=image_xyzc.shape, maxshape=tuple(maxshape), dtype=image_xyzc.dtype)
                f['data'].attrs['axistags'] = image_xyzc.axistags.toJSON()
                f['data'].attrs['slice-names'] = []

            # Write onto the end of the stack.
            f['data'][:, :, -1:, :] = image_xyzc

            # Maintain an attribute 'slice-names' to list each slice's name
            z_size = f['data'].shape[2]
            names = list(f['data'].attrs['slice-names'])
            names += ["{}: {}".format(z_size-1, name_prefix)]
            del f['data'].attrs['slice-names']
            f['data'].attrs['slice-names'] = names

    else:
        raise ValueError('Image write mode {} not recognised.'.format(repr(mode)))

    return filepath


def mkdir_p(path):
    """
    Like the bash command 'mkdir -p'
    """
    try:
        os.makedirs(path)
        return True
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            return False
        else:
            raise


def get_algo_notes(projects_dir):
    try:
        with open(os.path.join(projects_dir, 'algorithm_notes.json')) as f:
            algo_notes = json.load(f)
    except IOError:
        logger.warning('Algorithm notes not found, using empty strings')
        algo_notes = {'synapse_detection': '', 'skeleton_association': ''}

    return algo_notes


def create_label_volume(stack_info, open_image_store, name, tile_size=TILE_SIZE, dtype=LABEL_DTYPE, colour_channels=None):
    spatial_axes = 'zyx'
    dimension = tuple(stack_info['dimension'][dim] for dim in spatial_axes)
    chunksize = (1, tile_size, tile_size)

    axistags = spatial_axes
    if colour_channels is not None:
        axistags += 'c'
        dimension += (colour_channels, )
        chunksize += (colour_channels, )

    labels = open_image_store.create_dataset(
        name,
        dimension,  # zyx(c)
        chunks=chunksize,  # zyx(c)
        fillvalue=0,
        dtype=dtype
    )

    for key in ['translation', 'dimension', 'resolution']:
        labels.attrs[key] = json.dumps(stack_info[key], sort_keys=True)

    labels.attrs['axistags'] = axistags
    labels.attrs['order'] = 'C'

    return labels


def ensure_image_store(stack_info, image_store_path, force=False, throw_on_missing=False):
    if throw_on_missing and not os.path.exists(image_store_path):
        raise AssertionError('Image store missing missing from ' + image_store_path)

    if force or not os.path.exists(image_store_path):
        if os.path.isfile(image_store_path):
            backup_path = os.path.join(
                os.path.dirname(image_store_path),
                '{}BACKUP{}'.format(image_store_path, datetime.now().strftime('%Y-%m-%d_%H:%M:%S'))
            )
            os.rename(image_store_path, backup_path)
        logger.info('Creating image store volumes in %s', image_store_path)
        with h5py.File(image_store_path) as f:
            # f.attrs['workflow_id'] = workflow_id  # todo
            f.attrs['source_stack_id'] = stack_info['sid']

            create_label_volume(stack_info, f, 'slice_labels', TILE_SIZE)
            create_label_volume(
                stack_info, f, 'pixel_predictions', TILE_SIZE, dtype=PIXEL_PREDICTION_DTYPE, colour_channels=3
            )

            f.flush()


def write_predictions_synapses(image_store_path, predictions_xyc, mapped_synapse_cc_xy, bounds_xyz):
    slicing_zyx = bounds_xyz[0, 2], slice(bounds_xyz[0, 1], bounds_xyz[1, 1]), slice(bounds_xyz[0, 0], bounds_xyz[1, 0])
    with h5py.File(image_store_path, 'r+') as f:
        pixel_predictions_zyxc = f['pixel_predictions']
        pixel_predictions_zyxc[slicing_zyx] = predictions_xyc.transposeToOrder(pixel_predictions_zyxc.attrs['order'])

        slice_labels_zyx = f['slice_labels']
        slice_labels_zyx[slicing_zyx] = mapped_synapse_cc_xy.transposeToOrder(slice_labels_zyx.attrs['order'])


def hash_file(path, md5=None):
    if md5 is None:
        md5 = hashlib.md5()

    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(128 * md5.block_size), b''):
            md5.update(chunk)

    return md5


def hash_directory_contents(path, md5=None):
    if md5 is None:
        md5 = hashlib.md5()

    for root, dirnames, fnames in os.walk(path):
        for fname in fnames:
            hash_file(os.path.join(root, fname), md5)

    return md5


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
    commit_hash = sp.check_output(['git', 'rev-parse', 'HEAD']).strip()

    md5 = hashlib.md5(commit_hash)

    for path in sorted(paths):
        if os.path.isdir(path):
            logger.debug('Getting git commit hash of directory %s', path)
            result = sp.run(['git', '-C', path, 'rev-parse', 'HEAD'], stdout=sp.PIPE, stderr=sp.PIPE)
            if result.returncode and "ot a git repo" in result.stderr.decode("utf-8"):
                md5 = hash_directory_contents(path, md5)
            else:
                md5.update(result.stdout.decode("utf-8"))
        elif os.path.isfile(path):
            logger.debug('Getting hash of file %s', path)
            md5 = hash_file(path, md5)
        else:
            logger.warning('No file, symlink or directory found at %s', path)

    digest = md5.hexdigest()

    # todo: remove this
    if ALGO_HASH is not None:
        digest = ALGO_HASH
        logger.warning('Ignoring real algorithm hash, using hardcoded value'.format(ALGO_HASH))
    logger.debug('Algorithm hash is %s', digest)
    return digest


def dump_images(path, roi_xyz=None, **kwargs):
    """
    Writes set of vigra arrays to an HDF5 file.

    Parameters
    ----------
    path : str
    roi_xyz : np.array
    kwargs : dict of {str: vigra.VigraArray}
    """
    with h5py.File(path) as f:
        for name, arr in kwargs.items():
            arr.writeHDF5(f, name)
        if roi_xyz is not None:
            f.create_dataset('roi_xyz', data=roi_xyz)
