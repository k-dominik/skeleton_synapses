import datetime
import errno
import json
import os
import shutil
from datetime import datetime
import logging

import six
import h5py
import numpy as np
import vigra
from catpy import CatmaidClient

from skeleton_synapses.catmaid_interface import CatmaidSynapseSuggestionAPI

HDF5_NAME = "tilewise_image_store.hdf5"
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


def ensure_description_file(catmaid, description_path, stack_id, include_offset=False, force=False):
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
    if force or not os.path.isfile(description_path):
        volume_description_dict = catmaid.get_stack_description(stack_id, include_offset=include_offset)
        with open(description_path, 'w') as f:
            json.dump(volume_description_dict, f, sort_keys=True, indent=2)
        return True


def ensure_skel_output_dirs(output_file_dir, skel_ids, catmaid_ss, stack_id, force=False):
    skel_output_dirs = []
    for skeleton_id in ensure_list(skel_ids):
        # Name the output directory with the skeleton id
        skel_output_dir = os.path.join(output_file_dir, 'skeletons', str(skeleton_id))
        if force:
            try:
                shutil.rmtree(skel_output_dir, ignore_errors=True)
            except OSError:
                pass

        mkdir_p(skel_output_dir)

        skel_path = os.path.join(skel_output_dir, 'tree_geometry.json')
        skel_data = catmaid_ss.get_transformed_treenode_and_connector_geometry(stack_id, skeleton_id)
        with open(skel_path, 'w') as f:
            json.dump(skel_data, f)

        skel_output_dirs.append(skel_output_dir)

    return skel_output_dirs


def setup_files(
        credentials_path, stack_id, skeleton_ids, input_file_dir, force=False, output_file_dir=None
):
    """

    Parameters
    ----------
    credentials_path
    stack_id
    skeleton_ids
    input_file_dir
    force
    output_file_dir

    Returns
    -------
    tuple of (str, str, str, list of str, dict of {str:str})
    """
    catmaid = CatmaidSynapseSuggestionAPI(CatmaidClient.from_json(credentials_path), stack_id)
    projects_dir = os.path.join(input_file_dir, 'projects')

    autocontext_project = os.path.join(projects_dir, 'full-vol-autocontext.ilp')
    multicut_project = os.path.join(projects_dir, 'multicut', PROJECT_NAME + '-multicut.ilp')

    volume_description_path = os.path.join(
        input_file_dir, PROJECT_NAME + '-description-NO-OFFSET.json'
    )
    ensure_description_file(catmaid, volume_description_path, stack_id, include_offset=False)

    skeleton_ids = ensure_list(skeleton_ids)
    if output_file_dir:
        skel_output_dirs = ensure_skel_output_dirs(output_file_dir, skeleton_ids, catmaid, stack_id, force)
    else:
        skel_output_dirs = [None for _ in skeleton_ids]

    algo_notes = get_algo_notes(projects_dir)

    return autocontext_project, multicut_project, volume_description_path, skel_output_dirs, algo_notes


def cached_synapses_predictions_for_roi(roi_xyz, hdf5_path, squeeze=True):
    """

    Parameters
    ----------
    roi_xyz
    hdf5_path
    squeeze

    Returns
    -------
    (vigra.VigraArray, vigra.VigraArray)
    """
    # convert roi into a tuple of slice objects which can be used by numpy for indexing
    roi_slices = roi_xyz[0, 2], slice(roi_xyz[0, 1], roi_xyz[1, 1]), slice(roi_xyz[0, 0], roi_xyz[1, 0])

    with h5py.File(hdf5_path, 'r') as f:
        synapse_cc_xy = vigra.taggedView(
            f['slice_labels'], axistags=f['slice_labels'].attrs['axistags']
        )[roi_slices].transposeToOrder('V')
        predictions_xyc = vigra.taggedView(
            f['pixel_predictions'], axistags=f['pixel_predictions'].attrs['axistags']
        )[roi_slices].transposeToOrder('V')

    # if squeeze:
    #     return synapse_cc_xy.squeeze(), predictions_xyc.squeeze()
    # else:
    return synapse_cc_xy, predictions_xyc


initialized_files = set()


def write_output_image(output_dir, image_xyc, name, name_prefix="", mode="stacked"):
    """
    Write the given image to an hdf5 file.

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
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
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


def create_label_volume(stack_info, hdf5_file, name, tile_size=TILE_SIZE, dtype=LABEL_DTYPE, colour_channels=None):
    spatial_axes = 'zyx'
    dimension = tuple(stack_info['dimension'][dim] for dim in spatial_axes)
    chunksize = (1, tile_size, tile_size)

    axistags = spatial_axes
    if colour_channels is not None:
        axistags += 'c'
        dimension += (colour_channels, )
        chunksize += (colour_channels, )

    labels = hdf5_file.create_dataset(
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


def ensure_hdf5(stack_info, output_file_dir, force=False):
    hdf5_path = os.path.join(output_file_dir, HDF5_NAME)
    if force or not os.path.isfile(hdf5_path):
        if os.path.isfile(hdf5_path):
            hdf5_dir = os.path.dirname(hdf5_path)
            backup_path = os.path.join(hdf5_dir, '{}BACKUP{}'.format(hdf5_path, datetime.now().strftime('%Y-%m-%d_%H:%M:%S')))
            os.rename(hdf5_path, backup_path)
        logger.info('Creating HDF5 volumes in %s', hdf5_path)
        with h5py.File(hdf5_path) as f:
            # f.attrs['workflow_id'] = workflow_id  # todo
            f.attrs['source_stack_id'] = stack_info['sid']

            create_label_volume(stack_info, f, 'slice_labels', TILE_SIZE)
            create_label_volume(
                stack_info, f, 'pixel_predictions', TILE_SIZE, dtype=PIXEL_PREDICTION_DTYPE, colour_channels=3
            )

            f.flush()

    return hdf5_path


def write_predictions_synapses(hdf5_path, predictions_xyc, mapped_synapse_cc_xy, bounds_xyz):
    slicing_zyx = bounds_xyz[0, 2], slice(bounds_xyz[0, 1], bounds_xyz[1, 1]), slice(bounds_xyz[0, 0], bounds_xyz[1, 0])
    with h5py.File(hdf5_path, 'r+') as f:
        pixel_predictions_zyxc = f['pixel_predictions']
        pixel_predictions_zyxc[slicing_zyx] = predictions_xyc.transposeToOrder(pixel_predictions_zyxc.attrs['order'])

        slice_labels_zyx = f['slice_labels']
        slice_labels_zyx[slicing_zyx] = mapped_synapse_cc_xy.transposeToOrder(slice_labels_zyx.attrs['order'])
