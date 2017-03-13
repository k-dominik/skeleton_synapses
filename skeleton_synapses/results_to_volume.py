#!/usr/bin/env python

import h5py
import numpy as np
import os
import json
from tqdm import trange
import argparse
from catmaid_interface import CatmaidAPI

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')

VOLUME_HDF5_PATH = os.path.join(ROOT, 'projects-2017/synapse_volume.hdf5')
SKELETON_HDF5_PATH = os.path.join(ROOT, 'projects-2017/L1-CNS/skeletons/11524047/synapse_cc.h5')

DTYPE = np.uint32  # can only store up to 2^24
UNKNOWN_LABEL = 0  # should be smallest label
BACKGROUND_LABEL = 1  # should be smaller than synapse labels
CHUNK_SHAPE = (256, 256, 1)  # yxz

REFRESH = True


def load_json(path):
    with open(path) as f:
        return {key: value for key, value in json.load(f).items() if not key.startswith('##')}


def reorder(lst, order):
    return [lst[idx] for idx in order]


def parse_slice_name(s):
    """
    Take string of the form '{slice_index}: x{x_coord}-y{y_coord}-z{z_coord}' and return the integer index and a dict
    of x, y and z coords.

    Parameters
    ----------
    s : str
        Slice name of the form '{slice_index}: x{x_coord}-y{y_coord}-z{z_coord}'

    Returns
    -------
    tuple
        (slice_index, {'x': x_coord, 'y': y_coord, 'z': z_coord})
    """
    idx, name = s.split(': ')
    return int(idx), {item[0]: int(item[1:]) for item in name.split('-')}


def center_to_bounding_box(center, offset):
    """
    Assumes z-depth of 1 and square bounding box

    Parameters
    ----------
    center : dict
        x, y, z coords of the center of the tile
    offset : int
        aka radius of box, x or y distance from center to side/ top

    Returns
    -------
    dict
        {'x': (min, max), 'y': (min, max), 'z': number} boundaries
    """
    return {
        'x': (int(center['x'] - offset), int(center['x'] + offset + 1)),
        'y': (int(center['y'] - offset), int(center['y'] + offset + 1)),
        'z': int(center['z'])
    }


def ensure_volume_exists(volume_path, description, force=False):
    """
    If the volume HDF5 file does not exist, or create it

    Parameters
    ----------
    volume_path : str
        Path to HDF5 volume file
    description : dict
        Ilastik-style stack description dictionary
    force : bool
        Whether to delete and recreate an empty HDF5 volume file if it already exists

    Returns
    -------

    """
    if force or not os.path.isfile(volume_path):
        with h5py.File(volume_path, 'w') as volume_file:
            volume = volume_file.create_dataset(
                'volume',
                reorder(description['bounds_zyx'], [1, 2, 0]),  # zyx -> yxz
                chunks=CHUNK_SHAPE,
                fillvalue=UNKNOWN_LABEL,
                dtype=DTYPE
            )
            volume.attrs['max_id'] = BACKGROUND_LABEL
            volume.attrs['unknown'] = UNKNOWN_LABEL
            volume.attrs['background'] = BACKGROUND_LABEL


def main(credential_path, stack_id, skeleton_hdf5_path=SKELETON_HDF5_PATH, volume_hdf5_path=VOLUME_HDF5_PATH):
    catmaid = CatmaidAPI.from_json(credential_path)
    description = catmaid.get_stack_description(stack_id)

    ensure_volume_exists(
        volume_hdf5_path,
        description,
        REFRESH
    )

    with h5py.File(volume_hdf5_path, 'r+') as volume_file, h5py.File(skeleton_hdf5_path, 'r') as synapse_file:
        volume = volume_file['volume']
        synapses = synapse_file['data']
        offset = np.floor(synapses.shape[0] / 2)

        max_id = volume.attrs['max_id']
        next_max_id = max_id
        background_label = volume.attrs['background']

        slice_centers = dict(parse_slice_name(s) for s in synapses.attrs['slice-names'])
        slice_boundaries = {key: center_to_bounding_box(value, offset) for key, value in slice_centers.items()}

        for z_idx in trange(synapses.shape[2]):
            z_slice = np.array(synapses[:, :, z_idx, 0])
            z_slice[z_slice != 0] = z_slice[z_slice != 0] + max_id  # ensure no label clashes
            z_slice[z_slice == 0] = background_label

            next_max_id = max(next_max_id, np.max(z_slice))
            bbox = slice_boundaries[z_idx]
            volume[bbox['y'][0]:bbox['y'][1], bbox['x'][0]:bbox['x'][1], bbox['z']] = z_slice

        volume.attrs['max_id'] = next_max_id


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="" +
        "Commit the treenode-centered synapse label stack to the volume-sized label stack."
    )
    parser.add_argument(
        'credential_path',
        help='Path to a JSON file containing credentials for interacting with CATMAID'
    )
    parser.add_argument(
        'stack_id',
        help='ID or name of CATMAID image stack'
    )
    parser.add_argument(
        'skeleton_path', default=SKELETON_HDF5_PATH,
        help='Path to the HDF5 skeleton label stack'
    )
    parser.add_argument(
        'volume_path', default=VOLUME_HDF5_PATH,
        help='Path to the HDF5 volume-scale label stack'
    )
    args = parser.parse_args()

    main(args.credential_path, args.stack_id, args.skeleton_path, args.volume_path)
