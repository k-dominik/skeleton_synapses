#!/usr/bin/env python

import h5py
import numpy as np
import os
import json
from tqdm import trange
import argparse
import pandas as pd
from catmaid_interface import CatmaidAPI

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')

VOLUME_HDF5_PATH = os.path.join(ROOT, 'projects-2017/synapse_volume.hdf5')
ILASTIK_OUTPUT_PATH = os.path.join(ROOT, 'projects-2017/L1-CNS/')

DTYPE = np.uint32  # can only store up to 2^24
UNKNOWN_LABEL = 0  # should be smallest label
BACKGROUND_LABEL = 1  # should be smaller than synapse labels
CHUNK_SHAPE = (256, 256, 1)  # yxz

CSV_HEADERS = [
    'synapse_id',
    'skeleton_id'
    'overlaps_node_segment',
    'x_px',
    'y_px',
    'z_px',
    'size_px',
    'tile_x_px',
    'tile_y_px',
    'tile_index',
    'distance_to_node_px',
    'detection_uncertainty',
    'node_id',
    'node_x_px',
    'node_y_px',
    'node_z_px',
]


CSV_DTYPE = np.float64


REFRESH = True
DEBUGGING = True


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


def topleft_to_bounding_box(center, side_length):
    """
    Assumes z-depth of 1 and square bounding box

    Parameters
    ----------
    center : dict
        x, y, z coords of the top left corner of the tile
    side_length : int
        side length of bounding box

    Returns
    -------
    dict
        {'x': (min, max), 'y': (min, max), 'z': number} boundaries
    """
    return {
        'x': (int(center['x']), int(center['x'] + side_length)),
        'y': (int(center['y']), int(center['y'] + side_length)),
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
            volume.attrs['unknown'] = UNKNOWN_LABEL
            volume.attrs['background'] = BACKGROUND_LABEL

            synapse_info = volume_file.create_dataset(
                'synapse_info',
                shape=(0, len(CSV_HEADERS)),
                dtype=CSV_DTYPE
            )
            synapse_info.attrs['headers'] = CSV_HEADERS
            synapse_info.attrs['max_id'] = BACKGROUND_LABEL


# todo: deal with intersecting synapses (give them same ID)
def main(credential_path, stack_id, skel_id, ilastik_output_path=ILASTIK_OUTPUT_PATH, volume_hdf5_path=VOLUME_HDF5_PATH):
    catmaid = CatmaidAPI.from_json(credential_path)
    description = catmaid.get_stack_description(stack_id)

    ensure_volume_exists(
        volume_hdf5_path,
        description,
        REFRESH
    )

    skeleton_hdf5_path = os.path.join(ilastik_output_path, str(skel_id), 'synapse_cc.h5')
    skeleton_csv_path = os.path.join(
        ilastik_output_path, str(skel_id), 'skeleton-{}-synapses.csv'.format(skel_id)
    )

    with h5py.File(volume_hdf5_path, 'r+') as volume_file, h5py.File(skeleton_hdf5_path, 'r') as synapse_file:
        volume = volume_file['volume']
        synapses = synapse_file['data']
        synapse_info = volume_file['synapse_info']

        max_id = synapse_info.attrs['max_id']
        next_max_id = max_id
        background_label = volume.attrs['background']

        slice_centers = dict(parse_slice_name(s) for s in synapses.attrs['slice-names'])
        slice_boundaries = {key: topleft_to_bounding_box(value, synapses.shape[0]) for key, value in slice_centers.items()}

        for z_idx in trange(synapses.shape[2]):
            z_slice = np.array(synapses[:, :, z_idx, 0])
            z_slice[z_slice != 0] = z_slice[z_slice != 0] + max_id  # ensure no label clashes
            z_slice[z_slice == 0] = background_label

            next_max_id = max(next_max_id, np.max(z_slice))
            bbox = slice_boundaries[z_idx]
            volume[bbox['y'][0]:bbox['y'][1], bbox['x'][0]:bbox['x'][1], bbox['z']] = z_slice

        headers = synapse_info.attrs['headers']

        df = pd.read_csv(skeleton_csv_path, delimiter='\t')
        df['synapse_id'] += max_id
        old_info = np.array(synapse_info)
        new_info = df.astype(old_info.dtype).as_matrix()

        del volume_file['synapse_info']  # todo: race condition
        try:
            del synapse_info
        except NameError:
            pass

        synapse_info = volume_file.create_dataset(
            'synapse_info',
            dtype=old_info.dtype,
            data=np.vstack((old_info, new_info))
        )

        synapse_info.attrs['headers'] = headers
        synapse_info.attrs['max_id'] = next_max_id


if __name__ == '__main__':
    if DEBUGGING:
        main(
            'credentials_dev.json',
            '1',
            '11524047',
            "../projects-2017/L1-CNS/skeletons",
            "../projects-2017/L1-CNS/synapse_volume.hdf5"
        )
    else:
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
            'skel_id', default=VOLUME_HDF5_PATH,
            help='ID of skeleton being segmented'
        )
        parser.add_argument(
            'ilastik_output_path', default=ILASTIK_OUTPUT_PATH,
            help="Path to ilastik's output directory"
        )
        parser.add_argument(
            'volume_hdf5_path',
            help='Path to the output HDF5 volume file'
        )
        args = parser.parse_args()

        main(args.credential_path, args.stack_id, args.skel_id, args.ilastik_output_path, args.volume_hdf5_path)
