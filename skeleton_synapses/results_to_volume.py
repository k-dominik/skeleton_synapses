#!/usr/bin/env python

import h5py
import numpy as np
import os
import json
import argparse
import pandas as pd
from catmaid_interface import CatmaidAPI
import logging
import networkx as nx
from collections import namedtuple
from six import iteritems

logging.basicConfig(level=logging.DEBUG)

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')

VOLUME_HDF5_PATH = os.path.join(ROOT, 'projects-2017/synapse_volume.hdf5')
ILASTIK_OUTPUT_PATH = os.path.join(ROOT, 'projects-2017/L1-CNS/')

DTYPE = np.uint32  # can only store up to 2^24
UNKNOWN_LABEL = 0  # should be smallest label
BACKGROUND_LABEL = 1  # should be smaller than synapse labels
CHUNK_SHAPE = (256, 256, 1)  # yxz

CSV_HEADERS = [
    'synapse_id',
    'skeleton_id',
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
    'xmin',
    'xmax',
    'ymin',
    'ymax'
]


CSV_DTYPE = np.float64


REFRESH = False
DEBUGGING = False


def load_json(path):
    with open(path) as f:
        return {key: value for key, value in json.load(f).items() if not key.startswith('##')}


def reorder(lst, order):
    return [lst[idx] for idx in order]


def parse_slice_name(s):
    """
    Take string of the form 'x{x_coord}-y{y_coord}-z{z_coord}' and return the integer index and a dict of x,
    y and z coords.

    Parameters
    ----------
    s : str
        Slice name of the form 'x{x_coord}-y{y_coord}-z{z_coord}'

    Returns
    -------
    dict
        {'x': x_coord, 'y': y_coord, 'z': z_coord}
    """
    return {item[0]: int(item[1:]) for item in s.split('-')}


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


def ensure_volume_exists(volume_path, stack_info, force=False):
    """
    If the volume HDF5 file does not exist, or create it

    Parameters
    ----------
    volume_path : str
        Path to HDF5 volume file
    stack_info : dict
        Stack description as returned by CATMAID
    force : bool
        Whether to delete and recreate an empty HDF5 volume file if it already exists
    """
    if force or not os.path.isfile(volume_path):
        dimension = [stack_info['dimension'][dim] for dim in 'zyx']

        with h5py.File(volume_path, 'w') as volume_file:
            volume = volume_file.create_dataset(
                'volume',
                dimension,  # zyx
                chunks=CHUNK_SHAPE,
                fillvalue=UNKNOWN_LABEL,
                dtype=DTYPE
            )
            volume.attrs['unknown'] = UNKNOWN_LABEL
            volume.attrs['background'] = BACKGROUND_LABEL
            for key in ['translation', 'dimension', 'resolution']:
                volume.attrs[key] = json.dumps(stack_info[key])

            logging.debug('Creating synapse CSV array of shape {}'.format((0, len(CSV_HEADERS))))
            synapse_info = volume_file.create_dataset(
                'synapse_info',
                shape=(0, len(CSV_HEADERS)),
                dtype=CSV_DTYPE
            )
            synapse_info.attrs['headers'] = CSV_HEADERS
            synapse_info.attrs['max_id'] = BACKGROUND_LABEL


DfRow = namedtuple('DfRow', CSV_HEADERS)


class IntersectionDetector(object):
    def __init__(self, existing_df, existing_volume, slice_dir):
        """
        
        Parameters
        ----------
        existing_df : pandas.DataFrame
        existing_volume : h5py.Dataset
        slice_dir : str
        """
        self.df = existing_df
        self.volume = existing_volume
        self.slice_dir = slice_dir
        self.ancestry = nx.Graph()

    def process(self, new_df):
        background_label = self.volume.attrs['background']

        # merge old and new dataframes, noting where the boundary is and preventing ID clashes
        new_idx = len(self.df)
        current_max = self.df['synapse_id'].max()
        new_df['synapse_id'] += current_max if not np.isnan(current_max) else background_label
        self.df = pd.concat((self.df, new_df))

        # find all adjacencies/intersections in new synapse slices
        for idx, row in enumerate(new_df.itertuples(False, 'DfRow')):
            up_to = new_idx + idx
            self.ancestry.add_edges_from(
                (row['synapse_id'], existing_id) for existing_id in self.bbox_intersects(row, up_to)['synapse_id']
            )

        id_mapping = self._get_id_mapping()

        # relabel existing image labels and all df rows
        for old_id, new_id in iteritems(id_mapping):
            to_relabel = self.df['synapse_id'] == old_id
            for row in self.df[to_relabel].itertuples(False, 'DfRow'):
                slice_im = np.array(self.volume[row['z_px'], row['ymin']:row['ymax'], row['xmin']:row['xmax']])
                slice_im[slice_im == old_id] = new_id  # could maybe be done in one step
                self.volume[row['z_px'], row['ymin']:row['ymax'], row['xmin']:row['xmax']] = slice_im

            self.df.loc[to_relabel, 'synapse_id'] = new_id

        # dump label images into volume, remapping synapse IDs (and add background label)
        filenames = os.listdir(self.slice_dir)
        for idx, filename in enumerate(filenames, 1):
            logging.debug('Committing node slice {} of {}'.format(idx, len(filenames)))
            file_path = os.path.join(self.slice_dir, filename)
            logging.debug('Extracting data from {}'.format(file_path))
            topleft = parse_slice_name(os.path.splitext(filename)[0])
            with h5py.File(file_path, 'r') as synapse_file:
                slice_data = np.array(synapse_file['data'])[:, :, 0, 0]

                for old_id in slice_data.uniques():
                    if old_id in id_mapping:
                        slice_data[slice_data == old_id] = id_mapping[old_id]

                bbox = topleft_to_bounding_box(topleft, slice_data.shape[0])
                slice_data[slice_data == 0] = background_label

                # slice_data is in [x, y], where it needs to be in [y, x]
                self.volume[bbox['z'], bbox['y'][0]:bbox['y'][1], bbox['x'][0]:bbox['x'][1]] = slice_data.T

        # todo: clear rows of table if their bounding box has been entirely replaced?
        # Might be replaced by > 1 slice

    def _get_id_mapping(self):
        mappings = dict()
        for component in nx.connected_components(self.ancestry):
            master = min(component)
            for synapse_id in component:
                if synapse_id != master:
                    mappings[synapse_id] = master

        return mappings

    def bbox_intersects(self, new_row, up_to=None):
        df = self.df if up_to is None else self.df.iloc[:up_to]
        return df[
            np.abs(df['z_px'] - new_row['z_px']) <= 1 and  # same or adjacent Z slice
            not (
                df['xmin'] > new_row['xmax'] + 1 or    # new slice isn't to the left
                df['xmax'] < new_row['xmin'] - 1       # new slice isn't to the right
            ) and
            not (
                df['ymin'] > new_row['ymax'] + 1 or    # new slice isn't below
                df['ymax'] < new_row['ymin'] - 1       # new slice isn't above
            )
        ]


# todo: deal with intersecting synapses (give them same ID)
def main(credential_path, stack_id, skel_id, ilastik_output_path=ILASTIK_OUTPUT_PATH,
         volume_hdf5_path=VOLUME_HDF5_PATH, force=REFRESH):
    catmaid = CatmaidAPI.from_json(credential_path)
    stack_info = catmaid.get_stack_info(stack_id)

    ensure_volume_exists(
        volume_hdf5_path,
        stack_info,
        force
    )

    skeleton_hdf5_path = os.path.join(ilastik_output_path, str(skel_id), 'synapse_cc')
    skeleton_csv_path = os.path.join(
        ilastik_output_path, str(skel_id), 'skeleton-{}-synapses.csv'.format(skel_id)
    )

    logging.info('Opening volume file at {}'.format(volume_hdf5_path))
    with h5py.File(volume_hdf5_path, 'r+') as volume_file:
        volume = volume_file['volume']
        synapse_info = volume_file['synapse_info']

        max_id = synapse_info.attrs['max_id']
        background_label = volume.attrs['background']
        filenames = os.listdir(skeleton_hdf5_path)
        for idx, filename in enumerate(filenames, 1):
            logging.debug('Committing node slice {} of {}'.format(idx, len(filenames)))
            file_path = os.path.join(skeleton_hdf5_path, filename)
            logging.debug('Extracting data from {}'.format(file_path))
            topleft = parse_slice_name(os.path.splitext(filename)[0])
            with h5py.File(file_path, 'r') as synapse_file:
                slice_data = np.array(synapse_file['data'])[:, :, 0, 0]
                bbox = topleft_to_bounding_box(topleft, slice_data.shape[0])
                slice_data[slice_data != 0] = slice_data[slice_data != 0] + max_id  # ensure no label clashes
                slice_data[slice_data == 0] = background_label

                # slice_data is in [x, y], where it needs to be in [y, x]
                volume[bbox['z'], bbox['y'][0]:bbox['y'][1], bbox['x'][0]:bbox['x'][1]] = slice_data.T

        headers = synapse_info.attrs['headers']

        synapse_csv = pd.read_csv(skeleton_csv_path, delimiter='\t')
        synapse_csv['synapse_id'] += max_id
        old_info = np.array(synapse_info)
        new_info = synapse_csv.astype(old_info.dtype).as_matrix()

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
        synapse_info.attrs['max_id'] = synapse_csv['synapse_id'].max()


if __name__ == '__main__':
    if DEBUGGING:
        main(
            'credentials_dev.json',
            '1',
            '11524047',
            "../projects-2017/L1-CNS/skeletons",
            "../projects-2017/L1-CNS/synapse_volume.hdf5",
            True
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
        parser.add_argument(
            '-f', '--force', type=int, default=0,
            help='Create a new output HDF5 regardless of whether or not it exists'
        )
        args = parser.parse_args()

        main(args.credential_path, args.stack_id, args.skel_id, args.ilastik_output_path, args.volume_hdf5_path,
             args.force)
