from __future__ import absolute_import
import os
import argparse
import sys
import random
from datetime import datetime, timedelta
import json
from skeleton_synapses.catmaid_interface import CatmaidAPI
from tqdm import tqdm

TIME_WINDOW_YEARS = 3
DEFAULT_SAMPLE_SIZE = 10
XY_PADDING = 10
Z_PADDING = 1
TRUSTED_USERS = ['caseysm', 'albert']
STACK_ID = 1


def copy_dict(d):
    return json.loads(json.dumps(d))


def get_exemplar_synapses(credentials_path, sample_size=DEFAULT_SAMPLE_SIZE, output_path=None):
    """
    Randomly sample connectors recently annotated by trusted experts and construct a padded bounding box of the 
    connector and its associated treenodes. Separate these boxes into slices to be passed to Ilastik as bookmarks.
    
    Parameters
    ----------
    credentials_path : str
        Path to CATMAID credentials JSON
    sample_size : int
        How many connectors to sample
    output_path : str
        Where to save the JSON output

    Returns
    -------
    dict
        {
            f'connector{conn_id}_{slice_idx}': [[xmin, ymin, zmin], [zmax, ymax, zmax]], 
            ...
        }
    """
    catmaid = CatmaidAPI.from_json(credentials_path)

    responses = []
    date_to = datetime.now()
    date_from = date_to - timedelta(days=365.25*TIME_WINDOW_YEARS)

    # todo: speed this up with threading?
    for username in tqdm(TRUSTED_USERS, desc='Getting connectors by trusted annotators', file=sys.stdout):
        responses += catmaid.get_connectors(username, date_from, date_to)

    print('Sampling responses')
    sample = random.sample({response[0] for response in responses}, sample_size)

    bounds = {
        'min': {
            'x': sys.float_info.max,
            'y': sys.float_info.max,
            'z': sys.float_info.max
        },
        'max': {
            'x': -sys.float_info.max,
            'y': -sys.float_info.max,
            'z': -sys.float_info.max
        }
    }

    connectors = {connector_id: copy_dict(bounds) for connector_id in sample}

    coord_transformer = catmaid.get_coord_transformer(STACK_ID)

    for response in tqdm(responses, desc='Getting bounding boxes for connectors', file=sys.stdout):
        connector_id = response[0]
        if connector_id not in connectors:
            continue

        proj_coords = [
            {
                'x': response[1][0],
                'y': response[1][1],
                'z': response[1][2]
            },
            {
                'x': response[6][0],
                'y': response[6][1],
                'z': response[6][2]
            },
            {
                'x': response[11][0],
                'y': response[11][1],
                'z': response[11][2]
            }
        ]

        stack_coords = [
            coord_transformer.project_to_stack(coord_dict) for coord_dict in proj_coords
        ]

        for dim in ['x', 'y', 'z']:
            dim_coords = [stack_coord[dim] for stack_coord in stack_coords]

            connectors[connector_id]['min'][dim] = min(dim_coords + [connectors[connector_id]['min'][dim]])
            connectors[connector_id]['max'][dim] = max(dim_coords + [connectors[connector_id]['max'][dim]])

    slices = dict()
    for connector_id, bounds in tqdm(connectors.items(), desc='Converting bounding boxes to slice ROIs', file=sys.stdout):
        if abs(int(bounds['min']['z']) - int(bounds['max']['z']) + 1) > 30:
            continue
        for idx, z_index in enumerate(range(int(bounds['min']['z']) - Z_PADDING, int(bounds['max']['z']) + 1 + Z_PADDING)):
            roi_xyz = [
                [bounds['min']['x'] - XY_PADDING, bounds['min']['y'] - XY_PADDING, z_index],
                [bounds['max']['x'] + XY_PADDING, bounds['max']['y'] + XY_PADDING, z_index + 1]
            ]

            slices['connector{}_{}'.format(connector_id, idx)] = roi_xyz

    if output_path:
        print('Writing output')
        with open(output_path, 'w') as f:
            json.dump(slices, f, indent=2, sort_keys=True)

    return slices


if __name__ == "__main__":
    DEBUGGING = True
    if DEBUGGING:
        print("USING DEBUG ARGUMENTS")

        CREDENTIALS_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        credentials_path = os.path.join(CREDENTIALS_ROOT, 'credentials_real.json')
        OUTPUT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'synapse_sample.json')
        sample_size = 50

        args_list = [credentials_path, sample_size, OUTPUT_PATH]

    else:
        parser = argparse.ArgumentParser()
        parser.add_argument('credentials_path',
                            help='Path to a JSON file containing CATMAID credentials (see credentials.jsonEXAMPLE)')
        parser.add_argument('-o', '--output', default='',
                            help="Optional path to output JSON file")

        args = parser.parse_args()
        args_list = [args.credentials_path, DEFAULT_SAMPLE_SIZE, args.output]

    output = get_exemplar_synapses(*args_list)
    print('{} slices returned'.format(len(output)))
