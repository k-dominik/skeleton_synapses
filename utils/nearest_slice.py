from math import hypot
import os

DEBUG = True
DEBUG_ROOT = r'/home/cbarnes/work/synapse_detection/remote_results/L1-CNS/skeletons/11524047'


def path2coords(path):
    fname = os.path.basename(path)
    s = os.path.splitext(fname)[0]

    return {coord_str[0]: int(coord_str[1:]) for coord_str in s.split('-')}


def get_filename_for_coords(coords, directory, topleft=True):
    file_list = os.listdir(directory)
    coord_dist_list = []
    for filename in file_list:
        file_coords = path2coords(filename)
        if file_coords['z'] != coords['z']:
            continue
        if (not topleft) or (topleft and file_coords['x'] < coords['x'] and file_coords['y'] < coords['y']):
            coord_dist_list.append((hypot(file_coords['x'] - coords['x'], file_coords['y'] - coords['y']), filename))

    return sorted(coord_dist_list)[0][1]


if __name__ == '__main__':
    if DEBUG:
        directory = os.path.join(DEBUG_ROOT, 'raw')
        coords = {
            'x': 8768,
            'y': 8173,
            'z': 642
        }
        filename = get_filename_for_coords(coords, directory)
        print(filename)
