from __future__ import division
import h5py
import numpy as np
from matplotlib import pyplot as plt
import argparse
import os
from nearest_slice import get_filename_for_coords

plt.style.use('ggplot')

OFFSET = -121

DEFAULT_INNER_PATH = 'data'
DEFAULT_SHAPE = '16:9'

DEBUG_ROOT = r'/home/barnesc/synapse_detection/skeleton_synapses/projects-2017/L1-CNS/skeletons/11524047'

DIRS = ['raw', 'predictions', 'synapse_cc', 'segmentation']

types_kwargs = {
    'raw': {'cmap': 'gray'},
    'predictions': {},
    'segmentation': {'cmap': 'Set1'},
    'synapse_cc': {'cmap': 'Accent'}
}


def get_shape(number, width, height):
    """

    Parameters
    ----------
    number : int
        Number of images to be shown
    width
    height

    Returns
    -------
    (int, int)
        Number of rows (height), number of columns (width)
    """
    target_ratio = float(width)/float(height)
    nrows = 1
    ncols = 1

    while ncols * nrows < number:
        if ncols / nrows < target_ratio:
            ncols += 1
        else:
            nrows += 1

    return nrows, ncols


def get_data(file_path, inner_path):
    with h5py.File(file_path, 'r') as f:
        data = np.array(f[inner_path])

    return data


def view_data(root_dir, file_name, **kwargs):
    inner_path = kwargs.get('inner_path', DEFAULT_INNER_PATH)

    fig, ax_arr = plt.subplots(2, 2)
    # fig.suptitle('{} from {}'.format(inner_path, file_name))
    ax_dict = {im_type: ax for im_type, ax in zip(DIRS, list(ax_arr.flatten()))}

    for im_type in DIRS:
        file_path = os.path.join(root_dir, im_type, file_name)
        data = get_data(file_path, inner_path)
        ax = ax_dict[im_type]
        ax.set_title(im_type)

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        assert data.shape[2] == 1
        try:
            ax.imshow(np.transpose(data, (1, 0, 2, 3)).squeeze(), **types_kwargs[im_type])
        except TypeError:
            pass

    fig.tight_layout()
    # fig.set_size_inches(6, 6)
    plt.show()


if __name__ == "__main__":
    DEBUG = True
    if DEBUG:
        DEBUG_ROOT = '/home/cbarnes/work/synapse_detection/remote_results/L1-CNS/skeletons/11524047'
        filenames = sorted(os.listdir(os.path.join(DEBUG_ROOT, 'raw')))
        # filename = filenames[600]
        coords = {
            'x': 9507,
            'y': 8451,
            'z': 616 - OFFSET
        }
        filename = get_filename_for_coords(coords, os.path.join(DEBUG_ROOT, 'raw'))

        view_data(DEBUG_ROOT, filename, inner_path='data')
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument('-p', '--inner-path', default=DEFAULT_INNER_PATH, help='Path inside the HDF5 file to the image')
        parser.add_argument('-s', '--shape', default=DEFAULT_SHAPE, help='X:Y shape of subplots. Rows are filled first.')
        parser.add_argument('root_dir')
        parser.add_argument('file_name')

        args = parser.parse_args()

        view_data(args.root_dir, args.file_name, inner_path=args.inner_path)
