import h5py
from matplotlib import pyplot as plt
import numpy as np

PATH = '/home/cbarnes/work/synapse_detection/skeleton_synapses/projects-2017/L1-CNS/tilewise_image_store.hdf5'
INNER_PATH = 'volume'


def get_roi(path, inner_path, z_slice, y_bounds, x_bounds):
    with h5py.File(path) as f:
        vol = f[inner_path]
        roi_yx = np.array(vol[z_slice, y_bounds[0]:y_bounds[1], x_bounds[0]:x_bounds[1]]).T

    return roi_yx


def plot_roi(z_slice, y_bounds, x_bounds):
    data = get_roi(PATH, INNER_PATH, z_slice, y_bounds, x_bounds)

    f, ax = plt.subplots()

    ax.imshow(data.T)
    return data


def point2roi(zyx_px, side=512):
    """Assumes top left"""
    return zyx_px[0], (zyx_px[1], zyx_px[1] + side), (zyx_px[2], zyx_px[2] + side)


def plot_near_point(zyx_px):
    return plot_roi(*point2roi(zyx_px))

