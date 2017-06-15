from __future__ import division
import h5py
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm

from catpy import CatmaidClient, CoordinateTransformer


PATH = '../projects-2017/L1-CNS/tilewise_image_store.hdf5'
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


def point2roi(zyx_px, mode='topleft', side=512):
    """Assumes top left"""
    if mode == 'topleft':
        return zyx_px[0], (zyx_px[1], zyx_px[1] + side), (zyx_px[2], zyx_px[2] + side)
    elif mode == 'middle':
        radius = side // 2
        return zyx_px[0], (zyx_px[1] - radius, zyx_px[1] + radius), (zyx_px[2] - radius, zyx_px[2] + radius)


def plot_near_point(zyx_px, mode='topleft', side=512):
    return plot_roi(*point2roi(zyx_px, mode, side))


def plot_near_node(node_id, cred_path=None):
    if cred_path is None:
        cred_path = '../skeleton_synapses/credentials_dev.json'
    catmaid = CatmaidClient.from_json(cred_path)
    transformer = CoordinateTransformer.from_catmaid(catmaid, 1)
    tnid, x, y, z = catmaid.post((catmaid.project_id, 'node', 'get_location'), {'tnid': node_id})
    assert tnid == node_id
    coords = transformer.project_to_stack({'x': x, 'y': y, 'z': z})

    return plot_near_point([int(coords[dim]) for dim in 'zyx'], mode='middle')


def check_hdf5(path=PATH, inner_path=INNER_PATH):
    z_sums = []
    with h5py.File(path) as f:
        dset = f[inner_path]
        for z_idx in tqdm(range(dset.shape[0])):
            z_sums.append(0)
            for y_idx in tqdm(range(dset.shape[1])):
                z_sums[-1] += dset[z_idx, y_idx, :].sum()

    fig, ax = plt.subplots()

    ax.plot(z_sums)
    plt.show()


if __name__ == '__main__':
    check_hdf5()
