from __future__ import division
import h5py
from matplotlib import pyplot as plt

from catpy import CatmaidClient, CoordinateTransformer


PATH = '../projects-2017/L1-CNS/tilewise_image_store.hdf5'
SLICE_PATH = 'slice_labels'
PIXEL_PATH = 'pixel_predictions'


def get_roi(path, z_slice, y_bounds, x_bounds):
    with h5py.File(path) as f:
        px_vol = f[PIXEL_PATH]
        pixel_data_xyc = px_vol[z_slice, y_bounds[0]:y_bounds[1], x_bounds[0]:x_bounds[1], :].transpose((1, 0, 2))
        sl_vol = f[SLICE_PATH]
        slice_data_xy = sl_vol[z_slice, y_bounds[0]:y_bounds[1], x_bounds[0]:x_bounds[1]].T

    return pixel_data_xyc, slice_data_xy


def plot_roi(z_slice, y_bounds, x_bounds):
    pixel_data_xyc, slice_data_xy = get_roi(PATH, z_slice, y_bounds, x_bounds)

    f, (px_ax, sl_ax) = plt.subplots(1, 2)
    px_ax.imshow(pixel_data_xyc.transpose((1, 0, 2)))
    sl_ax.imshow(slice_data_xy.T)
    return pixel_data_xyc, slice_data_xy


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


if __name__ == '__main__':
    plot_near_point((582, 7413, 9485), mode='middle')
