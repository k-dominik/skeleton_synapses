#!/usr/bin/env python
import os

import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure

from catpy import CatmaidClient
from skeleton_synapses.catmaid_interface import CatmaidSynapseSuggestionAPI

CRED_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'skeleton_synapses', 'credentials_catsop.json')
HDF5_PATH = '../projects-2017/L1-CNS/tilewise_image_store.hdf5'
INNER_PATH = 'slice_labels'

cat = CatmaidSynapseSuggestionAPI(CatmaidClient.from_json(CRED_PATH))


def get_data(slice_ids, bounds_dict):
    with h5py.File(HDF5_PATH, 'r') as f:
        label_arr = f[INNER_PATH]
        data = np.array(label_arr[
            bounds_dict['zmin']:bounds_dict['zmax']+1,
            bounds_dict['ymin']:bounds_dict['ymax']+1,
            bounds_dict['xmin']:bounds_dict['xmax']+1
        ])

    for val in np.unique(data):
        if val not in slice_ids:
            data[data == val] = 0

    return data


def plot_block(data):
    verts, faces, normals, values = measure.marching_cubes(data, 1)
    # Display resulting triangular mesh using Matplotlib. This can also be done
    # with mayavi (see skimage.measure.marching_cubes_lewiner docstring).
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces])
    mesh.set_edgecolor('k')
    ax.add_collection3d(mesh)

    ax.set_xlabel("x-axis (maybe)")
    ax.set_ylabel("y-axis (maybe)")
    ax.set_zlabel("z-axis (maybe)")

    plt.tight_layout()
    plt.show()


def visualise_synapse(synapse_id):
    response = cat.get_synapse_bounds((synapse_id, ))[str(synapse_id)]
    print(response)
    slice_ids = response['synapse_slice_ids']
    bounds = response['extents']

    data = get_data(slice_ids, bounds)
    plot_block(data)


if __name__ == '__main__':
    # import sys
    #
    # syn_id = int(sys.argv[1])
    # 115163
    visualise_synapse(115163)
