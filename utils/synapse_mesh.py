#!/usr/bin/env python
"""
None of the plotting works, but making the mesh does?
"""
import os
import warnings
import json
import errno

import h5py
import numpy as np
import pickle
import networkx as nx

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure

from catpy import CatmaidClient
from skeleton_synapses.catmaid_interface import CatmaidSynapseSuggestionAPI

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

CRED_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'skeleton_synapses', 'credentials_catsop.json')
# PROJECTS_ROOT = '..'
PROJECTS_ROOT = '/run/user/1000/gvfs/sftp:host=10.101.50.112/home/barnesc/synapse_detection/skeleton_synapses'
# HDF5_PATH = '../projects-2017/L1-CNS/tilewise_image_store.hdf5'
HDF5_PATH = os.path.join(PROJECTS_ROOT, 'projects-2017/L1-CNS/tilewise_image_store.hdf5')
INNER_PATH = 'slice_labels'

catmaid = CatmaidSynapseSuggestionAPI(CatmaidClient.from_json(CRED_PATH))


def get_data(slice_ids, bounds_dict, path=None):
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

    if path:
        save_data(data, path)

    return data


def save_data(data, path):
    save_fns = {
        '.pickle': _save_pickle,
        '.hdf5': _save_hdf5,
        '.hdf': _save_hdf5,
        '.npy': _save_npy,
        '.json': _save_json
    }

    ext = os.path.splitext(path)[1]

    save_fns.get(ext, _save_default)(data, path)


def _save_pickle(data, path):
    with open(path, 'w') as f:
        pickle.dump(data, f)


def _save_hdf5(data, path):
    with h5py.File(path, 'w') as f:
        f.create_dataset('data', data=data)


def _save_npy(data, path):
    np.save(path, data)


def _save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data.tolist(), f)


def _save_default(data, path):
    warnings.warn('Unknown file type, saving in pickle format')
    _save_pickle(data, path)



def _plot_mpl(verts, faces, normals=None, values=None):

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


def _plot_visvis(verts, faces, normals, values):
    import visvis as vv
    vv.mesh(np.fliplr(verts), faces, normals, values)
    vv.use().Run()


def _plot_mayavi(verts, faces, normals=None, values=None):
    from mayavi import mlab
    mlab.triangular_mesh(verts[:, 0], verts[:, 1], verts[:, 2], faces)
    mlab.show()


def _plot_pyqt(verts, faces, normals, values=None):
    app = QApplication([])
    view = GLViewWidget()

    mesh = MeshData(verts / 100, faces)  # scale down - because camera is at a fixed position
    # or mesh = MeshData(smooth_vertices / 100, faces)
    mesh._vertexNormals = normals
    # or mesh._vertexNormals = smooth_normals

    item = GLMeshItem(meshdata=mesh, color=[1, 0, 0, 1], shader="normalColor")

    view.addItem(item)
    view.show()
    app.exec_()


def plot_block(data):
    # fig, ax_arr = plt.subplots(2, 4)
    # for z_idx, ax in enumerate(ax_arr.flatten()):
    #     try:
    #         ax.imshow(data[z_idx, :, :])
    #         ax.set_title('Z index {}'.format(z_idx))
    #     except IndexError:
    #         pass

    with open('data.pickle', 'w') as f:
        pickle.dump(data, f)

    # data2 = np.zeros(data.shape)
    # data2[data > 1] = 2

    # with open('data.pickle', 'w') as f:
    #     pickle.dump(data, f)

    verts, faces, normals, values = measure.marching_cubes(data, level=1, spacing=(50, 3.8, 3.8))


    with open('verts_faces.pickle', 'w') as f:
        pickle.dump((verts, faces), f)

    # verts, normals, faces = mc.march(data, 4)  # 4 smoothing rounds

    # _plot_pyqt(verts, faces, normals)

    # verts, faces, normals, values = measure.marching_cubes_lewiner(data, level=1, spacing=(50, 3.8, 3.8))


    # _plot_mayavi(verts, faces, normals)


def synapse_ids_to_data(synapse_ids, z_padding=1, xy_padding=10):
    response = catmaid.get_synapse_bounds(synapse_ids, z_padding=z_padding, xy_padding=xy_padding)

    for synapse_id, d in response.items():
        slice_ids = d['synapse_slice_ids']
        bounds = d['extents']

        yield int(synapse_id), get_data(slice_ids, bounds)


def save_data_multi(id_arr_list, root='', suffix='', ext='.npy'):
    if not root:
        root = os.path.dirname(os.path.realpath(__file__))
    mkdir_p(root)

    for syn_id, data in id_arr_list:
        fname = '{}{}.{}'.format(syn_id, suffix, ext.strip('.'))
        save_data(data, os.path.join(root, fname))


def mkdir_p(path):
    """
    Like the bash command 'mkdir -p'
    """
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def visualise_synapse(synapse_id):
    response = catmaid.get_synapse_bounds((synapse_id,), z_padding=1)[str(synapse_id)]
    print(response)
    slice_ids = response['synapse_slice_ids']
    bounds = response['extents']

    data = get_data(slice_ids, bounds)
    plot_block(data)


def to_obj(path, syn_id):
    data = dict(synapse_ids_to_data([syn_id]))[syn_id]
    verts, faces = measure.marching_cubes(data, level=1, spacing=(50, 3.8, 3.8))

    rows = [['## Mesh created by skimage.measure.marching_cubes_lewiner']]

    # vertices
    rows.append(['# List of xyz vertices'])
    for vert in np.fliplr(verts):
        rows.append(['v'] + [str(item) for item in vert])

    # # normals
    # rows.append(['# List of vertex normals in xyz'])
    # for normal in np.fliplr(normals):
    #     rows.append(['vn'] + [str(item) for item in normal])

    # faces
    rows.append(['# List of faces as triangles specified by 3 vertex indices'])
    for face in faces:
        rows.append(['f'] + [str(item) for item in face])

    with open(path, 'w') as f:
        for row in rows:
            f.write(' '.join(row) + '\n')


def catmull_clark(verts, faces, repeats=1):
    """
    WIP

    Requires new skimage and numpy

    https://en.wikipedia.org/wiki/Catmull%E2%80%93Clark_subdivision_surface#Recursive_evaluation
    """

    assert isinstance(repeats, int)

    if repeats < 1:
        return verts, faces

    facepoints = verts[faces].mean(axis=2)
    """F*3 array of coordinates of new face points"""

    # edges
    # 0 = 0 -> 1
    # 1 = 1 -> 2
    # 2 = 2 -> 0

    edges = np.concatenate((faces[:, (0, 1)], faces[:, (1, 2)], faces[:, (2, 0)]), axis=2)
    """F*2*3 array of vertex indices defining edges, where axis 0 is which triangle, and axis 2 is which edge of 
    the triangle"""

    edges_flat = np.reshape(edges, (faces.shape[0] * 3, 2))

    original = nx.Graph()
    original.add_edges_from(edges_flat, type='P')
    for node, data in original.nodes(data=True):
        data['coords'] = verts[node, :]

    facepoint_dict = {frozenset(vert_idxs): facepoint for vert_idxs, facepoint in zip(verts, facepoints)}

    for src, tgt, data in original.edges(data=True):
        if 'midpoint' not in data:
            data['midpoint'] = np.mean([src['coords'], tgt['coords']], axis=0)

        neighbours = list(nx.common_neighbors(original, src, tgt))

        facepoint1, facepoint2 = [facepoint_dict[frozenset([src, tgt, n])] for n in neighbours]

        data['edgepoint'] = {
            'coords': np.mean([src['coords'], tgt['coords'], facepoint1, facepoint2], axis=0),
            'src': min(src, tgt),
            'tgt': max(src, tgt),
        }

    for verts, facepoint_coords in facepoint_dict.items():
        pass


    edges_unique = np.unique(edges_flat, axis=0)

    sorted_edges = np.sort(edges, axis=1)
    """Edges sorted so that the lower index is always first"""

    faces_by_edge = []
    idx_counter = 0
    unique_edges = dict()

    for face in faces:
        faces_by_edge.append([])
        for idxs in [(0, 1), (1, 2), (0, 2)]:
            pair = tuple(sorted([face[idxs[0]], face[idxs[1]]]))
            if pair not in unique_edges:
                unique_edges[pair] = idx_counter
                idx_counter += 1
            faces_by_edge[-1].append(unique_edges[pair])
        faces_by_edge[-1] = sorted(faces_by_edge[-1])


    edge_midpoints = np.squeeze(verts[edges].mean(axis=1))
    """F*3*3 array of coordinates which are the midpoints of every edge, where axis 1 is which edge of the triangle, 
    and axis 2 is the 3D coordinate"""

    return catmull_clark(new_verts, new_faces, repeats-1)





if __name__ == '__main__':
    # import sys
    #
    # syn_id = int(sys.argv[1])
    # 115163
    synapse_ids = [115163, 114581, 114492, 113739, 113639]
    id_arrs = synapse_ids_to_data(synapse_ids)
    save_data_multi(id_arrs, root='synapse_examples', suffix='_zyx_50_3.8_3.8')

    to_obj('test.obj', synapse_ids[0])
