from __future__ import division
import os

from matplotlib import pyplot as plt

import numpy as np
import networkx as nx
import pandas as pd

from catpy.client import CatmaidClient, CatmaidClientApplication
from catpy.export import ExportWidget

from skeleton_synapses.catmaid_interface import CatmaidSynapseSuggestionAPI
from skeleton_synapses.constants import ROOT_DIR


STACK_ID = 1
RESOLUTION_XYZ = [3.8, 3.8, 50]
MAX_DISTANCE = 100


def add_node_to_graph(graph, tnid, parent_id, coords_xyz):
    node_id = int(tnid)
    parent_id = int(parent_id) if parent_id is not None else None
    graph.add_node(node_id,
                   id=node_id, parent_id=parent_id, coords_xyz=np.array(coords_xyz).astype(float), connectors=set())

    if parent_id is None:
        assert 'root' not in graph.graph
        graph.graph['root'] = tnid
    else:
        distance = np.linalg.norm(graph.node[parent_id]['coords_xyz'] - graph.node[node_id]['coords_xyz'])
        graph.add_edge(parent_id, node_id, distance=distance)


def response_to_graph(tnid_parentid_x_y_z):
    g = nx.DiGraph()
    for node_id, parent_id, x, y, z in tnid_parentid_x_y_z:
        add_node_to_graph(g, node_id, parent_id, [x, y, z])
    assert 'root' in g.graph
    return g


def geometry_to_graph(skeleton):
    g = nx.DiGraph()
    for tnid, treenode_data in skeleton['treenodes'].items():
        add_node_to_graph(g, tnid, treenode_data['parent_id'], treenode_data['location'])
    assert 'root' in g.graph
    return g


class ErrorAnalysisAPI(CatmaidClientApplication):
    def __init__(self, catmaid, stack_id=None):
        super(ErrorAnalysisAPI, self).__init__(catmaid)
        self.export = ExportWidget(catmaid)
        self.synsug = CatmaidSynapseSuggestionAPI(catmaid, stack_id)

    def get_intersecting_connectors(self, syn_obj_ids, tolerance):
        params = {'tolerance': tolerance, 'synapse_object_ids': syn_obj_ids}
        return self.get(('ext/synapsesuggestor/analysis', self.project_id, 'intersecting-connectors'), params)

    def get_skeleton_detail(self, skeleton_id, with_connectors=True):
        params = {
            'with_connectors': 'true' if with_connectors else 'false'
        }
        return self.get((self.project_id, 'skeletons', skeleton_id, 'compact-detail'), params)

    def get_skeleton_synapses(self, skeleton_id):
        params = {'skeleton_id': skeleton_id}
        return self.get(('ext/synapsesuggestor/analysis', self.project_id, 'skeleton-synapses'), params)


def dfs_to_graph(treenode_df, conn_df):
    tnid_parentid_x_y_z = treenode_df[['tnid', 'parent_id', 'x', 'y', 'z']].as_matrix()

    g = response_to_graph(tnid_parentid_x_y_z)

    for _, connector_row in conn_df.iterrows():
        if connector_row['relation_id'] not in [0, 1]:
            continue

        g.node[connector_row['tnid']]['connectors'].add(connector_row['connector_id'])

    return g


def skeleton_detail_to_dfs(skeleton_detail):
    tn_cols = ['tnid', 'parent_id', 'user_id', 'x', 'y', 'z', 'radius', 'confidence', 'edition_time', 'creation_time']
    treenode_df = pd.DataFrame(skeleton_detail[0], columns=tn_cols)

    conn_cols = ['tnid', 'connector_id', 'relation_id', 'x', 'y', 'z']
    conn_df = pd.DataFrame(skeleton_detail[1], columns=conn_cols)

    return treenode_df, conn_df


def response_to_df(response):
    assert 'data' in response
    assert 'columns' in response

    return pd.DataFrame(response['data'], columns=response['columns'])


def precision_recall(detected_syn_set, labelled_conn_set, intersecting_syn_set, intersecting_conn_set):
    precision = len(detected_syn_set.intersection(intersecting_syn_set)) / len(detected_syn_set)
    recall = len(labelled_conn_set.intersection(intersecting_conn_set)) / len(labelled_conn_set)

    return precision, recall


def get_data(cred_path, skeleton_id):
    catmaid = ErrorAnalysisAPI(CatmaidClient.from_json(cred_path), STACK_ID)

    skeleton_detail = catmaid.get_skeleton_detail(skeleton_id)

    treenodes_df, connectors_df = skeleton_detail_to_dfs(skeleton_detail)
    graph = dfs_to_graph(treenodes_df, connectors_df)

    skel_syn_response = catmaid.synsug.get_skeleton_synapses(skeleton_id)
    skel_syn_df = response_to_df(skel_syn_response)
    skel_syn_df['contact_nm'] = skel_syn_df['contact_px'] * np.prod(RESOLUTION_XYZ)
    skel_syn_df['size_nm'] = skel_syn_df['size_px'] * np.prod(RESOLUTION_XYZ)
    skel_syn_df['certainty_avg'] = 1 - skel_syn_df['uncertainty_avg']

    intersecting_conn_response = catmaid.get_intersecting_connectors(
        list(skel_syn_df['synapse']), MAX_DISTANCE
    )
    intersecting_conns_df = response_to_df(intersecting_conn_response)

    return treenodes_df, connectors_df, skel_syn_df, intersecting_conns_df, graph


def sweep_pr(skel_syn_df, connectors_df, intersecting_conns_df, constraint_name, bins=50):
    assert constraint_name in skel_syn_df.columns

    var_range = np.min(skel_syn_df[constraint_name]), np.max(skel_syn_df[constraint_name])
    values = np.linspace(*var_range, num=bins)

    labelled_conn_set = set(connectors_df['connector_id'])
    intersecting_syn_set = set(intersecting_conns_df['synapse_object_id'])

    precision_recalls = []

    for value in values:
        subset_skel_syn_df = skel_syn_df[skel_syn_df[constraint_name] > value]
        detected_syn_set = set(subset_skel_syn_df['synapse'])

        subset_intersecting_conns_df = intersecting_conns_df[
            intersecting_conns_df['synapse_object_id'].isin(detected_syn_set)
        ]
        intersecting_conn_set = set(subset_intersecting_conns_df['synapse_object_id'])

        precision_recalls.append(
            precision_recall(detected_syn_set, labelled_conn_set, intersecting_syn_set, intersecting_conn_set)
        )

    return np.array(precision_recalls)


if __name__ == '__main__':
    cred_path = os.path.join(ROOT_DIR, 'config', 'credentials', 'pogo.json')
    skel_id = 11524047

    treenodes_df, connectors_df, skel_syn_df, intersecting_conns_df, graph = get_data(cred_path, skel_id)

    prec_rec = sweep_pr(skel_syn_df, connectors_df, intersecting_conns_df, 'size_nm')
    precision, recall = prec_rec[:, 0], prec_rec[:, 1]

    plt.plot(recall, precision)
    plt.show()
