import os
import json
import collections
import numpy
import networkx as nx
from tree_util import partition
from catmaid_interface import CatmaidAPI
import tempfile


# Not  used in this file, but defined for cmd-line utilities to use.
CSV_FORMAT = { 'delimiter' : '\t', 'lineterminator' : '\n' }


NodeInfo = collections.namedtuple('NodeInfo', 'id x_px y_px z_px parent_id')
def parse_skeleton_swc(swc_path, x_res, y_res, z_res):
    """
    Parse the given swc file into a list of NodeInfo tuples.
    Coordinates are converted from nm to pixels.
    
    Note: In an swc file, a parentless node has parent_id = -1
    """
    node_infos = []
    with open(swc_path, 'r') as swc_file:
        for line in swc_file:
            if not line:
                continue
            node_id, _, x_nm, y_nm, z_nm, _, parent_id = line.split()

            # Convert from string
            node_id, parent_id = map(int, (node_id, parent_id))
            x_nm, y_nm, z_nm = map(float, (x_nm, y_nm, z_nm))

            # Convert to pixels
            x_px = int(x_nm / float(x_res))
            y_px = int(y_nm / float(y_res))
            z_px = int(z_nm / float(z_res))
            
            node_infos.append( NodeInfo(node_id, x_px, y_px, z_px, parent_id) )    
    return node_infos


def parse_skeleton_json_file(json_path, x_res, y_res, z_res):
    """
        Parse the given json file and return a list of NodeInfo tuples.
        Coordinates are converted from nm to pixels.

        Note: Mimicking the conventions above for swc files, 
              a parentless node will be assigned parent_id = -1    
        """
    assert os.path.splitext(json_path)[1] == '.json', "Skeleton file must end with .json"

    with open(json_path, 'r') as json_file:
        json_data = json.load(json_file)

    return parse_skeleton_json_data(json_data, x_res, y_res, z_res)


def parse_skeleton_json_data(json_data, x_res, y_res, z_res):
    """
    Parse the given json data and return a list of NodeInfo tuples.
    Coordinates are converted from nm to pixels.
    
    Note: Mimicking the conventions above for swc files, 
          a parentless node will be assigned parent_id = -1    
    """
    node_infos = []

    if len(json_data['skeletons']) == 0:
        raise Exception("File '{}' does not contain any skeleton data.".format( json_path ))
    if len(json_data['skeletons']) > 1:
        raise Exception("File '{}' contains more than one skeleton.  Can't process.".format( json_path ))
    node_dict = json_data['skeletons'].values()[0]['treenodes']

    for node_id, node_data in node_dict.iteritems():
        # Convert from string
        node_id = int( node_id )
        if node_data['parent_id']:
            parent_id = int( node_data['parent_id'] )
        else:
            parent_id = -1
        x_nm, y_nm, z_nm = map(float, node_data['location'])

        # Convert to pixels
        x_px = int(x_nm / float(x_res))
        y_px = int(y_nm / float(y_res))
        z_px = int(z_nm / float(z_res))
        
        node_infos.append( NodeInfo(node_id, x_px, y_px, z_px, parent_id) )
    
    skeleton_id = json_data['skeletons'].keys()[0]
    return skeleton_id, node_infos

def parse_skeleton_ids( json_path ):
    """
    Read the given skeleton json file and return the list of skeleton ids it contains.
    """
    assert os.path.splitext(json_path)[1] == '.json'
    with open(json_path, 'r') as json_file:
        json_data = json.load(json_file)
    return json_data['skeletons'].keys()

# Note that in ConnectorInfos, we do not transform the coordinates!
ConnectorInfo = collections.namedtuple('ConnectorInfo', 'id x_nm y_nm z_nm incoming_nodes outgoing_nodes')


def parse_connectors_from_file(json_path):
    """
        Parses skeleton files as returned by the CATMAID
        export widget's "Treenode and connector geometry" format.

        Read the skeleton json file and return:
        - A list of ConnectorInfo tuples
        - A dict of node -> connectors (regardless of whether the node is incoming or outgoing for the connector:
          { node : [connector_id, connector_id, ...] }

        """
    with open(json_path, 'r') as json_file:
        json_data = json.load(json_file)

    return parse_connectors_from_data(json_data)


def parse_connectors_from_data( json_data ):
    """
    Parses skeleton files as returned by the CATMAID
    export widget's "Treenode and connector geometry" format.
    
    Read the skeleton json data and return:
    - A list of ConnectorInfo tuples
    - A dict of node -> connectors (regardless of whether the node is incoming or outgoing for the connector:
      { node : [connector_id, connector_id, ...] }
    
    """
    connector_infos = []
    node_to_connector = {}

    if len(json_data['skeletons']) == 0:
        raise Exception("File '{}' does not contain any skeleton data.".format( json_path ))
    if len(json_data['skeletons']) > 1:
        raise Exception("File '{}' contains more than one skeleton.  Can't process.".format( json_path ))

    connection_dict = json_data['skeletons'].values()[0]['connectors']
    for connector_id, connector_data in connection_dict.iteritems():
        connector_id = int(connector_id)
        x_nm, y_nm, z_nm = map(float, connector_data['location'])
        #x_px = int(x_nm / float(X_RES))
        #y_px = int(y_nm / float(Y_RES))
        #z_px = int(z_nm / float(Z_RES))


        # Note the strange terminology of the json file:
        # The 'presynaptic_to' list means "all nodes in this list are presynaptic to the connector"
        #  (not "the connector is presynaptic to the following nodes")
        incoming = map(int, connector_data['presynaptic_to'] )
        outgoing = map(int, connector_data['postsynaptic_to'] )
        
        for node in incoming:
            node_connectors = node_to_connector.setdefault(node, [])
            node_connectors.append(connector_id)
        
        for node in outgoing:
            node_connectors = node_to_connector.setdefault(node, [])
            node_connectors.append(connector_id)
        
        connector_infos.append( ConnectorInfo(connector_id, x_nm, y_nm, z_nm, incoming, outgoing) )

    return connector_infos, node_to_connector

#
# A 'tree' is a networkx.DiGraph with a single root node (a node without parents)
#
def construct_tree(node_infos):
    """
    Construct a networkx.DiGraph() from a list of SWC NodeInfo instances.
    """
    tree = nx.DiGraph()
    for node_info in node_infos:
        if node_info.parent_id is None:
            tree.add_node(node_info.id)
            tree.graph['root'] = node_info.id
        else:
            tree.add_edge(node_info.parent_id, node_info.id)
        tree.node[node_info.id]['info'] = node_info
    return tree

def roi_around_point(coord_xyz, radius):
    """
    Produce a 3D roi (start, stop) tuple that surrounds the 
    node coordinates, with Z-thickness of 1.
    """
    coord_xyz = numpy.array(coord_xyz)
    start = coord_xyz - [radius, radius, 0]
    stop = coord_xyz + [radius+1, radius+1, 1]
    return numpy.array((tuple(start), tuple(stop)))

def roi_around_node(node_info, radius):
    coord_xyz = (node_info.x_px, node_info.y_px, node_info.z_px)
    return roi_around_point(coord_xyz, radius)

def branchwise_node_infos(tree):
    branches = []
    for branch in partition(tree):
        branch = filter(lambda node_id: node_id != -1, branch)
        branch_nodes = map(lambda node_id: tree.node[node_id]['info'], branch)
        branches.append(branch_nodes)
    return branches

def nodes_and_rois_for_tree(tree, radius):
    """
    Return a list of (coord, rois) for all nodes in the given skeleton,
    sorted in a reasonable order to increase cache hits
    (via CATMAID's partition() function).
    """
    branches = branchwise_node_infos(tree)
    branchwise_rois = []
    for branch in branches:
        rois = map(lambda n: roi_around_node(n, radius), branch)
        branchwise_rois.append( zip(branch, rois) )
    return branchwise_rois


class TransformedSkeleton(object):
    def __init__(self, json_path_or_data, resolution_xyz):
        """
        JSON should be in project coordinates. This transforms nodes into coordinates at stack scale, but still using 
        the project origin. Connector coordinates are not transformed
        
        Parameters
        ----------
        json_path
        resolution_xyz
        """

        if isinstance(json_path_or_data, str):
            skeleton_id, node_infos = parse_skeleton_json_file( json_path_or_data, *resolution_xyz )
            connector_infos_list, node_to_connector = parse_connectors_from_file(json_path_or_data)
        else:
            skeleton_id, node_infos = parse_skeleton_json_data(json_path_or_data, *resolution_xyz)
            connector_infos_list, node_to_connector = parse_connectors_from_data(json_path_or_data)
        
        self.skeleton_id = skeleton_id
        self.connector_infos = { info.id : info for info in connector_infos_list }
        self.node_to_connector = node_to_connector

        # Construct a networkx tree
        self.tree = construct_tree( node_infos )
        
        # And a list of the branches [[NodeInfo, NodeInfo,...], [NodeInfo, NodeInfo,...],...]
        self.branches = branchwise_node_infos(self.tree)

        self.x_res, self.y_res, self.z_res = resolution_xyz


class Skeleton(TransformedSkeleton):
    """
    Coordinates are not transformed in any way.
    """
    def __init__(self, json_path_or_data):
        super(Skeleton, self).__init__(json_path_or_data, (1, 1, 1))

    @classmethod
    def from_catmaid(cls, catmaid, skeleton_id, stack_id_or_title=None, save_path=None):
        """
        
        Parameters
        ----------
        catmaid : CatmaidAPI
        skeleton_id
        stack_id_or_title : int | str
            If set, coordinates will be transformed into stack coordinates. If not, project coordinates will be used.
        save_path : str
            If set, the treenode and connector geometry fetched from catmaid will be saved to this path
            
        Returns
        -------
        Skeleton
        """
        geom = catmaid.get_treenode_and_connector_geometry(skeleton_id, stack_id_or_title)
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(geom, f, sort_keys=True, indent=2)

        return cls(geom)


if __name__ == "__main__":
    X_RES = 3.8
    Y_RES = 3.8
    Z_RES = 50.0
    
    ROI_RADIUS = 100
    
    import os
    import skeleton_synapses
    package_dir = os.path.split(skeleton_synapses.__file__)[0]
    swc_path = os.path.join( package_dir, '../example/example_skeleton.swc' )

    node_infos = parse_swc( swc_path, X_RES, Y_RES, Z_RES )
    tree = construct_tree( node_infos )
    tree_nodes_and_rois = nodes_and_rois_for_tree(tree, radius=ROI_RADIUS)
    for i, branch_nodes_and_rois in enumerate(tree_nodes_and_rois[:2]):
        print "BRANCH {}:".format(i)
        for node, roi in branch_nodes_and_rois[:5]:
            roi = map(tuple, roi)
            print "coord = {}, roi = {}".format( node, roi )

