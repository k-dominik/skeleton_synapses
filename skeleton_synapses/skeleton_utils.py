import os
import json
import collections
import numpy
import networkx as nx
from tree_util import partition

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

def parse_skeleton_json(json_path, x_res, y_res, z_res):
    """
    Parse the given json file and return a list of NodeInfo tuples.
    Coordinates are converted from nm to pixels.
    
    Note: Mimicking the conventions above for swc files, 
          a parentless node will be assigned parent_id = -1    
    """
    assert os.path.splitext(json_path)[1] == '.json'
    
    node_infos = []
    with open(json_path, 'r') as json_file:
        json_data = json.load(json_file)
    
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
    return node_infos

def parse_skeleton_ids( json_path ):
    """
    Read the given skeleton json file and return the list of skeleton ids it contains.
    """
    assert os.path.splitext(json_path)[1] == '.json'
    with open(json_path, 'r') as json_file:
        json_data = json.load(json_file)
    return json_data['skeletons'].keys()

# Note that in ConnectorInfos, we keep the coordinates in nanometers!
ConnectorInfo = collections.namedtuple('ConnectorInfo', 'id x_nm y_nm z_nm incoming_nodes outgoing_nodes')
def parse_connectors( json_path ):
    connector_infos = []
    with open(json_path, 'r') as json_file:
        json_data = json.load(json_file)

    if len(json_data['skeletons']) == 0:
        raise Exception("File '{}' does not contain any skeleton data.".format( json_path ))
    if len(json_data['skeletons']) > 1:
        raise Exception("File '{}' contains more than one skeleton.  Can't process.".format( json_path ))

    connection_dict = json_data['skeletons'].values()[0]['connectors']
    for connector_id, connector_data in connection_dict.iteritems():
        connector_id = int(connector_id)
        x_nm, y_nm, z_nm = map(float, connector_data['location'])

        # Note the strange terminology of the json file:
        # The 'presynaptic_to' list means "all nodes in this list are presynaptic to the connector"
        #  (not "the connector is presynaptic to the following nodes")
        incoming = map(int, connector_data['presynaptic_to'] )
        outgoing = map(int, connector_data['postsynaptic_to'] )

        connector_infos.append( ConnectorInfo(connector_id, x_nm, y_nm, z_nm, incoming, outgoing) )

    return connector_infos

#
# A 'tree' is a networkx.DiGraph with a single root node (a node without parents)
#
def construct_tree(node_infos):
    """
    Construct a networkx.DiGraph() from a list of SWC NodeInfo instances.
    """
    tree = nx.DiGraph()
    for node_info in node_infos:
        tree.add_edge( node_info.parent_id, node_info.id )
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

def nodes_and_rois_for_tree(tree, radius):
    """
    Return a list of (coord, rois) for all nodes in the given skeleton,
    sorted in a reasonable order to increase cache hits
    (via CATMAID's partition() function).
    """
    tree_coords_and_rois = []
    for sequence in partition(tree):
        branch_coords_and_rois = []
        for node_id in sequence:
            if node_id != -1:
                node_info = tree.node[node_id]['info']
                coord_xyz = (node_info.x_px, node_info.y_px, node_info.z_px)
                branch_coords_and_rois.append( ( node_info, roi_around_point(coord_xyz, radius) ) )
        tree_coords_and_rois.append(branch_coords_and_rois)
    return tree_coords_and_rois

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

