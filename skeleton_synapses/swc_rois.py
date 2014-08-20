import collections
import numpy
import networkx as nx
from tree_util import partition

NodeInfo = collections.namedtuple('NodeInfo', 'id x y z parent_id')
def parse_swc(swc_path, x_res, y_res, z_res):
    """
    Parse the given swc file into a list of NodeInfo tuples.
    Coordinates are converted from nm to pixels.
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

def coords_and_rois_for_tree(tree, radius):
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
                info = tree.node[node_id]['info']
                coord_xyz = (info.x, info.y, info.z)
                branch_coords_and_rois.append( ( coord_xyz, roi_around_point(coord_xyz, radius) ) )
        tree_coords_and_rois.append(branch_coords_and_rois)
    return tree_coords_and_rois

if __name__ == "__main__":
    X_RES = 3.8
    Y_RES = 3.8
    Z_RES = 50.0
    
    ROI_RADIUS = 100
    
    import sys
    sys.argv.append('/home/anna/scripts/fruitfly/example_skeleton.swc')
    #sys.argv.append('/Users/bergs/Documents/workspace/anna_scripts/3034133.swc')
    
    swc_path = sys.argv[1]

    node_infos = parse_swc( swc_path, X_RES, Y_RES, Z_RES )
    tree = construct_tree( node_infos )
    tree_coords_and_rois = coords_and_rois_for_tree(tree, radius=ROI_RADIUS)
    for branch_coords_and_rois in tree_coords_and_rois[:5]:
        print "NEXT BRANCH"
        for coord, roi in branch_coords_and_rois[:5]:
            print "coord = {}, roi = {}".format( coord, roi )

