# A 'tree' is a networkx.DiGraph with a single root node (a node without parents)

# Copied from CATMAID/django/applications/catmaid/control/tree_util.py
def partition(tree, root_node=None):
    """ Partition the tree as a list of sequences of node IDs,
    with branch nodes repeated as ends of all sequences except the longest
    one that finishes at the root.
    Each sequence runs from an end node to either the root or a branch node. """
    distances = edge_count_to_root(tree, root_node=root_node) # distance in number of edges from root
    seen = set()
    # Iterate end nodes sorted from highest to lowest distance to root
    endNodeIDs = (nID for nID in tree.nodes() if 0 == len(tree.successors(nID)))
    for nodeID in sorted(endNodeIDs, key=distances.get, reverse=True):
        sequence = [nodeID]
        parentID = next(tree.predecessors_iter(nodeID), None)
        while parentID is not None:
            sequence.append(parentID)
            if parentID in seen:
                break
            seen.add(parentID)
            parentID = next(tree.predecessors_iter(parentID), None)

        if len(sequence) > 1:
            yield sequence

# Copied from CATMAID/django/applications/catmaid/control/tree_util.py
def edge_count_to_root(tree, root_node=None):
    """ Return a map of nodeID vs number of edges from the first node that lacks predecessors (aka the root). If root_id is None, it will be searched for."""
    distances = {}
    count = 1
    current_level = [root_node if root_node else find_root(tree)]
    next_level = []
    while current_level:
        # Consume all elements in current_level
        while current_level:
            node = current_level.pop()
            distances[node] = count
            next_level.extend(tree.successors_iter(node))
        # Rotate lists (current_level is now empty)
        current_level, next_level = next_level, current_level
        count += 1
    return distances

# Copied from CATMAID/django/applications/catmaid/control/tree_util.py
def find_root(tree):
    """ Search and return the first node that has zero predecessors.
    Will be the root node in directed graphs.
    Avoids one database lookup. """
    for node in tree:
        if not next(tree.predecessors_iter(node), None):
            return node
