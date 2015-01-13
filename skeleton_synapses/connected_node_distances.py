import csv
from skeleton_utils import parse_connectors, CSV_FORMAT

def connected_node_distances( skeleton_json_path,
                              raw_detection_csv_path,
                              merged_detection_csv_path,
                              output_csv_path ):
    """
    Useful for verifying the synapse detection procedure against annotated synapses (for which connectors exist).
    
    Locate all the "connected nodes" in the given skeleton 
        (i.e. nodes which are either ingoing to or outgoing from a connector in the skeleton).
    For each "connected node", determine the synapse_id it 
        is associated with (if any) using the given raw detections file.
        If there is more than one synapse associated with it, choose the one with 
        the smallest "membrane distance" to the node.
    For the synapse, extract the final "membrane distance" from the 
        given merged detections file, which lists the minimum "membrane distance" for each synapse.    
    """
    connector_infos = parse_connectors( skeleton_json_path )

    raw_detections, output_columns = _load_raw_detections( raw_detection_csv_path )
    merged_detections = _load_merged_detections( merged_detection_csv_path )

    nodes_without_detections = []
    with open( output_csv_path, 'w' ) as output_csv:
        csv_writer = csv.DictWriter(output_csv, output_columns, **CSV_FORMAT)
        csv_writer.writeheader()

        for connector_info in connector_infos:
            output_row = _get_row_dict( raw_detections, merged_detections, connector_info )
            if output_row["synapse_id"] == -1:
                nodes_without_detections.append( ( output_row["node_id"], output_row["connector_id"] ) )
            csv_writer.writerow( output_row )

    return len(connector_infos), nodes_without_detections

def _get_row_dict( raw_detections, merged_detections, connector_info ):
    """
    Locate the row from raw_detections that corresponds to the given connector info, 
    along with the appropriate synapse detection from merged_detections.
    """
    assert not connector_info.incoming_nodes or not connector_info.outgoing_nodes, \
        "We assume the given skeleton file does not reference any nodes outside the skeleton. \n"\
        "Therefore, there should be either exactly 1 incoming or 1 outgoing node, not more. \n"\
        "We assume no autapses exist in the skeleton..."

    connected_node_id = None
    if connector_info.incoming_nodes:
        connected_node_id = connector_info.incoming_nodes[0]
    if connector_info.outgoing_nodes:
        connected_node_id = connector_info.outgoing_nodes[0]

    try:
        raw_row = raw_detections[connected_node_id]
    except KeyError:
        output_row = { "node_id" : connected_node_id, 
                       "synapse_id" : -1, 
                       "distance" : -1.0,
                       "connector_id" : connector_info.id }
    else:
        synapse_id = int(raw_row["synapse_id"])

        merged_row = merged_detections[synapse_id]
        min_distance = float(merged_row["distance"])

        output_row = dict( raw_row )
        output_row["distance"] = min_distance
        output_row["connector_id"] = connector_info.id

    return output_row

def _load_raw_detections( raw_detection_csv_path ):
    """
    Read raw detections into a dict, indexed by node_id.
    Returns: raw_detections dict and output_columns from the csv file
    """
    raw_detections = {}
    with open( raw_detection_csv_path, 'r' ) as raw_detection_file:
        raw_csv_reader = csv.DictReader(raw_detection_file, **CSV_FORMAT)
        for row in raw_csv_reader:
            node_id = int(row["node_id"])
            if node_id not in raw_detections:
                raw_detections[node_id] = row
            else:
                # More than one synapse, keep the "closest" one.
                old_distance = float(raw_detections[node_id]["distance"])
                new_distance = float(row["distance"])
                if new_distance < old_distance:
                    raw_detections[node_id] = row

    output_columns = raw_csv_reader.fieldnames + ["connector_id"]
    return raw_detections, output_columns

def _load_merged_detections( merged_detection_csv_path ):
    """
    Read merged detections into a dict, indexed by synapse_id.
    """
    merged_detections = {}
    with open( merged_detection_csv_path, 'r' ) as merged_detection_file:
        merged_csv_reader = csv.DictReader(merged_detection_file, **CSV_FORMAT)
        for row in merged_csv_reader:
            synapse_id = int(row["synapse_id"])
            merged_detections[synapse_id] = row
    return merged_detections

if __name__ == "__main__":
    import sys
    import argparse

    DEBUG_ARGS = False    
    if DEBUG_ARGS:
        skeleton_id = 18689
        #skeleton_id = 133465
        #skeleton_id = 163751
        #skeleton_id = 94835
        sys.argv.append( "/Users/bergs/Documents/workspace/skeleton_synapses/test_skeletons/skeleton_{}.json".format( skeleton_id ) )
        sys.argv.append( "/Users/bergs/Documents/workspace/skeleton_synapses/test_skeletons/raw_detections_{}.csv".format( skeleton_id ) )
        sys.argv.append( "/Users/bergs/Documents/workspace/skeleton_synapses/test_skeletons/merged_detections_{}.csv".format( skeleton_id ) )
        sys.argv.append( "/Users/bergs/Documents/workspace/skeleton_synapses/test_skeletons/connected_node_distances_{}.csv".format( skeleton_id ) )
    
    parser = argparse.ArgumentParser()
    parser.add_argument("skeleton_json_path")
    parser.add_argument("raw_detection_csv_path")
    parser.add_argument("merged_detection_csv_path")
    parser.add_argument("output_csv_path")
    
    parsed_args = parser.parse_args()
    
    num_connectors, nodes_without_detections = connected_node_distances( parsed_args.skeleton_json_path,
                                                                         parsed_args.raw_detection_csv_path,
                                                                         parsed_args.merged_detection_csv_path,
                                                                         parsed_args.output_csv_path )
    
    print "DONE."
    print "Processed {} connectors".format( num_connectors )
    if nodes_without_detections:
        print "Warning: Found no synapses for {} synapses:".format( len(nodes_without_detections) )
        print nodes_without_detections
