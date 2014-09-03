import csv
import numpy
import scipy.spatial

from lazyflow.roi import getIntersectingBlocks
from lazyflow.utility.io import TiledVolume
from swc_rois import parse_connectors, ConnectorInfo

CSV_FORMAT = { 'delimiter' : '\t', 'lineterminator' : '\n' }

def store_connectors_blockwise( connectors, blockshape_xyz ):
    """
    Store the list of ConnectorInfos into buckets (a dictionary of lists), grouped by block location.
    The dict keys are simply the start coordinate of each block (as a tuple).
    The block boundaries are determined by blockshape_zyx.
    """
    blocks = {}
    for conn in connectors:
        coord = numpy.array( (conn.x_nm, conn.y_nm, conn.z_nm) ).astype(int)
        block_start = getIntersectingBlocks( blockshape_xyz, (coord, coord+1) )[0]
        block_start = tuple(block_start)
        try:
            blocks[block_start].append( conn )
        except KeyError:
            blocks[block_start] = [conn]
    return blocks


SEARCH_RADIUS = 500
def output_nearest_connectors( synapse_candidates_csv, connectors, resolution_xyz, output_csv ):
    """
    Read the synapse candidates csv file at the given path and write a copy of it 
    with extra columns appended for the distance to the nearest connector annotation.

    The extra output columns are:   nearest_connector_id, 
                                    nearest_connector_distance_nm, 
                                    nearest_connector_x_nm, 
                                    nearest_connector_y_nm, 
                                    nearest_connector_z_nm
                                    
    The nearest connector is only searched for within a maximum radius of SEARCH_RADIUS.
    If no nearby connector is found for a synapse candidate, a negative id is output, 
    with a very large distance.
    
    The extra columns are nearest_connector_id, nearest_connector_distance_nm
    
    Args:
        synapase_candidates_csv: A path to the output file from locate_synapses()
        connectors: A list of ConnectorInfo objects
        resolution_xyz: A tuple of the resolution in x,y,z order
        output_csv: The path to write the output file to    
    """
    # Avoid searching the whole list every time:
    # store the connectors in buckets by block.
    blockshape = (1000,1000,1000)
    blockwise_connectors = store_connectors_blockwise( connectors, blockshape )
    
    with open(synapse_candidates_csv, 'r') as candidates_file,\
         open(output_csv, 'w') as output_file:
        
        csv_reader = csv.DictReader(candidates_file, **CSV_FORMAT)
        output_fields = csv_reader.fieldnames + [ "nearest_connector_id", 
                                                  "nearest_connector_distance_nm", 
                                                  "nearest_connector_x_nm", 
                                                  "nearest_connector_y_nm", 
                                                  "nearest_connector_z_nm" ]

        csv_writer = csv.DictWriter(output_file, output_fields, **CSV_FORMAT)
        csv_writer.writeheader()
        
        for row in csv_reader:
            # Convert from pixels to nanometers
            x_nm = int(row["x_px"]) * resolution_xyz[0]
            y_nm = int(row["y_px"]) * resolution_xyz[1]
            z_nm = int(row["z_px"]) * resolution_xyz[2]

            candidate_coord = numpy.array( (x_nm, y_nm, z_nm) ).astype(int)

            # Find nearby blocks
            nearby_block_starts = getIntersectingBlocks( blockshape, ( candidate_coord - SEARCH_RADIUS, candidate_coord + SEARCH_RADIUS ))
            nearby_block_starts = map(tuple, nearby_block_starts)

            # Accumulate connectors found in nearby blocks
            nearby_connectors = []
            for block_start in nearby_block_starts:
                if block_start in blockwise_connectors:
                    nearby_connectors += blockwise_connectors[block_start]

            # Closure.  Distance from current point to given connector.
            def distance( conn ):
                return scipy.spatial.distance.euclidean( (conn.x_nm, conn.y_nm, conn.z_nm), (x_nm, y_nm, z_nm) )

            # Find closest connector.
            if nearby_connectors:
                nearest_connector = min(nearby_connectors, key=distance)
                min_distance = distance( nearest_connector )
            else:
                # No connectors nearby.  Emit default values.
                nearest_connector = ConnectorInfo(-1, -1, -1, -1, [], [])
                min_distance = 9999999.0

            # Write output row.
            row["nearest_connector_id"] = nearest_connector.id
            row["nearest_connector_distance_nm"] = min_distance
            row["nearest_connector_x_nm"] = nearest_connector.x_nm
            row["nearest_connector_y_nm"] = nearest_connector.y_nm
            row["nearest_connector_z_nm"] = nearest_connector.z_nm
            csv_writer.writerow( row )
                

if __name__ == "__main__":
    USE_DEBUG_FILES = False
    if USE_DEBUG_FILES:
        print "USING DEBUG ARGUMENTS"
        import sys        
        sys.argv.append( '/Users/bergs/Documents/workspace/skeleton_synapses/example/example_volume_description_2.json' )
        sys.argv.append( '/Users/bergs/Documents/workspace/skeleton_synapses/example/skeleton_18689.json' )
        sys.argv.append( '/Users/bergs/Documents/workspace/skeleton_synapses/merged_synapses.csv' )        
        sys.argv.append( '/Users/bergs/Documents/workspace/skeleton_synapses/filtered_with_distances.csv' )        

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('volume_description_json')
    parser.add_argument('skeleton_json')
    parser.add_argument('candidates_csv')
    parser.add_argument('output_csv')
    parsed_args = parser.parse_args()

    volume_description = TiledVolume.readDescription(parsed_args.volume_description_json)
    z_res, y_res, x_res = volume_description.resolution_zyx

    connectors = parse_connectors( parsed_args.skeleton_json )
    output_nearest_connectors( parsed_args.candidates_csv, connectors, ( x_res, y_res, z_res ), parsed_args.output_csv )
 
