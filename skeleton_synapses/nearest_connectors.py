import csv
import numpy
import scipy.spatial.distance

from lazyflow.roi import getIntersectingBlocks
from lazyflow.utility.io import TiledVolume
from skeleton_utils import parse_connectors, ConnectorInfo

class ConnectorStore(object):
    """
    Stores a list of connectors in buckets for faster location-based lookup.
    """
    SEARCH_RADIUS = 500
    
    def __init__(self, connectors, blockshape_xyz):
        self._blockshape = blockshape_xyz
        self._blocks = self._generate_connector_buckets( connectors, blockshape_xyz )
    
    @classmethod
    def _generate_connector_buckets(cls, connectors, blockshape_xyz ):
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

    def find_nearest_connector(self, detection_coord ):
        """
        Search the given buckets of connectors for the one that is nearest to the given coordinates.
        Buckets farther than SEARCH_RADIUS are not searched, in which case a default ConnectorInfo object is returned.
        
        Returns: nearest_connector, distance to the nearest connector
        """
        # Find nearby blocks
        detection_coord_int = detection_coord.astype(int)
        search_roi = ( detection_coord_int - self.SEARCH_RADIUS,
                       detection_coord_int + self.SEARCH_RADIUS )
        nearby_block_starts = getIntersectingBlocks(self._blockshape, search_roi)
        nearby_block_starts = map(tuple, nearby_block_starts)
    
        # Accumulate connectors found in nearby blocks
        nearby_connectors = []
        for block_start in nearby_block_starts:
            if block_start in self._blocks:
                nearby_connectors += self._blocks[block_start]
    
        # Closure.  Distance from current point to given connector.
        def distance( conn ):
            return scipy.spatial.distance.euclidean( (conn.x_nm, conn.y_nm, conn.z_nm), detection_coord )
    
        # Find closest connector.
        if nearby_connectors:
            nearest_connector = min(nearby_connectors, key=distance)
            min_distance = distance( nearest_connector )
        else:
            # No connectors nearby.  Emit default values.
            nearest_connector = ConnectorInfo(-1, -1, -1, -1, [], [])
            min_distance = 9999999.0
        
        return nearest_connector, min_distance
    

CSV_FORMAT = { 'delimiter' : '\t', 'lineterminator' : '\n' }
def output_nearest_connectors( synapse_detections_csv, connectors, resolution_xyz, output_csv ):
    """
    Read the synapse detections csv file at the given path and write a copy of it 
    with extra columns appended for the distance to the nearest connector annotation.

    The extra output columns are:   nearest_connector_id, 
                                    nearest_connector_distance_nm, 
                                    nearest_connector_x_nm, 
                                    nearest_connector_y_nm, 
                                    nearest_connector_z_nm
                                    
    The nearest connector is only searched for within a maximum radius of SEARCH_RADIUS.
    If no nearby connector is found for a synapse detection, a negative id is output, 
    with a very large distance.
    
    The extra columns are nearest_connector_id, nearest_connector_distance_nm
    
    Args:
        synapase_detections_csv: A path to the output file from locate_synapses()
        connectors: A list of ConnectorInfo objects
        resolution_xyz: A tuple of the resolution in x,y,z order
        output_csv: The path to write the output file to    
    """
    # Avoid searching the whole list every time:
    # store the connectors in buckets by block.
    connector_store = ConnectorStore( connectors, blockshape_xyz=(1000,1000,1000) )
    
    with open(synapse_detections_csv, 'r') as detections_file,\
         open(output_csv, 'w') as output_file:
        
        csv_reader = csv.DictReader(detections_file, **CSV_FORMAT)
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

            detection_coord = numpy.array( (x_nm, y_nm, z_nm) )
            nearest_connector, distance = connector_store.find_nearest_connector( detection_coord )

            # Write output row.
            row["nearest_connector_id"] = nearest_connector.id
            row["nearest_connector_distance_nm"] = distance
            row["nearest_connector_x_nm"] = nearest_connector.x_nm
            row["nearest_connector_y_nm"] = nearest_connector.y_nm
            row["nearest_connector_z_nm"] = nearest_connector.z_nm
            csv_writer.writerow( row )


if __name__ == "__main__":
    USE_DEBUG_FILES = False
    if USE_DEBUG_FILES:
        print "USING DEBUG ARGUMENTS"
        import sys        
        sys.argv.append( '/magnetic/workspace/skeleton_synapses/example/example_volume_description_2.json' )
        sys.argv.append( '/magnetic/workspace/skeleton_synapses/example/skeleton_18689.json' )
        sys.argv.append( '/magnetic/workspace/skeleton_synapses/merged_skeleton_18689_synapse_detections.csv' )        
        sys.argv.append( '/magnetic/workspace/skeleton_synapses/skeleton_18689_detections_with_distances_2.csv' )        

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('volume_description_json')
    parser.add_argument('skeleton_json')
    parser.add_argument('detections_csv')
    parser.add_argument('output_csv')
    parsed_args = parser.parse_args()

    volume_description = TiledVolume.readDescription(parsed_args.volume_description_json)
    z_res, y_res, x_res = volume_description.resolution_zyx

    connectors = parse_connectors( parsed_args.skeleton_json )
    output_nearest_connectors( parsed_args.detections_csv, connectors, ( x_res, y_res, z_res ), parsed_args.output_csv )
 
