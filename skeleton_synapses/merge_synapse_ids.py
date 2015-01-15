import csv
import collections
import numpy
from skeleton_utils import CSV_FORMAT

def merge_synapse_ids(input_path, output_path):
    """
    Read the given csv file and merge all rows with identical synapse ids into a single row for each id.

    The input csv file must contain a header row, and the following fields must be present (in any order):
    synapse_id, x_px, y_px, z_px, size_px, distance, detection_uncertainty

    Rows with the same synapse_id are merged.  Each column is handled as follows:
    x_px, y_px, z_px - weighted average across rows (weighted by size_px)
    size_px - sum of all rows
    distance - the minimum of all rows is chosen
    detection_uncertainty - weighted average across rows (weighted by size_px)
    
    A new column "node_count" will be appended to indicate how many rows were merged to create each output row.
    
    All other fields in the output row will be copied from one of the corresponding input rows.
    """
    with open(input_path, "r") as f:
        # Read all rows and group them by synapse id
        all_synapses = collections.OrderedDict()
        csv_reader = csv.DictReader(f, **CSV_FORMAT)
        for row in csv_reader:
            syn_id = row["synapse_id"]
            try:
                all_synapses[syn_id].append( row )
            except KeyError:
                all_synapses[syn_id] = [row]
            
    with open(output_path, "w") as f2:
        output_columns = csv_reader.fieldnames + ["node_count"]
        csv_writer = csv.DictWriter(f2, output_columns, **CSV_FORMAT)
        csv_writer.writeheader()
        for syn_id, rows in all_synapses.iteritems():
            if len(rows) == 1:
                # Fast path
                final_row = rows[0]
            else:
                min_distance = distance_hessian = distance_raw_probs = None
                if "distance" in output_columns:
                    # Find min distance
                    distances = map( lambda row: row["distance"], rows )
                    min_distance = numpy.asarray(distances, dtype=numpy.float32).min()
                
                connector_distances = map( lambda row: row["connector_distance"], rows)
                min_conn_distance = numpy.asarray(connector_distances, dtype=numpy.float32).min()

                if "distance_hessian" in output_columns:
                    # Find min hessian distance
                    distances_hessian = map( lambda row: row["distance_hessian"], rows )
                    min_distance_hessian = numpy.asarray(distances_hessian, dtype=numpy.float32).min()

                if "distance_raw_probs" in output_columns:
                    # Find min raw_probs distance
                    distances_raw_probs = map( lambda row: row["distance_raw_probs"], rows )
                    min_distance_raw_probs = numpy.asarray(distances_raw_probs, dtype=numpy.float32).min()

                # Sum sizes
                sizes = map( lambda row: row["size_px"], rows )
                sizes = numpy.asarray( sizes, dtype=numpy.uint32 )
                total_size = sizes.sum()
                size_weights = sizes/float(total_size)

                # Uncertainty: take weighted average
                uncertainties = map( lambda row: row["detection_uncertainty"], rows )
                uncertainties = numpy.asarray(uncertainties, dtype=numpy.float32)
                avg_uncertainty = numpy.average( uncertainties, weights=size_weights )
            
                # Create list of coord tuples for this synapse
                syn_coords = map( lambda row: (row["x_px"], row["y_px"], row["z_px"]), rows )
                syn_coord_array = numpy.asarray(syn_coords).astype(int)
                
                # Coord: take weighted average
                avg_coord = (numpy.average(syn_coord_array, axis=0, weights=size_weights) + 0.5).astype(int)

                # At least one row has the same z-coord as the average coord
                # Find it and use it as the final_row.
                #final_row = filter( lambda row: float(row["distance"]) == min_distance, rows )[0]
                final_row = rows[0]
                if "distance" in output_columns:
                    final_row = filter( lambda row: float(row["distance"]) == min_distance, rows )[0]
                if "distance_hessian" in output_columns:
                    final_row = filter( lambda row: numpy.float32(row["distance_hessian"]) == numpy.float32(min_distance_hessian), rows )[0]
                
                # Replace fields in the final row
                final_row["x_px"], final_row["y_px"], final_row["z_px"] = avg_coord
                final_row["connector_distance"] = min_conn_distance
                final_row["size_px"] = total_size
                final_row["detection_uncertainty"] = avg_uncertainty

                if "distance" in output_columns:
                    final_row["distance"] = min_distance
                if "distance_hessian" in output_columns:
                    final_row["distance_hessian"] = min_distance_hessian
                if "distance_raw_probs" in output_columns:
                    final_row["distance_raw_probs"] = min_distance_raw_probs

            final_row["node_count"] = len(rows)

            # Write.
            csv_writer.writerow( final_row )            

if __name__ == "__main__":
    import argparse

    USE_DEBUG_FILES = False
    if USE_DEBUG_FILES:
        print "USING DEBUG ARGUMENTS"
        import sys
        #sys.argv.append( '/home/akreshuk/data/abd1.5_output_synapse_1.csv' )
        #sys.argv.append( '/home/akreshuk/data/abd1.5_output_synapse_1_pp.csv' )
        
        sys.argv.append( '/Users/bergs/Documents/workspace/skeleton_synapses/synapses.csv' )
        sys.argv.append( '/Users/bergs/Documents/workspace/skeleton_synapses/merged_synapses.csv' )        

    parser = argparse.ArgumentParser() 
    parser.add_argument('input_csv')
    parser.add_argument('output_csv')
    parsed_args = parser.parse_args()
    
    merge_synapse_ids( parsed_args.input_csv, parsed_args.output_csv )
