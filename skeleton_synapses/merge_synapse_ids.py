import csv
import collections
import numpy

CSV_FORMAT = { 'delimiter' : '\t', 'lineterminator' : '\n' }

def merge_synapse_ids(input_path, output_path):
    """
    Read the given csv file and merge all rows with identical synapse ids into a single row for each id.

    The input csv file must contain a header row, and the following fields must be present (in any order):
    synapse_id, x_px, y_px, z_px, distance

    The output row's coordinate columns (x_px, y_px, z_px) will be an average of the coordinates from the merged rows.
    The output row's "distance" column will be the minimum entry from the corresponding input rows.
    A new column "slice_count" will be appended to indicate how many rows were merged to create each output row.
    
    All other fields in the output row will be copied from one of the corresponding input rows.
    """
    with open(input_path, "r") as f:
        # Read all rows and group them by synapse id
        all_synapses = collections.OrderedDict()
        csv_reader = csv.DictReader(f, **CSV_FORMAT)
        for row in csv_reader:
            syn_id = row["synpase_id"]
            try:
                all_synapses[syn_id].append( row )
            except KeyError:
                all_synapses[syn_id] = [row]
            
    with open(output_path, "w") as f2:
        output_columns = csv_reader.fieldnames + ["slice_count"]
        csv_writer = csv.DictWriter(f2, output_columns, **CSV_FORMAT)
        csv_writer.writeheader()
        for syn_id, rows in all_synapses.iteritems():
            if len(rows) == 1:
                # Fast path
                final_row = rows[0]
            else:
                # Create list of coord tuples for this synapse
                syn_coords = map( lambda row: (row["x_px"], row["y_px"], row["z_px"]), rows )
                syn_coord_array = numpy.asarray(syn_coords).astype(int)
                avg_coord = (numpy.average(syn_coord_array, 0) + 0.5).astype(int)

                # Find a row with the same z-index, and use its fields
                avg_z = avg_coord[2]
                final_row = filter( lambda row: int(row["z_px"]) == avg_z, rows )[0]
                
                # Replace coords with avg
                final_row["x_px"], final_row["y_px"], final_row["z_px"] = avg_coord

                # Replace distance with min distance                
                distances = map( lambda row: row["distance"], rows )
                distances = map( float, distances )
                final_row["distance"] = min( distances )
            
            final_row["slice_count"] = len(rows)
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
