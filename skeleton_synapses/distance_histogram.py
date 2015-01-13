import os
import csv
import numpy
import matplotlib.pyplot as plt
from skeleton_utils import CSV_FORMAT

def distance_histogram( detection_csv_path, include_negative_distances=False ):
    distances = []
    missing = 0
    with open( detection_csv_path, 'r' ) as detection_file:
        csv_reader = csv.DictReader(detection_file, **CSV_FORMAT)
        for row in csv_reader:
            distance = float( row["distance"] )
            if distance >= 0.0 or include_negative_distances:
                distances.append( distance )
            elif not include_negative_distances:
                missing += 1

    distances = numpy.asarray( distances, dtype=numpy.float32 )
    
    n, bins, patches = plt.hist(distances, bins=20, normed=1, facecolor='green', alpha=0.5)

    plt.xlabel('Membrane "Distance"')
    plt.ylabel('Detections')
    if missing > 0:
        plt.title( os.path.split(detection_csv_path)[1] + " ({} shown, {} missing)".format( len(distances), missing ) )
    else:
        plt.title( os.path.split(detection_csv_path)[1] + " ({} shown)".format( len(distances) ) )

    plt.show()

if __name__ == "__main__":
    import sys
    import argparse

    DEBUG_ARGS = False
    if DEBUG_ARGS:
        #sys.argv.append( "/Users/bergs/Documents/workspace/skeleton_synapses/test_skeletons/connected_node_distances_18689.csv" )
        #sys.argv.append( "/Users/bergs/Documents/workspace/skeleton_synapses/test_skeletons/connected_node_distances_133465.csv" )
        #sys.argv.append( "/Users/bergs/Documents/workspace/skeleton_synapses/test_skeletons/connected_node_distances_163751.csv" )
        sys.argv.append( "/Users/bergs/Documents/workspace/skeleton_synapses/test_skeletons/connected_node_distances_94835.csv" )
    
    parser = argparse.ArgumentParser()
    parser.add_argument("detection_csv_file")
    parsed_args = parser.parse_args()
    
    sys.exit( distance_histogram(parsed_args.detection_csv_file) )

