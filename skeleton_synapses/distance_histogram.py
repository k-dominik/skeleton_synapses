import os
import csv
import numpy
import matplotlib.pyplot as plt

CSV_FORMAT = { 'delimiter' : '\t', 'lineterminator' : '\n' }
def distance_histogram( detection_csv_path ):
    distances = []
    with open( detection_csv_path, 'r' ) as detection_file:
        csv_reader = csv.DictReader(detection_file, **CSV_FORMAT)
        for row in csv_reader:
            distance = float( row["distance"] )
            if distance > 0.0:
                distances.append( distance )

    distances = numpy.asarray( distances, dtype=numpy.float32 )
    
    n, bins, patches = plt.hist(distances, bins=20, normed=1, facecolor='green', alpha=0.5)

    plt.xlabel('Membrane "Distance"')
    plt.ylabel('Detections')
    plt.title( os.path.split(detection_csv_path)[1] )
    plt.show()

if __name__ == "__main__":
    import sys
    import argparse

    DEBUG_ARGS = True
    if DEBUG_ARGS:
        sys.argv.append( "/Users/bergs/Documents/workspace/skeleton_synapses/test_skeletons/connected_node_distances_18689.csv" )
    
    parser = argparse.ArgumentParser()
    parser.add_argument("detection_csv_file")
    parsed_args = parser.parse_args()
    
    sys.exit( distance_histogram(parsed_args.detection_csv_file) )
    