This is a distribution of ilastik with an extra library installed, which provides a script named 'locate_synapses'.

The script produces a CSV file with the following columns:

- "node_id"

  Which node was being processed when the synapse was detected.

- "node_x_px", "node_y_px", "node_z_px"

  The location of the node (in pixels)

- "synapse_id":

  A label integer for the synapse, which may
  appear in multiple neighboring slices

- "overlaps_node_segment",

  Either 'true' or 'false' indicating whether or not the
  synapse overlaps with the skeleton's neuron, i.e. the segment in
  the center of the fetched image data, where the skeleton node is located.

- "x_px", "y_px", "z_px"

  Location of the synapse, in pixels.

- "size_px"

  Size of the synapse as it appears in the current z-slice

- "tile_x_px", "tile_y_px", "tile_index"

  Location of the synapse within the output images produced by this script.

- "distance_to_node_px",

  Distance of the synapse to the skeleton node

- "detection_uncertainty"

  A measure of how certain the classifier is that this synapse is a true detection
  (Note: Don't take this measure too seriously.)


Additionally, the script writes a few image stacks (in HDF5 form) of the
image data that was used to produce the above CSV file:

- raw.h5

  The raw grayscale data extracted from around each skeleton node.
  (The central pixel corresponds to the location of the skeleton node.)

- predictions.h5

  Pixelwise probability maps computed over the raw data.
  (Channel 0: Membrane, Channel 1: Other, Channel 2: Synapse)

- synapse_cc.h5

  A label image containing any detected synapses in each slice.
  The label values correspond to the "synapse_id" CSV column described above.

- segmentation.h5

  A label image containing a segmentation of the neighborhood around each skeleton node.
  (Each 2D slice is an independent 2D segmentation -- they can't be stacked together in
  Z to form a 3D segmentation.)


USAGE:

$ ./bin/python bin/locate_synapses --help
usage: locate_synapses [-h] [--roi-radius-px ROI_RADIUS_PX]
                       skeleton_json autocontext_project multicut_project
                       volume_description output_dir [progress_port]

positional arguments:
  skeleton_json         A 'treenode and connector geometry' file exported from
                        CATMAID
  autocontext_project   ilastik autocontext project file (.ilp) with output
                        channels [membrane,other,synapse]. Must use axes
                        'xyt'.
  multicut_project      ilastik 2D multicut project file. Should expect the
                        probability channels from the autocontext project.
  volume_description    A file describing the CATMAID tile volume in the
                        ilastik 'TiledVolume' json format.
  output_dir            A directory to drop the output files.
  progress_port         An http server will be launched on the given port (if
                        nonzero), which can be queried to give information
                        about progress.

optional arguments:
  -h, --help            show this help message and exit
  --roi-radius-px ROI_RADIUS_PX
                        The radius (in pixels) around each skeleton node to
                        search for synapses


EXAMPLE:

./bin/python bin/locate_synapses \
  --roi-radius-px=150 \
  L1-CNS/skeletons/11524047/tree_geometry.json \
  L1-CNS/projects/full-vol-autocontext.ilp \
  L1-CNS/projects/multicut/L1-CNS-multicut.ilp \
  L1-CNS/L1-CNS-description.json
  L1-CNS/skeletons


ENVIRONMENT VARIABLES:

By default, ilastik uses all CPU cores on the machine, and (potentially) as much RAM as the machine can offer.
To limit the resources ilastik uses, define these environment variables before running the script:

# Example
export LAZYFLOW_THREADS=4
export LAZYFLOW_TOTAL_RAM_MB=4000

