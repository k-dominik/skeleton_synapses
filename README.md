Usage
-----
You must use the python interpreter and environment included with the binary. For example, `ilastik-1.2.0-Linux/bin/python`

    ilastik-1.2.0-Linux/bin/python skeleton_synapses/locate_synapses.py 12345.swc myproject.ilp L1-CNS-description.json 12345-detections.csv

Arguments:

- `skeleton_file`: The skeleton to process, as an `swc` or `json` file, exported from CATMAID.
- `autocontesxt_project`: The trained project file for 3D predictions (extension: `.ilp`)
- `volume_description`: A JSON file describing the tiled data source (see example below).  (Must have `.json` extension.)
- `output_file`: Where to store the detected synapses (csv).
- `progress_port`: (Optional.) Progress is reported via an http server, running on the given port. (default: 8000)

Output columns:

- `synapse_id`: A unique ID for the synapse within the file. The same synapse might appear in different slices.
- `x_px`, `y_px`, `z_px`: Coordinate of the synapse in pixels.
- `size_px`: Size of the synapse in pixels.
- `detection_uncertainty`: How (un)confident the synapse classifier is that this deteciton was real. (0.0: very confident, 1.0: not confident) 
- `node_id`: The ID of the skeleton node this synapse was found near.
- `node_x_px`, `node_y_px`, `node_z_px`: The location of the skeleton node, in pixels
- `distance`: Distance between the synapse and the node it was found near. (Currently, A simple euclidean distance.)


Example Volume Description JSON file
-------------------------------------

The description must be in this special JSON format.  Note that the fields are written in `zyx` order, not `xyz`!

    {
        "_schema_name" : "tiled-volume-description",
        "_schema_version" : 1.0,
    
        "## NOTES" : "Volume dimensions: x: 28128 px y: 31840 px z: 4841 (number of sections)",
        "## NOTES" : "Resolution: 3.8 x 3.8 x 50",
        "## NOTES" : "Tile size: 512 x 512 px",
    
        "name" : "L1 CNS (Janelia Local)",
        "format" : "jpg",
        "dtype" : "uint8",
        "bounds_zyx" : [4841, 31840, 28128],
        "##bounds_zyx" : [500, 31840, 28128],
    
        "resolution_zyx" : [50.0, 3.8, 3.8],
        "tile_shape_2d_yx" : [512, 512],
    
        "## JANELIA LOCAL  tile_url_format" : "https://neurocean.janelia.org/ssd-tiles-no-cache/0111-8/{z_index}/0/{y_index}_{x_index}.jpg",
        "## JANELIA PUBLIC tile_url_format" : "https://neurocean.janelia.org/ssd-tiles/0111-8/{z_index}/0/{y_index}_{x_index}.jpg",
        "tile_url_format" : "https://neurocean.janelia.org/ssd-tiles-no-cache/0111-8/{z_index}/0/{y_index}_{x_index}.jpg",
        
        "## NOTES" : "Don't touch the output_axes field.  The locate_synapses script requires xyz.",
        "output_axes" : "xyz",
        
        "## NOTES" : "Whether or not to keep a cache of the raw data in memory.  Useful for interactive navigation.",
        "cache_tiles" : true,
    
        "## NOTES" : "Many slices are missing from this dataset, so this mapping determines ",
        "##      " : "  which slices should be extended to replace the missing ones.",
        "##      " : "For example, slice 1103 is extended to also show up in slices 1104-1107",
        "extend_slices" : [ [1103, [1104, 1105, 1106, 1107]],
                            [1112, [1108, 1109, 1110, 1111]],
                            [1891, [1892, 1893, 1894, 1895]],
                            [1900, [1896, 1897, 1898, 1899]],
                            [1920, [1921]],
                            [1923, [1924]],
                            [1983, [1984, 1985, 1986]],
                            [1990, [1987, 1988, 1989]],
                            [2759, [2760, 2761, 2762, 2763, 2764]],
                            [2770, [2765, 2766, 2767, 2768, 2769]],
                            [2811, [2812, 2813]],
                            [3073, [3074, 3075, 3076, 3077, 3078]],
                            [3084, [3079, 3080, 3081, 3082, 3083]],
                            [3107, [3108]],
                            [3120, [3121, 3122]],
                            [3125, [3123, 3124]],
                            [3162, [3163, 3164]],
                            [3164, [3165, 3166]],
                            [3179, [3180, 3181, 3182]],
                            [3186, [3183, 3184, 3185]],
                            [3442, [3443, 3444, 3445, 3446, 3447]],
                            [3453, [3448, 3449, 3450, 3451, 3452]],
                            [3481, [3482]],
                            [3522, [3523, 3524, 3525, 3526, 3527]],
                            [3533, [3528, 3529, 3530, 3531, 3532]],
                            [4172, [4173]],
                            [4176, [4177]],
                            [4727, [4728]],
                            [4763, [4764]]
                           ]
    }
