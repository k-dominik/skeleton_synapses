Usage
-----
You must use the python interpreter and environment included with the binary. That is, use `ilastik_python.sh`:

    ilastik_python.sh skeleton_synapses/locate_synapses.py skeleton_swc project3d project2d volume_description output_file

Arguments:

- `skeleton_swc`: The skeleton to process, as an swc file
- `project3d`: The trained project file for 3D predictions (extension: .ilp)
- `project2d`: The trained project file for 2D predictions
- `volume_description`: A JSON file describing the tiled data source (see example below).  Must have .json extension.
- `output_file`: Where to store the detected synapses (csv).  The columns are: `id` `x` `y` `z` `distance`

Example Volume Description JSON file
-------------------------------------

The description must be in this special JSON format.  Note that the fields are in zyx order, not xyz.

    {
        "_schema_name" : "tiled-volume-description",
        "_schema_version" : 1.0,
    
        "## NOTES" : "Volume dimensions: x: 22775 px y: 18326 px z: 462 (number of sections)",
        "## NOTES" : "Resolution: 4.0 x 4.0 x 45.0",
        "## NOTES" : "Tile size: 256 x 256 px",
    
        "name" : "My Tiled Data",
        "format" : "jpg",
        "dtype" : "uint8",
        "bounds_zyx" : [462, 18326, 22775],    
        "resolution_zyx" : [45.0, 4.0, 4.0],
    
        "tile_shape_2d_yx" : [256, 256],
        "tile_url_format" : "http://neurocean.int.janelia.org:6081/ssd-3-tiles/abd1.5/{z_index}/{y_index}_{x_index}_0.jpg",
        
        "## NOTES" : "Don't touch the output_axes field.  The locate_synapses script requires xyz.",
        "output_axes" : "xyz",
        
        "## NOTES" : "Whether or not to keep a cache of the raw data in memory.  Useful for interactive navigation.",
        "cache_tiles" : false,
    
        "## NOTES" : "Many slices are missing from this dataset, so this mapping determines ",
        "##      " : "  which slices should be extended to replace the missing ones.",
        "##      " : "For example, slice 348 is extended to also show up in slices 349, 350, and 351",
        "extend_slices" : [ [98, [99]],
                            [348, [349, 350, 351]]
                          ]
    }
