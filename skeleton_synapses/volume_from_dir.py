import glob
import numpy
import h5py
import vigra

def volume_from_dir(dirpattern, output_filepath, offset=0, nfiles=None):
    filelist = glob.glob(dirpattern)
    filelist = sorted(filelist, key=str.lower)
    begin = offset
    if nfiles is not None and offset+nfiles<len(filelist):
        end=offset+nfiles
    else:
        end = len(filelist)
    filelist = filelist[begin:end]
    nx, ny = vigra.readImage(filelist[0]).squeeze().shape
    dt = vigra.readImage(filelist[0]).dtype
    nz = len(filelist)
    volume = numpy.zeros((nx, ny, nz, 1), dtype=dt)
    
    for i in range(len(filelist)):
        volume[:, :, i, 0] = vigra.readImage(filelist[i]).squeeze()[:]
        
    outfile = h5py.File(output_filepath, "w")
    outfile.create_dataset("data", data=volume)
    outfile.close()
    return volume

if __name__=="__main__":
    dirpattern = "/home/akreshuk/data/connector_archive_2g0y0b/distance_tests/16725035/*.tiff"
    output_filepath = "/home/akreshuk/data/connector_archive_2g0y0b/distance_tests/16725035_raw.h5"
    volume_from_dir(dirpattern, output_filepath)