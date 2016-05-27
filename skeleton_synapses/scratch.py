import numpy
import h5py
import vigra
import glob
import os.path

def synapse_images(two_dim = True):
    in_old = "/home/akreshuk/data/connector_archive_2g0y0b/distance_tests/*2d_pred.h5"
    in_new = "/data/connector_archive_2g0y0b/test_data/stage_2_output/*.h5"
    outdir = "/data/connector_archive_2g0y0b/compare_membranes/"

    in_old_list = glob.glob(in_old)
    in_old_list = sorted(in_old_list, key=str.lower)

    in_new_list = glob.glob(in_new)
    in_new_list = sorted(in_new_list, key=str.lower)

    for old_name, new_name in zip(in_old_list, in_new_list):
        print old_name
        dnew = vigra.readHDF5(new_name, "exported_data")
        dold = vigra.readHDF5(old_name, "exported_data")

        print dnew.shape, dnew.dtype, numpy.min(dnew), numpy.max(dnew)
        print dold.shape, dold.dtype, numpy.min(dold), numpy.max(dold)

        #convert the old ones to uint8, 0-255
        dold = dold*255
        dold = dold.astype(numpy.uint8)
        _, old_fname = os.path.split(old_name)
        parts = old_fname.split("_")
        dset_name = parts[0]
        if not os.path.exists(outdir+dset_name):
            os.makedirs(outdir+dset_name)
        if not os.path.exists(outdir+dset_name+"/old"):
            os.makedirs(outdir+dset_name+"/old")
        if not os.path.exists(outdir+dset_name+"/autocontext"):
            os.makedirs(outdir+dset_name+"/autocontext")
        if two_dim:
            vigra.impex.writeImage(dnew[:, :, 5, 1], outdir+dset_name+"/autocontext/membranes_z5.png" )
            vigra.impex.writeImage(dold[:, :, 1], outdir+dset_name+"/old/membranes_z5.png")
        else:
            for z in range(dnew.shape[2]):
                vigra.impex.writeImage(dnew[:, :, z, 2], outdir+dset_name+"/autocontext/%.02d"%z+".png")
                vigra.impex.writeImage(dold[:, :, z, 2], outdir+dset_name+"/old/%.02d"%z+".png")



if __name__=="__main__":
    synapse_images()