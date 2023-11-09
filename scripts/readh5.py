import h5py
import numpy as np

filename = "solution_10.h5"
outputfilename = "solution_10.txt"

with h5py.File(filename, "r") as f:
    # Print all root level object names (aka keys) 
    # these can be group or dataset names 
    print("Keys: %s" % f.keys())

    h = 10.0 / 1024

    center_rho = f['center_rho'][()]
    center_z = f['center_z'][()]
    center_value = f['center_value'][()]

    center_rho_index = np.rint((center_rho - h/2) * 1024 / 10.0).astype(int)
    center_z_index = np.rint((center_z - h/2)* 1024 / 10.0).astype(int)

    data = np.column_stack((center_rho_index, center_z_index, center_value))


    np.savetxt(outputfilename, data, delimiter=',')
