import h5py as h5
import os

filename = 'stateactionfile.h5'
try:
    os.remove(filename)
except OSError:
    pass
file = h5.File(filename, 'w')
file.attrs['data_count'] = 0
file.close()