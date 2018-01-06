import h5py as h5

file = h5.File('stateactionfile.h5', 'w')
file.attrs['data_count'] = 0
file.close()