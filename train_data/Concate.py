import h5py as h5
import numpy as np

file = h5.File("stateactionfile.h5", 'r')
fileW = h5.File("alldata.h5", 'r+')
datasets_count = file.attrs['data_count']

#max_dataset_index = file.attrs['data_count']

print(datasets_count)

dataset = np.array(fileW.get('sa'))

for i in range(0, datasets_count):
    array = file.get('sa' + str(i - 1))
    nparr = np.array(array)
    dataset = np.concatenate((dataset, nparr), axis=0)

file.close()

fileW.create_dataset('sa', data=dataset)
fileW.close()