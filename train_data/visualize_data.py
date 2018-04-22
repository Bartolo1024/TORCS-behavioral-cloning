import matplotlib.pyplot as plt
import h5py as h5
import numpy as np
import pandas as pd

track_list = ['track_%s' % ind for ind in range(0, 19)]
dataframes = []

with h5.File('train_data/stateactionfile.h5', 'r') as file:
    datasets_count = file.attrs['data_count']
    dataset = np.array(file.get('sa0'))
    columns = ['steer', 'acceleration', 'gear', 'brake', 'speed_x', 'speed_y', 'speed_z', 'rpm']
    columns.extend(['wheel_spin_%s' % ind for ind in range(0, 4)])
    columns.extend(track_list)
    columns.extend(['angle', 'track_position'])
    for i in range(1, datasets_count):
        array = file.get('sa' + str(i - 1))
        nparr = np.array(array)
        dataframes.append(pd.DataFrame(data=nparr, columns=columns))
    file.close()

plt.plot(np.arange(0, 10000), dataframes[1]['acceleration'])
plt.xlabel('time [steps]')
plt.ylabel('acceleration signal')
plt.show()

plt.plot(np.arange(0, 10000), dataframes[1]['steer'])
plt.xlabel('time [steps]')
plt.ylabel('steer signal')
plt.show()

plt.plot(np.arange(0, 10000), dataframes[1]['brake'])
plt.xlabel('time [steps]')
plt.ylabel('brake signal')
plt.show()

import sensors_animation as ani
animator = ani.SensorsAnimation(dataframes[1])
animator.run_animation()


