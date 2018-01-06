import h5py as h5
import numpy as np

class TrainInput(object):
    def __init__(self, filename = 'train_data/alldata.h5', DISPLAY_STEP = 100, number_of_sensors = 29):
        self.file = h5.File(filename, 'r')
        self.DISPLAY_STEP = DISPLAY_STEP

        self.data = np.array(self.file.get('sa'))
        self.data = np.delete(self.data, 2, 1)
        np.random.shuffle(self.data)
        self.max_train_index = int(self.data.shape[0]/2) - 1

        self.train_data_index = 0
        self.test_data_index = int(self.data.shape[0]/2)

        self.number_of_sensors = number_of_sensors

    def get_next_batch(self, batch = 100):
        state = self.data[self.train_data_index: self.train_data_index + batch, 3: 32]
        action = self.data[self.train_data_index: self.train_data_index + batch, 0: 3]

        self.train_data_index += batch

        if self.train_data_index > self.max_train_index:
            self.train_data_index = 0

        return state, action

    def get_next_test_data(self, batch = 100):
        state = self.data[self.test_data_index: self.test_data_index + batch - 1, 3: 32]
        action = self.data[self.test_data_index: self.test_data_index + batch - 1, 0: 3]

        self.test_data_index += batch

        if self.test_data_index > self.data.shape[0] - 1:
            self.test_data_index = int(self.data.shape[0]/2)

        return state, action

    def get_train_data_count(self):
        return self.max_train_index + 1

    def close(self):
        self.file.close()

