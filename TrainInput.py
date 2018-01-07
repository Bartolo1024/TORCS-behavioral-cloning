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

    def get_chain_train_data(self, batch_size, number_of_sensors, number_of_efectors):
        # return ordered chain of states and actions

        x = self.data[:, 3: 32]
        y = self.data[:, 0: 3]

        state_train = np.reshape(x, (batch_size, number_of_sensors, -1))
        action_train = np.reshape(y, (batch_size, number_of_efectors, -1))

        return state_train, action_train

    # def get_chain_test_data(self, batch_size, number_of_sensors, number_of_efectors):
    #     # return chain of states
    #
    #     x = self.data[self.test_data_index: self.data.shape[0], 3: 32]
    #     y = self.data[self.test_data_index: self.data.shape[0], 0: 3]
    #
    #     state_test = np.reshape(x, (batch_size, number_of_sensors, -1))
    #     action_test = np.reshape(y, (batch_size, number_of_efectors, -1))
    #
    #     return state_test, action_test

    def max_epochs_index(self, epoch_size):
        return int(self.data.shape[0]/epoch_size)

    def close(self):
        self.file.close()

