import h5py as h5
import numpy as np

class TrainInput(object):
    def __init__(self, filename = 'train_data/alldata.h5',
                 DISPLAY_STEP = 100,
                 number_of_sensors = 29,
                 shuffle_data = True,
                 single_race_data_size = 10000,
                 acceleration_brake_merged=False):

        self.file = h5.File(filename, 'r')
        self.DISPLAY_STEP = DISPLAY_STEP

        self.data = np.array(self.file.get('sa'))
        #delete gear
        self.data = np.delete(self.data, 2, 1)
        self.max_efector_index = 3

        #merge acceleration and brake
        if acceleration_brake_merged:
            self.data[:, 1] = self.data[:, 1] - self.data[:, 2]
            #self.data = np.delete(self.data, 2, 1)
            self.max_efector_index = 2

        self.shuffle_data = shuffle_data
        if self.shuffle_data == True:
            np.random.shuffle(self.data)

        self.max_train_index = int(self.data.shape[0]/2) - 1

        self.train_data_index = 0
        self.test_data_index = int(self.data.shape[0]/2)

        self.number_of_sensors = number_of_sensors
        self.single_race_data_size = single_race_data_size

        print(self.data.shape)

    def get_next_batch(self, batch = 100):
        state = self.data[self.train_data_index: self.train_data_index + batch, self.max_efector_index: 32]
        action = self.data[self.train_data_index: self.train_data_index + batch, 0: self.max_efector_index]

        self.train_data_index += batch

        if self.train_data_index > self.max_train_index:
            self.train_data_index = 0
            if self.shuffle_data == True:
                np.random.shuffle(self.data)

        return state, action

    def get_next_test_data(self, batch = 100):
        state = self.data[self.test_data_index: self.test_data_index + batch - 1, self.max_efector_index: 32]
        action = self.data[self.test_data_index: self.test_data_index + batch - 1, 0: self.max_efector_index]

        self.test_data_index += batch

        if self.test_data_index > self.data.shape[0] - 1:
            self.test_data_index = int(self.data.shape[0]/2)

        return state, action

    def get_train_data_count(self):
        return self.max_train_index + 1

    def max_epochs_index(self):
        return int(self.data.shape[0] / self.single_race_data_size)

    def get_batches_count(self, batch_size, backpropagation_lenght):
        return int(self.single_race_data_size / batch_size)

    def get_chain_train_data(self, batch_size, number_of_sensors, number_of_efectors, epoch):
        # return ordered chain of states and actions

        while epoch > self.max_epochs_index():
            epoch - self.max_epochs_index()

        x = self.data[epoch * self.single_race_data_size: epoch * self.single_race_data_size + self.single_race_data_size, self.max_efector_index: 32]
        y = self.data[epoch * self.single_race_data_size: epoch * self.single_race_data_size + self.single_race_data_size, 0: self.max_efector_index]

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



    def close(self):
        self.file.close()

