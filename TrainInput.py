import h5py as h5
import numpy as np

class TrainInput(object):
    def __init__(self, filename = 'stateactionfile.h5', DISPLAY_STEP = 100):
        self.file = h5.File(filename, 'r')
        self.DISPLAY_STEP = DISPLAY_STEP
        self.dataset_index = 0

        data_max_index = self.file.attrs['data_count'] - 1
        if data_max_index % 2 == 0:
            self.max_train_dataset_index = int(data_max_index / 2) - 1
        else:
            self.max_train_dataset_index = int(data_max_index / 2)

        self.test_dataset_index = self.max_train_dataset_index + 1
        self.max_test_dataset_index = data_max_index

        self.load_next_dataset()
        self.load_next_test_dataset()
        self.data_counter = 0

    def get_next_batch(self, batch = 100):
        state = self.train_data_numpy[self.data_counter: self.data_counter + batch, 3: 30]
        action = self.train_data_numpy[self.data_counter: self.data_counter + batch, 0: 3]

        self.data_counter += batch
        if self.data_counter >= self.train_data_size:
            self.load_next_dataset()

        return np.fliplr(state), np.fliplr(action)

    def get_next_test_data(self, batch = 100):
        state = self.test_data_numpy[self.test_data_counter: self.test_data_counter + batch - 1, 3: 30]
        action = self.test_data_numpy[self.test_data_counter: self.test_data_counter + batch - 1, 0: 3]

        self.test_data_counter += batch
        if self.test_data_counter >= self.test_data_size:
            self.load_next_test_dataset()

        return np.fliplr(state), np.fliplr(action)

    def get_train_data_count(self):
        return min((self.max_train_dataset_index + 1) * self.train_data_size, (self.max_test_dataset_index - self.max_train_dataset_index) * self.train_data_size) - 1

    def load_next_dataset(self):
        self.train_data = self.file.get('sa' + str(self.dataset_index))
        self.train_data_numpy = np.array(self.train_data)
        self.train_data_size = self.train_data_numpy.shape[0]

        self.train_data_numpy = np.delete(self.train_data_numpy, 2, 1)

        self.data_counter = 0
        self.dataset_index += 1

        print(str(self.dataset_index) + 'th dataset loaded')
        if self.dataset_index == self.max_train_dataset_index:
            self.dataset_index = 0
            print('Last dataset loaded')

    def load_next_test_dataset(self):
        self.test_data = self.file.get('sa' + str(self.test_dataset_index))
        self.test_data_numpy = np.array(self.test_data)
        self.test_data_size = self.test_data_numpy.shape[0]

        self.test_data_numpy = np.delete(self.test_data_numpy, 2, 1)

        self.test_data_counter = 0
        self.test_dataset_index += 1

        if self.test_dataset_index == self.max_test_dataset_index:
            self.test_dataset_index = 0
            print('Last test dataset loaded')

    def close(self):
        self.file.close()

