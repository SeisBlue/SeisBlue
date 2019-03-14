import numpy as np
from tensorflow.python.keras.utils import Sequence

from obspy import read


class DataGenerator(Sequence):
    def __init__(self, pkl_list, batch_size=32, dim=(1, 3001, 1), shuffle=False):
        self.dim = dim
        self.batch_size = batch_size
        self.pkl_list = pkl_list
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.pkl_list))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.pkl_list) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        temp_pkl_list = [self.pkl_list[k] for k in indexes]
        wavefile, probability = self.__data_generation(temp_pkl_list)

        return wavefile, probability

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, temp_pkl_list):
        wavefile = np.empty((self.batch_size, *self.dim))
        probability = np.empty((self.batch_size, *self.dim))
        for i, ID in enumerate(temp_pkl_list):
            trace = read(ID).traces[0]
            wavefile[i,] = trace.data.reshape(self.dim)
            probability[i,] = trace.pdf.reshape(self.dim)

        return wavefile, probability


class PredictGenerator(DataGenerator):
    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        temp_pkl_list = [self.pkl_list[k] for k in indexes]
        wavefile = self.__data_generation(temp_pkl_list)

        return wavefile

    def __data_generation(self, temp_pkl_list):
        wavefile = np.empty((self.batch_size, *self.dim))
        for i, ID in enumerate(temp_pkl_list):
            trace = read(ID).traces[0]
            wavefile[i,] = trace.data.reshape(*self.dim)

        return wavefile
