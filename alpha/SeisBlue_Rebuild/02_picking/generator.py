# -*- coding: utf-8 -*-
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset, random_split
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
import h5py
import numpy as np
from datetime import datetime
from itertools import islice

from seisblue import core


class H5Dataset(Dataset):
    def __init__(self, h5_paths, limit=-1, cache_size=3):
        self.limit = limit
        self.h5_paths = h5_paths
        self.cache_size = cache_size
        self.cache = {}
        self.indices = {}
        idx = 0

        for a, h5_path in enumerate(h5_paths):
            with h5py.File(h5_path, "r") as archive:
                for key, instance in archive.items():
                    self.indices[idx] = (a, key)
                    idx += 1

    def _load_archive(self, archive_index):
        if archive_index not in self.cache:
            if len(self.cache) >= self.cache_size:
                old_key = next(iter(self.cache))
                self.cache[old_key].close()
                del self.cache[old_key]
            self.cache[archive_index] = h5py.File(self.h5_paths[archive_index], "r")
        return self.cache[archive_index]

    def get_data(self, index):
        a, key = self.indices[index]
        archive = self._load_archive(a)
        instance = archive[key]
        features, labels = read_hdf5(instance)
        if features.shape != (3, 2048) or labels.shape != (3, 2048):
            print(f'Wrong shape {features.shape} for {key}')
        return features, labels

    def __getitem__(self, index):
        results = self.get_data(index)
        if results:
            return results
        else:
            return self.__getitem__((index + 1) % len(self.indices))

    def __len__(self):
        if self.limit > 0:
            return min([len(self.indices), self.limit])
        return len(self.indices)

    def close(self):
        for archive in self.cache.values():
            archive.close()
        self.cache.clear()


class PickDataset(Dataset):
    def __init__(self, instances):
        self.features = torch.zeros(len(instances), 3, instances[0].timewindow.npts)
        self.labels = torch.zeros_like(self.features)
        for i, instance in enumerate(instances):
            self.features[i, :, :] = torch.Tensor(instance.features.data)
            self.labels[i, :, :] = torch.tensor(instance.labels[0].data)
    def __getitem__(self, index):
        feature = self.features[index]
        label = self.labels[index]

        return feature, label

    def __len__(self):
        return len(self.features)


class PickTestDataset(Dataset):
    def __init__(self, instances):
        self.features = torch.zeros(len(instances), 3, instances[0].timewindow.npts)
        for i, instance in enumerate(instances):
            self.features[i, :, :] = torch.Tensor(instance.features.data)

    def __getitem__(self, index):
        feature = self.features[index]
        return feature

    def __len__(self):
        return len(self.features)


def read_hdf5(instance):
    features = torch.from_numpy(np.array(instance['features'].get('data'),
                                         dtype=np.float32))
    labels = torch.from_numpy(np.array(instance['labels/manual'].get('data'),
                                       dtype=np.float32))
    return features, labels


if __name__ == '__main__':
    pass