# -*- coding: utf-8 -*-
import h5py
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset, random_split, RandomSampler
import numpy as np

import seisblue.io



class H5Dataset(Dataset):
    def __init__(self, h5_paths, flip=None, limit=-1):
        self.limit = limit
        self.h5_paths = h5_paths
        self._archives = [h5py.File(h5_path, "r") for h5_path in self.h5_paths]
        self.indices = {}
        self.flip = flip

        idx = 0
        for a, archive in enumerate(self.get_archives):
            for e, event in enumerate(archive.values()):
                for i in range(len(event['instances'])):
                    self.indices[idx] = (a, e, i)
                    idx += 1
            archive.close()
        self._archives = None

    @property
    def get_archives(self):
        if self._archives is None:  # lazy loading here!
            self._archives = [h5py.File(h5_path, "r") for h5_path in
                              self.h5_paths]
        return self._archives

    def get_instance(self, index):
        a, e, i = self.indices[index]

        archive = list(self.get_archives[a].values())[e]
        instance = seisblue.io.read_hdf5_instance(archive, index=i)

        return instance

    def __getitem__(self, index):
        instance = self.get_instance(index).instances[0]
        features = [trace.data for trace in instance.traces if
                    list(trace.channel)[-1] == 'Z']
        features = torch.from_numpy(np.array(features)).to(torch.float32)
        label = [label for label in instance.labels if label.tag == 'manual'][0]
        flip_features = (-1) * features
        flip_label = None
        if label.data.any():
            label = torch.from_numpy(label.data).to(torch.float32)
            flip_label = label
            if np.argmax(label) == 0:
                flip_label = torch.tensor([0, 1, 0], dtype=torch.float32)
            elif np.argmax(label) == 1:
                flip_label = torch.tensor([1, 0, 0], dtype=torch.float32)
        else:
            label = torch.Tensor([0, 0, 1], dtype=torch.float32)

        data = (features, label, flip_features, flip_label) if self.flip else (features, label)
        return data

    def __len__(self):
        if self.limit > 0:
            return min([len(self.indices), self.limit])
        length = len(self.indices) * 2 if self.flip else len(self.indices)
        return length


class DualDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, drop_last=True, shuffle=True,
                 **kwargs):
        if shuffle:
            sampler = RandomSampler(dataset)
        else:
            sampler = None

        super(DualDataLoader, self).__init__(dataset, batch_size,
                                             drop_last=drop_last,
                                             sampler=sampler, **kwargs)
        self.next_features = None
        self.next_label = None

    def __iter__(self):
        return DualDataLoaderIterator(self)


class DualDataLoaderIterator:
    def __init__(self, loader):
        self.loader = loader
        self.iter = iter(loader.dataset)

    def __next__(self):
        batch_features = []
        batch_labels = []

        for _ in range(self.loader.batch_size):
            if self.loader.next_features is not None:
                features = self.loader.next_features
                label = self.loader.next_label
                self.loader.next_features = None
                self.loader.next_label = None
            else:
                try:
                    features, label, flip_features, flip_label = next(self.iter)
                    self.loader.next_features = flip_features
                    self.loader.next_label = flip_label
                except KeyError as e:
                    if self.loader.drop_last:
                        raise StopIteration
                    else:
                        print('The last batch is not full.')
                        break

            batch_features.append(features)
            batch_labels.append(label)
        batch_features = torch.stack(batch_features, dim=0)
        batch_labels = torch.stack(batch_labels, dim=0)
        return batch_features, batch_labels

