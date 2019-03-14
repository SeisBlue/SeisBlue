from obspy import read
import torch
from torch.utils.data import Dataset


class WaveProbDataset(Dataset):
    def __init__(self, pkl_list, dim=(1, 1, 3001), train=True):
        self.pkl_list = pkl_list
        self.dim = dim
        self.train = train

    def __getitem__(self, index):
        trace = read(self.pkl_list[index]).traces[0]
        wavefile = torch.FloatTensor(trace.data.reshape(self.dim))

        if self.train:
            probability = torch.FloatTensor(trace.pdf.reshape(self.dim))

            return wavefile, probability
        else:
            return wavefile

    def __len__(self):
        return len(self.pkl_list)
