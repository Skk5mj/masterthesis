import torch
from torch_geometric.data import InMemoryDataset, Data
from os.path import join, isfile
from os import listdir
import numpy as np
import os.path as osp
from mylib.read_data_parallel import read_data
from mylib.config import CFG

path_processed_data = CFG.path_processed_data


class NKIRSDataset(InMemoryDataset):
    def __init__(self, data_path, transform=None):
        self.transform = transform
        self.data_path = data_path
        self._indices = None
        if isfile(path_processed_data) == False:
            data, slices = read_data(self.data_path)
            self.data, self.slices = data, slices
            torch.save((self.data, self.slices), path_processed_data)
        else:
            self.data, self.slices = torch.load(path_processed_data)

    #     def __getitem__(self,idx): # No need to overwrite
    #         _data = self.data[idx]
    #         _slices = self.slices[idx]
    #         return _data, _slices
    def __repr__(self):
        return "{}({})".format(self.name, len(self))

    def len(self):
        return self.data.y.shape[0]
