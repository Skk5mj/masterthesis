import torch
from torch_geometric.data import InMemoryDataset,Data
from os.path import join, isfile
from os import listdir
import numpy as np
import os.path as osp
from mylib.read_data_parallel import read_data
from mylib.config import CFG

path_processed_data = CFG.path_processed_data

class NKIRSDataset(InMemoryDataset):
    def __init__(self, data_path):
        self.data_path = data_path
        if isfile(path_processed_data) == False:
            data, slices = read_data(self.data_path)
            self.data, self.slices = data, slices
            torch.save((self.data, self.slices), path_processed_data)
        else:
            self.data, self.slices = torch.load(path_processed_data)
            
#     def __getitem__(self,idx): # オーバーライドする必要なし
#         _data = self.data[idx]
#         _slices = self.slices[idx]
#         return _data, _slices
    def __repr__(self):
        return '{}({})'.format(self.name, len(self))




# 以下は箕輪さんのdatasetを使った定義だったがbraingnnに使うにはちょっと難しい感じだった
# import torch
# from torch_geometric.data import Dataset,Data
# class NKIRSDataset(Dataset):
#     def __init__(self, data, edge_idx, edge_att, label=None):
#         self.data = torch.tensor(data, dtype = torch.float32)
#         self.label = label
#         self.edge_idx = edge_idx
#         self.edge_att = edge_att
#         self.test = label is None
#         self.length = data.shape[0]
#     def __len__(self):
#         return self.length
#     def __getitem__(self,idx):
#         if self.test:
#             data = self.data[idx]
#             edge_idx = self.edge_idx[idx]
#             edge_att = self.edge_att[idx]
#             return data, edge_idx, edge_att
#         else:
#             data = self.data[idx]
#             label = self.label[idx]
#             edge_idx = self.edge_idx[idx]
#             edge_att = self.edge_att[idx]
#             return data, label, edge_idx, edge_att
