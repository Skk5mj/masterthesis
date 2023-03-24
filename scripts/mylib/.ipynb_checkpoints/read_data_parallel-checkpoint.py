'''
Author: Xiaoxiao Li
Date: 2019/02/24
'''

import os.path as osp
from os import listdir
import os
import glob
import h5py

from joblib import delayed, Parallel
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data
import networkx as nx
from networkx.convert_matrix import from_numpy_matrix
# import multiprocessing
from torch_sparse import coalesce
from torch_geometric.utils import remove_self_loops
# from functools import partial # 引数のいらない呼び出し可能オブジェクトに加工して使いやすくする
import deepdish as dd
from imports.gdc import GDC
from mylib import calc_feature
from mylib.config import CFG

path_subinfo = CFG.path_subinfo
df_subinfo = pd.read_csv(path_subinfo, index_col = 0)
label = df_subinfo.global_gm_bhq # タスクに合わせて変更する必要がある。柔軟性がないので将来的にこの部分は書き換える(2022/10/27)


def split(data, batch):
    node_slice = torch.cumsum(torch.from_numpy(np.bincount(batch)), 0)
    node_slice = torch.cat([torch.tensor([0]), node_slice])

    row, _ = data.edge_index
    edge_slice = torch.cumsum(torch.from_numpy(np.bincount(batch[row])), 0)
    edge_slice = torch.cat([torch.tensor([0]), edge_slice])

    # Edge indices should start at zero for every graph.
    data.edge_index -= node_slice[batch[row]].unsqueeze(0)

    slices = {'edge_index': edge_slice}
    if data.x is not None:
        slices['x'] = node_slice
    if data.edge_attr is not None:
        slices['edge_attr'] = edge_slice
    if data.y is not None:
        if data.y.size(0) == batch.size(0):
            slices['y'] = node_slice
        else:
            slices['y'] = torch.arange(0, batch[-1] + 2, dtype=torch.long)
    if data.pos is not None:
        slices['pos'] = node_slice

    return data, slices


def cat(seq):
    seq = [item for item in seq if item is not None]
    seq = [item.unsqueeze(-1) if item.dim() == 1 else item for item in seq]
    return torch.cat(seq, dim=-1).squeeze() if len(seq) > 0 else None

# 並列処理のためのクラス？
# class NoDaemonProcess(multiprocessing.Process):
#     @property
#     def daemon(self):
#         return False

#     @daemon.setter
#     def daemon(self, value):
#         pass


# class NoDaemonContext(type(multiprocessing.get_context())):
#     Process = NoDaemonProcess

    
# dataの読み込み
# ここを書き換えればいける？
def read_data(data_dir, use_gdc = False):
    data = np.load(data_dir)
    batch = []
    pseudo = []
    y_list = []
    edge_att_list, edge_index_list,att_list = [], [], []

    import timeit
    num_sub = data['pearson'].shape[0]
    start = timeit.default_timer()
    print('reading data starts.')
    res = Parallel(n_jobs = -1)(delayed(read_single_data)(data['pearson'][i], data['partial'][i], label[i], use_gdc) for i in range(num_sub))
#     res = pool.map(func, onlyfiles)

#     pool.close()
#     pool.join()

    stop = timeit.default_timer()

    print('Time: ', stop - start)



    for j in range(len(res)):
        edge_att_list.append(res[j][0])
        edge_index_list.append(res[j][1]+j*res[j][4])
        att_list.append(res[j][2])
        y_list.append(res[j][3])
        batch.append([j]*res[j][4])
        pseudo.append(np.diag(np.ones(res[j][4])))

    edge_att_arr = np.concatenate(edge_att_list)
    edge_index_arr = np.concatenate(edge_index_list, axis=1)
    att_arr = np.concatenate(att_list, axis=0)
    pseudo_arr = np.concatenate(pseudo, axis=0)
    y_arr = np.stack(y_list)
    edge_att_torch = torch.from_numpy(edge_att_arr.reshape(len(edge_att_arr), 1)).float()
    att_torch = torch.from_numpy(att_arr).float()
    y_torch = torch.from_numpy(y_arr).long()  # classification
    batch_torch = torch.from_numpy(np.hstack(batch)).long()
    edge_index_torch = torch.from_numpy(edge_index_arr).long()
    pseudo_torch = torch.from_numpy(pseudo_arr).float()
    data = Data(x=att_torch, edge_index=edge_index_torch, y=y_torch, edge_attr=edge_att_torch, pos = pseudo_torch )


    data, slices = split(data, batch_torch)

    return data, slices

# ここを書き換えればいける
def read_single_data(pearson_matrix, partialcorr_matrix, label, use_gdc = False):
    '''
    bold timeseriesから計算したピアソン相関行列と偏相関行列を入力にしてdataを返す。
    Paramters
    ---------
    matrix : np.array
        bold timeseriesから計算されたピアソン相関と偏相関一人分。keyで指定して取り出せる。
        keyは'pearson'と'partial'
    use_gdc : bool
        gdcを使うか否か。多分使わない。
        
    Returns
    -------
    edge_att.data.numpy(),edge_index.data.numpy(),att,label,num_nodes
    
    '''
    att = pearson_matrix
    pcorr = np.abs(partialcorr_matrix)
    # グラフを作成
    num_nodes = pcorr.shape[0]
    G = from_numpy_matrix(pcorr)
    A = nx.to_scipy_sparse_matrix(G)
    adj = A.tocoo()
    edge_att = np.zeros(len(adj.row))
    for i in range(len(adj.row)):
        edge_att[i] = pcorr[adj.row[i], adj.col[i]]

    edge_index = np.stack([adj.row, adj.col])
    edge_index, edge_att = remove_self_loops(torch.from_numpy(edge_index), torch.from_numpy(edge_att))
    edge_index = edge_index.long()
    edge_index, edge_att = coalesce(edge_index, edge_att, num_nodes,
                                    num_nodes)

    att_torch = torch.from_numpy(att).float()
    y_torch = torch.from_numpy(np.array(label)).float()  # regression

    data = Data(x=att_torch, edge_index=edge_index.long(), y=y_torch, edge_attr=edge_att)

    if use_gdc:
        '''
        Implementation of https://papers.nips.cc/paper/2019/hash/23c894276a2c5a16470e6a31f4618d73-Abstract.html
        '''
        data.edge_attr = data.edge_attr.squeeze()
        gdc = GDC(self_loop_weight=1, normalization_in='sym',
                  normalization_out='col',
                  diffusion_kwargs=dict(method='ppr', alpha=0.2),
                  sparsification_kwargs=dict(method='topk', k=20,
                                             dim=0), exact=True)
        data = gdc(data)
        return data.edge_attr.data.numpy(),data.edge_index.data.numpy(),data.x.data.numpy(),data.y.data.item(),num_nodes

    else:
        return edge_att.data.numpy(),edge_index.data.numpy(),att,label,num_nodes

# if __name__ == "__main__":
#     filename = '50346.h5'
#     read_single_data(data_dir, filename)






