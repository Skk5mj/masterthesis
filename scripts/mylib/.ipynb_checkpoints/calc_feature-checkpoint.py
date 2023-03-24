import os
from joblib import Parallel, delayed
import pandas as pd
import numpy as np
from nilearn.connectome import ConnectivityMeasure
import torch
import networkx as nx
from networkx.convert_matrix import from_numpy_matrix
from torch_geometric.utils import remove_self_loops
from torch_sparse import coalesce

def compute_connectivity(data, kind = 'correlation', save = False, save_dir = '', name = ''):
    """
    timeseriesをもとにピアソン相関か偏相関を計算する
    
    Parameters
    ----------
    data : np.array
        make_data()関数で整理したnumpy配列(n_sub, n_TR, n_ROI)
    kind : str
        接続性の種類。'correlation'=>ピアソン相関/'partial correlation'=>偏相関
    save : bool
        配列を保存するか否か
    save_dir : str
        保存するディレクトリのパス
    name : str
        保存時につける名前
    Returns
    -------
    connectivity_matrices : np.array
        全被験者の接続性行列(n_sub, n_ROI, n_ROI)
    """
    if kind not in ['correlation', 'partial correlation']:
        print(f'{kind} is a wrong keyword.')
    
    conn_measure = ConnectivityMeasure(kind = kind)
    conn_array = ConnectivityMeasure.fit_transform(data)
    if save:
        name += '.npz'
        path = os.path.join(save_dir, name)
        np.savez(path, conn_array)
        print(f'output {name} is saved.')
    return conn_array

def thresholding_matrix(pcorr_matrix, threshold = 0.1):
    '''
    入力された偏相関行列を閾値処理する
    
    Parameters
    ----------
    pcorr_matrices : np.array
        被験者の偏相関行列
    threshold : float
        閾値
    
    Returns
    -------
    pcorr_filterd : np.array
        閾値処理された偏相関行列
    '''
    n_sub = pcorr_matrix.shape[0]
    n_node = pcorr_matrix.shape[1]
    if threshold > 0.5:
            threshold = 1- threshold
    pcorr_filtered = np.zeros_like(pcorr_matrix)
    for i in range(n_sub):
        pcorr_1d = np.abs(pcorr_matrix[i]).flatten()
        pcorr_sorted = np.sort(pcorr_1d)[::-1] # 大きい順でsort
        _filter = np.where(pcorr_1d > pcorr_sorted[int((n_node ** 2) * threshold)], 1, 0) # 閾値を越えれば1,else 0
        pcorr_1d = pcorr_1d * _filter
        pcorr_filtered[i] = pcorr_1d.reshape(n_node, n_node)
    return pcorr_filtered



# ちょっと遅いから並列処理に後で書き換えたい
def get_edge_feature(pcorr_matrices): 
    '''
    閾値処理された偏相関行列からモデルの入力のためのエッジ特徴量へ変換する。
    
    Parameters
    ----------
    pcorr_matrices : np.array
        閾値処理された偏相関行列. n_sub x n_rois x n_rois
    
    Returns
    -------
    edge_idx_list : torch.tensor
        dtype = torch.long
    edge_att_list : torch.tensor
        dtype = torch.float
    '''
    num_sub = pcorr_matrices.shape[0] 
    num_nodes = pcorr_matrices.shape[-1]
    edge_idx_list = []
    edge_att_list = []
    for i in range(num_sub):
        pcorr_matrix = pcorr_matrices[i]
        G = from_numpy_matrix(pcorr_matrix)
        A = nx.to_scipy_sparse_matrix(G)
        adj = A.tocoo()
        edge_att = np.zeros(len(adj.row))
        for j in range(len(adj.row)):
            edge_att[j] = pcorr_matrix[adj.row[j], adj.col[j]]
        edge_idx = np.stack([adj.row, adj.col])
        edge_idx, edge_att = remove_self_loops(torch.from_numpy(edge_idx), torch.from_numpy(edge_att))
        edge_idx = edge_idx.long()
        edge_idx, edge_att = coalesce(edge_idx, edge_att, num_nodes,
                                        num_nodes)
        edge_idx_list.append(edge_idx) # listのなかにtensorがいっぱい入っているので
        edge_att_list.append(edge_att)
        
    edge_idx_list = np.array([t.numpy() for t in edge_idx_list])
    edge_att_list = np.array([t.numpy() for t in edge_att_list])
    
    return torch.from_numpy(edge_idx_list), torch.from_numpy(edge_att_list)

# def get_edge_feature_single(pcorr_matrix):
#     num_nodes = pcorr_matrix.shape[-1]
#     G = from_numpy_matrix(pcorr_matrix)
#     A = nx.to_scipy_sparse_matrix(G)
#     adj = A.tocoo()
#     edge_att = np.zeros(len(adj.row))
#     for j in range(len(adj.row)):
#         edge_att[j] = pcorr_matrix[adj.row[j], adj.col[j]]
#     edge_idx = np.stack([adj.row, adj.col])
#     edge_idx, edge_att = remove_self_loops(torch.from_numpy(edge_idx), torch.from_numpy(edge_att))
#     edge_idx = edge_idx.long()
#     edge_idx, edge_att = coalesce(edge_idx, edge_att, num_nodes,
#                                     num_nodes)
#     return edge_idx, edge_att

# def get_edge_feature_parallel(pcorr_matrices):
#     num_sub = pcorr_matrices.shape[0]
#     edge_idx_list = Parallel(n_jobs=-1)(delayed(get_edge_feature_single)(pcorr_matrices[i]) for i in range(num_sub))
    
#     return edge_idx_list, edge_att_list