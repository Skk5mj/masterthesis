import torch
import torch.nn.functional as F
import torch.nn as nn
# torch_geometricそのままのtopkpoolingをr-poolとして用いると論文の実装と異なっている気がする
# スコアを重みのL2ノルムで正規化したあとの標準化のステップがない。
from torch_geometric.nn import TopKPooling 
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.utils import add_self_loops, sort_edge_index, remove_self_loops
from torch_sparse import spspmm
from net.braingraphconv import MyNNConv

# 畳み込み層を3つに増やす
class Network3layers(torch.nn.Module):
    def __init__(self, indim, ratio, nclass, k=8, R=116):
        """
        :param indim: (int) node feature dimension
        :param ratio: (float) pooling ratio in (0,1)
        :param nclass: (int)  number of classes
        :param k: (int) number of communities
        :param R: (int) number of ROIs
        """
        super(Network3layers, self).__init__()

        self.indim = indim
        self.dim1 = 32
        self.dim2 = 32
        self.dim3 = 32
        self.k = k
        self.R = R

        self.n1 = nn.Sequential(
            nn.Linear(self.R, self.k, bias=False),
            nn.ReLU(),
            nn.Linear(self.k, self.dim1 * self.indim),
        )
        self.conv1 = MyNNConv(self.indim, self.dim1, self.n1, normalize=False)
        self.pool1 = TopKPooling(
            self.dim1, ratio=ratio, multiplier=1, nonlinearity=torch.sigmoid
        )
        self.n2 = nn.Sequential(
            nn.Linear(self.R, self.k, bias=False),
            nn.ReLU(),
            nn.Linear(self.k, self.dim2 * self.dim1),
        )
        self.conv2 = MyNNConv(self.dim1, self.dim2, self.n2, normalize=False)
        self.pool2 = TopKPooling(
            self.dim2, ratio=ratio, multiplier=1, nonlinearity=torch.sigmoid
        )
        self.n3 = nn.Sequential(
            nn.Linear(self.R, self.k, bias=False),
            nn.ReLU(),
            nn.Linear(self.k, self.dim2 * self.dim3),
        )
        self.conv3 = MyNNConv(self.dim2, self.dim3, self.n3, normalize=False)
        self.pool3 = TopKPooling(
            self.dim3, ratio=ratio, multiplier=1, nonlinearity=torch.sigmoid
        )

        self.fc1 = torch.nn.Linear(
            (self.dim1 + self.dim2 + self.dim3) * 2,
            (self.dim1 + self.dim2 + self.dim3) // 4,
        )
        self.bn1 = torch.nn.BatchNorm1d((self.dim1 + self.dim2 + self.dim3) // 4)
        self.fc2 = torch.nn.Linear((self.dim1 + self.dim2 + self.dim3) // 4, nclass)

    def forward(self, data):
        x, edge_index, batch, edge_attr, pos = (
            data.x,
            data.edge_index,
            data.batch,
            data.edge_attr,
            data.pos,
        )
        # 1st conv layer
        x = self.conv1(x, edge_index, edge_attr, pos)  # conv
        x, edge_index, edge_attr, batch, perm, score1, score1_all = self.pool1(
            x, edge_index, edge_attr, batch
        )  # pooling
        pos = pos[perm]
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)  # readout

        edge_attr = edge_attr.squeeze()
        edge_index, edge_attr = self.augment_adj(edge_index, edge_attr, x.size(0))

        # 2nd conv layer
        x = self.conv2(x, edge_index, edge_attr, pos)  # conv
        x, edge_index, edge_attr, batch, perm, score2, score2_all = self.pool2(
            x, edge_index, edge_attr, batch
        )  # pooling
        pos = pos[perm]
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)  # readout
        edge_attr = edge_attr.squeeze()
        edge_index, edge_attr = self.augment_adj(edge_index, edge_attr, x.size(0))
        # 3rd conv layer
        x = self.conv3(x, edge_index, edge_attr, pos)
        x, edge_index, edge_attr, batch, perm, score3, score3_all = self.pool3(
            x, edge_index, edge_attr, batch
        )
        # pos = pos[perm]
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        # edge_attr = edge_attr.squeeze()
        # edge_index, edge_attr = self.augment_adj(edge_index, edge_attr, x.size(0))
        # aggregation
        x = torch.cat([x1, x2, x3], dim=1)
        # MLP for regression
        x = self.bn1(F.relu(self.fc1(x)))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)

        # output
        output = {
            "pred": x,
            "w1": self.pool1.weight,
            "w2": self.pool2.weight,
            "w3": self.pool3.weight,
            "s1": torch.sigmoid(score1).view(x.size(0), -1),
            "s1_all": torch.sigmoid(score1_all).view(x.size(0), -1),
            "s2": torch.sigmoid(score2).view(x.size(0), -1),
            "s2_all": torch.sigmoid(score2_all).view(x.size(0), -1),
            "s3": torch.sigmoid(score3).view(x.size(0), -1),
            "s3_all": torch.sigmoid(score3_all).view(x.size(0), -1),
        }
        return output

    def augment_adj(self, edge_index, edge_weight, num_nodes):
        edge_index, edge_weight = add_self_loops(
            edge_index, edge_weight, num_nodes=num_nodes
        )
        edge_index, edge_weight = sort_edge_index(edge_index, edge_weight, num_nodes)
        edge_index, edge_weight = spspmm(
            edge_index,
            edge_weight,
            edge_index,
            edge_weight,
            num_nodes,
            num_nodes,
            num_nodes,
        )
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        return edge_index, edge_weight
