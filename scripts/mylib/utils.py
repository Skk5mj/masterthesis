import random
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold
import torch
import torch.nn.functional as F
from scipy.io import loadmat
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold

# https://qiita.com/si1242/items/d2f9195c08826d87d6ad
def seed_worker(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True # <= 処理速度が落ちるかも


# loss function
def topk_loss(s, ratio, EPS=1e-10):
    # if ratio > 0.5:
    #     ratio = 1 - ratio
    s = s.sort(dim=1).values
    res = (
        -torch.log(s[:, -int(s.size(1) * ratio) :] + EPS).mean()
        - torch.log(1 - s[:, : int(s.size(1) * ratio)] + EPS).mean()
    )
    return res


def unit_loss(weight):
    return (torch.norm(weight, p=2) - 1) ** 2


def criterion(output, data_list, ratio, lambda_p=0.1, lambda_tpk=0.1):
    y_pred = output["pred"]
    loss_mse = F.mse_loss(
        y_pred.squeeze(dim=-1),
        torch.cat([data.y for data in data_list]).to(y_pred.device),
    )
    loss_p1 = unit_loss(output["w1"])
    loss_p2 = unit_loss(output["w2"])
    loss_p3 = unit_loss(output["w3"])
    loss_tpk1 = topk_loss(output["s1"], ratio)
    loss_tpk2 = topk_loss(output["s2"], ratio)
    loss_tpk3 = topk_loss(output["s3"], ratio)
    loss = (
        loss_mse
        + lambda_p * (loss_p1 + loss_p2 + loss_p3)
        + lambda_tpk * (loss_tpk1 + loss_tpk2 + loss_tpk3)
    )
    return loss


# learning function
def train_step(data_list, model, optimizer, ratio, lambda_p=0.1, lambda_tpk=0.1):
    model.train()
    output = model(data_list)
    loss = criterion(output, data_list, ratio)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return output, loss


def val_step(data_list, model, ratio, lambda_p=0.1, lambda_tpk=0.1):
    model.eval()
    with torch.no_grad():
        output = model(data_list)
        loss = criterion(output, data_list, ratio, lambda_p, lambda_tpk)
    return output, loss


def test_step(data_list, model, ratio, lambda_p=0.1, lambda_tpk=0.1):
    model.eval()
    with torch.no_grad():
        output = model(data_list)
        loss = criterion(output, data_list, ratio, lambda_p, lambda_tpk)
    return output, loss


class EarlyStopping:
    """
    早期終了 (early stopping)
    """

    def __init__(self, patience=0, verbose=0, mode="loss"):
        self._step = 0
        self.patience = patience
        self.verbose = verbose

        self.mode = mode

        if self.mode == "loss":
            self._loss = float("inf")
        elif self.mode == "score":
            self._score = 0.0
        else:
            raise Exception("error")

    def __call__(self, value):
        if self.mode == "loss":
            if self._loss < value:
                self._step += 1
                if self._step > self.patience:
                    if self.verbose:
                        print("early stopping")
                    return True
            else:
                self._step = 0
                self._loss = value

            return False

        elif self.mode == "score":
            if self._score > value:
                self._step += 1
                if self._step > self.patience:
                    if self.verbose:
                        print("early stopping")
                    return True
            else:
                self._step = 0
                self._score = value

            return False
