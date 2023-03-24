import os
import os.path as osp
import sys
import pickle
import numpy as np
import argparse
import time
import copy
import gc
import warnings
import yaml
from attrdict import AttrDict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
from torch_geometric.seed import seed_everything

# for multi-gpu training
from torch_geometric.nn import DataParallel as DP
from torch_geometric.data import DataListLoader

from mylib.NKIRSDataset import NKIRSDataset

# from mylib import calc_feature
from mylib.config import CFG

# utils
from mylib.utils import seed_worker
from mylib.utils import criterion
from mylib.utils import train_step
from mylib.utils import val_step
from mylib.utils import test_step
from mylib.utils import EarlyStopping

# vizualization
from mylib.visualization import plot_loss_curve
from mylib.visualization import plot_scatter
from mylib.visualization import plot_cluster_AAL
from net.braingnn import Network3layers

warnings.filterwarnings("ignore")

sub_info = CFG.path_subinfo
path_features = CFG.path_features

df_subjects_info = pd.read_csv(sub_info, index_col=0)
features = np.load(path_features)

bhq_mean = df_subjects_info.gm_bhq.mean()
bhq_std = df_subjects_info.gm_bhq.std()
print(f"bhq mean : {bhq_mean} || bhq std : {bhq_std}")

torch.manual_seed(42)
EPS = 1e-10

name = "NKI-RS"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load data
data_path = CFG.path_features
dataset = NKIRSDataset(data_path)
dataset.data.y = dataset.data.y.squeeze()
dataset.data.x[dataset.data.x == float("inf")] = 0

with open("param.yaml") as f:
    tmp = yaml.safe_load(f)  # config is dict
    prm_set = AttrDict(tmp)
prm_set1 = prm_set.param_set1
prm_set2 = prm_set.param_set2
prm_set3 = prm_set.param_set3
prm_set4 = prm_set.param_set4


def cross_val(dataset_all, param_dict, exp_name, save_model=True):
    seed_everything(seed=42)
    print(f"parameters : {param_dict}")
    # setting parameters
    (
        n_epochs,
        n_splits,
        lr_init,
        weight_decay,
        patience,
        ratio,
        k,
        lambda_p,
        lambda_tpk,
        batch_size,
    ) = param_dict.values()
    lr_init = float(lr_init)
    weight_decay = float(weight_decay)
    kf = KFold(n_splits=n_splits, shuffle=False)
    save_dir = osp.join("./log", exp_name)
    if not osp.exists(save_dir):
        os.makedirs(save_dir)
    else:
        raise Exception(
            f"This experiment has been done! Check f{save_dir} exists and {exp_name}!!"
        )
    loss_best_fold = 1e10
    mae_list = []
    predictions = []
    results = dict()

    # cross validation
    s1_all = []
    s2_all = []
    s3_all = []
    for n_fold, (train_idx, test_idx) in enumerate(kf.split(dataset_all)):
        best_loss = 1e10
        best_epoch = None
        print(
            f"===================== This is fold #{n_fold + 1} / {n_splits}. ====================="
        )
        exp_dir = osp.join(save_dir, "fold_" + str(n_fold + 1))
        if not os.path.exists(exp_dir):
            os.mkdir(exp_dir)
        # deviding data into train_outer and test
        train_dataset_outer = dataset[train_idx]
        test_dataset = dataset[test_idx]

        model = Network3layers(indim=116, ratio=ratio, k=k, nclass=1, R=116).to(device)
        if torch.cuda.device_count() > 1:
            print(f"Let's use {torch.cuda.device_count()} GPUs!!")
            model = DP(model)
        # optimizer & scheduler
        optimizer = optim.AdamW(
            model.parameters(), lr=lr_init, weight_decay=weight_decay
        )
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-4)
        # prepare dataloaders
        train_loader = DataListLoader(
            train_dataset_outer, batch_size=batch_size, shuffle=False
        )
        test_loader = DataListLoader(test_dataset, batch_size=batch_size)
        train_loss_list = []
        test_loss_list = []
        # early stopping
        es = EarlyStopping(patience=patience, verbose=1, mode="loss")
        stopped_epoch = n_epochs
        for epoch in range(n_epochs):
            # training phase with outer train dataset
            model.train()
            train_loss_all = 0.0
            for data_list in train_loader:
                optimizer.zero_grad()
                output, train_loss = train_step(
                    data_list,
                    model,
                    optimizer,
                    ratio=ratio,
                    lambda_p=lambda_p,
                    lambda_tpk=lambda_tpk,
                )
                scheduler.step()
                train_loss_all += train_loss.item() * len(data_list)
            train_loss_all = train_loss_all / len(train_loader.dataset)
            # evaluating phase with test dataset
            model.eval()
            with torch.no_grad():
                test_loss_all = 0.0
                for data_list in test_loader:
                    output, test_loss = test_step(
                        data_list,
                        model,
                        ratio=ratio,
                        lambda_p=lambda_p,
                        lambda_tpk=lambda_tpk,
                    )
                    test_loss_all += test_loss.item() * len(data_list)
                test_loss_all = test_loss_all / len(test_loader.dataset)
            if epoch == 0 or (epoch + 1) % (n_epochs // 10) == 0:
                print(
                    f"Epoch : [{epoch + 1} / {n_epochs}] || Train Loss : {train_loss_all} | Test Loss : {test_loss_all}"
                )
            train_loss_list.append(train_loss_all)
            test_loss_list.append(test_loss_all)
            if test_loss_all < best_loss and (epoch + 1) > 5:
                best_epoch = epoch + 1
                best_loss = test_loss_all
                best_model_wts = copy.deepcopy(model.state_dict())
                if save_model:
                    model_path = osp.join(exp_dir, "model_weights.pth")
                    torch.save(best_model_wts, model_path)
            # early stopping
            if epoch <= 50:
                pass
            elif es(test_loss_all):
                stopped_epoch = epoch + 1
                print(f"Training phase stopped in epoch#{stopped_epoch}.")
                break
        # *********************** all epochs done here ***********************
        print(f"fold {n_fold} is done!!")
        print(f"best epoch is #{best_epoch}. test loss : {best_loss}")
        # plotting loss curve
        plot_loss_curve(
            train_loss_list=train_loss_list,
            test_loss_list=test_loss_list,
            stopped_epoch=stopped_epoch,
            n_fold=n_fold,
            save_fig=True,
            save_path=exp_dir,
            bhq_type="gm",
        )
        # predicting with best model for vizualizing results
        model.load_state_dict(torch.load(model_path))
        with torch.no_grad():
            train_pred = []
            test_pred = []
            # added (2023/01/18)
            s1_list = []
            s2_list = []
            s3_list = []
            for data_list in train_loader:
                output, train_loss = test_step(
                    data_list,
                    model,
                    ratio=ratio,
                    lambda_p=lambda_p,
                    lambda_tpk=lambda_tpk,
                )
                train_pred.append(output["pred"])
                s1_list.append(output["s1_all"].view(-1).detach().cpu().numpy())
                s2_list.append(output["s2_all"].view(-1).detach().cpu().numpy())
                s3_list.append(output["s3_all"].view(-1).detach().cpu().numpy())
                s1_arr = np.hstack(s1_list)
                s2_arr = np.hstack(s2_list)
                s3_arr = np.hstack(s3_list)
            s1_all.append(s1_arr)
            s2_all.append(s2_arr)
            s3_all.append(s3_arr)
            for data_list in test_loader:
                output, test_loss = test_step(
                    data_list,
                    model,
                    ratio=ratio,
                    lambda_p=lambda_p,
                    lambda_tpk=lambda_tpk,
                )
                test_pred.append(output["pred"])
        train_label = train_loader.dataset.data.y[train_idx].numpy()
        train_pred = torch.cat(train_pred).to("cpu").detach().numpy().copy().squeeze()
        test_label = test_loader.dataset.data.y[test_idx].numpy()
        test_pred = torch.cat(test_pred).to("cpu").detach().numpy().copy().squeeze()
        predictions.append(test_pred)
        # ************************ result visualization ***********************
        # train data
        plot_scatter(
            y_true=train_label,
            y_pred=train_pred,
            train=True,
            exp_name=exp_name,
            save_fig=True,
            save_path=exp_dir,
        )
        # test data
        plot_scatter(
            y_true=test_label,
            y_pred=test_pred,
            train=False,
            exp_name=exp_name,
            save_fig=True,
            save_path=exp_dir,
        )
        mae = MAE(test_label, test_pred) * bhq_std
        mae_list.append(mae)
        if mae < loss_best_fold:
            best_fold = n_fold + 1
            loss_best_fold = mae

        # freeing memory
        del (
            model,
            train_dataset_outer,
            train_loader,
            test_dataset,
            test_loader,
            train_pred,
            train_label,
            test_pred,
            test_label,
        )
        gc.collect()
        torch.cuda.empty_cache()
    print(f"best fold is {best_fold}. MAE = {loss_best_fold}.")
    print(f"Mean MAE = {np.mean(mae_list)} / std = {np.std(mae_list)}.")
    predictions = np.concatenate(predictions)
    np.save(osp.join(save_dir, "predictions.npy"), predictions)
    results["params"] = param_dict
    results["mae"] = mae_list
    results["best_fold"] = best_fold
    results["s1"] = s1_all
    results["s2"] = s2_all
    results["s3"] = s3_all
    results_path = osp.join(save_dir, "results.pkl")
    with open(results_path, "wb") as f:
        pickle.dump(results, f)


# implement main calculation.
cross_val(dataset, param_dict=prm_set1, exp_name="Model1")
cross_val(dataset, param_dict=prm_set2, exp_name="Model2")
cross_val(dataset, param_dict=prm_set3, exp_name="Model3")
cross_val(dataset, param_dict=prm_set4, exp_name="Model4")
