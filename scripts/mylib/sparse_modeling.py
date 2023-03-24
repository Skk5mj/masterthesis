import os
import os.path as osp
import glob
import copy
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import Lasso
from group_lasso import GroupLasso
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error as MSE
import matplotlib.pyplot as plt
import seaborn as sns
from mylib.config import CFG

"""
sparse group lasso 
https://www.tandfonline.com/doi/abs/10.1080/10618600.2012.681250
https://www.frontiersin.org/articles/10.3389/fnagi.2018.00252/full
- sparse group lasso = group lasso + lasso
"""


def sparsegrouplassocv(
    data, group_ids, n_splits=10, n_lambdas=50, alpha=0.5, pheno=True
):
    kf = KFold(n_splits=n_splits)
    scaler = StandardScaler()
    # lambdas = np.logspace(-5, 1, num=n_lambdas) # <- 平らな部分が多かった
    lambdas = np.logspace(-4, 0, num=n_lambdas)
    mean_list = []
    std_list = []
    sem_list = []
    coef_list = []
    best_score = 1e5
    for c in lambdas:
        mse_list = []
        for n_fold, (train_idx, test_idx) in enumerate(kf.split(data)):
            # print(f"========================== {n_fold + 1} / {n_splits} ==========================")
            # subID除去,trainとtestにsplit -> 標準化
            train_data = scaler.fit_transform(data.iloc[train_idx, 1:])
            test_data = scaler.transform(data.iloc[test_idx, 1:])
            if pheno:  # FCS + (age, gender)
                X_train, y_train = train_data[:, 1:], train_data[:, 0]
                X_test, y_test = test_data[:, 1:], test_data[:, 0]
            else:  # FCSのみ
                X_train, y_train = train_data[:, 3:], train_data[:, 0]
                X_test, y_test = test_data[:, 3:], test_data[:, 0]
            reg = GroupLasso(
                groups=group_ids,
                group_reg=(1 - alpha) * c,
                l1_reg=alpha * c,
                n_iter=100000,  # 大きめにしておかないと収束しない
                random_state=42,
                old_regularisation=False,  # Trueにするとバグでfitすらしない
                fit_intercept=False,  # 標準化済みなので切片いらない
                supress_warning=True,
            )
            reg.fit(X_train, y_train)
            y_pred = reg.predict(X_test)
            mse = MSE(y_test, y_pred)
            mse_list.append(mse)
            coef_list.append(reg.coef_)
        mean = np.mean(mse_list)
        std = np.std(mse_list)
        sem = std / np.sqrt(len(mse_list))
        if mean < best_score:
            best_score = mean
            tmp_std = std
            best_param = {"lambda": c, "alpha": alpha}
        mean_list.append(mean)
        std_list.append(std)
        sem_list.append(sem)
    return {
        "best_param": best_param,
        "best_score": best_score,
        "best_score_std": tmp_std,
        "mean": mean_list,
        "std": std_list,
        "sem": sem_list,
        "mse": mse_list,
        "coef": coef_list,
        "lambdas": lambdas,
    }
