import os
import os.path as osp
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import nilearn
from nilearn import plotting, image
from nilearn.datasets import fetch_atlas_aal
import torch
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
import skimage
import skimage.measure as me

from mylib.config import CFG

# bhq_mean = df_subinfo.global_gm_bhq.mean()
# bhq_std = df_subinfo.global_gm_bhq.std()
df_subinfo_gm = pd.read_csv(CFG.path_subinfo, header=0, index_col=0)


def plot_loss_curve(
    train_loss_list,
    test_loss_list,
    stopped_epoch,
    n_fold,
    save_fig=False,
    save_path=None,
):
    """_summary_

    Args:
        train_loss_list (_type_): _description_
        test_loss_list (_type_): _description_
        stopped_epoch (_type_): _description_
        n_fold (_type_): _description_
        save_fig (bool, optional): _description_. Defaults to False.
        save_path (_type_, optional): _description_. Defaults to None.
    """
    bhq_mean = df_subinfo_gm["gm_bhq"].mean()
    bhq_std = df_subinfo_gm["gm_bhq"].std

    fig = plt.figure()
    x = np.arange(stopped_epoch)
    plt.plot(x, train_loss_list, color="coral", label="Train")
    plt.plot(x, test_loss_list, color="limegreen", label="Test")
    plt.xlabel("Epoch No.")
    plt.ylabel("Loss value.")
    plt.title(f"Train & validation loss curves in Fold#{n_fold + 1}")
    plt.legend()
    plt.grid()
    plt.show()
    if save_fig:
        fig.savefig(osp.join(save_path, "train_validation_loss_curves.png"))
        print("Saved fig.")


def plot_scatter(
    y_true,
    y_pred,
    train=True,
    exp_name="",
    save_fig=False,
    save_path=None,
):

    bhq_mean = df_subinfo_gm["gm_bhq"].mean()
    bhq_std = df_subinfo_gm["gm_bhq"].std()
    xlabel = "Gray Matter BHQ"

    corr, p = pearsonr(y_true, y_pred)
    mae = MAE(y_pred, y_true) * bhq_std
    if train:
        title = exp_name + " " + "scatter train"
    else:
        title = exp_name + " " + "scatter test"
    new_line = "\n"
    fontsize = 14
    scatter_label = f"corrcoef : {corr:.3f}{new_line}p_value : {p}{new_line}MAE : {mae}"
    fig = plt.figure(figsize=(6, 6), dpi=80)
    plt.scatter(
        y_true * bhq_std + bhq_mean,
        y_pred * bhq_std + bhq_mean,
        label=scatter_label,
    )
    if not train:  # testの際は回帰直線を引く
        linear_reg = LinearRegression()
        linear_reg.fit(
            y_true.reshape(-1, 1) * bhq_std + bhq_mean, y_pred * bhq_std + bhq_mean
        )
        plt.plot(
            y_true * bhq_std + bhq_mean,
            linear_reg.predict(y_true.reshape(-1, 1) * bhq_std + bhq_mean),
            color="red",
            label=f"y = {linear_reg.coef_[0]:.3f}x + {linear_reg.intercept_:.3f}",
        )
        print("intercept : ", linear_reg.intercept_)
        print("coef : ", linear_reg.coef_[0])
    plt.title(title, fontsize=fontsize)
    plt.xlabel(xlabel=xlabel, fontsize=fontsize)
    plt.ylabel("FC-based BHQ", fontsize=fontsize)
    plt.legend(fontsize=10.5)
    plt.grid()
    plt.show()
    if save_fig:
        if train:
            fig.savefig(osp.join(save_path, "scatter_train.png"))
        else:
            fig.savefig(osp.join(save_path, "scatter_test.png"))


def plot_cluster_AAL(model, weight_path, model_number, shrink=0.2):
    """
    学習済みモデルの第一RaGConv層でクラスタリングを行っている部分の重みを抽出.
    使ったatlasに基づいてどのROIがどのクラスタに所属しているかの結果を表示する.
    クラスタごとに色を変えたいけどどうやってやったらいいかわからないのでいったん放置.

    Args:
        model (_type_): 使ったモデル.
        weight_path (_type_): 保存した学習済みモデルの重み
        atlas_name (str, optional): atlasの名前. Defaults to "AAL".
    """
    # AAL atlasのロード
    atlas_data = fetch_atlas_aal()
    atlas_filename = atlas_data.maps
    labels = atlas_data.labels
    # 重みのロードと抽出
    weights = torch.load(weight_path)
    nn1_weight = model.state_dict(weights)["module.n1.0.weight"].detach().cpu().numpy()
    nn1_weight = np.array(nn1_weight)
    nn1_weight_th = nn1_weight * np.where(
        nn1_weight > 0, 1, 0
    )  # ReLU, calc non-negative score
    # plotting results of soft clustering.
    fig = plt.figure(figsize=(12, 12), dpi=80)
    plt.imshow(nn1_weight_th, cmap="binary")
    plt.colorbar(shrink=shrink)
    plt.xlabel("ROI defined in AAL atlas")
    plt.ylabel("Community")
    plt.title(f"Visualization of membership scores(Model {model_number})")
    plt.show()
    # ハードクラスタリング
    results = np.argmax(nn1_weight, axis=0)
    # クラスタのユニークな番号リスト
    clstr_list = np.unique(results)
    cmap = ListedColormap(["salmon"])
    subnet_dict = {}
    for i in clstr_list:
        print("=======================================================")
        print(f"This is cluster #{i + 1}")
        rois_in_clstr_i = np.where(results == i)[0]
        roilabels_in_clstr_i = [labels[int(j)] for j in rois_in_clstr_i]
        roiidx_in_clstr_i = [
            atlas_data.indices[atlas_data.labels.index(labels[int(j)])]
            for j in rois_in_clstr_i
        ]
        # クラスタ内のROIの画像を抽出
        img_list = [
            image.math_img("img == %s" % j, img=atlas_data.maps)
            for j in roiidx_in_clstr_i
        ]
        # 足し合わせてクラスタの画像とする
        img_clstr_i = image.math_img(
            "np.sum(imgs, axis = -1, dtype = 'float')", imgs=img_list
        )
        display = plotting.plot_roi(
            img_clstr_i,
            title=f"cluster {i + 1}(Model {model_number})",
            draw_cross=False,
        )
        plotting.show()
        print(f"ROIs in cluster {i + 1} is listed below.")
        print(roilabels_in_clstr_i)
        print("=======================================================")
        subnet_dict[f"cluster_{i + 1}"] = roilabels_in_clstr_i
    return subnet_dict


def cluster_separation(fimage, clustername, save_path, minimumsize=100):
    if os.path.isfile(fimage) == True:
        func_image = image.load_img(fimage)
        fdata = func_image.get_fdata()
        fdata2 = np.copy(fdata)
        fdata[fdata != 0] = 1
        clusdata = me.label(fdata)
        clusindex = np.argwhere(clusdata != 0)
        clusindexlist = clusindex.tolist()
        cluslabel = [
            clusdata[clusindexlist[i][0], clusindexlist[i][1], clusindexlist[i][2]]
            for i in range(len(clusindexlist))
        ]
        clusnumbering = list(set(cluslabel))
        k = 0
        for j in range(len(clusnumbering)):
            eachclusdata = clusdata - np.zeros(clusdata.shape)
            eachclusdata[eachclusdata != clusnumbering[j]] = 0
            eachclusdata[eachclusdata == clusnumbering[j]] = 1
            appliedimagedata = np.multiply(fdata2, eachclusdata)
            eachclusimage = image.new_img_like(func_image, appliedimagedata)
            if np.count_nonzero(eachclusdata) > minimumsize:
                eachclusimage.to_filename(
                    osp.join(save_path, clustername + "_clus_" + str(k + 1) + ".nii.gz")
                )
                # plotting.plot_glass_brain(eachclusimage, title='clus_' + str(k + 1),colorbar=True,display_mode='ortho')
                plotting.plot_roi(
                    eachclusimage,
                    title="clus_" + str(k + 1),
                    display_mode="ortho",
                    draw_cross=False,
                )
                plotting.show()
                k = k + 1
            del eachclusdata
