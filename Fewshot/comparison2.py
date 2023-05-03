import copy

import torch
from main import *
from dataloader import d2v_pairer
from AllDataloader import SplitDataloader, MyDataSet
from config import get_config

import os
import toml
import numpy as np
import random
import sys, io, warnings
from scipy import stats
from abc import ABC, abstractmethod
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import RandomForestClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
from catboost import CatBoostClassifier, CatboostError
from tab_transformer_pytorch import FTTransformer

# import sys
# sys.path.append('/mnt/storage_ssd/FairFewshot/STUNT_main')
# from STUNT_interface import STUNT_utils, MLPProto

def get_batch(dl, num_rows):
    xs, ys, model_id = next(iter(dl))
    xs_meta, xs_target = xs[:, :num_rows], xs[:, num_rows:]
    ys_meta, ys_target = ys[:, :num_rows], ys[:, num_rows:]
    xs_meta, xs_target = xs_meta.contiguous(), xs_target.contiguous()
    ys_meta, ys_target = ys_meta.contiguous(), ys_target.contiguous()
    # ys_target = ys_target.view(-1)
    return xs_meta, xs_target, ys_meta, ys_target


class Model(ABC):
    # Process batch of data
    def get_accuracy(self, batch):
        xs_metas, xs_targets, ys_metas, ys_targets = batch
        accs = []
        for xs_meta, xs_target, ys_meta, ys_target in zip(xs_metas, xs_targets, ys_metas, ys_targets):
            # print(xs_meta.shape)
            self.fit(xs_meta, ys_meta)
            a = self.get_acc(xs_target, ys_target)

            accs.append(a)


        accs = np.concatenate(accs)

        mean, std = np.mean(accs), np.std(accs, ddof=1) / np.sqrt(accs.shape[0])

        return mean, std

    @abstractmethod
    def fit(self, xs_meta, ys_meta):
        pass

    @abstractmethod
    def get_acc(self, xs_target, ys_target) -> np.array:
        pass


# class STUNT(STUNT_utils, Model):
#     model: torch.nn.Module

#     def __init__(self):
#         self.lr = 0.0001
#         self.model_size = (256, 256) # num_cols, out_dim, hid_dim
#         self.steps =0
#         self.shot = 4
#         self.tasks_per_batch = 4
#         self.test_num_way = 2
#         self.query = 1
#         self.kmeans_iter = 5


#     def fit(self, xs_meta, ys_meta):
#         ys_meta = ys_meta.flatten()
#         # Reset the model
#         self.model = MLPProto(xs_meta.shape[-1], self.model_size[0], self.model_size[1])
#         self.optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)
#         with warnings.catch_warnings():
#             # warnings.simplefilter("ignore")
#             for _ in range(self.steps):
#                 try:
#                     train_batch = self.get_batch(xs_meta.clone())
#                     self.protonet_step(train_batch)
#                 except AttributeError as e:
#                     pass

#         with torch.no_grad():
#             meta_embed = self.model(xs_meta)
#         self.prototypes = self.get_prototypes(meta_embed.unsqueeze(0), ys_meta.unsqueeze(0), 2)


#     def get_acc(self, xs_target, ys_target):
#         self.model.eval()
#         with torch.no_grad():
#             support_target = self.model(xs_target)

#         self.prototypes = self.prototypes[0]
#         support_target = support_target.unsqueeze(1)


#         sq_distances = torch.sum((self.prototypes
#                                   - support_target) ** 2, dim=-1)

#         # print(sq_distances.shape)
#         _, preds = torch.min(sq_distances, dim=-1)

#         # print(preds.numpy(), ys_target.numpy())
#         return (preds == ys_target).numpy()


class TabnetModel(Model):
    def __init__(self):
        self.model = TabNetClassifier(device_name="cpu")
        self.bs = 64
        self.patience = 17

    def fit(self, xs_meta, ys_meta):
        ys_meta = ys_meta.flatten().numpy()
        xs_meta = xs_meta.numpy()

        if ys_meta.min() == ys_meta.max():
            self.identical_batch = True
            self.pred_val = ys_meta[0]
        else:
            self.identical_batch = False

            sys.stdout = open(os.devnull, "w")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                try:
                    self.model.fit(xs_meta, ys_meta,
                                   eval_name=["accuracy"], eval_set=[(xs_meta, ys_meta)],
                                   batch_size=self.bs, patience=self.patience, drop_last=False)
                except RuntimeError:
                    # Tabnet fails if multiple columns are exactly identical. Add a irrelevant amount of random noise to stop this.
                    xs_meta += np.random.normal(size=xs_meta.shape) * 1e-6
                    print(xs_meta)
                    self.model.fit(xs_meta, ys_meta,
                                   eval_set=[(xs_meta, ys_meta)], eval_name=["accuracy"],
                                   batch_size=self.bs, patience=self.patience, drop_last=False)

            sys.stdout = sys.__stdout__

            self.pred_val = None

    def get_acc(self, xs_target, ys_target):
        if self.identical_batch:
            predictions = np.ones_like(ys_target) * self.pred_val
        else:
            with torch.no_grad():
                predictions = self.model.predict(X=xs_target)

        ys_lr_target_labels = np.array(predictions).flatten()

        return ys_lr_target_labels == np.array(ys_target)

    def __repr__(self):
        return "TabNet"


class FTTrModel(Model):
    model: torch.nn.Module

    def __init__(self):
        self.null_categ = torch.tensor([[]])

    def fit(self, xs_meta, ys_meta):
        ys_meta = ys_meta.flatten()
        xs_meta = xs_meta            
        # Reset the model
        self.model = FTTransformer(
            categories=(),  # tuple containing the number of unique values within each category
            num_continuous=xs_meta.shape[-1],  # number of continuous values
            dim=24,  # dimension, paper set at 32
            dim_out=2,  # binary prediction, but could be anything
            depth=4,  # depth, paper recommended 6
            heads=2,  # heads, paper recommends 8
            attn_dropout=0.1,  # post-attention dropout
            # ff_dropout=0.1  # feed forward dropout
        )

        optim = torch.optim.Adam(self.model.parameters(), lr=2.25e-3)

        for _ in range(30):
            x_categ = torch.tensor([[]])
            clf = self.model(x_categ, xs_meta)
            loss = torch.nn.functional.cross_entropy(clf, ys_meta)
            loss.backward()
            optim.step()
            optim.zero_grad()

    def get_acc(self, xs_target, ys_target):
        self.model.eval()
        with torch.no_grad():
            target_preds = self.model(self.null_categ, xs_target)
        preds = torch.argmax(target_preds, dim=1)

        return (preds == ys_target).numpy()

    def __repr__(self):
        return "FTTransformer"


class BasicModel(Model):
    def __init__(self, name):
        match name:
            case "LR":
                self.model = LogisticRegression(max_iter=1000)
            case "SVC":
                self.model = SVC()
            case "KNN":
                self.model = KNN(n_neighbors=3, p=1, weights="distance")
            case "CatBoost":
                self.model = CatBoostClassifier(iterations=20, depth=4, learning_rate=0.5,
                                                loss_function='Logloss', allow_const_label=True, verbose=False)
            case "R_Forest":
                self.model = RandomForestClassifier(n_estimators=30)
            case _:
                raise Exception("Invalid model specified")

        self.name = name
        self.identical_batch = False

    def fit(self, xs_meta, ys_meta):
        ys_meta = ys_meta.flatten().numpy()
        xs_meta = xs_meta.numpy()

        if ys_meta.min() == ys_meta.max():
            self.identical_batch = True
            self.pred_val = ys_meta[0]
        else:
            self.identical_batch = False

            try:
                self.model.fit(xs_meta, ys_meta)
            except CatboostError:
                # Catboost fails if every input element is the same
                self.identical_batch = True
                mode = stats.mode(ys_meta, keepdims=False)[0]
                self.pred_val = mode

    def get_acc(self, xs_target, ys_target):
        xs_target = xs_target.numpy()
        if self.identical_batch:
            predictions = np.ones_like(ys_target) * self.pred_val
        else:
            predictions = self.model.predict(xs_target)

        return np.array(predictions).flatten() == np.array(ys_target)

    def __repr__(self):
        return self.name


class FLAT(Model):
    def __init__(self, save_dir, save_ep=None):
        print(f'Loading model at {save_dir = }')

        if save_ep is None:
            state_dict = torch.load(f'{save_dir}/model.pt')
        else:
            state_dict = torch.load(f'{save_dir}/model_{save_ep}.pt')
        self.model = ModelHolder(cfg_all=get_config(cfg_file=f'{save_dir}/defaults.toml'))
        self.model.load_state_dict(state_dict['model_state_dict'])

    def fit(self, xs_meta, ys_meta):
        xs_meta, ys_meta = xs_meta.unsqueeze(0), ys_meta.unsqueeze(0)
        pairs_meta = d2v_pairer(xs_meta, ys_meta)
        with torch.no_grad():
            self.embed_meta, self.pos_enc = self.model.forward_meta(pairs_meta)

    def get_acc(self, xs_target, ys_target) -> np.array:
        xs_target = xs_target.unsqueeze(0)
        with torch.no_grad():
            ys_pred_target = self.model.forward_target(xs_target, self.embed_meta, self.pos_enc)

        ys_pred_target_labels = torch.argmax(ys_pred_target.view(-1, 2), dim=1)

        return (ys_pred_target_labels == ys_target).numpy()

    def __repr__(self):
        return "FLAT"

class FLAT_MAML(Model):
    def __init__(self, save_dir, save_ep=None):
        print(f'Loading model at {save_dir = }')

        if save_ep is None:
            state_dict = torch.load(f'{save_dir}/model.pt')
        else:
            state_dict = torch.load(f'{save_dir}/model_{save_ep}.pt')
        self.model = ModelHolder(cfg_all=get_config(cfg_file=f'{save_dir}/defaults.toml'))
        self.model.load_state_dict(state_dict['model_state_dict'])

    def fit(self, xs_meta, ys_meta):
        xs_meta, ys_meta = xs_meta.unsqueeze(0), ys_meta.unsqueeze(0)
        pairs_meta = d2v_pairer(xs_meta, ys_meta)
        with torch.no_grad():
            embed_meta, pos_enc = self.model.forward_meta(pairs_meta)

        embed_meta.requires_grad = True
        pos_enc.requires_grad = True
        #optim_pos = torch.optim.Adam([pos_enc], lr=0.005)
        # optim_embed = torch.optim.SGD([embed_meta, ], lr=50, momentum=0.75)  # torch.optim.Adam([embed_meta], lr=0.01)  #
        optim_embed = torch.optim.Adam([embed_meta], lr=0.05)
        for _ in range(5):
            # Make predictions on meta set and calc loss
            preds = self.model.forward_target(xs_meta, embed_meta, pos_enc)
            loss = torch.nn.functional.cross_entropy(preds.squeeze(), ys_meta.squeeze())
            loss.backward()
            #optim_pos.step()
            optim_embed.step()
            optim_embed.zero_grad()
            #optim_pos.zero_grad()

        self.embed_meta = embed_meta
        self.pos_enc = pos_enc

        # print(self.embed_meta)

    def get_acc(self, xs_target, ys_target) -> np.array:
        xs_target = xs_target.unsqueeze(0)
        with torch.no_grad():
            ys_pred_target = self.model.forward_target(xs_target, self.embed_meta, self.pos_enc)

        ys_pred_target_labels = torch.argmax(ys_pred_target.view(-1, 2), dim=1)

        return (ys_pred_target_labels == ys_target).numpy()

    def __repr__(self):
        return "FLAT_MAML"
    
def get_results_by_dataset(test_data_names, models, num_rows=10, num_targets=5, num_samples=3, agg=False):
    """
    Evaluates the model and baseline_models on the test data sets.
    Results are groupped by: data set, model, number of test columns.
    """

    datasets = [
        MyDataSet(d, num_rows=5, num_targets=5, binarise=True, split="all")
        for d in test_data_names
    ]

    n_cols = [d.ds_cols - 1 for d in datasets]
    max_test_col = max([d.ds_cols - 1 for d in datasets])
    n_cols = dict(zip(test_data_names, n_cols))

    results = pd.DataFrame(columns=['data_name', 'model', 'num_cols', 'acc'])
    num_cols = 2
    # while num_cols <= max_test_col and num_cols <= 60:
    #     print(num_cols)
    #     if agg:
    #         test_dl = SplitDataloader(
    #             bs=num_samples * len(test_data_names), num_rows=num_rows, num_targets=num_targets,
    #             num_cols=[num_cols - 1, num_cols], ds_group=test_data_names
    #         )
    #         batch = get_batch(test_dl, num_rows)
    #         for model in models:
    #             mean_acc, std_acc = model.get_accuracy(batch)
    #             result = pd.DataFrame({
    #                 'data_name': "all seen",
    #                 'model': str(model),
    #                 'num_cols': num_cols,
    #                 'acc': mean_acc,
    #                 'std': std_acc
    #             }, index=[0])
    #
    #             results = pd.concat([results, result])
    #     else:
    #         for data_name in test_data_names:
    #             if n_cols[data_name] >= num_cols:
    #                 test_dl = SplitDataloader(
    #                     bs=num_samples, num_rows=num_rows,
    #                     num_targets=num_targets, num_cols=[num_cols, num_cols],
    #                     ds_group=data_name
    #                 )
    #                 batch = get_batch(test_dl, num_rows)
    #                 for model in models:
    #                     mean_acc, std_acc = model.get_accuracy(batch)
    #
    #                     result = pd.DataFrame({
    #                         'data_name': data_name,
    #                         'model': str(model),
    #                         'num_cols': num_cols,
    #                         'acc': mean_acc,
    #                         'std': std_acc
    #                     }, index=[0])
    #
    #                     results = pd.concat([results, result])
    #     num_cols *= 2

    # Test on full dataset
    if not agg:
        for data_name in test_data_names:
            test_dl = SplitDataloader(
                bs=num_samples, num_rows=num_rows,
                num_targets=num_targets, num_cols=[n_cols[data_name], n_cols[data_name]],
                ds_group=data_name
            )
            batch = get_batch(test_dl, num_rows)
            for model in models:
                print(data_name, str(model))
                mean_acc, std_acc = model.get_accuracy(batch)
                result = pd.DataFrame({
                    'data_name': data_name,
                    'model': str(model),
                    'num_cols': -1,
                    'acc': mean_acc,
                    'std': std_acc
                }, index=[0])

                results = pd.concat([results, result])

    results.reset_index(drop=True, inplace=True)
    return results


def main(save_no, num_rows, save_ep, dir_path=f'./saves'):
    files = [f for f in os.listdir(dir_path) if os.path.isdir(f'{dir_path}/{f}')]
    existing_saves = sorted([int(f[5:]) for f in files if f.startswith("save")])  # format: save_{number}
    save_no = existing_saves[save_no]
    save_dir = f'{dir_path}/save_{save_no}'

    all_cfg = toml.load(os.path.join(save_dir, 'defaults.toml'))
    cfg = all_cfg["DL_params"]
    ds = all_cfg["Settings"]["dataset"]
    ds_group = cfg["ds_group"]

    if ds == "my_split":
        split_file = f"./datasets/grouped_datasets/{cfg['split_file']}"
        with open(split_file) as f:
            split = toml.load(f)
        train_data_names = split[str(ds_group)]["train"]
        test_data_names = split[str(ds_group)]["test"]

    elif ds == "total":
        fold_no, split_no = ds_group
        splits = toml.load(f'./datasets/grouped_datasets/splits_{fold_no}')
        if split_no == -1:
            get_splits = range(6)
        else:
            get_splits = [split_no]

        test_data_names = []
        for split in get_splits:
            ds_name = splits[str(split)]["test"]
            test_data_names += ds_name

        train_data_names = []
        for split in get_splits:
            ds_name = splits[str(split)]["train"]
            train_data_names += ds_name

    elif ds == "custom":
        test_data_names = []
        train_data_names = []

    else:
        raise Exception("Invalid data split")
    
    print("Train datases:", train_data_names)
    print("Test datasets:", test_data_names)

    num_targets = cfg["num_targets"]
    num_samples = 1000 // num_targets

    models = [
        FLAT_MAML(save_dir, save_ep=save_ep),
        FLAT(save_dir, save_ep=save_ep),
        BasicModel("LR"), BasicModel("CatBoost"),  BasicModel("KNN"),# BasicModel("R_Forest"),  ,
        TabnetModel(),
        FTTrModel(),
        # STUNT(),
    ]

    unseen_results = get_results_by_dataset(
        test_data_names, models,
        num_rows=num_rows, num_targets=num_targets,
        num_samples=num_samples
    )
    unseen_results.to_csv(f'{save_dir}/unseen_results_{num_rows}_row.csv')
    detailed_results = unseen_results.copy()

    mean, std = detailed_results["acc"], detailed_results["std"]
    mean_std = [f'{m * 100:.2f}±{s * 100:.2f}' for m, s in zip(mean, std)]
    detailed_results['acc'] = mean_std

    detailed_results = detailed_results.pivot(
        columns=['data_name', 'model'], index='num_cols', values=['acc'])

    # Aggreate results
    agg_results = unseen_results.copy()

    # Move FLAT and FALT MAML to first 2 columns
    agg_results = agg_results.groupby(['num_cols', 'model'])['acc'].mean().unstack()
    new_column_order = ["FLAT", "FLAT_MAML"] + [col for col in agg_results.columns if (col != "FLAT" and col != "FLAT_MAML")]
    agg_results = agg_results.reindex(columns=new_column_order)
    
    # Difference between FLAT and the best baseline
    agg_results["FLAT_diff"] = (agg_results["FLAT"] - agg_results.iloc[:, 2:].max(axis=1)) * 100
    agg_results["FLAT_MAML_diff"] = (agg_results["FLAT_MAML"] - agg_results.iloc[:, 2:-1].max(axis=1)) * 100
    agg_results["FLAT_diff"] = agg_results["FLAT_diff"].apply(lambda x: f'{x:.2f}')
    agg_results["FLAT_MAML_diff"] = agg_results["FLAT_MAML_diff"].apply(lambda x: f'{x:.2f}')
    
    # Get errors using appropriate formulas.
    pivot_acc = unseen_results.pivot(
        columns=['data_name', 'model'], index='num_cols', values=['acc'])
    pivot_std = unseen_results.pivot(
        columns=['data_name', 'model'], index='num_cols', values=['std'])
    model_names = pivot_acc.columns.get_level_values(2).unique()
    for model_name in model_names:

        model_accs = pivot_acc.loc[:, ("acc", slice(None), model_name)]
        model_stds = pivot_std.loc[:, ("std", slice(None), model_name)]

        mean_stds = []
        for i in range(pivot_acc.shape[0]):
            accs = np.array(model_accs.iloc[i].dropna())
            std = np.array(model_stds.iloc[i].dropna())

            assert std.shape == accs.shape
            mean_acc = np.mean(accs)
            std_acc = np.sqrt(np.sum(std ** 2)) / std.shape[0]
            mean_std = f'{mean_acc * 100:.2f}±{std_acc * 100:.2f}'
            mean_stds.append(mean_std)

        agg_results[model_name] = mean_stds

    print(agg_results.to_string(index=False))


    return unseen_results


if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    num_test_rows = [5, 10]
    save_no_ls = [0, 1, 2]

    for ep in [None]:
        print("======================================================")
        print("Epoch number", ep)
        for i in save_no_ls:
            for j in num_test_rows:
                random.seed(0)
                np.random.seed(0)
                torch.manual_seed(0)
                print()
                print(i, j)
                main(save_no=i, num_rows=j, save_ep=ep)


