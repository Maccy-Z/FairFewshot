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
            accs.append(self.get_acc(xs_target, ys_target))

        return np.mean(accs)

    @abstractmethod
    def fit(self, xs_meta, ys_meta):
        pass

    @abstractmethod
    def get_acc(self, xs_target, ys_target):
        pass


class TabnetModel(Model):
    def __init__(self):
        self.model = TabNetClassifier(device_name="cpu")

    def fit(self, xs_meta, ys_meta):
        ys_meta = ys_meta[0].flatten().numpy()
        xs_meta = xs_meta[0].numpy()

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
                                   eval_set=[(xs_meta, ys_meta)], eval_name=["accuracy"],
                                   batch_size=8, patience=10, drop_last=False)
                except RuntimeError:
                    # Tabnet fails if multiple columns are exactly identical. Add a irrelevant amount of random noise to stop this.
                    xs_meta += np.random.normal(size=xs_meta.shape) * 1e-6
                    print(xs_meta)
                    self.model.fit(xs_meta, ys_meta,
                                   eval_set=[(xs_meta, ys_meta)], eval_name=["accuracy"],
                                   batch_size=8, patience=10, drop_last=False)

            sys.stdout = sys.__stdout__

            self.pred_val = None

    def get_acc(self, xs_target, ys_target):
        if self.identical_batch:
            predictions = np.ones_like(ys_target) * self.pred_val
        else:
            xs_target = xs_target.detach()[0]
            with torch.no_grad():
                predictions = self.model.predict(X=xs_target)

        ys_lr_target_labels = np.array(predictions).flatten()
        accuracy = (ys_lr_target_labels == np.array(ys_target)).sum().item() / len(ys_target)

        return accuracy

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
            dim=16,  # dimension, paper set at 32
            dim_out=2,  # binary prediction, but could be anything
            depth=4,  # depth, paper recommended 6
            heads=2,  # heads, paper recommends 8
            attn_dropout=0.1,  # post-attention dropout
            # ff_dropout=0.1  # feed forward dropout
        )

        optim = torch.optim.Adam(self.model.parameters(), lr=2e-3)

        for _ in range(30):
            x_categ = torch.tensor([[]])
            clf = self.model(x_categ, xs_meta)

            loss = torch.nn.functional.cross_entropy(clf, ys_meta.squeeze())
            loss.backward()
            optim.step()
            optim.zero_grad()

    def get_acc(self, xs_target, ys_target):
        self.model.eval()
        with torch.no_grad():
            target_preds = self.model(self.null_categ, xs_target)
        preds = torch.argmax(target_preds, dim=1)
        accuracy = (preds == ys_target).sum().item() / len(ys_target)

        return accuracy

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
                self.model = KNN(n_neighbors=2, p=1, weights="distance")
            case "CatBoost":
                self.model = CatBoostClassifier(iterations=6, depth=4, learning_rate=1,
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
            # xs_target = xs_target[0]
            predictions = self.model.predict(xs_target)

        ys_lr_target_labels = np.array(predictions).flatten()
        accuracy = (ys_lr_target_labels == np.array(ys_target)).sum().item() / len(ys_target)

        return accuracy

    def __repr__(self):
        return self.name


class FLAT(Model):
    def __init__(self, save_dir):
        print(f'Loading model at {save_dir = }')

        state_dict = torch.load(f'{save_dir}/model.pt')
        self.model = ModelHolder(cfg_all=get_config(cfg_file=f'{save_dir}/defaults.toml'))
        self.model.load_state_dict(state_dict['model_state_dict'])

    def fit(self, xs_meta, ys_meta):
        xs_meta, ys_meta = xs_meta.unsqueeze(0), ys_meta.unsqueeze(0)
        pairs_meta = d2v_pairer(xs_meta, ys_meta)
        with torch.no_grad():
            self.embed_meta, self.pos_enc = self.model.forward_meta(pairs_meta)

    def get_acc(self, xs_target, ys_target):
        xs_target = xs_target.unsqueeze(0)
        with torch.no_grad():
            ys_pred_target = self.model.forward_target(xs_target, self.embed_meta, self.pos_enc)

        ys_pred_target_labels = torch.argmax(ys_pred_target.view(-1, 2), dim=1)
        accuracy = (ys_pred_target_labels == ys_target).sum().item() / len(ys_target)

        return accuracy

    def __repr__(self):
        return "FLAT"


def get_results_by_dataset(test_data_names, models, num_rows=10, num_targets=5, num_samples=3, by_dataset=True, by_cols=True):
    """
    Evaluates the model and baseline_models on the test data sets.
    Results are groupped by: data set, model, number of test columns.
    """

    datasets = [
        MyDataSet(d, num_rows=0, num_targets=0, binarise=True, split="all")
        for d in test_data_names
    ]
    n_cols = [d.ds_cols - 1 for d in datasets]
    max_test_col = max([d.ds_cols - 1 for d in datasets])
    n_cols = dict(zip(test_data_names, n_cols))

    results = pd.DataFrame(columns=['data_name', 'model', 'num_cols', 'acc'])

    if by_cols:
        num_cols = 2
        while num_cols <= max_test_col:
            if not by_dataset:
                test_dl = SplitDataloader(
                    bs=num_samples, num_rows=num_rows, num_targets=num_targets,
                    num_cols=[num_cols - 1, num_cols], ds_group=test_data_names
                )
                batch = get_batch(test_dl, num_rows)
                for model in models:
                    result = pd.DataFrame({
                        'data_name': "all seen",
                        'model': str(model),
                        'num_cols': num_cols,
                        'acc': model.get_accuracy(batch)
                    }, index=[0])

                    results = pd.concat([results, result])
            else:
                for data_name in test_data_names:
                    if n_cols[data_name] >= num_cols:
                        test_dl = SplitDataloader(
                            bs=num_samples * len(test_data_names), num_rows=num_rows,
                            num_targets=num_targets, num_cols=[num_cols - 1, num_cols], 
                            ds_group=data_name
                        )
                        batch = get_batch(test_dl, num_rows)
                        for model in models:
                            acc = model.get_accuracy(batch)
                            result = pd.DataFrame({
                                'data_name': data_name,
                                'model': str(model),
                                'num_cols': num_cols,
                                'acc': acc
                            }, index=[0])

                            results = pd.concat([results, result])
            num_cols *= 2

    # Test on full dataset
    elif not by_cols and by_dataset:
        for data_name in test_data_names:
            test_dl = SplitDataloader(
                bs=num_samples, num_rows=num_rows,
                num_targets=num_targets, num_cols=[n_cols[data_name], n_cols[data_name]],
                ds_group=data_name
            )
            batch = get_batch(test_dl, num_rows)
            for model in models:
                result = pd.DataFrame({
                    'data_name': data_name,
                    'model': str(model),
                    'num_cols': num_cols,
                    'acc': model.get_accuracy(batch)
                }, index=[0])

                results = pd.concat([results, result])
    
    else:
        Exception("At least one of by_cols or by_dataset must be True")

    results.reset_index(drop=True, inplace=True)

    return results


def main(save_no):
    BASEDIR = '.'
    dir_path = f'{BASEDIR}/saves'
    # files = [f for f in os.listdir(dir_path) if os.path.isdir(f'{dir_path}/{f}')]
    # existing_saves = sorted([int(f[5:]) for f in files if f.startswith("save")])  # format: save_{number}
    # save_no = existing_saves[save_no]
    save_dir = f'{BASEDIR}/saves/save_{save_no}'

    all_cfg = toml.load(os.path.join(save_dir, 'defaults.toml'))
    cfg = all_cfg["DL_params"]
    ds = all_cfg["Settings"]["dataset"]
    ds_group = cfg["ds_group"]

    if ds == "my_split":
        split_file = cfg["split_file"]
        split_file = f'./datasets/grouped_datasets/{split_file}'
        with open(split_file) as f:
            split = toml.load(f)
        train_data_names = split[str(ds_group)]["train"]
        test_data_names = split[str(ds_group)]["test"]

        print("Train datases:", train_data_names)
        print("Test datasets:", test_data_names)

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

    else:
        raise Exception("Invalid data split")

    num_rows = cfg["num_rows"]
    num_targets = cfg["num_targets"]
    num_samples = 50

    models = [FLAT(save_dir),
              BasicModel("LR"), BasicModel("KNN"),  # , BasicModel("R_Forest"),  BasicModel("CatBoost"),
              # TabnetModel(),
              # FTTrModel(),
              ]

    unseen_results = get_results_by_dataset(
        test_data_names, models,
        num_rows=num_rows, num_targets=num_targets,
        num_samples=num_samples, by_cols=True, by_dataset=True
    )
    print("======================================================")
    print("Test accuracy on unseen datasets (by dataset)")
    unseen_print = unseen_results.pivot(
        columns=['data_name', 'model'], index='num_cols', values='acc')
    print((unseen_print * 100).round(2).to_string())
    unseen_results.to_csv(f'{BASEDIR}/saves/save_{save_no}/unseen_results.csv')
    
    print("======================================================")
    print("Test accuracy on unseen datasets (aggregated)")
    df = unseen_results.groupby(['num_cols', 'model'])['acc'].mean().unstack()
    df["FLAT_diff"] = df["FLAT"] - df.iloc[:, 1:].max(axis=1)
    print((df * 100).round(2).to_string())

    seen_results = get_results_by_dataset(
        train_data_names, models,
        num_rows=num_rows, num_targets=num_targets,
        num_samples=num_samples, by_cols=True, by_dataset=False
    )
    print()
    print("======================================================")
    print("Test accuracy on seen datasets (aggregated)")
    df = seen_results.pivot(columns='model', index='num_cols', values='acc')
    df["FLAT_diff"] = df["FLAT"] - df.iloc[:, 1:].max(axis=1)
    print((df * 100).round(2).to_string())
    seen_results.to_csv(f'{BASEDIR}/saves/save_{save_no}/seen_results.csv')



if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    # save_number = int(input("Enter save number:\n"))
    main(save_no=19)

    #col_accs = main(save_no=2)

    # print(col_accs)
