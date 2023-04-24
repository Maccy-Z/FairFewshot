import torch
from main import *
from dataloader import d2v_pairer
from AllDataloader import SplitDataloader
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
    ys_target = ys_target.view(-1)

    return xs_meta, xs_target, ys_meta, ys_target

class Model(ABC):
    def get_accuracy(self, batch):
        xs_meta, xs_target, ys_meta, ys_target = batch

        self.fit(xs_meta, ys_meta)

        return self.get_acc(xs_target, ys_target)

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
        ys_meta = ys_meta[0].flatten()
        xs_meta = xs_meta[0]
        # Reset the model
        self.model = FTTransformer(
            categories=(),  # tuple containing the number of unique values within each category
            num_continuous=xs_meta.shape[-1],  # number of continuous values
            dim=16,  # dimension, paper set at 32
            dim_out=2,  # binary prediction, but could be anything
            depth=4,  # depth, paper recommended 6
            heads=2,  # heads, paper recommends 8
            # attn_dropout=0.1,  # post-attention dropout
            # ff_dropout=0.1  # feed forward dropout
        )

        optim = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        for _ in range(50):
            x_categ = torch.tensor([[]])
            clf = self.model(x_categ, xs_meta)

            loss = torch.nn.functional.cross_entropy(clf, ys_meta.squeeze())
            loss.backward()
            optim.step()
            optim.zero_grad()


    def get_acc(self, xs_target, ys_target):
        xs_target = xs_target[0]

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
                           loss_function='Logloss',allow_const_label=True, verbose=False)
            case "R_Forest":
                self.model = RandomForestClassifier(n_estimators=30)
            case _:
                raise Exception("Invalid model specified")

        self.name = name
        self.identical_batch = False

    def fit(self, xs_meta, ys_meta):
        ys_meta = ys_meta[0].flatten().numpy()
        xs_meta = xs_meta[0].numpy()

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
            xs_target = xs_target[0]
            predictions = self.model.predict(xs_target)

        ys_lr_target_labels = np.array(predictions).flatten()
        accuracy = (ys_lr_target_labels == np.array(ys_target)).sum().item() / len(ys_target)

        return accuracy

    def __repr__(self):
        return self.name


class Fewshot(Model):
    def __init__(self, save_dir):
        print(f'Loading model at {save_dir = }')

        state_dict = torch.load(f'{save_dir}/model.pt')
        self.model = ModelHolder(cfg_all=get_config(cfg_file=f'{save_dir}/defaults.toml'))
        self.model.load_state_dict(state_dict['model_state_dict'])

    def fit(self, xs_meta, ys_meta):
        pairs_meta = d2v_pairer(xs_meta, ys_meta)
        with torch.no_grad():
            self.embed_meta, self.pos_enc = self.model.forward_meta(pairs_meta)

    def get_acc(self, xs_target, ys_target):
        with torch.no_grad():
            ys_pred_target = self.model.forward_target(xs_target, self.embed_meta, self.pos_enc)

        ys_pred_target_labels = torch.argmax(ys_pred_target.view(-1, 2), dim=1)
        accuracy = (ys_pred_target_labels == ys_target).sum().item() / len(ys_target)

        return accuracy

    def __repr__(self):
        return "Fewshot"


def get_results_by_dataset(test_data_names, models, num_rows=10, agg=False):
    """
    Evaluates the model and baseline_models on the test data sets.
    Results are groupped by: data set, model, number of test columns.
    """

    results = pd.DataFrame(columns=['data_name', 'model', 'num_cols', 'acc'])

    for data_name in test_data_names:
        save_name = "SEEN" if agg else data_name

        for num_cols in range(1, 20, 4):

            # get batch
            test_dl = SplitDataloader(bs=1, num_rows=num_rows, num_targets=5,
                                 num_cols=num_cols, get_ds=data_name, split="test")

            batch = get_batch(test_dl, num_rows)

            for model in models:
                result = pd.DataFrame({
                                        'data_name': save_name,
                                        'model': str(model),
                                        'num_cols': num_cols,
                                        'acc': model.get_accuracy(batch)
                                        }, index=[0])

                results = pd.concat([results, result])

    results.reset_index(drop=True, inplace=True)
    return results


def main(save_no):
    BASEDIR = '.'
    dir_path = f'{BASEDIR}/saves'
    files = [f for f in os.listdir(dir_path) if os.path.isdir(f'{dir_path}/{f}')]
    existing_saves = sorted([int(f[5:]) for f in files if f.startswith("save")])  # format: save_{number}
    save_no = existing_saves[save_no]
    save_dir = f'{BASEDIR}/saves/save_{save_no}'

    cfg = toml.load(os.path.join(save_dir, 'defaults.toml'))["DL_params"]

    num_rows = 10  # cfg["num_rows"]

    models = [#Fewshot(save_dir),
              BasicModel("LR"), BasicModel("CatBoost"), BasicModel("KNN"),
              #TabnetModel(),
              #FTTrModel(),
              #BasicModel("R_Forest"),
              ]

    unseen_results = get_results_by_dataset(cfg["test_data_names"], models, num_rows=num_rows)
    print(unseen_results.pivot(columns=['data_name', 'model'], index='num_cols', values='acc'))

    seen_results = get_results_by_dataset([cfg["train_data_names"]], models, num_rows=num_rows, agg=True)

    print()
    print()
    print(seen_results.pivot(columns='model', index='num_cols', values='acc'))


    return unseen_results



if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    # save_number = int(input("Enter save number:\n"))
    # main(save_no=save_number)

    col_accs = main(save_no=-2)

    # print(col_accs)
