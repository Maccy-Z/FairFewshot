import torch
from main import *
from dataloader import d2v_pairer
from AllDataloader10 import SplitDataloader
from config import get_config
import time, os, toml, random, pickle, warnings
import numpy as np
from scipy import stats
from abc import ABC, abstractmethod
import pandas as pd
from collections import defaultdict

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import RandomForestClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
from catboost import CatBoostClassifier, CatboostError
from tab_transformer_pytorch import FTTransformer
from tabpfn import TabPFNClassifier
from ds_base import InfModel, ff_block

import sys

sys.path.append('/mnt/storage_ssd/fewshot_learning/FairFewshot/STUNT_main')
from STUNT_interface import STUNT_utils, MLPProto

BASEDIR = '.'


def load_batch(ds_name, num_rows):
    with open(f"./datasets/data/{ds_name}/batches/{num_rows}_5_-3", "rb") as f:
        batch = pickle.load(f)

    if batch is None:
        raise IndexError(f"Batch not found for file {ds_name}")
    return batch


def get_batch(dl, num_rows):
    xs, ys, model_id = next(iter(dl))
    xs_meta, xs_target = xs[:, :num_rows], xs[:, num_rows:]
    ys_meta, ys_target = ys[:, :num_rows], ys[:, num_rows:]
    xs_meta, xs_target = xs_meta.contiguous(), xs_target.contiguous()
    ys_meta, ys_target = ys_meta.contiguous(), ys_target.contiguous()
    return xs_meta, xs_target, ys_meta, ys_target


class Model(ABC):
    # Process batch of data
    def get_accuracy(self, batch):
        xs_metas, xs_targets, ys_metas, ys_targets = batch
        accs = []

        # n = 200
        # xs_metas, xs_targets = torch.rand([200, 10, n]), torch.rand([200, 5, n])
        # st = time.time()

        for xs_meta, xs_target, ys_meta, ys_target in zip(xs_metas, xs_targets, ys_metas, ys_targets):
            self.fit(xs_meta, ys_meta)
            a = self.get_acc(xs_target, ys_target)

            accs.append(a)

        # print(f'{time.time() - st :.3g}')
        # exit(6)
        accs = np.concatenate(accs)

        mean, std = np.mean(accs), np.std(accs, ddof=1) / np.sqrt(accs.shape[0])

        return mean, std

    @abstractmethod
    def fit(self, xs_meta, ys_meta):
        pass

    @abstractmethod
    def get_acc(self, xs_target, ys_target) -> np.array:
        pass


class STUNT(STUNT_utils, Model):
    model: torch.nn.Module

    def __init__(self):
        self.lr = 0.0001
        self.model_size = (1024, 1024)  # num_cols, out_dim, hid_dim
        self.steps = 5
        self.tasks_per_batch = 4
        self.test_num_way = 2
        self.query = 1
        self.kmeans_iter = 5

    def fit(self, xs_meta, ys_meta):
        self.shot = (xs_meta.shape[0] - 2) // 2
        ys_meta = ys_meta.flatten().long()

        # Reset the model
        self.model = MLPProto(xs_meta.shape[-1], self.model_size[0], self.model_size[1])
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        with warnings.catch_warnings():
            # warnings.simplefilter("ignore")
            for _ in range(self.steps):
                try:
                    train_batch = self.get_batch(xs_meta.clone())
                    self.protonet_step(train_batch)
                except NameError as e:
                    pass

        with torch.no_grad():
            meta_embed = self.model(xs_meta)

        self.prototypes = self.get_prototypes(meta_embed.unsqueeze(0), ys_meta.unsqueeze(0), 3)

    def get_acc(self, xs_target, ys_target):
        self.model.eval()
        with torch.no_grad():
            support_target = self.model(xs_target)

        self.prototypes = self.prototypes[0]
        support_target = support_target.unsqueeze(1)

        sq_distances = torch.sum((self.prototypes
                                  - support_target) ** 2, dim=-1)

        # print(sq_distances.shape)
        _, preds = torch.min(sq_distances, dim=-1)

        # print(preds.numpy(), ys_target.numpy())
        return (preds == ys_target).numpy()


class TabnetModel(Model):
    def __init__(self):
        self.model = TabNetClassifier(device_name="cpu")
        self.bs = 64
        self.patience = 17

    def fit(self, xs_meta, ys_meta):
        ys_meta = ys_meta.flatten().float().numpy()
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
                    # self.model.fit(xs_meta, ys_meta, drop_last=False)
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
        ys_meta = ys_meta.flatten().long()
        xs_meta = xs_meta
        # Reset the model
        self.model = FTTransformer(
            categories=(),  # tuple containing the number of unique values within each category
            num_continuous=xs_meta.shape[-1],  # number of continuous values
            dim=24,  # dimension, paper set at 32
            dim_out=3,  # binary prediction, but could be anything
            depth=4,  # depth, paper recommended 6
            heads=2,  # heads, paper recommends 8
            attn_dropout=0.1,  # post-attention dropout
            # ff_dropout=0.1  # feed forward dropout
        )

        optim = torch.optim.Adam(self.model.parameters(), lr=2.25e-3)

        for _ in range(70):
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

        return (preds == ys_target).numpy()

    def __repr__(self):
        return "FTTransformer"


class BasicModel(Model):
    def __init__(self, name):
        match name:
            case "LR":
                self.model = LogisticRegression(max_iter=1000)
            case "SVC":
                self.model = SVC(C=10, kernel="sigmoid", gamma=0.02)
            case "KNN":
                self.model = KNN(n_neighbors=2, p=1, weights="distance")
            case "CatBoost":
                self.model = CatBoostClassifier(iterations=200, learning_rate=0.03, allow_const_label=True, verbose=False)
                # iterations=20, depth=4, learning_rate=0.5,
                #                             loss_function='Logloss', allow_const_label=True, verbose=False)

            case "R_Forest":
                self.model = RandomForestClassifier(n_estimators=150, n_jobs=5)

            case "TabPFN":
                self.model = TabPFNClassifier(device="cpu", N_ensemble_configurations=32)
            case _:
                raise Exception("Invalid model specified")

        self.name = name
        self.identical_batch = False
        self.pfn_error = False

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
                # print(self.model.get_all_params())
                # exit(2)
            except CatboostError:
                # Catboost fails if every input element is the same
                self.identical_batch = True
                mode = stats.mode(ys_meta, keepdims=False)[0]
                self.pred_val = mode
            except ValueError as e:
                assert self.name == "TabPFN"
                # TabPFN cant do more than 100 attributes
                self.pfn_error = True
                mode = stats.mode(ys_meta, keepdims=False)[0]
                self.pred_val = mode
                print(e)

    def get_acc(self, xs_target, ys_target):
        xs_target = xs_target.numpy()
        if self.identical_batch or self.pfn_error:
            predictions = np.ones_like(ys_target) * self.pred_val
            self.pfn_error = False
        else:
            predictions = self.model.predict(xs_target)

        return np.array(predictions).flatten() == np.array(ys_target)

    def __repr__(self):
        return self.name


class FLAT(Model):
    def __init__(self, load_no, save_ep=0):
        save_dir = f'{BASEDIR}/saves/save_{load_no}'
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

        ys_pred_target_labels = torch.argmax(ys_pred_target.view(-1, 10), dim=1)
        return (ys_pred_target_labels == ys_target).numpy()

    def __repr__(self):
        return "FLAT"


class FLAT_MAML(Model):
    def __init__(self, load_no, save_ep=None):
        save_dir = f'{BASEDIR}/saves/old_final/save_{load_no}'
        print(f'Loading model at {save_dir = }')

        if save_ep is None:
            state_dict = torch.load(f'{save_dir}/model.pt')
        else:
            state_dict = torch.load(f'{save_dir}/model_{save_ep}.pt')
        self.model = ModelHolder(cfg_all=get_config(cfg_file=f'{save_dir}/defaults.toml'))
        self.model.load_state_dict(state_dict['model_state_dict'])

    def fit(self, xs_meta, ys_meta):

        if xs_meta.shape[1] > 100:
            print("FF slow dataset")
            steps = 1
        else:
            steps = 3


        xs_meta, ys_meta = xs_meta.unsqueeze(0), ys_meta.unsqueeze(0)
        pairs_meta = d2v_pairer(xs_meta, ys_meta)
        with torch.no_grad():
            embed_meta, pos_enc = self.model.forward_meta(pairs_meta)

        embed_meta.requires_grad = True
        pos_enc.requires_grad = True
        optim_pos = torch.optim.Adam([pos_enc], lr=0.001)
        # optim_embed = torch.optim.SGD([embed_meta, ], lr=50, momentum=0.75)
        optim_embed = torch.optim.Adam([embed_meta], lr=0.075)
        for _ in range(steps):
            # Make predictions on meta set and calc loss
            preds = self.model.forward_target(xs_meta, embed_meta, pos_enc)
            loss = torch.nn.functional.cross_entropy(preds.squeeze(), ys_meta.long().squeeze())
            loss.backward()
            optim_pos.step()
            optim_embed.step()
            optim_embed.zero_grad()
            optim_pos.zero_grad()

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
        return "FLAT_maml"


class Iwata(Model):
    def __init__(self, load_no):
        save_dir = f'{BASEDIR}/iwata/{load_no}'
        print(f'Loading model at {save_dir = }')

        self.model = torch.load(f'{save_dir}/model.pt')

    def fit(self, xs_meta, ys_meta):
        xs_meta, ys_meta = xs_meta.unsqueeze(-1).unsqueeze(0), ys_meta.unsqueeze(-1).unsqueeze(-1).unsqueeze(0)

        with torch.no_grad():
            self.model.forward_meta(xs_meta, ys_meta)

    def get_acc(self, xs_target, ys_target) -> np.array:
        xs_target = xs_target.unsqueeze(-1).unsqueeze(0)
        with torch.no_grad():
            ys_pred_target = self.model.forward_target(xs_target)

        ys_pred_target_labels = torch.argmax(ys_pred_target, dim=-1)

        ys_target = ys_target.squeeze().flatten()

        return (ys_pred_target_labels == ys_target).numpy()

    def __repr__(self):
        return "Iwata"


def get_results_by_dataset(test_data_names, models, num_rows):
    """
    Evaluates the model and baseline_models on the test data sets.
    Results are groupped by: data set, model, number of test columns.
    """

    results = pd.DataFrame(columns=['data_name', 'model', 'num_cols', 'acc', 'std'])

    # Test on full dataset
    for data_name in test_data_names:
        print(data_name)
        try:
            #batch = load_batch(ds_name=data_name, num_rows=num_rows)
            dl = SplitDataloader(ds_group=data_name, bs=75, num_rows=num_rows, num_targets=num_rows, num_cols=-3, ds_split="test")
            batch = get_batch(dl, num_rows=num_rows)

        except IndexError as e:
            print(e)
            continue

        model_acc_std = defaultdict(list)
        for model in models:
            mean_acc, std_acc = model.get_accuracy(batch)

            model_acc_std[str(model)].append([mean_acc, std_acc])

        for model_name, acc_stds in model_acc_std.items():
            acc_stds = np.array(acc_stds)
            # For baselines, variance is sample variance.
            if len(acc_stds) == 1:
                mean_acc, std_acc = acc_stds[0, 0], acc_stds[0, 1]

            # Average over all FLAT and FLAT_MAML models.
            # For FLAT, variance is variance between models
            else:
                means, std = acc_stds[:, 0], acc_stds[:, 1]
                mean_acc = np.mean(means)
                std_acc = np.std(means, ddof=1) / np.sqrt(means.shape[0])

            result = pd.DataFrame({
                'data_name': data_name,
                'model': str(model_name),
                'num_cols': -1,
                'acc': mean_acc,
                'std': std_acc
            }, index=[0])
            results = pd.concat([results, result])

    results.reset_index(drop=True, inplace=True)
    return results


def main(load_no, num_rows, ds_group):
    fold_no, split_no = ds_group

    #fold_no = ds_split

    splits = toml.load(f'./datasets/grouped_datasets/splits_{fold_no}')

    get_splits = range(6)

    test_data_names = []
    for split in get_splits:
        ds_name = splits[str(split)]["test"]
        test_data_names += ds_name

    # print("Train datases:", train_data_names)
    print("Test datasets:", test_data_names)

    #test_data_names.remove("semeion")

    models = [BasicModel("LR"), FLAT(29)] #[FLAT(num) for num in load_no]

    unseen_results = get_results_by_dataset(
        test_data_names, models, num_rows
    )

    # Results for each dataset
    detailed_results = unseen_results.copy()

    mean, std = detailed_results["acc"], detailed_results["std"]
    mean_std = [f'{m * 100:.2f}±{s * 100:.2f}' for m, s in zip(mean, std)]
    detailed_results['acc_std'] = mean_std

    results = detailed_results.pivot(columns=['data_name', 'model'], index='num_cols', values=['acc_std'])
    # print("======================================================")
    # print("Test accuracy on unseen datasets")
    # print(results.to_string())

    det_results = detailed_results.pivot(columns=['data_name', 'model'], index='num_cols', values=['acc'])
    det_results = det_results.to_string()

    # Aggreate results
    agg_results = unseen_results.copy()

    # Move flat to first column
    agg_results = agg_results.groupby(['num_cols', 'model'])['acc'].mean().unstack()
    new_column_order = ["FLAT", "FLAT_maml"] + [col for col in agg_results.columns if (col != "FLAT" and col != "FLAT_maml")]
    agg_results = agg_results.reindex(columns=new_column_order)

    # Difference between FLAT and best model
    # agg_results["FLAT_diff"] = (agg_results["FLAT"] - agg_results.iloc[:, 2:].max(axis=1)) * 100
    # agg_results["FLAT_maml_diff"] = (agg_results["FLAT_maml"] - agg_results.iloc[:, 2:-1].max(axis=1)) * 100
    # agg_results["FLAT_diff"] = agg_results["FLAT_diff"].apply(lambda x: f'{x:.2f}')
    # agg_results["FLAT_maml_diff"] = agg_results["FLAT_maml_diff"].apply(lambda x: f'{x:.2f}')

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

    # print()
    # print("======================================================")
    # print("Test accuracy on unseen datasets (aggregated)")
    #print(agg_results["FLAT_diff"].to_string(index=False))
    # print(agg_results.to_string(index=False))
    print(agg_results.to_string())
    agg_results = agg_results.to_string()


    # with open(f'{result_dir}/aggregated', "w") as f:
    #     for line in agg_results:
    #         f.write(line)
    #
    # with open(f'{result_dir}/detailed', "w") as f:
    #     for line in det_results:
    #         f.write(line)
    #
    # with open(f'{result_dir}/raw.pkl', "wb") as f:
    #     pickle.dump(unseen_results, f)


if __name__ == "__main__":
    # random.seed(0)
    # np.random.seed(0)
    # torch.manual_seed(0)

    main(load_no=[], num_rows=20, ds_group=(0, -1))
