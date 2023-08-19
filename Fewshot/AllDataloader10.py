# %%
import torch
import numpy as np
import pandas as pd
import os
import random
import toml
from itertools import islice
import matplotlib.pyplot as plt

DATADIR = './datasets'

N_class = 10


def to_tensor(array: np.array, device, dtype=torch.float32):
    return torch.from_numpy(array).to(device).to(dtype)


def binarise_data(ys):
    median = torch.median(ys)

    if np.random.randint(2) == 1:
        ys = (ys >= median)
    else:
        ys = (ys > median)
    return ys.long()


def one_vs_all(ys):
    # identify the most common class and set it to 1, set everything else as 0
    mode = ys.flatten().mode()[0]
    idx = ys == mode
    ys = torch.zeros_like(ys).long()
    ys[idx] = 1

    return ys


class MyDataSet:
    def __init__(
            self, ds_name, num_rows, num_targets, balance, split,
            dtype=torch.float32, device="cpu"):
        self.ds_name = ds_name
        self.num_rows = num_rows
        self.num_targets = num_targets
        self.tot_rows = num_rows + num_targets
        self.num_meta = num_rows
        self.balance = balance

        self.device = device
        self.dtype = dtype

        self.train, self.valid, self.test = False, False, False

        """
        Dataset format: {folder}_py.dat             predictors
                        labels_py.dat               labels for predictors
                        folds_py.dat                test fold
                        validation_folds_py.dat     validation fold
        folds_py == 0 for train
        vfold_py == 1 for validation
        folds_py == 1 for test                 
        
        Here, combine test and valid folds. 
        """
        ds_dir = f'{DATADIR}/data'
        # get train fold
        folds = pd.read_csv(
            f"{ds_dir}/{ds_name}/folds_py.dat", header=None)[0]
        folds = np.asarray(folds)
        # get validation fold
        vldfold = pd.read_csv(
            f"{ds_dir}/{ds_name}/validation_folds_py.dat", header=None)[0]
        vldfold = np.asarray(vldfold)

        # read predictors
        predictors = pd.read_csv(
            f"{ds_dir}/{ds_name}/{self.ds_name}_py.dat", header=None)
        predictors = np.asarray(predictors)
        # read internal target
        targets = pd.read_csv(f"{ds_dir}/{ds_name}/labels_py.dat", header=None)
        targets = np.asarray(targets)

        if split == "train":
            idx = (1 - folds) == 1 & (vldfold == 0)
            self.train = True
        elif split == "val":
            self.valid = True
            idx = (vldfold == 1)
        elif split == "test":
            self.test = True
            idx = (folds == 1)
        elif split == "all":
            idx = np.ones_like(folds).astype(bool)
        else:
            raise Exception("Split must be train, val, test or all")

        preds = predictors[idx]
        labels = targets[idx]
        # labels = (labels - np.mean(labels)) / np.std(labels)

        data = np.concatenate((preds, labels), axis=-1)
        self.data = to_tensor(data, device=device)
        self.ds_cols = self.data.shape[-1]
        self.ds_rows = self.data.shape[0]

        col_data = self.data[:, -1]
        unique_lab, unique_idx, counts = np.unique(
            col_data, return_counts=True, return_inverse=True)
        self.num_classes = len(unique_lab)
        sorted_indices = sorted(range(len(counts)), key=lambda i: counts[i], reverse=True)
        largest_labels = sorted_indices[:N_class]

        row_probs = np.zeros(self.ds_rows)
        self.wanted_rows, lens = [], []
        for l in largest_labels:
            top_idx = (unique_idx == l)
            row_probs[top_idx] = 1 / counts[l]

            self.wanted_rows.append(np.where(top_idx)[0])
            lens.append(len(np.where(top_idx)[0]))

        # if min(lens) < 5:
        #     self.ds_rows = 0
        if len(counts) < N_class:
            self.ds_rows = 0

        row_probs = row_probs / np.sum(row_probs)

        self.row_probs = row_probs.T

        self.test = len(counts)

    def sample(self, num_cols):
        # print(self.test, self.min_ds_rows, self)
        predict_cols = np.random.choice(self.ds_cols - 1, size=num_cols, replace=False)

        if self.balance is None:
            rows = np.random.choice(
                self.ds_rows, size=self.tot_rows, replace=False, p=self.row_probs)
        else:
            exit(5)
            rows = []

            class_split = [self.balance, self.balance, self.num_meta - 2 * self.balance]
            class_split = np.random.permutation(class_split)


            for label, num_rows in enumerate(class_split):
                wanted_row = np.random.choice(self.wanted_rows[label], size=num_rows, replace=False)
                rows.append(wanted_row)

            targ_rows = np.random.choice(
                self.ds_rows, size=self.num_targets, replace=False, p=self.row_probs)

            rows .append(targ_rows)
            rows = np.concatenate(rows)

        select_data = self.data[rows]

        # Pick out wanted columns
        xs = select_data[:, predict_cols]
        ys = select_data[:, -1]

        # Normalise xs
        m = xs.mean(0, keepdim=True)
        s = xs.std(0, unbiased=False, keepdim=True)
        xs -= m
        xs /= (s + 10e-4)

        unique_values = torch.unique(ys)
        mapping = {unique_values[i].item(): i for i in range(len(unique_values))}

        ys = torch.tensor([mapping[v.item()] for v in ys])
        return xs, ys

    def __repr__(self):
        return self.ds_name

    def __len__(self):
        return self.ds_rows

    def cols(self):
        return self.ds_cols


class SplitDataloader:
    def __init__(
            self, bs, num_rows, num_targets, balance=None,
            num_cols=-1, ds_group=-1, ds_split="train", device="cpu",
            split_file='./datasets/grouped_datasets/splits'):
        """

        :param bs: Number of datasets to sample from each batch
        :param num_rows: Number of meta rows
        :param num_targets: Number of target rows
        :param binarise: Binarise the dataset upon loading instead of with each sample
        :param num_cols: Number of columns to sample from. 
            If 0, random no. columns between 2 and the largest number of 
                columns among all datasets
            If -1, random no. columns between 2 and the smallest number of 
                columns among all datasets
            If -2, sample datasets with equal probability, then sample valid number of columns.
            If -3, sample datasets with equal probability, take max allowed number of columns.
            If list, sample from a range of no. columns specified in the list
        :param ds_group: Which datasets to sample from. 
            If -1, sample all available datasets
            If int >= 0, referes to premade group specified in the split_file 
            If string or list of strings, sample from that specified dataset(s).
        :param ds_split: If ds_group is int >= 0, the test or train split.
        :param split_fmt: format of data to load.
        """

        self.bs = bs
        self.tot_rows = num_rows + num_targets
        self.num_rows = num_rows
        self.num_targets = num_targets
        self.balance = balance
        self.num_cols = num_cols
        self.ds_group = ds_group
        self.ds_split = ds_split
        self.split_file = split_file


        self.device = device

        self._get_valid_datasets()
        if isinstance(num_cols, list):
            self._check_num_cols()

    def _get_valid_datasets(self):
        ds_dir = f'{DATADIR}/data/'
        if isinstance(self.ds_group, tuple):
            fold_no, split_no = self.ds_group
            splits = toml.load(f'./datasets/grouped_datasets/splits_{fold_no}')
            if split_no == -1:
                get_splits = range(6)
            else:
                get_splits = [split_no]

            ds_names = []
            for split in get_splits:
                names = splits[str(split)][self.ds_split]
                ds_names += names

        elif isinstance(self.ds_group, int):
            if self.ds_group == -1:
                # get all datasets
                ds_names = os.listdir(ds_dir)
                ds_names.remove('info.json')
                if '.DS_Store' in ds_names:
                    ds_names.remove('.DS_Store')
            else:
                # get datasets from pre-defined split
                splits = toml.load(self.split_file)
                ds_names = splits[str(self.ds_group)][self.ds_split]

        elif isinstance(self.ds_group, str):
            ds_names = [self.ds_group]

        elif isinstance(self.ds_group, list):
            ds_names = self.ds_group
        else:
            raise Exception("Invalid ds_group")

        self.all_datasets = [
            MyDataSet(name, num_rows=self.num_rows,
                      num_targets=self.num_targets,
                      balance=self.balance,
                      device=self.device, split="all")
            for name in ds_names]

        valid_datasets = []
        for d in self.all_datasets:
            if len(d) >= self.tot_rows:
                valid_datasets.append(d)
            else:
                print(f"WARN: Discarding {d}, due to not enough rows")
        self.all_datasets = valid_datasets

        if len(self.all_datasets) == 0:
            raise IndexError(f"No datasets with enough rows. Required: {self.tot_rows}")

        ds_len = [ds.ds_cols for ds in self.all_datasets]
        self.min_ds_cols = min(ds_len)

    def _check_num_cols(self):
        max_num_cols = max(self.num_cols)
        valid_datasets = [
            d for d in self.all_datasets if d.ds_cols > max_num_cols]
        if not valid_datasets:
            raise IndexError(
                "Provided range of columns to sample exceeds the "
                + "dimension of the largest dataset available" + f' {max_num_cols}')

    def __iter__(self):
        """
        :return: [bs, num_rows, num_cols], [bs, num_rows, 1]
        """
        while True:
            # Sample columns uniformly
            if self.num_cols == 0 or self.num_cols == -1 or isinstance(self.num_cols, list):
                if isinstance(self.num_cols, int):
                    if self.num_cols == 0:
                        max_num_cols = max([d.ds_cols for d in self.all_datasets]) - 1

                    elif self.num_cols == -1:
                        max_num_cols = min([d.ds_cols for d in self.all_datasets]) - 1
                    num_cols_range = [2, max_num_cols]

                else:
                    num_cols_range = self.num_cols

                if self.decrease_col_prob == -1:
                    num_cols = np.random.choice(
                        list(range(num_cols_range[0], num_cols_range[1] + 1)), size=1)[0]
                else:
                    num_cols = np.random.geometric(p=self.decrease_col_prob, size=1) + 1
                    num_cols = max(num_cols_range[0], num_cols)
                    num_cols = min(num_cols, num_cols_range[1])
                valid_datasets = [d for d in self.all_datasets if d.ds_cols > num_cols]
                datasets = random.choices(valid_datasets, k=self.bs)

            # Sample datasets uniformly
            elif self.num_cols == -2:
                datasets = random.choices(self.all_datasets, k=self.bs)
                max_num_cols = min([d.ds_cols for d in datasets]) - 1
                num_cols = np.random.randint(2, max_num_cols)

            elif self.num_cols == -3:
                datasets = random.choices(self.all_datasets, k=self.bs)
                num_cols = min([d.ds_cols for d in datasets]) - 1
            else:
                raise Exception("Invalid num_cols")

            datanames = [str(d) for d in datasets]

            xs, ys = list(zip(*[
                ds.sample(num_cols=num_cols)
                for ds in datasets]))
            xs = torch.stack(xs)
            ys = torch.stack(ys)
            yield xs, ys, datanames

    def __repr__(self):
        return str(self.all_datasets)

    def __len__(self):
        return self.all_datasets[0].cols()


if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)

    datasets = os.listdir("/mnt/storage_ssd/fewshot_learning/FairFewshot/datasets/data")
    datasets = sorted([f for f in datasets if os.path.isdir(f'/mnt/storage_ssd/fewshot_learning/FairFewshot/datasets/data/{f}')])
    dl = SplitDataloader(
        bs=1, num_rows=20, num_targets=5, num_cols=-3, ds_group=datasets, ds_split="train",
    )

    print(str(dl).split(","))
    # formatted_strings = [f"'{item}'" for item in str(dl)]
    # print(formatted_strings)
    # output = ', '.join(formatted_strings)
    #
    # print(output)
