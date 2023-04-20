import torch
import numpy as np
import pandas as pd
import os
import random
import toml

os.chdir("/Users/kasiakobalczyk/FairFewshot")
DATADIR = './datasets'


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
    def __init__(self, data_name, num_rows, split="train", balance_train=False,
                 one_v_all=False, dtype=torch.float32, device="cpu"):
        # data_name = "adult"
        self.data_name = data_name
        self.device = device
        self.dtype = dtype
        self.balance_train = balance_train
        self.train, self.valid, self.test = False, False, False
        self.num_rows = num_rows
        self.one_v_all = one_v_all

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
        folds = pd.read_csv(f"{ds_dir}/{data_name}/folds_py.dat", header=None)[0]
        folds = np.asarray(folds)
        # get validation fold
        vldfold = pd.read_csv(f"{ds_dir}/{data_name}/validation_folds_py.dat", header=None)[0]
        vldfold = np.asarray(vldfold)

        # read predictors
        predictors = pd.read_csv(f"{ds_dir}/{data_name}/{self.data_name}_py.dat", header=None)
        predictors = np.asarray(predictors)
        # read internal target
        targets = pd.read_csv(f"{ds_dir}/{data_name}/labels_py.dat", header=None)
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
            raise Exception("Split must be train, val or test")

        preds = predictors[idx]
        labels = targets[idx]
        labels = (labels - np.mean(labels)) / np.std(labels)
        num_ys = labels.shape[-1]

        data = np.concatenate((preds, labels), axis=-1)
        self.data = to_tensor(data, device=device)
        self.num_cols = self.data.shape[-1]
        self.tot_rows = self.data.shape[0]
        self.cols = np.arange(self.num_cols)

        # Select columns to return as ys
        if False:  # self.train:
            self.allow_targs = np.arange(self.num_cols)
        else:
            self.allow_targs = np.arange(self.num_cols - num_ys, self.num_cols)

        # If one label makes up more than 50% of the column, downweight its sampling probability of category to 50%.
        # Always done for val/test, optional for train
        row_probs = []
        for col_no in range(self.data.shape[1]):
            probs = np.ones(self.tot_rows)

            col_data = self.data[:, col_no]
            unique_lab, unique_idx, counts = np.unique(col_data, return_counts=True, return_inverse=True)
            if np.max(counts) / self.tot_rows > 0.5:
                max_prob = self.tot_rows / np.max(counts) - 1

                # Some cols have all entries identical.
                if max_prob == 0:
                    pass
                    # print(f"Warning: Entire column contains 1 unique entry, ds={self.data_name}, {col_no=}")
                    # print(np.max(counts) / self.tot_rows)
                else:
                    top_idx = (unique_idx == np.argmax(counts))
                    probs[top_idx] = max_prob

            probs = probs / np.sum(probs)
            row_probs.append(probs)
        row_probs = np.stack(row_probs).T

        self.row_probs = row_probs.T

    def sample(self, num_xs, force_next=False):

        if num_xs > self.num_cols - 1:  # or not self.train:
            num_xs = self.num_cols - 1

        targ_col = np.random.choice(self.allow_targs, size=1)

        row_cols = np.setdiff1d(self.cols, targ_col)
        predict_cols = np.random.choice(row_cols, size=num_xs, replace=False)
        if not self.train or self.balance_train:
            rows = np.random.choice(self.tot_rows, size=self.num_rows, replace=False, p=self.row_probs[targ_col].squeeze())
        else:
            rows = np.random.choice(self.tot_rows, size=self.num_rows, replace=False)
            assert False
        select_data = self.data[rows]

        # Pick out wanted columns
        xs = select_data[:, predict_cols]
        ys = select_data[:, targ_col]

        # Normalise xs
        m = xs.mean(0, keepdim=True)
        s = xs.std(0, unbiased=False, keepdim=True)
        xs -= m
        xs /= (s + 10e-4)

        if self.train and not self.one_v_all:
            ys = binarise_data(ys)
            assert False
        else:
            ys = one_vs_all(ys)

        # If a batch is exclusively 1 or 0 as label, try to regenerate the batch, once only
        if force_next or ys.min() != ys.max():
            return xs, ys
        else:
            return self.sample(num_xs, force_next=True)

    def __repr__(self):
        return self.data_name

    def __len__(self):
        return len(self.data)


class AllDatasetDataLoader:
    def __init__(self, bs, num_rows, num_targets, num_cols=-1, ds_group=-1, one_v_all=False, balance_train=True, device="cpu", split="train"):
        assert False
        self.bs = bs
        self.num_rows = num_rows + num_targets
        self.split = split
        self.device = device
        self.num_cols = num_cols
        self.ds_group = ds_group
        self.train = (split == "train")
        self.balance_train = balance_train
        self.one_v_all = one_v_all

        if not self.train and self.bs != 1:
            raise Exception("During val/test, BS must be 1 since full datasets are used.")

        self.get_valid_datasets()

    def get_valid_datasets(self):
        if self.ds_group == -1:
            ds_dir = f'{DATADIR}/data'
            all_data_names = os.listdir(ds_dir)
            all_data_names.remove('info.json')
        else:
            ds_dir = f'{DATADIR}/grouped_datasets/{self.ds_group}'
            all_data_names = os.listdir(ds_dir)

        all_datasets = [
            MyDataSet(d, num_rows=self.num_rows, split=self.split, device=self.device,
                      balance_train=self.balance_train, one_v_all=self.one_v_all)
            for d in all_data_names
        ]
        min_ds = 500 if self.train else self.num_rows * 2
        self.datasets = [
            d for d in all_datasets if d.tot_rows >= min_ds
        ]

    def __iter__(self):
        """
        :return: [bs, num_rows, num_cols], [bs, num_rows, 1]
        """
        while True:
            datasets = random.sample(self.datasets, self.bs)
            datanames = [d.data_name for d in datasets]
            if self.num_cols == -1:
                num_xs = min([d.num_cols for d in datasets]) - 1
            else:
                num_xs = self.num_cols
            xs, ys = list(zip(*[
                datasets[i].sample(num_xs=num_xs)
                for i in range(self.bs)]))
            xs = torch.stack(xs)
            ys = torch.stack(ys)
            yield xs, ys  # , datanames


class SplitDataloader:
    def __init__(self, bs, num_rows, num_targets, num_cols=-1, ds_group=-1, one_v_all=False, balance_train=True, device="cpu", split="train"):

        self.bs = bs
        self.num_rows = num_rows + num_targets
        self.device = device
        self.num_cols = num_cols
        self.ds_group = ds_group
        self.balance_train = balance_train
        self.one_v_all = one_v_all

        self.train = (split == "train")
        if split == "val":
            split = "test"
        self.split = split

        if not self.train and self.bs != 1:
            raise Exception("During val/test, BS must be 1 since full datasets are used.")

        self.get_valid_datasets()

        # print(f'Split: {split}, datasets: {self.datasets}')

    def get_valid_datasets(self):
        ds_dir = f'{DATADIR}/grouped_datasets/'
        splits = toml.load(f'{ds_dir}/splits')
        all_datasets = []
        if self.ds_group == -1:
            get_splits = sorted([f for f in os.listdir(ds_dir) if os.path.isdir(f'{ds_dir}/{f}')])
        else:
            get_splits = [str(self.ds_group)]

        for split in get_splits:
            ds_names = splits[str(split)][self.split]
            for name in ds_names:
                ds = MyDataSet(name, num_rows=self.num_rows, device=self.device, split="test",
                               balance_train=self.balance_train, one_v_all=self.one_v_all)
                all_datasets.append(ds)

        min_ds = self.num_rows * 2
        self.datasets = [
            d for d in all_datasets if len(d) >= min_ds
        ]

    def __iter__(self):
        """
        :return: [bs, num_rows, num_cols], [bs, num_rows, 1]
        """
        while True:
            datasets = random.sample(self.datasets, self.bs)
            datanames = [d.data_name for d in datasets]
            if self.num_cols == -1:
                num_xs = min([d.num_cols for d in datasets]) - 1
            else:
                num_xs = self.num_cols
            xs, ys = list(zip(*[
                datasets[i].sample(num_xs=num_xs)
                for i in range(self.bs)]))
            xs = torch.stack(xs)
            ys = torch.stack(ys)
            yield xs, ys  # , datanames

    def __repr__(self):
        return str(self.datasets)

if __name__ == "__main__":
    dl = SplitDataloader(bs=1, num_rows=16, num_targets=3, num_cols=-1, ds_group=4, one_v_all=True, split="train")
    print(dl)

    exit(10)

    means = []
    dl = iter(dl)
    y_count = {i: 0 for i in range(20)}
    for _ in range(1000):
        x, y = next(dl)

        print(x.shape)

        num = torch.sum(y).item()
        y_count[num] += 1

        y_mean = torch.mean(y, dtype=float)
        means.append(y_mean)

    means = torch.stack(means)
    means = torch.mean(means)
    print("Mean y value", f'{means.item():.4g}')
    print("Histogram of number of positive samples", y_count)

# {0: 14, 1: 10, 2: 13, 3: 26, 4: 33, 5: 76, 6: 348, 7: 311, 8: 67, 9: 46, 10: 12, 11: 7, 12: 7, 13: 30}
# {0: 28, 1: 21, 2: 22, 3: 23, 4: 36, 5: 52, 6: 313, 7: 343, 8: 49, 9: 37, 10: 17, 11: 10, 12: 9, 13: 40}
