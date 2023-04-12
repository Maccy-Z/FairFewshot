import torch
import numpy as np
import pandas as pd
import os
import random

DATADIR = './datasets/data'

def to_tensor(array: np.array, device, dtype=torch.float32):
    return torch.from_numpy(array).to(device).to(dtype)


def binarise_data(ys):
    median = torch.median(ys)
    ys = (ys > median)
    return ys.long()

def one_vs_all(ys):
    # identify the most common class and set it to 1, set everything else as 0
    mode = ys.mode()[0]
    idx = ys == mode
    ys = torch.zeros(ys.shape[0])
    ys[idx] = 1
    return ys


class MyDataSet:
    def __init__(self, data_name, num_rows, split="train", dtype=torch.float32, device="cpu"):
        self.data_name = data_name
        self.device = device
        self.dtype = dtype
        self.train, self.valid, self.test = False, False, False
        self.num_rows = num_rows

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

        # get train fold
        folds = pd.read_csv(f"{DATADIR}/{data_name}/folds_py.dat", header=None)[0]
        folds = np.asarray(folds)
        # get validation fold
        vldfold = pd.read_csv(f"{DATADIR}/{data_name}/validation_folds_py.dat", header=None)[0]
        vldfold = np.asarray(vldfold)

        # read predictors
        predictors = pd.read_csv(f"{DATADIR}/{data_name}/{self.data_name}_py.dat", header=None)
        predictors = np.asarray(predictors)
        # read internal target
        targets = pd.read_csv(f"{DATADIR}/{data_name}/labels_py.dat", header=None)
        targets = np.asarray(targets)

        if split=="train":
            idx = (folds == 0)
            self.train = True
        elif split=="val":
            self.valid = True
            idx = (vldfold == 1)
        elif split=="test":
            self.ttest = True
            idx = (folds == 1)
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
        if split == "train":
            self.allow_targs = np.arange(self.num_cols)
        else:
            self.allow_targs = np.arange(self.num_cols - num_ys, self.num_cols)


    def sample(self, num_xs, force_next=False):

        if num_xs > self.num_cols or not self.train:
            num_xs = self.num_cols - 1

        target_col = np.random.choice(self.allow_targs, size=1)

        row_cols = np.setdiff1d(self.cols, target_col)
        predict_cols = np.random.choice(row_cols, size=num_xs, replace=False)


        rows = np.random.choice(self.tot_rows, size=self.num_rows, replace=False)
        select_data = self.data[rows]

        # Pick out wanted columns
        xs = select_data[:, predict_cols]
        ys = select_data[:, target_col]

        # If a batch is exclusively 1 or 0 as label, regenerate the batch
        ys = binarise_data(ys)

        if force_next or ys.min() != ys.max():
            return xs, ys
        else:
            return self.sample(num_xs, force_next=True)



class AllDatasetDataLoader:
    def __init__(
            self, bs, num_rows, num_targets, num_cols=-1, device="cpu", split="train"):

        self.bs = bs
        self.num_rows = num_rows + num_targets
        self.split = split
        self.device = device
        self.num_cols = num_cols
        self.train = (split=="train")

        if not self.train and self.bs != 1:
            raise Exception("During val/test, BS must be 1 since full datasets are used.")

        self.get_valid_datasets()

    def get_valid_datasets(self):
        all_data_names = os.listdir(DATADIR)
        all_data_names.remove('info.json')
        all_datasets = [
            MyDataSet(d, num_rows=self.num_rows, split=self.split, device=self.device)
            for d in all_data_names
        ]
        self.datasets = [
            d for d in all_datasets if d.tot_rows >= 250
        ]


    def __iter__(self):
        """
        :return: [bs, num_rows, num_cols], [bs, num_rows, 1]
        """
        while True:
            datasets = random.sample(self.datasets, self.bs)
            datanames = [d.data_name for d in datasets]
            num_xs = min([d.num_cols for d in datasets]) - 1

            xs, ys = list(zip(*[
                datasets[i].sample(num_xs = num_xs)
                for i in range(self.bs)]))
            xs = torch.stack(xs)
            ys = torch.stack(ys)

            yield xs, ys # , datanames

if __name__ == "__main__":
    dl = AllDatasetDataLoader(bs=1, num_rows=10, num_targets=3, split="train")

    means = []
    dl = iter(dl)
    for _ in range(1000):
        x, y = next(dl)
        y_mean = torch.mean(y, dtype=float)
        means.append(y_mean)


    means = torch.stack(means)
    means = torch.mean(means)
    print(means.item())


