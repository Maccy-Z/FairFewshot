import torch
import numpy as np
import pandas as pd
import os
import random
from itertools import islice

DATADIR = './datasets/data_kasia'

class SimpleDataset():
    def __init__(self, data_name, split="train"):
        self.split = split
        self.data_name = data_name
        self.train = pd.read_csv(
            f'{DATADIR}/{data_name}/{data_name}_train_py.dat'
        )
        self.val = pd.read_csv(
            f'{DATADIR}/{data_name}/{data_name}_val_py.dat'
        )
        self.test = pd.read_csv(
            f'{DATADIR}/{data_name}/{data_name}_test_py.dat'
        )
        self.num_cols = self.train.shape[1] - 1

    def _shuffle_cols(self, xs):
        col_idx = np.random.permutation(xs.shape[1])
        return xs.iloc[:, col_idx]
    
    def _normalize(self, xs):
        m = xs.mean(0, keepdim=True)
        s = xs.std(0, unbiased=False, keepdim=True)
        xs -= m
        xs /= (s + 10e-4)
        return xs

    def stratified_sample(self, num_rows, norm=True, shuffle_cols=False, num_cols=None):
        if self.split == "train":
            dataset = self.train
        elif self.split == "val":
            dataset = self.val
        elif self.split == "test":
            dataset = self.test

        sample_data = dataset.groupby('label', group_keys=False).apply(
            lambda x: x.sample(num_rows // 2 + 1, replace=True))

        sample_data = sample_data.sample(num_rows)

        xs, ys = sample_data.iloc[:, :-1], sample_data.iloc[:, -1]
        if shuffle_cols:
            xs = self._shuffle_cols(xs)
        if num_cols:
            col_idx = np.random.permutation(num_cols)
            xs = xs.iloc[:, col_idx]
        xs = torch.tensor(xs.values, device='cpu', dtype=torch.float32)
        ys = torch.tensor(ys.values, device='cpu', dtype=torch.float32)
        if norm:
            xs = self._normalize(xs)
        return xs, ys

class MyDataLoader:
    def __init__(self, bs, num_rows, num_targets, data_names, num_cols = None,
                 split="train", norm=True, shuffle_cols=False, min_num_cols=2):
        """
        :param:
            bs: batch size
            num_rows: num meta-training rows
            num_targets: num meta-target rows
            num_cols: (optional) list of no. predictor columns to 
                be sampled by the data loader
            data_names: list of dataset names to be used by the loader
            split: train, val, or test
            norm: if inputs should be normalized
            shuffle_cols: if columns should be randomly shuffled
            min_num_cols: if num_cols is None, sample no. columns from range 
                between min_num_cols and the smallest number of columns among 
                the datasets
        """
        self.bs = bs
        self.num_rows = num_rows + num_targets
        self.num_cols = num_cols
        self.norm = norm
        self.shuffle_cols = shuffle_cols
        self.num_cols = num_cols
        self.data_names = data_names
        self.min_num_cols = min_num_cols
        self.datasets = [SimpleDataset(d, split=split) for d in self.data_names]


    def __iter__(self):
        """
        :return: [bs, num_rows, num_cols], [bs, num_rows, 1]
        """
        while True:
            num_data_cols = [d.num_cols for d in self.datasets]
            if self.num_cols:
                num_cols = np.random.choice(self.num_cols)
            else:
                min_col, max_col = self.min_num_cols, min(num_data_cols)
                num_cols = np.random.randint(min_col, max_col)

            datasets = random.choices(self.datasets, k=self.bs)
            datanames = [d.data_name for d in datasets]
            xs, ys = list(zip(*[d.stratified_sample(
                self.num_rows, norm=self.norm, shuffle_cols=self.shuffle_cols, num_cols=num_cols) 
                for d in datasets]))
            xs = torch.stack(xs)
            ys = torch.stack(ys)
            yield xs, ys.long()# , datanames
        
if __name__ == "__main__":
    # Test if dataloader works.
    np.random.seed(0)
    torch.manual_seed(0)

    dl = MyDataLoader(
        bs=1, num_rows=5, num_targets=5, num_cols=None, 
        data_names=['oocytes_merluccius_nucleus_4d', 'oocytes_merluccius_states_2f'], 
        split="train", shuffle_cols=True)

    for xs, ys, datanames in islice(dl, 10):
        print(xs.shape)