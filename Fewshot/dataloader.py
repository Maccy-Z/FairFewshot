# Loads datasets.
import torch
import numpy as np
import os
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from itertools import islice

DATADIR = './datasets/data'

# Loads a single dataset. Split into train and valid.
class Dataset:
    def __init__(self, data_name, split="train", dtype=torch.float32, device="cpu"):
        self.data_name = data_name
        self.device = device
        self.dtype = dtype
        """      
        Data columns and num_catagories, 
        0, Age: 76
        1, Workclass: 9
        2, fnlwgt: 18299
        3, Education: 16
        4, Education-num: 16
        5, Marital: 7
        6, Occupation: 15
        7, Relation: 6 
        8, Race: 5
        9, Sex: 2
        10, Cap-gain: 118
        11, Cap-loss: 91
        12, Hr-per-wk: 92
        13, Native_cont: 41
        14, Income >50k: 2
        """

        datadir = f'./datasets/data_split/{self.data_name}'

        # Read data and labels
        full_dataset = pd.read_csv(f"{datadir}/{split}", header=None)
        full_dataset = np.asarray(full_dataset)

        # Combine validation and test folds together

        self.data = self._to_tensor(full_dataset)

    def _to_tensor(self, array: np.array):
        return torch.from_numpy(array).to(self.device).to(self.dtype)


# Randomly samples from dataset. Returns a batch of tables for use in the GNN
class AdultDataLoader:
    def __init__(self, *, bs, num_rows, num_targets, num_xs=10, flip=True, device="cpu", split="train"):
        self.bs = bs
        self.num_rows = num_rows
        self.num_targets = num_targets
        self.num_xs = num_xs
        self.flip = flip

        self.ds = Dataset("adult", device=device, split=split)

        self.data = self.ds.data
        self.len = self.data.shape[0]
        self.cols = self.data.shape[1]

    # Pick out bs rows from dataset. Select 1 column to be target and random columns as predictors.
    # Only certain column can be targets since some are continuous.
    def __iter__(self):
        """
        :return: [bs, num_rows, num_cols], [bs, num_rows, 1]
        """
        num_rows = self.num_rows + self.num_targets

        while True:
            # Randomise data order
            permutation = torch.randperm(self.data.shape[0])
            data = self.data[permutation]

            allowed_targets = [0]#range(15)
            cols = np.arange(self.cols)

            for st in torch.arange(0, self.len - num_rows * self.bs, num_rows * self.bs):
                # TODO: Allow flexible number of columns within a batch. Currently, each batch is fixed num_xs.
                self.num_xs = 10

                # Get target and predictor columns
                target_cols = np.random.choice(allowed_targets, size=self.bs)

                predict_cols = np.empty((self.bs, self.num_xs))
                for batch, target in enumerate(target_cols):
                    row_cols = np.setdiff1d(cols, target)
                    predict_cols[batch] = np.random.choice(row_cols, size=self.num_xs, replace=False)

                target_cols = torch.from_numpy(target_cols).int()
                predict_cols = torch.from_numpy(predict_cols).int()

                # Rows of data to select
                select_data = data[st:st + num_rows * self.bs]

                # Pick out wanted columns
                predict_idxs = predict_cols.repeat_interleave(num_rows, dim=0)
                target_idxs = target_cols.repeat_interleave(num_rows)

                xs = select_data[np.arange(self.bs * num_rows).reshape(-1, 1), predict_idxs]
                ys = select_data[np.arange(self.bs * num_rows), target_idxs]

                xs = xs.view(self.bs, num_rows, self.num_xs)  # [bs, num_rows, num_cols]
                ys = ys.view(self.bs, num_rows)  # [bs, num_rows]

                ys = self.binarise_data(ys)
                yield xs, ys

    # Convert target columns into a binary classification task.
    def binarise_data(self, ys):
        median = torch.median(ys[:, :self.num_rows], dim=1, keepdim=True)[0]
        ys = (ys > median)

        if self.flip:
            # Random flip to balance dataset
            flip = torch.randint(0, 2, (self.bs, 1))
            ys = torch.logical_xor(ys, flip)

        return ys.long()

    def __len__(self):
        num_batches = self.len // ((self.num_rows + self.num_targets) * self.bs)
        return num_batches


# Dataset2vec requires different dataloader from GNN. Returns all pairs of x and y.
def d2v_pairer(xs, ys):
    #    # torch.Size([2, 5, 10]) torch.Size([2, 5, 1])
    bs, num_rows, num_xs = xs.shape

    xs = xs.view(bs * num_rows, num_xs)
    ys = ys.view(bs * num_rows, 1)

    pair_flat = torch.empty(bs * num_rows, num_xs, 2, device=xs.device)
    for k, (xs_k, ys_k) in enumerate(zip(xs, ys)):
        # Only allow 1D for ys
        ys_k = ys_k.repeat(num_xs)
        pairs = torch.stack([xs_k, ys_k], dim=-1)

        pair_flat[k] = pairs

    pairs = pair_flat.view(bs, num_rows, num_xs, 2)

    return pairs

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

def to_tensor(array: np.array, device, dtype=torch.float32):
    return torch.from_numpy(array).to(device).to(dtype)

class MyDataSet():
    def __init__(self, data_name, split="train", dtype=torch.float32, device="cpu"):
        self.data_name = data_name
        self.device = device
        self.dtype = dtype
        self.split = split

        if split == "train":
            self.train = pd.read_csv(
                f'{DATADIR}/{data_name}/{data_name}_train_py.dat', 
                header=None)
            self.train = to_tensor(np.asarray(self.train), device=self.device, dtype=dtype)
            self.num_predictors = self.train.shape[1] - 1

        elif split == "val":
            self.predictors = pd.read_csv(
                f'{DATADIR}/{data_name}/{data_name}_val_py.dat', 
                header=None)
            self.predictors = to_tensor(
                np.asarray(self.predictors), device=self.device, dtype=dtype)
            self.targets = pd.read_csv(
                f'{DATADIR}/{data_name}/labels_py.dat', 
                header=None).iloc[:, 0]
            n_classes = len(np.unique(self.targets))
            # if n_classes > 1:
            #     self.targets = one_vs_all(self.targets)
            self.targets = to_tensor(
                np.asarray(self.targets), device=self.device, dtype=dtype)
            self.num_predictors = self.predictors.shape[1]
            self.num_total_rows = self.predictors.shape[0]

    def sample(self, num_rows, num_cols, norm):
        if num_cols > self.num_predictors:
            raise Exception("More columns requested than available predictors")
        
        if self.split == "train":
            # subset rows
            row_idx = np.random.permutation(self.train.shape[0])
            subset = self.train[row_idx, :][:num_rows, :]
            # subset random number of columns
            col_idx = np.random.permutation(self.train.shape[1])
            subset = subset[:, col_idx]
            xs, ys = subset[:, 1:num_cols+1], subset[:, 0]
            ys = binarise_data(ys)

        elif self.split == "val":
            row_idx = np.random.permutation(self.predictors.shape[0])
            col_idx = np.random.permutation(self.predictors.shape[1])
            xs = self.predictors[:, col_idx][row_idx, :]
            xs = xs[:num_rows, :num_cols]
            ys = self.targets[row_idx][:num_rows]
            if max(ys) > 1:
                ys = one_vs_all(ys)
        
        if norm:
            m = xs.mean(0, keepdim=True)
            s = xs.std(0, unbiased=False, keepdim=True)
            xs -= m
            xs /= (s + 10e-4)

        return xs, ys.long()

class AllDatasetDataLoader():
    def __init__(
            self, bs, num_rows, num_targets, num_cols=-1, 
            device="cpu", split="train", norm=True):

        self.bs = bs
        self.num_rows = num_rows + num_targets
        self.split = split
        self.device = device
        self.num_cols = num_cols
        self.get_valid_datasets()
        self.norm = norm

    def get_valid_datasets(self):
        all_data_names = os.listdir(DATADIR)
        all_data_names.remove('info.json')
        self.all_datasets = [
                MyDataSet(d, split="val", device=self.device) 
                for d in all_data_names
            ]
        self.datasets = [
            d for d in self.all_datasets if d.num_total_rows >= self.num_rows
        ]
        if self.num_cols != -1:
            self.datasets = [
                d for d in self.datasets if d.num_predictors == self.num_cols
            ]
    
    def __iter__(self):
        """
        :return: [bs, num_rows, num_cols], [bs, num_rows, 1]
        """
        while True:
            datasets = random.sample(self.datasets, self.bs)
            datanames = [d.data_name for d in datasets]
            max_num_cols = min([d.num_predictors for d in datasets])
            if self.num_cols == -1:
                num_cols = np.random.randint(1, max_num_cols + 1)
            else:
                num_cols = self.num_cols
            xs, ys = list(zip(*[
                datasets[i].sample(
                    num_rows=self.num_rows, num_cols=num_cols, norm=self.norm) 
                for  i in range(self.bs)]))
            xs = torch.stack(xs)
            ys = torch.stack(ys)
            yield xs, ys, datanames

class SimpleDataset():
    def __init__(self, data_name, split="train"):
        self.split = split
        self.data_name = data_name
        self.X = pd.read_csv(
                f'{DATADIR}/{data_name}/{data_name}_py.dat', 
                header=None)
        self.y = pd.read_csv(
                f'{DATADIR}/{data_name}/labels_py.dat', 
                header=None)
        self.y = one_vs_all(pd.Series(self.y[0]))
        self.num_cols = self.X.shape[1]

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X, self.y, test_size=0.2, stratify=self.y, random_state=0)
        self.data_train  = pd.DataFrame(self.X_train)
        self.data_train['label'] = self.y_train
        self.data_val  = pd.DataFrame(self.X_val)
        self.data_val['label'] = self.y_val        
        self.X_train.drop('label', axis=1, inplace=True)
        self.X_val.drop('label', axis=1, inplace=True)

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
        # TO DO: handle imbalanced-classes better (do not sample with replacement)
        if self.split == "train":
            sample_data = self.data_train.groupby('label', group_keys=False).apply(
                lambda x: x.sample(num_rows // 2, replace=True)).sample(num_rows)
        else:
            sample_data = self.data_val.groupby('label', group_keys=False).apply(
                lambda x: x.sample(num_rows // 2, replace=True)).sample(num_rows)
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
    
    def _binary_vector(self, size, miss_rate):
        while True:
            vector = np.random.binomial(n=1, p=1-miss_rate, size=size)
            if vector.sum() > 0:
                return vector
    
    def get_missing(self, num_rows, miss_rate=0.5):
        # generate x number of missing patterns and sample from them
        num_total_rows = self.X_train.shape[0]
        num_patterns = (num_total_rows // num_rows) // 5
        patterns = [
            self._binary_vector(self.X_train.shape[1], miss_rate=miss_rate)
            for i in range(num_patterns)]
        print("Number of generated missing patterns:", num_patterns)
        self.train_mask = np.stack([
            patterns[np.random.randint(num_patterns)] 
            for i in range(self.X_train.shape[0])])
        self.val_mask = np.stack([
            patterns[np.random.randint(num_patterns)] 
            for i in range(self.X_val.shape[0])])

    def missing_sample(self, num_cols, num_rows, norm=True, shuffle_cols=False):
        if self.split == "train":
            data = self.data_train.copy()
            mask = self.train_mask
        elif self.split == "val":
            data = self.data_val.copy()
            mask = self.val_mask

        pattern = mask[mask.sum(axis=1) == num_cols, :]
        pattern = pattern[np.random.randint(pattern.shape[0]), :] # sample a pattern of missing columns
        row_idx = np.where((mask == pattern).sum(axis=1) == mask.shape[1])[0] # select all rows with this pattern
        col_idx = np.where(pattern == 1)[0]
        sample_data = data.iloc[row_idx, list(col_idx) + [-1]]
        sample_data = sample_data.groupby('label', group_keys=False).apply(
                lambda x: x.sample(num_rows // 2, replace=True)).sample(num_rows)
        xs, ys = sample_data.iloc[:, :-1], sample_data.iloc[:, -1]
        if shuffle_cols:
            xs = self._shuffle_cols(xs)
        xs = torch.tensor(xs.values, device='cpu', dtype=torch.float32)
        ys = torch.tensor(ys.values, device='cpu', dtype=torch.float32)
        if norm:
            xs = self._normalize(xs)
        return xs, ys, pattern


class DummyDataLoader():
    def __init__(self, bs, num_rows, num_targets, num_cols, fixed_num_cols=False,
                 split="train", norm=True, shuffle_cols=False, data_names=None):
        self.bs = bs
        self.num_rows = num_rows + num_targets
        self.num_cols = num_cols
        self.norm = norm
        self.shuffle_cols = shuffle_cols
        self.num_cols = num_cols
        self.fixed_num_cols = fixed_num_cols
        if data_names:
            self.data_names = data_names
            self.all_datasets = [SimpleDataset(d, split=split) for d in self.data_names]
        else:
            self.data_names = os.listdir(DATADIR)
            self.data_names.remove('info.json')
            self.all_datasets = [SimpleDataset(d, split=split) for d in self.data_names]    


    def _fix_col_sample(self):
        if isinstance(self.num_cols, list):
            num_cols = np.random.choice(self.num_cols)
        else:
            num_cols = self.num_cols
        self.datasets = [d for d in self.all_datasets if d.num_cols == num_cols]
        datasets = random.choices(self.datasets, k=self.bs)
        datanames = [d.data_name for d in datasets]
        xs, ys = list(zip(*[d.stratified_sample(
            self.num_rows, norm=self.norm, shuffle_cols=self.shuffle_cols) 
            for d in datasets]))
        xs = torch.stack(xs)
        ys = torch.stack(ys)
        return xs, ys.long(), datanames
    
    def _random_col_sample(self):
        num_data_cols = [d.num_cols for d in self.all_datasets]
        if self.num_cols:
            num_cols = np.random.choice(self.num_cols)
        else:
            min_col, max_col = 5, min(num_data_cols)
            num_cols = np.random.randint(min_col, max_col)

        datasets = random.choices(self.all_datasets, k=self.bs)
        datanames = [d.data_name for d in datasets]
        xs, ys = list(zip(*[d.stratified_sample(
            self.num_rows, norm=self.norm, shuffle_cols=self.shuffle_cols, num_cols=num_cols) 
            for d in datasets]))
        xs = torch.stack(xs)
        ys = torch.stack(ys)
        return xs, ys.long(), datanames

    def __iter__(self):
        """
        :return: [bs, num_rows, num_cols], [bs, num_rows, 1]
        """
        while True:
            if self.fixed_num_cols:
                yield self._fix_col_sample()
            else:
                yield self._random_col_sample()

    


class MissingDataLoader():
    def __init__(self, bs, num_rows, num_targets, data_name, miss_rate=0.5,
                 split="train", norm=True, shuffle_cols=True):
        self.bs = bs
        self.num_rows = num_rows + num_targets
        self.norm = norm
        self.dataset = SimpleDataset(data_name, split=split)
        self.dataset.get_missing(num_rows=self.num_rows, miss_rate=miss_rate)
        self.shuffle_cols = shuffle_cols
        self.split = split

    def __iter__(self):
        """
        :return: [bs, num_rows, num_cols], [bs, num_rows, 1]
        """
        while True:
            if self.split == "train":
                num_cols = np.random.choice(np.unique(self.dataset.train_mask.sum(axis=1)))
            else:
                num_cols = np.random.choice(np.unique(self.dataset.val_mask.sum(axis=1)))
            xs, ys, miss_patterns = list(zip(*[self.dataset.missing_sample(
                num_rows=self.num_rows, num_cols=num_cols, 
                norm=self.norm, shuffle_cols=self.shuffle_cols
            ) for i in range(self.bs)]))
            xs = torch.stack(xs)
            ys = torch.stack(ys)
            yield xs, ys.long(), miss_patterns

if __name__ == "__main__":
    # Test if dataloader works.
    np.random.seed(0)
    torch.manual_seed(0)

    dl = DummyDataLoader(
        bs=1, num_rows=5, num_targets=5, num_cols=None, 
        data_names=['oocytes_merluccius_nucleus_4d', 'oocytes_merluccius_states_2f'], 
        split="val", shuffle_cols=True, fixed_num_cols=False)
    # dl = MissingDataLoader(
    #     data_name='breast-cancer-wisc', bs=1, num_rows=5, num_targets=5, 
    #     miss_rate=0.5, split="val", norm=False, shuffle_cols=True
    # )

    for xs, ys, datanames in islice(dl, 10):
        print(xs)