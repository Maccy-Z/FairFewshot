#%%
import torch
import numpy as np
import pandas as pd
import os
import random
import toml
from itertools import islice
import toml
import matplotlib.pyplot as plt
from config import Config

cfg = Config()
RNG = np.random.default_rng()

DATADIR = './datasets'

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

# Sample n items from k catagories. Return samples per catagory.
def sample(n, k):
    q, r = divmod(n, k)
    counts = [q+1]*r + [q]*(k-r)
    return counts


class MyDataSet:
    def __init__(
            self, ds_name, num_rows, num_targets, split,
            dtype=torch.float32, device="cpu"):
        self.ds_name = ds_name
        self.num_rows = num_rows
        self.num_targets = num_targets
        self.tot_rows = num_rows + num_targets

        self.device = device
        self.dtype = dtype

        self.train, self.valid, self.test = False, False, False

        print(ds_name, num_rows, num_targets, split)

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

        data = np.concatenate((preds, labels), axis=-1)
        data = to_tensor(data, device=device)
        self.tot_cols = data.shape[-1]
        self.tot_rows = data.shape[0]

        labels, position = np.unique(data[:, -1], return_inverse=True)
        labels = labels.astype(int)
        self.num_labels = len(labels)

        # Sort data by label
        self.data = {}
        for label in labels:
            mask = (position == label)
            label_data = data[mask]
            self.data[label] = label_data

            # Check dataset is large enough
            num_label_row = len(label_data)
            if cfg.min_row_per_label > num_label_row:
                raise ValueError(f'Not enough labels for class {label}, require {cfg.min_row_per_label}, has {num_label_row}')

        if self.tot_cols < cfg.min_cols:
            raise ValueError(f'Not enough columns. Require {cfg.min_cols}, has {self.tot_cols}')
        # # Binarise labels:
        # if self.binarise:
        #     self.data[:, -1] = one_vs_all(self.data[:, -1])
        #     # Sort based on label
        #     ones = (self.data[:, -1] == 1)
        #     self.ones = self.data[ones]
        #     self.zeros = self.data[torch.logical_not(ones)]
        #
        #     self.num_1s = self.ones.shape[0]
        #     self.num_0s = self.zeros.shape[0]
        #
        #     if self.num_0s < self.tot_rows or self.num_1s < self.tot_rows:
        #         print("WARN: Discarding dataset due to lack of labels", self.ds_name)
        #         self.ds_rows = 0
        # else:
        #     assert False
        #     # If one label makes up more than 50% of the column, downweight its sampling probability of category to 50%.
        #     row_probs = np.ones(self.ds_rows)
        #
        #     col_data = self.data[:, -1]
        #     unique_lab, unique_idx, counts = np.unique(
        #         col_data, return_counts=True, return_inverse=True)
        #     if np.max(counts) / self.ds_rows > 0.5:
        #         max_prob = self.ds_rows / np.max(counts) - 1
        #
        #         # Some cols have all entries identical.
        #         if max_prob == 0:
        #             pass
        #             # print(f"Warning: Entire column contains 1 unique entry, ds={self.data_name}, {col_no=}")
        #             # print(np.max(counts) / self.tot_rows)
        #         else:
        #             top_idx = (unique_idx == np.argmax(counts))
        #             row_probs[top_idx] = max_prob
        #
        #     row_probs = row_probs / np.sum(row_probs)
        #
        #     self.row_probs = row_probs.T


    def sample(self, num_cols):
        # Columns to sample from
        predict_cols = RNG.choice(self.tot_cols - 1, size=num_cols, replace=False)

        # Uniformly divide labels to fit n_meta / target.
        sample_meta = RNG.permutation(sample(cfg.N_meta, self.num_labels))
        sample_target = RNG.permutation(sample(cfg.N_target, self.num_labels))

        # Draw number of samples from each label.
        metas, targets = [], []
        for (label, label_rows), N_meta, N_target in zip(self.data.items(), sample_meta, sample_target, strict=True):
            # Draw rows and shuffle to make meta and target batch
            rows = RNG.choice(label_rows, size=N_meta + N_target, replace=False)

            meta_rows = rows[:N_meta]
            target_rows = rows[N_meta:]

            metas.append(meta_rows)
            targets.append(target_rows)


        print("ifdngiosdf")
        metas, targets = np.concatenate(metas), np.concatenate(targets)
        meta_pred, meta_label = metas[:, :-1], metas[:, -1]
        target_pred, target_label = targets[:, :-1], targets[:, -1]

        if cfg.normalise:
            all_data = np.concatenate([meta_pred, target_pred])
            mean, std = np.mean(all_data, axis=0), np.std(all_data, axis=0)
            meta_pred = (meta_pred - mean) / std
            target_pred = (target_pred - mean) / std

        return meta_pred, meta_label, target_pred, target_label


    def __repr__(self):
        return self.ds_name

    def __len__(self):
        return self.tot_rows


class SplitDataloader:
    def __init__(
            self, bs, num_meta, num_targets, num_cols=-1, ds_group=-1, ds_split="train", device="cpu",
            split_file='./datasets/grouped_datasets/splits',
            num_1s = None):
        """

        :param bs: Number of datasets to sample from each batch
        :param num_meta: Number of meta rows
        :param num_targets: Number of target rows
        :param num_cols: Number of columns to sample from.
            If 0, random no. columns between 2 and the largest number of 
                columns among all datasets
            If -1, random no. columns between 2 and the smallest number of 
                columns among all datasets
            If -2, sample datasets with equal probability, then sample valid number of columns.
            If -3, sample datasets with equal probability, take max allowed number of columns.
            If list, sample from a range of no. columns specified in the list
        :param ds_group: Which datasets to sample from.
            If None: All datasets
            If -1, sample all available datasets
            If strings, sample from that specified dataset(s).
        :param ds_split: If ds_group is int >= 0, the test or train split.
        """

        self.bs = bs
        self.num_meta = num_meta
        self.tot_rows = num_meta + num_targets
        self.num_rows = num_meta
        self.num_targets = num_targets
        self.num_cols = num_cols
        self.ds_group = ds_group
        self.ds_split = ds_split
        self.split_file = split_file

        self.device = device

        self._get_valid_datasets()


    def _get_valid_datasets(self):
        ds_dir = f'{DATADIR}/data/'
        # if isinstance(self.ds_group, tuple):
        #     fold_no, split_no = self.ds_group
        #     splits = toml.load(f'./datasets/grouped_datasets/splits_{fold_no}')
        #     if split_no == -1:
        #         get_splits = range(6)
        #     else:
        #         get_splits = [split_no]
        #
        #     ds_names = []
        #     for split in get_splits:
        #         names = splits[str(split)][self.ds_split]
        #         ds_names += names
        #
        # elif isinstance(self.ds_group, int):
        #     if self.ds_group == -1:
        #         # get all datasets
        #         ds_names = os.listdir(ds_dir)
        #         ds_names.remove('info.json')
        #         if '.DS_Store' in ds_names:
        #             ds_names.remove('.DS_Store')
        #     else:
        #         # get datasets from pre-defined split
        #         splits = toml.load(self.split_file)
        #         ds_names = splits[str(self.ds_group)][self.ds_split]

        if self.ds_group is None:
            ds_names = [f for f in os.listdir(ds_dir) if os.path.isdir(f'{ds_dir}/{f}')]
            print(ds_names)

        elif isinstance(self.ds_group, list):
            ds_names = self.ds_group
        else:
            raise Exception("Invalid ds_group")

        self.all_datasets = []
        for ds_name in ds_names:
            try:
                ds = MyDataSet(ds_name, num_rows=self.num_rows, num_targets=self.num_targets,
                        device=self.device, split="all")
                self.all_datasets.append(ds)

            except ValueError as e:
                print(f'Discarding dataset {ds_name}')
                print(e)

        if len(self.all_datasets) == 0:
            raise IndexError(f"No datasets with enough rows. Required: {self.tot_rows}")

        self.min_ds_cols = min([ds.tot_cols for ds in self.all_datasets])

    # def _check_num_cols(self):
    #     max_num_cols = max(self.num_cols)
    #     valid_datasets = [
    #         d for d in self.all_datasets if d.ds_cols > max_num_cols]
    #     if not valid_datasets:
    #         raise IndexError(
    #             "Provided range of columns to sample exceeds the "
    #             + "dimension of the largest dataset available" + f' {max_num_cols}')

    def __iter__(self):
        """
        :return: [bs, num_rows, num_cols], [bs, num_rows, 1]
        """
        while True:
            # Sample columns uniformly
            # if self.num_cols == 0 or self.num_cols == -1 or isinstance(self.num_cols, list):
            #     if isinstance(self.num_cols, int):
            #         if self.num_cols == 0:
            #             max_num_cols = max([d.ds_cols for d in self.all_datasets]) - 1
            #
            #         elif self.num_cols == -1:
            #             max_num_cols = min([d.ds_cols for d in self.all_datasets]) - 1
            #         num_cols_range = [2, max_num_cols]
            #
            #     else:
            #         num_cols_range = self.num_cols
            #
            #     if self.decrease_col_prob == -1:
            #         num_cols = np.random.choice(
            #             list(range(num_cols_range[0], num_cols_range[1] + 1)), size=1)[0]
            #     else:
            #         num_cols = np.random.geometric(p=self.decrease_col_prob, size=1) + 1
            #         num_cols = max(num_cols_range[0], num_cols)
            #         num_cols = min(num_cols, num_cols_range[1])
            #     valid_datasets = [d for d in self.all_datasets if d.ds_cols > num_cols]
            #     datasets = random.choices(valid_datasets, k=self.bs)

            # Sample datasets uniformly
            if cfg.col_fmt == 'uniform':
                datasets = RNG.choice(self.all_datasets, size=self.bs)
                max_num_cols = min([d.tot_cols for d in datasets]) - 1
                num_cols = RNG.integers(2, max_num_cols)

            elif cfg.col_fmt == 'all':
                datasets = RNG.choice(self.all_datasets, size=self.bs)
                num_cols = min([d.tot_cols for d in datasets]) - 1
            else:
                raise Exception("Invalid num_cols")

            datanames = [str(d) for d in datasets]


            meta_pred, meta_label, target_pred, target_label = list(zip(*[
                ds.sample(num_cols=num_cols) for ds in datasets]))
            meta_pred = torch.stack(meta_pred)
            meta_label = torch.stack(meta_label)
            target_pred = torch.stack(target_pred)
            target_label = torch.stack(target_label)
            yield xs, ys, datanames

    def __repr__(self):
        return str(self.all_datasets)


if __name__ == "__main__":
    torch.manual_seed(0)
    random.seed(0)

    dl = SplitDataloader(
        bs=1, num_meta=10, num_targets=10, num_cols="all", ds_group=['adult'], ds_split="train",
        num_1s={'meta': 1, 'target': 0}
    )

    for xs, ys, datanames in islice(dl, 10):
        print(ys[:, :5].sum(), ys[:, 5:].sum())