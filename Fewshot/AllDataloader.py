#%%
import torch
import numpy as np
import pandas as pd
import os
import toml
from itertools import islice
import toml
from config import Config

cfg = Config()
RNG = cfg.RNG

def d2v_pairer(xs, ys):
    #    # torch.Size([2, 5, 10]) torch.Size([2, 5, 1])
    bs, num_rows, num_xs = xs.shape

    xs = xs.reshape(bs * num_rows, num_xs)
    ys = ys.reshape(bs * num_rows, 1)

    pair_flat = torch.empty(bs * num_rows, num_xs, 2, device=xs.device)
    for k, (xs_k, ys_k) in enumerate(zip(xs, ys)):
        # Only allow 1D for ys
        ys_k = ys_k.repeat(num_xs)
        pairs = torch.stack([xs_k, ys_k], dim=-1)

        pair_flat[k] = pairs

    pairs = pair_flat.view(bs, num_rows, num_xs, 2)

    return pairs


def to_tensor(array: np.array, device=torch.device('cpu'), dtype=torch.float32):
    return torch.from_numpy(array).to(device).to(dtype)


# Sample n items from k catagories. Return samples per catagory.
def sample(n, k):
    q, r = divmod(n, k)
    counts = [q+1]*r + [q]*(k-r)
    return counts


class MyDataSet:
    def __init__(self, ds_name, split, dtype=torch.float32, device="cpu"):
        self.ds_name = ds_name

        self.device = device
        self.dtype = dtype

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

        ds_dir = f'{cfg.DS_DIR}/data'
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
        elif split == "val":
            idx = (vldfold == 1)
        elif split == "test":
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


    def sample(self, num_cols):
        # Columns to sample from
        pred_cols = RNG.choice(self.tot_cols - 1, size=num_cols, replace=False)

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


        metas, targets = np.concatenate(metas), np.concatenate(targets)

        meta_label, target_label = metas[:, -1], targets[:, -1]
        meta_pred, target_pred= metas[:, pred_cols], targets[:, pred_cols]

        if cfg.normalise:
            all_data = np.concatenate([meta_pred, target_pred])
            mean, std = np.mean(all_data, axis=0), np.std(all_data, axis=0)
            meta_pred = (meta_pred - mean) /(std + 1e-8)
            target_pred = (target_pred - mean) / (std + 1e-8)

        return meta_pred, meta_label, target_pred, target_label


    def __repr__(self):
        return self.ds_name

    def __len__(self):
        return self.tot_rows


class SplitDataloader:
    def __init__(
            self, bs, ds_group=-1, ds_split="train", device="cpu",
            split_file='./datasets/grouped_datasets/splits'):
        """

        :param bs: Number of datasets to sample from each batch
        :param ds_group: Which datasets to sample from.
            If None: All datasets
            If -1, sample all available datasets
            If strings, sample from that specified dataset(s).
        :param ds_split: If ds_group is int >= 0, the test or train split.
        """

        self.bs = bs
        self.ds_group = ds_group
        self.ds_split = ds_split
        self.split_file = split_file

        self.device = device

        ds_dir = f'{cfg.DS_DIR}/data/'

        # All datasets
        if self.ds_group is None:
            ds_names = [f for f in os.listdir(ds_dir) if os.path.isdir(f'{ds_dir}/{f}')]
            print(ds_names)
        # Specific datasets
        elif isinstance(self.ds_group, list):
            ds_names = self.ds_group
        # Premade splits
        elif isinstance(ds_group, str):
            with open(f'{cfg.DS_DIR}/splits/{ds_group}', "r") as f:
                ds_names = toml.load(f)[ds_split]
        else:
            raise Exception("Invalid ds_group")

        self.all_datasets = []
        for ds_name in ds_names:
            try:
                ds = MyDataSet(ds_name, device=self.device, split="all")
                self.all_datasets.append(ds)

            except ValueError as e:
                print(f'Discarding dataset {ds_name}')
                print(e)

        if len(self.all_datasets) == 0:
            raise IndexError(f"No datasets with enough rows")

        self.min_ds_cols = min([ds.tot_cols for ds in self.all_datasets])

    def __iter__(self):
        """
        :return: [bs, num_rows, num_cols], [bs, num_rows, 1]
        """
        while True:

            # Number of columns to sample dataset. Testing always uses full dataset
            if cfg.col_fmt == 'all' or self.ds_split == "test":
                datasets = RNG.choice(self.all_datasets, size=self.bs)  # Allow repeats.
                num_cols = min([d.tot_cols for d in datasets]) - 1

            elif cfg.col_fmt == 'uniform':
                datasets = RNG.choice(self.all_datasets, size=self.bs)  # Allow repeats.
                max_num_cols = min([d.tot_cols for d in datasets]) - 1
                num_cols = RNG.integers(2, max_num_cols)

            else:
                raise Exception("Invalid num_cols")

            meta_pred, meta_label, target_pred, target_label = list(zip(*[
                ds.sample(num_cols=num_cols) for ds in datasets]))

            meta_pred, meta_label = np.stack(meta_pred), np.stack(meta_label)
            target_pred, target_label = np.stack(target_pred), np.stack(target_label)


            xs_meta, xs_target = to_tensor(meta_pred), to_tensor(target_pred)
            ys_meta, ys_target = to_tensor(meta_label, dtype=torch.int64), to_tensor(target_label, dtype=torch.int64)

            # Get maximum number of labels in batch
            max_N_label = max([d.num_labels for d in datasets]) + 1

            yield xs_meta, ys_meta, xs_target, ys_target, max_N_label

    def __repr__(self):
        return str(self.all_datasets)


if __name__ == "__main__":
    torch.manual_seed(0)

    dl = SplitDataloader(
        bs=2, ds_group="0", ds_split="train")

    for mp, ml, tp, tl, datanames in islice(dl, 10):
        print(mp.shape, ml.shape)
        print(tp.shape, tl.shape)
        print()