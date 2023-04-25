import torch
import numpy as np
import pandas as pd
import os
import random
import toml

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
    def __init__(self, ds_name, num_rows, num_targets, binarise, split, dtype=torch.float32, device="cpu"):
        # data_name = "adult"
        self.ds_name = ds_name
        self.num_rows = num_rows
        self.num_targets = num_targets
        self.tot_rows = num_rows + num_targets
        self.binarise = binarise

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
        folds = pd.read_csv(f"{ds_dir}/{ds_name}/folds_py.dat", header=None)[0]
        folds = np.asarray(folds)
        # get validation fold
        vldfold = pd.read_csv(f"{ds_dir}/{ds_name}/validation_folds_py.dat", header=None)[0]
        vldfold = np.asarray(vldfold)

        # read predictors
        predictors = pd.read_csv(f"{ds_dir}/{ds_name}/{self.ds_name}_py.dat", header=None)
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

        # Binarise labels:
        if self.binarise:
            self.data[:, -1] = one_vs_all(self.data[:, -1])
            # Sort based on label
            ones = (self.data[:, -1] == 1)
            self.ones = self.data[ones]
            self.zeros = self.data[torch.logical_not(ones)]

            self.num_1s = self.ones.shape[0]
            self.num_0s = self.zeros.shape[0]

            if self.num_0s < self.tot_rows or self.num_1s < self.tot_rows:
                print("WARN: Discarding dataset due to lack of labels", self.ds_name)
                self.ds_rows = 0
        else:

            # If one label makes up more than 50% of the column, downweight its sampling probability of category to 50%.
            row_probs = np.ones(self.ds_rows)

            col_data = self.data[:, -1]
            unique_lab, unique_idx, counts = np.unique(col_data, return_counts=True, return_inverse=True)
            if np.max(counts) / self.ds_rows > 0.5:
                max_prob = self.ds_rows / np.max(counts) - 1

                # Some cols have all entries identical.
                if max_prob == 0:
                    pass
                    # print(f"Warning: Entire column contains 1 unique entry, ds={self.data_name}, {col_no=}")
                    # print(np.max(counts) / self.tot_rows)
                else:
                    top_idx = (unique_idx == np.argmax(counts))
                    row_probs[top_idx] = max_prob

            row_probs = row_probs / np.sum(row_probs)

            self.row_probs = row_probs.T


    def sample(self, num_cols):
        if num_cols > self.ds_cols - 1:  # or not self.train:
            num_cols = self.ds_cols - 1

        targ_col = -1
        predict_cols = np.random.choice(self.ds_cols - 1, size=num_cols, replace=False)
        if self.binarise:
            # Select meta and target rows separately. Pick number of 1s from binomial then sample without replacement.
            # TODO: This also allows for fixing number of meta / target easily.
            meta_1s = np.random.binomial(self.num_rows, 0.5)
            target_1s = np.random.binomial(self.num_targets, 0.5)

            meta_0s, target_0s = self.num_rows - meta_1s, self.num_targets - target_1s

            meta_0s_row = np.random.choice(self.num_0s, size=meta_0s, replace=False)
            meta_1s_row = np.random.choice(self.num_1s, size=meta_1s, replace=False)

            rem_0 = np.setdiff1d(np.arange(self.num_0s), meta_0s_row)
            rem_1 = np.setdiff1d(np.arange(self.num_1s), meta_1s_row)
            targ_0s_row = np.random.choice(rem_0, size=target_0s, replace=False)
            targ_1s_row = np.random.choice(rem_1, size=target_1s, replace=False)

            meta_rows = torch.concatenate((self.zeros[meta_0s_row], self.ones[meta_1s_row]))
            targ_rows = torch.concatenate((self.zeros[targ_0s_row], self.ones[targ_1s_row]))

            # Join meta and target. Split apart later.
            select_data = torch.concatenate([meta_rows, targ_rows])

        else:
            rows = np.random.choice(self.ds_rows, size=self.tot_rows, replace=False, p=self.row_probs)
            select_data = self.data[rows]

        # Pick out wanted columns
        xs = select_data[:, predict_cols]
        ys = select_data[:, targ_col]

        # Normalise xs
        m = xs.mean(0, keepdim=True)
        s = xs.std(0, unbiased=False, keepdim=True)
        xs -= m
        xs /= (s + 10e-4)
        if not self.binarise:
            ys = one_vs_all(ys)
        return xs, ys


    def __repr__(self):
        return self.ds_name

    def __len__(self):
        return self.ds_rows


class SplitDataloader:
    def __init__(self, bs, num_rows, num_targets, binarise=False, num_cols=-1, get_ds=-1, split="train", device="cpu"):
        """

        :param bs: Number of datasets to sample from each batch
        :param num_rows: Number of meta rows
        :param num_targets: Number of target rows
        :param binarise: Binarise the dataset upon loading instead of with each sample
        :param num_cols: Number of columns to sample from. -1 for all, 0 for random.
        :param get_ds: Which datasets to sample from. If int, referes to premade group. If string or list of string,
                        sample from that specific dataset(s).
        :param split: If ds_group is int, the test or train split.
        """

        self.bs = bs
        self.tot_rows = num_rows + num_targets
        self.num_rows = num_rows
        self.num_targets = num_targets
        self.binarise = binarise
        self.num_cols = num_cols
        self.get_ds = get_ds

        self.device = device

        self.split = split

        self.get_valid_datasets()


    def get_valid_datasets(self):
        ds_dir = f'{DATADIR}/grouped_datasets/'
        ds_to_get = None
        if isinstance(self.get_ds, int):
            splits = toml.load(f'{ds_dir}/splits')
            if self.get_ds == -1:
                get_splits = sorted([f for f in os.listdir(ds_dir) if os.path.isdir(f'{ds_dir}/{f}')])
            else:
                get_splits = [str(self.get_ds)]

            ds_to_get = []
            for split in get_splits:
                ds_names = splits[str(split)][self.split]
                ds_to_get += ds_names

        elif isinstance(self.get_ds, str):
            ds_to_get = [self.get_ds]

        elif isinstance(self.get_ds, list):
            ds_to_get = self.get_ds

        all_datasets = []
        for name in ds_to_get:
            ds = MyDataSet(name, num_rows=self.num_rows, num_targets=self.num_targets,
                           binarise=self.binarise, device=self.device, split="all",)
            all_datasets.append(ds)

        min_ds = self.tot_rows * 2

        self.datasets = [
            d for d in all_datasets if len(d) >= min_ds
        ]

        self.max_cols = min([d.ds_cols for d in self.datasets]) - 1



    def __iter__(self):
        """
        :return: [bs, num_rows, num_cols], [bs, num_rows, 1]
        """
        while True:
            datasets = random.choices(self.datasets, k=self.bs)
            datanames = [str(d) for d in datasets]

            max_num_cols = min([d.ds_cols for d in datasets]) - 1
            if self.num_cols == -1:
                num_cols = max_num_cols
            elif self.num_cols == 0:
                num_cols = np.random.randint(2, max_num_cols)
            else:
                num_cols = min(self.num_cols, max_num_cols)

            xs, ys = list(zip(*[
                datasets[i].sample(num_cols=num_cols)
                for i in range(self.bs)]))
            xs = torch.stack(xs)
            ys = torch.stack(ys)
            yield xs, ys, datanames

    def __repr__(self):
        return str(self.datasets)



if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)

    dl = SplitDataloader(bs=5, num_rows=16, binarise=False, num_targets=3, num_cols=0, get_ds=["adult", "car"], split="train")
    # dl = SingleDataloader(ds_name="adult", bs=1, num_rows=16, num_targets=3, num_cols=-1, ds_group=4, one_v_all=True, split="train")

    means = []
    dl = iter(dl)
    # y_count = {i: 0 for i in range(20)}
    for _ in range(1000):
        x, y, _ = next(dl)
        print(x.shape)
        num = torch.sum(y).item()
       # y_count[num] += 1

        y_mean = torch.mean(y, dtype=float)
        means.append(y_mean)

    means = torch.stack(means)
    means = torch.mean(means)
    print("Mean y value", f'{means.item():.4g}')
    # print("Histogram of number of positive samples", y_count)

# {0: 14, 1: 10, 2: 13, 3: 26, 4: 33, 5: 76, 6: 348, 7: 311, 8: 67, 9: 46, 10: 12, 11: 7, 12: 7, 13: 30}
# {0: 28, 1: 21, 2: 22, 3: 23, 4: 36, 5: 52, 6: 313, 7: 343, 8: 49, 9: 37, 10: 17, 11: 10, 12: 9, 13: 40}
