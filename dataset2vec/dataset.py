import torch
import numpy as np
import pandas as pd
import random

torch.manual_seed(0)


class Dataset:
    def __init__(self, data_name, nsamples=200):
        self.nsamples = nsamples
        self.data_name = data_name

        self.data, self.labels = self.init_data()

        self.data_len = self.data["train"].shape[0]

        # TODO: implement for varying nsamples
        assert self.data_len > self.nsamples

    def init_data(self):
        """
        Dataset format: {folder}_py.dat             predictors
                        labels_py.dat               labels for predictors
                        folds_py.dat                test fold
                        validation_folds_py.dat     validation fold

        """
        datadir = f'./datasets/{self.data_name}'

        # get train fold
        folds = pd.read_csv(f"{datadir}/folds_py.dat", header=None)[0]
        folds = np.asarray(folds)
        # get validation fold
        vldfold = pd.read_csv(f"{datadir}/validation_folds_py.dat", header=None)[0]
        vldfold = np.asarray(vldfold)

        # read predictors
        predictors = pd.read_csv(f"{datadir}/{self.data_name}_py.dat", header=None)
        predictors = np.asarray(predictors)

        # read internal target
        targets = pd.read_csv(f"{datadir}/labels_py.dat", header=None)
        targets = np.asarray(targets)

        # get data folds
        data = {}
        data.update({'train': predictors[(1 - folds) == 1 & (vldfold == 0)]})
        data.update({'test': predictors[folds == 1]})
        data.update({'valid': predictors[vldfold == 1]})

        # get label folds
        labels = {}
        labels.update({'train': targets[(1 - folds) == 1 & (vldfold == 0)]})
        labels.update({'test': targets[folds == 1]})
        labels.update({'valid': targets[vldfold == 1]})

        return data, labels

    def get_data(self, split="train"):
        want_idx = torch.randperm(self.data_len)[:self.nsamples]
        xs = self.data[split][want_idx]
        ys = self.labels[split][want_idx]

        xs = np.reshape(xs, [self.nsamples, -1])  # = [num_samples, xdim]
        ys = np.reshape(ys, [self.nsamples, -1])  # = [num_samples, ydim]
        xdim, ydim = xs.shape[-1], ys.shape[-1]

        # Turn data from table of x and table of y to all paris of (x, y)
        pair_output = torch.zeros([xdim, ydim, self.nsamples, 2])

        for k in range(self.nsamples):
            for i, x in enumerate(xs[k]):
                for j, y in enumerate(ys[k]):
                    pair_output[i, j, k] = torch.tensor([x, y])

        return pair_output

    def __repr__(self):
        return f'Ds: {self.data_name}'


class Dataloader:
    def __init__(self, bs=6, bs_num_ds=3, steps=5000, device="cpu"):
        """
        :param bs: Number of datasets to sample from
        :param bs_num_ds: Number of uniqie datasets to sample from.
        """
        self.bs = bs
        self.steps = steps
        self.device = device
        self.bs_num_ds = bs_num_ds

        ds = ["adult", "car", "blood", "chess-krvk", "bank", "ionosphere"]
        self.num_ds = len(ds)

        self.ds = [Dataset(data_name=name) for name in ds]

        # Ensure number of datasets can fit into batch exactly
        self.num_repeat, remainder = divmod(bs, bs_num_ds)
        assert remainder == 0

    def __iter__(self):
        for _ in range(self.steps):
            chosen_ds = random.sample(self.ds, self.bs_num_ds) * self.num_repeat
            # print(chosen_ds)
            labels = list(range(self.bs_num_ds)) * self.num_repeat

            meta_dataset = [ds.get_data().to(self.device) for ds in chosen_ds]
            # meta_dataset = torch.stack(meta_dataset)

            yield meta_dataset, torch.tensor(labels)


if __name__ == "__main__":
    ds = Dataloader()

    for _ in ds:
        print(_)
