import torch
import numpy as np
import pandas as pd
import random

torch.manual_seed(0)


class Dataset:
    def __init__(self, data_name, nsamples=100):
        self.nsamples = nsamples
        self.data_name = data_name

        self.all_data, self.all_labels = self.init_data()
        self.train(True)

    def init_data(self):
        """
        Dataset format: {folder}_py.dat             predictors
                        labels_py.dat               labels for predictors
                        folds_py.dat                test fold
                        validation_folds_py.dat     validation fold

        """
        datadir = f'./datasets/data/{self.data_name}'

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
        #data.update({'test': predictors[folds == 1]})
        data.update({'valid': predictors[(vldfold == 1) | (folds == 1)]})

        # get label folds
        labels = {}
        labels.update({'train': targets[(1 - folds) == 1 & (vldfold == 0)]})
        #labels.update({'test': targets[folds == 1]})
        labels.update({'valid': targets[(vldfold == 1) | (folds == 1)]})

        return data, labels

    def get_data(self):
        want_idx = torch.randperm(self.data_len)[:self.nsamples]
        xs = self.data[want_idx]
        ys = self.labels[want_idx]

        xs = np.reshape(xs, [self.nsamples, -1])  # = [num_samples, xdim]
        ys = np.reshape(ys, [self.nsamples, -1])  # = [num_samples, ydim]
        xdim, ydim = xs.shape[-1], ys.shape[-1]

        if self.training:
            num_xdim = np.random.randint(xdim) + 1
            num_ydim = np.random.randint(ydim) + 1

            x_idx = np.random.choice(np.arange(xdim), size=num_xdim, replace=False)
            y_idx = np.random.choice(np.arange(ydim), size=num_ydim, replace=False)

            xs = xs[:, x_idx]
            ys = ys[:, y_idx]

        # Turn data from table of x and table of y to all paris of (x, y)
        pair_output = []
        for k in range(self.nsamples):
            xs_k, ys_k = xs[k], ys[k]
            pairs = np.transpose([np.repeat(xs_k, len(ys_k)), np.tile(ys_k, len(xs_k))])
            pairs = pairs.reshape(len(xs_k), len(ys_k), 2)

            pair_output.append(pairs)

        pair_output = np.stack(pair_output)
        pair_output = torch.tensor(pair_output).permute(1, 2, 0, 3).to(torch.float32)

        return pair_output

    def __repr__(self):
        return f'Ds: {self.data_name}'

    def train(self, train: bool):
        if train:
            self.data = self.all_data["train"]
            self.labels = self.all_labels["train"]

            self.training = True
        else:
            self.data = self.all_data["valid"]
            self.labels = self.all_labels["valid"]

            self.training = False

        self.data_len = self.data.shape[0]
        if self.data_len < self.nsamples:
            self.nsamples = self.data_len


class Dataloader:
    def __init__(self, bs=6, bs_num_ds=3, steps=5000, device="cpu", split="train"):
        """
        :param bs: Number of datasets to sample from
        :param bs_num_ds: Number of uniqie datasets to sample from.
        """
        self.bs = bs
        self.steps = steps
        self.device = device
        self.bs_num_ds = bs_num_ds

        ds = ["adult", "car", "blood", "chess-krvk", "bank", "ionosphere", "magic", "musk-1",
              "optical", "titanic", "ecoli", "thyroid", "waveform"]
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

            yield meta_dataset, torch.tensor(labels)


    def train(self, train):
        for d in self.ds:
            d.train(train)


if __name__ == "__main__":

    dl = Dataloader()
    dl.split = "train"

    import time

    st = time.time()
    for ia, (data, labels) in enumerate(dl):
        if ia % 10 == 0:
            print(time.time() - st)
            st = time.time()
