import torch
import numpy as np
import pandas as pd
import random
import json
from scipy import stats

torch.manual_seed(0)

DATADIR = './datasets/data'

class Dataset:
    def __init__(self, data_name):
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

        self.train_data = pd.read_csv(
            f'{DATADIR}/{self.data_name}/{self.data_name}_train_py.dat'
        )
        self.val_data = pd.read_csv(
            f'{DATADIR}/{self.data_name}/{self.data_name}_val_py.dat'
        )
        self.test_data = pd.read_csv(
            f'{DATADIR}/{self.data_name}/{self.data_name}_test_py.dat'
        )

        data = {
            'train': self.train_data.drop('label', axis=1).values,
            'val': self.val_data.drop('label', axis=1).values,
            'test': self.test_data.drop('label', axis=1).values
        }

        labels = {
            'train': self.train_data['label'].values,
            'val': self.val_data['label'].values,
            'test': self.test_data['label'].values
        }

        return data, labels

    def get_data(self, nsamples):
        nsamples = min(nsamples, self.data_len)

        want_idx = torch.randperm(self.data_len)[:nsamples]
        xs = self.data[want_idx]
        ys = self.labels[want_idx]

        xs = np.reshape(xs, [nsamples, -1])  # = [num_samples, xdim]
        ys = np.reshape(ys, [nsamples, -1])  # = [num_samples, ydim]
        xdim, ydim = xs.shape[-1], ys.shape[-1]

        if self.training:
            num_xdim = max(np.random.randint(xdim) + 1, 2)
            num_ydim = np.random.randint(ydim) + 1

            x_idx = np.random.choice(np.arange(xdim), size=num_xdim, replace=False)
            y_idx = np.random.choice(np.arange(ydim), size=num_ydim, replace=False)

            xs = xs[:, x_idx]
            ys = ys[:, y_idx]

        # Normalise batch
        m = xs.mean(0)
        s = xs.std(0)
        xs -= m
        xs /= (s + 10e-4)


        # Turn data from table of x and table of y to all paris of (x, y)
        pair_output = []
        for k in range(nsamples):
            xs_k, ys_k = xs[k], ys[k]
            pairs = np.transpose([np.repeat(xs_k, len(ys_k)), np.tile(ys_k, len(xs_k))])
            pairs = pairs.reshape(len(xs_k), len(ys_k), 2)

            pair_output.append(pairs)

        pair_output = np.stack(pair_output)
        pair_output = torch.from_numpy(pair_output).permute(1, 2, 0, 3).to(torch.float32)

        return pair_output

    def __repr__(self):
        return f'Ds: {self.data_name}'

    def train(self, train: bool):
        if train:
            self.data = self.all_data["train"]
            self.labels = self.all_labels["train"]

            self.training = True
        else:
            self.data = self.all_data["val"]
            self.labels = self.all_labels["val"]

            self.training = False

        self.data_len = self.data.shape[0]


class Dataloader:
    def __init__(self, bs=6, bs_num_ds=3, steps=5000, nsamples=10, min_ds_len = 500, device="cpu", split="train"):
        """
        :param bs: Number of datasets to sample from
        :param bs_num_ds: Number of uniqie datasets to sample from.
        """
        self.bs = bs
        self.steps = steps
        self.device = device
        self.bs_num_ds = bs_num_ds
        self.nsamples = nsamples

        # # ds = ["adult", "car", "blood", "chess-krvk", "bank", "ionosphere", "magic", "musk-1",
        # #       "optical", "titanic", "ecoli", "thyroid", "waveform", "nursery", "musk-2", "pendigits",
        # #       "mushroom", "miniboone", "ringnorm", "twonorm", "statlog-shuttle"]
        # ds = ['molec-biol-splice', 'twonorm', 'plant-texture', 'ringnorm', 'steel-plates', 'chess-krvk', 'statlog-shuttle', 'semeion', 'connect-4', 'wall-following', 'cardiotocography-3clases', 'plant-margin', 'nursery', 'titanic', 'tic-tac-toe', 'waveform', 'wine-quality-red', 'wine-quality-white', 'spambase', 'thyroid', 'mammographic', 'waveform-noise', 'letter', 'yeast', 'adult', 'annealing', 'contrac', 'statlog-landsat', 'musk-2', 'abalone', 'statlog-vehicle', 'page-blocks', 'plant-shape', 'bank', 'pendigits', 'mushroom', 'optical', 'oocytes_merluccius_states_2f', 'chess-krvkp', 'led-display', 'oocytes_trisopterus_nucleus_2f', 'statlog-german-credit', 'car', 'oocytes_trisopterus_states_5b', 'oocytes_merluccius_nucleus_4d', 'ozone', 'magic', 'statlog-image', 'cardiotocography-10clases', 'miniboone']

        with open("./datasets/data/info.json") as f:
            json_data = json.load(f)

        dataset_lengths = {}
        for k, v in json_data.items():
            length = int(v['cardinality']['train'])
            if length > min_ds_len:
                dataset_lengths[k] = v['cardinality']['train']
        ds = dataset_lengths.keys()


        self.num_ds = len(ds)

        self.ds = [Dataset(data_name=name) for name in ds]

        # Ensure number of datasets can fit into batch exactly
        self.num_repeat, remainder = divmod(bs, bs_num_ds)
        assert remainder == 0

    def __iter__(self):
        for _ in range(self.steps):
            chosen_ds = random.sample(self.ds, self.bs_num_ds) * self.num_repeat

            labels = list(range(self.bs_num_ds)) * self.num_repeat

            meta_dataset = [ds.get_data(nsamples=self.nsamples).to(self.device) for ds in chosen_ds]

            yield meta_dataset, torch.tensor(labels)

    def train(self, train):
        if train:
            self.nsamples = 10
        else:
            self.nsampels = 10
        # self.nsamples = 1000
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
