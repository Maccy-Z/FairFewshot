from sklearn import datasets
from matplotlib import pyplot as plt
import torch
import numpy as np
import random

random.seed(0)


class ToyDataset:
    def __init__(self, ds_type: str, nsamples=200):
        self.nsamples = nsamples
        self.ds_type = ds_type

        if ds_type == "moons":
            ds_fn = datasets.make_moons
        elif ds_type == "blobs":
            ds_fn = datasets.make_blobs
        elif ds_type == "circles":
            ds_fn = datasets.make_circles
        else:
            assert 0

        self.ds_fn = lambda: ds_fn(nsamples)

    def get_data(self, split=None):
        xs, ys = self.ds_fn()
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

    def plot_ds(self):
        x, y = self.ds_fn()
        # fn = lambda x: datasets.make_blobs(200)
        # x, y = fn(0)

        y = ["red" if i == 0 else "blue" for i in y]

        print(y)
        plt.scatter(x[:, 0], x[:, 1], c=y)

        plt.show()

    def __repr__(self):
        return f'Ds: {self.ds_type}'


class ToyDataloader:
    def __init__(self, bs=6, repeat_frac=1 / 2, steps=5000):
        """
        :param bs: Number of datasets to sample from
        :param repeat_frac: 1 / number of times to repeat datasets.
        """
        self.bs = bs
        self.repeat_frac = repeat_frac
        self.steps = steps

        ds = ["blobs", "circles", "moons"]
        self.num_ds = len(ds)

        self.ds = [ToyDataset(name) for name in ds]

        # Sanity check that bs and repeat_frac are possible
        assert (bs * repeat_frac).is_integer()
        if self.num_ds / self.repeat_frac < self.bs:
            # Requested too large of a batch, not enough datasets for given repeat fraction.
            print("Requested too large of a batch, not enough datasets for given repeat fraction.")
            assert 0

        self.num_samples = int(self.bs * self.repeat_frac)

        if self.num_samples > self.num_ds:
            self.num_samples = self.num_ds

    # Sample items from datasets. Make sure there are always repeated datasets.
    def __iter__(self):
        for _ in range(self.steps):
            chosen_ds = random.sample(self.ds, self.num_samples) * int(1 / self.repeat_frac)
            # print(chosen_ds)
            labels = list(range(self.num_samples)) * int(1 / self.repeat_frac)

            meta_dataset = [ds.get_data() for ds in chosen_ds]
            meta_dataset = torch.stack(meta_dataset)

            yield meta_dataset, torch.tensor(labels)


if __name__ == "__main__":
    dl = ToyDataloader(bs=9, repeat_frac=1/3)
    for i in dl:
        pass
        # print(i)
