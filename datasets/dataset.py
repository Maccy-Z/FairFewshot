# Loads datasets.
import torch
import numpy as np
import pandas as pd
import random

torch.manual_seed(0)

# Loads a single dataset. Split into train and valid.
class Dataset:
    def __init__(self, data_name, dtype=torch.float32, device="cpu"):
        self.data_name = data_name
        self.device = device
        self.dtype = dtype
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

        # Combine validation and test folds together
        data, labels = {}, {}
        pred_train = predictors[(1 - folds) == 1 & (vldfold == 0)]
        pred_valid = predictors[(vldfold == 1) | (folds == 1)]
        data["train"] = self._to_tensor(pred_train)
        data["valid"] = self._to_tensor(pred_valid)

        # get label folds
        labels_train = targets[(1 - folds) == 1 & (vldfold == 0)]
        labels_valid = targets[(vldfold == 1) | (folds == 1)]
        labels["train"] = self._to_tensor(labels_train)
        labels["valid"] = self._to_tensor(labels_valid)

        self.all_data, self.all_labels = data, labels

        self.train(True)

    def _to_tensor(self, array: np.array):
        return torch.from_numpy(array).to(self.device).to(self.dtype)

    def train(self, train: bool):
        if train:
            data = self.all_data["train"]
            labels = self.all_labels["train"]

        else:
            data = self.all_data["valid"]
            labels = self.all_labels["valid"]

        return data, labels


# Randomly samples from dataset.
class DataLoader:
    def __init__(self, ds_name, *, bs, train, device="cpu"):
        self.ds_name = ds_name
        self.bs = bs
        self.train = train

        self.ds = Dataset(ds_name, device=device)
        self.data, self.labels = self.ds.train(train)
        self.len = self.labels.shape[0]

    def __iter__(self):
        bs = self.bs
        permutation = torch.randperm(self.labels.shape[0])

        data, labels = self.data[permutation], self.labels[permutation]
        for st in torch.arange(0, self.len - bs, bs):
            xs = data[st:st + bs]
            ys = labels[st:st + bs]

            yield xs, ys


if __name__ == "__main__":
    import json

    torch.manual_seed(0)

    with open("./data/info.json") as f:
        json_data = json.load(f)

    dataset_lengths = {}
    for k, v in json_data.items():
        length = int(v['cardinality']['train'])
        if length > 1000:
            dataset_lengths[k] = v['cardinality']['train']

    print(dataset_lengths)
    print("Long datasets:", len(dataset_lengths))
    print()
    dl = DataLoader("statlog-shuttle", bs=1, train=True)

    for xs, ys in dl:
        print(xs.shape, ys.shape)
        break
