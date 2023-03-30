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
        
        Combine data and label into single table to sample from. 
        
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
        datadir = f'./datasets/data/{self.data_name}'

        # Get train and test folds
        folds = pd.read_csv(f"{datadir}/folds_py.dat", header=None)[0]
        folds = np.asarray(folds)
        vldfold = pd.read_csv(f"{datadir}/validation_folds_py.dat", header=None)[0]
        vldfold = np.asarray(vldfold)

        # Read data and labels
        predictors = pd.read_csv(f"{datadir}/{self.data_name}_py.dat", header=None)
        predictors = np.asarray(predictors)
        targets = pd.read_csv(f"{datadir}/labels_py.dat", header=None)
        targets = np.asarray(targets)

        full_dataset = np.concatenate([predictors, targets], axis=-1)

        # Combine validation and test folds together
        data = {}
        data_train = full_dataset[(1 - folds) == 1 & (vldfold == 0)]
        data_valid = full_dataset[(vldfold == 1) | (folds == 1)]

        data["train"] = self._to_tensor(data_train)
        data["valid"] = self._to_tensor(data_valid)

        self.all_data = data

    def _to_tensor(self, array: np.array):
        return torch.from_numpy(array).to(self.device).to(self.dtype)

    def train(self, train: bool):
        if train:
            return self.all_data["train"]

        else:
            return self.all_data["valid"]


# Randomly samples from dataset.
class AdultDataLoader:
    def __init__(self, *, num_rows, bs, num_xs, train, device="cpu"):
        self.num_rows = num_rows
        self.train = train
        self.bs = bs
        self.num_xs = num_xs

        self.ds = Dataset("adult", device=device)

        self.data = self.ds.train(train)
        self.len = self.data.shape[0]
        self.cols = self.data.shape[1]

        # for i in range(len(self.data[0])):
        #     column = self.data[:, i]
        #
        #     unique_values = np.unique(column)
        #
        #     print(f"{i}, Number of unique values:", len(unique_values))

    # Pick out bs rows from dataset. Select 1 column to be target and random columns as predictors.
    # Only certain column can be targets since some are continuous.
    def __iter__(self):
        num_rows = self.num_rows
        permutation = torch.randperm(self.data.shape[0])
        data = self.data[permutation]

        allowed_targets = [9, 14]
        cols = np.arange(self.cols)

        for st in torch.arange(0, self.len - num_rows, num_rows):
            # Target
            target_col = np.random.choice(allowed_targets)
            target_col = torch.tensor([target_col])

            # Predictors
            predict_cols = np.setdiff1d(cols, target_col)
            predict_cols = np.random.choice(predict_cols, size=self.num_xs, replace=False)
            predict_cols = torch.from_numpy(predict_cols)

            # Rows of data to select
            selected_data = data[st:st + num_rows]

            xs = torch.index_select(selected_data, dim=1, index=predict_cols)
            ys = torch.index_select(selected_data, dim=1, index=target_col)

            yield xs, ys.int()


if __name__ == "__main__":
    dl = AdultDataLoader(num_rows=5, bs=5, num_xs=10, train=True)

    for _ in dl:
        print(_)
