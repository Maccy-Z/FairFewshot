# Make custom dataset folds.
import torch
import numpy as np
import pandas as pd
import random
import os
import csv

torch.manual_seed(0)


# Loads a single dataset. Split into train and valid.
def dataset(data_name, train_frac, val_frac):
    datadir = f'./data/{data_name}'

    # read predictors
    predictors = pd.read_csv(f"{datadir}/{data_name}_py.dat", header=None)
    predictors = np.asarray(predictors)
    # read internal target
    targets = pd.read_csv(f"{datadir}/labels_py.dat", header=None)
    targets = np.asarray(targets)

    # Combine predictors and target into single array, with targets as last column
    full_dataset = np.concatenate([predictors, targets], axis=-1)

    data_len = full_dataset.shape[0]
    print(f'Dataset length: {data_len}')

    train_num = int(data_len * train_frac)
    val_num = int(data_len * val_frac)
    test_num = data_len - train_num - val_num
    print("Data split:", train_num, val_num, test_num)

    # Randomly select portions of data to act as train, validation and test splits.
    rand_idx = np.random.choice(data_len, size=data_len, replace=False)
    train_idx, val_idx, test_idx = np.split(rand_idx, [train_num, val_num + train_num])

    data = {}
    data["train"] = full_dataset[train_idx]
    data["val"] = full_dataset[val_idx]
    data["test"] = full_dataset[test_idx]

    print()
    savedir = f'./data_split/{data_name}'
    print(f"Saving split files at '{savedir}/[train, val, test]'")
    assert os.path.exists(savedir)

    for split in ["train", "val", "test"]:
        with open(f'{savedir}/{split}', "w", newline='') as f:
            writer = csv.writer(f)
            for row in data[split]:
                writer.writerow(row)


if __name__ == "__main__":
    torch.manual_seed(0)

    dataset("adult", train_frac=0.7, val_frac=0.1)
