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
    datadir = f'.'

    # read predictors
    data = pd.read_csv(f"{datadir}/adult.data", header=None)
    data = np.asarray(data)

    num_rows, num_cols = data.shape[0], data.shape[1]
    print(f'Dataset length: {num_rows}')

    col_classes = [set() for _ in range(num_cols)]
    for row in data:
        for classes, col in zip(col_classes, row):
            classes.add(col)

    # Quantise classes
    col_maps = []
    discrete_cols = []
    for i, classes in enumerate(col_classes):
        classes = list(classes)

        if isinstance(classes[0], int):
            classes = sorted(classes)
            col_map = {v: v for v in classes}
        else:
            discrete_cols.append(i)
            col_map = {k: v for k, v in zip(classes, range(len(classes)))}

        col_maps.append(col_map)
    print("Discrete columns:", discrete_cols)

    quantised_data = []
    for row in data:
        new_row = []
        for col_map, col in zip(col_maps, row):
            new_row.append(col_map[col])

        quantised_data.append(new_row)

    data = np.array(quantised_data)
    means, std = np.mean(data, axis=0), np.std(data, axis=0)
    data = data - means
    data = data / std

    train_num = int(num_rows * train_frac)
    val_num = int(num_rows * val_frac)
    test_num = num_rows - train_num - val_num
    print("Data split:", train_num, val_num, test_num)

    # Randomly select portions of data to act as train, validation and test splits.
    rand_idx = np.random.choice(num_rows, size=num_rows, replace=False)
    train_idx, val_idx, test_idx = np.split(rand_idx, [train_num, val_num + train_num])
    data_split = {}
    data_split["train"] = data[train_idx]
    data_split["val"] = data[val_idx]
    data_split["test"] = data[test_idx]

    print()
    savedir = f'.'
    print(f"Saving split files at '{savedir}/[train, val, test]'")
    assert os.path.exists(savedir)

    for split in ["train", "val", "test"]:
        with open(f'{savedir}/{split}', "w", newline='') as f:
            writer = csv.writer(f)
            for row in data_split[split]:
                writer.writerow(row)


if __name__ == "__main__":
    torch.manual_seed(0)

    dataset("adult", train_frac=0.7, val_frac=0.1)
