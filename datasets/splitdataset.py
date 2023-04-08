import pandas as pd
import numpy as np
import os

DATADIR = './datasets/data'
def split_data(datadir, data_name):
    df = pd.read_csv(f"{datadir}/{data_name}/{data_name}_py.dat", header=None)

    # pick ~20% of rows and ~20% (or min 2) of columns for testing
    frac = 0.2
    n_val_rows = int(df.shape[0] * frac)
    n_val_cols = max(int(df.shape[1] * frac), 2)

    cols_rand_idx = np.random.permutation(list(range(df.shape[1])))
    rows_rand_idx = np.random.permutation(list(range(df.shape[0])))
    train_cols, val_cols = cols_rand_idx[:-n_val_cols], cols_rand_idx[-n_val_cols:]
    train_rows, val_rows = rows_rand_idx[:-n_val_cols], rows_rand_idx[-n_val_rows:]

    train_df = df.iloc[train_rows, train_cols]
    val_df = df.iloc[val_rows, val_cols]

    train_df.to_csv(f'{datadir}/{data_name}/{data_name}_train_py.dat', header=False)
    val_df.to_csv(f'{datadir}/{data_name}/{data_name}_val_py.dat', header=False)


if __name__ == "__main__":
    data_names = os.listdir(DATADIR)
    data_names.remove('info.json')
    for data_name in data_names:
        split_data(DATADIR, data_name)

