import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

def one_vs_all(ys):
    # identify the most common class and set it to 1, set everything else as 0
    mode = ys.mode().iloc[0, 0]
    idx = ys == mode
    ys = np.zeros(shape=ys.shape)
    ys[idx] = 1
    return ys

DATADIR = './datasets/data'
def split_data(datadir, data_name):
    X = pd.read_csv(f"{datadir}/{data_name}/{data_name}_py.dat", header=None)
    y = pd.read_csv(f"{datadir}/{data_name}/labels_py.dat", header=None)

    y = one_vs_all(y)

    # pick ~30% for val / test
    train_X, test_X, train_y, test_y = train_test_split(
        X, y, test_size=0.3, random_state=0, stratify=y)

    # split test 50/50 into val and test
    val_X, test_X, val_y, test_y = train_test_split(
        test_X, test_y, test_size=0.5, random_state=0, stratify=test_y)
    
    train = pd.DataFrame(train_X)
    train['label'] = train_y

    val = pd.DataFrame(val_X)
    val['label'] = val_y

    test = pd.DataFrame(test_X)
    test['label'] = test_y
    
    pd.DataFrame(train).to_csv(f'{DATADIR}/{data_name}/{data_name}_train_py.dat', index=False)
    pd.DataFrame(val).to_csv(f'{DATADIR}/{data_name}/{data_name}_val_py.dat', index=False)
    pd.DataFrame(test).to_csv(f'{DATADIR}/{data_name}/{data_name}_test_py.dat', index=False)

if __name__ == "__main__":
    data_names = os.listdir(DATADIR)
    data_names.remove('info.json')
    data_names.remove('.DS_Store')
    for data_name in data_names:
        print(data_name)
        split_data(DATADIR, data_name)

