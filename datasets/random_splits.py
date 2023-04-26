import random
import numpy as np
import toml
import os
import sys
sys.path.insert(0, '/Users/kasiakobalczyk/FairFewshot')
from Fewshot.AllDataloader import MyDataSet
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

random.seed(123)
np.random.seed(123)

def get_datasets(ds_type):
    if ds_type == 'all':
        all_data_names = os.listdir('./datasets/data')
        all_data_names.remove('info.json')
        if '.DS_Store' in all_data_names:
            all_data_names.remove('.DS_Store')

    elif ds_type == "medical":
        all_data_names = [
            'acute-inflammation', 'acute-nephritis', 'arrhythmia',
            'blood', 'breast-cancer', 'breast-cancer-wisc', 'breast-cancer-wisc-diag', 
            'breast-cancer-wisc-prog', 'breast-tissue', 'cardiotocography-3clases', 
            'dermatology', 'echocardiogram', 'fertility', 'heart-cleveland', 
            'heart-hungarian', 'heart-switzerland', 'heart-va', 'hepatitis', 'horse-colic',
            'ilpd-indian-liver', 'lung-cancer', 'lymphography', 'mammographic', 
            'parkinsons', 'post-operative', 'primary-tumor', 'spect', 'spectf', 
            'statlog-heart', 'thyroid', 'vertebral-column-2clases'
        ]
    return all_data_names

def get_n_col(all_data_names):
    all_datasets = [
        MyDataSet(d, num_rows=0, num_targets=0, binarise=True, split="all") 
        for d in all_data_names
    ]
    n_col = [d.ds_cols - 1 for d in all_datasets]
    n_col = dict(zip(all_data_names, n_col))
    return n_col

def split_datasets(all_data_names):
    n_col = get_n_col(all_data_names)

    kf = KFold(n_splits=5, shuffle=True)
    kf.get_n_splits(all_data_names)

    splits = {}

    for i, (train_index, test_index) in enumerate(kf.split(all_data_names)):
        ds_train = [all_data_names[i] for i in train_index]
        ds_test = [all_data_names[i] for i in test_index]
        ds_val, ds_test = train_test_split(ds_test, test_size=0.5, shuffle=True)
        max_train_col = max([n_col[d] for d in ds_train])
        max_test_col = max([n_col[d] for d in ds_test])
        max_val_col = max([n_col[d] for d in ds_val])
        splits[str(i)] = {
            'train': ds_train,
            'val': ds_val,
            'test': ds_test,
            'max_val_col': max_val_col,
            'max_test_col': max_test_col
        }
    print(splits)
    return splits

if __name__ == "__main__":
    all_data_names = get_datasets("all")
    splits = split_datasets(all_data_names)
    with open("./datasets/grouped_datasets/my_splits", "w") as fp:
        toml.dump(splits, fp) 

