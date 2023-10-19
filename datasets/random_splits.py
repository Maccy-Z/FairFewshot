"""Generate medical dataset splits"""
import random
import numpy as np
import toml
import os
from Fewshot.AllDataloader import MyDataSet, SplitDataloader
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import train_test_split
import pandas as pd

random.seed(123)
np.random.seed(123)

def get_datasets(ds_type, num_rows, num_targets, binarise=True):
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
    dl = SplitDataloader(
        bs=1, num_rows=num_rows, num_targets=num_targets, 
        ds_group=all_data_names, binarise=binarise
    )
    all_datasets = [d for d in dl.all_datasets]
    all_data_names = [d.ds_name for d in all_datasets]
    return all_data_names

def get_ds_df(all_data_names):
    all_datasets = [
        MyDataSet(d, num_rows=0, num_targets=0, binarise=True, split="all") 
        for d in all_data_names
    ]
    n_col = [d.ds_cols - 1 for d in all_datasets]
    df = pd.DataFrame({'data_name': all_data_names, 'n_col': n_col})
    df['size'] = pd.qcut(df['n_col'], 3, labels=[0, 1, 2])
    return df

def split_datasets(all_data_names, get_val=False, n_splits=10, stratify=True):
    df = get_ds_df(all_data_names)

    if stratify:
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True)
        kf.get_n_splits(X=df['data_name'], y=df['size'])
    else:
        kf = KFold(n_splits=n_splits, shuffle=True)
        kf.get_n_splits(X=df['data_name'], y=df['size'])
        
    splits = {}

    for i, (train_index, test_index) in enumerate(kf.split(df['data_name'], df['size'])):
        df_train = df.iloc[train_index, :]
        test_ds = df.iloc[test_index, :]['data_name']
        if get_val:
            train_ds , val_ds = train_test_split(df_train['data_name'], test_size=3, shuffle=True, stratify=df_train['size'])
        else:
            val_ds = []
            train_ds = df_train['data_name']
        splits[str(i)] = {
            'train': list(train_ds),
            'val': list(val_ds),
            'test': list(test_ds),
            'max_train_col':  int(df_train['n_col'].max()),
            'max_val_col':  int(df[df.data_name.isin(val_ds)]['n_col'].max()) if val_ds else 0,
            'max_test_col':  int(df[df.data_name.isin(test_ds)]['n_col'].max())
        }
    print(splits)
    return splits

if __name__ == "__main__":
    all_data_names = get_datasets("medical", num_rows=10, num_targets=15)
    #all_data_names = ['fertility', 'lung-cancer', 'breast-cancer', 'mammographic', 'echocardiogram', 'heart-va', 'post-operative', 'heart-switzerland']
    splits = split_datasets(all_data_names, n_splits=10, stratify=False)

    with open("./datasets/grouped_datasets/med_splits_3", "w") as fp:
      toml.dump(splits, fp) 