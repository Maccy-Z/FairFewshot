#%%
import pandas as pd
import seaborn as sns
import numpy as np 

names = ['heart-cleveland', 'heart-hungarian', 'heart-va', 'heart-switzerland']
dataframe_dict = {}
for name in names:
    dir_name = f'datasets/data/{name}'
    dataframe_dict[name] = {
        'data' : pd.read_csv(
            f'{dir_name}/{name}_py.dat', header=None).iloc[:, :12],
        'labels' : pd.read_csv(
            f'{dir_name}/labels_py.dat', header=None),
        'folds' : pd.read_csv(
            f'{dir_name}/folds_py.dat', header=None),
        'vfolds' : pd.read_csv(
            f'{dir_name}/validation_folds_py.dat', header=None),
    }

train = {
    'heart-hungarian': dataframe_dict['heart-hungarian'],
    'heart-va': dataframe_dict['heart-va']
}

test = {
    'heart-cleveland': dataframe_dict['heart-cleveland']
}

extra = {
    'heart-switzerland': dataframe_dict['heart-switzerland']
}

#%%
def get_columns(n_overlap, seed):
    np.random.seed(seed)
    columns = np.array(list(range(12)))
    n_remaining = 12 - n_overlap
    common_columns = np.random.choice(columns, n_overlap, replace=False)
    remaining_columns = set(list(range(24))).difference(set(common_columns))
    remaining_columns = np.array(list(remaining_columns))
    remaining_columns = np.random.choice(
        remaining_columns, n_remaining * 2, replace=False)
    
    train_columns, test_columns = remaining_columns[ :n_remaining], remaining_columns[n_remaining:]

    return common_columns, train_columns, test_columns

def get_overlap_data(n_overlap, seed):
    common_columns, train_columns, test_columns = get_columns(n_overlap, seed)
    write_files(train, common_columns, train_columns)
    write_files(test, common_columns, test_columns)

def write_files(df_dict, common_columns, other_columns):
    for name, data_df in df_dict.items():
        columns = list(common_columns) + list(other_columns)
        name = f'{name}_{n_overlap}_{seed}'
        dir = f'overlapdatasets2/data/{name}'
        try:
            os.mkdir(dir)
        except(FileExistsError):
            continue
        df = pd.concat([data_df['data'], extra['heart-switzerland']['data']], axis=1)
        df.iloc[:, columns].to_csv(f'{dir}/{name}_py.dat', header=False, index=False)
        data_df['labels'].to_csv(f'{dir}/labels_py.dat', header=False, index=False)
        data_df['folds'].to_csv(f'{dir}/folds_py.dat', header=False, index=False)
        data_df['vfolds'].to_csv(f'{dir}/validation_folds_py.dat', header=False, index=False)
    
# %%
for n_overlap in list(range(0, 13, 2)):
    for seed in list(range(6)):
        get_overlap_data(n_overlap, seed)
# %%
import os 
import toml

base_dir = "overlapdatasets2/grouped_datasets"

dataset_splits = {}

for n_overlap in list(range(0, 13, 2)):
    for seed in list(range(6)):
        group_no = n_overlap * 10 + seed

        trains = [f'{name}_{n_overlap}_{seed}' for name in train.keys()]
        tests = [f'{name}_{n_overlap}_{seed}' for name in test.keys()]

        dataset_splits[str(group_no)] = {"train": sorted(trains), "test": sorted(tests)}

with open(f'{base_dir}/overlapping_splits', "w") as f:
    toml.dump(dataset_splits, f)
# %%
