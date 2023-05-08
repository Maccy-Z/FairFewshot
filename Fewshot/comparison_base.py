from main import *
import time, os, toml, random, pickle, warnings
import numpy as np
from scipy import stats
from abc import ABC, abstractmethod
import pandas as pd
from collections import defaultdict


import sys
sys.path.append('/mnt/storage_ssd/FairFewshot/STUNT_main')
#from STUNT_interface import STUNT_utils, MLPProto

BASEDIR = '.'


class Model(ABC):
    # Process batch of data
    def __init__(self, model_name):
        self.model_name = model_name


    def get_accuracy(self, ds_name, num_rows, num_cols):
        with open(f'./datasets/data/{ds_name}/baselines.dat', "r") as f:
            lines = f.read()

        lines = lines.split("\n")[1:]

        for config in lines:
            if config.startswith(f'{self.model_name},{num_rows},{num_cols}'):
                config = config.split(",")

                mean, std = float(config[-2]), float(config[-1])
                return mean, std


        raise FileNotFoundError(f"Requested config does not exist: {self.model_name}, {ds_name}, {num_rows=}, {num_cols=}")

    def __repr__(self):
        return self.model_name


def get_results_by_dataset(test_data_names, models, num_rows=10, num_targets=5, num_samples=3, agg=False, binarise=True):
    """
    Evaluates the model and baseline_models on the test data sets.
    Results are groupped by: data set, model, number of test columns.
    """

    results = pd.DataFrame(columns=['data_name', 'model', 'num_cols', 'acc', 'std'])

    # Test on full dataset
    for data_name in test_data_names:
        model_acc_std = defaultdict(list)
        for model in models:
            try:
                mean_acc, std_acc = model.get_accuracy(data_name, num_rows, -3)
            except FileNotFoundError as e:
                print(e)
                continue

            model_acc_std[str(model)].append([mean_acc, std_acc])

        for model_name, acc_stds in model_acc_std.items():
            acc_stds = np.array(acc_stds)
            # For baselines, variance is sample variance.
            if len(acc_stds) == 1:
                mean_acc, std_acc = acc_stds[0, 0], acc_stds[0, 1]

            # Average over all FLAT and FLAT_MAML models.
            # For FLAT, variance is variance between models
            else:
                means, std = acc_stds[:, 0], acc_stds[:, 1]
                mean_acc = np.mean(means)
                std_acc = np.std(means, ddof=1) / np.sqrt(means.shape[0])

            result = pd.DataFrame({
                        'data_name': data_name,
                        'model': str(model_name),
                        'num_cols': -1,
                        'acc': mean_acc,
                        'std': std_acc
                        }, index=[0])
            results = pd.concat([results, result])

    results.reset_index(drop=True, inplace=True)
    return results


def main(load_no, num_rows, save_ep=None):
    dir_path = f'{BASEDIR}/saves'
    files = [f for f in os.listdir(dir_path) if os.path.isdir(f'{dir_path}/{f}')]
    existing_saves = sorted([int(f[5:]) for f in files if f.startswith("save")])  # format: save_{number}
    load_no = [existing_saves[num] for num in load_no]
    load_dir = f'{BASEDIR}/saves/save_{load_no[-1]}'

    all_cfg = toml.load(os.path.join(load_dir, 'defaults.toml'))
    cfg = all_cfg["DL_params"]
    ds = all_cfg["Settings"]["dataset"]
    ds_group = cfg["ds_group"]

    if ds == "my_split":
        exit(2)
        split_file = f"./datasets/grouped_datasets/{cfg['split_file']}"
        with open(split_file) as f:
            split = toml.load(f)
        train_data_names = split[str(ds_group)]["train"]
        test_data_names = split[str(ds_group)]["test"]

        print("Train datases:", train_data_names)
        print("Test datasets:", test_data_names)

    elif ds == "total":
        ds_group = save_ep
        fold_no, split_no = ds_group

        splits = toml.load(f'./datasets/grouped_datasets/splits_{fold_no}')
        print("Testing group:", ds_group)

        if split_no == -1:
            get_splits = range(6)
        else:
            get_splits = [split_no]

        test_data_names = []
        for split in get_splits:
            ds_name = splits[str(split)]["test"]
            test_data_names += ds_name

        train_data_names = []
        for split in get_splits:
            ds_name = splits[str(split)]["train"]
            train_data_names += ds_name

        print("Test datasets:", test_data_names)

    else:
        raise Exception("Invalid data split")

    num_targets = 5
    binarise = cfg["binarise"]

    models = [Model("LR"), Model("CatBoost"), Model("R_Forest"),  Model("KNN"), Model("TabNet"), Model("FTTransformer")
              ]

    unseen_results = get_results_by_dataset(
        test_data_names, models,
        num_rows=num_rows, num_targets=num_targets, binarise=binarise
    )


    # Results for each dataset
    detailed_results = unseen_results.copy()

    mean, std = detailed_results["acc"], detailed_results["std"]
    mean_std = [f'{m * 100:.2f}±{s * 100:.2f}' for m, s in zip(mean, std)]
    detailed_results['acc_std'] = mean_std

    results = detailed_results.pivot(columns=['data_name', 'model'], index='num_cols', values=['acc_std'])
    # print("======================================================")
    # print("Test accuracy on unseen datasets")
    # print(results.to_string())

    det_results = detailed_results.pivot(columns=['data_name', 'model'], index='num_cols', values=['acc'])
    det_results = det_results.to_string()


    # Aggreate results
    agg_results = unseen_results.copy()

    # Move flat to first column
    agg_results = agg_results.groupby(['num_cols', 'model'])['acc'].mean().unstack()


    # Get errors using appropriate formulas.
    pivot_acc = unseen_results.pivot(
        columns=['data_name', 'model'], index='num_cols', values=['acc'])
    pivot_std = unseen_results.pivot(
        columns=['data_name', 'model'], index='num_cols', values=['std'])
    model_names = pivot_acc.columns.get_level_values(2).unique()
    for model_name in model_names:

        model_accs = pivot_acc.loc[:, ("acc", slice(None), model_name)]
        model_stds = pivot_std.loc[:, ("std", slice(None), model_name)]

        mean_stds = []
        for i in range(pivot_acc.shape[0]):
            accs = np.array(model_accs.iloc[i].dropna())
            std = np.array(model_stds.iloc[i].dropna())

            assert std.shape == accs.shape
            mean_acc = np.mean(accs)
            std_acc = np.sqrt(np.sum(std ** 2)) / std.shape[0]
            mean_std = f'{mean_acc * 100:.2f}±{std_acc * 100:.2f}'
            mean_stds.append(mean_std)

        agg_results[model_name] = mean_stds

    # print()
    # print("======================================================")
    # print("Test accuracy on unseen datasets (aggregated)")
    # print(agg_results.to_string(index=False))
    print(agg_results.to_string())


if __name__ == "__main__":

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)


    col_accs = main(load_no=[-1, -2, -3, -4, -5], num_rows=3, save_ep=[3, -1])
