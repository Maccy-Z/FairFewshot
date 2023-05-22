import os
import toml
import random

base_dir = "./data"
datasets = sorted([f for f in os.listdir(base_dir) if os.path.isdir(f'{base_dir}/{f}')])

short = []
long = []

for dataset in datasets:
    new_baselines = []
    with open(f'./data/{dataset}/{dataset}_py.dat', "r") as f:
        labels = f.readline().split(",")

        if len(labels) < 50:
            short.append(dataset)
        else:
            long.append(dataset)

dataset_splits = {}

short_train = random.sample(short, int(2 * len(short) / 3))
short_test = [e for e in short if e not in short_train]

long_train = random.sample(long, int(2 * len(long) / 3))
long_test = [e for e in long if e not in long_train]


with open("grouped_datasets/short_splits", "w") as f:

    dataset_splits["short"] = {"train": sorted(short_train), "test": sorted(short_test)}
    dataset_splits["long"] = {"train": sorted(long_train), "test": sorted(long_test)}

    print(dataset_splits)
    toml.dump(dataset_splits, f)




