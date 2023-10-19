# Split into 4 folds.
import os
import random
import toml

random.seed(0)

base_dir = "./datasets/grouped_datasets"


groups = sorted([f for f in os.listdir(base_dir) if os.path.isdir(f'{base_dir}/{f}')])
# Give each dataset a number from 0 to 3, denoting the split.
keys = [0, 1, 2, 3] * 100

ds_groups = {}
for group in groups:

    # print(f'{base_dir}/{split}')
    split_data_names = os.listdir(f'{base_dir}/{group}')
    random.shuffle(split_data_names)
    key = keys[:len(split_data_names)]

    ds_groups[group] = {k: v for k, v in zip(split_data_names, key)}

for fold in range(4):
    dataset_splits = {}

    for group_no, group in ds_groups.items():
        trains, tests = [], []
        for name, fold_group in group.items():
            if fold_group == fold:
                tests.append(name)
            else:
                trains.append(name)
        dataset_splits[group_no] = {"train": sorted(trains), "test": sorted(tests)}

    with open(f'{base_dir}/splits_{fold}', "w") as f:
        toml.dump(dataset_splits, f)



