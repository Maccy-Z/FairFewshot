#%%
import os
import random
import toml

random.seed(0)

base_dir = "./datasets/grouped_datasets"
frac = 0.25


d_splits = sorted([f for f in os.listdir(base_dir) if os.path.isdir(f'{base_dir}/{f}')])

dataset_splits = {}
for i, split in enumerate(d_splits):
    # print(f'{base_dir}/{split}')
    all_data_names = os.listdir(f'{base_dir}/{split}')

    num_val = int(len(all_data_names) * frac)
    test_ds = random.sample(all_data_names, num_val)
    train_ds = [item for item in all_data_names if item not in test_ds]

    dataset_splits[split] = {"train": train_ds, "test": test_ds}

print(dataset_splits)


with open(f'{base_dir}/splits', "w") as f:
    toml.dump(dataset_splits, f)


# %%
