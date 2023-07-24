# Sample and save batches
from AllDataloader import SplitDataloader
from itertools import islice
import os
import pickle
from config import Config

batches = 2000

data_dir = "./datasets/data"
datasets = sorted([f for f in os.listdir(data_dir) if os.path.isdir(f'{data_dir}/{f}')])


def save_batch(ds_name, num_batches, num_targets, num_rows):
    cfg = Config()

    if not os.path.exists(f"{data_dir}/{ds_name}/batches"):
        os.makedirs(f"{data_dir}/{ds_name}/batches")

    for num_row in num_rows:
        try:
            dl = SplitDataloader(cfg, ds_group=ds_name, bs=num_batches)
            batch = next(iter(dl))

            # Save format: num_rows, num_targets, num_cols
            with open(f"{data_dir}/{ds_name}/batches/{num_row}_{num_targets}_{-3}", "wb") as f:
                pickle.dump(batch, f)


        except IndexError as e:
            print(e)
            with open(f"{data_dir}/{ds_name}/batches/{num_row}_{num_targets}_{-3}", "wb") as f:
                pickle.dump(None, f)



for ds in datasets:
    try:
        dl = SplitDataloader(cfg, bs=2, ds_group=[ds], ds_split="train")
    except IndexError as e:
        print(e)
        continue

    for xs_meta, ys_meta, xs_target, ys_target, max_N_label in islice(dl, 2000):

        print(xs_meta, ys_meta, xs_target, ys_target, max_N_label)

