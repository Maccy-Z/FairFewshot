# Sample and save batches
from AllDataloader import SplitDataloader
from itertools import islice
import os
import pickle
from config import Config

data_dir = "./datasets/data"


def save_batch(cfg, ds_name, num_batches):
    save_file = f"{data_dir}/{ds_name}/batches/{cfg.N_meta}_{cfg.N_target}"


    if not os.path.exists(f"{data_dir}/{ds_name}/batches"):
        os.makedirs(f"{data_dir}/{ds_name}/batches")

    try:
        dl = SplitDataloader(cfg, ds_group=[ds_name], bs=num_batches)
        batch = next(iter(dl))
        print(batch)
        exit(2)
        # Save format: num_rows, num_targets, num_cols
        with open(save_file, "wb") as f:
            pickle.dump(batch, f)

    except IndexError as e:
        print(e)
        exit(2)
        with open(save_file, "wb") as f:
            pickle.dump(None, f)


def main():
    datasets = sorted([f for f in os.listdir(data_dir) if os.path.isdir(f'{data_dir}/{f}')])

    N_batches = 2000
    for N_meta in [1, 2, 3, 5, 10, 15]:

        cfg = Config()
        cfg.N_meta = N_meta
        cfg.N_target = 3


        save_batch(cfg, "adult", N_batches)


if __name__ == "__main__":
    main()

# for ds in datasets:
#     try:
#         dl = SplitDataloader(cfg, bs=2, ds_group=[ds], ds_split="train")
#     except IndexError as e:
#         print(e)
#         continue
#
#     for xs_meta, ys_meta, xs_target, ys_target, max_N_label in islice(dl, 2000):
#
#         print(xs_meta, ys_meta, xs_target, ys_target, max_N_label)

