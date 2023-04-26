import torch
from comparison2 import TabnetModel, FTTrModel, BasicModel, get_batch
from AllDataloader import SplitDataloader
import os


def main(f, num_batches, num_rows, num_targets):
    dl = SplitDataloader(ds_group=f, bs=num_batches, num_rows=num_rows, num_targets=num_targets, num_cols=-3)
    batch = get_batch(dl, num_rows=num_rows)

    print(xs_meta.shape)

    exit(3)
    print(dl)


if __name__ == "__main__":
    data_dir = './datasets/data'
    num_batches = 2000
    num_rows = 10
    num_targets = 5


    files = [f for f in os.listdir(data_dir) if os.path.isdir(f'{data_dir}/{f}')]


    for f in files:
        main(f, num_batches=num_batches, num_rows=num_rows, num_targets=num_targets)






