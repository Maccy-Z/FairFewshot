import torch
from comparison2 import TabnetModel, FTTrModel, BasicModel
from AllDataloader import SplitDataloader
import pickle
import os
import csv
data_dir = './datasets/data'


def load_batch(ds_name, num_rows, num_targets, num_cols):
    with open(f"./datasets/data/{ds_name}/batches/{num_rows}_{num_targets}_{num_cols}", "rb") as f:
        batch = pickle.load(f)

    if batch is None:
        raise IndexError(f"Batch not found for file {ds_name}")
    return batch

def get_batch(dl, num_rows):
    xs, ys, model_id = next(iter(dl))
    xs_meta, xs_target = xs[:, :num_rows], xs[:, num_rows:]
    ys_meta, ys_target = ys[:, :num_rows], ys[:, num_rows:]
    xs_meta, xs_target = xs_meta.contiguous(), xs_target.contiguous()
    ys_meta, ys_target = ys_meta.contiguous(), ys_target.contiguous()
    # ys_target = ys_target.view(-1)

    return xs_meta, xs_target, ys_meta, ys_target

# Get batch and save to disk. All columns.
def save_batch(ds_name, num_batches, num_targets):
    if not os.path.exists(f"{data_dir}/{ds_name}/batches"):
        os.makedirs(f"{data_dir}/{ds_name}/batches")

    for num_rows in [10]:
        try:
            dl = SplitDataloader(ds_group=ds_name, bs=num_batches, num_rows=num_rows, num_targets=num_targets, num_cols=-3, binarise=True)
            batch = get_batch(dl, num_rows=num_rows)

            # Save format: num_rows, num_targets, num_cols
            with open(f"{data_dir}/{ds_name}/batches/{num_rows}_{num_targets}_{-3}_3_7", "wb") as f:
                pickle.dump(batch, f)


        except IndexError as e:
            with open(f"{data_dir}/{ds_name}/batches/{num_rows}_{num_targets}_{-3}_3_7", "wb") as f:
                pickle.dump(None, f)


def main_append(f, num_targets):

    models = [
              BasicModel("LR") , BasicModel("CatBoost"), BasicModel("R_Forest"),  BasicModel("KNN"),
              TabnetModel(),
              FTTrModel(),
              ]

    model_accs = [] # Save format: [model, num_rows, num_cols, acc, std]

    for model in models:
        print(model)
        for num_rows in [3]:
            for num_cols in [-3,]:
                try:
                    batch = load_batch(ds_name=f, num_rows=num_rows, num_cols=-3, num_targets=num_targets)
                except IndexError as e:
                    break
                mean_acc, std_acc = model.get_accuracy(batch)
                model_accs.append([model, num_rows, num_cols, mean_acc, std_acc])

    with open(f'{data_dir}/{f}/baselines.dat', 'a', newline='') as f:
        writer = csv.writer(f)
        for row in model_accs:
            writer.writerow(row)


def main(f, num_targets):

    models = [
              BasicModel("LR") , BasicModel("CatBoost"), BasicModel("R_Forest"),  BasicModel("KNN"),
              TabnetModel(),
              FTTrModel(),
              ]

    model_accs = [] # Save format: [model, num_rows, num_cols, acc, std]

    for model in models:
        print(model)
        for num_rows in [5, 10, 15]:
            for num_cols in [-3,]:
                try:
                    batch = load_batch(ds_name=f, num_rows=num_rows, num_cols=-3, num_targets=num_targets)
                except IndexError as e:
                    break
                mean_acc, std_acc = model.get_accuracy(batch)
                model_accs.append([model, num_rows, num_cols, mean_acc, std_acc])

    with open(f'{data_dir}/{f}/baselines.dat', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "num_rows", "num_cols", "acc", "std"])
        for row in model_accs:
            writer.writerow(row)

if __name__ == "__main__":
    import numpy as np
    import random
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)

    num_bs = 200
    num_targs = 5

    files = [f for f in sorted(os.listdir(data_dir)) if os.path.isdir(f'{data_dir}/{f}')]

    for f in files:
        print("---------------------")
        print(f)

        # main_append(f, num_targets=num_targs)
