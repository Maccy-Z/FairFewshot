import torch
from comparison2 import TabnetModel, FTTrModel, BasicModel, get_batch
from AllDataloader import SplitDataloader
import os
import csv
data_dir = './datasets/data'


def main(f, num_batches, num_targets):

    models = [
              # BasicModel("LR"), BasicModel("CatBoost"), BasicModel("R_Forest"),  BasicModel("KNN"),
              # TabnetModel(),
              FTTrModel(),
              ]


    model_accs = [] # Save format: [model, num_rows, num_cols, acc, std]

    for model in models:
        print(model)
        for num_rows in [2, 5, 10, 15]:
            for num_cols in [1, 2, 4, 8, 16, 32, 64]:
                print("Next")
                dl = SplitDataloader(ds_group=f, bs=num_batches, num_rows=num_rows, num_targets=num_targets, num_cols=-3)
                if num_cols > dl.min_ds_cols:
                    break
                batch = get_batch(dl, num_rows=num_rows)
                mean_acc, std_acc = model.get_accuracy(batch)
                model_accs.append([model, num_rows, num_cols, mean_acc, std_acc])

    print(model_accs)

    with open(f'{data_dir}/{f}/baselines.dat', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "num_rows", "num_cols", "acc", "std"])
        for row in model_accs:
            writer.writerow(row)

    exit(3)
    print(dl)


if __name__ == "__main__":
    num_batches = 100
    num_targets = 5


    files = [f for f in os.listdir(data_dir) if os.path.isdir(f'{data_dir}/{f}')]


    for f in files:
        main(f, num_batches=num_batches, num_targets=num_targets)






