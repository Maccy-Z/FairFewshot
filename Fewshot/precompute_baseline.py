import torch
from comparison2 import TabnetModel, FTTrModel, BasicModel, get_batch
from AllDataloader import SplitDataloader
import os
import csv
data_dir = './datasets/data'


def main(f, num_batches, num_targets):

    models = [
              BasicModel("LR"), BasicModel("CatBoost"), BasicModel("R_Forest"),  BasicModel("KNN"),
              # TabnetModel(),
              # FTTrModel(),
              ]


    model_accs = [] # Save format: [model, num_rows, num_cols, acc, std]

    for model in models:
        print(model)
        for num_rows in [5, 10, 15]:
            for num_cols in [-3, 2, 4, 8, 16, 32]:

                try:
                    if num_cols == -3:
                        dl = SplitDataloader(ds_group=f, bs=num_batches, num_rows=num_rows, num_targets=num_targets, num_cols=-3)
                    else:
                        dl = SplitDataloader(ds_group=f, bs=num_batches, num_rows=num_rows, num_targets=num_targets, num_cols=[num_cols, num_cols])
                except IndexError:
                    print(IndexError)
                    break
                batch = get_batch(dl, num_rows=num_rows)
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

    num_batches = 250
    num_targets = 5


    files = [f for f in os.listdir(data_dir) if os.path.isdir(f'{data_dir}/{f}')]


    for f in files:
        main(f, num_batches=num_batches, num_targets=num_targets)






