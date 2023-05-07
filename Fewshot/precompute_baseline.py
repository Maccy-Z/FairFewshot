import torch
from comparison2 import TabnetModel, FTTrModel, BasicModel
from AllDataloader import SplitDataloader
from utils import load_batch, get_batch
import pickle
import os
import csv

data_dir = './datasets/data'

# Get batch and save to disk. All columns.
def save_batch(ds_name, num_batches, num_targets, tag=None):
    if not os.path.exists(f"{data_dir}/{ds_name}/batches"):
        os.makedirs(f"{data_dir}/{ds_name}/batches")

    for num_rows in [2, 6, 10]:
        try:
            dl = SplitDataloader(
                ds_group=ds_name, 
                bs=num_batches, 
                num_rows=num_rows, 
                num_targets=num_targets, 
                num_cols=-3, 
                binarise=True,
                num_1s={'meta': num_rows // 2, 'target': num_targets // 2}
            )
            batch = get_batch(dl, num_rows=num_rows)

            # Save format: num_rows, num_targets, num_cols
            file_name = f'{num_rows}_{num_targets}_{-3}'
            if tag:
                file_name += f'_{tag}'
            with open(f"{data_dir}/{ds_name}/batches/{file_name}", "wb") as f:
                pickle.dump(batch, f)


        except IndexError as e:
            with open(f"{data_dir}/{ds_name}/batches/{file_name}", "wb") as f:
                pickle.dump(None, f)


def main(f, num_targets, batch_tag=None):
    models = [
              BasicModel("LR") , BasicModel("CatBoost"), BasicModel("R_Forest"),  BasicModel("KNN"),
              TabnetModel(),
              FTTrModel(),
              ]

    model_accs = [] # Save format: [model, num_rows, num_cols, acc, std]

    for model in models:
        for num_rows in [2, 6, 10]:
            for num_cols in [-3,]:
                try:
                    batch = load_batch(
                        ds_name=f, 
                        num_rows=num_rows,
                        num_cols=-3, 
                        num_targets=num_targets,
                        tag=batch_tag
                    )
                except IndexError as e:
                    break
                mean_acc, std_acc = model.get_accuracy(batch)
                print(f, num_rows, model, np.round(mean_acc, 2) * 100)
                model_accs.append([model, num_rows, num_cols, mean_acc, std_acc])

    
    if batch_tag:
        file_path = f'{data_dir}/{f}/baselines_{batch_tag}.dat'
    else:
        file_path = f'{data_dir}/{f}/baselines.dat'
    with open(file_path, 'a+', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["model", "num_rows", "num_cols", "acc", "std"])
        for row in model_accs:
            writer.writerow(row)

if __name__ == "__main__":
    import numpy as np
    import random
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)

    num_bs = 167
    num_targs = 6

    # files = [f for f in sorted(os.listdir(data_dir)) if os.path.isdir(f'{data_dir}/{f}')]
    files = [
            'acute-inflammation', 'acute-nephritis', 'arrhythmia',
            'blood', 'breast-cancer', 'breast-cancer-wisc', 'breast-cancer-wisc-diag', 
            'breast-cancer-wisc-prog', 'breast-tissue', 'cardiotocography-3clases', 
            'dermatology', 'echocardiogram', 'fertility', 'heart-cleveland', 
            'heart-hungarian', 'heart-switzerland', 'heart-va', 'hepatitis', 'horse-colic',
            'ilpd-indian-liver', 'lung-cancer', 'lymphography', 'mammographic', 
            'parkinsons', 'post-operative', 'primary-tumor', 'spect', 'spectf', 
            'statlog-heart', 'thyroid', 'vertebral-column-2clases'
        ]

    for f in files:
        print("---------------------")
        print(f)

        main(f, num_targets=num_targs, batch_tag='kshot')
