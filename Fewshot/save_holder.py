import os
import torch
import pickle
import datetime
from matplotlib import pyplot as plt
import shutil
import numpy as np


# Save into file. Automatically make a new folder for every new save.
class SaveHolder:
    def __init__(self, base_dir):
        dir_path = f'{base_dir}/saves'
        files = [f for f in os.listdir(dir_path) if os.path.isdir(f'{dir_path}/{f}')]
        existing_saves = sorted([int(f[5:]) for f in files if f.startswith("save")])  # format: save_{number}

        if existing_saves:
            save_no = existing_saves[-1] + 1
        else:
            save_no = 0

        self.save_dir = f'{base_dir}/saves/save_{save_no}'

        print("Making new save folder at: ")
        print(self.save_dir)
        os.mkdir(self.save_dir)

        shutil.copy(f'{base_dir}/Fewshot/defaults.toml', f'{self.save_dir}/defaults.toml')

    def save_model(self, model: torch.nn.Module):
        torch.save(model, f'{self.save_dir}/model.pt')

    def save_history(self, hist_dict: dict):
        with open(f'{self.save_dir}/history.pkl', 'wb') as f:
            pickle.dump(hist_dict, f)


# Needs to be imported after SaveHolder or there will be a circular import
from main import *


class SaveLoader:
    def __init__(self, save_dir):
        print(f"Loading save at {save_dir}")
        # self.model = torch.load(f'{save_dir}/model.pt')

        with open(f'{save_dir}/history.pkl', "rb") as f:
            self.history = pickle.load(f)

    def plot_history(self):
        # val_accs = self.history["val_accs"]
        # val_accs = np.array(val_accs)
        # val_accs = np.mean(val_accs, axis=-1)

        train_accs = self.history["accs"]
        train_accs = np.array(train_accs)
        train_accs = np.array_split(train_accs, len(train_accs) // 500)
        train_accs = np.stack([np.mean(ary) for ary in train_accs])

        plt.plot(train_accs, label="Train Acc")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    BASEDIR = "/mnt/storage_ssd/FairFewshot"
    saves = os.listdir(f'{BASEDIR}/saves')
    saves = sorted(saves)

    # h = SaveHolder(base_dir=f'{BASEDIR}')
    h = SaveLoader(save_dir=f'{BASEDIR}/saves/{saves[-1]}')
    h.plot_history()
