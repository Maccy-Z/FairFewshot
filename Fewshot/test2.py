import torch
from main import *
from dataloader import d2v_pairer
from AllDataloader import SplitDataloader
from config import get_config
import os
import toml
import numpy as np
import random

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from pytorch_tabnet.tab_model import TabNetClassifier


def get_batch(dl, num_rows):
    try:
        xs, ys, model_id = next(iter(dl))
    except:
        xs, ys = next(iter(dl))
        model_id = []

    xs_meta, xs_target = xs[:, :num_rows], xs[:, num_rows:]
    ys_meta, ys_target = ys[:, :num_rows], ys[:, num_rows:]
    xs_meta, xs_target = xs_meta.contiguous(), xs_target.contiguous()
    ys_meta, ys_target = ys_meta.contiguous(), ys_target.contiguous()
    ys_target = ys_target.view(-1)

    if len(model_id) > 0:
        return model_id, xs_meta, xs_target, ys_meta, ys_target
    return xs_meta, xs_target, ys_meta, ys_target


class SKLModel:
    def __init__(self, model, name):
        self.model = model
        self.name = name
        self.identical_batch = False

    def fit(self, xs_meta, ys_meta):
        ys_meta = ys_meta.detach()[0].flatten()
        xs_meta = xs_meta.detach()[0]

        if ys_meta.min() == ys_meta.max():
            self.identical_batch = True
            self.pred_val = ys_meta[0].numpy()
        else:
            self.identical_batch = False
            self.model.fit(X=xs_meta, y=ys_meta)
            self.pred_val = None

    def get_acc(self, xs_target, ys_target):
        if self.identical_batch:
            predictions = np.ones_like(ys_target) * self.pred_val
        else:
            xs_target = xs_target.detach()[0]
            predictions = self.model.predict(X=xs_target)

        ys_lr_target_labels = np.array(predictions).flatten()
        accuracy = (ys_lr_target_labels == np.array(ys_target)).sum().item() / len(ys_target)

        return accuracy

    def __repr__(self):
        return self.name


class TabnetModel:
    def __init__(self):
        self.model = TabNetClassifier(device_name="cpu")

    def fit(self, xs_meta, ys_meta):
        ys_meta = ys_meta.detach()[0].flatten()
        xs_meta = xs_meta.detach()[0]

        if ys_meta.min() == ys_meta.max():
            self.identical_batch = True
            self.pred_val = ys_meta[0].numpy()
        else:
            self.identical_batch = False
            self.model.fit(xs_meta.numpy(), ys_meta.numpy(),
                           batch_size=8, virtual_batch_size=8, patience=10, drop_last=False)
            self.pred_val = None

    def get_acc(self, xs_target, ys_target):
        if self.identical_batch:
            predictions = np.ones_like(ys_target) * self.pred_val
        else:
            xs_target = xs_target.detach()[0]
            predictions = self.model.predict(X=xs_target)

        ys_lr_target_labels = np.array(predictions).flatten()
        accuracy = (ys_lr_target_labels == np.array(ys_target)).sum().item() / len(ys_target)

        return accuracy

    def __repr__(self):
        return "TabNet"


class Fewshot:
    def __init__(self, save_dir):
        print(f'Loading model at {save_dir = }')

        state_dict = torch.load(f'{save_dir}/model.pt')
        self.model = ModelHolder(cfg_all=get_config(cfg_file=f'{save_dir}/defaults.toml'))
        self.model.load_state_dict(state_dict['model_state_dict'])

    def fit(self, xs_meta, ys_meta):
        pairs_meta = d2v_pairer(xs_meta, ys_meta)
        with torch.no_grad():
            self.embed_meta, self.pos_enc = self.model.forward_meta(pairs_meta)

    def get_acc(self, xs_target, ys_target):
        with torch.no_grad():
            ys_pred_target = self.model.forward_target(xs_target, self.embed_meta, self.pos_enc)

        ys_pred_target_labels = torch.argmax(ys_pred_target.view(-1, 2), dim=1)
        accuracy = (ys_pred_target_labels == ys_target).sum().item() / len(ys_target)

        return accuracy

    def __repr__(self):
        return "Fewshot"


def main(save_no, ds_group=-1, print_result=True):
    BASEDIR = '.'
    dir_path = f'{BASEDIR}/saves'
    files = [f for f in os.listdir(dir_path) if os.path.isdir(f'{dir_path}/{f}')]
    existing_saves = sorted([int(f[5:]) for f in files if f.startswith("save")])  # format: save_{number}
    save_no = existing_saves[save_no]
    save_dir = f'{BASEDIR}/saves/save_{save_no}'

    # state_dict = torch.load(os.path.join(save_dir, 'model.pt'))
    # model = ModelHolder(cfg_all=get_config(cfg_file=f'{save_dir}/defaults.toml'))
    # model.load_state_dict(state_dict['model_state_dict'])

    cfg = toml.load(os.path.join(save_dir, 'defaults.toml'))["DL_params"]

    num_rows = 16  # cfg["num_rows"]
    num_targets = cfg["num_targets"]
    # ds_group = 2 # cfg["ds_group"]

    # models = [LogisticRegression(max_iter=1000), SVC()]
    models = [Fewshot(save_dir),
              SKLModel(LogisticRegression(max_iter=1000), name="LR"), SKLModel(SVC(), name="SVC"),
              TabnetModel()]
    # baseline_model_names = ['LR', 'SVC']

    col_accs = {}
    for num_cols in range(1, 20, 2):
        accs = {str(model): [] for model in models}
        val_dl = SplitDataloader(bs=1, num_rows=num_rows, num_targets=5,
                                 num_cols=num_cols, ds_group=ds_group, split="val")

        for j in range(10):
            # Fewshot predictions
            xs_meta, xs_target, ys_meta, ys_target = get_batch(val_dl, num_rows)

            for model in models:
                model.fit(xs_meta, ys_meta)
                acc = model.get_acc(xs_target, ys_target)
                accs[str(model)].append(acc)

        for model_name, all_accs in accs.items():
            mean_acc = np.mean(all_accs)
            accs[model_name] = mean_acc
        col_accs[num_cols] = accs

        if print_result:
            print()
            for model_name, all_accs in accs.items():
                print(f'{all_accs:.3f}')

    return col_accs


if __name__ == "__main__":
    # random.seed(0)
    # np.random.seed(0)
    # torch.manual_seed(0)

    # save_number = int(input("Enter save number:\n"))
    # main(save_no=save_number)

    for eval_no in range(1):
        print()
        print("Eval number", eval_no)
        col_accs = main(save_no=-(eval_no + 1), ds_group=-1)

    print(col_accs)
