import torch
from main import *
from dataloader import d2v_pairer
from AllDataloader import AllDatasetDataLoader
from config import get_config
import os
import toml
import numpy as np
import random

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


class ZeroModel:
    def fit(self, X, y):
        self.y = torch.mode(y)[0]
        return

    def predict(self, X):
        return np.ones(X.shape[0]) * self.y.numpy()


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


def get_predictions(xs_meta, xs_target, ys_meta, model):
    pairs_meta = d2v_pairer(xs_meta, ys_meta)
    with torch.no_grad():
        embed_meta, pos_enc = model.forward_meta(pairs_meta)
        ys_pred_target = model.forward_target(xs_target, embed_meta, pos_enc)
    return ys_pred_target


def get_accuracy(ys_pred_target, ys_target):
    ys_pred_target_labels = torch.argmax(ys_pred_target.view(-1, 2), dim=1)
    accuracy = (ys_pred_target_labels == ys_target).sum().item() / len(ys_target)
    return accuracy


def get_baseline_predictions(model, xs_meta, ys_meta, xs_target, i):
    ys_meta = ys_meta.detach()[i].flatten()
    xs_meta = xs_meta.detach()[i]
    xs_target = xs_target.detach()[i]
    # print(ys_meta)
    if ys_meta.min() == ys_meta.max():
        predictions = np.ones(xs_target.shape[0]) * ys_meta[0].numpy()

    else:
        model.fit(X=xs_meta, y=ys_meta)
        predictions = model.predict(X=xs_target)
    return predictions


def get_baseline_accuracy(model, bs, xs_meta, ys_meta, xs_target, ys_target):
    ys_lr_target_labels = [get_baseline_predictions(model,
                                                    xs_meta,
                                                    ys_meta,
                                                    xs_target,
                                                    i) for i in range(bs)]

    ys_lr_target_labels = np.array(ys_lr_target_labels).flatten()
    accuracy = (ys_lr_target_labels == np.array(ys_target)).sum().item() / len(ys_target)
    return accuracy


def main(save_no):
    BASEDIR = '.'
    dir_path = f'{BASEDIR}/saves'
    files = [f for f in os.listdir(dir_path) if os.path.isdir(f'{dir_path}/{f}')]
    existing_saves = sorted([int(f[5:]) for f in files if f.startswith("save")])  # format: save_{number}
    save_no = existing_saves[save_no]
    save_dir = f'{BASEDIR}/saves/save_{save_no}'
    print(f'Loading model at {save_dir = }')

    state_dict = torch.load(os.path.join(save_dir, 'model.pt'))
    model = ModelHolder(cfg_all=get_config(cfg_file=f'{save_dir}/defaults.toml'))
    model.load_state_dict(state_dict['model_state_dict'])

    cfg = toml.load(os.path.join(save_dir, 'defaults.toml'))["DL_params"]

    num_rows = 10 #cfg["num_rows"]
    num_targets = cfg["num_targets"]
    ds_group = -1  # cfg["ds_group"]

    bs = 1
    baseline_models = [LogisticRegression(max_iter=1000), SVC(), ZeroModel()]
    baseline_model_names = ['LR']  # , 'SVC', "Baseline model"]

    for num_cols in range(1, 20, 2):
        acc = []
        baseline_acc = {name: [] for name in baseline_model_names}
        val_dl = AllDatasetDataLoader(bs=bs, num_rows=num_rows, num_targets=5,
                                      num_cols=num_cols, ds_group=ds_group, split="val")

        for j in range(2000):
            # Fewshot predictions
            xs_meta, xs_target, ys_meta, ys_target = get_batch(val_dl, num_rows)
            ys_pred_target = get_predictions(xs_meta=xs_meta, xs_target=xs_target, ys_meta=ys_meta, model=model)
            acc.append(get_accuracy(ys_pred_target, ys_target))

            # Predictions for baseline models
            for base_model, model_name in zip(baseline_models, baseline_model_names):
                baseline_acc[model_name].append(get_baseline_accuracy(
                    model=base_model,
                    bs=bs,
                    xs_meta=xs_meta,
                    xs_target=xs_target,
                    ys_meta=ys_meta,
                    ys_target=ys_target
                ))
        # print('---------------------')
        # print(f'num_cols: {num_cols}')
        # print(f'Fewshot mean acc: {np.mean(acc):.3f}')
        # for model_name in baseline_model_names:
        #     print(f'{model_name} mean acc: {np.mean(baseline_acc[model_name]):.3f}')
        print(f'{np.mean(acc) - np.mean(baseline_acc["LR"]):.3f}')


if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    # save_number = int(input("Enter save number:\n"))
    # main(save_no=save_number)

    for eval_no in range(3):
        print()
        print("Eval number", eval_no)
        main(save_no=-(eval_no + 1))

