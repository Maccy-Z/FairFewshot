import torch
import sys
sys.path.insert(0, '/Users/kasiakobalczyk/FairFewshot/Fewshot')
from Fewshot.main import *
from Fewshot.mydataloader import MyDataLoader
import os
import toml

import pandas as pd
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb

np.random.seed(0)
random.seed(0)
sns.set_style('ticks')
sns.set_palette('Set2')

all_data_names = os.listdir('./datasets/data')
all_data_names.remove('info.json')
all_data_names.remove('.DS_Store')
dl = MyDataLoader(bs=1, num_rows=5, num_targets=5, data_names=all_data_names)
num_col_dict = dict(zip(all_data_names, [d.num_cols for d in dl.datasets]))


def load_model(save_no):
    save_dir = os.path.join(f'./saves/save_{save_no}')
    model_save = torch.load(os.path.join(save_dir, 'model.pt'))
    all_cfgs = toml.load(os.path.join(save_dir, 'defaults.toml'))
    model = ModelHolder(cfg_all=all_cfgs)
    model.load_state_dict(model_save['model_state_dict'])
    return model, all_cfgs

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

def get_flat_embedding(model, xs_meta, ys_meta):
    pairs_meta = d2v_pairer(xs_meta, ys_meta)
    embed_meta, pos_enc = model.forward_meta(pairs_meta)
    return embed_meta, pos_enc

def get_flat_preddictions(model, xs_target, embed_meta, pos_enc):
    ys_pred_target =  model.forward_target(xs_target, embed_meta, pos_enc)
    return ys_pred_target

def get_flat_accuracy(ys_pred_target, ys_target):
    ys_pred_target_labels = torch.argmax(ys_pred_target.view(-1, 2), dim=1)
    accuracy = (ys_pred_target_labels == ys_target).sum().item() / len(ys_target)
    return accuracy

def get_baseline_predictions(model, xs_meta, ys_meta, xs_target, i):
    """
    Fits a model on meta (x, y) pairs and returns predictions on xs_target
    """
    ys_meta = ys_meta.detach()[i].flatten()
    xs_meta = xs_meta.detach()[i]
    xs_target = xs_target.detach()[i]
    try:
        model.fit(X=xs_meta, y=ys_meta)
        predictions = model.predict(X=xs_target)
    except ValueError:
        if len(np.unique(ys_meta)) == 1:
            if ys_meta[0] == 0:
                predictions = np.zeros(ys_meta.shape[0])
            else:
                predictions = np.ones(ys_meta.shape[0])
    return predictions

def get_baseline_accuracy(model, batch):
    """
    Fits the baseline models and returns accuracy of baseline model predictions
    on (x, y) meta-target data.
    """
    try: 
        _, xs_meta, xs_target, ys_meta, ys_target = batch
    except: 
        xs_meta, xs_target, ys_meta, ys_target = batch
    
    ys_lr_target_labels = np.array([
        get_baseline_predictions(
            model,
            xs_meta,
            ys_meta,
            xs_target,
            i
        ) for i in range(xs_meta.shape[0])]).flatten()
    accuracy = (
        ys_lr_target_labels == np.array(ys_target)).sum().item() / len(ys_target)
    return accuracy

def get_flat_acc(model, batch):
    """
    Makes predictions and returns the accuracy on meta-target data
    for out fewshot model.
    """
    try: 
        _, xs_meta, xs_target, ys_meta, ys_target = batch
    except: 
        xs_meta, xs_target, ys_meta, ys_target = batch
    embed_meta, pos_enc = get_flat_embedding(model, xs_meta, ys_meta)
    ys_pred_target = get_flat_preddictions(model, xs_target, embed_meta, pos_enc)
    return get_flat_accuracy(ys_pred_target, ys_target)

def get_results_by_dataset(test_data_names, model, baseline_models, baseline_model_names):
    results = pd.DataFrame(columns=['data_name', 'model', 'num_cols', 'acc'])

    for data_name in test_data_names:
        num_cols = 2
        while num_cols <= num_col_dict[data_name]:
            # get batch 
            test_dl = MyDataLoader(
                bs=200, num_rows=cfg['num_rows'], 
                num_targets=cfg['num_targets'], data_names=[data_name], 
                num_cols=[num_cols], split="test"
            )
            batch = get_batch(test_dl, cfg["num_rows"])
            result = pd.DataFrame({
                'data_name': data_name, 
                'model': 'FLAT', 
                'num_cols': num_cols, 
                'acc': get_flat_acc(model, batch)
            }, index=[0])
            results = pd.concat([results, result])
            for base_model, model_name in zip(baseline_models, baseline_model_names):
                result = pd.DataFrame({
                    'data_name': data_name, 
                    'model': model_name, 
                    'num_cols': num_cols, 
                    'acc': get_baseline_accuracy(base_model, batch)
                }, index=[0])
                results = pd.concat([results, result])
            num_cols *= 2
    results.reset_index(drop=True, inplace=True)
    return results

def get_agg_results(test_data_names, model, baseline_models, baseline_model_names):
    results = pd.DataFrame(columns=['data_name', 'model', 'num_cols', 'acc'])
    num_cols = 2
    while num_cols <= 32:
        # get batch 
        test_dl = MyDataLoader(
            bs=200, num_rows=cfg['num_rows'], 
            num_targets=cfg['num_targets'], data_names=test_data_names, 
            num_cols=[num_cols], split="test"
        )
        batch = get_batch(test_dl, cfg["num_rows"])
        result = pd.DataFrame({
            'model': 'FLAT', 
            'num_cols': num_cols, 
            'acc': get_flat_acc(model, batch)
        }, index=[0])
        results = pd.concat([results, result])

        for base_model, model_name in zip(baseline_models, baseline_model_names):
            baseline_acc = get_baseline_accuracy(base_model, batch)
            result = pd.DataFrame({
                'model': model_name, 
                'num_cols': num_cols, 
                'acc': get_baseline_accuracy(base_model, batch)
            }, index=[0])
            results = pd.concat([results, result])
        num_cols *= 2
    results.reset_index(drop=True, inplace=True)
    return results

if __name__ == "__main__":

    save_no = 5

    model, all_cfg = load_model(save_no)
    cfg = all_cfg["DL_params"]

    train_data_names = cfg["train_data_names"]
    test_data_names= cfg["test_data_names"]

    baseline_models = [
        LogisticRegression(max_iter=1000), 
        KNeighborsClassifier(n_neighbors=2),
        # xgb.XGBClassifier()
    ]
    baseline_model_names = ['LR', 'kNN'] #'XGB']

    unseen_results = get_results_by_dataset(
        test_data_names=test_data_names,
        model=model,
        baseline_models=baseline_models,
        baseline_model_names=baseline_model_names,
    )

    seen_results = get_agg_results(
        test_data_names=train_data_names,
        model=model,
        baseline_models=baseline_models,
        baseline_model_names=baseline_model_names,
    )

    unseen_results.to_csv(f'./results/unseen_test_save_{save_no}.csv')
    seen_results.to_csv(f'./results/seen_test_save_{save_no}.csv')

    print("========================================")
    print("Test accuracy on unseen datasets")
    print(unseen_results.pivot(columns=['data_name', 'model'], index='num_cols', values='acc'))
    print()
    print("========================================")
    print("Test accuracy on seen datasets (aggregated)")
    print(seen_results.pivot(columns='model', index='num_cols', values='acc'))
    print()
    print("========================================")
    print("Test accuracy on unseen datasets (aggregated)")
    print(unseen_results.groupby(['num_cols', 'model'])['acc'].mean().unstack())


