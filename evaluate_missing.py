#%%
%load_ext autoreload
%autoreload 2

import torch
import sys
sys.path.insert(0, '/Users/kasiakobalczyk/FairFewshot/Fewshot')
from Fewshot.main import *
from Fewshot.dataloader import d2v_pairer, MissingDataLoader
import os
import toml
import random

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style('ticks')
sns.set_palette('Set2')
save_no = 53
base_dir = '.'
save_dir = os.path.join(base_dir, f'saves/save_{save_no}')
model_save = torch.load(os.path.join(save_dir, 'model.pt'))

all_cfgs = toml.load(os.path.join(save_dir, 'defaults.toml'))
cfg = all_cfgs["DL_params"]

bs = cfg["bs"]
num_rows = cfg["num_rows"]
num_targets = cfg["num_targets"]
num_cols = cfg["num_cols"]
shuffle_cols = cfg["shuffle_cols"]
data_name = cfg["data_names"][0]
miss_rate = cfg["miss_rate"]

model = ModelHolder()
model.load_state_dict(model_save['model_state_dict'])


# %%
def get_batch(dl):
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

def get_embedding(xs_meta, ys_meta, model):
    pairs_meta = d2v_pairer(xs_meta, ys_meta)
    embed_meta, pos_enc = model.forward_meta(pairs_meta)
    return embed_meta, pos_enc

def get_preddictions(xs_target, embed_meta, pos_enc):
    ys_pred_target =  model.forward_target(xs_target, embed_meta, pos_enc)
    return ys_pred_target

def get_accuracy(ys_pred_target, ys_target):
    ys_pred_target_labels = torch.argmax(ys_pred_target.view(-1, 2), dim=1)
    accuracy = (ys_pred_target_labels == ys_target).sum().item() / len(ys_target)
    return accuracy
# %%
# %%
np.random.seed(0)
random.seed(0)

train_dl = MissingDataLoader(
    bs=bs, num_rows=num_rows, num_targets=num_targets, shuffle_cols=False,
    data_name=data_name, miss_rate=miss_rate, split="train"
)

val_dl = MissingDataLoader(
    bs=bs, num_rows=num_rows, num_targets=num_targets, shuffle_cols=False,
    data_name=data_name, miss_rate=miss_rate, split="val"
)


# %%
val_data = train_dl.dataset.X_val
val_mask = train_dl.dataset.val_mask

train_data = val_dl.dataset.X_train
train_label = val_dl.dataset.y_train
train_mask = val_dl.dataset.train_mask

train_data = pd.DataFrame(train_data).reset_index(drop=True)
val_data = pd.DataFrame(val_data).reset_index(drop=True)
train_data[train_mask == 0] = None
val_data[val_mask == 0] = None

#%%
from sklearn.linear_model import LogisticRegression
# Logistic model fitted on the complete dataset
clf = LogisticRegression()

def get_fewshot_acc(batch):
    try: 
        model_id, xs_meta, xs_target, ys_meta, ys_target = batch
    except: 
        xs_meta, xs_target, ys_meta, ys_target = batch
    embed_meta, pos_enc = get_embedding(xs_meta, ys_meta, model)
    ys_pred_target = get_preddictions(xs_target, embed_meta, pos_enc)
    return get_accuracy(ys_pred_target, ys_target) * 100

def fit_full_baseline_model(patterns, model):
    pattern = patterns[0]
    X = train_data.iloc[:, np.where(pattern == 1)[0]].dropna(axis=0)
    y = train_label[X.index]
    return model.fit(X, y)

def get_full_baseline_accuracy(model, patterns, x_meta, y_meta, x_target, y_target):
    pattern = patterns[0]
    X = train_data.iloc[:, np.where(pattern == 1)[0]].dropna(axis=0)
    y = train_label[X.index]
    X = np.concatenate([X.values, np.array(x_meta)])
    y = np.concatenate([y, np.array(y_meta)])
    model.fit(X, y)
    pred = model.predict(x_target)
    return (pred == np.array(y_target)).sum() / len(y_target) * 100

def get_fewshot_baseline_accuracy(model, x_meta, y_meta, x_target, y_target):
    if len(np.unique(y_meta)) == 1:
        if y_meta[0] == 0:
            pred = np.zeros(shape=y_target.shape)
        else:
            pred = np.ones(shape=y_target.shape)
    else:
        model.fit(x_meta, y_meta)
        pred = model.predict(x_target)
    return (pred == np.array(y_target)).sum() / len(y_target) * 100

our_train_acc = []
our_val_acc = []
full_lr_acc = []
fewshot_lr_acc = []
for j in range(400):
    # Fewshot predictions
    val_batch = get_batch(val_dl)
    train_batch = get_batch(train_dl)
    patterns, xs_meta, xs_target, ys_meta, ys_target = val_batch
    our_val_acc.append(get_fewshot_acc(val_batch))
    our_train_acc.append(get_fewshot_acc(train_batch))
    ys_target = ys_target.reshape(bs, num_targets)
    for i in range(bs):
        full_lr_acc.append(get_full_baseline_accuracy(clf, patterns, x_meta=xs_meta[i], y_meta=ys_meta[i], x_target=xs_target[i], y_target=ys_target[i]))
        fewshot_lr_acc.append(get_fewshot_baseline_accuracy(clf, x_meta=xs_meta[i], y_meta=ys_meta[i], x_target=xs_target[i], y_target=ys_target[i]))

print(f'Our train fewshot: {np.mean(our_train_acc):.2f}')
print(f'Our val fewshot: {np.mean(our_val_acc):.2f}')
print(f'LR full data: {np.mean(full_lr_acc):.2f}')
print(f'LR fewshot data: {np.mean(fewshot_lr_acc):.2f}')
# %%
print(val_dl.dataset.X_train.shape)
print(val_dl.dataset.X_val.shape)
# %%
