#%%
%load_ext autoreload
%autoreload 2

import torch
import sys
sys.path.insert(0, '/Users/kasiakobalczyk/FairFewshot/Fewshot')
from Fewshot.main import *
from Fewshot.config import get_config
from Fewshot.mydataloader import MyDataLoader
import os
import toml

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

np.random.seed(0)
random.seed(0)
sns.set_style('ticks')
sns.set_palette('Set2')

save_no = 63
base_dir = '.'
save_dir = os.path.join(base_dir, f'saves/save_{save_no}')
model_save = torch.load(os.path.join(save_dir, 'model.pt'))

all_cfgs = toml.load(os.path.join(save_dir, 'defaults.toml'))
cfg = all_cfgs["DL_params"]
bs = cfg["bs"]
num_rows = cfg["num_rows"]
num_targets = cfg["num_targets"]
num_cols = cfg.get("num_cols")
shuffle_cols = cfg["shuffle_cols"]

model = ModelHolder()
model.load_state_dict(model_save['model_state_dict'])

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

def get_baseline_accuracy(model, xs_meta, ys_meta, xs_target, ys_target):
    """
    Fits the baseline models and returns accuracy of baseline model predictions
    on (x, y) meta-target data.
    """
    ys_lr_target_labels = np.array([
        get_baseline_predictions(
            model,
            xs_meta,
            ys_meta,
            xs_target,
            i
        ) for i in range(bs)]).flatten()
    accuracy = (
        ys_lr_target_labels == np.array(ys_target)).sum().item() / len(ys_target)
    return accuracy

def get_fewshot_acc(batch):
    """
    Makes predictions and returns the accuracy on meta-target data
    for out fewshot model.
    """
    try: 
        _, xs_meta, xs_target, ys_meta, ys_target = batch
    except: 
        xs_meta, xs_target, ys_meta, ys_target = batch
    embed_meta, pos_enc = get_embedding(xs_meta, ys_meta, model)
    ys_pred_target = get_preddictions(xs_target, embed_meta, pos_enc)
    return get_accuracy(ys_pred_target, ys_target)

# %%
# Accuracy
# ----------

baseline_models = [LogisticRegression(max_iter=1000)]#, SVC(), RandomForestClassifier(), GradientBoostingClassifier()]
baseline_model_names = ['LR'] #, 'SVC', 'RF', 'GB']

seen_data_names = cfg["train_data_names"] # Datasets used during training
unseen_data_names = ['heart-cleveland'] # Datasets for testing

num_cols_ls = list(range(2, 9))
results = pd.DataFrame()

for num_cols in num_cols_ls:
    seen_acc = []
    unseen_acc = []
    baseline_acc = dict(zip(baseline_model_names, [[] for i in range(1)]))
    seen_val_dl = MyDataLoader(
        bs=bs, num_rows=num_rows, num_targets=num_targets, 
        num_cols=[num_cols], data_names=seen_data_names, 
        shuffle_cols=shuffle_cols, split="test"
    )
    unseen_val_dl = MyDataLoader(
        bs=bs, num_rows=num_rows, num_targets=num_targets, 
        num_cols=[num_cols], data_names=unseen_data_names, 
        shuffle_cols=shuffle_cols, split="test"
    )
    
    for j in range(200):
        # Fewshot predictions
        seen_val_batch = get_batch(seen_val_dl)
        unseen_val_batch = get_batch(unseen_val_dl)
        try:
            model_id, xs_meta, xs_target, ys_meta, ys_target = unseen_val_batch
        except:
            xs_meta, xs_target, ys_meta, ys_target = unseen_val_batch
        seen_acc.append(get_fewshot_acc(seen_val_batch))
        unseen_acc.append(get_fewshot_acc(unseen_val_batch))

         # Predictions for baseline models - fewshot
        for base_model, model_name in zip(baseline_models, baseline_model_names):
            baseline_acc[model_name].append(get_baseline_accuracy(
                model=base_model,
                xs_meta=xs_meta,
                xs_target=xs_target,
                ys_meta=ys_meta,
                ys_target=ys_target
            )) 

    print('---------------------')
    print(f'num_cols: {num_cols}') 
    print(f'Fewshot seen mean acc: {np.mean(seen_acc):.3f}')
    print(f'Fewshot unseen mean acc: {np.mean(unseen_acc):.3f}')
    results.loc[num_cols, 'fewshot seen'] = np.mean(seen_acc)
    results.loc[num_cols, 'fewshot unseen'] = np.mean(unseen_acc)
    for model_name in baseline_model_names:
        print(f'{model_name} unseen mean acc: {np.mean(baseline_acc[model_name]):.3f}')
        results.loc[num_cols, model_name] = np.mean(baseline_acc[model_name])

results.index.name = 'num_col'
(results * 100).round(2)
# %%
# Embeddings
# ----------

embed_meta_ls = []
pos_enc_ls = []
model_id_ls = []
num_cols_ls = []

dl = MyDataLoader(
        bs=bs, num_rows=num_rows, num_targets=num_targets, 
        num_cols=None, data_names=seen_data_names + unseen_data_names, 
        shuffle_cols=shuffle_cols, split="test"
)
for i in range(400):
    model_id, xs_meta, xs_target, ys_meta, ys_target = get_batch(dl)
    embed_meta, pos_enc = get_embedding(xs_meta, ys_meta, model)
    pos_enc_ls.append(pos_enc)
    embed_meta_ls.append(embed_meta)
    model_id_ls.append(model_id)

model_id_ls = [item for sublist in model_id_ls for item in sublist]
embed_meta = torch.stack(embed_meta_ls).detach().reshape(400 * bs, 64)

scaler = StandardScaler()
reducer = TSNE(n_components=3, perplexity=100)
# reducer = PCA(n_components=3)
reduced_embeddings = reducer.fit_transform(scaler.fit_transform(embed_meta))

reduced_embeddings_df = pd.DataFrame({
    'dim_1' : reduced_embeddings[:, 0],
    'dim_2' : reduced_embeddings[:, 1],
    'dim_3' : reduced_embeddings[:, 2],
    'model_id' : np.array(model_id_ls).astype(str),
})

plot_df = reduced_embeddings_df
fig, axs = plt.subplots(3, 1, figsize=(4, 12), sharex=True, sharey=True)
sns.scatterplot(plot_df, x='dim_1', y='dim_2', ax=axs[0], hue='model_id', hue_order=seen_data_names + unseen_data_names)
sns.scatterplot(plot_df[~plot_df.model_id.isin(unseen_data_names)], x='dim_1', y='dim_2', ax=axs[1], hue='model_id', hue_order=seen_data_names)
sns.scatterplot(plot_df[plot_df.model_id.isin(unseen_data_names)], x='dim_1', y='dim_2', ax=axs[2], hue='model_id', palette=['C6'], hue_order=unseen_data_names)
fig.tight_layout()
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles=handles, labels=labels, bbox_to_anchor=(0.75, 1.1))
[ax.get_legend().remove() for ax in axs]
plt.show()
# %%
