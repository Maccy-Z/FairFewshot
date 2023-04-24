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

all_data_names = os.listdir('./datasets/data')
all_data_names.remove('info.json')
all_data_names.remove('.DS_Store')
test_data_names = np.random.choice(all_data_names, size= len(all_data_names) // 5, replace=False)

medical_datasets = [
    'acute-inflammation', 'acute-nephritis', 'arrhythmia',
    'blood', 'breast-cancer', 'breast-cancer-wisc', 'breast-cancer-wisc-diag', 
    'breast-cancer-wisc-prog', 'breast-tissue', 'cardiotocography-3clases', 
    'dermatology', 'echocardiogram', 'fertility', 'heart-cleveland', 
    'heart-hungarian', 'heart-switzerland', 'heart-va', 'hepatitis', 'horse-colic',
    'ilpd-indian-liver', 'lung-cancer', 'lymphography', 'mammographic', 
    'parkinsons', 'post-operative', 'primary-tumor', 'spect', 'spectf', 
    'statlog-heart', 'thyroid', 'vertebral-column-2clases'
]

dl = MyDataLoader(bs=1, num_rows=5, num_targets=5, data_names=all_data_names)
n_col = dict(zip(all_data_names, [d.num_cols for d in dl.datasets]))


financial_datasets = [
    'statlog-german-credit', 'statlog-australian-credit', 'credit-approval', 'bank']

[n_col[d] for d in ['acute-inflammation', 'hepatitis', 'blood', 'dermatology', 'vertebral-column-2clases']]

#%%
save_no = 2
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

model = ModelHolder(cfg_all=all_cfgs)
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

all_data_names = cfg['all_data_names']
unseen_data_names = cfg['test_data_names']
seen_data_names = list(set(all_data_names).difference(set(unseen_data_names)))
print('Seen', seen_data_names)
print('Unseen', unseen_data_names)

# %%
# Unseen
# --------
results = dict()
for unseen_data_name in unseen_data_names:
    print("==============")
    print(unseen_data_name)
    num_cols = 2
    results[unseen_data_name] = pd.DataFrame()

    while num_cols <= n_col[unseen_data_name]:
        unseen_acc = []
        baseline_acc = dict(zip(baseline_model_names, [[] for i in range(1)]))

        unseen_val_dl = MyDataLoader(
            bs=bs, num_rows=num_rows, num_targets=num_targets, 
            num_cols=[num_cols], data_names=[unseen_data_name], 
            shuffle_cols=shuffle_cols, split="test"
        )
        
        for j in range(200):
            unseen_val_batch = get_batch(unseen_val_dl)
            try:
                model_id, xs_meta, xs_target, ys_meta, ys_target = unseen_val_batch
            except:
                xs_meta, xs_target, ys_meta, ys_target = unseen_val_batch
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
        print(f'Fewshot unseen mean acc: {np.mean(unseen_acc):.3f}')
        results[unseen_data_name].loc[num_cols, 'fewshot unseen'] = np.mean(unseen_acc)
        for model_name in baseline_model_names:
            print(f'{model_name} unseen mean acc: {np.mean(baseline_acc[model_name]):.3f}')
            results[unseen_data_name].loc[num_cols, model_name] = np.mean(baseline_acc[model_name])
        num_cols *= 2

    results[unseen_data_name].index.name = 'num_col'
    results[unseen_data_name] = (results[unseen_data_name] * 100).round(2)

#%%
# %%
# Seen accuracy
# --------
seen_results = dict()
print("==============")
num_cols = 2
seen_results = pd.DataFrame()

while num_cols <= 32:
    seen_acc = []
    baseline_acc = dict(zip(baseline_model_names, [[] for i in range(1)]))
    seen_val_dl = MyDataLoader(
        bs=bs, num_rows=num_rows, num_targets=num_targets, 
        num_cols=[num_cols], data_names=seen_data_names, 
        shuffle_cols=shuffle_cols, split="test"
    )
    
    for j in range(200):
        seen_val_batch = get_batch(seen_val_dl)
        try:
            model_id, xs_meta, xs_target, ys_meta, ys_target = seen_val_batch
        except:
            xs_meta, xs_target, ys_meta, ys_target = seen_val_batch
        seen_acc.append(get_fewshot_acc(seen_val_batch))

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
    seen_results.loc[num_cols, 'fewshot seen'] = np.mean(seen_acc)
    for model_name in baseline_model_names:
        print(f'{model_name} seen mean acc: {np.mean(baseline_acc[model_name]):.3f}')
        seen_results.loc[num_cols, model_name] = np.mean(baseline_acc[model_name])
    num_cols *= 2

seen_results.index.name = 'num_col'
seem_results = (seen_results * 100).round(2)


# %%
# Embeddings
# ----------

embed_meta_ls = []
pos_enc_ls = []
model_id_ls = []
num_cols_ls = []

for d in seen_data_names + unseen_data_names:
    dl = MyDataLoader(
            bs=bs, num_rows=20, num_targets=0, 
            num_cols=[n_col[d]], data_names=[d],
            shuffle_cols=shuffle_cols, split="train"
    )
    for i in range(100):
        model_id, xs_meta, xs_target, ys_meta, ys_target = get_batch(dl)
        embed_meta, pos_enc = get_embedding(xs_meta, ys_meta, model)
        pos_enc_ls.append(pos_enc)
        embed_meta_ls.append(embed_meta)
        model_id_ls.append(model_id)

model_id_ls = [item for sublist in model_id_ls for item in sublist]
embed_meta = torch.stack(embed_meta_ls).detach().reshape(100 * bs * len(unseen_data_names + seen_data_names), 64)

#%%
scaler = StandardScaler()
reducer = TSNE(n_components=3, perplexity=100, random_state=0)
# reducer = PCA(n_components=3)
reduced_embeddings = reducer.fit_transform(scaler.fit_transform(embed_meta))
reduced_embeddings_df = pd.DataFrame({
    'dim_1' : reduced_embeddings[:, 0],
    'dim_2' : reduced_embeddings[:, 1],
    'dim_3' : reduced_embeddings[:, 2],
    'model_id' : np.array(model_id_ls).astype(str),
})
#%%
plot_df = reduced_embeddings_df
fig, axs = plt.subplots(3, 1, figsize=(4, 12), sharex=True, sharey=True)
# sns.scatterplot(plot_df, x='dim_1', y='dim_2', ax=axs[0], hue='model_id', hue_order=seen_data_names + unseen_data_names)
sns.scatterplot(plot_df[plot_df.model_id.isin(seen_data_names)], x='dim_1', y='dim_2', ax=axs[1], hue='model_id')#, hue_order=seen_data_names + unseen_data_names)
sns.scatterplot(plot_df[plot_df.model_id.isin(unseen_data_names)], x='dim_1', y='dim_2', ax=axs[2], hue='model_id')#, hue_order=seen_data_names + unseen_data_names)
# fig.tight_layout()
# handles, labels = axs[0].get_legend_handles_labels()
# fig.legend(handles=handles, labels=labels, bbox_to_anchor=(0.75, 1.1))
[ax.get_legend().remove() for ax in axs]
plt.show()
# %%
#%%
sns.scatterplot(plot_df[plot_df.model_id.isin(unseen_data_names)], x='dim_1', y='dim_2', hue='model_id')
#%%
sns.scatterplot(plot_df[plot_df.model_id.isin([
    'thyroid', 'primary-tumor', 'parkinsons', 
    'breast-cancer-wisc', 'mammographic', 'fertility'
])], x='dim_1', y='dim_2', hue='model_id')
# %%
sns.scatterplot(plot_df[plot_df.model_id.isin(seen_data_names)],  x='dim_1', y='dim_2', hue='model_id')
# %%
mean_embeddings = reduced_embeddings_df.groupby('model_id').mean().drop('arrhythmia')
x = mean_embeddings['dim_1']
y = mean_embeddings['dim_2']

fig, ax = plt.subplots(figsize = (10, 10))
mean_embeddings.plot(x='dim_1', y='dim_2', kind = 'scatter', ax=ax)
for i, txt in enumerate(mean_embeddings.index):
    ax.annotate(txt, (x[i], y[i]))
plt.show()
# %%
from sklearn.cluster import KMeans

fit_df = reduced_embeddings_df[reduced_embeddings_df.model_id.isin(seen_data_names)].groupby('model_id').mean()

km = KMeans(n_clusters = 3)
clust = km.fit_predict(fit_df[['dim_1', 'dim_2', 'dim_3']])
#%%

fit_df[clust == 0].index

# %%
