
#%%
%load_ext autoreload
%autoreload 2

import torch
import pickle
import sys
sys.path.insert(0, '/Users/kasiakobalczyk/FairFewshot/Fewshot')
from Fewshot.main import *
from Fewshot.dataloader import d2v_pairer, AdultDataLoader, DummyDataLoader, SimpleDataset
from Fewshot.data_generation import MLPDataLoader
from Fewshot.config import get_config
import os
import toml

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier

sns.set_style('ticks')
sns.set_palette('Set2')
save_no = 43
base_dir = '.'
save_dir = os.path.join(base_dir, f'saves/save_{save_no}')
model_save = torch.load(os.path.join(save_dir, 'model.pt'))

model = ModelHolder()
model.load_state_dict(model_save['model_state_dict'])

all_cfgs = toml.load(os.path.join(save_dir, 'defaults.toml'))
cfg = all_cfgs["DL_params"]

bs = cfg["bs"]
num_rows = cfg["num_rows"]
num_targets = cfg["num_targets"]
num_cols = cfg["num_cols"]
shuffle_cols = cfg["shuffle_cols"]

#%%
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


#%%

def get_fewshot_acc(batch):
    model_id, xs_meta, xs_target, ys_meta, ys_target = batch
    embed_meta, pos_enc = get_embedding(xs_meta, ys_meta, model)
    ys_pred_target = get_preddictions(xs_target, embed_meta, pos_enc)
    return get_accuracy(ys_pred_target, ys_target)

baseline_models = [LogisticRegression(max_iter=1000), SVC()]
baseline_model_names = ['LR', 'SVC']

data_names = ['teaching']

for num_cols in [5]:
    train_acc = []
    val_acc = []
    baseline_acc = dict(zip(baseline_model_names, [[] for i in range(2)]))
    val_dl = DummyDataLoader(bs=bs, num_rows=num_rows, num_targets=num_targets, num_cols=num_cols, data_names=data_names, shuffle_cols=shuffle_cols, split="val")
    train_dl = DummyDataLoader(bs=bs, num_rows=num_rows, num_targets=num_targets, num_cols=num_cols, data_names=data_names, shuffle_cols=shuffle_cols, split="train")

    for j in range(200):
        # Fewshot predictions
        train_batch = get_batch(train_dl)
        val_batch = get_batch(val_dl)
        model_id, xs_meta, xs_target, ys_meta, ys_target = val_batch
        train_acc.append(get_fewshot_acc(train_batch))
        val_acc.append(get_fewshot_acc(val_batch))

         # Predictions for baseline models - fewshot
        for base_model, model_name in zip(baseline_models, baseline_model_names):
            baseline_acc[model_name].append(get_baseline_accuracy(
                model=base_model,
                xs_meta=xs_meta,
                xs_target=xs_target,
                ys_meta=ys_meta,
                ys_target=ys_target
            )) 

        # Predicitions for baseline models - fitted on the whole datasets
        # TODO
        
    print('---------------------')
    print(f'num_cols: {num_cols}') 
    print(f'Fewshot train mean acc: {np.mean(train_acc):.3f}')
    print(f'Fewshot val mean acc: {np.mean(val_acc):.3f}')
    for model_name in baseline_model_names:
        print(f'{model_name} mean acc: {np.mean(baseline_acc[model_name]):.3f}')


# # %%
# # # Base line
# def get_fit_full_mod(model, data_name):
#     dataset = SimpleDataset(data_name)
#     X = dataset.data_train.iloc[:, :-1].values
#     y = dataset.data_train.iloc[:, -1].values
#     return model.fit(X, y)

# def _get_full_acc(data_name, model_name, x_target, y_target):
#     mod = full_fitted_models[model_name][data_name]
#     pred = mod.predict(x_target)
#     return (pred == np.array(y_target)).sum() / len(y_target)

# def get_full_acc(baseline_full_acc, model_id, xs_target, ys_target):
#     for mod_name in baseline_model_names:
#         baseline_full_acc[mod_name].append(np.mean([
#             _get_full_acc(
#                 model_id[i],
#                 mod_name,
#                 xs_target[i], 
#                 ys_target[i]
#             ) for i in range(bs)]))

# val_dl = DummyDataLoader(
#     bs=bs, num_rows=num_rows, num_targets=num_targets, num_cols=3, shuffle_cols=False, split="val")
    
# model_id, xs_meta, xs_target, ys_meta, ys_target = get_batch(val_dl)
# datanames = [d.data_name for d in val_dl.datasets]
# full_fitted_models = {}
# baseline_full_acc = {}
# for mod, mod_name in zip(baseline_models, baseline_model_names):
#     full_fitted_models[mod_name] = {}
#     baseline_full_acc[mod_name] = []
#     for d in datanames:
#             full_fitted_models[mod_name][d] = get_fit_full_mod(mod, d)

# for i in range(200):
#     model_id, xs_meta, xs_target, ys_meta, ys_target = get_batch(val_dl)
#     ys_target = torch.split(ys_target, num_targets)
#     get_full_acc(baseline_full_acc, model_id, xs_target, ys_target)

# for mod_name in baseline_model_names:
#     baseline_full_acc[mod_name] = np.mean(baseline_full_acc[mod_name])

4# %%
embed_meta_ls = []
pos_enc_ls = []
model_id_ls = []
num_cols_ls = []
num_cols = cfg['num_cols']

for n in [5]:
    #val_dl = AllDatasetDataLoader(bs=bs, num_rows=num_rows, num_targets=num_targets, num_cols=num_cols)
    dl = DummyDataLoader(bs=bs, num_rows=num_rows, num_targets=num_targets, num_cols=n, split="val")

    for i in range(100):
        model_id, xs_meta, xs_target, ys_meta, ys_target = get_batch(dl)
        embed_meta, pos_enc = get_embedding(xs_meta, ys_meta, model)
        pos_enc_ls.append(pos_enc)
        embed_meta_ls.append(embed_meta)
        model_id_ls.append(model_id)
        num_cols_ls.append([n] * bs)


model_id_ls = [item for sublist in model_id_ls for item in sublist]
num_cols_ls = [item for sublist in num_cols_ls for item in sublist]
embed_meta = torch.stack(embed_meta_ls).detach().reshape(100 * bs, 32)
# pos_enc = torch.stack(pos_enc_ls).detach().reshape(100 * bs, num_cols * 3)
#%%
print(np.unique(model_id_ls))

#%%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

reducer = TSNE(n_components=3)
# reducer = PCA(n_components=3)
reduced_embeddings = reducer.fit_transform(scaler.fit_transform(embed_meta))

reduced_embeddings_df = pd.DataFrame({
    'dim_1' : reduced_embeddings[:, 0],
    'dim_2' : reduced_embeddings[:, 1],
    'dim_3' : reduced_embeddings[:, 2],
    'model_id' : np.array(model_id_ls).astype(str),
    'num_cols' : np.array(num_cols_ls).astype(str),
})

plot_df = reduced_embeddings_df
fig, axs = plt.subplots(figsize=(7, 5))
sns.scatterplot(plot_df, x='dim_1', y='dim_2', ax=axs, hue='model_id')
# sns.scatterplot(plot_df, x='dim_2', y='dim_3', ax=axs[1], hue='model_id', legend=False)
# sns.scatterplot(plot_df, x='dim_1', y='dim_3', ax=axs[2], hue='model_id', legend=False)
axs.legend(bbox_to_anchor=(1.05, 1))
fig.tight_layout()
plt.show()
# %%
# from sklearn.cluster import KMeans

# km = KMeans(n_clusters=7, n_init=20)
# km.fit(reduced_embeddings_df.iloc[:, :2])
# reduced_embeddings_df['cluster'] = km.predict(reduced_embeddings_df.iloc[:, :2])


# fig, ax = plt.subplots(figsize=(4, 4))
# sns.scatterplot(data=reduced_embeddings_df, x='dim_1', y='dim_2', ax=ax, hue='cluster', legend=False)
# fig.tight_layout()
# plt.show()
# # %%
# reduced_embeddings_df.groupby(['model_id', 'cluster'])['cluster'].count().unstack()
# # %%
# #%%
# num_cols_range = [2, 5, 10, 15, 20]
# model_acc = []
# baseline_models = [LogisticRegression(), SVC()]
# baseline_model_names = ['LR', 'SVC']
# baseline_acc = dict(zip(baseline_model_names, [[] for i in range(2)]))

# for num_cols in num_cols_range:
#     print("-----------------------------")
#     print("Number of features:", num_cols)
#     dl = MLPDataLoader(
#         bs=bs, num_rows=num_rows, num_targets=num_targets, 
#         num_cols=num_cols, num_models=-1, config={}, 
#     )
#     # Predictions for fewshot
#     model_id, xs_meta, xs_target, ys_meta, ys_target = get_batch(dl)
#     embed_meta, pos_enc = get_embedding(xs_meta, ys_meta, model)
#     ys_pred_target = get_preddictions(xs_target, embed_meta, pos_enc)
#     model_acc.append(get_accuracy(ys_pred_target, ys_target))
#     print("Fewshot:", model_acc[-1])
#     # Predictions for baseline models
#     for base_model, model_name in zip(baseline_models, baseline_model_names):
#         baseline_acc[model_name].append(get_baseline_accuracy(
#             model=base_model,
#             xs_meta=xs_meta,
#             xs_target=xs_target,
#             ys_meta=ys_meta,
#             ys_target=ys_target
#         ))
#         print(f'{model_name}: {baseline_acc[model_name][-1]}')
    
# accuracy_summary_df = pd.DataFrame({
#     'N_features': num_cols_range,
#     'Fewshot': model_acc,
#     'LR' : baseline_acc['LR'],
#     'SVC' : baseline_acc['SVC']
# })
# accuracy_summary_df

# # %%

# new_dl = AdultDataLoader(bs=bs, num_rows=num_rows, num_targets=num_targets, flip=True)
# xs_meta, xs_target, ys_meta, ys_target  = get_batch(new_dl)
# xs_meta = torch.nn.functional.normalize(xs_meta, dim=1)
# xs_target = torch.nn.functional.normalize(xs_target, dim=1)
# embed_meta, pos_enc = get_embedding(xs_meta, ys_meta, model)
# ys_pred_target = get_preddictions(xs_target, embed_meta, pos_enc)
# print('Fewshot:', get_accuracy(ys_pred_target, ys_target))

# for base_model, model_name in zip(baseline_models, baseline_model_names):
#     acc = get_baseline_accuracy(
#         model=base_model,
#         xs_meta=xs_meta,
#         xs_target=xs_target,
#         ys_meta=ys_meta,
#         ys_target=ys_target
#     )
#     print(f'{model_name}: {baseline_acc[model_name][-1]}')


                              
#  # %%

# dl = MLPDataLoader(bs=bs, num_rows=num_rows, num_targets=num_targets, num_cols=num_cols, config={}, num_models=2)
# model_id, xs_meta, xs_target, ys_meta, ys_target = get_batch(dl)
# embed_meta, pos_enc = get_embedding(xs_meta, ys_meta, model)

# reducer = TSNE(n_components=3)
# # reducer = PCA(n_components=3)
# reduced_embeddings = reducer.fit_transform(embed_meta.detach())

# reduced_embeddings_df = pd.DataFrame({
#     'dim_1' : reduced_embeddings[:, 0],
#     'dim_2' : reduced_embeddings[:, 1],
#     'dim_3' : reduced_embeddings[:, 2],
#     'model_id' : np.array(model_id).astype(str)
# })
# fig, axs = plt.subplots(1, 3, figsize=(12, 4))
# sns.scatterplot(data=reduced_embeddings_df, x='dim_1', y='dim_2', hue = 'model_id', ax=axs[0])
# sns.scatterplot(data=reduced_embeddings_df, x='dim_2', y='dim_3', hue = 'model_id', ax=axs[1])
# sns.scatterplot(data=reduced_embeddings_df, x='dim_1', y='dim_3', hue = 'model_id', ax=axs[2])
# fig.tight_layout()
# plt.show()

# # %%
# from scipy.linalg import logm

# corrs = np.array([
#     np.corrcoef(
#         np.array(xs_meta.detach()[i]).T, 
#         np.array(ys_meta.detach())[i].flatten()).flatten()
#     for i in range(bs)
# ])

# sns.clustermap(corrs, col_cluster=False)

# # %%
# from sklearn.cluster import KMeans

# sum_of_squared_distances = []
# K = range(1,15)
# for k in K:
#     km = KMeans(n_clusters=k, n_init=10)
#     km = km.fit(corrs)
#     sum_of_squared_distances.append(km.inertia_)

# plt.plot(K, sum_of_squared_distances, 'bx-')
# plt.xlabel('k')
# plt.ylabel('Sum_of_squared_distances')
# plt.title('Elbow Method For Optimal k')
# plt.show()

# # %%

# km = KMeans(n_clusters=2, n_init=10)
# clusters = km.fit_predict(corrs)

# fig, axs = plt.subplots(1, 3, figsize=(12, 4))
# sns.scatterplot(data=reduced_embeddings_df, x='dim_1', y='dim_2', hue = clusters, ax=axs[0])
# sns.scatterplot(data=reduced_embeddings_df, x='dim_2', y='dim_3', hue = clusters, ax=axs[1])
# sns.scatterplot(data=reduced_embeddings_df, x='dim_1', y='dim_3', hue = clusters, ax=axs[2])
# fig.tight_layout()
# plt.show()
# %%
