#%%
%load_ext autoreload
%autoreload 2

import torch
import sys
import random
import numpy as np

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, '/Users/kasiakobalczyk/FairFewshot/Fewshot')
from Fewshot.main import *
from Fewshot.AllDataloader import SplitDataloader
from Fewshot.utils import get_batch, load_model, get_num_rows_cols

np.random.seed(0)
random.seed(0)
sns.set_style('ticks')
sns.set_palette('Set2')

save_no = 36

num_rows_dict, num_cols_dict = get_num_rows_cols()
model, cfg_all = load_model(save_no)
cfg = cfg_all["DL_params"]

# %%
# Embeddings
# ----------
split_file = cfg["split_file"]
split_file = f'/Users/kasiakobalczyk/FairFewshot/datasets/grouped_datasets/{split_file}'
train_dl = SplitDataloader(bs=0, num_rows=0, num_targets=0, binarise=cfg["binarise"],
                    ds_group=cfg["ds_group"], ds_split="train", 
                    split_file=split_file)
test_dl = SplitDataloader(bs=0, num_rows=0, num_targets=0, binarise=cfg["binarise"],
                    ds_group=cfg["ds_group"], ds_split="test", 
                    split_file=split_file)

seen_datasets = train_dl.all_datasets
unseen_datasets = test_dl.all_datasets
seen_data_names = [str(d) for d in seen_datasets]
unseen_data_names = [str(d) for d in unseen_datasets]
datasets = seen_datasets + unseen_datasets
datanames = seen_data_names + unseen_data_names

#%%
def get_embeddings(num_rows_func, num_samples=50):
    embed_meta_ls = []
    weight_meta_ls = []
    model_id_ls = []
    num_cols_ls = []

    for d in datanames:
        d = str(d)
        n_col = num_cols_dict[d]
        num_rows = num_rows_func(d)
        dl = SplitDataloader(
                bs=1, num_rows=num_rows, num_targets=0, 
                num_cols=[n_col - 1, n_col], ds_group=[d],
                #num_1s={'meta': num_rows // 2, 'target': 0}
        )
        for i in range(num_samples):
            model_id, xs_meta, xs_target, ys_meta, ys_target = get_batch(dl, num_rows)
            pairs_meta = d2v_pairer(xs_meta, ys_meta)
            with torch.no_grad():
                embed_meta, pos_enc = model.forward_meta(pairs_meta)
                layer_weights, lin_weights = model.weight_model(embed_meta)
                layer_weights
                first_layer_weights = layer_weights[0][0][0].flatten()
                second_layer_weights = layer_weights[0][1][0].flatten()
                first_layer_bias = layer_weights[0][0][3]
                second_layer_bias = layer_weights[0][1][3]
                layer_weights = torch.concat([
                    first_layer_weights, second_layer_weights, 
                    #first_layer_bias, #second_layer_bias
                ])
                shape = layer_weights.shape[0]
            weight_meta_ls.append(layer_weights)
            embed_meta_ls.append(embed_meta)
            model_id_ls.append(model_id)

    model_id_ls = [item for sublist in model_id_ls for item in sublist]
    embed_meta = torch.stack(embed_meta_ls).detach().reshape(
        num_samples * (len(datanames)), 64)
    weight_meta = torch.stack(weight_meta_ls).detach().reshape(
        num_samples * (len(datanames)), shape)

    return embed_meta, weight_meta, model_id_ls

def get_reduced_df(embed, model_id_ls):
    scaler = StandardScaler()
    reducer = TSNE(n_components=2, perplexity=100, random_state=0)
    # reducer = PCA(n_components=3)
    reduced_embeddings = reducer.fit_transform(scaler.fit_transform(embed))
    reduced_embeddings_df = pd.DataFrame({
        'dim_1' : reduced_embeddings[:, 0],
        'dim_2' : reduced_embeddings[:, 1],
        'model_id' : np.array(model_id_ls).astype(str),
    })
    return reduced_embeddings_df

def plot_embeddings(
        ax, reduced_embeddings_df, plot_centroids=False, 
        annot=False, alpha=0.5
    ):
    sns.set_style('whitegrid')
    
    all_data_names = seen_data_names + unseen_data_names
    num_datasets = len(all_data_names)
    colors = sns.husl_palette(num_datasets)
    plot_df = reduced_embeddings_df
    sns.scatterplot(
        plot_df, x='dim_1', y='dim_2', hue='model_id', 
        legend=False, ax=ax, size=1, alpha=alpha, palette=colors)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])

    if plot_centroids:
        mean_embeddings = reduced_embeddings_df.groupby('model_id').median()
        plot_df = mean_embeddings.loc[plot_df.model_id.unique(), :]
        ax.scatter(x=plot_df.dim_1, y=plot_df.dim_2, color=colors, s=50, edgecolors='white')
    if annot:
        x = mean_embeddings['dim_1']
        y = mean_embeddings['dim_2']
        for i, txt in enumerate(mean_embeddings.index):
                offset = 0.8
                if txt in ['heart-cleveland', 'breast-cancer']:
                    offset = 2
                ax.annotate(txt, (x[i], y[i] + offset), ha='center')#(x[i] - np.random.rand(1) * 0.8, y[i] + np.random.rand(1) * 0.5))

#%%
num_rows = []
all_model_id = []
all_embed_ls = []
for i, n in enumerate([10, 25, 50, 100]):
    num_rows_func = lambda d: min(n, num_rows_dict[d] // 2)
    embed_meta, weight_meta, model_id_ls = get_embeddings(num_rows_func, num_samples=50)
    num_rows += [n] * embed_meta.shape[0]
    all_model_id += model_id_ls
    all_embed_ls.append(embed_meta)

all_embed_meta = torch.concat(all_embed_ls)
all_embed_meta.shape

reduced_embeddings_df = get_reduced_df(all_embed_meta, all_model_id)
reduced_embeddings_df['num_rows'] = num_rows
#%%
fig, axs  = plt.subplots(1, 4, figsize=(12, 3))
for n, ax in zip([10, 25, 50, 100], axs.ravel()):
    plot_embeddings(ax, reduced_embeddings_df[reduced_embeddings_df.num_rows == n])
    ax.set_title(f'$n={n}$', fontsize=14)
plt.tight_layout()
plt.savefig('figures/medical_embeddings_byrows.pdf', bbox_inches='tight')
plt.show()
# %%
fig, ax = plt.subplots(figsize=(8, 8))
plot_embeddings(
    ax, reduced_embeddings_df[reduced_embeddings_df.num_rows == 100],
    plot_centroids=True, annot=True, alpha=0.35)
plt.tight_layout()
plt.savefig('figures/medical_embeddings.pdf', bbox_inches='tight')
plt.show()

#%%
unseen_datasets

#%%
def get_variance(x):
    center = x.mean(axis=0)
    var = ((x - center)**2).sum(axis=1).mean()
    return var

stat_df = pd.DataFrame()
stat_df['within_clust_var'] = reduced_embeddings_df.groupby('model_id')[['dim_1', 'dim_2']].apply(lambda x: get_variance(x))
stat_df['num_rows'] = stat_df.index.map(num_rows_dict)
stat_df.sort_values('within_clust_var')

print('global variance:', get_variance(reduced_embeddings_df[['dim_1', 'dim_2']]))
print('mean within cluster variance:', stat_df['within_clust_var'].mean())

stat_df['within_clust_var'].sort_values()
# %%
