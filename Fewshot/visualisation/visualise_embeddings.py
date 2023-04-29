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
from main import *
from AllDataloader import SplitDataLoader
from utils import get_batch, load_model, get_num_rows_cols

np.random.seed(0)
random.seed(0)
sns.set_style('ticks')
sns.set_palette('Set2')

save_no = 41

num_rows_dict, num_cols_dict = get_num_rows_cols()
model, cfg_all = load_model(save_no)
cfg = cfg_all["DL_params"]

# %%
# Embeddings
# ----------
split_file = cfg["split_file"]
split_file = f'/Users/kasiakobalczyk/FairFewshot/datasets/grouped_datasets/{split_file}'
train_dl = SplitDataLoader(bs=0, num_rows=0, num_targets=0, binarise=cfg["binarise"],
                    ds_group=cfg["ds_group"], ds_split="train", 
                    num_classes=cfg["num_classes"], split_file=split_file)
test_dl = SplitDataLoader(bs=0, num_rows=0, num_targets=0, binarise=cfg["binarise"],
                    ds_group=cfg["ds_group"], ds_split="test", 
                    num_classes=cfg["num_classes"], split_file=split_file)

seen_datasets = train_dl.all_datasets
unseen_datasets = test_dl.all_datasets
seen_data_names = [str(d) for d in seen_datasets]
unseen_data_names = [str(d) for d in unseen_datasets]
datasets = seen_datasets + unseen_datasets
datanames = seen_data_names + unseen_data_names

#%%
embed_meta_ls = []
model_id_ls = []
num_cols_ls = []

num_samples = 50

for d in datanames:
    d = str(d)
    n_col = num_cols_dict[d]
    num_rows = num_rows_dict[d] // 2
    dl = SplitDataLoader(
            bs=1, num_rows=num_rows, num_targets=0, 
            num_cols=[n_col - 1, n_col], ds_group=[d],
            num_classes=cfg["num_classes"],
    )
    for i in range(num_samples):
        model_id, xs_meta, xs_target, ys_meta, ys_target = get_batch(dl, num_rows)
        pairs_meta = d2v_pairer(xs_meta, ys_meta)
        with torch.no_grad():
            embed_meta, pos_enc = model.forward_meta(pairs_meta)
        embed_meta_ls.append(embed_meta)
        model_id_ls.append(model_id)

model_id_ls = [item for sublist in model_id_ls for item in sublist]
embed_meta = torch.stack(embed_meta_ls).detach().reshape(
    num_samples * (len(datanames)), 64)

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
plot_df = reduced_embeddings_df

#%%
# All unseen datasets
sns.scatterplot(
    plot_df[plot_df.model_id.isin(unseen_data_names)], 
    x='dim_1', y='dim_2', hue='model_id')

# %%
# All seen datasets
sns.scatterplot(
    plot_df[plot_df.model_id.isin(seen_data_names)], 
      x='dim_1', y='dim_2', hue='model_id', legend=True)

# %%
# All datasets
sns.scatterplot(
    plot_df[plot_df.model_id.isin(seen_data_names + unseen_data_names)], 
      x='dim_1', y='dim_2', hue='model_id', legend=True)

# %%
# Average embeddings
mean_embeddings = reduced_embeddings_df.groupby('model_id').mean()
x = mean_embeddings['dim_1']
y = mean_embeddings['dim_2']

fig, ax = plt.subplots(figsize = (10, 10))
mean_embeddings.loc[seen_data_names, :].plot(x='dim_1', y='dim_2', kind = 'scatter', ax=ax, color='C0')
mean_embeddings.loc[unseen_data_names, :].plot(x='dim_1', y='dim_2', kind = 'scatter', ax=ax, color='C1')
for i, txt in enumerate(mean_embeddings.index):
    ax.annotate(txt, (x[i] - np.random.rand(1) * 0.8, y[i] + np.random.rand(1) * 0.5))
plt.show()
# %%
import toml
my_split = {
    'train' : ['post-operative', 'lung-cancer', 'hepatitis', 'breast-cancer', 'heart-va', 'heart-switzerland', 'fertility', 'echocardiogram', 'lymphography'],
    'test' : ['post-operative', 'breast-cancer', 'echocardiogram']
}
my_split['max_col'] = max([num_cols_dict[d] for d in my_split['test']])

my_split = {'0': my_split}

with open('../datasets/grouped_datasets/my_split', 'w') as fp:
    toml.dump(my_split, fp)
# %%


