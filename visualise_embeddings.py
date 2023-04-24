#%%
%load_ext autoreload
%autoreload 2

import torch
import sys
import os
sys.path.insert(0, '/Users/kasiakobalczyk/FairFewshot/Fewshot')

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from Fewshot.main import *
from Fewshot.mydataloader import MyDataLoader
from myutils import get_batch, load_model, get_flat_embedding

np.random.seed(0)
random.seed(0)
sns.set_style('ticks')
sns.set_palette('Set2')

all_data_names = os.listdir('./datasets/data')
all_data_names.remove('info.json')
all_data_names.remove('.DS_Store')
dl = MyDataLoader(bs=1, num_rows=5, num_targets=5, data_names=all_data_names)
num_col_dict = dict(zip(all_data_names, [d.num_cols for d in dl.datasets]))

save_no = 5
base_dir = '.'
model, all_cfg = load_model(save_no)
cfg = all_cfg["DL_params"]

# %%
# Embeddings
# ----------
seen_data_names = cfg["train_data_names"]
unseen_data_names = cfg["test_data_names"]

embed_meta_ls = []
model_id_ls = []
num_cols_ls = []
num_rows = 40

for d in seen_data_names + unseen_data_names:
    dl = MyDataLoader(
            bs=cfg["bs"], num_rows=num_rows, num_targets=0, 
            num_cols=[num_col_dict[d]], data_names=[d],
            shuffle_cols=cfg["shuffle_cols"], split="train"
    )
    for i in range(50):
        model_id, xs_meta, xs_target, ys_meta, ys_target = get_batch(dl, num_rows)
        embed_meta, pos_enc = get_flat_embedding(xs_meta, ys_meta, model)
        embed_meta_ls.append(embed_meta)
        model_id_ls.append(model_id)

model_id_ls = [item for sublist in model_id_ls for item in sublist]
embed_meta = torch.stack(embed_meta_ls).detach().reshape(
    50 * cfg["bs"] * len(unseen_data_names + seen_data_names), 64)

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
      x='dim_1', y='dim_2', hue='model_id', legend=False)

# %%
# Average embeddings
mean_embeddings = reduced_embeddings_df.groupby('model_id').mean()
x = mean_embeddings['dim_1']
y = mean_embeddings['dim_2']

fig, ax = plt.subplots(figsize = (10, 10))
mean_embeddings.loc[seen_data_names, :].plot(x='dim_1', y='dim_2', kind = 'scatter', ax=ax, color='C0')
mean_embeddings.loc[unseen_data_names, :].plot(x='dim_1', y='dim_2', kind = 'scatter', ax=ax, color='C1')
for i, txt in enumerate(mean_embeddings.index):
    ax.annotate(txt, (x[i] - np.random.rand(1), y[i] + np.random.rand(1)))
plt.show()
