import torch
import sys
import os
import random
import numpy as np

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, '/mnt/storage_ssd/FairFewshot/Fewshot')
from main import *
from AllDataloader import SplitDataloader, MyDataSet

np.random.seed(0)
random.seed(0)
sns.set_style('ticks')
sns.set_palette('Set2')

all_data_names = os.listdir('./datasets/data')
all_data_names.remove('info.json')

all_datasets = [
    MyDataSet(d, num_rows=0, num_targets=0, binarise=True, split="all") 
    for d in all_data_names]
num_cols_ls = [d.ds_cols - 1 for d in all_datasets]
num_cols_dict = dict(zip(all_data_names, num_cols_ls))
num_rows_ls = [d.ds_rows for d in all_datasets]
num_rows_dict = dict(zip(all_data_names, num_rows_ls))

save_no = 31
save_dir = f'./saves/save_{save_no}'
state_dict = torch.load(f'{save_dir}/model.pt')
cfg_all = get_config(cfg_file=f'{save_dir}/defaults.toml')
cfg = cfg_all["DL_params"]
model = ModelHolder(cfg_all=cfg_all)
model.load_state_dict(state_dict['model_state_dict'])


def get_batch(dl, num_rows):
    xs, ys, model_id = next(iter(dl))
    xs_meta, xs_target = xs[:, :num_rows], xs[:, num_rows:]
    ys_meta, ys_target = ys[:, :num_rows], ys[:, num_rows:]
    xs_meta, xs_target = xs_meta.contiguous(), xs_target.contiguous()
    ys_meta, ys_target = ys_meta.contiguous(), ys_target.contiguous()
    ys_target = ys_target.view(-1)

    return model_id, xs_meta, xs_target, ys_meta, ys_target

# %%
# Embeddings
# ----------'

train_dl = SplitDataloader(bs=1, num_rows=15, num_targets=0, binarise=True,
                    ds_group=tuple(cfg["ds_group"]), ds_split="train",
                    )
test_dl = SplitDataloader(bs=1, num_rows=15, num_targets=0, binarise=True,
                    ds_group=tuple(cfg["ds_group"]), ds_split="test",
                    )

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

num_rows = 30
discard_cnt = 0

for d in datanames:
    d = str(d)
    n_col = num_cols_dict[d]
    try:
        dl = SplitDataloader(
                bs=1, num_rows=15, num_targets=0,
                num_cols=[n_col - 1, n_col], ds_group=[d],
        )
    except IndexError as e:
        print(e)

    for i in range(20):
        model_id, xs_meta, xs_target, ys_meta, ys_target = get_batch(dl, num_rows)
        pairs_meta = d2v_pairer(xs_meta, ys_meta)
        with torch.no_grad():
            embed_meta, pos_enc = model.forward_meta(pairs_meta)
        embed_meta_ls.append(embed_meta)
        model_id_ls.append(model_id)

model_id_ls = [item for sublist in model_id_ls for item in sublist]
embed_meta = torch.stack(embed_meta_ls).detach().reshape(-1, 64)

#%%
scaler = StandardScaler()
reducer = TSNE(n_components=3, perplexity=21, random_state=1)
# reducer = PCA(n_components=3)
reduced_embeddings = reducer.fit_transform(scaler.fit_transform(embed_meta))
reduced_embeddings_df = pd.DataFrame({
    'dim_1' : reduced_embeddings[:, 0],
    'dim_2' : reduced_embeddings[:, 1],
    'dim_3' : reduced_embeddings[:, 2],
    'model_id' : np.array(model_id_ls).astype(str),
})
plot_df = reduced_embeddings_df

# #%%
# # All unseen datasets
# sns.scatterplot(
#     plot_df[plot_df.model_id.isin(unseen_data_names)],
#     x='dim_1', y='dim_2', hue='model_id')

# # %%
# # All seen datasets
# sns.scatterplot(
#     plot_df[plot_df.model_id.isin(seen_data_names)],
#       x='dim_1', y='dim_2', hue='model_id', legend=False)

# %%
# Average embeddings
mean_embeddings = reduced_embeddings_df.groupby('model_id').mean()
x = mean_embeddings['dim_1']
y = mean_embeddings['dim_2']

fig, ax = plt.subplots(figsize = (10, 10))
mean_embeddings.loc[seen_data_names, :].plot(x='dim_1', y='dim_2', kind = 'scatter', ax=ax, color='C0')

mean_embeddings.loc[unseen_data_names, :].plot(x='dim_1', y='dim_2', kind = 'scatter', ax=ax, color='C1')

for i, txt in enumerate(mean_embeddings.index):
    ax.annotate(txt, (x[i], y[i]))
plt.show()

# # %%
# import toml
# my_split = {
#     'train' : ['post-operative', 'lung-cancer', 'hepatitis', 'breast-cancer', 'heart-va', 'heart-switzerland', 'fertility', 'echocardiogram', 'lymphography'],
#     'test' : ['post-operative', 'breast-cancer', 'echocardiogram']
# }
# my_split['max_col'] = max([num_cols_dict[d] for d in my_split['test']])
#
# my_split = {'0': my_split}
#
# with open('../datasets/grouped_datasets/my_split', 'w') as fp:
#     toml.dump(my_split, fp)
# # %%
#
#
