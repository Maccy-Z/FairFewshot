#%%
import pandas as pd
import toml
import seaborn as sns
import matplotlib.pyplot as plt

#%%

sns.set_palette('deep')
sns.set_style('ticks')

flat_results_df = pd.DataFrame()

for save_no in range(12):
    result = pd.read_pickle(
        f'./results/all_heart_results/result_{save_no}_heart/raw.pkl')
    flat_results_df = pd.concat([flat_results_df, result])

flat_results_df.groupby(['data_name', 'model'])[['acc']].mean()
# %%
baseline_results_df = pd.DataFrame()
for i, data_name in enumerate(flat_results_df.data_name.unique()):
    result = pd.read_csv(f'./datasets/max_data/{data_name}/baselines.dat', header=None)
    result.columns = ['model', 'num_rows', 'num_cols', 'acc', 'std']
    result['data_name'] = data_name
    baseline_results_df = pd.concat([baseline_results_df, result])

baseline_results_df = baseline_results_df[baseline_results_df.num_rows == '10']

mean_results = baseline_results_df.groupby(['data_name', 'model'])[['acc']].mean()
mean_results.sort_values(['data_name', 'acc'])
# %%
baseline = 'KNN'

cols = ['data_name', 'model', 'acc', 'std']
all_results_df = pd.concat([baseline_results_df[cols], flat_results_df[cols]])

mean_acc = all_results_df.groupby(['data_name', 'model'])[['acc']].mean()
mean_acc = mean_acc.sort_values(['data_name', 'acc'], ascending=False).reset_index()

mean_acc.pivot(columns =['data_name'], index=['model']).loc[['FLAT_maml', 'FLAT', baseline], :]
#%%
compare_df = all_results_df[
    all_results_df.model.isin(['FLAT', 'FLAT_maml', baseline])]

mean_comp_acc = compare_df.pivot_table(index=['data_name'], columns=['model'], values=['acc']).droplevel(0, axis=1)
mean_comp_acc.columns = ['FLAT', 'FLAT_maml', baseline]
mean_comp_acc['FLAT_diff'] = mean_comp_acc['FLAT'] - mean_comp_acc[baseline]
mean_comp_acc['FLAT_maml_diff'] = mean_comp_acc['FLAT_maml'] - mean_comp_acc[baseline]

mean_comp_acc
# %%
names = ['heart-cleveland', 'heart-hungarian', 'heart-va', 'heart-switzerland']
dataframe_dict = {}

for name in names:
    dir_name = f'datasets/data/{name}'
    dataframe_dict[name] = {
        'data' : pd.read_csv(
            f'{dir_name}/{name}_py.dat', header=None).iloc[:, :12],
        'labels' : pd.read_csv(
            f'{dir_name}/labels_py.dat', header=None),
        'folds' : pd.read_csv(
            f'{dir_name}/folds_py.dat', header=None),
        'vfolds' : pd.read_csv(
            f'{dir_name}/validation_folds_py.dat', header=None),
    }
# %%
fig, axs = plt.subplots(2, 2, figsize=(7, 6))

cmap = sns.diverging_palette(230, 20, as_cmap=True)
corr_dict = {}
for k, name in enumerate(names):
    df = pd.concat([
        dataframe_dict[name]['data'], dataframe_dict[name]['labels']], axis=1)
    corr_dict[name] = df.corr()
    i = k % 2
    j = k // 2
    sns.heatmap(df.corr(), ax=axs[i][j], cbar=False, cmap=cmap)
    axs[i][j].set_title(name)
    axs[i][j].set_xticks([])
    axs[i][j].set_yticks([])
plt.show()
# %%
import numpy as np
def dcov(X, Y):
    """Computes the distance eucledian distance between matrices X and Y.
    """
    return ((X - Y) ** 2).sum().sum()

distance_df = pd.DataFrame(columns=names, index=names, dtype=float)

for name_x in names:
    for name_y in names:
        distance_df.loc[name_x, name_y] = dcov(
            corr_dict[name_x], corr_dict[name_y]
            )
    
# %%
sns.heatmap(distance_df, cmap='Blues', annot=True)
# %%
distance_df.sum()
# %%
