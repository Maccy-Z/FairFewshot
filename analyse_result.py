#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

flat_results_df = pd.DataFrame()
for num_row in [1, 2, 3, 5, 6, 10]:
    for i in range(10):
        df = pd.read_pickle(f'./results/med_results_new/results_{num_row}_rows/result_{i}_fold_{num_row}_rows/raw.pkl')
        df['num_rows'] = num_row
        flat_results_df = pd.concat([flat_results_df, df])
flat_results_df

#%%
data_names = flat_results_df.data_name.unique()

base_results_df = pd.DataFrame()
for data_name in data_names:
    df = pd.read_csv(f'./datasets/data/{data_name}/baselines.dat')
    df['data_name'] = data_name
    base_results_df = pd.concat([base_results_df, df])
base_results_df.rename({'Model':'model'}, axis=1, inplace=True)
base_results_df
#%%
results_df = pd.concat([flat_results_df, base_results_df])

models = results_df.model.unique()
flat_models = ['FLAT_maml', 'FLAT'] 
baseline_models = [m for m in models if m not in flat_models]
model_order = flat_models + baseline_models

agg_results_df = results_df.groupby(['num_rows', 'model'])[['acc']].mean().unstack()
agg_results_df = agg_results_df.droplevel(0, axis=1)
agg_results_df = agg_results_df.loc[:, model_order] * 100
agg_results_df['FLAT_maml_diff'] = agg_results_df['FLAT_maml'] - agg_results_df.loc[:, baseline_models].max(axis=1)
agg_results_df['FLAT_diff'] = agg_results_df['FLAT'] - agg_results_df.loc[:, baseline_models].max(axis=1) 
agg_results_df
# %%0
num_row = 5
single_result_df = pd.DataFrame()
for i in range(10):
    df = pd.read_pickle(f'./results/med_results/results_{num_row}_rows/result_{i}_fold_{num_row}_rows/raw.pkl')
    df['num_rows'] = 3
    single_result_df= pd.concat([single_result_df, df])
single_result_df = single_result_df.pivot(index='data_name', columns='model', values=['acc', 'std'])
data_names = single_result_df.index
single_result_df.reset_index(inplace=True)
best_baselines = single_result_df.loc[:, 'acc'].loc[:, baseline_models].idxmax(axis=1).values
best_base_acc = [single_result_df.loc[i, ('acc', best_baselines[i])] for i in range(29)]
best_base_std = [single_result_df.loc[i, ('std', best_baselines[i])] for i in range(29)]
# %%
plot_df = pd.DataFrame(
    index = data_names,
    columns=pd.MultiIndex.from_product(
    [['acc', 'std'], ['FLAT', 'best_baseline']]))
plot_df.iloc[:, 0] = single_result_df.loc[:, ('acc', 'FLAT')]
plot_df.loc[:, ('std', 'FLAT')] =  single_result_df.loc[:, ('std', 'FLAT')]
plot_df.loc[:, ('acc', 'best_baseline')] = best_base_acc
plot_df.loc[:, ('std', 'best_baseline')] = best_base_std
plot_df = pd.DataFrame(plot_df.unstack()).reset_index()
plot_df.columns = ['kind', 'model', 'data_name']
#%%
plot_df = pd.DataFrame(plot_df.unstack()).reset_index()
plot_df.columns = ['model','data_name', 'acc']
sns.barplot(data=plot_df, y='data_name', x='acc', hue='model')
plot_df
# %%
single_result_df.loc[:, 'acc'].loc[:, baseline_models].idxmax(axis=1)
# %%
