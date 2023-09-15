#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/Users/kasiakobalczyk/FairFewshot/Fewshot')
from utils import get_num_rows_cols

#num_rows = [2, 6, 10] 
num_rows = [1, 3, 5, 10, 15]
# get FLAT results
flat_results_df = pd.DataFrame()
iwata_results_df = pd.DataFrame()
for num_row in num_rows:
    for i in range(10):
        #df = pd.read_pickle(f'./results/fixed_kshot_results_v2/results_{num_row}_rows/result_kshot_v2_{i}_fold_{num_row}_rows/raw.pkl')
        #df = pd.read_pickle(f'./results/binomial_results_v2/results_{num_row}_rows/result_binomial_v2_{i}_fold_{num_row}_rows/raw.pkl')
        df = pd.read_pickle(f'./results/med_results_v2/results_{num_row}_rows/result__{i}_fold_{num_row}_rows/raw.pkl')
        iwata_df  = pd.read_pickle(f'./andrija_pc_results/iwata/result_iwata_{i}_row_{num_row}/raw.pkl')
        iwata_df['num_rows'] = num_row
        # filter to only testing datasets
        iwata_df = iwata_df[iwata_df.data_name.isin(df.data_name.unique())]
        df['num_rows'] = num_row
        flat_results_df = pd.concat([flat_results_df, df])
        iwata_results_df = pd.concat([iwata_results_df, iwata_df])

#%%
# get baseline results
data_names = flat_results_df.data_name.unique()
base_results_df = pd.DataFrame()
for data_name in data_names:
    header = ['model', 'num_rows', 'num_cols', 'acc', 'std']
    df = pd.read_csv(f'./datasets/max_data/{data_name}/baselines.dat', header=None)
    df.columns = header
    try:
        df_tabpfn = pd.read_csv(f'./datasets/data/{data_name}/baselines_tabpfn.dat')
        df_tabpfn.columns = header
        #print(df_tabpfn)
    except(FileNotFoundError):
        df_tabpfn = pd.DataFrame(columns=header)
    df = pd.concat([df, df_tabpfn])
    # df = pd.read_csv(f'./datasets/data/{data_name}/baselines_binomial_v2.dat', header=0)
    # df =  pd.read_csv(f'./datasets/data/{data_name}/baselines_kshot_v2.dat')
    df = df[df.model != 'Model']
    df['num_rows'] = df['num_rows'].astype(int)
    df['acc'] = df['acc'].astype(float)
    df['std'] = df['std'].astype(float)
    df = df[df.num_rows.isin(num_rows)]
    df['data_name'] = data_name
    base_results_df = pd.concat([base_results_df, df])
base_results_df

base_results_df = pd.concat([base_results_df, iwata_results_df])
#%%

# combine dataframes
results_df = pd.concat([flat_results_df, base_results_df])

# aggregate results over all datasets
models = results_df.model.unique()
flat_models = ['FLAT', 'FLAT_maml']
baseline_models = [m for m in models if m not in flat_models]
model_order = baseline_models + flat_models

agg_acc_df = results_df.groupby(['num_rows', 'model'])[['acc']].mean().unstack()
agg_acc_df = agg_acc_df.droplevel(0, axis=1)
agg_acc_df = agg_acc_df.loc[:, model_order] * 100

agg_acc_df['FLAT_maml_diff'] = agg_acc_df['FLAT_maml'] - agg_acc_df.loc[:, baseline_models].fillna(0).max(axis=1)
agg_acc_df['FLAT_diff'] = agg_acc_df['FLAT'] - agg_acc_df.loc[:, baseline_models].fillna(0).max(axis=1) 

agg_std_df = results_df.groupby(['num_rows', 'model'])[['std']].apply(lambda x: np.sqrt((x**2).sum()) / len(data_names) * 100).unstack()
agg_std_df = agg_std_df.droplevel(0, axis=1)

# find the std of the best model
best_models = agg_acc_df.loc[:, baseline_models].fillna(0).idxmax(axis=1).values
best_base_std = [agg_std_df.loc[num_row, best_models[i]] for i, num_row in enumerate(num_rows)]
best_base_std = pd.Series(best_base_std, index=num_rows)
best_std = np.sqrt(agg_std_df['FLAT']**2 + best_base_std**2) / 2
best_std

# display the results
display_df = pd.DataFrame()

for m in model_order:
    display_df[m] = agg_acc_df[m].round(2).astype(str) + ' ± ' +  agg_std_df[m].round(2).astype(str)
display_df['FLAT_diff'] = agg_acc_df['FLAT_diff'].round(2).astype(str) +  ' ± ' + best_std.round(2).astype(str)
display_df.transpose()

#%%
# find the number of successes
better_df = results_df.pivot(index=['num_rows', 'data_name'], columns=['model'], values='acc')
better_df['better'] = better_df['FLAT'] > better_df.loc[:, baseline_models].max(axis=1)
better_df['better'].unstack().sum(axis=1).loc[num_rows]

#%%
# Find model ranks

#%%
det_results_df = results_df[results_df.num_rows == 5].pivot(index=['data_name'], columns=['model'], values='acc')
det_results_df['FLAT_diff'] = det_results_df['FLAT'] - det_results_df.loc[:, baseline_models].max(axis=1)
det_results_df = (det_results_df * 100).round(1).sort_values(by='FLAT_diff')
model_order = list(det_results_df.mean().sort_values().index.values[1:]) + ['FLAT_diff']
model_order.remove('FLAT_maml')
det_results_df.loc[:, model_order]
#det_results_df.loc[:, model_order].mean().round(1).values

#%%
# num_cols vs results

num_rows_dict, num_cols_dict = get_num_rows_cols()


plt.scatter([num_cols_dict[d] for d in det_results_df.index.values], det_results_df['FLAT_diff'])
plt.xlim([0, 50])
plt.xlabel('num_cols')
plt.ylabel('FLAT_diff')

# %% 
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
# Compare the results against the number of rows / classses
DATADIR = './datasets/data'

data_dim_df = pd.DataFrame(index=results_df.data_name.unique())
data_dim_df['n_cols'] = data_dim_df['n_classes'] = None

for ds_name in data_dim_df.index:
    labels = pd.read_csv(f'{DATADIR}/{ds_name}/labels_py.dat', header=None)
    data = pd.read_csv(f'{DATADIR}/{ds_name}/{ds_name}_py.dat', header=None)
    n_classes = labels.iloc[:, 0].nunique()
    n_cols = data.shape[1]
    data_dim_df.loc[ds_name, 'n_cols'] = n_cols
    data_dim_df.loc[ds_name, 'n_classes'] = n_classes
data_dim_df

#%%
from scipy.stats import ttest_ind
for baseline in baseline_models:
    compare_df = results_df[
        (results_df.model.isin(['FLAT', baseline])) & 
        (results_df.num_rows == 5)
    ][['data_name', 'model', 'acc']].pivot(
        index='data_name', columns='model').droplevel(0, axis=1)
    compare_df['diff'] = compare_df['FLAT'] - compare_df[baseline]

    compare_df = compare_df.join(data_dim_df)

    # Test the hypothesis that the number of original classes influences the performance
    a = compare_df[compare_df.n_classes == 2]['diff']
    b = compare_df[compare_df.n_classes > 2]['diff']

    print(baseline, ttest_ind(a, b).pvalue)
# %%
