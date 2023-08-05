#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import toml
sys.path.insert(0, '/Users/kasiakobalczyk/FairFewshot/Fewshot')
from utils import get_num_rows_cols

def is_seen(data_name, ds_group):
    split_file = f"./datasets/grouped_datasets/med_splits_2"
    with open(split_file) as f:
        split = toml.load(f)
    train_data_names = split[str(ds_group)]["train"]
    test_data_names = split[str(ds_group)]["test"]
    if data_name in train_data_names:
        return True
    else:
        return False

def is_medical(data_name):
    medical = [
        'acute-inflammation', 'acute-nephritis', 'arrhythmia',
        'blood', 'breast-cancer', 'breast-cancer-wisc', 'breast-cancer-wisc-diag', 
        'breast-cancer-wisc-prog', 'breast-tissue', 'cardiotocography-3clases', 
        'dermatology', 'echocardiogram', 'fertility', 'heart-cleveland', 
        'heart-hungarian', 'heart-switzerland', 'heart-va', 'hepatitis', 'horse-colic',
        'ilpd-indian-liver', 'lung-cancer', 'lymphography', 'mammographic', 
        'parkinsons', 'post-operative', 'primary-tumor', 'spect', 'spectf', 
        'statlog-heart', 'thyroid', 'vertebral-column-2clases'
    ]
    medical.remove('fertility')
    medical.remove('lung-cancer')
    if data_name in medical:
        return True
    else:
        return False

#num_rows = [2, 6, 10] 
num_rows = [5]
# get FLAT results
flat_results_df = pd.DataFrame()
for num_row in num_rows:
    for i in range(10):
        #df = pd.read_pickle(f'./results/fixed_kshot_results_v2/results_{num_row}_rows/result_kshot_v2_{i}_fold_{num_row}_rows/raw.pkl')
        #df = pd.read_pickle(f'./results/binomial_results_v2/results_{num_row}_rows/result_binomial_v2_{i}_fold_{num_row}_rows/raw.pkl')
        df = pd.read_pickle(f'./results/med_results_v2/results_{num_row}_rows/result__{i}_fold_{num_row}_rows/raw.pkl')
        #df = pd.read_pickle(f'./results/all_results/result_{i}all_fold_{num_row}_rows/raw.pkl')
        df['num_rows'] = num_row
        flat_results_df = pd.concat([flat_results_df, df])

# get baseline results
data_names = flat_results_df.data_name.unique()
base_results_df = pd.DataFrame()
for data_name in data_names:
    df = pd.read_csv(f'./datasets/max_data/{data_name}/baselines.dat', header=None)
    # df = pd.read_csv(f'./datasets/data/{data_name}/baselines_binomial_v2.dat', header=0)
    # df =  pd.read_csv(f'./datasets/data/{data_name}/baselines_kshot_v2.dat')
    df.columns = ['model', 'num_rows', 'num_cols', 'acc', 'std']
    df = df[df.model != 'Model']
    df['num_rows'] = df['num_rows'].astype(int)
    df['acc'] = df['acc'].astype(float)
    df['std'] = df['std'].astype(float)
    df = df[df.num_rows.isin(num_rows)]
    df['data_name'] = data_name
    base_results_df = pd.concat([base_results_df, df])
base_results_df


#%%

# combine dataframes
results_df = pd.concat([flat_results_df, base_results_df])

# aggregate results over all datasets
models = results_df.model.unique()
#flat_models = ['FLAT', 'FLAT_maml']
flat_models = ['FLAT']
baseline_models = [m for m in models if m not in flat_models]
model_order = baseline_models + flat_models

agg_acc_df = results_df.groupby(['num_rows', 'model'])[['acc']].mean().unstack()
agg_acc_df = agg_acc_df.droplevel(0, axis=1)
agg_acc_df = agg_acc_df.loc[:, model_order] * 100

#agg_acc_df['FLAT_maml_diff'] = agg_acc_df['FLAT_maml'] - agg_acc_df.loc[:, baseline_models].fillna(0).max(axis=1)
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
# Analyse all results

num_rows = [5]
# get FLAT results
flat_results_df = pd.DataFrame()
for num_row in num_rows:
    for i in range(10):
        df = pd.read_pickle(f'./results/all_results/result_{i}all_fold_{num_row}_rows/raw.pkl')
        df['num_rows'] = num_row
        df['fold'] = i
        df['seen'] = df['data_name'].apply(lambda x: is_seen(x, i))
        df['medical'] = df['data_name'].apply(is_medical)
        flat_results_df = pd.concat([flat_results_df, df])

# %%
flat_results_df[~flat_results_df.seen & flat_results_df.medical]

#%%
# get baseline results
data_names = flat_results_df.data_name.unique()
base_results_df = pd.DataFrame()
for data_name in data_names:
    df = pd.read_csv(f'./datasets/max_data/{data_name}/baselines.dat', header=None)
    df.columns = ['model', 'num_rows', 'num_cols', 'acc', 'std']
    df = df[df.model != 'Model']
    df['num_rows'] = df['num_rows'].astype(int)
    df['acc'] = df['acc'].astype(float)
    df['std'] = df['std'].astype(float)
    df = df[df.num_rows.isin(num_rows)]
    df['data_name'] = data_name
    base_results_df = pd.concat([base_results_df, df])
base_results_df

# %%
results_df = flat_results_df.loc[:, ['data_name', 'fold', 'seen', 'medical', 'acc']]
results_df.rename({'acc' : 'FLAT'}, axis=1, inplace=True)

baseline_models = base_results_df.model.unique()
base_results_df = base_results_df.pivot(index=['data_name'], columns = ['model'], values = ['acc'])
base_results_df = base_results_df.droplevel(0, axis=1)
base_results_df.reset_index(inplace=True)
results_df = results_df.merge(base_results_df, how='left', on='data_name')
# %%
results_df[~results_df.seen & results_df.medical].mean() * 100
#%%
results_df['FLAT_diff'] = results_df['FLAT'] - results_df.loc[:, baseline_models].max(axis=1)
for base_mod in baseline_models:
    results_df[f'FLAT_{base_mod}_diff'] = results_df['FLAT'] - results_df[base_mod]
results_df
# %%
diff_cols = [f'FLAT_{m}_diff' for m in baseline_models]
(results_df.groupby(['seen', 'medical'])[diff_cols].mean() * 100).round(2)

# %%
(results_df.groupby(['seen', 'medical'])[diff_cols].mean().mean(axis=1) * 100).round(2)
# %%
results_df.groupby(['seen', 'medical'])['FLAT_diff'].mean() * 100
# %%
