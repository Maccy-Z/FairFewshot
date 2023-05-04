#%%
import pandas as pd

BASEDIR = '/Users/kasiakobalczyk/FairFewshot'
SAVEDIR = f'{BASEDIR}/saves'
#%%
similar_datasets = [
    'fertility', 'lung-cancer', 'breast-cancer', 
    'mammographic', 'echocardiogram', 'heart-va', 
    'post-operative', 'heart-switzerland'
]

def get_results(save_no_ls):
    all_results = pd.DataFrame()

    for save_no in save_no_ls:
        results = pd.read_csv(f'{SAVEDIR}/save_{save_no}/unseen_results.csv', index_col=0)
        results['save_no'] = save_no
        all_results = pd.concat([all_results, results])
    return all_results

global_results = get_results(list(range(10, 20)))
local_results = get_results(list(range(20, 27)))

view_global_results = global_results.pivot(
    index='num_cols', columns=['data_name', 'model'], values='acc')
view_local_results = local_results.pivot(
    index='num_cols', columns=['data_name', 'model'], values='acc')

# %%
compared_results = pd.read_csv(f'{BASEDIR}/results/global_local_comp.csv', index_col=0)
#%%
data_name = similar_datasets[0]
print(data_name)
compared_results.pivot(
    index='num_cols', columns=['data_name', 'model'], values='acc').sort_index()[data_name].dropna()
 # %%
idx = compared_results[compared_results.num_cols.str.startswith('total')].index
compared_results.loc[idx, 'num_cols'] = 'total'
compared_results.groupby(['num_cols', 'model'])[['acc']].mean().unstack()
# %%

all_results = pd.DataFrame()
for i, save_no in enumerate(list(range(0, 10))):
    results = pd.read_csv(f'{BASEDIR}/saves/save_{save_no}/unseen_results.csv', index_col=0)
    results['split'] = i
    all_results = pd.concat([all_results, results])
all_results.reset_index(drop=True, inplace=True)
 # %%
all_total_results = all_results[all_results.num_cols.str.startswith('t')]
all_col_results = all_results[~all_results.num_cols.str.startswith('t')]
all_total_results['num_cols'] = 'total'

#%%
all_col_results['num_cols'] = all_col_results['num_cols'].astype(int)
view_col_results = all_col_results.groupby(['num_cols', 'model'])['acc'].mean().unstack()
view_col_results['FLAT_diff'] = view_col_results['FLAT'] - view_col_results[['KNN', 'LR']].max(axis=1)
view_col_results * 100
# %%
view_total_results = all_total_results.groupby(['num_cols', 'model'])['acc'].mean().unstack()
view_total_results['FLAT_diff'] = view_total_results['FLAT'] - view_total_results[['KNN', 'LR']].max(axis=1)
view_total_results * 100
# %%

num_rows = 5
num_targets = 5
all_results = pd.DataFrame()
for i, save_no in enumerate(list(range(20, 30))):
    for j in [1, 3, 5, 10]:
        results = pd.read_csv(
            f'{BASEDIR}/saves/all_new_saves/new_saves_{num_rows}_{num_targets}/save_{save_no}/unseen_results_{j}_row.csv', index_col=0)
        results['split'] = i
        results['n_row'] = j
        all_results = pd.concat([all_results, results])


models = list(all_results.model.unique())
base_models = models.copy()
base_models.remove('FLAT_MAML')
base_models.remove('FLAT')
all_results.reset_index(drop=True, inplace=True)
all_results['num_cols'] = 'total'
view_results = all_results.pivot(index = ['split', 'n_row', 'data_name'], columns = ['model'], values = 'acc')
# %%
print('num_rows:', num_rows, 'num_targets:', num_targets)
mean_results = view_results.groupby('n_row')[models].mean() * 100
mean_results['FLAT_MAML_diff'] = mean_results['FLAT_MAML'] - mean_results.loc[:, base_models].max(axis=1)
mean_results['FLAT_diff'] = mean_results['FLAT'] - mean_results.loc[:, base_models].max(axis=1)  
mean_results.round(2)
# %%
