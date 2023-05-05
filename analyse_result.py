#%%
import pandas as pd
import numpy as np

results_df = pd.DataFrame()
for num_row in [1, 3, 5, 10]:
    for i in range(10):
        df = pd.read_pickle(f'./results/med_results/results_{num_row}_rows/result_{i}_fold_{num_row}_rows/raw.pkl')
        df['num_rows'] = num_row
        results_df = pd.concat([results_df, df])

# acc_df = results_df.pivot(index='data_name', columns='model', values=['acc'])
# std_df = results_df.pivot(index='data_name', columns='model', values=['std'])

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
result_3_row_df = pd.DataFrame()
for i in range(10):
    df = pd.read_pickle(f'./results/med_results/results_3_rows/result_{i}_fold_3_rows/raw.pkl')
    df['num_rows'] = 3
    result_3_row_df= pd.concat([result_3_row_df, df])
result_3_row_df[result_3_row_df.model == "FLAT"]
# %%
