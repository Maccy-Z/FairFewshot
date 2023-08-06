#%%
import pandas as pd
import toml
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
sns.set_palette('deep')
sns.set_style('ticks')

#%%
flat_results_df = pd.DataFrame()

def get_overlap_seed(x):
    n_overlap = (x // 5) * 2
    seed = x % 5  
    return n_overlap, seed

for num_rows in [3, 5, 10]:
    for save_no in range(42):
        result = pd.read_pickle(
            f'./andrija_pc_results/all_overlap_cleveland/result_{num_rows}_{save_no}_overlap/raw.pkl')
        result['num_rows'] = num_rows
        flat_results_df = pd.concat([flat_results_df, result])

flat_results_df['seed'] = flat_results_df['data_name'].str.split('_').apply(lambda x: int(x[-1]))
flat_results_df['n_overlap'] = flat_results_df['data_name'].str.split('_').apply(lambda x: int(x[-2]))
# %%s
flat_results_df.groupby(['model', 'num_rows', 'n_overlap'])['acc'].mean()

# %%
data_names = flat_results_df['data_name'].dropna().unique()

baseline_results_df = pd.DataFrame()
for i, data_name in enumerate(data_names):
    n_overlap, seed = get_overlap_seed(i)
    result = pd.read_csv(f'./overlapdatasets_all_cleveland/data/{data_name}/baselines_overlap.dat')
    result['data_name'] = data_name
    baseline_results_df = pd.concat([baseline_results_df, result])

baseline_results_df['seed'] = baseline_results_df['data_name'].str.split('_').apply(lambda x: int(x[-1]))
baseline_results_df['n_overlap'] = baseline_results_df['data_name'].str.split('_').apply(lambda x: int(x[-2]))
baseline_results_df['num_rows'] = baseline_results_df['num_rows'].astype(int)
mean_results = baseline_results_df.groupby(['model', 'num_rows', 'n_overlap'])[['acc']].mean().reset_index()
mean_results.groupby(['model', 'num_rows', 'n_overlap'])['acc'].mean()

# %%
# Performance gains
cols = ['data_name', 'model', 'n_overlap', 'num_rows', 'seed', 'acc', 'std']

all_results_df = pd.concat([flat_results_df[cols], baseline_results_df[cols]])

flat_models = ['FLAT', 'FLAT_maml']
baseline = 'LR'

comparison_df = all_results_df[all_results_df.model.isin(flat_models + [baseline])]
comparison_df = comparison_df.pivot(
    index=['data_name', 'num_rows', 'n_overlap', 'seed'], 
    columns=['model'], values=['acc']
).reset_index()

comparison_df.columns = ['data_name', 'num_rows', 'n_overlap', 'seed'] + flat_models + [baseline]
comparison_df

comparison_std_df = all_results_df[all_results_df.model.isin(flat_models + [baseline])]
comparison_std_df = comparison_std_df.pivot(
    index=['data_name', 'num_rows', 'n_overlap', 'seed'], 
    columns=['model'], values=['std']
).reset_index()

comparison_std_df.columns = ['data_name', 'num_rows', 'n_overlap', 'seed'] + flat_models + [baseline]
comparison_std_df

comparison_df['FLAT_acc_diff'] = comparison_df['FLAT'] - comparison_df[baseline]
comparison_df['FLAT_maml_acc_diff'] = comparison_df['FLAT_maml'] - comparison_df[baseline]

comparison_std_df['FLAT_diff_std'] = np.sqrt(
    comparison_std_df['FLAT'] ** 2 + comparison_std_df[baseline] ** 2)
comparison_std_df['FLAT_maml_diff_std'] = np.sqrt(
    comparison_std_df['FLAT_maml'] ** 2 + comparison_std_df[baseline] ** 2)

#%%
mean_difference = comparison_df.groupby(
    ['num_rows', 'n_overlap'])[['FLAT_acc_diff', 'FLAT_maml_acc_diff']].mean()
mean_difference.columns = ['FLAT', 'FLAT_maml']
mean_difference = mean_difference.stack().reset_index()
mean_difference.columns = ['num_rows', 'n_overlap', 'model', 'mean']

errors = comparison_std_df.groupby(
    ['num_rows', 'n_overlap'])[['FLAT_diff_std', 'FLAT_maml_diff_std']].agg(
        lambda x: np.sqrt((x**2).sum()) / len(x))

errors.columns = ['FLAT', 'FLAT_maml']
errors = errors.stack().reset_index()
errors.columns = ['num_rows', 'n_overlap', 'model', 'std']
# errors['value_type'] = 'error'

mean_difference_df = pd.merge(
    mean_difference, errors, 
    on=['num_rows', 'n_overlap', 'model'],
)

mean_difference_df['lower'] = mean_difference_df['mean'] -  mean_difference_df['std']
mean_difference_df['upper'] = mean_difference_df['mean'] +  mean_difference_df['std']

# %%
from matplotlib import ticker

fig, axs = plt.subplots(1, 3, figsize=(12, 3), sharex=True, sharey=True)

for i, num_rows in enumerate([3, 5, 10]):
    plot_df = mean_difference_df[
        (mean_difference_df.num_rows == num_rows)
    ]
    axs[i].hlines(0, 0, 12, color = 'black', linestyles='dotted', alpha=0.5)
    sns.scatterplot(plot_df, x='n_overlap', y='mean', hue='model', ax=axs[i], legend=False)
    for model in ['FLAT', 'FLAT_maml']:
        model_df =  plot_df[plot_df.model == model]
        axs[i].errorbar(
            model_df['n_overlap'],
            model_df['mean'],
            model_df['std']
        )
    axs[i].set_xlabel('')
    axs[i].set_ylabel('')
    axs[i].set_title('$N_{meta}=$'+f'{num_rows}')
    
axs[0].xaxis.set_major_formatter(ticker.PercentFormatter(xmax=12))
axs[0].yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))

fig.supxlabel('% overlap')
fig.supylabel('Accuracy gain')
fig.tight_layout()

plt.show()

# %%
fig, ax = plt.subplots()
plot_df =  all_results_df[
    (all_results_df.n_overlap <= 12) &
    (all_results_df.num_rows == 5) &
    (all_results_df.model.isin(['FLAT', 'FLAT_maml']))
]

mean_acc = plot_df.groupby(
    ['n_overlap', 'model'])[['acc']].mean().reset_index()


sns.lineplot(mean_acc, x='n_overlap', y='acc', hue='model', ax=ax)
sns.scatterplot(mean_acc, x='n_overlap', y='acc', hue='model', ax=ax, legend=False)


plt.legend()

# %%
