import pickle
from Fewshot.comparison_base2 import main as baselines

with open(f'/mnt/storage_ssd/FairFewshot/Results/24/raw.pkl', "rb") as f:
    flat_results = pickle.load(f)

flat_results = flat_results.pivot(columns=['data_name', 'model'], index='num_cols', values=['acc'])
flat_results.columns = flat_results.columns.droplevel(0)

wanted_columns = sorted(list(set(list(flat_results.columns.get_level_values(0)))))

best_flat = {}
for col in wanted_columns:
    ds_results = flat_results[col][['FLAT', 'FLAT_maml']]


    best_flat[col] = ds_results.iloc[0].tolist()


baseline_results = baselines(num_rows=3)
baseline_results.columns = baseline_results.columns.droplevel(0)


best_baseline = {}
for col in wanted_columns:
    ds_results = baseline_results[col].iloc[0].tolist()

    best_baseline[col] = max(ds_results)

flat_win, maml_win, baseline_win = 0, 0, 0
for col in wanted_columns:
    flat, maml = best_flat[col]

    results = [flat, maml, best_baseline[col]]
    best_result = max(results)
    idx = results.index(best_result)

    if idx == 0:
        flat_win += 1
    elif idx == 1:
        maml_win += 1
    else:
        baseline_win += 1

print(flat_win, maml_win, baseline_win)

