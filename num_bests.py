import pickle
from old.comparison_base2 import main as baselines


save_no = 13
num_rows= 15

with open(f'/mnt/storage_ssd/FairFewshot/Results/{save_no}/raw.pkl', "rb") as f:
    flat_results = pickle.load(f)

flat_results = flat_results.pivot(columns=['data_name', 'model'], index='num_cols', values=['acc'])
flat_results.columns = flat_results.columns.droplevel(0)

wanted_columns = sorted(list(set(list(flat_results.columns.get_level_values(0)))))

best_flat = {}
for col in wanted_columns:
    ds_results = flat_results[col][['FLAT', 'FLAT_maml']]


    best_flat[col] = ds_results.iloc[0].tolist()


baseline_results = baselines(num_rows=num_rows)
baseline_results.columns = baseline_results.columns.droplevel(0)


best_baseline = {}
for col in wanted_columns:
    ds_results = baseline_results[col].iloc[0].tolist()

    best_baseline[col] = max(ds_results)

flat_win, maml_win, base_flat, base_maml= 0, 0, 0, 0
for col in wanted_columns:
    flat, maml = best_flat[col]

    results = [flat, maml, best_baseline[col]]
    best_result = max(results)
    idx = results.index(best_result)

    if flat > best_baseline[col]:
        flat_win += 1
    else:
        base_flat += 1

    if maml > best_baseline[col]:
        maml_win += 1
    else:
        base_maml += 1


print("FLAT")
print(flat_win, base_flat)
print("MAML")
print(maml_win, base_maml)

